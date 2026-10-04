"""
Кредитный скоринг: Give Me Some Credit
LogReg (baseline) -> LightGBM (+ подбор параметров) -> CatBoost
Метрики: ROC-AUC, Gini, KS, PSI. Затем: калибровка, выбор порога по стоимости ошибок, SHAP.

Запуск:
    pip install lightgbm catboost scikit-learn pandas numpy matplotlib scipy shap
    python credit_scoring.py --data cs-training.csv
    python credit_scoring.py --data cs-training.csv --fast        # без подбора параметров
    python credit_scoring.py --data cs-training.csv --cost-fn 5   # цена пропуска дефолта
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from scipy.stats import ks_2samp
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
TARGET = "SeriousDlqin2yrs"
OUT = "results"
os.makedirs(OUT, exist_ok=True)


def gini(y, p):
    """Gini = 2 * AUC - 1"""
    return 2 * roc_auc_score(y, p) - 1


def ks_stat(y, p):
    """KS: макс. расстояние между распределениями скора 'плохих' и 'хороших'"""
    y, p = np.asarray(y), np.asarray(p)
    return ks_2samp(p[y == 1], p[y == 0]).statistic


def psi(expected, actual, bins=10):
    """PSI скора: <0.1 стабильно, 0.1-0.25 умеренный сдвиг, >0.25 сильный."""
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df = df.rename(columns={
        "NumberOfTime30-59DaysPastDueNotWorse": "Late30_59",
        "NumberOfTimes90DaysLate": "Late90",
        "NumberOfTime60-89DaysPastDueNotWorse": "Late60_89",
    })
    df.loc[df["age"] < 18, "age"] = np.nan                 
    for c in ["Late30_59", "Late60_89", "Late90"]:          
        df.loc[df[c] >= 96, c] = np.nan
    for c in ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"]:
        df[c] = df[c].clip(upper=df[c].quantile(0.995))     
    df["income_missing"] = df["MonthlyIncome"].isna().astype(int)
    df["total_late"] = df[["Late30_59", "Late60_89", "Late90"]].sum(axis=1, min_count=1)
    return df


def tune_lgbm(X_tr, y_tr, n_iter):
    """RandomizedSearchCV по ROC-AUC, 3-fold CV на train."""
    grid = {
        "num_leaves": [15, 31, 63],
        "min_child_samples": [50, 100, 200],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "subsample": [0.7, 0.8, 1.0],
        "reg_lambda": [0, 1, 5, 10],
        "learning_rate": [0.03, 0.05, 0.1],
    }
    base = LGBMClassifier(n_estimators=300, subsample_freq=1, n_jobs=1,
                          random_state=SEED, verbose=-1)
    search = RandomizedSearchCV(
        base, grid, n_iter=n_iter, scoring="roc_auc", n_jobs=-1, random_state=SEED,
        cv=StratifiedKFold(3, shuffle=True, random_state=SEED))
    search.fit(X_tr, y_tr)
    print(f"Подбор LightGBM: CV AUC = {search.best_score_:.4f}, параметры = {search.best_params_}")
    return search.best_params_


def fit_lgbm(X_tr, y_tr, X_va, y_va, params):
    p = dict(num_leaves=31, min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
             reg_lambda=1.0, learning_rate=0.03)
    p.update(params)
    m = LGBMClassifier(n_estimators=3000, subsample_freq=1, random_state=SEED, verbose=-1, **p)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
          callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
    return m


def calibrate(model, X_va, y_va):
    """Изотоническая калибровка на valid. Модель уже обучена, поэтому замораживаем её."""
    try:
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    except ImportError:                                     
        cal = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    return cal.fit(X_va, y_va)


def threshold_table(y, p, c_fn, c_fp, grid):
    """Отказываем, если p >= t.
    Стоимость = c_fn * (одобрили плохого) + c_fp * (отказали хорошему)."""
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    for t in grid:
        approve = p < t
        fn = int(((y == 1) & approve).sum())      
        fp = int(((y == 0) & ~approve).sum())     
        rows.append({
            "threshold": t, "approval_rate": approve.mean(),
            "bad_rate_approved": y[approve].mean() if approve.any() else np.nan,
            "FN": fn, "FP": fp, "cost": c_fn * fn + c_fp * fp,
        })
    return pd.DataFrame(rows)


def main(a):
    df = load_and_clean(a.data)
    print(f"Строк: {len(df):,}, доля дефолтов: {df[TARGET].mean():.2%}")
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

    models = {}
    models["LogReg"] = Pipeline([
        ("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]).fit(X_tr, y_tr)

    params = {} if a.fast else tune_lgbm(X_tr, y_tr, a.n_iter)
    models["LightGBM"] = fit_lgbm(X_tr, y_tr, X_va, y_va, params)
    print(f"LightGBM: лучшая итерация = {models['LightGBM'].best_iteration_}")

    models["CatBoost"] = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=6, eval_metric="AUC",
        random_seed=SEED, verbose=0, early_stopping_rounds=100,
    ).fit(X_tr, y_tr, eval_set=(X_va, y_va))

    pr = lambda m, X: m.predict_proba(X)[:, 1]
    sc = {n: {"train": pr(m, X_tr), "valid": pr(m, X_va), "test": pr(m, X_te)} for n, m in models.items()}

    rows = []
    for n, s in sc.items():
        rows.append({
            "model": n,
            "Gini valid": gini(y_va, s["valid"]),
            "ROC-AUC test": roc_auc_score(y_te, s["test"]),
            "Gini test": gini(y_te, s["test"]),
            "KS test": ks_stat(y_te, s["test"]),
            "Gini train": gini(y_tr, s["train"]),
            "PSI train->test": psi(s["train"], s["test"]),
        })
    res = pd.DataFrame(rows).set_index("model").round(4)
    print("\n=== Сравнение моделей ===")
    print(res.to_string())
    res.to_csv(f"{OUT}/metrics.csv")

    best = res["Gini valid"].idxmax()       
    print(f"\nЛучшая модель по Gini на valid: {best}")

    plt.figure(figsize=(6, 5))
    for n, s in sc.items():
        fpr, tpr, _ = roc_curve(y_te, s["test"])
        plt.plot(fpr, tpr, label=f"{n} (AUC={roc_auc_score(y_te, s['test']):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC-кривые (test)")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/roc.png", dpi=130); plt.close()

    p, yy = sc[best]["test"], y_te.values
    grid = np.linspace(p.min(), p.max(), 300)
    plt.figure(figsize=(6, 5))
    plt.plot(grid, [(p[yy == 1] <= t).mean() for t in grid], label="плохие (дефолт)")
    plt.plot(grid, [(p[yy == 0] <= t).mean() for t in grid], label="хорошие")
    plt.title(f"KS = {ks_stat(yy, p):.3f} ({best})")
    plt.xlabel("скор"); plt.ylabel("CDF"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT}/ks.png", dpi=130); plt.close()

    cal = calibrate(models[best], X_va, y_va)
    p_raw, p_cal = sc[best]["test"], pr(cal, X_te)
    brier = {"до калибровки": brier_score_loss(y_te, p_raw),
             "после калибровки": brier_score_loss(y_te, p_cal)}
    print("\n=== Калибровка (Brier score, меньше = лучше) ===")
    for k, v in brier.items():
        print(f"  {k}: {v:.5f}")
    print(f"  Gini после калибровки: {gini(y_te, p_cal):.4f} (ранжирование почти не меняется)")

    plt.figure(figsize=(6, 5))
    for lab, pp in [("до", p_raw), ("после", p_cal)]:
        fr, mp = calibration_curve(y_te, pp, n_bins=10, strategy="quantile")
        plt.plot(mp, fr, "o-", label=lab)
    plt.plot([0, 1], [0, 1], "k--", lw=0.8, label="идеал")
    plt.xlabel("предсказанная вероятность"); plt.ylabel("доля реальных дефолтов")
    plt.title(f"Калибровка ({best})"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT}/calibration.png", dpi=130); plt.close()

    p_cal_va = pr(cal, X_va)
    tgrid = np.unique(np.quantile(p_cal_va, np.linspace(0.5, 0.999, 200)))
    tv = threshold_table(y_va, p_cal_va, a.cost_fn, a.cost_fp, tgrid)
    t_opt = float(tv.loc[tv["cost"].idxmin(), "threshold"])
    tt = threshold_table(y_te, p_cal, a.cost_fn, a.cost_fp, [t_opt]).iloc[0]
    all_cost = a.cost_fn * int((y_te == 1).sum())            
    print(f"\n=== Порог отсечения (цена пропуска дефолта = {a.cost_fn}, цена отказа хорошему = {a.cost_fp}) ===")
    print(f"  Оптимальный порог на valid: {t_opt:.4f}")
    print(f"  TEST: одобрено {tt['approval_rate']:.1%} заявок, доля дефолтов среди одобренных "
          f"{tt['bad_rate_approved']:.2%} (против {y_te.mean():.2%} без модели)")
    print(f"  TEST: стоимость ошибок {tt['cost']:.0f} против {all_cost:.0f} при одобрении всех")
    tt.to_frame("value").to_csv(f"{OUT}/threshold_result.csv")

    full = threshold_table(y_te, p_cal, a.cost_fn, a.cost_fp, tgrid)
    plt.figure(figsize=(6, 5))
    plt.plot(full["threshold"], full["cost"])
    plt.axvline(t_opt, color="r", ls="--", label=f"порог = {t_opt:.3f}")
    plt.xlabel("порог отказа"); plt.ylabel("стоимость ошибок (test)")
    plt.title("Выбор порога"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT}/threshold.png", dpi=130); plt.close()

    try:
        import shap
        sample = X_te.sample(min(5000, len(X_te)), random_state=SEED)
        sv = shap.TreeExplainer(models["LightGBM"]).shap_values(sample)
        if isinstance(sv, list):
            sv = sv[1]
        shap.summary_plot(sv, sample, show=False)
        plt.tight_layout(); plt.savefig(f"{OUT}/shap_summary.png", dpi=130, bbox_inches="tight"); plt.close()
        shap.summary_plot(sv, sample, plot_type="bar", show=False)
        plt.tight_layout(); plt.savefig(f"{OUT}/shap_bar.png", dpi=130, bbox_inches="tight"); plt.close()
        print("\nSHAP-графики сохранены.")
    except ImportError:
        print("\nSHAP пропущен: pip install shap")

    print(f"\nГотово. Все графики и таблицы в папке '{OUT}/'")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cs-training.csv")
    ap.add_argument("--fast", action="store_true", help="без подбора параметров LightGBM")
    ap.add_argument("--n-iter", type=int, default=15, help="число попыток RandomizedSearchCV")
    ap.add_argument("--cost-fn", type=float, default=5, help="цена одобрения будущего дефолтника")
    ap.add_argument("--cost-fp", type=float, default=1, help="цена отказа хорошему клиенту")
    main(ap.parse_args())
