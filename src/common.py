"""Shared building blocks: data loading, leakage audit, preprocessing, models, metrics, cost.

Convention used everywhere: y = 1 means the ADVERSE outcome
  - Loan Default dataset : Status = 1 (borrower defaulted)
  - Loan Approval dataset: Loan_Status = 'N' (application rejected by the lender)
A model's probability p is therefore "probability of the adverse outcome" and
the decision rule is: predict adverse (reject) if p >= threshold.
A false negative (missed adverse outcome) = a FALSE APPROVAL, the expensive error.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

SEED = 42


# ----------------------------------------------------------------------------- data
def load_default(path):
    """Dataset 1 (Loan Default, 148,670 rows). ID and year are identifiers, dropped."""
    df = pd.read_csv(path)
    y = df["Status"].astype(int)                      # 1 = defaulted
    X = df.drop(columns=["Status", "ID", "year"])
    return X, y


def load_approval(path):
    """Dataset 2 (Loan Approval, 614 rows). Loan_ID dropped; y = 1 means REJECTED."""
    df = pd.read_csv(path)
    y = (df["Loan_Status"] == "N").astype(int)
    X = df.drop(columns=["Loan_Status", "Loan_ID"])
    return X, y


def split_60_20_20(X, y, seed=SEED):
    """Stratified train (60%) / validation (20%) / test (20%) split.
    Done BEFORE any imputation, scaling, encoding or resampling."""
    X_dev, X_te, y_dev, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=seed)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


# ----------------------------------------------------------------------------- leakage audit
def leakage_audit(X, y, auc_thr=0.60, rate_thr=0.90, level_thr=0.99):
    """Audit candidate predictors on TRAINING data only.

    A column is flagged when
      (a) its missing-value indicator alone separates the classes (AUC >= auc_thr or <= 1-auc_thr), or
      (b) at least min_n rows are missing and the adverse rate among them is >= rate_thr, or
      (c) a categorical level with at least min_n rows has an adverse rate >= level_thr (or <= 1-level_thr).
    Such columns are almost certainly recorded AFTER the outcome is known, i.e. not available at origination.
    """
    min_n = max(50, int(0.001 * len(X)))
    rows = []
    for c in X.columns:
        reason, auc, rate_m, n_m = [], np.nan, np.nan, int(X[c].isna().sum())
        if n_m > 0:
            m = X[c].isna().astype(int)
            auc = roc_auc_score(y, m)
            rate_m = float(y[m == 1].mean())
            if auc >= auc_thr or auc <= 1 - auc_thr:
                reason.append(f"missing-indicator AUC={auc:.3f}")
            if n_m >= min_n and rate_m >= rate_thr:
                reason.append(f"adverse rate {rate_m:.1%} when missing")
        if X[c].dtype == object or str(X[c].dtype).startswith("str"):
            g = y.groupby(X[c]).agg(["mean", "size"])
            g = g[g["size"] >= max(min_n, 500)] if len(X) > 5000 else g[g["size"] >= min_n]
            bad = g[(g["mean"] >= level_thr) | (g["mean"] <= 1 - level_thr)]
            for lvl, r in bad.iterrows():
                reason.append(f"level '{lvl}' has adverse rate {r['mean']:.2%} (n={int(r['size'])})")
        rows.append(dict(feature=c, n_missing=n_m, pct_missing=round(100 * n_m / len(X), 2),
                         missing_indicator_auc=auc, adverse_rate_if_missing=rate_m,
                         flagged=bool(reason), reason="; ".join(reason)))
    out = pd.DataFrame(rows)
    return out, out.loc[out.flagged, "feature"].tolist()


# ----------------------------------------------------------------------------- preprocessing
def make_preprocessor(X):
    """Impute + scale numerics (median, with missing-value indicators) and impute + one-hot
    categoricals (missing becomes its own level). Always fitted on training data only."""
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in X.columns if c not in cat]
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                         ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                         ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)])


# ----------------------------------------------------------------------------- models
GRIDS = {
    "LogReg": [dict(C=c) for c in (0.01, 0.1, 1.0, 10.0)],
    "RandForest": [dict(max_depth=d, min_samples_leaf=m) for d in (6, 12, None) for m in (5, 20)],
    "LightGBM": [dict(num_leaves=n, min_child_samples=m, learning_rate=lr)
                 for n in (15, 31, 63) for m in (20, 50) for lr in (0.05, 0.1)],
    "XGBoost": [dict(max_depth=d, learning_rate=lr, n_estimators=ne)
                for d in (4, 8, 12) for lr in (0.05, 0.1) for ne in (200, 500)],
    "CatBoost": [dict(depth=d, learning_rate=lr, iterations=iters)
                 for d in (4, 8, 12) for lr in (0.05, 0.1) for iters in (200, 500)],
}


def make_estimator(name, params=None, class_weight=None, seed=SEED):
    p = dict(params or {})
    if name == "LogReg":
        return LogisticRegression(max_iter=3000, class_weight=class_weight, random_state=seed, **p)
    if name == "RandForest":
        return RandomForestClassifier(n_estimators=150, n_jobs=2, class_weight=class_weight,
                                      random_state=seed, **p)
    if name == "LightGBM":
        return LGBMClassifier(
            learning_rate=p.pop("learning_rate", 0.05),
            n_estimators=p.pop("n_estimators", 200),
            num_leaves=p.pop("num_leaves", 31),
            min_child_samples=p.pop("min_child_samples", 20),
            subsample=p.pop("subsample", 0.8),
            subsample_freq=1,
            colsample_bytree=p.pop("colsample_bytree", 0.8),
            reg_lambda=p.pop("reg_lambda", 1.0),
            reg_alpha=p.pop("reg_alpha", 0.0),
            class_weight=class_weight,
            random_state=seed, n_jobs=2, verbose=-1, **p)
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=p.pop("n_estimators", 200),
            max_depth=p.pop("max_depth", 6),
            learning_rate=p.pop("learning_rate", 0.05),
            subsample=p.pop("subsample", 0.8),
            colsample_bytree=p.pop("colsample_bytree", 0.8),
            reg_lambda=p.pop("reg_lambda", 1.0),
            reg_alpha=p.pop("reg_alpha", 0.0),
            random_state=seed, n_jobs=2, verbosity=0, eval_metric="auc", **p)
    if name == "CatBoost":
        return CatBoostClassifier(
            iterations=p.pop("iterations", 200),
            depth=p.pop("depth", 6),
            learning_rate=p.pop("learning_rate", 0.05),
            l2_leaf_reg=p.pop("l2_leaf_reg", 1.0),
            border_count=p.pop("border_count", 255),
            random_state=seed, verbose=0, **p)
    raise ValueError(name)


def make_pipeline(X, name, params=None, class_weight=None, smote=False, seed=SEED):
    """preprocess -> (optional SMOTE, training folds only) -> estimator."""
    est = make_estimator(name, params, class_weight, seed)
    pre = make_preprocessor(X)
    if smote:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        return ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=seed)), ("clf", est)])
    return Pipeline([("pre", pre), ("clf", est)])


# ----------------------------------------------------------------------------- metrics / cost
def metrics(y, p, t=0.5):
    """Threshold metrics at t plus AUC. Positive class = adverse outcome."""
    yhat = (p >= t).astype(int)
    return dict(accuracy=accuracy_score(y, yhat), precision=precision_score(y, yhat, zero_division=0),
                recall=recall_score(y, yhat, zero_division=0), f1=f1_score(y, yhat, zero_division=0),
                auc=roc_auc_score(y, p), mcc=matthews_corrcoef(y, yhat))


def decision_cost(y, p, t, ratio):
    """Total cost = ratio * FN + FP  (FN = false approval, FP = false rejection)."""
    yhat = p >= t
    fn = int(((~yhat) & (y == 1)).sum())
    fp = int((yhat & (y == 0)).sum())
    return ratio * fn + fp, fn, fp


THRESHOLDS = np.round(np.arange(0.05, 0.9501, 0.01), 2)


def cost_curve(y, p, ratio):
    y = np.asarray(y)
    return np.array([decision_cost(y, p, t, ratio)[0] for t in THRESHOLDS]) / len(y)


def best_threshold(y, p, ratio):
    c = cost_curve(y, p, ratio)
    return float(THRESHOLDS[int(np.argmin(c))])


def permutation_importance(pipe, X, y, n_permutations=10, seed=SEED):
    """Permutation importance: model-agnostic feature importance.
    
    Permutes each feature and measures the decrease in metric.
    Works for any classifier model.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    pipe.fit(X_tr, y_tr)
    
    # Baseline score
    p_baseline = pipe.predict_proba(X_te)[:, 1]
    baseline = roc_auc_score(y_te, p_baseline)
    
    importances = {}
    for col in X_te.columns:
        X_permuted = X_te.copy()
        X_permuted[col] = np.random.permutation(X_permuted[col].values)
        p_perm = pipe.predict_proba(X_permuted)[:, 1]
        perm_score = roc_auc_score(y_te, p_perm)
        importances[col] = baseline - perm_score  # decrease in AUC
    
    return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))


def psi_calc(expected, actual, n_bins=10):
    """Population Stability Index (PSI) between two distributions.
    
    Commonly used in credit risk monitoring to detect distribution shifts.
    PSI < 0.2: No significant change
    PSI 0.2-0.25: Moderate change - investigate
    PSI > 0.25: Significant change - take action
    """
    import pandas as pd
    import numpy as np
    
    expected = np.asarray(expected).reshape(-1)
    actual = np.asarray(actual).reshape(-1)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_counts_expected = pd.cut(expected, bins=bin_edges).value_counts().sort_index()
    bin_counts_actual = pd.cut(actual, bins=bin_edges).value_counts().sort_index()
    
    # Convert to proportions, handling missing bins
    all_bins = pd.Index([f"({bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}]" 
                         for i in range(n_bins)],
                        dtype='object')
    prop_expected = bin_counts_expected.reindex(all_bins, fill_value=0) / len(expected)
    prop_actual = bin_counts_actual.reindex(all_bins, fill_value=0) / len(actual)
    
    # PSI calculation (only bins with expected > 0)
    psi = 0.0
    for bin_label in all_bins:
        pe = prop_expected[bin_label]
        pa = prop_actual[bin_label]
        if pe > 0:
            psi += (pa - pe) * np.log(pa / pe)
    
    # Interpretation
    if psi < 0.2:
        interpretation = "No significant change"
    elif psi < 0.25:
        interpretation = "Moderate change - investigate"
    else:
        interpretation = "Significant change - take action"
    
    return {
        "psi": psi,
        "interpretation": interpretation,
        "detailed": {
            "expected_props": prop_expected.to_dict(),
            "actual_props": prop_actual.to_dict()
        }
    }


def learning_curve_analysis(pipe, X, y, train_sizes=np.linspace(0.1, 1.0, 5), 
                            cv=5, seed=SEED):
    """Learning curve analysis: model performance vs training set size.
    
    Returns train/test AUC scores across different training set proportions.
    """
    from sklearn.model_selection import learning_curve
    from sklearn.metrics import roc_auc_score
    import numpy as np
    
    train_sizes, train_scores, test_scores = learning_curve(
        pipe, X, y, cv=cv, scoring="roc_auc",
        train_sizes=train_sizes, random_state=seed
    )
    
    return {
        "train_sizes": train_sizes,
        "train_mean": np.mean(train_scores, axis=1),
        "train_std": np.std(train_scores, axis=1),
        "test_mean": np.mean(test_scores, axis=1),
        "test_std": np.std(test_scores, axis=1),
    }


def calibration_intercept_slope(y, p):
    """Calculate calibration intercept and slope.
    
    Fits: y = intercept + slope * p (via logistic regression calibration)
    Ideal: intercept=0, slope=1
    """
    import numpy as np
    
    p = np.asarray(p).reshape(-1)
    y = np.asarray(y).reshape(-1)
    
    n = len(y)
    p_mean = p.mean()
    y_mean = y.mean()
    
    # slope = cov(y, p) / var(p)
    # intercept = y_mean - slope * p_mean
    cov_lp = ((y - y_mean) * (p - p_mean)).mean()
    var_p = ((p - p_mean) ** 2).mean()
    
    if var_p == 0:
        return 0.0, 1.0
    
    slope = cov_lp / var_p
    intercept = y_mean - slope * p_mean
    
    return float(intercept), float(slope)


def decision_curve_analysis(y, p, ratios=range(1, 21)):
    """Decision Curve Analysis: net benefit across threshold probabilities.
    
    Used in medical/credit risk to evaluate clinical usefulness
    of a model across different risk tolerance levels.
    """
    import numpy as np
    
    p = np.asarray(p).reshape(-1)
    y = np.asarray(y).reshape(-1)
    
    results = {}
    for ratio in ratios:
        best_t = best_threshold(y, p, ratio)
        # Net benefit = TP/n - ratio * FP/n
        yhat = (p >= best_t).astype(int)
        tp = int((yhat & (y == 1)).sum())
        fp = int((yhat & (y == 0)).sum())
        n = len(y)
        net_benefit = tp / n - ratio * fp / n
        
        # Also calculate "treat all" and "treat none" baselines
        net_benefit_treat_all = y.mean() - ratio * (1 - y.mean())
        net_benefit_treat_none = 0.0
        
        results[ratio] = {
            "best_threshold": best_t,
            "net_benefit": net_benefit,
            "net_benefit_treat_all": net_benefit_treat_all,
            "net_benefit_treat_none": net_benefit_treat_none,
        }
    
    return results
    """Expected Calibration Error: weighted average of |avg_pred - avg_actual| per bin."""
    y = np.asarray(y)
    p = np.asarray(p)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p > lo) & (p <= hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = p[mask].mean()
        bin_acc = y[mask].mean()
        ece += (mask.sum() / len(y)) * abs(bin_conf - bin_acc)
    return float(ece)


def reliability_curve(y, p, n_bins=10):
    """Returns per-bin mean predicted probability and observed adverse rate."""
    y = np.asarray(y)
    p = np.asarray(p)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p > lo) & (p <= hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        rows.append(dict(bin=i, mean_pred=float(p[mask].mean()),
                         frac_pos=float(y[mask].mean()), n=int(mask.sum())))
    return pd.DataFrame(rows)


def ece_score(y, p, n_bins=10):
    """Expected Calibration Error: weighted average of |avg_pred - avg_actual| per bin."""
    y = np.asarray(y)
    p = np.asarray(p)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p > lo) & (p <= hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = p[mask].mean()
        bin_acc = y[mask].mean()
        ece += (mask.sum() / len(y)) * abs(bin_conf - bin_acc)
    return float(ece)


def reliability_curve(y, p, n_bins=10):
    """Returns per-bin mean predicted probability and observed adverse rate."""
    y = np.asarray(y)
    p = np.asarray(p)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p > lo) & (p <= hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        rows.append(dict(bin=i, mean_pred=float(p[mask].mean()),
                         frac_pos=float(y[mask].mean()), n=int(mask.sum())))
    return pd.DataFrame(rows)
