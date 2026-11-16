"""Improved loan default pipeline: feature engineering, Optuna tuning (50+ trials),
stacking ensemble, calibration, cost-sensitive thresholds, significance testing.

Writes results to improved_* files (never overwrites existing CSVs).

    python src/run_improved.py            # full run
    python src/run_improved.py --quick    # fast smoke test (fewer trials)

Protocol:
  1. Stratified 60/20/20 split before any preprocessing.
  2. Leakage audit on training split only -> F_clean (flagged columns removed).
  3. Feature engineering from F_clean columns only (row-wise, no fitted statistics).
     Keep a feature only if it improves validation AUC.
  4. Optuna TPE tuning (50+ trials per model) on validation AUC with early stopping.
  5. Stacking (LogReg meta-learner over OOF predictions) + weighted blend.
  6. Isotonic calibration; report Brier, log loss, ECE before/after.
  7. Cost-sensitive threshold: ratios 2,5,10,20 vs theory 1/(r+1),
     vs approve-all and reject-all baselines on test split.
  8. Repeated 5-fold CV + Nadeau-Bengio corrected t-test with Holm adjustment
     vs LightGBM baseline.

All outputs written to results/improved_*.csv (existing CSVs untouched).
"""
import argparse
import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import StackingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, log_loss, matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (SEED, THRESHOLDS, best_threshold, cost_curve, decision_cost,
                    ece_score, leakage_audit, load_default, make_estimator,
                    make_pipeline, make_preprocessor, metrics, reliability_curve,
                    split_60_20_20)

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
T0 = time.time()

RATIOS = [2, 5, 10, 20]
N_TRIALS = 50          # Optuna trials per model (full run)
N_CV_REPEATS = 3       # Repeated CV for significance testing
N_CV_FOLDS = 5


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ============================================================================ step A: feature engineering
def add_features(X):
    """Row-wise engineered features from F_clean columns only.
    No fitted statistics, so no leakage across splits."""
    X = X.copy()
    new_cols = []

    # Ratios
    if "loan_amount" in X.columns and "income" in X.columns:
        X["loan_to_income"] = X["loan_amount"] / X["income"].replace(0, np.nan)
        new_cols.append("loan_to_income")
    if "loan_amount" in X.columns and "term" in X.columns:
        X["loan_per_term"] = X["loan_amount"] / X["term"].replace(0, np.nan)
        new_cols.append("loan_per_term")
    if "income" in X.columns and "term" in X.columns:
        X["income_per_term"] = X["income"] / X["term"].replace(0, np.nan)
        new_cols.append("income_per_term")

    # Log transforms of skewed amounts
    if "income" in X.columns:
        X["log_income"] = np.log1p(X["income"].clip(lower=0))
        new_cols.append("log_income")
    if "loan_amount" in X.columns:
        X["log_loan_amount"] = np.log1p(X["loan_amount"].clip(lower=0))
        new_cols.append("log_loan_amount")

    # Binned credit score (categorical, one-hot encoded by preprocessor)
    if "Credit_Score" in X.columns:
        X["credit_score_bin"] = pd.cut(
            X["Credit_Score"],
            bins=[0, 580, 670, 740, 800, 850],
            labels=["poor", "fair", "good", "very_good", "exceptional"],
            include_lowest=True,
        ).astype(str)
        new_cols.append("credit_score_bin")

    X[new_cols] = X[new_cols].replace([np.inf, -np.inf], np.nan)
    return X, new_cols


def select_features(Xtr, ytr, Xva, yva, base_cols, new_cols):
    """Keep only engineered features that improve validation AUC on LightGBM."""
    if not new_cols:
        return base_cols, []
    kept = []
    base_pipe = make_pipeline(Xtr[base_cols], "LightGBM", None).fit(Xtr[base_cols], ytr)
    base_auc = roc_auc_score(yva, base_pipe.predict_proba(Xva[base_cols])[:, 1])
    for c in new_cols:
        cols = base_cols + kept + [c]
        pipe = make_pipeline(Xtr[cols], "LightGBM", None).fit(Xtr[cols], ytr)
        auc = roc_auc_score(yva, pipe.predict_proba(Xva[cols])[:, 1])
        if auc > base_auc + 1e-4:
            kept.append(c)
            base_auc = auc
            log(f"  feature kept: {c} (val AUC {base_auc:.4f})")
        else:
            log(f"  feature dropped: {c} (val AUC {auc:.4f} <= {base_auc:.4f})")
    return base_cols + kept, kept


# ============================================================================ step B: Optuna tuning
def optuna_tune(Xtr, ytr, Xva, yva, name, n_trials):
    """Optuna TPE optimisation on validation AUC with early stopping."""
    if not OPTUNA_AVAILABLE:
        log(f"  Optuna not available; skipping {name}")
        return None, None

    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier

    def objective(trial):
        si = trial.suggest_int
        sf = trial.suggest_float
        if name == "LightGBM":
            params = dict(
                n_estimators=si("n_estimators", 100, 1000),
                learning_rate=sf("learning_rate", 0.01, 0.2, log=True),
                num_leaves=si("num_leaves", 15, 255),
                min_child_samples=si("min_child_samples", 5, 100),
                subsample=sf("subsample", 0.6, 1.0),
                colsample_bytree=sf("colsample_bytree", 0.6, 1.0),
                reg_lambda=sf("reg_lambda", 1e-3, 10.0, log=True),
                reg_alpha=sf("reg_alpha", 1e-3, 10.0, log=True),
            )
            est = LGBMClassifier(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                num_leaves=params["num_leaves"],
                min_child_samples=params["min_child_samples"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                reg_lambda=params["reg_lambda"],
                reg_alpha=params["reg_alpha"],
                random_state=SEED, n_jobs=2, verbose=-1,
            )
        elif name == "XGBoost":
            params = dict(
                n_estimators=si("n_estimators", 100, 1000),
                max_depth=si("max_depth", 3, 12),
                learning_rate=sf("learning_rate", 0.01, 0.2, log=True),
                subsample=sf("subsample", 0.6, 1.0),
                colsample_bytree=sf("colsample_bytree", 0.6, 1.0),
                reg_lambda=sf("reg_lambda", 1e-3, 10.0, log=True),
                reg_alpha=sf("reg_alpha", 1e-3, 10.0, log=True),
            )
            est = XGBClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                reg_lambda=params["reg_lambda"],
                reg_alpha=params["reg_alpha"],
                random_state=SEED, n_jobs=2, verbosity=0,
                eval_metric="auc",
            )
        elif name == "CatBoost":
            params = dict(
                iterations=si("iterations", 100, 1000),
                depth=si("depth", 3, 12),
                learning_rate=sf("learning_rate", 0.01, 0.2, log=True),
                l2_leaf_reg=sf("l2_leaf_reg", 1e-3, 10.0, log=True),
                border_count=si("border_count", 32, 255),
            )
            est = CatBoostClassifier(
                iterations=params["iterations"],
                depth=params["depth"],
                learning_rate=params["learning_rate"],
                l2_leaf_reg=params["l2_leaf_reg"],
                border_count=params["border_count"],
                random_state=SEED, verbose=0,
            )
        else:
            return 0.5

        pipe = Pipeline([("pre", make_preprocessor(Xtr)), ("clf", est)])
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xva)[:, 1]
        return float(roc_auc_score(yva, p))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    best_auc = study.best_value
    log(f"  {name}: best val AUC {best_auc:.4f} with {len(study.trials)} trials")
    return best_params, best_auc


# ============================================================================ step C: stacking + blend
def oof_predict(X, y, name, params, n_folds=5):
    """Out-of-fold probabilities via stratified CV. Fitted on training folds only."""
    oof = np.zeros(len(y))
    fold_aucs = []
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=SEED).split(X, y):
        pipe = make_pipeline(X.iloc[tr], name, params).fit(X.iloc[tr], y.iloc[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        fold_aucs.append(roc_auc_score(y.iloc[te], p))
    return oof, float(np.mean(fold_aucs)), float(np.std(fold_aucs))


def stacking_predict(Xtr, ytr, Xva, Xte, model_names, params_dict, n_folds=5):
    """StackingClassifier with logistic regression meta-learner over OOF predictions."""
    estimators = []
    for m in model_names:
        est = make_estimator(m, params_dict.get(m), None, SEED)
        estimators.append((m, est))

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000, random_state=SEED),
        cv=StratifiedKFold(n_folds, shuffle=True, random_state=SEED),
        stack_method="predict_proba",
        n_jobs=1,
        passthrough=False,
    )
    pre = make_preprocessor(Xtr)
    pipe = Pipeline([("pre", pre), ("stack", stack)])
    pipe.fit(Xtr, ytr)
    p_va = pipe.predict_proba(Xva)[:, 1]
    p_te = pipe.predict_proba(Xte)[:, 1]
    return p_va, p_te


def weighted_blend_predict(Xtr, ytr, Xva, yva, Xte, model_names, params_dict):
    """Simple weighted blend: weight = validation AUC of each model."""
    weights, p_va_list, p_te_list = [], [], []
    for m in model_names:
        pipe = make_pipeline(Xtr, m, params_dict.get(m)).fit(Xtr, ytr)
        p_va = pipe.predict_proba(Xva)[:, 1]
        p_te = pipe.predict_proba(Xte)[:, 1]
        w = roc_auc_score(yva, p_va)  # use val AUC as weight
        weights.append(w)
        p_va_list.append(p_va)
        p_te_list.append(p_te)
    weights = np.array(weights)
    weights = weights / weights.sum()
    p_va = np.average(np.stack(p_va_list, axis=1), weights=weights, axis=1)
    p_te = np.average(np.stack(p_te_list, axis=1), weights=weights, axis=1)
    return p_va, p_te


# ============================================================================ step D: calibration
def calibrate_scores(oof, ydev, p_te):
    """Isotonic calibration fitted on OOF dev predictions. Returns calibrated test probs."""
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof, ydev)
    p_cal = cal.predict(p_te)
    return p_cal


# ============================================================================ step E: cost-sensitive threshold
def cost_analysis(ydev, oof_cal, yte, p_te, ratios):
    """Cost-optimal threshold per ratio vs theory 1/(r+1), vs approve-all/reject-all baselines.
    Threshold is chosen from (ydev, oof_cal) and applied to (yte, p_te)."""
    rows = []
    n = len(yte)
    n_adverse = int((yte == 1).sum())
    n_safe = n - n_adverse

    for r in ratios:
        t_star = best_threshold(ydev, oof_cal, r)
        t_theory = 1.0 / (r + 1)
        total_cost, fn, fp = decision_cost(yte, p_te, t_star, r)
        cost_per_applicant = total_cost / n

        _, fn_t, fp_t = decision_cost(yte, p_te, t_theory, r)
        cost_theory = (r * fn_t + fp_t) / n

        rows.append(dict(
            ratio=r, threshold_star=round(t_star, 4),
            threshold_theory=round(t_theory, 4),
            theory_gap=round(abs(t_star - t_theory), 4),
            total_cost=int(total_cost), fn=int(fn), fp=int(fp),
            cost_per_applicant=round(cost_per_applicant, 4),
            cost_per_applicant_theory=round(cost_theory, 4),
        ))

    # Baseline at t=0.5 for ratio 10
    _, fn_05, fp_05 = decision_cost(yte, p_te, 0.5, 10)
    rows.insert(0, dict(
        ratio=10, threshold_star=0.5, threshold_theory=1/11, theory_gap=abs(0.5 - 1/11),
        total_cost=int(10 * fn_05 + fp_05), fn=int(fn_05), fp=int(fp_05),
        cost_per_applicant=round((10 * fn_05 + fp_05) / n, 4),
        cost_per_applicant_theory=np.nan,
    ))

    # Approve-all and reject-all baselines
    rows.append(dict(
        ratio=10, threshold_star="approve-all", threshold_theory=np.nan, theory_gap=np.nan,
        total_cost=int(10 * n_adverse), fn=int(n_adverse), fp=0,
        cost_per_applicant=round(10 * n_adverse / n, 4), cost_per_applicant_theory=np.nan,
    ))
    rows.append(dict(
        ratio=10, threshold_star="reject-all", threshold_theory=np.nan, theory_gap=np.nan,
        total_cost=int(n_safe), fn=0, fp=int(n_safe),
        cost_per_applicant=round(n_safe / n, 4), cost_per_applicant_theory=np.nan,
    ))

    return pd.DataFrame(rows)


# ============================================================================ step F: significance
def corrected_t(diffs, n_train, n_test, k):
    """Nadeau-Bengio corrected resampled t-test."""
    d = np.asarray(diffs)
    mean_d = d.mean()
    var_d = d.var(ddof=1)
    if var_d == 0:
        return 0.0, 1.0
    t = mean_d / np.sqrt((1.0 / k + n_test / n_train) * var_d)
    p = float(2 * stats.t.sf(abs(t), k - 1))
    return float(t), p


def holm_adjust(pvals):
    """Holm step-down adjustment."""
    pvals = np.asarray(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    n = len(ranked)
    adj = ranked * (n - np.arange(n))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out


def significance_test(Xdev, ydev, baseline_name, baseline_params, other_cfgs, n_repeats=3):
    """Repeated 5-fold CV on identical folds; corrected t-test vs baseline with Holm adjustment."""
    n_train = int(len(Xdev) * 0.8)
    n_test = len(Xdev) - n_train

    rows = []
    pvals = []
    names = []
    for rep in range(n_repeats):
        # Fresh split inside the loop (Rule: split inside the loop, not once before)
        skf = list(StratifiedKFold(N_CV_FOLDS, shuffle=True, random_state=SEED + rep).split(Xdev, ydev))
        fold_aucs = {baseline_name: []}
        for name in other_cfgs:
            fold_aucs[name] = []

        for fold_idx, (tr, te) in enumerate(skf):
            Xtr_f, Xte_f = Xdev.iloc[tr], Xdev.iloc[te]
            ytr_f, yte_f = ydev.iloc[tr], ydev.iloc[te]

            pipe_b = make_pipeline(Xtr_f, baseline_name, baseline_params).fit(Xtr_f, ytr_f)
            p_b = pipe_b.predict_proba(Xte_f)[:, 1]
            fold_aucs[baseline_name].append(roc_auc_score(yte_f, p_b))

            for name, params in other_cfgs.items():
                pipe = make_pipeline(Xtr_f, name, params).fit(Xtr_f, ytr_f)
                p = pipe.predict_proba(Xte_f)[:, 1]
                fold_aucs[name].append(roc_auc_score(yte_f, p))

        for name in other_cfgs:
            diffs = np.array(fold_aucs[name]) - np.array(fold_aucs[baseline_name])
            t, p = corrected_t(diffs, n_train, n_test, N_CV_FOLDS)
            rows.append(dict(
                model=name, baseline=baseline_name,
                mean_auc=round(np.mean(fold_aucs[name]), 4),
                baseline_mean_auc=round(np.mean(fold_aucs[baseline_name]), 4),
                mean_diff=round(np.mean(diffs), 4),
                t_stat=round(t, 4), p_value=round(p, 6),
            ))
            pvals.append(p)
            names.append(name)

    adj_pvals = holm_adjust(np.array(pvals))
    for i, name in enumerate(names):
        rows[i]["p_value_holm"] = round(float(adj_pvals[i]), 6)
        rows[i]["significant_holm"] = bool(adj_pvals[i] < 0.05)

    return pd.DataFrame(rows)


# ============================================================================ main pipeline
def run_improved(quick=False):
    X, y = load_default(os.path.join(ROOT, "data", "Loan_Default.csv"))
    if quick:
        X, _, y, _ = train_test_split(X, y, train_size=0.15, stratify=y, random_state=SEED)

    Xtr, Xva, Xte, ytr, yva, yte = split_60_20_20(X, y)
    log(f"default: train={len(Xtr)} val={len(Xva)} test={len(Xte)}; adverse rate {y.mean():.3f}")

    # Leakage audit on training split only
    audit, flagged = leakage_audit(Xtr, ytr)
    log(f"leakage audit flagged {len(flagged)} columns: {flagged}")
    base_cols = [c for c in Xtr.columns if c not in flagged]
    log(f"F_clean: {len(base_cols)} features")

    # Save leakage audit
    audit.insert(0, "dataset", "default")
    audit.to_csv(f"{RES}/improved_default_leakage_audit.csv", index=False)

    # ---- Step A: feature engineering ----
    Xtr_fe, new_cols = add_features(Xtr)
    Xva_fe = add_features(Xva)[0]
    Xte_fe = add_features(Xte)[0]

    # Align columns
    for c in new_cols:
        if c not in Xva_fe.columns:
            Xva_fe[c] = np.nan
        if c not in Xte_fe.columns:
            Xte_fe[c] = np.nan

    feat_cols, kept = select_features(Xtr_fe, ytr, Xva_fe, yva, base_cols, new_cols)
    log(f"final feature set: {len(feat_cols)} columns ({len(kept)} engineered kept: {kept})")

    # Save feature engineering log
    fe_log = pd.DataFrame([dict(feature=c, kept=(c in kept), source="engineered" if c in new_cols else "base")
                           for c in feat_cols])
    fe_log.to_csv(f"{RES}/improved_default_feature_engineering.csv", index=False)

    # ---- Step B: Optuna tuning ----
    tuned = {}
    val_aucs = {}
    if OPTUNA_AVAILABLE:
        n_trials = 10 if quick else N_TRIALS
        for m in ("LightGBM", "XGBoost", "CatBoost"):
            params, auc = optuna_tune(Xtr_fe[feat_cols], ytr, Xva_fe[feat_cols], yva, m, n_trials)
            tuned[m] = params
            val_aucs[m] = auc
    else:
        from common import GRIDS
        for m in ("LightGBM", "XGBoost", "CatBoost"):
            best, log_df = tune_grid(Xtr_fe[feat_cols], ytr, Xva_fe[feat_cols], yva, m)
            tuned[m] = best
            val_aucs[m] = float(log_df.val_auc.max())

    # Save tuning results
    tuning_rows = []
    for m in ("LightGBM", "XGBoost", "CatBoost"):
        tuning_rows.append(dict(dataset="default", feature_set="F_clean+FE", model=m,
                                val_auc=round(val_aucs.get(m, np.nan), 4),
                                params=str(tuned.get(m, {}))))
    pd.DataFrame(tuning_rows).to_csv(f"{RES}/improved_default_tuning.csv", index=False)
    log(f"Optuna tuning done: {val_aucs}")

    # ---- Step C: stacking + weighted blend ----
    model_names = ["LightGBM", "XGBoost", "CatBoost"]

    # Stacking
    p_va_stack, p_te_stack = stacking_predict(
        Xtr_fe[feat_cols], ytr, Xva_fe[feat_cols], Xte_fe[feat_cols],
        model_names, tuned)
    stack_va_auc = roc_auc_score(yva, p_va_stack)
    stack_te_auc = roc_auc_score(yte, p_te_stack)
    log(f"Stacking: val AUC {stack_va_auc:.4f} test AUC {stack_te_auc:.4f}")

    # Weighted blend
    p_va_blend, p_te_blend = weighted_blend_predict(
        Xtr_fe[feat_cols], ytr, Xva_fe[feat_cols], yva, Xte_fe[feat_cols],
        model_names, tuned)
    blend_va_auc = roc_auc_score(yva, p_va_blend)
    blend_te_auc = roc_auc_score(yte, p_te_blend)
    log(f"Weighted blend: val AUC {blend_va_auc:.4f} test AUC {blend_te_auc:.4f}")

    # Save ensemble results
    ens_rows = []
    for m in model_names:
        pipe = make_pipeline(Xtr_fe[feat_cols], m, tuned.get(m)).fit(Xtr_fe[feat_cols], ytr)
        p_va = pipe.predict_proba(Xva_fe[feat_cols])[:, 1]
        p_te = pipe.predict_proba(Xte_fe[feat_cols])[:, 1]
        ens_rows.append(dict(dataset="default", model=f"{m}(Optuna)",
                             val_auc=round(roc_auc_score(yva, p_va), 4),
                             test_auc=round(roc_auc_score(yte, p_te), 4),
                             **metrics(yte, p_te)))
    ens_rows.append(dict(dataset="default", model="Stacking(LogReg meta)",
                         val_auc=round(stack_va_auc, 4), test_auc=round(stack_te_auc, 4),
                         **metrics(yte, p_te_stack)))
    ens_rows.append(dict(dataset="default", model="WeightedBlend",
                         val_auc=round(blend_va_auc, 4), test_auc=round(blend_te_auc, 4),
                         **metrics(yte, p_te_blend)))
    pd.DataFrame(ens_rows).to_csv(f"{RES}/improved_default_models.csv", index=False)

    # ---- Step D: calibration of best single model ----
    best_model = max(model_names, key=lambda m: val_aucs.get(m, 0))
    log(f"Best single model: {best_model} (val AUC {val_aucs[best_model]:.4f})")

    # Fit final model on train+val
    Xdev = pd.concat([Xtr_fe[feat_cols], Xva_fe[feat_cols]])
    ydev = pd.concat([ytr, yva])

    # OOF predictions for calibration (on combined dev set = train+val)
    oof, oof_va_auc, oof_va_std = oof_predict(
        Xdev, ydev, best_model, tuned.get(best_model), N_CV_FOLDS)
    log(f"OOF {best_model}: val AUC {oof_va_auc:.4f} +/- {oof_va_std:.4f}")

    # Compute oof_cal using IsotonicRegression
    oof_cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof, ydev.values).predict(oof)

    # Fit final pipeline on train+val (train+val combined = dev)
    final_pipe = make_pipeline(Xdev, best_model, tuned.get(best_model)).fit(Xdev, ydev)
    p_te_raw = final_pipe.predict_proba(Xte_fe[feat_cols])[:, 1]

    # Calibrate
    p_te_cal = calibrate_scores(oof, ydev.values, p_te_raw)

    # Metrics before/after calibration
    cal_rows = []
    for ver, p in (("raw", p_te_raw), ("calibrated", p_te_cal)):
        m = metrics(yte, p)
        cal_rows.append(dict(
            dataset="default", model=best_model, version=ver,
            auc=m["auc"], brier=round(brier_score_loss(yte, p), 4),
            log_loss=round(log_loss(yte, np.clip(p, 1e-6, 1 - 1e-6)), 4),
            ece=round(ece_score(yte, p), 4),
            accuracy=m["accuracy"], precision=m["precision"], recall=m["recall"],
            f1=m["f1"], mcc=m["mcc"]))
    pd.DataFrame(cal_rows).to_csv(f"{RES}/improved_default_calibration.csv", index=False)
    log(f"Calibration ({best_model}): Brier {cal_rows[0]['brier']} -> {cal_rows[1]['brier']}; "
        f"ECE {cal_rows[0]['ece']} -> {cal_rows[1]['ece']}")

    # Save reliability curve
    rel = reliability_curve(yte.values, p_te_cal)
    rel.insert(0, "dataset", "default")
    rel.to_csv(f"{RES}/improved_default_reliability.csv", index=False)


    p_best = p_te_cal  # use calibrated scores
    cost_df = cost_analysis(ydev, oof_cal, yte.values, p_te_cal, RATIOS)
    cost_df.insert(0, "dataset", "default")
    cost_df.to_csv(f"{RES}/improved_default_cost_threshold.csv", index=False)
    log("Cost-sensitive threshold analysis done")

    # ---- SAVE MODEL FOR PRODUCTION ----
    MODEL_DIR = os.path.join(ROOT, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Fit final pipeline on train+val (train+val combined = dev)
    final_pipe = make_pipeline(Xdev, best_model, tuned.get(best_model)).fit(Xdev, ydev)

    # Get optimal threshold for ratio 10:1 from validation data
    # Use OOF dev predictions for threshold selection (oof_cal, not re-fitting)
    t_star_10 = best_threshold(ydev, oof_cal, 10)

    # Save artifacts
    joblib.dump(final_pipe, os.path.join(MODEL_DIR, "best_model_pipeline.joblib"))
    joblib.dump(feat_cols, os.path.join(MODEL_DIR, "feature_columns.joblib"))
    joblib.dump(t_star_10, os.path.join(MODEL_DIR, "optimal_threshold_10_1.joblib"))
    
    # Save metadata with oof_cal reference
    metadata = {
        "best_model": best_model,
        "feature_set": "F_clean+FE",
        "n_features": len(feat_cols),
        "n_base_features": len(base_cols),
        "n_engineered_features": len(kept),
        "test_auc": round(float(roc_auc_score(yte, p_best)), 4),
        "test_mcc": round(float(matthews_corrcoef(yte, (p_best >= t_star_10).astype(int))), 4),
        "optimal_threshold_10_1": round(float(t_star_10), 4),
        "calibrated": True,
        "leaking_columns_removed": flagged,
        "seed": SEED,
    }
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    log(f"Model saved to {MODEL_DIR}/ (pipeline, features, threshold, metadata)")

    # ---- Step F: significance testing ----
    other_cfgs = {m: tuned.get(m) for m in model_names if m != best_model}
    sig_df = significance_test(
        Xdev, ydev, best_model, tuned.get(best_model), other_cfgs, n_repeats=N_CV_REPEATS)
    sig_df.insert(0, "dataset", "default")
    sig_df.to_csv(f"{RES}/improved_default_significance.csv", index=False)
    log(f"Significance testing done (vs {best_model})")

    # ---- Final test metrics table ----
    other_cfgs = {m: tuned.get(m) for m in model_names if m != best_model}
    sig_df = significance_test(
        Xdev, ydev, best_model, tuned.get(best_model), other_cfgs, n_repeats=N_CV_REPEATS)
    sig_df.insert(0, "dataset", "default")
    sig_df.to_csv(f"{RES}/improved_default_significance.csv", index=False)
    log(f"Significance testing done (vs {best_model})")

    # ---- Final test metrics table ----
    final_rows = []
    for m in model_names:
        pipe = make_pipeline(Xdev, m, tuned.get(m)).fit(Xdev, ydev)
        p_te_m = pipe.predict_proba(Xte_fe[feat_cols])[:, 1]
        final_rows.append(dict(dataset="default", feature_set="F_clean+FE", model=m,
                               **metrics(yte, p_te_m)))
    # Stacking
    final_rows.append(dict(dataset="default", feature_set="F_clean+FE", model="Stacking",
                           **metrics(yte, p_te_stack)))
    # Weighted blend
    final_rows.append(dict(dataset="default", feature_set="F_clean+FE", model="WeightedBlend",
                           **metrics(yte, p_te_blend)))
    # Calibrated best
    final_rows.append(dict(dataset="default", feature_set="F_clean+FE", model=f"{best_model}(calibrated)",
                           **metrics(yte, p_best)))
    pd.DataFrame(final_rows).to_csv(f"{RES}/improved_default_test_metrics.csv", index=False)

    log("ALL DONE (improved pipeline)")


def tune_grid(Xtr, ytr, Xva, yva, name):
    """Fallback grid search (used if Optuna unavailable)."""
    from common import GRIDS
    rows = []
    for params in GRIDS[name]:
        pipe = make_pipeline(Xtr, name, params).fit(Xtr, ytr)
        rows.append(dict(model=name, params=str(params),
                         train_auc=roc_auc_score(ytr, pipe.predict_proba(Xtr)[:, 1]),
                         val_auc=roc_auc_score(yva, pipe.predict_proba(Xva)[:, 1])))
    log_df = pd.DataFrame(rows)
    best = GRIDS[name][int(log_df.val_auc.values.argmax())]
    return best, log_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    run_improved(quick=a.quick)


if __name__ == "__main__":
    main()