"""Course diagnostics on the final (headline) model of each dataset.

    python src/run_course_methods.py            # full run (~3-5 min on 2 cores)
    python src/run_course_methods.py --quick    # smoke test on a 15% sample of Dataset 1

Run AFTER run_experiments.py: the best model and its hyper-parameters are read from
results/<dataset>_test_metrics.csv (highest validation AUC on the headline feature set).

For each dataset it uses the same stratified 60/20/20 split and leakage audit as the main pipeline, then:
  1. Permutation importance   - model fitted on train+val, columns shuffled on the TEST split.
  2. Population Stability Idx - score PSI (validation vs test scores of a train-only model) and
                                per-feature PSI (train vs test) for every numeric column.
  3. Learning curve           - train vs cross-validated AUC as training size grows (bias-variance).
  4. Calibration slope/int.   - logistic recalibration of test-set probabilities (ideal a=0, b=1).
  5. Decision curve analysis  - net benefit of the model vs reject-all / approve-all on the test split.

Outputs: results/<dataset>_course_*.csv  (figures are drawn by make_figures.py from these CSVs).
"""
import argparse
import ast
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from common import (SEED, calibration_intercept_slope, decision_curve_analysis, leakage_audit,
                    learning_curve_analysis, load_approval, load_default, make_pipeline,
                    permutation_importance, psi_calc, split_60_20_20)

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# Final models diagnosed here. Dataset 1: CatBoost (the extension study's final model; settings from the
# CatBoost validation grid). Dataset 2: the best baseline model by validation AUC (logistic regression).
FIXED = {"default": ("CatBoost", {"depth": 8, "learning_rate": 0.05, "iterations": 200})}


def headline_model(ds, feature_set):
    """Final model for the dataset: fixed for Dataset 1, else best by VALIDATION AUC (never test AUC)."""
    if ds in FIXED:
        return FIXED[ds]
    tm = pd.read_csv(f"{RES}/{ds}_test_metrics.csv")
    tm = tm[tm.feature_set == feature_set]
    row = tm.loc[tm.val_auc.idxmax()]
    return row.model, ast.literal_eval(row.params)


def run(ds, X, y, feature_set, n_perm, lc_rows, lc_cv):
    log(f"=== {ds}: feature set {feature_set}")
    Xtr, Xva, Xte, ytr, yva, yte = split_60_20_20(X, y)
    _, flagged = leakage_audit(Xtr, ytr)                       # audit on TRAINING data only
    cols = [c for c in X.columns if not (feature_set == "F_clean" and c in flagged)]
    if feature_set == "F_noCreditHistory":
        cols = [c for c in cols if c != "Credit_History"]
    Xtr, Xva, Xte = Xtr[cols], Xva[cols], Xte[cols]
    Xdev, ydev = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])
    name, params = headline_model(ds, feature_set)
    log(f"  model {name} {params}")

    # Final model: refit on train+val, scored once on the untouched test split
    final = make_pipeline(Xdev, name, params).fit(Xdev, ydev)
    p_te = final.predict_proba(Xte)[:, 1]

    # 1. Permutation importance on the test split
    pi = permutation_importance(final, Xte, yte, n_permutations=n_perm)
    pi.insert(0, "model", name)
    pi.insert(0, "dataset", ds)
    pi.to_csv(f"{RES}/{ds}_course_permutation_importance.csv", index=False)
    log(f"  permutation importance: top = {pi.feature.iloc[0]} ({pi.importance_mean.iloc[0]:.4f} AUC)")

    # 2. PSI. Score PSI compares two OUT-OF-SAMPLE score sets (validation vs test) of a train-only model,
    #    which mimics "development sample vs new applicants". Feature PSI compares train vs test inputs.
    m_tr = make_pipeline(Xtr, name, params).fit(Xtr, ytr)
    s = psi_calc(m_tr.predict_proba(Xva)[:, 1], m_tr.predict_proba(Xte)[:, 1])
    rows = [dict(dataset=ds, variable="model_score", kind="score", psi=s["psi"], interpretation=s["interpretation"])]
    for c in cols:
        if pd.api.types.is_numeric_dtype(Xtr[c]) and Xtr[c].nunique() > 2:
            r = psi_calc(Xtr[c], Xte[c])
            rows.append(dict(dataset=ds, variable=c, kind="feature", psi=r["psi"], interpretation=r["interpretation"]))
    psi_df = pd.DataFrame(rows)
    psi_df.to_csv(f"{RES}/{ds}_course_psi.csv", index=False)
    log(f"  score PSI = {s['psi']:.4f} ({s['interpretation']}); max feature PSI = {psi_df.psi[1:].max():.4f}")

    # 3. Learning curve on train+val (sub-sampled for the large dataset to keep runtime reasonable)
    Xl, yl = Xdev, ydev
    if lc_rows and len(Xdev) > lc_rows:
        Xl, _, yl, _ = train_test_split(Xdev, ydev, train_size=lc_rows, stratify=ydev, random_state=SEED)
    lc = learning_curve_analysis(make_pipeline(Xl, name, params), Xl, yl, cv=lc_cv)
    lc.insert(0, "model", name)
    lc.insert(0, "dataset", ds)
    lc.to_csv(f"{RES}/{ds}_course_learning_curve.csv", index=False)
    gap = lc.train_auc_mean.iloc[-1] - lc.cv_auc_mean.iloc[-1]
    log(f"  learning curve: final train-CV AUC gap = {gap:.4f}")

    # 4. Calibration intercept / slope of the final model's test probabilities (plus LogReg for contrast)
    cal = []
    a, b = calibration_intercept_slope(yte, p_te)
    cal.append(dict(dataset=ds, model=name, intercept=a, slope=b, mean_pred=p_te.mean(), event_rate=yte.mean()))
    lr = make_pipeline(Xdev, "LogReg", {"C": 1.0}).fit(Xdev, ydev)
    p_lr = lr.predict_proba(Xte)[:, 1]
    a, b = calibration_intercept_slope(yte, p_lr)
    cal.append(dict(dataset=ds, model="LogReg", intercept=a, slope=b, mean_pred=p_lr.mean(), event_rate=yte.mean()))
    pd.DataFrame(cal).to_csv(f"{RES}/{ds}_course_calibration_slope.csv", index=False)
    log(f"  calibration {name}: intercept {cal[0]['intercept']:.3f}, slope {cal[0]['slope']:.3f}")

    # 5. Decision curve analysis on the test split
    dca = decision_curve_analysis(yte, p_te)
    dca.insert(0, "model", name)
    dca.insert(0, "dataset", ds)
    dca.to_csv(f"{RES}/{ds}_course_decision_curve.csv", index=False)
    r10 = dca.iloc[(dca.threshold - 1 / 11).abs().argmin()]
    log(f"  DCA at pt~1/11 (10:1): model NB {r10.net_benefit_model:.4f} vs reject-all {r10.net_benefit_reject_all:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    a = ap.parse_args()

    X, y = load_approval(os.path.join(a.data, "loan_data_set.csv"))
    run("approval", X, y, "F_full", n_perm=20, lc_rows=None, lc_cv=5)

    X, y = load_default(os.path.join(a.data, "Loan_Default.csv"))
    if a.quick:
        X, _, y, _ = train_test_split(X, y, train_size=0.15, stratify=y, random_state=SEED)
    run("default", X, y, "F_clean", n_perm=3 if a.quick else 5, lc_rows=10000 if a.quick else 40000, lc_cv=3)
    log("ALL DONE")


if __name__ == "__main__":
    main()
