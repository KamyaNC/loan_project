"""Train the final models from the raw CSVs, end to end, and print every step.

    python train.py                     # both datasets (Dataset 1 takes a few minutes)
    python train.py --dataset approval  # Dataset 2 only (seconds)
    python train.py --dataset default   # Dataset 1 only
    python train.py --no-save           # train and evaluate, but leave models/ untouched

What it does (the same leakage-safe protocol as the full experiment scripts, without the long grid searches):
  1. load the CSV               5. pick the cost-optimal threshold on out-of-fold training scores
  2. stratified 60/20/20 split  6. refit on train+validation, evaluate ONCE on the test split
  3. leakage audit (train only) 7. compare with approve-everyone / reject-everyone
  4. train candidate models,    8. save the model to models/ (used by src/predict.py)
     compare on validation AUC
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from common import (SEED, best_threshold, decision_cost, leakage_audit, load_approval, load_default,
                    make_pipeline, metrics, split_60_20_20)

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()
RATIO = 10                                  # a false approval costs 10x a false rejection

# Tuned settings found by the full experiments (results/*_tuning.csv, results/coursemodels_*.csv)
CANDIDATES = {
    "default": {
        "LogReg":       dict(C=1.0),
        "DecisionTree": None,                                   # depth 11, set below
        "LightGBM":     dict(num_leaves=31, min_child_samples=20, learning_rate=0.05),
        "CatBoost":     dict(depth=7, learning_rate=0.055, iterations=442, l2_leaf_reg=1.07),
    },
    "approval": {
        "LogReg":       dict(C=0.1),
        "DecisionTree": None,                                   # depth 3
        "RandForest":   dict(max_depth=12, min_samples_leaf=5),
        "CatBoost":     dict(depth=4, learning_rate=0.05, iterations=200),
    },
}
FINAL = {"default": "CatBoost", "approval": "LogReg"}


def step(n, text):
    print(f"\n[{time.time() - T0:6.1f}s] STEP {n}  {text}")


def line(text=""):
    print(f"          {text}")


def build(name, params, X, depth=None):
    """Pipeline = train-only preprocessing + model (decision tree built here; others from common.py)."""
    if name == "DecisionTree":
        from sklearn.pipeline import Pipeline
        from sklearn.tree import DecisionTreeClassifier
        from common import make_preprocessor
        return Pipeline([("pre", make_preprocessor(X)), ("clf", DecisionTreeClassifier(max_depth=depth, random_state=SEED))])
    return make_pipeline(X, name, params)


def show_table(rows, cols):
    w = [max(len(c), *(len(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])) for r in rows)) for c in cols]
    line("  ".join(c.ljust(x) for c, x in zip(cols, w)))
    line("  ".join("-" * x for x in w))
    for r in rows:
        line("  ".join((f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])).ljust(x) for c, x in zip(cols, w)))


def run(ds, save):
    print("\n" + "=" * 78)
    print(f"  {'DATASET 1: LOAN DEFAULT (y = 1: borrower defaulted)' if ds == 'default' else 'DATASET 2: LOAN APPROVAL (y = 1: lender rejected)'}")
    print("=" * 78)

    step(1, "Load data")
    if ds == "default":
        X, y = load_default(os.path.join(ROOT, "data", "Loan_Default.csv"))
    else:
        X, y = load_approval(os.path.join(ROOT, "data", "loan_data_set.csv"))
    line(f"{len(X):,} rows, {X.shape[1]} candidate columns, adverse rate {y.mean():.1%}")

    step(2, "Stratified 60/20/20 split (before any preprocessing)")
    Xtr, Xva, Xte, ytr, yva, yte = split_60_20_20(X, y)
    line(f"train {len(Xtr):,} | validation {len(Xva):,} | test {len(Xte):,}  (test is not touched until step 6)")

    step(3, "Leakage audit on the training rows only")
    _, flagged = leakage_audit(Xtr, ytr)
    cols = [c for c in X.columns if c not in flagged]
    line(f"flagged {len(flagged)} column(s): {', '.join(flagged) if flagged else 'none'}")
    if ds == "default":                      # row-wise engineered ratio (no fitted statistics, cannot leak)
        for d in (Xtr, Xva, Xte):
            d["loan_to_income"] = (d["loan_amount"] / d["income"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        cols.append("loan_to_income")
    line(f"model uses {len(cols)} columns")
    Xtr, Xva, Xte = Xtr[cols], Xva[cols], Xte[cols]

    step(4, "Train candidate models on train, compare on validation AUC")
    rows = []
    for name, params in CANDIDATES[ds].items():
        t = time.time()
        pipe = build(name, params, Xtr, depth=11 if ds == "default" else 3).fit(Xtr, ytr)
        tr = roc_auc_score(ytr, pipe.predict_proba(Xtr)[:, 1])
        va = roc_auc_score(yva, pipe.predict_proba(Xva)[:, 1])
        rows.append(dict(model=name, train_AUC=tr, val_AUC=va, gap=tr - va, seconds=f"{time.time() - t:.1f}"))
    show_table(rows, ["model", "train_AUC", "val_AUC", "gap", "seconds"])
    final = FINAL[ds]
    line(f"final model: {final}" + ("  (simplest model; differences on 123 validation rows are not significant)" if ds == "approval" else ""))

    Xdev, ydev = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])
    step(5, f"Cost-optimal threshold for a {RATIO}:1 cost ratio, from 5-fold out-of-fold scores on train+validation")
    oof = np.zeros(len(ydev))
    for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xdev, ydev)):
        m = build(final, CANDIDATES[ds][final], Xdev).fit(Xdev.iloc[a], ydev.iloc[a])
        oof[b] = m.predict_proba(Xdev.iloc[b])[:, 1]
        print(f"          fold {k + 1}/5 done", end="\r", flush=True)
    print(" " * 40, end="\r")
    t_star = best_threshold(ydev.values, oof, RATIO)
    line(f"t* = {t_star:.2f}   (theory for calibrated scores: 1/(r+1) = {1 / (RATIO + 1):.3f})")

    step(6, "Refit on train+validation, evaluate ONCE on the test split")
    model = build(final, CANDIDATES[ds][final], Xdev).fit(Xdev, ydev)
    p = model.predict_proba(Xte)[:, 1]
    res = [dict(threshold="0.50", **metrics(yte, p, 0.5)), dict(threshold=f"{t_star:.2f} (t*)", **metrics(yte, p, t_star))]
    show_table(res, ["threshold", "accuracy", "precision", "recall", "f1", "auc", "mcc"])

    step(7, f"Cost per applicant at {RATIO}:1 vs trivial policies")
    yv = yte.values
    pol = [("Approve everyone", 1.01), ("Model at t = 0.50", 0.5), ("Reject everyone", -0.01), (f"Model at t* = {t_star:.2f}", t_star)]
    crow = []
    for name, t in pol:
        c, fn, fp = decision_cost(yv, p, t, RATIO)
        crow.append(dict(policy=name, false_approvals=fn, false_rejections=fp, cost_per_applicant=c / len(yv),
                         approval_rate=f"{(p < t).mean():.1%}"))
    show_table(crow, ["policy", "false_approvals", "false_rejections", "cost_per_applicant", "approval_rate"])
    rej = crow[2]["cost_per_applicant"]
    line(f"saving vs rejecting everyone: {100 * (1 - crow[3]['cost_per_applicant'] / rej):.1f}%")

    step(8, "Save the model")
    if save and ds == "default":
        md = os.path.join(ROOT, "models")
        os.makedirs(md, exist_ok=True)
        joblib.dump(model, os.path.join(md, "best_model_pipeline.joblib"))
        joblib.dump(cols, os.path.join(md, "feature_columns.joblib"))
        joblib.dump(t_star, os.path.join(md, "optimal_threshold_10_1.joblib"))
        with open(os.path.join(md, "model_metadata.json"), "w") as f:
            json.dump(dict(best_model=final, feature_set="F_clean+loan_to_income", n_features=len(cols),
                           leaking_columns_removed=flagged, optimal_threshold_10_1=t_star,
                           test_auc=round(res[0]["auc"], 4), test_mcc_at_0_50=round(res[0]["mcc"], 4),
                           trained_by="train.py", seed=SEED), f, indent=2)
        line("saved to models/  ->  now run: python src/predict.py --input data/sample_applications.csv --output predictions.csv")
    elif ds == "default":
        line("skipped (--no-save)")
    else:
        line("not saved (src/predict.py scores Dataset 1 applications)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["default", "approval", "both"], default="both")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    for ds in (["approval", "default"] if a.dataset == "both" else [a.dataset]):
        run(ds, save=not a.no_save)
    print(f"\nDone in {time.time() - T0:.0f} s.")


if __name__ == "__main__":
    main()
