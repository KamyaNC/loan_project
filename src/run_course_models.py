"""Course algorithms (BITS F464, CLO2) under the same leakage-safe protocol as the main pipeline.

    python src/run_course_models.py            # full run (~10 min on 2 cores)
    python src/run_course_models.py --quick    # smoke test on a 15% sample of Dataset 1

Adds the classical models named in the course plan, each with a sweep over its main capacity knob so that
underfitting -> overfitting (bias-variance) and the effect of regularisation can be seen directly:

  Decision Tree           max_depth 1..20                     (week 4: decision tree learning)
  Logistic Regression     C, L1 vs L2 penalty                 (weeks 2 and 4: regularisation, discriminative models)
  LDA                     shrinkage 'auto' vs none            (week 3: discriminant functions / Fisher)
  Gaussian Naive Bayes    var_smoothing                       (week 8: Bayesian classifiers)
  k-NN                    k = 1..201                          (week 9: instance-based learning)
  Linear SVM / RBF SVM    C (and gamma for RBF)               (week 7: maximum margin, kernels)

Protocol: same stratified 60/20/20 split and leakage audit as run_experiments.py; preprocessing (impute, scale,
one-hot) is fitted on the training rows only, inside each pipeline. Every knob is chosen on VALIDATION AUC, the
chosen model is refit on train+val and evaluated ONCE on the test split. On Dataset 1, k-NN and the SVMs are fitted
on a stratified 20,000-row sample of the training data (kernel SVMs scale roughly quadratically with n).

Outputs: results/<dataset>_coursemodels_sweep.csv, results/<dataset>_coursemodels_test.csv
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(__file__))
from common import SEED, leakage_audit, load_approval, load_default, make_preprocessor, split_60_20_20
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- model factory + sweeps
# Each entry: (family, knob name, list of knob values, function value -> estimator).
# The knob is the model's main capacity / regularisation control.
def sweeps(n_train):
    k_max = min(201, n_train // 2)
    return [
        ("DecisionTree", "max_depth", list(range(1, 21)),
         lambda d: DecisionTreeClassifier(max_depth=d, random_state=SEED)),
        ("LogReg-L2", "C", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
         lambda c: LogisticRegression(C=c, penalty="l2", solver="liblinear", max_iter=3000, random_state=SEED)),
        ("LogReg-L1", "C", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
         lambda c: LogisticRegression(C=c, penalty="l1", solver="liblinear", max_iter=3000, random_state=SEED)),
        ("LDA", "shrinkage", ["none", "auto"],
         lambda s: LinearDiscriminantAnalysis(solver="lsqr", shrinkage=None if s == "none" else "auto")),
        ("NaiveBayes", "var_smoothing", [1e-9, 1e-6, 1e-3, 1e-1],
         lambda v: GaussianNB(var_smoothing=v)),
        ("kNN", "k", [k for k in (1, 3, 5, 11, 21, 51, 101, 201) if k <= k_max],
         lambda k: KNeighborsClassifier(n_neighbors=k, weights="uniform", n_jobs=2)),
        ("LinearSVM", "C", [0.001, 0.01, 0.1, 1.0, 10.0],
         lambda c: LinearSVC(C=c, dual=False, max_iter=5000, random_state=SEED)),
        ("RBF-SVM", "C|gamma", [(c, g) for c in (0.1, 1.0, 10.0) for g in ("scale", 0.01)],
         lambda cg: SVC(C=cg[0], gamma=cg[1], kernel="rbf", random_state=SEED)),
    ]


SUBSAMPLED = {"kNN", "LinearSVM", "RBF-SVM"}     # fitted on a 20k-row training sample on Dataset 1


def scores(pipe, X):
    """Continuous score for AUC: probability when available, else the SVM decision value."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)


def evaluate(pipe, X, y):
    """Threshold metrics at the model's own decision rule (p >= 0.5, or SVM margin >= 0) + AUC."""
    s, yhat = scores(pipe, X), pipe.predict(X)
    return dict(accuracy=accuracy_score(y, yhat), precision=precision_score(y, yhat, zero_division=0),
                recall=recall_score(y, yhat, zero_division=0), f1=f1_score(y, yhat, zero_division=0),
                auc=roc_auc_score(y, s), mcc=matthews_corrcoef(y, yhat))


def run(ds, X, y, feature_set, sub_n):
    log(f"=== {ds} [{feature_set}]")
    Xtr, Xva, Xte, ytr, yva, yte = split_60_20_20(X, y)
    _, flagged = leakage_audit(Xtr, ytr)                       # audit on TRAINING rows only
    cols = [c for c in X.columns if not (feature_set == "F_clean" and c in flagged)]
    Xtr, Xva, Xte = Xtr[cols], Xva[cols], Xte[cols]
    Xdev, ydev = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])

    # Training sample for the models that do not scale to ~90k rows (Dataset 1 only)
    if sub_n and len(Xtr) > sub_n:
        Xs, _, ys, _ = train_test_split(Xtr, ytr, train_size=sub_n, stratify=ytr, random_state=SEED)
        Xds, _, yds, _ = train_test_split(Xdev, ydev, train_size=sub_n, stratify=ydev, random_state=SEED)
    else:
        Xs, ys, Xds, yds = Xtr, ytr, Xdev, ydev

    sweep_rows, test_rows = [], []
    for fam, knob, values, make in sweeps(len(Xtr)):
        big = fam in SUBSAMPLED
        Xa, ya = (Xs, ys) if big else (Xtr, ytr)
        best, best_auc = None, -1.0
        for v in values:
            pipe = Pipeline([("pre", make_preprocessor(Xa)), ("clf", make(v))]).fit(Xa, ya)
            tr_auc = roc_auc_score(ya, scores(pipe, Xa))
            va_auc = roc_auc_score(yva, scores(pipe, Xva))
            sweep_rows.append(dict(dataset=ds, model=fam, knob=knob, value=str(v),
                                   n_train=len(Xa), train_auc=tr_auc, val_auc=va_auc))
            if va_auc > best_auc:
                best, best_auc = v, va_auc
        # Refit the chosen setting on train+val, evaluate ONCE on the untouched test split
        Xb, yb = (Xds, yds) if big else (Xdev, ydev)
        final = Pipeline([("pre", make_preprocessor(Xb)), ("clf", make(best))]).fit(Xb, yb)
        m = evaluate(final, Xte, yte)
        test_rows.append(dict(dataset=ds, feature_set=feature_set, model=fam, chosen=f"{knob}={best}",
                              n_fit=len(Xb), val_auc=best_auc, **m))
        log(f"  {fam:<12} best {knob}={best}  val AUC {best_auc:.4f}  test AUC {m['auc']:.4f}  MCC {m['mcc']:.4f}")

    pd.DataFrame(sweep_rows).to_csv(f"{RES}/{ds}_coursemodels_sweep.csv", index=False)
    pd.DataFrame(test_rows).to_csv(f"{RES}/{ds}_coursemodels_test.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    X, y = load_approval(os.path.join(a.data, "loan_data_set.csv"))
    run("approval", X, y, "F_full", sub_n=None)

    X, y = load_default(os.path.join(a.data, "Loan_Default.csv"))
    if a.quick:
        X, _, y, _ = train_test_split(X, y, train_size=0.15, stratify=y, random_state=SEED)
    run("default", X, y, "F_clean", sub_n=5000 if a.quick else 20000)
    log("ALL DONE")


if __name__ == "__main__":
    main()
