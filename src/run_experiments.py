"""Runs every experiment promised in Phase 1 on both datasets and writes CSVs to results/.

    python src/run_experiments.py            # full run
    python src/run_experiments.py --quick    # fast smoke test on a 15% sample of Dataset 1

Protocol (identical for both datasets)
  1. Stratified 60/20/20 train/validation/test split BEFORE any preprocessing.
  2. Leakage audit on the training split only -> feature sets (F_full, F_clean / F_noCH).
  3. Hyper-parameters chosen on the validation split (validation AUC). Train AUC is logged too, so
     the bias-variance / regularisation trade-off can be shown.
  4. Stability: stratified 5-fold CV on train+val (5 repeats for the small dataset, 1 for the large one).
  5. Final model = best config refit on train+val ("dev"); evaluated ONCE on the untouched test split.
  6. Cost-sensitive threshold chosen on out-of-fold dev predictions, then applied to the test split.
  7. Fairness, TreeSHAP and Fairlearn mitigation on the test split for the headline feature set.
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from common import (GRIDS, SEED, THRESHOLDS, best_threshold, cost_curve, decision_cost, leakage_audit,
                    load_approval, load_default, make_estimator, make_pipeline, make_preprocessor, metrics,
                    split_60_20_20)

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
MODELS = ["LogReg", "RandForest", "LightGBM", "XGBoost", "CatBoost"]
RATIOS = [2, 5, 10, 20]         # 10:1 is the Phase 1 choice; the others are the sensitivity sweep
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ============================================================================ model selection
def tune(Xtr, ytr, Xva, yva, name, tag):
    """Grid search on the validation split. Returns best params and the full log."""
    rows = []
    for params in GRIDS[name]:
        pipe = make_pipeline(Xtr, name, params).fit(Xtr, ytr)
        rows.append(dict(**tag, model=name, params=str(params),
                         train_auc=roc_auc_score(ytr, pipe.predict_proba(Xtr)[:, 1]),
                         val_auc=roc_auc_score(yva, pipe.predict_proba(Xva)[:, 1])))
    log_df = pd.DataFrame(rows)
    best = GRIDS[name][int(log_df.val_auc.values.argmax())]
    return best, log_df


def cv_eval(X, y, name, params, n_rep):
    """Repeated stratified 5-fold CV. Returns per-fold metrics and the first repeat's OOF probabilities."""
    rows, oof = [], None
    for r in range(n_rep):
        o = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(X, y):
            p = make_pipeline(X, name, params).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
            o[te] = p
            rows.append(metrics(y.iloc[te], p))
        if r == 0:
            oof = o
    return pd.DataFrame(rows), oof


# ============================================================================ fairness helpers
def group_report(g, y, p, t):
    """Per-group approval rates / error rates plus gap summaries with a bootstrap CI on the DP gap."""
    from fairlearn.metrics import equalized_odds_difference
    m = g.notna().values
    g, y, p = g[m].astype(str).values, np.asarray(y)[m], np.asarray(p)[m]
    yhat = (p >= t).astype(int)
    rows = []
    for lv in sorted(set(g)):
        k = g == lv
        rows.append(dict(group=lv, n=int(k.sum()), approval_rate_pred=1 - yhat[k].mean(),
                         approval_rate_actual=1 - y[k].mean(),
                         tpr_adverse=yhat[k][y[k] == 1].mean() if (y[k] == 1).any() else np.nan,
                         fpr_adverse=yhat[k][y[k] == 0].mean() if (y[k] == 0).any() else np.nan))
    grp = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED)

    def gap(idx):
        ar = [1 - yhat[idx][g[idx] == lv].mean() for lv in set(g[idx])]
        return max(ar) - min(ar)
    boots = [gap(rng.integers(0, len(g), len(g))) for _ in range(500)]
    ar = grp.approval_rate_pred
    summ = dict(dp_gap=ar.max() - ar.min(), dp_gap_ci_lo=np.percentile(boots, 2.5),
                dp_gap_ci_hi=np.percentile(boots, 97.5), dp_ratio=ar.min() / ar.max(),
                eo_gap=equalized_odds_difference(y, yhat, sensitive_features=g), smallest_group_n=int(grp.n.min()))
    return grp, summ


def make_attrs(X, dataset):
    """Sensitive attributes used for the audit -> (Series aligned with X, model feature it relates to)."""
    if dataset == "approval":
        return {"Gender": (X["Gender"], "Gender"), "Married": (X["Married"], "Married"),
                "Education": (X["Education"], "Education")}
    age = X["age"].map({"<25": "<=34", "25-34": "<=34", "35-44": "35-54", "45-54": "35-54",
                        "55-64": "55+", "65-74": "55+", ">74": "55+"})
    return {"Gender": (X["Gender"], "Gender"), "AgeGroup": (age, "age")}


def shap_disparity(Xdev, ydev, Xte, params, attrs):
    """TreeSHAP on a LightGBM model; one-hot columns are summed back to original features.
    Disparity_j = max over groups of mean(phi_j) minus min over groups of mean(phi_j)."""
    import shap
    pre = make_preprocessor(Xdev).fit(Xdev)
    clf = make_estimator("LightGBM", params).fit(pre.transform(Xdev), ydev)
    idx = np.random.default_rng(SEED).choice(len(Xte), size=min(6000, len(Xte)), replace=False)
    sv = shap.TreeExplainer(clf).shap_values(pre.transform(Xte.iloc[idx]))
    sv = sv[1] if isinstance(sv, list) else sv
    cols = sorted(Xdev.columns, key=len, reverse=True)
    owner = []
    for n in pre.get_feature_names_out():
        base = n.split("__", 1)[1].replace("missingindicator_", "")
        owner.append(next(c for c in cols if base == c or base.startswith(c + "_")))
    phi = pd.DataFrame(sv).T.groupby(np.array(owner)).sum().T            # (rows x original features)
    out = []
    for a, (g, feat) in attrs.items():
        g = g.iloc[idx].reset_index(drop=True)
        ok = g.notna()
        means = phi[ok].groupby(g[ok].astype(str)).mean()
        for f in phi.columns:
            out.append(dict(attribute=a, feature=f, disparity=means[f].max() - means[f].min(),
                            high_group=means[f].idxmax(), low_group=means[f].idxmin(),
                            is_sensitive_feature=(f == feat), mean_abs_phi=phi[f].abs().mean()))
    return pd.DataFrame(out)


def mitigate(Xdev, ydev, Xte, yte, C, g_dev, g_te, t_star):
    """Baseline LR (t=0.5 and cost-tuned t) vs Fairlearn ExponentiatedGradient(DemographicParity)."""
    from fairlearn.reductions import DemographicParity, ExponentiatedGradient
    kd, kt = g_dev.notna().values, g_te.notna().values
    Xd, yd, gd = Xdev[kd], ydev[kd], g_dev[kd].astype(str)
    Xt, yt, gt = Xte[kt], yte[kt], g_te[kt].astype(str)
    pre = make_preprocessor(Xd).fit(Xd)
    A, B = pre.transform(Xd), pre.transform(Xt)
    lr = LogisticRegression(C=C, max_iter=3000, random_state=SEED).fit(A, yd)
    p_lr = lr.predict_proba(B)[:, 1]
    eg = ExponentiatedGradient(LogisticRegression(C=C, max_iter=3000, random_state=SEED), DemographicParity(),
                               eps=0.005, max_iter=50)
    eg.fit(A, yd, sensitive_features=gd)
    p_eg = eg._pmf_predict(B)[:, 1]
    yhat_eg = eg.predict(B, random_state=SEED)
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
    rows = []
    # (label, probabilities, hard decisions)
    variants = [("LogReg @0.50", p_lr, (p_lr >= 0.5).astype(int)),
                (f"LogReg @cost-optimal t={t_star:.2f}", p_lr, (p_lr >= t_star).astype(int)),
                ("LogReg + ExponentiatedGradient (DP)", p_eg, yhat_eg.astype(int))]
    for label, p, yh in variants:
        fn = int(((yh == 0) & (yt.values == 1)).sum())
        fp = int(((yh == 1) & (yt.values == 0)).sum())
        _, sm = group_report(gt, yt, yh.astype(float), 0.5)
        rows.append(dict(model=label, accuracy=accuracy_score(yt, yh), precision=precision_score(yt, yh, zero_division=0),
                         recall=recall_score(yt, yh), f1=f1_score(yt, yh),
                         # EG returns hard 0/1 decisions, so an AUC for it would not be comparable
                         auc=np.nan if "Exponentiated" in label else roc_auc_score(yt, p),
                         mcc=matthews_corrcoef(yt, yh), approval_rate=1 - yh.mean(), dp_gap=sm["dp_gap"],
                         eo_gap=sm["eo_gap"], cost_per_applicant_10to1=(10 * fn + fp) / len(yt)))
    return pd.DataFrame(rows)


# ============================================================================ one dataset
def run_dataset(name, X, y, sets_fn, headline, n_rep):
    tag0 = dict(dataset=name)
    Xtr, Xva, Xte, ytr, yva, yte = split_60_20_20(X, y)
    Xdev, ydev = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])
    log(f"{name}: train={len(Xtr)} val={len(Xva)} test={len(Xte)}; adverse rate {y.mean():.3f}")

    audit, flagged = leakage_audit(Xtr, ytr)
    audit.insert(0, "dataset", name)
    audit.to_csv(f"{RES}/{name}_leakage_audit.csv", index=False)
    log(f"{name}: audit flagged {flagged}")
    sets = sets_fn(X.columns.tolist(), flagged)

    tuning, cvs, tests, costs, curves = [], [], [], [], []
    store = {}
    for sname, cols in sets.items():
        tag = dict(dataset=name, feature_set=sname)
        log(f"{name}/{sname}: {len(cols)} features")
        best = {}
        for m in MODELS:
            best[m], tl = tune(Xtr[cols], ytr, Xva[cols], yva, m, tag)
            tuning.append(tl)
            cvdf, oof = cv_eval(Xdev[cols], ydev, m, best[m], n_rep)
            cvs.append(cvdf.agg(["mean", "std"]).T.assign(dataset=name, feature_set=sname, model=m)
                       .reset_index().rename(columns={"index": "metric"}))
            pipe = make_pipeline(Xdev[cols], m, best[m]).fit(Xdev[cols], ydev)
            p_te = pipe.predict_proba(Xte[cols])[:, 1]
            val_auc = tl.val_auc.max()
            tests.append(dict(**tag, model=m, params=str(best[m]), val_auc=val_auc, **metrics(yte, p_te, 0.5)))
            store[(sname, m)] = dict(oof=oof, p_te=p_te, params=best[m], val_auc=val_auc)
            log(f"  {m:10s} val AUC {val_auc:.4f}  test AUC {tests[-1]['auc']:.4f}  MCC {tests[-1]['mcc']:.4f}")

        # cost-sensitive threshold for the best model (by validation AUC) of this feature set
        bm = max(MODELS, key=lambda m: store[(sname, m)]["val_auc"])
        oof, p_te = store[(sname, bm)]["oof"], store[(sname, bm)]["p_te"]
        for r in RATIOS:
            t_star = best_threshold(ydev, oof, r)
            c50, fn50, fp50 = decision_cost(yte.values, p_te, 0.5, r)
            cs, fns, fps = decision_cost(yte.values, p_te, t_star, r)
            mt = metrics(yte, p_te, t_star)
            costs.append(dict(**tag, model=bm, ratio=r, t_star=t_star, cost_at_0_50=c50 / len(yte),
                              cost_at_t_star=cs / len(yte), reduction_pct=100 * (c50 - cs) / c50 if c50 else np.nan,
                              fn_at_0_50=fn50, fp_at_0_50=fp50, fn_at_t_star=fns, fp_at_t_star=fps,
                              recall_at_t_star=mt["recall"], precision_at_t_star=mt["precision"],
                              approval_rate_at_t_star=1 - (p_te >= t_star).mean(),
                              approval_rate_at_0_50=1 - (p_te >= 0.5).mean(),
                              cost_approve_all=r * yte.mean(), cost_reject_all=1 - yte.mean()))
            curves.append(pd.DataFrame(dict(**tag, ratio=r, threshold=THRESHOLDS,
                                            dev_oof_cost=cost_curve(ydev, oof, r), test_cost=cost_curve(yte, p_te, r))))
        store[(sname, "best")] = bm
        log(f"  best model {bm}; 10:1 threshold {costs[-2]['t_star']:.2f}, "
            f"test cost reduction {costs[-2]['reduction_pct']:.1f}%, approval rate {costs[-2]['approval_rate_at_t_star']:.1%}")

    pd.concat(tuning).to_csv(f"{RES}/{name}_tuning.csv", index=False)
    pd.concat(cvs).to_csv(f"{RES}/{name}_cv.csv", index=False)
    pd.DataFrame(tests).to_csv(f"{RES}/{name}_test_metrics.csv", index=False)
    pd.DataFrame(costs).to_csv(f"{RES}/{name}_cost_threshold.csv", index=False)
    pd.concat(curves).to_csv(f"{RES}/{name}_cost_curves.csv", index=False)

    # ---------------- learning curves + balancing on the headline feature set
    cols = sets[headline]
    bm = store[(headline, "best")]
    lc = []
    for m in ("LogReg", "LightGBM"):
        for f in (0.05, 0.1, 0.2, 0.4, 0.7, 1.0):
            Xs, ys = (Xtr[cols], ytr) if f == 1.0 else train_test_split(Xtr[cols], ytr, train_size=f, stratify=ytr, random_state=SEED)[0::2]
            pipe = make_pipeline(Xs, m, store[(headline, m)]["params"]).fit(Xs, ys)
            lc.append(dict(dataset=name, feature_set=headline, model=m, n_train=len(Xs),
                           train_auc=roc_auc_score(ys, pipe.predict_proba(Xs)[:, 1]),
                           val_auc=roc_auc_score(yva, pipe.predict_proba(Xva[cols])[:, 1])))
    pd.DataFrame(lc).to_csv(f"{RES}/{name}_learning_curve.csv", index=False)
    log(f"{name}: learning curves done")

    bal = []
    for m in ("LogReg", "LightGBM"):
        for strat in ("none", "class_weight", "SMOTE"):
            pipe = make_pipeline(Xdev[cols], m, store[(headline, m)]["params"],
                                 class_weight="balanced" if strat == "class_weight" else None,
                                 smote=(strat == "SMOTE")).fit(Xdev[cols], ydev)
            bal.append(dict(dataset=name, feature_set=headline, model=m, balancing=strat,
                            **metrics(yte, pipe.predict_proba(Xte[cols])[:, 1], 0.5)))
    pd.DataFrame(bal).to_csv(f"{RES}/{name}_balancing.csv", index=False)
    log(f"{name}: balancing done")

    # ---------------- fairness: group report, TreeSHAP disparity, mitigation
    t10 = best_threshold(ydev, store[(headline, bm)]["oof"], 10)
    p_te = store[(headline, bm)]["p_te"]
    attrs = make_attrs(X, name)
    gr_all, sm_all = [], []
    for a, (g, _) in attrs.items():
        for op, t in (("t=0.50", 0.5), ("cost-optimal 10:1", t10)):
            gr, sm = group_report(g.loc[Xte.index], yte, p_te, t)
            gr_all.append(gr.assign(dataset=name, attribute=a, model=bm, operating_point=op, threshold=t))
            sm_all.append(dict(dataset=name, attribute=a, model=bm, operating_point=op, threshold=t, **sm))
    pd.concat(gr_all).to_csv(f"{RES}/{name}_fairness_groups.csv", index=False)
    pd.DataFrame(sm_all).to_csv(f"{RES}/{name}_fairness_summary.csv", index=False)
    log(f"{name}: fairness report done")

    sd = shap_disparity(Xdev[cols], ydev, Xte[cols], store[(headline, "LightGBM")]["params"],
                        {a: (g.loc[Xte.index], f) for a, (g, f) in attrs.items()})
    sd.insert(0, "dataset", name)
    sd.to_csv(f"{RES}/{name}_shap_disparity.csv", index=False)
    log(f"{name}: TreeSHAP done")

    C = store[(headline, "LogReg")]["params"]["C"]
    mts = []
    for a, (g, _) in attrs.items():
        mt = mitigate(Xdev[cols], ydev, Xte[cols], yte, C, g.loc[Xdev.index], g.loc[Xte.index], t10)
        mts.append(mt.insert(0, "attribute", a) or mt)
    pd.concat(mts).assign(dataset=name).to_csv(f"{RES}/{name}_mitigation.csv", index=False)
    log(f"{name}: mitigation done")


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    # Dataset 2 (small): F_full vs F without the dominant feature Credit_History
    X, y = load_approval(os.path.join(a.data, "loan_data_set.csv"))
    run_dataset("approval", X, y,
                lambda cols, flagged: {"F_full": cols, "F_noCreditHistory": [c for c in cols if c != "Credit_History"]},
                headline="F_full", n_rep=5)

<<<<<<< HEAD
    # Dataset 1 (large): leakage-filtered F_clean only (columns flagged by the audit are removed;
    # F_full is intentionally excluded because it contains post-outcome leakage columns)
=======
    # Dataset 1 (large): F_full (includes the 9 leaking columns, evidence that the audit works)
    #                    vs F_clean (audit-flagged columns removed; headline feature set)
>>>>>>> 3f1ba31 (Initial commit)
    X, y = load_default(os.path.join(a.data, "Loan_Default.csv"))
    if a.quick:
        X, _, y, _ = train_test_split(X, y, train_size=0.15, stratify=y, random_state=SEED)
    run_dataset("default", X, y,
<<<<<<< HEAD
                lambda cols, flagged: {"F_clean": [c for c in cols if c not in flagged]},
=======
                lambda cols, flagged: {"F_full": cols, "F_clean": [c for c in cols if c not in flagged]},
>>>>>>> 3f1ba31 (Initial commit)
                headline="F_clean", n_rep=1)
    log("ALL DONE")


if __name__ == "__main__":
    main()
