"""Phase 2 extensions: feature engineering, stronger models (Optuna-tuned LightGBM / XGBoost / CatBoost and a
stacking ensemble), probability calibration, paired significance tests, and cost-optimal decisions under a
fairness constraint (group-specific thresholds).

    python src/run_experiments.py      # first: baseline experiments (writes results/*.csv)
    python src/run_extensions.py       # then: this file (reads the baseline tuning CSVs, writes ext_*.csv)
    python src/run_extensions.py --quick

Same protocol as run_experiments.py: stratified 60/20/20 split before preprocessing; everything is fitted on
training data; the test split is scored once per final model.
"""
import argparse
import ast
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import StackingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(__file__))
from common import (SEED, THRESHOLDS, best_threshold, decision_cost, leakage_audit, load_approval, load_default,
                    make_estimator, make_preprocessor, metrics, split_60_20_20)

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
T0 = time.time()
RATIOS = [5, 10]


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ============================================================================ feature engineering
def add_features(X, ds):
    """Row-wise ratios only (no fitted statistics), so they cannot leak across splits."""
    X = X.copy()
    if ds == "approval":
        inc = X["ApplicantIncome"] + X["CoapplicantIncome"]
        emi = X["LoanAmount"] * 1000 / X["Loan_Amount_Term"]
        X["total_income"] = inc
        X["loan_to_income"] = X["LoanAmount"] * 1000 / inc
        X["emi"] = emi
        X["emi_to_income"] = emi / inc
        X["log_total_income"] = np.log1p(inc)
        new = ["total_income", "loan_to_income", "emi", "emi_to_income", "log_total_income"]
    else:
        X["loan_to_income"] = X["loan_amount"] / X["income"]
        X["loan_per_term"] = X["loan_amount"] / X["term"]
        X["log_income"] = np.log1p(X["income"])
        new = ["loan_to_income", "loan_per_term", "log_income"]
    X[new] = X[new].replace([np.inf, -np.inf], np.nan)
    return X, new


# ============================================================================ models
def build(name, params=None, seed=SEED):
    p = dict(params or {})
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        d = dict(n_estimators=200, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                 reg_lambda=1.0, random_state=seed, n_jobs=2, verbose=-1)
        d.update(p)  # tuned values override the baseline defaults
        return LGBMClassifier(**d)
    if name in ("LogReg", "RandForest"):
        return make_estimator(name, p, None, seed)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, tree_method="hist", n_jobs=2, random_state=seed,
                             eval_metric="auc", verbosity=0, **p)
    if name == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=300, thread_count=2, random_seed=seed, verbose=0, allow_writing_files=False, **p)
    raise ValueError(name)


def pipe_for(X, name, params=None):
    return Pipeline([("pre", make_preprocessor(X)), ("clf", build(name, params))])


def space(name, tr):
    if name == "LightGBM":
        return dict(num_leaves=tr.suggest_int("num_leaves", 7, 63), min_child_samples=tr.suggest_int("min_child_samples", 10, 100),
                    learning_rate=tr.suggest_float("learning_rate", 0.02, 0.15, log=True),
                    reg_lambda=tr.suggest_float("reg_lambda", 0.01, 10, log=True))
    if name == "XGBoost":
        return dict(max_depth=tr.suggest_int("max_depth", 3, 8), min_child_weight=tr.suggest_int("min_child_weight", 1, 20),
                    learning_rate=tr.suggest_float("learning_rate", 0.03, 0.2, log=True),
                    subsample=tr.suggest_float("subsample", 0.6, 1.0), colsample_bytree=tr.suggest_float("colsample_bytree", 0.5, 1.0),
                    reg_lambda=tr.suggest_float("reg_lambda", 0.01, 10, log=True))
    if name == "CatBoost":
        return dict(depth=tr.suggest_int("depth", 4, 8), learning_rate=tr.suggest_float("learning_rate", 0.03, 0.2, log=True),
                    l2_leaf_reg=tr.suggest_float("l2_leaf_reg", 1, 10))
    raise ValueError(name)


def optuna_tune(name, Xtr, ytr, Xva, yva, n_trials, ds):
    """Bayesian (TPE) search on validation AUC. Returns best params and the trial log."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rows = []

    def obj(tr):
        params = space(name, tr)
        pipe = pipe_for(Xtr, name, params).fit(Xtr, ytr)
        v = roc_auc_score(yva, pipe.predict_proba(Xva)[:, 1])
        rows.append(dict(dataset=ds, model=name, trial=tr.number, val_auc=v, params=str(params)))
        return v
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=n_trials)
    return st.best_params, pd.DataFrame(rows)


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if (idx == b).any())


def prob_metrics(y, p):
    return dict(auc=roc_auc_score(y, p), brier=brier_score_loss(y, p), logloss=log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
                ece=ece(y, p))


def oof_predict(X, y, name, params, seed=SEED):
    o = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        o[te] = pipe_for(X, name, params).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
    return o


# ============================================================================ significance
def corrected_t(diffs, n_train, n_test):
    """Nadeau-Bengio corrected resampled t-test on per-fold score differences."""
    d = np.asarray(diffs)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), k - 1))


def significance(Xdev, ydev, cfgs, n_rep, ds):
    folds = []
    for r in range(n_rep):
        for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(Xdev, ydev)):
            row = dict(rep=r, fold=f)
            for m, params in cfgs.items():
                p = pipe_for(Xdev, m, params).fit(Xdev.iloc[tr], ydev.iloc[tr]).predict_proba(Xdev.iloc[te])[:, 1]
                row[m] = roc_auc_score(ydev.iloc[te], p)
            folds.append(row)
        log(f"  {ds}: significance repeat {r + 1}/{n_rep}")
    F = pd.DataFrame(folds)
    F.insert(0, "dataset", ds)
    ntr, nte = int(len(ydev) * 0.8), int(len(ydev) * 0.2)
    ref = max(cfgs, key=lambda m: F[m].mean())
    out = []
    for m in cfgs:
        if m == ref:
            continue
        t, p = corrected_t(F[ref] - F[m], ntr, nte)
        out.append(dict(dataset=ds, reference=ref, other=m, mean_auc_ref=F[ref].mean(), mean_auc_other=F[m].mean(),
                        mean_diff=(F[ref] - F[m]).mean(), t=t, p_value=p, n_folds=len(F)))
    S = pd.DataFrame(out)
    order = S.p_value.argsort().values
    holm = np.empty(len(S))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, S.p_value.values[i] * (len(S) - rank)))
        holm[i] = run
    S["p_holm"] = holm
    return F, S


# ============================================================================ fairness-constrained decisions
def group_thresholds_dp(s_dev, y_dev, g_dev, ratio, t_global, min_n=30):
    """Equal approval rate q in every group: t_g = q-quantile of the group's scores. q is chosen to minimise dev cost."""
    groups = [lv for lv in pd.unique(g_dev.dropna()) if (g_dev == lv).sum() >= min_n]
    best = None
    for q in np.arange(0.02, 0.99, 0.01):
        th = {lv: np.quantile(s_dev[(g_dev == lv).values], q) for lv in groups}
        c = _cost_with(s_dev, y_dev, g_dev, th, t_global, ratio)
        if best is None or c < best[0]:
            best = (c, q, th)
    return best[2], best[1]


def group_thresholds_eo(s_dev, y_dev, g_dev, ratio, t_global, min_n=30):
    """Equal recall of adverse outcomes rho in every group: t_g = (1-rho)-quantile of that group's adverse scores."""
    groups = [lv for lv in pd.unique(g_dev.dropna())
              if (g_dev == lv).sum() >= min_n and ((g_dev == lv) & (y_dev == 1)).sum() >= 10]
    best = None
    for rho in np.arange(0.05, 0.99, 0.01):
        th = {lv: np.quantile(s_dev[((g_dev == lv) & (y_dev == 1)).values], 1 - rho) for lv in groups}
        c = _cost_with(s_dev, y_dev, g_dev, th, t_global, ratio)
        if best is None or c < best[0]:
            best = (c, rho, th)
    return best[2], best[1]


def _thr_vector(g, th, t_global):
    return pd.Series(g).map(th).astype(float).fillna(t_global).values


def _cost_with(s, y, g, th, t_global, ratio):
    yhat = s >= _thr_vector(g, th, t_global)
    fn = ((~yhat) & (y == 1)).sum()
    fp = (yhat & (y == 0)).sum()
    return (ratio * fn + fp) / len(y)


def policy_row(ds, attr, ratio, policy, s_te, y_te, g_te, th, t_global, extra=None):
    from fairlearn.metrics import equalized_odds_difference
    thr = _thr_vector(g_te, th, t_global) if th is not None else np.full(len(s_te), t_global)
    yhat = (s_te >= thr).astype(int)
    fn = ((yhat == 0) & (y_te == 1)).sum()
    fp = ((yhat == 1) & (y_te == 0)).sum()
    m = g_te.notna().values
    gg, yy, hh = g_te[m].astype(str).values, np.asarray(y_te)[m], yhat[m]
    ar = {lv: 1 - hh[gg == lv].mean() for lv in set(gg) if (gg == lv).sum() >= 10}
    rng = np.random.default_rng(SEED)

    def gap(ix):
        a = [1 - hh[ix][gg[ix] == lv].mean() for lv in ar if (gg[ix] == lv).any()]
        return max(a) - min(a)
    boots = [gap(rng.integers(0, len(gg), len(gg))) for _ in range(300)]
    keep = np.isin(gg, list(ar))
    tpr = {lv: hh[(gg == lv) & (yy == 1)].mean() for lv in ar if ((gg == lv) & (yy == 1)).any()}
    return dict(dataset=ds, attribute=attr, ratio=ratio, policy=policy, cost=(ratio * fn + fp) / len(y_te),
                approval_rate=1 - yhat.mean(), recall=hh[yy == 1].mean() if (yy == 1).any() else np.nan,
                precision=(yy[hh == 1].mean() if (hh == 1).any() else np.nan),
                dp_gap=max(ar.values()) - min(ar.values()), dp_ci_lo=np.percentile(boots, 2.5), dp_ci_hi=np.percentile(boots, 97.5),
                tpr_gap=max(tpr.values()) - min(tpr.values()),
                eo_gap=equalized_odds_difference(yy[keep], hh[keep], sensitive_features=gg[keep]),
                smallest_group_n=min((gg == lv).sum() for lv in ar), **(extra or {}))


# ============================================================================ per dataset
def run(ds, X, y, base_sets, n_rep, trials, headline):
    Xtr, Xva, Xte, ytr, yva, yte = split_60_20_20(X, y)
    Xdev, ydev = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])
    _, flagged = leakage_audit(Xtr, ytr)
    base_cols = base_sets(X.columns.tolist(), flagged)
    XF, new = add_features(X, ds)
    tun = pd.read_csv(f"{RES}/{ds}_tuning.csv")
    tun = tun[tun.feature_set == headline]

    def best_params(m):
        d = tun[tun.model == m]
        return ast.literal_eval(d.loc[d.val_auc.idxmax(), "params"])
    lg_base, lr_base = best_params("LightGBM"), best_params("LogReg")

    # ---------------- 1. feature engineering ablation
    rows = []
    for fs_name, cols in (("base", base_cols), ("base+engineered", base_cols + new)):
        for m, params in (("LogReg", lr_base), ("LightGBM", lg_base)):
            cvs = []
            for r in range(n_rep):
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(Xdev, ydev):
                    Xd = XF.loc[Xdev.index, cols]
                    p = pipe_for(Xd, m, params).fit(Xd.iloc[tr], ydev.iloc[tr]).predict_proba(Xd.iloc[te])[:, 1]
                    cvs.append(roc_auc_score(ydev.iloc[te], p))
            pipe = pipe_for(XF[cols], m, params).fit(XF.loc[Xdev.index, cols], ydev)
            pt = pipe.predict_proba(XF.loc[Xte.index, cols])[:, 1]
            rows.append(dict(dataset=ds, feature_set=fs_name, n_features=len(cols), model=m, cv_auc_mean=np.mean(cvs),
                             cv_auc_std=np.std(cvs, ddof=1), **{f"test_{k}": v for k, v in metrics(yte, pt).items()}))
            log(f"{ds} FE {fs_name:16s} {m:9s} CV AUC {np.mean(cvs):.4f}  test AUC {rows[-1]['test_auc']:.4f}")
    fe = pd.DataFrame(rows)
    fe.to_csv(f"{RES}/ext_{ds}_feature_engineering.csv", index=False)
    gain = (fe[(fe.feature_set == "base+engineered") & (fe.model == "LightGBM")].cv_auc_mean.values[0]
            - fe[(fe.feature_set == "base") & (fe.model == "LightGBM")].cv_auc_mean.values[0])
    use_fe = gain > 0  # rule fixed in advance: keep the engineered features only if CV AUC (LightGBM) improves
    cols = base_cols + new if use_fe else base_cols
    log(f"{ds}: engineered features {'KEPT' if use_fe else 'DROPPED'} (CV AUC gain {gain:+.4f}); {len(cols)} features")
    Xtr_, Xva_, Xte_, Xdev_ = XF.loc[Xtr.index, cols], XF.loc[Xva.index, cols], XF.loc[Xte.index, cols], XF.loc[Xdev.index, cols]

    # ---------------- 2. stronger models (Optuna) + stacking
    tuned, trials, res = {"LogReg": lr_base}, [], []
    for m in ("LightGBM", "XGBoost", "CatBoost"):
        bp, tl = optuna_tune(m, Xtr_, ytr, Xva_, yva, TRIALS[m], ds)
        tuned[m] = bp
        trials.append(tl)
        log(f"{ds}: Optuna {m}: best val AUC {tl.val_auc.max():.4f} after {len(tl)} trials")
    pd.concat(trials).to_csv(f"{RES}/ext_{ds}_optuna_trials.csv", index=False)
    fits = {}
    for m in ("LightGBM", "XGBoost", "CatBoost"):
        va = pipe_for(Xtr_, m, tuned[m]).fit(Xtr_, ytr)
        fits[m] = dict(val_auc=roc_auc_score(yva, va.predict_proba(Xva_)[:, 1]))
    stack_est = [("lr", build("LogReg", lr_base)), ("lgbm", build("LightGBM", tuned["LightGBM"])),
                 ("xgb", build("XGBoost", tuned["XGBoost"])), ("cat", build("CatBoost", tuned["CatBoost"]))]
    mk_stack = lambda X_: Pipeline([("pre", make_preprocessor(X_)),
                                    ("clf", StackingClassifier(stack_est, final_estimator=LogisticRegression(max_iter=2000),
                                                               cv=3, stack_method="predict_proba", n_jobs=1))])
    va = mk_stack(Xtr_).fit(Xtr_, ytr)
    fits["Stack"] = dict(val_auc=roc_auc_score(yva, va.predict_proba(Xva_)[:, 1]))
    test_p = {}
    for m in ("LightGBM", "XGBoost", "CatBoost", "Stack"):
        pipe = (mk_stack(Xdev_) if m == "Stack" else pipe_for(Xdev_, m, tuned[m])).fit(Xdev_, ydev)
        test_p[m] = pipe.predict_proba(Xte_)[:, 1]
        res.append(dict(dataset=ds, model=m + ("(Optuna)" if m != "Stack" else ""), features=len(cols), val_auc=fits[m]["val_auc"],
                        **prob_metrics(yte, test_p[m]), mcc=metrics(yte, test_p[m])["mcc"], params=str(tuned.get(m, {}))))
        log(f"{ds}: {m:9s} val AUC {fits[m]['val_auc']:.4f} test AUC {res[-1]['auc']:.4f} Brier {res[-1]['brier']:.4f}")
    pd.DataFrame(res).to_csv(f"{RES}/ext_{ds}_models.csv", index=False)

    # ---------------- 3. calibration of the best single model (calibrator fitted on out-of-fold predictions)
    bm = max(("LightGBM", "XGBoost", "CatBoost"), key=lambda m: fits[m]["val_auc"])
    oof = oof_predict(Xdev_, ydev, bm, tuned[bm])
    if len(ydev) > 1000:
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof, ydev)
        kind, f = "isotonic", cal.predict
    else:
        sig = LogisticRegression(C=1e6, max_iter=1000).fit(np.log(np.clip(oof, 1e-6, 1 - 1e-6) / (1 - np.clip(oof, 1e-6, 1 - 1e-6))).reshape(-1, 1), ydev)
        kind = "sigmoid"
        f = lambda p: sig.predict_proba(np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1))[:, 1]
    p_raw, p_cal, oof_cal = test_p[bm], f(test_p[bm]), f(oof)
    cm = pd.DataFrame([dict(dataset=ds, model=bm, version="raw", **prob_metrics(yte, p_raw)),
                       dict(dataset=ds, model=bm, version=f"calibrated ({kind})", **prob_metrics(yte, p_cal))])
    cm.to_csv(f"{RES}/ext_{ds}_calibration.csv", index=False)
    bins = np.linspace(0, 1, 11)
    cur = []
    for ver, pp in (("raw", p_raw), ("calibrated", p_cal)):
        ix = np.minimum((pp * 10).astype(int), 9)
        for b in range(10):
            if (ix == b).sum():
                cur.append(dict(dataset=ds, version=ver, bin=b, mean_pred=pp[ix == b].mean(), frac_pos=yte.values[ix == b].mean(), n=int((ix == b).sum())))
    pd.DataFrame(cur).to_csv(f"{RES}/ext_{ds}_reliability.csv", index=False)
    th_rows = []
    for r in (2, 5, 10, 20):
        th_rows.append(dict(dataset=ds, ratio=r, theory=1 / (r + 1), t_raw=best_threshold(ydev, oof, r), t_calibrated=best_threshold(ydev, oof_cal, r)))
    pd.DataFrame(th_rows).to_csv(f"{RES}/ext_{ds}_threshold_theory.csv", index=False)
    log(f"{ds}: calibration ({kind}) Brier {cm.brier[0]:.4f} -> {cm.brier[1]:.4f}; ECE {cm.ece[0]:.4f} -> {cm.ece[1]:.4f}")

    # ---------------- 4. paired significance of model differences (identical folds)
    cfgs = {"LogReg": lr_base, "RandForest": best_params("RandForest"), "LightGBM": tuned["LightGBM"],
            "XGBoost": tuned["XGBoost"], "CatBoost": tuned["CatBoost"]}
    F, S = significance(Xdev_, ydev, cfgs, n_rep, ds)
    F.to_csv(f"{RES}/ext_{ds}_cv_folds.csv", index=False)
    S.to_csv(f"{RES}/ext_{ds}_significance.csv", index=False)
    log(f"{ds}: significance done; reference {S.reference.iloc[0]}")

    # ---------------- 5. cost-optimal decisions under a fairness constraint (calibrated scores of the best model)
    if ds == "approval":
        attrs = {"Gender": X["Gender"], "Married": X["Married"], "Education": X["Education"]}
    else:
        age = X["age"].map({"<25": "<=34", "25-34": "<=34", "35-44": "35-54", "45-54": "35-54",
                            "55-64": "55+", "65-74": "55+", ">74": "55+"})
        attrs = {"Gender": X["Gender"], "AgeGroup": age}
    pol, thr_rows = [], []
    for attr, g in attrs.items():
        g_dev, g_te = g.loc[Xdev.index], g.loc[Xte.index].reset_index(drop=True)
        for r in RATIOS:
            t_star = best_threshold(ydev, oof_cal, r)
            pol.append(policy_row(ds, attr, r, "P0 standard t=0.50", p_cal, yte.values, g_te, None, 0.5))
            pol.append(policy_row(ds, attr, r, "P1 cost-optimal, one threshold", p_cal, yte.values, g_te, None, t_star, dict(t_global=t_star)))
            th_dp, q = group_thresholds_dp(oof_cal, ydev.values, g_dev.reset_index(drop=True), r, t_star)
            pol.append(policy_row(ds, attr, r, "P2 cost-optimal + equal approval rate", p_cal, yte.values, g_te, th_dp, t_star, dict(target=q)))
            th_eo, rho = group_thresholds_eo(oof_cal, ydev.values, g_dev.reset_index(drop=True), r, t_star)
            pol.append(policy_row(ds, attr, r, "P3 cost-optimal + equal recall", p_cal, yte.values, g_te, th_eo, t_star, dict(target=rho)))
            for pname, th in (("P2", th_dp), ("P3", th_eo)):
                for lv, t in th.items():
                    thr_rows.append(dict(dataset=ds, attribute=attr, ratio=r, policy=pname, group=lv, threshold=t))
        log(f"{ds}: policies done for {attr}")
    pd.DataFrame(pol).to_csv(f"{RES}/ext_{ds}_policy_comparison.csv", index=False)
    pd.DataFrame(thr_rows).to_csv(f"{RES}/ext_{ds}_group_thresholds.csv", index=False)


TRIALS = {}


def main():
    global TRIALS
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    a = ap.parse_args()
    X, y = load_approval(os.path.join(a.data, "loan_data_set.csv"))
    TRIALS = dict(LightGBM=4, XGBoost=4, CatBoost=3) if a.quick else dict(LightGBM=30, XGBoost=30, CatBoost=20)
    run("approval", X, y, lambda cols, fl: cols, n_rep=2 if a.quick else 5, trials=None, headline="F_full")
    X, y = load_default(os.path.join(a.data, "Loan_Default.csv"))
    if a.quick:
        X, _, y, _ = train_test_split(X, y, train_size=0.15, stratify=y, random_state=SEED)
    TRIALS = dict(LightGBM=3, XGBoost=3, CatBoost=2) if a.quick else dict(LightGBM=20, XGBoost=20, CatBoost=8)
    run("default", X, y, lambda cols, fl: [c for c in cols if c not in fl], n_rep=1 if a.quick else 2, trials=None, headline="F_clean")
    log("ALL DONE")


if __name__ == "__main__":
    main()
