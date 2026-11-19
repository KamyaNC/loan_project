"""One-command summary of the whole project: every key result, organised by section.

    python report.py                 # print the report and save it to REPORT.md
    python report.py --no-save       # print only

Reads the CSVs in results/ (nothing is retrained, so it runs in about a second).
Sections:
  1. Datasets            4. Bias-variance and regularisation   7. Fairness
  2. Leakage audit       5. Calibration                         8. Key findings
  3. Model comparison    6. Cost-sensitive decisions            9. Files to look at
"""
import argparse
import ast
import os

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
OUT = []                                           # every printed line, also written to REPORT.md


def rd(name):
    return pd.read_csv(os.path.join(RES, f"{name}.csv"))


def say(line=""):
    print(line)
    OUT.append(line)


def section(n, title):
    say()
    say(f"## {n}. {title}")
    say()


def table(df, cols, headers=None, fmt=None):
    """Print a Markdown table (readable in the terminal and rendered on GitHub / in VS Code preview)."""
    fmt = fmt or {}
    headers = headers or cols
    rows = [[fmt.get(c, lambda v: f"{v:.3f}" if isinstance(v, float) else str(v))(r[c]) for c in cols]
            for _, r in df.iterrows()]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    say("| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |")
    say("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        say("| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + " |")
    say()


pct = lambda v: f"{100 * v:.1f}%"
pp = lambda v: f"{100 * v:.1f} pp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()

    say("# Loan Approval and Default Prediction: Phase 2 results")
    say()
    say("BITS F464 Machine Learning, Concept Integration Assessment 2 · BITS Pilani, Dubai Campus")
    say("Kamya Chandebhamar · Anirudh Rakesh · Prerana CS")
    say()
    say("Convention: y = 1 is the adverse outcome (default / rejection), so a false negative is a FALSE APPROVAL.")

    # ------------------------------------------------------------------ 1
    section(1, "Datasets")
    ds = pd.DataFrame([
        dict(d="1: Loan Default", rows="148,670", label="defaulted (24.6%)", meaning="repayment outcome",
             split="89,202 / 29,734 / 29,734"),
        dict(d="2: Loan Approval", rows="614", label="rejected (31.3%)", meaning="past lender decision",
             split="368 / 123 / 123")])
    table(ds, ["d", "rows", "label", "meaning", "split"],
          ["Dataset", "Rows", "Label (y = 1)", "Label means", "Train / val / test"])

    # ------------------------------------------------------------------ 2
    section(2, "Leakage audit (training rows only)")
    au = rd("default_leakage_audit")
    fl = au[au.flagged].copy()
    say(f"Dataset 1: {len(fl)} of {len(au)} columns flagged and removed (F_clean). Dataset 2: none flagged.")
    say()
    table(fl, ["feature", "pct_missing", "reason"], ["Flagged column", "% missing", "Why"],
          {"pct_missing": lambda v: f"{v:.1f}"})
    cv1, cv2 = rd("default_cv"), rd("approval_cv")
    ab = []
    for m in ("LogReg", "LightGBM"):
        g = cv1[(cv1.metric == "auc") & (cv1.model == m)].set_index("feature_set")["mean"]
        ab.append(dict(ds="1", model=m, full=g["F_full"], clean=g["F_clean"]))
    for m in ("LogReg", "LightGBM"):
        g = cv2[(cv2.metric == "auc") & (cv2.model == m)].set_index("feature_set")["mean"]
        ab.append(dict(ds="2", model=m, full=g["F_full"], clean=g["F_noCreditHistory"]))
    say("Cross-validated AUC with and without the leaked (Dataset 1) / dominant (Dataset 2) features:")
    say()
    table(pd.DataFrame(ab), ["ds", "model", "full", "clean"],
          ["Dataset", "Model", "All features", "Without leaked / Credit_History"])

    # ------------------------------------------------------------------ 3
    section(3, "Model comparison on the untouched test split (threshold 0.50)")
    cm1 = rd("default_coursemodels_test")
    im = rd("improved_default_test_metrics")
    im = im[im.model.isin(["LightGBM", "XGBoost", "CatBoost", "Stacking"])].assign(
        model=lambda d: d.model.map(lambda m: m if m == "Stacking" else f"{m} (Optuna)"))
    d1 = pd.concat([cm1[cm1.model != "LogReg-L1"], im]).sort_values("auc")
    d1["model"] = d1.model.replace({"CatBoost (Optuna)": "CatBoost (Optuna)  <- final"})
    cols = ["model", "accuracy", "precision", "recall", "f1", "auc", "mcc"]
    hdr = ["Model", "Accuracy", "Precision", "Recall", "F1", "AUC", "MCC"]
    say("Dataset 1, 29,734 test loans (k-NN and SVMs fitted on a 20,000-row sample):")
    say()
    table(d1, cols, hdr)
    cm2 = rd("approval_coursemodels_test")
    t2 = rd("approval_test_metrics")
    t2 = t2[(t2.feature_set == "F_full") & t2.model.isin(["RandForest", "LightGBM"])]
    d2 = pd.concat([cm2[cm2.model != "LogReg-L1"], t2]).sort_values("auc")
    d2["model"] = d2.model.replace({"LogReg-L2": "LogReg-L2  <- final"})
    say("Dataset 2, 123 test applications:")
    say()
    table(d2, cols, hdr)
    sig = rd("ext_approval_significance")
    say(f"Dataset 2 significance (corrected t-test, Holm): smallest adjusted p = {sig.p_holm.min():.2f}, "
        "so no model is significantly better than another.")

    # ------------------------------------------------------------------ 4
    section(4, "Bias-variance and regularisation (train vs validation AUC)")
    for dsn, label in (("default", "Dataset 1"), ("approval", "Dataset 2")):
        sw = rd(f"{dsn}_coursemodels_sweep")
        rows = []
        for m, g in sw.groupby("model", sort=False):
            b = g.loc[g.val_auc.idxmax()]
            lo, hi = g.iloc[0], g.iloc[-1]
            rows.append(dict(model=m, knob=b.knob, best=b.value, val=b.val_auc, gap=b.train_auc - b.val_auc,
                             ends=f"{lo.value}: {lo.train_auc:.2f}/{lo.val_auc:.2f} -> {hi.value}: {hi.train_auc:.2f}/{hi.val_auc:.2f}"))
        say(f"{label} (train/val AUC at the two ends of each sweep):")
        say()
        table(pd.DataFrame(rows), ["model", "knob", "best", "val", "gap", "ends"],
              ["Model", "Knob", "Chosen", "Val AUC", "Train-val gap", "Sweep ends (train/val)"])
    lc = rd("default_course_learning_curve")
    say(f"Learning curve (CatBoost, Dataset 1): train-CV gap {lc.train_auc_mean.iloc[0] - lc.cv_auc_mean.iloc[0]:.3f} "
        f"at {lc.n_train.iloc[0]:,} rows -> {lc.train_auc_mean.iloc[-1] - lc.cv_auc_mean.iloc[-1]:.3f} at "
        f"{lc.n_train.iloc[-1]:,} rows. More data, less variance.")

    # ------------------------------------------------------------------ 5
    section(5, "Calibration of the final Dataset 1 model")
    cal = rd("improved_default_calibration")
    table(cal, ["version", "auc", "brier", "log_loss", "ece"], ["Version", "AUC", "Brier", "Log loss", "ECE"],
          {"ece": lambda v: f"{v:.4f}"})
    th = rd("ext_default_threshold_theory")
    say("Cost-optimal thresholds found by search vs theory 1/(r+1): " +
        ", ".join(f"{int(r.ratio)}:1 -> {r.t_calibrated:.2f} (theory {r.theory:.3f})" for _, r in th.iterrows()))

    # ------------------------------------------------------------------ 6
    section(6, "Cost-sensitive decisions (Dataset 1, final model)")
    ct = rd("improved_default_cost_threshold")
    c10 = ct[ct.ratio == 10].copy()
    def policy(t):
        if t == "approve-all":
            return "Approve everyone"
        if t == "reject-all":
            return "Reject everyone"
        return "Model, t = 0.50" if float(t) == 0.5 else f"Model, cost-optimal t* = {float(t):.2f}"
    c10["policy"] = c10.threshold_star.astype(str).map(policy)
    say("Cost ratio 10:1 (a false approval costs 10 times a false rejection):")
    say()
    table(c10.sort_values("cost_per_applicant", ascending=False), ["policy", "fn", "fp", "cost_per_applicant"],
          ["Policy", "False approvals", "False rejections", "Cost per applicant"])
    rej = c10[c10.threshold_star == "reject-all"].cost_per_applicant.iloc[0]
    best = c10.cost_per_applicant.min()
    say(f"The cost-optimal threshold beats rejecting everyone by {100 * (1 - best / rej):.1f}%.")
    say()
    allr = ct[~ct.threshold_star.isin(["approve-all", "reject-all"])].copy()
    allr = allr[allr.threshold_star.astype(float) != 0.5]
    table(allr.sort_values("ratio"), ["ratio", "threshold_star", "threshold_theory", "cost_per_applicant"],
          ["Ratio FN:FP", "Chosen t*", "Theory 1/(r+1)", "Cost per applicant"],
          {"ratio": lambda v: f"{int(v)}:1", "threshold_star": lambda v: f"{float(v):.2f}"})

    # ------------------------------------------------------------------ 7
    section(7, "Fairness (Dataset 1, gender, 10:1)")
    pc = rd("ext_default_policy_comparison")
    pc = pc[(pc.attribute == "Gender") & (pc.ratio == 10)]
    table(pc, ["policy", "cost", "approval_rate", "dp_gap", "tpr_gap"],
          ["Policy", "Cost", "Approval rate", "Approval-rate gap", "Recall gap"],
          {"approval_rate": pct, "dp_gap": pp, "tpr_gap": pp})
    pi = rd("default_course_permutation_importance").head(5)
    say("Top 5 features by permutation importance (drop in test AUC when shuffled): " +
        ", ".join(f"{r.feature} {r.importance_mean:.3f}" for _, r in pi.iterrows()))

    # ------------------------------------------------------------------ 8
    section(8, "Key findings")
    best1 = im[im.model.str.startswith("CatBoost")].iloc[0]
    for line in [
        f"1. Leakage: {len(fl)} post-outcome columns give every model AUC 1.000; without them the honest test AUC is {best1.auc:.3f}.",
        "2. Dataset 2: removing Credit_History drops AUC from about 0.76 to 0.57; one field carries the signal.",
        f"3. Model choice: tuned boosting (CatBoost, MCC {best1.mcc:.2f}) beats linear, kernel and instance-based models on 148k loans;"
        " on 614 rows, logistic regression is as good as anything.",
        "4. Bias-variance: tree depth, k, C and the L1/L2 penalty all show underfitting turning into overfitting.",
        f"5. Cost: t* = 0.09 halves the model's cost vs 0.50, but beats rejecting everyone by only {100 * (1 - best / rej):.0f}%.",
        "6. Fairness: the cost-optimal threshold nearly doubles the gender approval gap; equal approval rates cost under 2%.",
    ]:
        say(line)

    # ------------------------------------------------------------------ 9
    section(9, "Files to look at")
    for f in sorted(os.listdir(os.path.join(ROOT, "figures"))):
        say(f"- figures/{f}")
    say("- models/best_model_pipeline.joblib (final model) + models/model_metadata.json")

    if not a.no_save:
        with open(os.path.join(ROOT, "REPORT.md"), "w") as fh:
            fh.write("\n".join(OUT) + "\n")
        print("\nSaved: REPORT.md (open it in VS Code and press Ctrl+Shift+V for the formatted view)")


if __name__ == "__main__":
    main()
