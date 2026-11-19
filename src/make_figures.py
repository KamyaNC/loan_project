"""Builds every figure used in the article from the CSVs in results/ (run after run_experiments.py)."""
import ast
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = f"{ROOT}/results", f"{ROOT}/figures"
os.makedirs(FIG, exist_ok=True)

# Okabe-Ito colour-blind-safe palette, assigned in a fixed order (never cycled)
BLUE, ORANGE, GREEN, RED, GREY, INK = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#8A8F98", "#222222"
MODEL_COL = {"LogReg": BLUE, "RandForest": ORANGE, "LightGBM": GREEN}
TITLES = {"approval": "Dataset 2: Loan Approval (614 rows)", "default": "Dataset 1: Loan Default (148,670 rows)"}
HEADLINE = {"approval": "F_full", "default": "F_clean"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#666",
                     "axes.titlesize": 10, "axes.titleweight": "bold", "figure.facecolor": "white",
                     "axes.facecolor": "white", "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False})


def rd(name):
    return pd.read_csv(f"{RES}/{name}.csv")


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- 1. leakage audit
def fig_leakage():
    a = rd("default_leakage_audit")
    a = a[a.n_missing > 0].copy()
    a["dev"] = (a.missing_indicator_auc - 0.5).abs() + 0.5
    a = a.sort_values("dev")
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.barh(a.feature, a.dev, color=[RED if f else GREY for f in a.flagged], height=0.6)
    ax.axvline(0.5, color=INK, lw=0.8)
    ax.axvline(0.6, color=INK, lw=0.8, ls="--")
    ax.text(0.602, -0.9, "flag threshold 0.60", fontsize=8, color=INK, va="top")
    ax.set_xlim(0.45, 1.02)
    ax.set_xlabel("AUC of the 'value is missing' indicator alone (0.5 = no information)")
    ax.set_title("Which missing-value patterns reveal the outcome? (training split)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_leakage_audit")


# ---------------------------------------------------------------- 2. ablation (CV AUC +- sd)
def fig_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for ax, ds in zip(axes, ["default", "approval"]):
        cv = rd(f"{ds}_cv")
        cv = cv[cv.metric == "auc"]
        sets = list(dict.fromkeys(cv.feature_set))
        w = 0.38
        for i, (s, col) in enumerate(zip(sets, [GREY, BLUE])):
            d = cv[cv.feature_set == s].set_index("model").loc[["LogReg", "RandForest", "LightGBM"]]
            x = np.arange(3) + (i - 0.5) * w
            ax.bar(x, d["mean"], w * 0.92, yerr=d["std"].fillna(0), color=col, label=s, capsize=2, error_kw=dict(lw=0.8))
            for xi, v, sd in zip(x, d["mean"], d["std"].fillna(0)):
                ax.text(xi, min(v + sd + 0.008, 1.005), f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["LogReg", "RandForest", "LightGBM"])
        ax.axhline(0.5, color=INK, lw=0.8, ls="--")
        ax.set_ylim(0.5, 1.06)
        ax.set_title(TITLES[ds], fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Cross-validated AUC (mean ± sd)")
    fig.suptitle("Removing leaking / dominant features changes the picture", fontsize=10, fontweight="bold", y=1.02)
    save(fig, "fig_ablation")


# ---------------------------------------------------------------- 3. cost curves
def fig_cost():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    cols = {2: BLUE, 5: ORANGE, 10: GREEN, 20: RED}
    ct = None
    for ax, ds in zip(axes, ["default", "approval"]):
        cc = rd(f"{ds}_cost_curves")
        cc = cc[cc.feature_set == HEADLINE[ds]]
        ct = rd(f"{ds}_cost_threshold")
        ct = ct[ct.feature_set == HEADLINE[ds]]
        for r, col in cols.items():
            d = cc[cc.ratio == r]
            base = d.loc[d.threshold == 0.5, "test_cost"].iloc[0]
            ax.plot(d.threshold, d.test_cost / base, color=col, lw=1.6, label=f"{r}:1")
            t = ct[ct.ratio == r].t_star.iloc[0]
            ax.plot([t], [d.loc[d.threshold == t, "test_cost"].iloc[0] / base], "o", color=col, ms=6, mec="white", mew=1)
        ax.axvline(0.5, color=INK, lw=0.8, ls="--")
        ax.text(0.51, 0.03, "standard 0.50", fontsize=8, transform=ax.get_xaxis_transform())
        ax.set_title(TITLES[ds], fontsize=9)
        ax.set_xlabel("decision threshold (reject if P(adverse) ≥ t)")
        ax.set_ylim(bottom=0)
        ax.legend(title="cost ratio FN:FP", fontsize=8, title_fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.12))
    axes[0].set_ylabel("test cost relative to t = 0.50")
    fig.suptitle("Cost-sensitive thresholds (dots = threshold chosen on out-of-fold training predictions)",
                 fontsize=10, fontweight="bold", y=1.02)
    save(fig, "fig_cost_curves")


# ---------------------------------------------------------------- 4. learning curves
def fig_learning():
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=False)
    for ax, ds in zip(axes, ["default", "approval"]):
        lc = rd(f"{ds}_learning_curve")
        for m in ("LogReg", "LightGBM"):
            d = lc[lc.model == m]
            ax.plot(d.n_train, d.train_auc, color=MODEL_COL[m], ls="--", lw=1.4)
            ax.plot(d.n_train, d.val_auc, color=MODEL_COL[m], lw=1.8)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("training examples")
        ax.set_title(f"{TITLES[ds]} [{HEADLINE[ds]}]", fontsize=9)
    axes[0].set_ylabel("AUC")
    h = [Line2D([], [], color=MODEL_COL["LogReg"], lw=1.8, label="LogReg"),
         Line2D([], [], color=MODEL_COL["LightGBM"], lw=1.8, label="LightGBM"),
         Line2D([], [], color=INK, lw=1.4, ls="--", label="train"),
         Line2D([], [], color=INK, lw=1.8, label="validation")]
    axes[0].legend(handles=h, fontsize=8, loc="upper right", ncol=2)
    fig.suptitle("Bias-variance: the train/validation gap shrinks with data", fontsize=10, fontweight="bold", y=1.02)
    save(fig, "fig_learning_curves")


# ---------------------------------------------------------------- 5. regularisation
def fig_regularisation():
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.6))
    for r, ds in enumerate(["default", "approval"]):
        t = rd(f"{ds}_tuning")
        t = t[t.feature_set == HEADLINE[ds]].copy()
        t["p"] = t.params.apply(ast.literal_eval)
        lr = t[t.model == "LogReg"].assign(x=lambda d: d.p.apply(lambda p: p["C"]))
        gb = t[(t.model == "LightGBM") & t.p.apply(lambda p: p.get("min_child_samples") == 20)].assign(
            x=lambda d: d.p.apply(lambda p: p["num_leaves"]))
        for c, (d, lab, lg) in enumerate([(lr, "LogReg: inverse regularisation strength C", True),
                                          (gb, "LightGBM: num_leaves (model complexity)", False)]):
            ax = axes[r, c]
            d = d.sort_values("x")
            ax.plot(d.x, d.train_auc, color=GREY, ls="--", marker="o", ms=5, lw=1.4, label="train")
            ax.plot(d.x, d.val_auc, color=BLUE, marker="o", ms=5, lw=1.8, label="validation")
            if lg:
                ax.set_xscale("log")
            ax.set_xlabel(lab, fontsize=8.5)
            ax.set_title(f"{TITLES[ds].split(':')[0]} [{HEADLINE[ds]}]", fontsize=9)
            if c == 0:
                ax.set_ylabel("AUC")
            ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig_regularisation")


# ---------------------------------------------------------------- 6. SHAP disparity
def fig_shap():
    for ds in ["approval", "default"]:
        s = rd(f"{ds}_shap_disparity")
        attrs = list(dict.fromkeys(s.attribute))
        fig, axes = plt.subplots(1, len(attrs), figsize=(3.6 * len(attrs), 3.4))
        axes = np.atleast_1d(axes)
        for ax, a in zip(axes, attrs):
            d = s[s.attribute == a].nlargest(6, "disparity").iloc[::-1]
            ax.barh(d.feature, d.disparity, color=[RED if x else BLUE for x in d.is_sensitive_feature], height=0.6)
            ax.set_title(f"by {a}", fontsize=9)
            ax.set_xlabel("group gap in mean SHAP value (log-odds)", fontsize=8)
            ax.grid(axis="y", visible=False)
        fig.suptitle(f"{TITLES[ds]}: features that drive group differences (red = the sensitive feature itself)",
                     fontsize=9.5, fontweight="bold", y=1.03)
        fig.tight_layout()
        save(fig, f"fig_shap_disparity_{ds}")


# ---------------------------------------------------------------- 7. extensions (run_extensions.py outputs)
def fig_policy():
    """Cost vs demographic-parity gap for the four decision policies (Dataset 1, Gender and AgeGroup, two cost ratios)."""
    d = rd("ext_default_policy_comparison")
    cols = {"P0": GREY, "P1": BLUE, "P2": GREEN, "P3": ORANGE}
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0), sharex="row")
    for r_i, attr in enumerate(["Gender", "AgeGroup"]):
        for c_i, ratio in enumerate([5, 10]):
            ax = axes[r_i, c_i]
            q = d[(d.attribute == attr) & (d.ratio == ratio)]
            for _, row in q.iterrows():
                k = row.policy[:2]
                ax.errorbar(row.dp_gap * 100, row.cost, xerr=[[(row.dp_gap - row.dp_ci_lo) * 100], [(row.dp_ci_hi - row.dp_gap) * 100]],
                            fmt="o", color=cols[k], ms=7, capsize=2, lw=1, label=row.policy)
                ax.annotate(k, (row.dp_gap * 100, row.cost), textcoords="offset points", xytext=(6, 5), fontsize=8)
            ax.margins(y=0.14)
            ax.set_title(f"{attr}, cost ratio {ratio}:1", fontsize=9)
            if r_i == 1:
                ax.set_xlabel("approval-rate gap between groups (percentage points)")
            if c_i == 0:
                ax.set_ylabel("test cost per applicant")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Dataset 1: what a fairness constraint costs (lower left is better)", fontsize=10, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig_policy_comparison")


def fig_calibration():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    for ax, ds in zip(axes, ["default", "approval"]):
        r = rd(f"ext_{ds}_reliability")
        ax.plot([0, 1], [0, 1], color=INK, lw=0.8, ls="--")
        for ver, col in (("raw", RED), ("calibrated", BLUE)):
            q = r[r.version == ver]
            ax.plot(q.mean_pred, q.frac_pos, marker="o", ms=5, color=col, lw=1.6, label=ver)
        ax.set_title(TITLES[ds], fontsize=9)
        ax.set_xlabel("predicted probability of the adverse outcome")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("observed adverse rate")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Calibration on the test split (dashed = perfectly calibrated)", fontsize=10, fontweight="bold", y=1.02)
    save(fig, "fig_calibration")


# ---------------------------------------------------------------- 8. course methods (run_course_methods.py outputs)
def fig_permutation():
    """Top-10 features by drop in test AUC when the column is shuffled (error bar = sd over repeats)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for ax, ds in zip(axes, ["default", "approval"]):
        d = rd(f"{ds}_course_permutation_importance").head(10).iloc[::-1]
        ax.barh(d.feature, d.importance_mean, xerr=d.importance_sd, color=BLUE, height=0.6,
                error_kw=dict(lw=0.8, capsize=2))
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(f"{TITLES[ds]} [{d.model.iloc[0]}]", fontsize=9)
        ax.set_xlabel("drop in test AUC when the column is shuffled")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Permutation importance of the final models (test split)", fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig_permutation_importance")


def fig_decision_curve():
    """Net benefit of the model vs the two trivial policies; dotted line marks the 10:1 cost ratio (pt = 1/11)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, ds in zip(axes, ["default", "approval"]):
        d = rd(f"{ds}_course_decision_curve")
        ax.plot(d.threshold, d.net_benefit_model, color=BLUE, lw=1.8, label=f"model ({d.model.iloc[0]})")
        ax.plot(d.threshold, d.net_benefit_reject_all, color=RED, lw=1.4, ls="--", label="reject everyone")
        ax.plot(d.threshold, d.net_benefit_approve_all, color=GREY, lw=1.4, label="approve everyone")
        ax.axvline(1 / 11, color=INK, lw=0.8, ls=":")
        ax.text(1 / 11 + 0.01, 0.95, "10:1", fontsize=8, transform=ax.get_xaxis_transform())
        top = d.net_benefit_model.max()
        ax.set_ylim(-0.05 * max(top, 0.05), top * 1.25 + 0.01)
        ax.set_title(TITLES[ds], fontsize=9)
        ax.set_xlabel("threshold probability pt (reject if P(adverse) ≥ pt)")
    axes[0].set_ylabel("net benefit")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Decision curve analysis: where the model beats the trivial policies (test split)",
                 fontsize=10, fontweight="bold", y=1.02)
    save(fig, "fig_decision_curve")


# ---------------------------------------------------------------- 9. course models (run_course_models.py outputs)
def fig_course_models():
    """Train vs validation AUC across each classical model's capacity knob (bias-variance, regularisation)."""
    panels = [("DecisionTree", "tree depth (more = more complex)", False),
              ("LogReg", "C = 1/regularisation strength", True),
              ("kNN", "k neighbours (more = simpler)", True),
              ("LinearSVM", "C (more = narrower margin)", True)]
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 5.8))
    for r, ds in enumerate(["default", "approval"]):
        sw = rd(f"{ds}_coursemodels_sweep")
        for c, (fam, xlab, logx) in enumerate(panels):
            ax = axes[r, c]
            fams = ["LogReg-L2", "LogReg-L1"] if fam == "LogReg" else [fam]
            for f, col in zip(fams, [BLUE, ORANGE]):
                d = sw[sw.model == f].copy()
                d["x"] = d.value.astype(float)
                d = d.sort_values("x")
                lab = f.split("-")[-1] if fam == "LogReg" else None
                ax.plot(d.x, d.train_auc, color=col, ls="--", lw=1.3)
                ax.plot(d.x, d.val_auc, color=col, lw=1.8, marker="o", ms=3.5, label=lab)
                b = d.loc[d.val_auc.idxmax()]
                ax.plot([b.x], [b.val_auc], "o", ms=8, mfc="none", mec=INK, mew=1.2)
            if logx:
                ax.set_xscale("log")
            ax.set_xlabel(xlab, fontsize=8)
            if r == 0:
                ax.set_title(fam if fam != "LogReg" else "LogReg: L1 vs L2", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{TITLES[ds].split(':')[0]}\nAUC", fontsize=9)
            if fam == "LogReg":
                ax.legend(fontsize=7.5, loc="lower right")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=INK, ls="--", lw=1.3, label="train"),
         Line2D([], [], color=INK, lw=1.8, label="validation"),
         Line2D([], [], color=INK, marker="o", ls="", mfc="none", ms=8, label="chosen (best validation)")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Underfitting to overfitting: train vs validation AUC across each model's capacity knob",
                 fontsize=10, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig_course_models_bias_variance")


if __name__ == "__main__":
    for f in (fig_leakage, fig_ablation, fig_cost, fig_learning, fig_regularisation, fig_shap):
        f()
    if os.path.exists(f"{RES}/ext_default_policy_comparison.csv"):   # written by run_extensions.py
        fig_policy()
        fig_calibration()
    if os.path.exists(f"{RES}/default_course_decision_curve.csv"):  # written by run_course_methods.py
        fig_permutation()
        fig_decision_curve()
    if os.path.exists(f"{RES}/default_coursemodels_sweep.csv"):     # written by run_course_models.py
        fig_course_models()
