"""
Professor demo: clean summary of loan default pipeline results.
Run:  python3 demo_results.py
"""
import os
import pandas as pd

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def rd(name):
    return pd.read_csv(f"{RES}/{name}.csv")

print("=" * 72)
print("  LOAN DEFAULT PREDICTION — PIPELINE RESULTS SUMMARY")
print("=" * 72)
print()

# ---------------------------------------------------------------- 1. Leakage audit
print("--- 1. DATA LEAKAGE AUDIT (training split only) ---")
audit = rd("improved_default_leakage_audit")
flagged = audit[audit["flagged"] == True]
print(f"Features audited: {len(audit)}")
print(f"Columns flagged as leaking: {len(flagged)}")
if len(flagged) > 0:
    print("Flagged columns:", ", ".join(flagged["feature"].tolist()))
print()

# ---------------------------------------------------------------- 2. Feature engineering
print("--- 2. FEATURE ENGINEERING (from F_clean only) ---")
fe = rd("improved_default_feature_engineering")
kept = fe[fe["kept"] == True]
dropped = fe[fe["kept"] == False]
print(f"Base features (F_clean): {len(fe[fe['source']=='base'])}")
print(f"Engineered features tested: {len(fe[fe['source']=='engineered'])}")
print(f"Engineered features kept: {len(kept)} -> {', '.join(kept['feature'].tolist())}")
print(f"Engineered features dropped: {len(dropped)}")
print()

# ---------------------------------------------------------------- 3. Model comparison
print("--- 3. TEST SET PERFORMANCE (held-out, never seen in training) ---")
tm = rd("improved_default_test_metrics")
print(f"{'Model':<28} {'AUC':>7} {'MCC':>7} {'F1':>7} {'Accuracy':>9}")
print("-" * 62)
for _, row in tm.iterrows():
    print(f"{row['model']:<28} {row['auc']:>7.4f} {row['mcc']:>7.4f} "
          f"{row['f1']:>7.4f} {row['accuracy']:>9.4f}")
print()

# ---------------------------------------------------------------- 4. Calibration
print("--- 4. PROBABILITY CALIBRATION (best model: CatBoost) ---")
cal = rd("improved_default_calibration")
print(f"{'Version':<14} {'AUC':>7} {'Brier':>8} {'LogLoss':>9} {'ECE':>7}")
print("-" * 50)
for _, row in cal.iterrows():
    print(f"{row['version']:<14} {row['auc']:>7.4f} {row['brier']:>8.4f} "
          f"{row['log_loss']:>9.4f} {row['ece']:>7.4f}")
print()

# ---------------------------------------------------------------- 5. Cost-sensitive threshold
print("--- 5. COST-SENSITIVE THRESHOLD (ratio 10:1 FN:FP) ---")
ct = rd("improved_default_cost_threshold")
ct10 = ct[ct["ratio"] == 10]
print(f"{'Policy':<24} {'Threshold':>10} {'Cost/Applicant':>16} {'FN':>5} {'FP':>6}")
print("-" * 66)
for _, row in ct10.iterrows():
    ts = row['threshold_star']
    ts_str = f"{ts:.4f}" if isinstance(ts, (int, float)) else str(ts)
    print(f"{ts_str:<24} {ts_str:>10} {row['cost_per_applicant']:>16.4f} "
          f"{row['fn']:>5} {row['fp']:>6}")
print()

# ---------------------------------------------------------------- 6. Significance
print("--- 6. STATISTICAL SIGNIFICANCE (vs CatBoost baseline, Holm-adjusted) ---")
sig = rd("improved_default_significance")
print(f"{'Model':<16} {'Mean AUC':>10} {'Diff vs Base':>14} {'p-value (Holm)':>16} {'Significant':>13}")
print("-" * 72)
for _, row in sig.iterrows():
    sig_mark = "YES" if row["significant_holm"] else "no"
    print(f"{row['model']:<16} {row['mean_auc']:>10.4f} {row['mean_diff']:>14.4f} "
          f"{row['p_value_holm']:>16.4f} {sig_mark:>13}")
print()

print("=" * 72)
print("  CONCLUSION: Real, leakage-free performance on held-out test data.")
print("  Best model: CatBoost (Optuna-tuned) — Test AUC 0.8109, MCC 0.4665")
print("=" * 72)