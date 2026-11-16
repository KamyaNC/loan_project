# Loan Approval / Default Prediction: Phase 2 (Model Design and Implementation)

Implements the Phase 1 plan on two public Kaggle datasets: leakage-safe pipeline, feature ablation,
cost-sensitive decision thresholds, and TreeSHAP + Fairlearn fairness analysis.

## 1. Data

| Dataset | File in `data/` | Kaggle source | Rows x cols |
|---|---|---|---|
| 1. Loan Default | `Loan_Default.csv` | `yasserh/loan-default-dataset` | 148,670 x 34 |
| 2. Loan Approval | `loan_data_set.csv` | `burak3ergun/loan-data-set` | 614 x 13 |

If `Loan_Default.csv` is missing from the ZIP (it is 27 MB), download it from the Kaggle page above and put it
in `data/`. Use `python src/run_experiments.py --data <folder>` to read the CSVs from another folder.

## 2. Setup and run

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt                        # Python 3.10+
python src/run_experiments.py                          # ~4-5 min on 2 CPU cores, writes results/*.csv
python src/run_extensions.py                           # ~10 min: engineered features, XGBoost/CatBoost/Optuna/stacking,
                                                       #   calibration, significance tests, fairness-constrained thresholds
python src/make_figures.py                             # writes figures/*.png (run after both experiment scripts)
python src/run_experiments.py --quick                  # 1-minute smoke test (15% of Dataset 1)
python src/run_extensions.py --quick                   # 1-minute smoke test of the extensions
```

Everything is seeded (`SEED = 42` in `src/common.py`), so reruns reproduce the same tables.

## 3. Code layout

| File | Purpose |
|---|---|
| `src/common.py` | loading, 60/20/20 split, leakage audit, preprocessing, models and grids, metrics, cost function |
| `src/run_experiments.py` | baseline experiments (leakage audit, ablation, cost thresholds, fairness), one CSV per table |
| `src/run_extensions.py` | extensions: engineered ratios, Optuna-tuned LightGBM/XGBoost/CatBoost, stacking, calibration, paired significance tests, cost-optimal decisions under a fairness constraint (`ext_*.csv`); run it after `run_experiments.py`, since it reads the tuned settings from `results/*_tuning.csv` |
| `src/make_figures.py` | all figures, built from the CSVs only |

## 4. Protocol (same for both datasets)

1. Stratified 60/20/20 train/validation/test split **before** any preprocessing.
2. Leakage audit on the training split only (`results/*_leakage_audit.csv`). It flags columns whose
   missing-value indicator alone separates the classes, columns with a near-pure missing group, and categorical
   levels that are almost perfectly pure. Flagged columns are removed to build `F_clean`.
   Feature sets: Dataset 1 = `F_full` vs `F_clean`; Dataset 2 = `F_full` vs `F_noCreditHistory`.
3. Imputation (median plus missing indicators / constant), scaling and one-hot encoding live inside a scikit-learn
   `Pipeline`, so they are fitted on training data only. SMOTE (when used) is inside an imbalanced-learn pipeline,
   so it only touches training folds.
4. Models: Logistic Regression (L2), Random Forest, LightGBM. Hyper-parameters are chosen on the validation split.
   Train and validation AUC are both logged (`*_tuning.csv`) to show the bias-variance / regularisation trade-off.
5. Stability: stratified 5-fold CV on train+val (5 repeats for Dataset 2, 1 for Dataset 1) -> `*_cv.csv` (mean and sd).
6. Final model = best configuration refit on train+val, evaluated once on the untouched test split -> `*_test_metrics.csv`.
7. Cost-sensitive threshold: minimise `ratio x FN + FP` over t = 0.05...0.95 on out-of-fold train+val predictions,
   then apply to the test split. Ratios 2:1, 5:1, 10:1 (the Phase 1 choice) and 20:1 -> `*_cost_threshold.csv`.
   The same file contains the approve-everyone and reject-everyone costs, so gains can be judged against trivial policies.
8. Fairness on the test split: per-group approval rates, demographic-parity gap with a bootstrap 95% CI,
   equalized-odds gap (`*_fairness_*.csv`); TreeSHAP group disparity on a LightGBM model (`*_shap_disparity.csv`);
   Fairlearn `ExponentiatedGradient(DemographicParity)` mitigation (`*_mitigation.csv`).

Convention: `y = 1` is the adverse outcome (default / rejection), so "false negative" = false approval.
Dataset 2's label is a past lender decision, not a repayment outcome. Its cost figures measure agreement
with that history, not money saved.

## 4b. Extensions (`run_extensions.py`)

9. Engineered ratios (loan-to-income and similar; row-wise, no fitted statistics). Kept only if LightGBM cross-validated AUC does not fall.
10. LightGBM, XGBoost and CatBoost tuned with Optuna (TPE) on validation AUC; stacking ensemble (logistic-regression meta-learner).
11. Calibration of the best single booster on its out-of-fold predictions (isotonic for Dataset 1, sigmoid for Dataset 2); check of t* against 1/(r+1).
12. Paired model comparison on identical folds: corrected resampled t-test (Nadeau-Bengio) with Holm adjustment.
13. Four decision policies on cost and parity: P0 t=0.50, P1 one cost-optimal threshold, P2 per-group thresholds with equal approval rate,
    P3 per-group thresholds with equal recall (`ext_*_policy_comparison.csv`). Group thresholds use the protected attribute and are a
    measurement of the price of parity, not a deployment recommendation.

## 5. Expected outputs

`results/` holds one CSV per table named `<dataset>_<table>.csv` (`default` = Dataset 1, `approval` = Dataset 2).
`figures/` holds `fig_leakage_audit`, `fig_ablation`, `fig_cost_curves`, `fig_learning_curves`, `fig_regularisation`,
`fig_shap_disparity_*`, `fig_calibration` and `fig_policy_comparison`. The console prints a timestamped progress log and ends with `ALL DONE`.

## 6. Known limitations

* Dataset 2's test split has only 123 rows (19 female applicants), so its test metrics and fairness gaps are noisy.
  Prefer the repeated-CV numbers and the bootstrap intervals.
* `dtir1`, `age` and `submission_of_application` are flagged by the audit rules and removed in `F_clean`, so
  `F_clean` is a conservative feature set.
* SMOTE is applied to one-hot encoded data (plain SMOTE, not SMOTENC).
* On Dataset 2 the Optuna validation AUC (about 0.84-0.86) overstates test AUC (about 0.73-0.75): 123 validation rows
  are used to choose among 20-30 settings. Use the cross-validated and paired-test results.
* Fairlearn's `ExponentiatedGradient` returns randomised hard decisions, so no AUC is reported for it.
