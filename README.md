# Loan Approval / Default Prediction: Phase 2 (Model Design and Implementation)

Implements the Phase 1 plan on two public Kaggle datasets: leakage-safe pipeline, feature ablation,
cost-sensitive decision thresholds, and TreeSHAP + Fairlearn fairness analysis.

## 0. Quick start in VS Code

1. **Open the folder:** File > Open Folder > `loan_project`. Install the Python extension if VS Code suggests it.
2. **Create the environment** (Terminal > New Terminal):

   ```bash
   python -m venv .venv
   # Windows:      .venv\Scripts\activate
   # macOS/Linux:  source .venv/bin/activate
   pip install -r requirements.txt
   ```

   Then Ctrl+Shift+P > "Python: Select Interpreter" > pick the one in `.venv`.
3. **See every result, organised:** `python report.py`. It prints nine sections (datasets, leakage audit,
   model comparison, bias-variance, calibration, cost, fairness, key findings, figures) in about a second, using the
   saved results, and writes them to `REPORT.md` (open it and press Ctrl+Shift+V for the formatted view).
4. **Score new applications with the saved model:** `python src/predict.py --input data/sample_applications.csv --output predictions.csv`.

The same runs are one click away in the Run and Debug panel (Ctrl+Shift+D): pick a configuration, press F5.
Retraining is only needed if you change the code; the committed `results/`, `figures/` and `models/` are the full-run outputs.

## 1. Data

| Dataset | File in `data/` | Kaggle source | Rows x cols |
|---|---|---|---|
| 1. Loan Default | `Loan_Default.csv` | `yasserh/loan-default-dataset` | 148,670 x 34 |
| 2. Loan Approval | `loan_data_set.csv` | `burak3ergun/loan-data-set` | 614 x 13 |

If `Loan_Default.csv` is missing from the ZIP (it is 27 MB), download it from the Kaggle page above and put it
in `data/`. Use `python src/run_experiments.py --data <folder>` to read the CSVs from another folder.

## 2. Setup and run

Dependencies (all in `requirements.txt`, Python 3.10+): numpy, pandas, scipy, scikit-learn >= 1.4, imbalanced-learn,
lightgbm, xgboost, catboost, optuna, shap, fairlearn, matplotlib.

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
bash run_all.sh                                        # everything below, in the right order
```

Or step by step (order matters: later scripts read earlier results):

```bash
python src/run_experiments.py      # 1. baseline: leakage audit, ablation, 5 models, CV, cost thresholds, fairness
python src/run_extensions.py       # 2. engineered features, Optuna-tuned boosters, stacking, calibration,
                                   #    significance tests, fairness-constrained thresholds (reads results/*_tuning.csv)
python src/run_improved.py         # 3. final Dataset 1 pipeline; saves the deployable model to models/
python src/run_course_methods.py   # 4. permutation importance, PSI, learning curve, calibration slope, decision curves
python src/run_course_models.py    # 4b. course algorithms: decision tree, L1/L2 LogReg, LDA, Naive Bayes, k-NN, linear/RBF SVM
python src/make_figures.py         # 5. all figures in figures/, built from the CSVs only
python demo_results.py             # 6. one-screen summary of the final results (for the presentation)
```

Every experiment script accepts `--quick` (15% sample of Dataset 1, fewer trials) for a 1-2 minute smoke test, and
`--data <folder>` to read the CSVs from another folder. The full baseline run tunes 5 models x 12 grid points on
148k rows and is slow on a 2-core laptop (CatBoost/XGBoost at depth 12 dominate); the committed `results/` already
contain the full-run outputs, so `make_figures.py` and `demo_results.py` work without re-running anything.
**Note:** `--quick` writes to the same `results/` files, so run it on a copy if you want to keep the full-run tables.

Scoring new applications with the saved model:

```bash
python src/predict.py --input data/new_applications.csv --output predictions.csv   # uses the 10:1 threshold
```

Everything is seeded (`SEED = 42` in `src/common.py`), so reruns reproduce the same tables
(tiny differences can appear across library versions / thread counts for tree ensembles).

## 3. Code layout

| File | Purpose |
|---|---|
| `src/common.py` | loading, 60/20/20 split, leakage audit, preprocessing, models and grids, metrics, cost function, course diagnostics (permutation importance, PSI, learning curve, calibration slope, decision curve) |
| `src/run_experiments.py` | baseline experiments (leakage audit, ablation, cost thresholds, fairness), one CSV per table |
| `src/run_extensions.py` | extensions: engineered ratios, Optuna-tuned LightGBM/XGBoost/CatBoost, stacking, calibration, paired significance tests, cost-optimal decisions under a fairness constraint (`ext_*.csv`) |
| `src/run_improved.py` | final Dataset 1 pipeline (feature selection, 50+ Optuna trials, stacking/blend, isotonic calibration, cost thresholds vs theory); writes `improved_*.csv` and `models/` |
| `src/run_course_methods.py` | course diagnostics on each dataset's final model (`*_course_*.csv`) |
| `src/run_course_models.py` | classical course algorithms with capacity sweeps (train vs validation AUC) and test metrics (`*_coursemodels_*.csv`) |
| `src/predict.py` | applies the saved pipeline and threshold to new applications |
| `src/make_figures.py` | all figures, built from the CSVs only |
| `report.py` | one-command organised summary of every result; writes `REPORT.md` |
| `demo_results.py` | short summary of the final Dataset 1 pipeline |
| `run_all.sh` | runs the whole pipeline in order |

## 4. Protocol (same for both datasets)

1. Stratified 60/20/20 train/validation/test split **before** any preprocessing.
2. Leakage audit on the training split only (`results/*_leakage_audit.csv`). It flags columns whose
   missing-value indicator alone separates the classes, columns with a near-pure missing group, and categorical
   levels that are almost perfectly pure. Flagged columns are removed to build `F_clean`.
   Feature sets: Dataset 1 = `F_full` vs `F_clean`; Dataset 2 = `F_full` vs `F_noCreditHistory`.
3. Imputation (median plus missing indicators / constant), scaling and one-hot encoding live inside a scikit-learn
   `Pipeline`, so they are fitted on training data only. SMOTE (when used) is inside an imbalanced-learn pipeline,
   so it only touches training folds.
4. Models: Logistic Regression (L2), Random Forest, LightGBM, XGBoost, CatBoost. Hyper-parameters are chosen on the validation split.
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

## 4c. Course diagnostics (`run_course_methods.py`)

Run on the headline model of each dataset (chosen by validation AUC; CatBoost on Dataset 1, LogReg on Dataset 2):

14. Permutation importance: drop in test AUC when one column is shuffled (repeated, seeded).
15. Population Stability Index: score PSI (validation vs test scores of a train-only model) and per-feature PSI
    (train vs test). Rule of thumb: < 0.10 stable, 0.10-0.25 moderate, > 0.25 significant shift.
16. Learning curve: train vs cross-validated AUC as the training set grows (bias-variance diagnosis).
17. Calibration intercept and slope from logistic recalibration (ideal 0 and 1).
18. Decision curve analysis: net benefit vs reject-all / approve-all across threshold probabilities.

## 4d. Course algorithms (`run_course_models.py`)

19. Decision tree (max_depth 1-20), logistic regression with L1 and L2 penalties (C = 0.001-100), LDA (with / without
    shrinkage), Gaussian Naive Bayes, k-NN (k = 1-201), linear SVM (C) and RBF-kernel SVM (C x gamma), all under the
    same split, audit and train-only preprocessing. Each knob is chosen on validation AUC, the chosen model is refit on
    train+val and tested once. On Dataset 1, k-NN and the SVMs use a stratified 20,000-row training sample.
    Outputs: `*_coursemodels_sweep.csv` (train vs validation AUC per setting) and `*_coursemodels_test.csv`;
    figure `fig_course_models_bias_variance`.

## 5. Expected outputs

`results/` holds one CSV per table named `<dataset>_<table>.csv` (`default` = Dataset 1, `approval` = Dataset 2),
plus `ext_*` (extensions), `improved_*` (final Dataset 1 pipeline) and `*_course_*` (course diagnostics).
Test-set accuracy, precision, recall, F1, AUC and MCC are in `*_test_metrics.csv`.
`figures/` holds `fig_leakage_audit`, `fig_ablation`, `fig_cost_curves`, `fig_learning_curves`, `fig_regularisation`,
`fig_shap_disparity_*`, `fig_calibration`, `fig_policy_comparison`, `fig_permutation_importance`, `fig_decision_curve` and `fig_course_models_bias_variance`.
`models/` holds the fitted final pipeline, its feature list, the 10:1 threshold and `model_metadata.json`.
Each script prints a timestamped progress log and ends with `ALL DONE`.

## 6. Known limitations

* **Sample size of the Dataset 1 baseline tables.** The committed `default_*.csv` baseline tables (grid search over
  5 models, F_full vs F_clean ablation, CV, cost thresholds, fairness) were produced with `--quick`, i.e. on a
  stratified 15% sample (22,300 rows; test split 4,460 rows), because the full grid is very slow on a laptop.
  On that sample the audit flags 7 columns (`age` and `submission_of_application` are not flagged).
  The extensions (`ext_default_*`), the final pipeline (`improved_default_*`, `models/`), the course diagnostics
  (`default_course_*`) and `default_leakage_audit.csv` use all 148,670 rows (test split 29,734 rows; 9 flagged columns).
  Run `python src/run_experiments.py` without `--quick` to rebuild the baseline tables on the full data.* Dataset 2's test split has only 123 rows (19 female applicants), so its test metrics and fairness gaps are noisy.
  Prefer the repeated-CV numbers and the bootstrap intervals.
* `dtir1`, `age` and `submission_of_application` are flagged by the audit rules and removed in `F_clean`, so
  `F_clean` is a conservative feature set.
* SMOTE is applied to one-hot encoded data (plain SMOTE, not SMOTENC).
* On Dataset 2 the Optuna validation AUC (about 0.84-0.86) overstates test AUC (about 0.73-0.75): 123 validation rows
  are used to choose among 20-30 settings. Use the cross-validated and paired-test results.
* Fairlearn's `ExponentiatedGradient` returns randomised hard decisions, so no AUC is reported for it.
