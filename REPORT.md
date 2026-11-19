# Loan Approval and Default Prediction: Phase 2 results

BITS F464 Machine Learning, Concept Integration Assessment 2 · BITS Pilani, Dubai Campus
Kamya Chandebhamar · Anirudh Rakesh · Prerana CS

Convention: y = 1 is the adverse outcome (default / rejection), so a false negative is a FALSE APPROVAL.

## 1. Datasets

| Dataset          | Rows    | Label (y = 1)     | Label means          | Train / val / test       |
|------------------|---------|-------------------|----------------------|--------------------------|
| 1: Loan Default  | 148,670 | defaulted (24.6%) | repayment outcome    | 89,202 / 29,734 / 29,734 |
| 2: Loan Approval | 614     | rejected (31.3%)  | past lender decision | 368 / 123 / 123          |


## 2. Leakage audit (training rows only)

Dataset 1: 9 of 31 columns flagged and removed (F_clean). Dataset 2: none flagged.

| Flagged column            | % missing | Why                                                           |
|---------------------------|-----------|---------------------------------------------------------------|
| rate_of_interest          | 24.5      | missing-indicator AUC=0.997; adverse rate 100.0% when missing |
| Interest_rate_spread      | 24.6      | missing-indicator AUC=1.000; adverse rate 100.0% when missing |
| Upfront_charges           | 26.6      | missing-indicator AUC=0.984; adverse rate 92.2% when missing  |
| property_value            | 10.1      | missing-indicator AUC=0.706; adverse rate 100.0% when missing |
| credit_type               | 0.0       | level 'EQUI' has adverse rate 100.00% (n=9152)                |
| age                       | 0.1       | adverse rate 100.0% when missing                              |
| submission_of_application | 0.1       | adverse rate 100.0% when missing                              |
| LTV                       | 10.1      | missing-indicator AUC=0.706; adverse rate 100.0% when missing |
| dtir1                     | 16.2      | missing-indicator AUC=0.687                                   |

Cross-validated AUC with and without the leaked (Dataset 1) / dominant (Dataset 2) features:

| Dataset | Model    | All features | Without leaked / Credit_History |
|---------|----------|--------------|---------------------------------|
| 1       | LogReg   | 1.000        | 0.733                           |
| 1       | LightGBM | 1.000        | 0.787                           |
| 2       | LogReg   | 0.762        | 0.561                           |
| 2       | LightGBM | 0.764        | 0.574                           |


## 3. Model comparison on the untouched test split (threshold 0.50)

Dataset 1, 29,734 test loans (k-NN and SVMs fitted on a 20,000-row sample):

| Model                       | Accuracy | Precision | Recall | F1    | AUC   | MCC   |
|-----------------------------|----------|-----------|--------|-------|-------|-------|
| NaiveBayes                  | 0.776    | 0.652     | 0.199  | 0.305 | 0.706 | 0.269 |
| LinearSVM                   | 0.788    | 0.699     | 0.246  | 0.364 | 0.728 | 0.323 |
| kNN                         | 0.775    | 0.813     | 0.114  | 0.200 | 0.730 | 0.249 |
| LDA                         | 0.788    | 0.657     | 0.290  | 0.402 | 0.732 | 0.333 |
| LogReg-L2                   | 0.790    | 0.685     | 0.270  | 0.388 | 0.736 | 0.334 |
| RBF-SVM                     | 0.782    | 0.751     | 0.170  | 0.277 | 0.741 | 0.284 |
| DecisionTree                | 0.807    | 0.742     | 0.333  | 0.460 | 0.760 | 0.406 |
| CatBoost (Optuna)  <- final | 0.815    | 0.748     | 0.379  | 0.503 | 0.791 | 0.440 |
| Stacking                    | 0.816    | 0.730     | 0.402  | 0.518 | 0.791 | 0.444 |
| LightGBM (Optuna)           | 0.816    | 0.746     | 0.381  | 0.505 | 0.791 | 0.440 |
| XGBoost (Optuna)            | 0.816    | 0.750     | 0.381  | 0.505 | 0.791 | 0.442 |

Dataset 2, 123 test applications:

| Model               | Accuracy | Precision | Recall | F1    | AUC   | MCC   |
|---------------------|----------|-----------|--------|-------|-------|-------|
| DecisionTree        | 0.707    | 0.528     | 0.500  | 0.514 | 0.611 | 0.305 |
| LightGBM            | 0.748    | 0.630     | 0.447  | 0.523 | 0.689 | 0.368 |
| kNN                 | 0.813    | 0.895     | 0.447  | 0.596 | 0.696 | 0.542 |
| NaiveBayes          | 0.789    | 0.750     | 0.474  | 0.581 | 0.701 | 0.470 |
| RBF-SVM             | 0.821    | 0.900     | 0.474  | 0.621 | 0.712 | 0.564 |
| RandForest          | 0.813    | 0.895     | 0.447  | 0.596 | 0.715 | 0.542 |
| LDA                 | 0.813    | 0.857     | 0.474  | 0.610 | 0.747 | 0.538 |
| LinearSVM           | 0.813    | 0.857     | 0.474  | 0.610 | 0.760 | 0.538 |
| LogReg-L2  <- final | 0.805    | 0.818     | 0.474  | 0.600 | 0.762 | 0.514 |

Dataset 2 significance (corrected t-test, Holm): smallest adjusted p = 0.73, so no model is significantly better than another.

## 4. Bias-variance and regularisation (train vs validation AUC)

Dataset 1 (train/val AUC at the two ends of each sweep):

| Model        | Knob          | Chosen         | Val AUC | Train-val gap | Sweep ends (train/val)                               |
|--------------|---------------|----------------|---------|---------------|------------------------------------------------------|
| DecisionTree | max_depth     | 11             | 0.748   | 0.032         | 1: 0.53/0.53 -> 20: 0.90/0.67                        |
| LogReg-L2    | C             | 10.0           | 0.724   | 0.012         | 0.001: 0.73/0.72 -> 100.0: 0.74/0.72                 |
| LogReg-L1    | C             | 1.0            | 0.724   | 0.012         | 0.001: 0.71/0.70 -> 100.0: 0.74/0.72                 |
| LDA          | shrinkage     | auto           | 0.720   | 0.012         | none: 0.73/0.72 -> auto: 0.73/0.72                   |
| NaiveBayes   | var_smoothing | 0.1            | 0.693   | 0.009         | 1e-09: 0.68/0.67 -> 0.1: 0.70/0.69                   |
| kNN          | k             | 51             | 0.723   | 0.047         | 1: 1.00/0.60 -> 201: 0.74/0.72                       |
| LinearSVM    | C             | 0.1            | 0.721   | 0.020         | 0.001: 0.74/0.72 -> 10.0: 0.74/0.72                  |
| RBF-SVM      | C|gamma       | (0.1, 'scale') | 0.741   | 0.028         | (0.1, 'scale'): 0.77/0.74 -> (10.0, 0.01): 0.77/0.74 |

Dataset 2 (train/val AUC at the two ends of each sweep):

| Model        | Knob          | Chosen      | Val AUC | Train-val gap | Sweep ends (train/val)                               |
|--------------|---------------|-------------|---------|---------------|------------------------------------------------------|
| DecisionTree | max_depth     | 8           | 0.772   | 0.161         | 1: 0.70/0.72 -> 20: 1.00/0.71                        |
| LogReg-L2    | C             | 0.1         | 0.811   | -0.003        | 0.001: 0.77/0.79 -> 100.0: 0.82/0.81                 |
| LogReg-L1    | C             | 1.0         | 0.812   | -0.003        | 0.001: 0.50/0.50 -> 100.0: 0.82/0.81                 |
| LDA          | shrinkage     | auto        | 0.808   | 0.003         | none: 0.81/0.79 -> auto: 0.81/0.81                   |
| NaiveBayes   | var_smoothing | 0.1         | 0.797   | 0.002         | 1e-09: 0.81/0.77 -> 0.1: 0.80/0.80                   |
| kNN          | k             | 51          | 0.814   | -0.015        | 1: 1.00/0.76 -> 101: 0.80/0.80                       |
| LinearSVM    | C             | 0.01        | 0.810   | -0.004        | 0.001: 0.79/0.80 -> 10.0: 0.82/0.81                  |
| RBF-SVM      | C|gamma       | (1.0, 0.01) | 0.796   | 0.058         | (0.1, 'scale'): 0.86/0.79 -> (10.0, 0.01): 0.90/0.76 |

Learning curve (CatBoost, Dataset 1): train-CV gap 0.233 at 2,666 rows -> 0.045 at 26,666 rows. More data, less variance.

## 5. Calibration of the final Dataset 1 model

| Version    | AUC   | Brier | Log loss | ECE    |
|------------|-------|-------|----------|--------|
| raw        | 0.791 | 0.136 | 0.432    | 0.0047 |
| calibrated | 0.790 | 0.137 | 0.432    | 0.0037 |

Cost-optimal thresholds found by search vs theory 1/(r+1): 2:1 -> 0.32 (theory 0.333), 5:1 -> 0.17 (theory 0.167), 10:1 -> 0.09 (theory 0.091), 20:1 -> 0.05 (theory 0.048)

## 6. Cost-sensitive decisions (Dataset 1, final model)

Cost ratio 10:1 (a false approval costs 10 times a false rejection):

| Policy                        | False approvals | False rejections | Cost per applicant |
|-------------------------------|-----------------|------------------|--------------------|
| Approve everyone              | 7328            | 0                | 2.465              |
| Model, t = 0.50               | 4407            | 1069             | 1.518              |
| Reject everyone               | 0               | 22406            | 0.753              |
| Model, cost-optimal t* = 0.09 | 426             | 16526            | 0.699              |

The cost-optimal threshold beats rejecting everyone by 7.2%.

| Ratio FN:FP | Chosen t* | Theory 1/(r+1) | Cost per applicant |
|-------------|-----------|----------------|--------------------|
| 2:1         | 0.32      | 0.333          | 0.318              |
| 5:1         | 0.17      | 0.167          | 0.560              |
| 10:1        | 0.09      | 0.091          | 0.699              |
| 20:1        | 0.05      | 0.048          | 0.749              |


## 7. Fairness (Dataset 1, gender, 10:1)

| Policy                                | Cost  | Approval rate | Approval-rate gap | Recall gap |
|---------------------------------------|-------|---------------|-------------------|------------|
| P0 standard t=0.50                    | 1.544 | 87.2%         | 12.0 pp           | 25.4 pp    |
| P1 cost-optimal, one threshold        | 0.698 | 16.7%         | 21.8 pp           | 5.6 pp     |
| P2 cost-optimal + equal approval rate | 0.710 | 15.3%         | 2.8 pp            | 2.3 pp     |
| P3 cost-optimal + equal recall        | 0.708 | 17.2%         | 4.2 pp            | 0.7 pp     |

Top 5 features by permutation importance (drop in test AUC when shuffled): co-applicant_credit_type 0.108, income 0.091, Gender 0.063, Neg_ammortization 0.037, loan_amount 0.027

## 8. Key findings

1. Leakage: 9 post-outcome columns give every model AUC 1.000; without them the honest test AUC is 0.791.
2. Dataset 2: removing Credit_History drops AUC from about 0.76 to 0.57; one field carries the signal.
3. Model choice: tuned boosting (CatBoost, MCC 0.44) beats linear, kernel and instance-based models on 148k loans; on 614 rows, logistic regression is as good as anything.
4. Bias-variance: tree depth, k, C and the L1/L2 penalty all show underfitting turning into overfitting.
5. Cost: t* = 0.09 halves the model's cost vs 0.50, but beats rejecting everyone by only 7%.
6. Fairness: the cost-optimal threshold nearly doubles the gender approval gap; equal approval rates cost under 2%.

## 9. Files to look at

- figures/fig_ablation.png
- figures/fig_calibration.png
- figures/fig_cost_curves.png
- figures/fig_course_models_bias_variance.png
- figures/fig_decision_curve.png
- figures/fig_leakage_audit.png
- figures/fig_learning_curves.png
- figures/fig_permutation_importance.png
- figures/fig_policy_comparison.png
- figures/fig_regularisation.png
- figures/fig_shap_disparity_approval.png
- figures/fig_shap_disparity_default.png
- models/best_model_pipeline.joblib (final model) + models/model_metadata.json
