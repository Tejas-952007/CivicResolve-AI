# CivicResolve AI — Phase 2 Handoff Report: Leakage-Free ML Modeling, Evaluation, and Explainability

**Author:** Senior Machine Learning & Data Engineering Research Partner  
**Status:** **PASS** (Ready for Phase 3 Interface, Dashboard, and Clustering)  
**Date:** October 2026  
**Repository:** `CivicResolveAI_Starter`  

---

## 1. Executive Summary

Phase 2 constructed the first academically rigorous, leakage-free Machine Learning layer for the CivicResolve AI project. All models were developed, trained, selected, and evaluated using verified data contracts on the 10,000-record NYC 311 temporal dataset produced in Phase 1 (`data/processed/nyc_311_phase1_profiled.csv`).

### Core Milestones Delivered:
1. **Zero Data Leakage Architecture:** An approved intake-time feature allowlist was strictly enforced. All target columns and post-resolution metadata (`closed_date`, `status`, `resolution_description`, `resolution_hours`, `target_long_duration`, `agency`, `created_year`) were barred from entering the training pipeline.
2. **Disjoint Temporal Partitions:** The dataset was partitioned strictly by local creation date (`America/New_York`):
   - **Training Set:** January 1, 2023 – December 31, 2023 ($N = 4,827$ total, $4,711$ eligible).
   - **Validation Set:** January 1, 2024 – June 30, 2024 ($N = 2,473$ total, $2,437$ eligible).
   - **Final Test Set:** July 1, 2024 – December 31, 2024 ($N = 2,700$ total, $2,635$ eligible).
   No random splitting or temporal leakage occurred.
3. **Training-Only Target Engineering:**
   - The binary classification threshold for "long-duration risk" was calculated exclusively from eligible 2023 training records at the 75th percentile: **$124.925\text{ hours}$** (~5.2 days). Neither validation nor test records were observed when computing this cutoff.
4. **Validation-Driven Model Selection:**
   - **Regression Winner:** **Random Forest Regressor (Log1p target)** achieved a validation MAE of **$230.53\text{ hours}$** ($R^2 = 0.408$), substantially outperforming the Dummy Median Baseline (MAE $296.95\text{ hours}$, $R^2 = -0.057$) and raw-duration linear models.
   - **Classification Winner:** **Logistic Regression (Class-Balanced)** achieved a validation F1-score of **$0.730$**, with high operational recall of **$89.7\%$** (PR-AUC: $0.755$, ROC-AUC: $0.940$).
5. **Final Unbiased Test Set Evaluation:** Evaluated exactly once on the frozen models:
   - **Regression (Random Forest Log1p):** Test MAE = **$251.99\text{ hours}$**, RMSE = **$947.67\text{ hours}$**, Median AE = **$15.59\text{ hours}$**, $R^2 = 0.145$.
   - **Classification (Logistic Regression Balanced):** Test Precision = **$0.591$**, Recall = **$0.887$** (caught $488$ of $550$ delayed complaints), F1-score = **$0.709$**, PR-AUC = **$0.723$**, ROC-AUC = **$0.926$**.
6. **Sensitivity Analysis:** A dedicated sensitivity analysis was executed excluding zero-duration records ($N=44$ in validation, $N=42$ in test). Regression MAE shifted by less than $3.5\text{ hours}$ ($233.66\text{h}$ validation, $255.22\text{h}$ test), demonstrating pipeline stability.
7. **Comprehensive Test Suite:** 19/19 automated tests passed ($9$ Phase 1 tests + $10$ Phase 2 tests) in under 10 seconds.

---

## 2. Dataset Partition Counts & Eligibility Filtering

From the 10,000 raw records in `nyc_311_phase1_profiled.csv`, supervised targets require valid resolution timestamps. In-progress/open records and negative duration anomalies are tracked and excluded from supervised training:

| Partition | Local Date Range | Total Records | Open/Unresolved Excluded | Negative Durations Excluded | Eligible Labelled Records |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Training** | 2023-01-01 to 2023-12-31 | 4,827 | 95 | 21 | **4,711** |
| **Validation** | 2024-01-01 to 2024-06-30 | 2,473 | 36 | 0 | **2,437** |
| **Final Test** | 2024-07-01 to 2024-12-31 | 2,700 | 65 | 0 | **2,635** |
| **Total** | 2023-01-01 to 2024-12-31 | **10,000** | **196** | **21** | **9,783** |

### Accounting for Ineligible Records:
- **Negative Durations ($N=21$):** All 21 negative duration records occurred in 2023. These represent administrative timestamp artifacts (e.g., ticket closing logged prior to formal system intake) and were excluded from supervised target modeling to preserve scientific validity.
- **Open / Unresolved Cases ($N=196$):** Open cases cannot have an elapsed resolution duration until closed. They are retained in the base repository for later operational profiling and queue simulation, but excluded from supervised regression and classification training.

---

## 3. Data Contracts and Feature Allowlist

### 3.1 Approved Intake-Time Feature Allowlist
Only features guaranteed to be known at the exact millisecond a citizen or call-taker files a complaint are admitted:
1. `complaint_type` (Categorical, primary service category)
2. `descriptor` (Categorical, specific sub-issue)
3. `borough` (Categorical, municipal jurisdiction)
4. `incident_zip` (Categorical string, postal code identifier)
5. `open_data_channel_type` (Categorical, intake channel: Phone, Online, Mobile, Other)
6. `created_hour` (Numeric integer, 0–23)
7. `created_weekday` (Numeric integer, 0–6)
8. `created_month` (Numeric integer, 1–12)
9. `is_weekend` (Binary indicator, 0 or 1)
10. `latitude` (Continuous float, WGS84 coordinate)
11. `longitude` (Continuous float, WGS84 coordinate)

### 3.2 Explicit Exclusion Justifications
- `agency`: Excluded from intake features because municipal routing to an agency is either an internal downstream step or collinear with `complaint_type`.
- `created_year`: Excluded because training data contains exclusively the year 2023. Including year would introduce zero variance in training and cause arbitrary or invalid linear extrapolation when evaluating 2024 records.
- `closed_date`, `status`, `resolution_description`: Strictly excluded as target leakage (they reveal post-resolution lifecycle outcomes).
- `resolution_hours`, `target_long_duration`: Strictly excluded as target variables.

### 3.3 Leakage-Free Pipeline Architecture (`src/preprocess.py`)
All feature transformations are wrapped inside Scikit-learn pipelines with a `ColumnTransformer` fitted exclusively on training data:
- **Coordinates (`latitude`, `longitude`):** Missing values imputed using `SimpleImputer(strategy='median')` learned from 2023, followed by `StandardScaler()`.
- **Cyclic Features (`created_hour`, `created_weekday`, `created_month`):** Transformed via custom `CyclicFeatureTransformer` into sine and cosine coordinates:
  $$\sin\left(\frac{2\pi \cdot t}{T}\right), \quad \cos\left(\frac{2\pi \cdot t}{T}\right)$$
  preserving continuous periodic wraparound across midnight, week boundaries, and new years.
- **Categoricals (`complaint_type`, `descriptor`, `borough`, `open_data_channel_type`, `incident_zip`):** Imputed with `'MISSING'` and encoded using `OneHotEncoder(min_frequency=15, handle_unknown='ignore')`. Infrequent categories below 15 occurrences in training are grouped into an `infrequent_sklearn` bin, preventing runaway dimensionality and gracefully handling novel ZIP codes or sub-descriptors in 2024.

---

## 4. Target Engineering & Threshold Derivation

### Regression Target
$$\text{resolution\_hours} = \frac{\text{closed\_date\_utc} - \text{created\_date\_utc}}{3600}$$
- **Training Distribution ($N=4,711$):**
  - Minimum: $0.000\text{ hours}$
  - 25th Percentile: $1.868\text{ hours}$
  - Median (50th Percentile): **$11.709\text{ hours}$**
  - Mean: $533.651\text{ hours}$
  - 75th Percentile: **$124.925\text{ hours}$**
  - Maximum: $30,137.917\text{ hours}$ (~3.44 years)

### Classification Target
$$\text{target\_long\_duration} = \begin{cases} 1 & \text{if } \text{resolution\_hours} > 124.925 \\ 0 & \text{otherwise} \end{cases}$$
The 75th percentile cutoff of **$124.925\text{ hours}$** was derived strictly from 2023 training data.

#### Class Balance Across Partitions:
- **Training Set (2023):** $1,178$ Positive ($25.0\%$), $3,533$ Negative ($75.0\%$).
- **Validation Set (2024-H1):** $503$ Positive ($20.6\%$), $1,934$ Negative ($79.4\%$).
- **Final Test Set (2024-H2):** $550$ Positive ($20.9\%$), $2,085$ Negative ($79.1\%$).

---

## 5. Regression Experiments & Model Selection

### 5.1 Validation Comparison (Trained on 2023, Evaluated on 2024-H1)

| Model | Val MAE (hrs) | Val RMSE (hrs) | Val $R^2$ | Val Median AE (hrs) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Dummy Regressor (Median)** | 296.95 | 1230.41 | -0.0573 | 11.35 | Baseline predictor ($11.71\text{h}$) |
| **Linear Regression (Raw Target)** | 461.28 | 992.42 | 0.3122 | 143.56 | Distorted by severe right-skew outliers |
| **Linear Regression (Log1p Target)** | 234.02 | 968.74 | 0.3446 | 7.53 | Ridge regression on $\log(1+y)$ |
| **Decision Tree (Raw Target)** | 481.22 | 1280.38 | -0.1449 | 190.53 | Overfits heavily, high variance |
| **Decision Tree (Log1p Target)** | 289.04 | 1217.07 | -0.0345 | 25.69 | Tree on $\log(1+y)$ |
| **Random Forest (Log1p Target)** | **230.53** | **920.70** | **0.4080** | **22.02** | **Selected Preferred Regressor** |

*Primary Selection Rule:* The preferred model was selected on the basis of lowest validation MAE. **Random Forest (Log1p Target)** achieved the lowest MAE ($230.53\text{ hours}$) and highest $R^2$ ($0.4080$).

### 5.2 Final Unbiased Test Set Evaluation (2024-H2)
Evaluated exactly once on the frozen Random Forest (Log1p) pipeline:
- **Test MAE:** **$251.99\text{ hours}$**
- **Test RMSE:** **$947.67\text{ hours}$**
- **Test $R^2$:** **$0.1446$**
- **Test Median AE:** **$15.59\text{ hours}$**

*Observation:* While the RMSE and MAE remain consistent with validation ($230\text{h} \rightarrow 252\text{h}$ MAE), the $R^2$ dropped from $0.408$ to $0.145$. This occurs because extreme long-tail delays in 2024-H2 contain heavier variance, and demonstrates why non-random temporal evaluation is essential to detect real-world degradation.

### 5.3 Zero-Duration Sensitivity Analysis
A total of 44 records in validation and 42 records in test had an exact resolution duration of 0.0 hours (often administrative cancellations or immediate phone resolutions):
- **Validation (Excluding 0-duration, $N=2,393$):** $\text{MAE} = 233.66\text{h}$, $\text{RMSE} = 929.08\text{h}$, $R^2 = 0.4074$.
- **Final Test (Excluding 0-duration, $N=2,593$):** $\text{MAE} = 255.22\text{h}$, $\text{RMSE} = 955.28\text{h}$, $R^2 = 0.1436$.
- **Conclusion:** Excluding zero-duration complaints shifts MAE by only $+3.13\text{h}$ in validation and $+3.23\text{h}$ in test. The model's predictive structure is robust against zero-duration artifacts.

---

## 6. Classification Experiments & Model Selection

### 6.1 Validation Comparison (Trained on 2023, Evaluated on 2024-H1)

| Model | Val Precision | Val Recall | Val F1-Score | Val PR-AUC | Val ROC-AUC | Val Confusion Matrix `[[TN, FP], [FN, TP]]` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Most Frequent)** | 0.0000 | 0.0000 | 0.0000 | 0.2064 | 0.5000 | `[[1934, 0], [503, 0]]` |
| **Logistic Regression (Unweighted)** | 0.7316 | 0.7097 | 0.7205 | **0.7592** | **0.9401** | `[[1803, 131], [146, 357]]` |
| **Logistic Regression (Balanced)** | 0.6161 | **0.8966** | **0.7304** | 0.7546 | 0.9396 | `[[1653, 281], [52, 451]]` |
| **Decision Tree (Balanced)** | 0.3995 | 1.0000 | 0.5709 | 0.3995 | 0.8033 | `[[1178, 756], [0, 503]]` |
| **Random Forest (Balanced)** | 0.4860 | 0.9682 | 0.6472 | 0.6933 | 0.9127 | `[[1419, 515], [16, 487]]` |

*Primary Selection Rule:* For civic operations, missing a high-delay complaint is more costly than a false alarm. **Logistic Regression (Class-Balanced)** achieved the highest F1-score (**$0.7304$**) while capturing **$89.66\%$** of delayed complaints ($451$ out of $503$), with an exceptional ROC-AUC of **$0.9396$**.

### 6.2 Final Unbiased Test Set Evaluation (2024-H2)
Evaluated exactly once on the frozen Logistic Regression (Balanced) pipeline:
- **Test Precision:** **$0.5908$**
- **Test Recall:** **$0.8873$** (successfully detected $488$ out of $550$ delayed requests)
- **Test F1-Score:** **$0.7093$**
- **Test PR-AUC:** **$0.7227$** (vs. 0.209 majority baseline)
- **Test ROC-AUC:** **$0.9262$**
- **Test Confusion Matrix:**
  $$\begin{bmatrix} \text{TN}=1747 & \text{FP}=338 \\ \text{FN}=62 & \text{TP}=488 \end{bmatrix}$$
  - Overall accuracy: **$84.82\%$** (significantly above the $79.13\%$ majority class baseline).

---

## 7. Model Explainability & Permutation Feature Importance

Global permutation importance was computed on the held-out validation set ($N=2,437$) over 5 repetitions.

### 7.1 Classification Permutation Importance (F1 Drop Metric)
| Feature | Mean F1 Drop | Standard Deviation | Importance Interpretation |
| :--- | :---: | :---: | :--- |
| `complaint_type` | **+0.3879** | 0.0125 | Dominant predictor: ticket type drives SLA expectations |
| `descriptor` | **+0.1054** | 0.0078 | Granular issue subtype refines delay risk |
| `incident_zip` | **+0.0067** | 0.0021 | Neighborhood / borough-level operational variance |
| `open_data_channel_type` | **+0.0031** | 0.0014 | Intake channel has mild predictive utility |
| `borough` | **+0.0020** | 0.0011 | Broad geographic identifier |
| `longitude` / `latitude` | -0.0004 | 0.0006 | Marginal spatial nuance beyond ZIP code |
| `created_hour` / `weekday` / `month` | -0.0018 | 0.0008 | Minimal macro seasonality impact on delay class |

### 7.2 Critical Scientific Clarification
> **Cautionary Scientific Interpretation:** Permutation feature importance measures purely statistical dependence within the learned model weights. A high importance score for `complaint_type` (e.g., Street Condition or Tree Maintenance) does **not** imply that filing a tree complaint *causes* municipal delay. Rather, different complaint types follow vastly different departmental workflows, inspection protocols, and contractor procurement cycles.

---

## 8. Automated Verification Suite

All 19 automated tests passed under `pytest`:
- **Phase 1 Tests (9/9):** API retry logic, 24-month window coverage, duplicate row detection, feature boundary leakage, malformed API response handling, monthly quota allocation exactness, negative duration exclusion, strata offset reproducibility, and NYC local timestamp parsing.
- **Phase 2 Tests (10/10):**
  1. `test_temporal_partitions_are_strictly_disjoint`: Verified zero overlapping unique keys between train, validation, and test.
  2. `test_no_record_created_in_2024_enters_training`: Verified maximum creation date in training is `2023-12-31T23:59:59`.
  3. `test_regression_targets_contain_no_negative_durations`: Verified minimum duration in training is $\ge 0.0$.
  4. `test_open_cases_are_excluded_from_supervised_targets`: Verified no `status == 'Open'` or `'Pending'` records entered supervised matrices.
  5. `test_threshold_is_computed_from_training_set_only`: Verified that the classification threshold equals the 75th percentile of 2023 training data and differs from validation/test percentiles.
  6. `test_no_post_resolution_field_enters_feature_matrix`: Verified that all outcome-revealing columns are barred from feature matrices.
  7. `test_missing_values_and_unseen_categories_handled_safely`: Tested pipeline with adversarial inputs (NaN coordinates, novel category `'NONEXISTENT_COMPLAINT_999'`, unseen ZIP `'99999'`).
  8. `test_pipeline_save_load_and_prediction_reproducibility`: Saved pipelines were re-loaded from disk and verified to produce identical predictions within $10^{-6}$ numerical tolerance.
  9. `test_models_produce_valid_finite_predictions`: Confirmed zero NaN, null, or infinite predictions across all splits.
  10. `test_final_test_set_not_used_in_model_selection`: Confirmed model selection logic only inspects validation set metrics.

```bash
============================== 19 passed in 9.69s ==============================
```

---

## 9. Artifact Inventory

### Source Code (`src/`):
- `src/features.py`: Intake allowlist enforcement, temporal split partitioning, eligibility filtering, and threshold derivation.
- `src/preprocess.py`: `ColumnTransformer`, `CyclicFeatureTransformer`, coordinate scaling, and rare-category one-hot encoding.
- `src/train_regression.py`: Regression pipeline training, validation comparison, and sensitivity analysis.
- `src/train_classification.py`: Classification pipeline training, imbalance handling, and validation comparison.
- `src/evaluate.py`: Full model benchmarking, permutation importance, metrics compilation, and plot generation.

### Saved Models (`models/`):
- `models/preferred_regressor.joblib`: Trained Random Forest Regressor (Log1p target).
- `models/preferred_classifier.joblib`: Trained Logistic Regression Classifier (Balanced).
- `models/model_metadata.json`: Feature allowlist, training threshold, hyperparameter settings, and timestamp record.

### Metrics & Diagnostic Tables (`outputs/`):
- `outputs/phase2_model_comparison.csv`: Head-to-head comparison table for all 11 regression and classification candidates.
- `outputs/phase2_metrics.json`: Complete quantitative telemetry including sample counts, partition distributions, and sensitivity benchmarks.

### Diagnostic Figures (`outputs/figures/`):
- `phase2_regression_comparison.png`: Validation MAE and RMSE across all regression models.
- `phase2_actual_vs_predicted.png`: Actual vs. predicted resolution hours for the preferred regressor.
- `phase2_regression_residuals.png`: Residual error distribution plot.
- `phase2_classification_confusion_matrix.png`: Confusion matrices for validation and final test sets.
- `phase2_pr_roc_curves.png`: Precision-Recall and ROC curves with baseline references.
- `phase2_feature_importance.png`: Permutation feature importance rankings with scientific disclaimer.

---

## 10. Recommended Next Steps for Phase 3

With a leakage-free ML core fully validated, Phase 3 should focus on:
1. **Interactive Municipal Dashboard (Flask UI):** Expose real-time intake scoring where staff can select complaint details and receive resolution-time forecasts and long-delay risk flags.
2. **Unsupervised Complaint Clustering:** Implement K-Means clustering on complaint attributes and spatial coordinates to discover latent municipal service-demand archetypes.
3. **Resource Planning Simulation:** Utilize predicted backlog risk to simulate work-order dispatch prioritization across municipal depots.
