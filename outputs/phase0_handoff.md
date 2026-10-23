# CivicResolve AI — Phase 0: Real Dataset Verification and Project Initialization Handoff Report

**Date:** 2026-10-24  
**Status Decision:** **PASS WITH LIMITATIONS**  
**Repository:** `CivicResolveAI_Starter`  
**Target Academic Scope:** Explainable Municipal Complaint Intelligence, Resolution Forecasting, and Resource Planning (College Mini-Project)

---

## 1. Executive Summary

Phase 0 aimed to verify the official NYC Open Data 311 API, validate the project environment, test bounded data ingestion without downloading multi-million-row tables, profile data quality and missingness, and establish baseline scientific integrity rules before any machine learning models are designed or trained.

The pipeline executed cleanly and produced all required artifacts. The public API was verified as live and accessible without API keys or paid tiers. However, empirical profiling revealed that the quarter-by-quarter ordered sampling strategy introduces severe temporal clustering and complaint-type bias. Therefore, Phase 0 is marked **PASS WITH LIMITATIONS**: the data pipeline is technically sound and verified, but the preliminary sample must not be used for final model training without the recommended sampling adjustments.

---

## 2. Repository Components Found

Prior to code execution, a full inspection of `CivicResolveAI_Starter` identified:

| Component | Path | Status Prior to Phase 0 | Description |
|---|---|---|---|
| Project Scope | `PROJECT_SCOPE.md` | Present | Scope lock specifying MVP techniques, features, and guardrails |
| Documentation | `README.md` | Present | Setup guide, command references, and sampling caveats |
| Dependency List | `requirements.txt` | Present | 10 core packages (pandas, numpy, requests, scikit-learn, flask, etc.) |
| Version Control Config | `.gitignore` | Present | Configured to ignore virtual environments, raw/processed CSVs, and figures |
| Ingestion Script | `scripts/01_fetch_sample.py` | Present | Script to fetch quarterly bounded chunks from NYC 311 API |
| Profiling Script | `scripts/02_profile_sample.py` | Present | Script to profile missingness, durations, and output figures |
| Source Module | `src/` | Empty directory | Awaiting Phase 1 modularization |
| Data Folders | `data/raw`, `data/processed` | Present (empty) | Target directories for fetched and profiled CSVs |
| Output Folders | `outputs/figures` | Present (empty) | Target directory for figures and reports |
| Model Store | `models/` | Empty directory | Awaiting Phase 2 model serialization |
| Web UI Assets | `static/css`, `templates/` | Present (empty) | Awaiting Phase 3 Flask dashboard |

---

## 3. Environment & Dependency Setup

- **Operating System:** Linux x86_64
- **Base Python:** `Python 3.12.3`
- **Virtual Environment:** Dedicated project `.venv` created at `/home/tejas/Downloads/CivicResolveAI_Starter/.venv`. System Python was untouched.
- **Installed Packages (`pip list` verification):**
  - `pandas` 3.1.0
  - `numpy` 2.5.4
  - `requests` 2.34.2
  - `scikit-learn` 1.9.1
  - `flask` 3.1.3
  - `matplotlib` 3.11.2
  - `seaborn` 0.13.2
  - `plotly` 7.1.0
  - `reportlab` 5.1.0
  - `joblib` 1.6.0
  - Plus standard sub-dependencies (`scipy` 1.18.1, `pillow` 13.0.0, `werkzeug` 3.1.9, `jinja2` 3.1.6, etc.)

---

## 4. Commands Executed & Real Execution Results

All commands were executed directly inside the local environment and inspected:

1. **Virtual environment creation:**
   ```bash
   python3 -m venv .venv
   ```
   *Result:* Exit code 0. `.venv` created.

2. **Package installation:**
   ```bash
   .venv/bin/python -m pip install --upgrade pip && .venv/bin/python -m pip install -r requirements.txt
   ```
   *Result:* Exit code 0. All 10 packages and wheels successfully compiled and installed.

3. **Sample API Fetch (Task 3):**
   ```bash
   .venv/bin/python scripts/01_fetch_sample.py --start-year 2023 --end-year 2024 --per-quarter 100
   ```
   *Result:* Exit code 0.
   - Quarters queried: 8 quarters (2023-Q1 through 2024-Q4).
   - Rows fetched: 800 total (100 per quarter).
   - Rows after deduplication: 800 (unique keys preserved).
   - File saved: `data/raw/nyc_311_preliminary_sample.csv`.

4. **Data Profiling (Task 4):**
   ```bash
   .venv/bin/python scripts/02_profile_sample.py
   ```
   *Result:* Exit code 0.
   - Output JSON: `outputs/data_quality_report.json`
   - Output CSV: `data/processed/nyc_311_preliminary_profiled.csv`
   - Output Figures: `outputs/figures/01_top_problem_categories.png`, `02_resolution_hours.png`, `03_sample_rows_by_month.png`

---

## 5. Observed Dataset & Profiling Statistics

The profiling script produced the following verified statistics recorded in `outputs/data_quality_report.json`:

### A. Volume & Dimensions
- **Sample Rows:** 800
- **Raw Columns:** 15
- **Profiled Columns:** 17 (`resolution_hours` and `created_month` added)
- **Duplicate `unique_key` rows:** 0 (0.00%)
- **Valid `created_date` and `closed_date` pairs:** 800 (100.00%)

### B. Observed Missingness by Field
| Column Name | Missing Count | Missing Percent | Role & Risk Assessment |
|---|---|---|---|
| `due_date` | 799 / 800 | 99.88% | Sparse; unusable as a reliable model feature |
| `descriptor` | 36 / 800 | 4.50% | High value sub-category; requires imputation/null category |
| `latitude` | 14 / 800 | 1.75% | Geospatial coordinate; imputable or drop nulls for geo models |
| `longitude` | 14 / 800 | 1.75% | Geospatial coordinate; matches latitude nullness |
| `resolution_description` | 9 / 800 | 1.12% | Post-resolution field (target leakage!); drop from inputs |
| `incident_zip` | 3 / 800 | 0.38% | Clean postal coverage |
| `unique_key` | 0 / 800 | 0.00% | Primary key |
| `created_date` | 0 / 800 | 0.00% | Timestamp |
| `closed_date` | 0 / 800 | 0.00% | Outcome timestamp (used only to compute target) |
| `agency` | 0 / 800 | 0.00% | Handling department |
| `agency_name` | 0 / 800 | 0.00% | Full department title |
| `complaint_type` | 0 / 800 | 0.00% | Primary category |
| `status` | 0 / 800 | 0.00% | Complaint status ("Closed" for all sampled rows) |
| `borough` | 0 / 800 | 0.00% | Municipal administrative borough |
| `open_data_channel_type` | 0 / 800 | 0.00% | Intake channel (PHONE, ONLINE, MOBILE) |

### C. Resolution Duration Metrics (`resolution_hours`)
- **Negative duration rows:** 0
- **Zero duration rows (`created_date == closed_date`):** 6 (0.75%)
  - *Investigation:* Analysis of resolution text reveals these represent instant administrative closures (jurisdictional redirection to state agencies, invalid address triage, or administrative transfer).
- **Non-negative duration records:** 800
- **Minimum resolution time:** 0.000 hours
- **Median resolution time:** 1.167 hours (~70.0 minutes)
- **90th percentile resolution time:** 43.990 hours (~1.83 days)
- **95th percentile resolution time:** 259.569 hours (~10.8 days)
- **Maximum resolution time:** 8,272.024 hours (~344.7 days)

### D. Top Complaint Categories in the Sample
1. `Noise - Residential`: 250 (31.25%)
2. `Noise - Street/Sidewalk`: 115 (14.38%)
3. `Illegal Parking`: 91 (11.38%)
4. `Noise - Commercial`: 60 (7.50%)
5. `Blocked Driveway`: 50 (6.25%)
6. `Illegal Fireworks`: 29 (3.62%)
7. `Noise - Vehicle`: 28 (3.50%)
8. `HEAT/HOT WATER`: 18 (2.25%)
9. `Noise`: 16 (2.00%)
10. `Street Condition`: 10 (1.25%)
- **Total distinct categories observed:** 69

---

## 6. Problems Discovered and Code Changes Made

1. **Defensive Response Validation & Schema Guarantee in `scripts/01_fetch_sample.py`:**
   - *Problem:* Socrata Open Data endpoints may return error dictionaries or omit columns completely when all rows in a batch contain `null` values for that field (such as `due_date`).
   - *Fix:* Added `isinstance(rows, list)` check inside the API query loop to fail explicitly if a non-list JSON payload is received. Added explicit reindexing to ensure all 15 expected `FIELDS` are preserved in consistent column order in the output CSV.

2. **Timezone Conversion Warning in `scripts/02_profile_sample.py`:**
   - *Problem:* Calling `df["created_date"].dt.to_period("M")` on UTC-aware timestamps caused pandas 3.1.0 to issue a `UserWarning: Converting to PeriodArray/Index representation will drop timezone information`.
   - *Fix:* Replaced `.dt.to_period("M").astype("string")` with `df["created_date"].dt.strftime("%Y-%m")`, eliminating the warning while producing standardized month strings.

---

## 7. Sampling Limitations and Scientific Integrity Risks

1. **Severe Temporal Compression & Boundary Clustering:**
   - Querying with `$order: created_date DESC` and `$limit: 100` per quarter fetched records only from the final ~15 to 30 minutes of the last day of each quarter (`2023-03-31`, `2023-06-30`, `2023-09-30`, `2023-12-31`, `2024-03-31`, `2024-06-30`, `2024-09-30`, `2024-12-31`).
   - Visualized clearly in `outputs/figures/03_sample_rows_by_month.png`: 100 records in months 3, 6, 9, 12, and 0 records in all intermediate months.

2. **Extreme Category Distortion (Nighttime Noise Bias):**
   - Because all records in this preliminary sample were filed late at night (between 23:40 and 23:59), noise complaints constitute **over 58% of the dataset** (Noise - Residential, Noise - Street/Sidewalk, Noise - Commercial, Noise - Vehicle, Noise).
   - This sample cannot be claimed as representative of general municipal complaints or daytime municipal service requests (e.g., building permits, water leaks, road repairs, sanitation).

3. **Target Leakage Prohibition:**
   - `closed_date`, `status`, and `resolution_description` occur downstream of intake. Using them as predictive features in regression or classification would represent fatal data leakage. They must remain strictly excluded from any feature engineering matrix.

4. **Duration vs. SLA Caution:**
   - Elapsed time (`closed_date - created_date`) reflects historical system operational throughput. It does not measure official regulatory SLA compliance or civic satisfaction unless explicitly mapped to formal agency service standards.

5. **External Validity & Municipal Context:**
   - NYC municipal infrastructure, agency hierarchy (NYPD, HPD, DSNY), and public complaint dynamics differ fundamentally from Indian municipalities (such as PCMC / Pimpri-Chinchwad Municipal Corporation). Findings must be presented academically as a transferable machine-learning methodology, not a direct operational deployment on Indian civic bodies without local data and administrative approvals.

---

## 8. Milestone Decision & Required Evidence for Phase 1

### Decision: **PASS WITH LIMITATIONS**

The Phase 0 acceptance criteria have been met:
- [x] Repository inspected and documented.
- [x] Dedicated `.venv` created and dependencies cleanly installed without touching system Python.
- [x] Official NYC 311 API successfully queried and verified without paid services or API keys.
- [x] Raw sample downloaded, deduplicated, and inspected.
- [x] Profile script executed, producing structured JSON and visualization artifacts.
- [x] Code defects/warnings fixed.
- [x] Scientific integrity checks, target leakage risks, and temporal biases thoroughly identified.

### Prerequisite Evidence Required Before Phase 1 Modelling:
1. **Sampling Revision:** Formulate a uniform or stratified temporal sampling mechanism (e.g. distributed daily or weekly samples across daytime and nighttime hours) to capture a representative distribution of complaint types.
2. **Feature Boundary Definition:** Explicitly define an intake-time-only feature contract (timestamp attributes, borough, zip, complaint_type, agency, channel).
3. **Temporal Evaluation Split:** Plan a strict out-of-time test split (e.g. train on 2023 data, validate on 2024-H1, test on 2024-H2) to test generalization without data leakage.
