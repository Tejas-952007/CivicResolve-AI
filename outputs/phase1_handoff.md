# CivicResolve AI — Phase 1: Sampling Redesign and Reliable Modeling Dataset Handoff Report

**Date:** 2026-10-24  
**Status Decision:** **PASS**  
**Repository:** `CivicResolveAI_Starter`  
**Milestone:** Phase 1: Stratified Temporal Sampling and Reliable Modeling Dataset

---

## 1. Executive Summary

In Phase 0, initial API connectivity was confirmed, but data profiling detected critical sampling flaws: the quarter-by-quarter ordered sampling (`ORDER BY created_date DESC LIMIT 100`) caused **100% of rows to cluster within the final 15–30 minutes of each quarter**, skewing complaint categories toward nighttime noise complaints (>58% of all rows).

In **Phase 1**, we completely redesigned the ingestion architecture:
1. Queried official Socrata population totals across all 24 calendar months of 2023–2024 (total population: **6,681,493 complaints**).
2. Deterministically allocated a target sample of **10,000 records** proportionally using the **Largest Remainder (Hamilton-Hare)** method.
3. Divided monthly row ranks into equal strata and selected reproducible offset positions using seed `42`.
4. Extracted small bounded chunks ($n=25$) with Socrata SODA 2 pagination (`$limit`, `$offset`, `$order: created_date ASC, unique_key ASC`), exponential backoff on HTTP 429/5xx, and progress checkpointing.
5. Preserved open/in-progress tickets alongside closed complaints without outcome-filtering.
6. Localized naive Socrata Floating Timestamps to `America/New_York` (US Eastern Time), calculating exact elapsed hours via UTC conversion to account for Daylight Saving Time (DST).
7. Verified that quarter-end clustering dropped from **100.0% to 0.9%**, distinct dates expanded from **8 to 389**, and distinct complaint categories increased from **69 to 148**.
8. Validated the pipeline via a 9-test automated test suite in `pytest` (100% pass rate).

Phase 1 achieves a **PASS** status. The resulting dataset is academically defensible, reproducible, and ready for Phase 2 feature engineering.

---

## 2. Primary Data Source & API Verification

- **Data Portal:** NYC Open Data (City of New York)
- **Dataset:** 311 Service Requests from 2020 to Present
- **Official Documentation:** [NYC Open Data 311 About Page](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9/about_data)
- **API Endpoint:** `https://data.cityofnewyork.us/resource/erm2-nwe9.json`
- **Protocol:** Socrata Open Data API (SODA 2)
- **Authentication:** Public authorized access without paid API keys or private credentials.

---

## 3. Timestamp Semantics & Parsing Audit

### A. Socrata Floating Timestamp Semantics
NYC 311 timestamps are stored as **Socrata Floating Timestamps** (`YYYY-MM-DDTHH:MM:SS.000` without a UTC offset suffix). According to City of New York Open Data documentation, these floating timestamps represent **publisher local time: America/New_York (US Eastern Time: EST/EDT)**.

### B. Accurate Parsing & Calendar Feature Derivation
In Phase 0, naive timestamps were converted using `pd.to_datetime(..., utc=True)`, which mistakenly treated NYC local clock times as UTC. In Phase 1, we established a mathematically and scientifically sound parsing protocol:
1. **Raw Preservation:** The raw string is preserved verbatim in `data/raw/nyc_311_temporal_sample.csv`.
2. **Local Localization:** 
   ```python
   df["created_dt_local"] = pd.to_datetime(df["created_date"]).dt.tz_localize(
       "America/New_York", ambiguous="NaT", nonexistent="shift_forward"
   )
   ```
3. **Elapsed Duration (Accounting for DST Transitions):**
   ```python
   created_utc = df["created_dt_local"].dt.tz_convert("UTC")
   closed_utc = df["closed_dt_local"].dt.tz_convert("UTC")
   df["resolution_hours"] = (closed_utc - created_utc).dt.total_seconds() / 3600.0
   ```
   Converting both endpoints to UTC before duration arithmetic accounts for 23-hour and 25-hour days caused by Eastern Time Daylight Saving Time shifts (e.g., spring forward in March, fall back in November).
4. **Intake Feature Engineering:**
   - `created_hour`: 0–23 (citizen call hour in New York City local time).
   - `created_weekday`: Monday through Sunday.
   - `created_month`: 1–12.
   - `created_year`: 2023 or 2024.
   - `is_weekend`: Binary flag (Saturday or Sunday).

---

## 4. Sampling Architecture (`scripts/03_build_temporal_sample.py`)

### A. Configuration Parameters
- **Target Date Range:** `2023-01-01` to `2024-12-31` (24 calendar months).
- **Target Sample Size:** 10,000 records.
- **Request Chunk Size:** 25 records per API query.
- **Random Seed:** `42`.
- **Paging Order:** `created_date ASC, unique_key ASC`.

### B. Observed 24-Month Population Counts & Allocated Quotas
Across the 24 calendar months, the total population was **6,681,493 service requests**. The Largest Remainder method allocated the following quotas:

| Month | Population ($N_m$) | Population Share | Quota ($Q_m$) | Planned Chunks | Sampled Rows |
|---|---|---|---|---|---|
| **2023-01** | 242,173 | 3.62% | 363 | 15 | 363 |
| **2023-02** | 225,002 | 3.37% | 337 | 14 | 337 |
| **2023-03** | 251,716 | 3.77% | 377 | 16 | 377 |
| **2023-04** | 247,496 | 3.70% | 370 | 15 | 370 |
| **2023-05** | 280,932 | 4.20% | 421 | 17 | 421 |
| **2023-06** | 267,168 | 4.00% | 400 | 16 | 400 |
| **2023-07** | 289,330 | 4.33% | 433 | 18 | 433 |
| **2023-08** | 282,757 | 4.23% | 423 | 17 | 423 |
| **2023-09** | 283,517 | 4.24% | 424 | 17 | 424 |
| **2023-10** | 302,769 | 4.53% | 453 | 19 | 453 |
| **2023-11** | 280,046 | 4.19% | 419 | 17 | 419 |
| **2023-12** | 271,817 | 4.07% | 407 | 17 | 407 |
| **2024-01** | 287,186 | 4.30% | 430 | 18 | 430 |
| **2024-02** | 240,436 | 3.60% | 360 | 15 | 360 |
| **2024-03** | 266,604 | 3.99% | 399 | 16 | 399 |
| **2024-04** | 267,565 | 4.00% | 401 | 17 | 401 |
| **2024-05** | 283,982 | 4.25% | 425 | 17 | 425 |
| **2024-06** | 306,200 | 4.58% | 458 | 19 | 458 |
| **2024-07** | 298,936 | 4.47% | 447 | 18 | 447 |
| **2024-08** | 284,823 | 4.26% | 426 | 18 | 426 |
| **2024-09** | 306,770 | 4.59% | 459 | 19 | 459 |
| **2024-10** | 306,803 | 4.59% | 459 | 19 | 459 |
| **2024-11** | 293,516 | 4.39% | 439 | 18 | 439 |
| **2024-12** | 313,949 | 4.70% | 470 | 19 | 470 |
| **Total** | **6,681,493** | **100.00%** | **10,000** | **411** | **10,000** |

### C. Stratified Offset Generation
To prevent boundary clustering:
- For each month with quota $Q_m$ and chunk size 25, $K_m = \lceil Q_m / 25 \rceil$ strata were constructed across the population rank space $[0, N_m - 25]$.
- An offset was sampled within each stratum using `np.random.RandomState(seed + month_index * 1000)`.
- This ensures that API calls span early-month, mid-month, and late-month records systematically.

### D. Ingestion Execution Metrics
- **Total Requests Executed:** 411 chunk requests.
- **Total Rows Downloaded:** 10,000.
- **Unique Records After Deduplication:** 10,000 (0 duplicates).
- **Target Shortfall:** 0.
- **API Retries / Rate Limiting:** Handled gracefully via exponential backoff; no fatal request failures.

---

## 5. Profiling & Empirical Data Quality Audit

Profiling results from [outputs/phase1_data_quality.json](file:///home/tejas/Downloads/CivicResolveAI_Starter/outputs/phase1_data_quality.json):

### A. Dimensions & Representation
- **Total Sample Records:** 10,000
- **Raw Columns:** 14
- **Profiled Columns:** 24
- **Total Distinct Creation Dates:** **389 unique calendar dates** (spanning all 24 months, with 13 to 19 distinct dates sampled in every month).

### B. Missingness by Field
| Field | Null Count | Missing % | Nature of Column |
|---|---|---|---|
| `descriptor` | 331 | 3.31% | Specific problem description (imputable via 'Unknown' token) |
| `target_long_duration` | 222 | 2.22% | Missing only on unresolved open tickets |
| `closed_date` | 196 | 1.96% | Unresolved open tickets |
| `resolution_hours` | 196 | 1.96% | Unresolved open tickets |
| `resolution_description` | 192 | 1.92% | Outcome field |
| `latitude` | 147 | 1.47% | Geospatial coordinate |
| `longitude` | 147 | 1.47% | Geospatial coordinate |
| `incident_zip` | 90 | 0.90% | Postal code |
| `complaint_type` | 0 | 0.00% | Primary category |
| `borough` | 0 | 0.00% | Administrative borough |
| `open_data_channel_type` | 0 | 0.00% | Intake channel |
| `agency` | 0 | 0.00% | Assigned agency |
| `created_date` | 0 | 0.00% | Intake timestamp |
| `unique_key` | 0 | 0.00% | Identifier |

### C. Resolution Durations & Anomalies
- **Completed Records:** 9,778 (97.78%)
- **Unresolved / Open Records:** 222 (2.22%) (Status: `In Progress`: 109, `Pending`: 24, `Open`: 11, `Assigned`: 7, `Started`: 2, `Unspecified`: 1)
- **Zero Duration Records:** 181 (1.81%). Investigation shows these represent instantaneous administrative triage closures (duplicate notifications, immediate state agency transfer, or invalid address redirection).
- **Negative Duration Records:** 21 (0.21%). These represent manual administrative recording errors in source Socrata data where `closed_date < created_date`. They are flagged and excluded from duration modeling.
- **Duration Quantiles for Valid Completed Cases ($n=9,777$):**
  - **Min:** 0.000 hours
  - **25th Percentile (p25):** 0.959 hours (~58 minutes)
  - **Median (p50):** **8.717 hours**
  - **75th Percentile (p75):** **99.303 hours (~4.14 days)**
  - **90th Percentile (p90):** 598.347 hours (~24.9 days)
  - **95th Percentile (p95):** 1,851.414 hours (~77.1 days)
  - **99th Percentile (p99):** 9,498.681 hours (~395.8 days)
  - **Max:** 30,168.850 hours (~3.44 years)
  - **Interquartile Range (IQR):** 98.344 hours

### D. Comparison: Phase 0 vs. Phase 1 Elimination of Sampling Bias

| Dimension | Phase 0 (Feasibility) | Phase 1 (Stratified Temporal Sample) | Improvement Verdict |
|---|---|---|---|
| **Total Rows** | 800 | 10,000 | +1,150% data volume |
| **Distinct Dates** | 8 dates | **389 dates** | Comprehensive calendar coverage |
| **Quarter-End Clustered Rows** | **100.0%** (800 / 800) | **0.90%** (90 / 10,000) | **Temporal clustering eliminated** |
| **Top Category Share** | 31.3% (`Noise - Residential`) | 14.7% (`Illegal Parking`) | Natural demand diversity restored |
| **Total Noise Complaints** | **58.6%** of sample | **17.2%** of sample | Nighttime noise bias resolved |
| **Distinct Problem Categories** | 69 categories | **148 categories** | Captures municipal breadth |
| **Diurnal Cycle** | 100% late night (23:40–23:59) | Full 24-hour cycle (peaks at 10 AM–2 PM) | Realistic civic engagement |
| **Open Complaints Included** | 0% (filtered out) | 2.22% (222 open cases) | Reflects real-world pipeline |

---

## 6. Strict Feature Boundaries & Target Leakage Prevention

To ensure strict academic defense, fields are formally categorized into an approved intake allowlist and a banned post-resolution list:

### A. Approved Intake Feature Allowlist
Only features observable at the exact moment a complaint is logged may be used as model inputs:
1. `complaint_type`: Primary grievance category.
2. `descriptor`: Subcategory detail.
3. `borough`: Municipal borough (Brooklyn, Queens, Manhattan, Bronx, Staten Island).
4. `incident_zip`: Postal code.
5. `open_data_channel_type`: Submission channel (Online, Phone, Mobile, Unknown).
6. `created_hour`: Hour of intake in NYC local time (0–23).
7. `created_weekday`: Day of week (Monday–Sunday).
8. `created_month`: Calendar month (1–12).
9. `created_year`: Intake year (2023 or 2024).
10. `is_weekend`: Binary indicator (1 if Saturday/Sunday, else 0).
11. `latitude`, `longitude`: Geospatial location coordinates.

### B. Banned Post-Resolution Fields (Target Leakage)
These fields are populated downstream of intake and must **never** enter the feature matrix:
- `closed_date`: Post-resolution outcome timestamp.
- `status`: Outcome state (`Closed`).
- `resolution_description`: Agency post-action closing comments.
- `resolution_hours`: Derived target variable.
- `target_long_duration`: Binary classification target.

### C. Policy on `agency`
`agency` is retained in `data/processed/nyc_311_phase1_profiled.csv` for descriptive service demand dashboards and routing audits. However, it is **withheld from baseline predictive models** until it is empirically verified whether agency assignment happens instantaneously at intake or via asynchronous department triage.

---

## 7. Future Temporal Evaluation Plan

To guarantee out-of-time generalization and prevent temporal data leakage, Phase 2 modeling will follow a strict temporal train/validation/test split:

```
[------------- 2023 Training Set -------------] [--- 2024-H1 Val ---] [--- 2024-H2 Test ---]
  Jan 1, 2023                      Dec 31, 2023   Jan 1, 2024    Jun 30   Jul 1, 2024    Dec 31
  n = 4,827 records (48.3%)                      n = 2,473 (24.7%)       n = 2,700 (27.0%)
```

### Targets Defined:
1. **Regression Target (`resolution_hours`):**
   - Predicted elapsed hours between `created_date` and `closed_date` on valid completed tickets ($n = 9,777$).
   - Baselines: Mean baseline, Median baseline.
   - Metrics: MAE, RMSE, $R^2$.
2. **Classification Target (`target_long_duration`):**
   - Binary risk indicator defining long-duration complaints.
   - **Leakage Prevention Rule:** The classification threshold is computed **strictly from the 2023 training data**.
   - Training 75th percentile threshold = **125.001 hours (~5.2 days)**.
   - A complaint is labelled $1$ (high risk) if `resolution_hours >= 125.001`, and $0$ otherwise.
   - Metrics: Precision, Recall, F1-Score, PR-AUC, Confusion Matrix.
   - **Academic Guardrail:** This threshold is project-defined based on historical empirical distributions. It is explicitly not presented as an official legal SLA violation.

---

## 8. Automated Test Suite Results

The test suite [tests/test_phase1.py](file:///home/tejas/Downloads/CivicResolveAI_Starter/tests/test_phase1.py) was executed via `pytest`:

```bash
.venv/bin/pytest tests/test_phase1.py
```

**Results:**
- `test_monthly_quota_allocation_exact_sum`: PASSED (Hamilton-Hare sums to target exactly).
- `test_date_range_windows_24_months`: PASSED (24 contiguous non-overlapping months).
- `test_strata_offsets_reproducibility_and_bounds`: PASSED (Ranks bounded, reproducible under seed 42).
- `test_malformed_api_response_handling`: PASSED (Non-list API payloads safely raise errors).
- `test_duplicate_row_detection`: PASSED (Duplicate unique keys detected and dropped).
- `test_timestamp_handling_nyc_local_semantics`: PASSED (NYC local hour preserved, UTC converted for DST).
- `test_resolution_duration_and_invalid_pairs`: PASSED (Negative and zero durations correctly caught).
- `test_feature_boundary_leakage_prevention`: PASSED (Banned fields verified absent from allowlist).
- `test_api_retry_on_transient_failure`: PASSED (HTTP 429 backoff and retry succeeds).

**Summary: 9 passed in 0.51s (100% pass rate).**

---

## 9. Artifact Manifest

| Category | Artifact Path | Description |
|---|---|---|
| Ingestion Script | [scripts/03_build_temporal_sample.py](file:///home/tejas/Downloads/CivicResolveAI_Starter/scripts/03_build_temporal_sample.py) | Stratified temporal sampling script with Hamilton-Hare allocation |
| Profiling Script | [scripts/04_profile_temporal_sample.py](file:///home/tejas/Downloads/CivicResolveAI_Starter/scripts/04_profile_temporal_sample.py) | Profile calculation and visualization generator |
| Test Suite | [tests/test_phase1.py](file:///home/tejas/Downloads/CivicResolveAI_Starter/tests/test_phase1.py) | 9 automated unit and integrity tests |
| Raw Data | [data/raw/nyc_311_temporal_sample.csv](file:///home/tejas/Downloads/CivicResolveAI_Starter/data/raw/nyc_311_temporal_sample.csv) | 10,000 raw sampled records across 2023–2024 |
| Processed Data | [data/processed/nyc_311_phase1_profiled.csv](file:///home/tejas/Downloads/CivicResolveAI_Starter/data/processed/nyc_311_phase1_profiled.csv) | 10,000 profiled records with local calendar and duration features |
| Sampling Manifest | [outputs/sampling_manifest.json](file:///home/tejas/Downloads/CivicResolveAI_Starter/outputs/sampling_manifest.json) | Population counts, quotas, planned requests, execution log |
| Data Quality Report | [outputs/phase1_data_quality.json](file:///home/tejas/Downloads/CivicResolveAI_Starter/outputs/phase1_data_quality.json) | Complete statistical breakdown, missingness, and quantiles |
| Documentation | [README.md](file:///home/tejas/Downloads/CivicResolveAI_Starter/README.md) | Updated guide covering Phase 0 limits and Phase 1 workflow |
| Chart 1 | [outputs/figures/phase1_month_coverage.png](file:///home/tejas/Downloads/CivicResolveAI_Starter/outputs/figures/phase1_month_coverage.png) | Monthly sample counts across 24 months |
| Chart 2 | [outputs/figures/phase1_hour_distribution.png](file:///home/tejas/Downloads/CivicResolveAI_Starter/outputs/figures/phase1_hour_distribution.png) | Local hour-of-day distribution (NYC Eastern Time) |
| Chart 3 | [outputs/figures/phase1_weekday_distribution.png](file:///home/tejas/Downloads/CivicResolveAI_Starter/outputs/figures/phase1_weekday_distribution.png) | Day-of-week distribution (Monday–Sunday) |
| Chart 4 | [outputs/figures/phase1_top_categories.png](file:///home/tejas/Downloads/CivicResolveAI_Starter/outputs/figures/phase1_top_categories.png) | Top 15 complaint categories in stratified sample |
| Chart 5 | [outputs/figures/phase1_resolution_duration.png](file:///home/tejas/Downloads/CivicResolveAI_Starter/outputs/figures/phase1_resolution_duration.png) | Elapsed resolution duration histogram (clipped at 95th percentile) |

---

## 10. Remaining Limitations & Recommendations for Phase 2

1. **Rank-Stratified vs. Simple Random Sample:**
   Although rank-stratified sampling successfully disperses observations across all 24 months, individual requests within each chunk of 25 rows are consecutive in the API table. Sample proportions should not be quoted as exact population census parameters.
2. **Extreme Duration Outliers:**
   Max resolution time reached 30,168 hours (~3.4 years). For Phase 2 regression modeling, robust scalers, log transformations (`log1p(resolution_hours)`), or reasonable outlier handling must be explored.
3. **Open Records Handling in Supervised ML:**
   The 222 open tickets cannot participate in supervised regression training because their true resolution time is not yet known. They will remain in descriptive dashboards and can be scored as real-time inference candidates.
4. **Local Indian Municipal Context:**
   The methodology is built to demonstrate explainable complaint analytics on public open data. Deploying this system to an Indian municipality (such as PCMC) requires local schema mapping, agency hierarchy adaptation, and authorized municipal data.

### Milestone Status: **PASS**
Phase 1 is complete. The dataset is scientifically verified and ready for Phase 2 (Preprocessing Pipelines, Baseline Regression, and Risk Classification).
