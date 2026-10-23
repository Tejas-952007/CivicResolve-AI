# CivicResolve AI — Explainable Municipal Complaint Intelligence & Resource-Allocation Simulator

[![Tests](https://img.shields.io/badge/pytest-45%20passed-brightgreen.svg)](#7-automated-test-suite)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Academic%20Evaluation-lightgrey.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/web-Flask%20%2B%20ReportLab-orange.svg)](https://flask.palletsprojects.com/)

**Academic Mini-Project (B.E. Artificial Intelligence & Data Science)**  
CivicResolve AI is an end-to-end municipal operational intelligence and decision-support system developed on public NYC 311 service-request data (canonical N = 10,000 rank-stratified records spanning 2023–2024). It combines **leakage-free machine learning** (historical resolution-time estimation and long-duration risk classification), **unsupervised pattern discovery** (K-Means demand archetypes across 274 municipality-category segments), **automated administrative reporting** (dynamic ReportLab PDF generation), and a **reinforcement learning resource-allocation simulator** (tabular Q-learning vs. multi-policy queuing benchmarks).

- **Official NYC Open Data 311 Endpoint:** <https://data.cityofnewyork.us/resource/erm2-nwe9.json>  
- **Dataset Documentation:** <https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9/about_data>
- **Final Academic Report (PDF):** [`report/CivicResolve_AI_Final_Report.pdf`](report/CivicResolve_AI_Final_Report.pdf) (Markdown source: [`report/CivicResolve_AI_Final_Report.md`](report/CivicResolve_AI_Final_Report.md))
- **Examiner Demo Guide:** [`report/DEMO_GUIDE.md`](report/DEMO_GUIDE.md)

---

## 1. Quickstart & Local Web Application Run

### Prerequisites
- Linux / macOS / Windows with WSL
- Python 3.10, 3.11, or 3.12
- Google Chrome or Chromium (optional, used for headless report rendering)

### Execution Steps
```bash
# 1. Activate the pre-configured virtual environment
source .venv/bin/activate

# 2. Run the complete automated test suite (45 tests across Phases 1, 2, 3, and 4)
pytest tests/ -v

# 3. Launch the local Flask application
python app.py
```

The web application runs locally at: **<http://127.0.0.1:5000>**

---

## 2. Application Modules & Complete Route Map

| Module | Route(s) | Method | Functional Scope & Governance Disclaimers |
| :--- | :--- | :--- | :--- |
| **System Health** | `/health` | `GET` | Verifies loaded pipeline artifacts, data integrity, and service status. |
| **Overview Dashboard** | `/dashboard` | `GET` | Dynamic KPI cards, monthly trends, borough delay rates, and distribution histograms with live slice filtering. |
| **Single Complaint Analysis** | `/predict` | `GET`, `POST` | Dual model inference: Expected duration (hours) and Delay Risk % at intake time. Strict leakage-free validation. |
| **Batch CSV Scoring** | `/batch`<br>`/batch/sample`<br>`/batch/export` | `GET`, `POST`<br>`GET`<br>`POST` | Bulk scoring with automatic quarantine of post-resolution fields (`closed_date`, `resolution_description`). Downloadable scored CSV. |
| **Model Evaluation** | `/evaluation` | `GET` | Transparent held-out test benchmarks (2024-H2), confusion matrix, PR/ROC curves, and non-causal feature rankings. |
| **Pattern Discovery** | `/clustering`<br>`/clustering/export` | `GET`<br>`GET` | Unsupervised K-Means clustering ($k=5$, Silhouette: 0.3836) with 2D PCA visual scatter plot and operational archetype segmentation. |
| **Resource Planning Simulator** | `/resource-planning`<br>`/resource-planning/run`<br>`/resource-planning/export-csv` | `GET`<br>`POST`<br>`GET` | 5-policy capacity planning simulation benchmark (Q-Learning, FIFO, Risk-Plus-Wait, Risk-First, Round-Robin). |
| **Diagnostic PDF Export** | `/report/download` | `GET` | Generates a dynamic multi-page ReportLab PDF report containing KPI tables, ML benchmarks, and simulation appendix. |
| **Filtered Data Export** | `/export/filtered-csv` | `GET` | Exports dynamically filtered dashboard complaints as a CSV file. |
| **Diagnostic Figures** | `/figures/<filename>` | `GET` | Serves statically generated evaluation and simulation diagnostic plots. |

---

## 3. Project Architecture & Leakage Prevention

```
                                  [ NYC 311 Public API ]
                                             │
                                             ▼
                             [ Data Ingestion & Profiling ]
                             (10,000 rank-stratified records)
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
          [ Temporal Partitioning ]                     [ Unsupervised Clustering ]
       Train: 2023 (N=4,827)                         K-Means (k=5 Archetypes)
       Val:   2024-H1 (N=2,473)                      PCA 2D Spatial Projections
       Test:  2024-H2 (N=2,700)                                     │
                      │                                             │
                      ▼                                             │
          [ Intake Feature Allowlist ]                              │
      (Agency, Complaint Type, Descriptor,                          │
       Borough, Incident Zip, Time/Coords)                          │
                      │                                             │
           ┌──────────┴──────────┐                                  │
           ▼                     ▼                                  │
     [ Regressor ]        [ Classifier ]                            │
    RF (Log1p Target)     LogisticRegression                        │
    (Expected Hours)      (Delay Risk %)                            │
           │                     │                                  │
           └──────────┬──────────┘                                  │
                      ▼                                             │
       [ Q-Learning Capacity Simulator ] ◄──────────────────────────┘
       (Discrete State MDP, Multi-Policy)
                      │
                      ▼
            [ Flask Web Application ]
         (Interactive UI + ReportLab PDF)
```

### Strict Leakage-Free Intake Feature Allowlist
Only features verified to be available at ticket intake are permitted into predictive pipelines:
- **Categorical:** `complaint_type`, `descriptor`, `borough`, `incident_zip`, `open_data_channel_type`
- **Temporal:** `created_hour`, `created_weekday`, `created_month`, `is_weekend`
- **Geospatial:** `latitude`, `longitude`

### Strictly Excluded Post-Resolution Columns
`closed_date`, `status`, `resolution_description`, `resolution_hours`, `target_long_duration`, `agency`, `created_year`.

### Disjoint Temporal Partitioning
- **Training Set (2023):** N = 4,827 total rows (4,711 eligible closed non-negative records).
- **Validation Set (2024-H1):** N = 2,473 total rows (2,437 eligible closed non-negative records).
- **Final Test Set (2024-H2):** N = 2,700 total rows (2,635 eligible closed non-negative records).
- **Total Canonical Sample:** Exactly 10,000 rows (9,783 eligible supervised records, 217 exclusions).
- **Empirical Delay Threshold:** Derived strictly from 2023 training data at the 75th percentile: **124.925 hours (~5.2 days)**.

---

## 4. Empirical ML Benchmarks (Held-Out Final Test: 2024-H2)

Evaluated exactly once on frozen models without post-test tuning:

### Classification: Logistic Regression (Class-Balanced)
- **F1-Score:** 0.7093
- **Recall:** 88.7% (Identified 488 of 550 delayed complaints in test set)
- **Precision:** 59.1% (Generates 338 false-positive alerts; designed as a screening filter)
- **PR-AUC:** 0.7227 (vs. 0.2087 no-skill baseline)
- **ROC-AUC:** 0.9262
- **Confusion Matrix:** `[[TN=1747, FP=338], [FN=62, TP=488]]`

### Regression: Random Forest Regressor (Log1p Target)
- **Test Median Absolute Error:** 15.59 hours (~0.65 days)
- **Test MAE:** 251.99 hours (~10.5 days)
- **Test RMSE:** 947.67 hours
- **Test R²:** 0.1446
- *Note:* Extreme municipal tail events (tickets requiring thousands of hours for multi-month capital repairs or structural litigation) inflate MAE/RMSE relative to median absolute error.

---

## 5. Reinforcement Learning Capacity Simulator & Scientific Findings

The simulator models a resource-constrained municipal dispatch intake over discrete planning periods (Horizon T = 10 periods, Capacity C = 5 items/period, Arrival Batch B = 6 items/period) across 5 evaluation seeds (99–103) on the 2024-H1 validation cohort.

### Canonical Multi-Policy Benchmark Comparison

| Policy | Mean Reward ± Std | Slot Util % | Handled Items | Mean Wait (periods) | Max Starvation (periods) | True Long Capture % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Risk-Plus-Wait** | **39.93 ± 4.61** | 100.0% | 50.0 | 0.81 | 2.4 | 86.0% |
| **FIFO (Baseline)** | 38.31 ± 4.08 | 100.0% | 50.0 | 0.82 | 1.8 | 83.4% |
| **Risk-First** | 38.31 ± 4.82 | 100.0% | 50.0 | **0.47** | 6.6 | **98.8%** |
| **Q-Learning** | 37.33 ± 2.69 | 100.0% | 50.0 | 0.78 | 2.4 | 82.2% |
| **Round-Robin** | 31.55 ± 3.85 | 100.0% | 50.0 | 0.67 | **1.0** | 72.6% |

### Critical Academic Honesty Notice on Q-Learning
> **Q-learning is NOT the top-performing policy on mean reward.**  
> In canonical multi-seed evaluation, the heuristic `Risk-Plus-Wait` policy achieves the highest mean reward (39.93 ± 4.61), followed by `FIFO` (38.31 ± 4.08) and `Q-Learning` (37.33 ± 2.69).  
> - **Reason:** Tabular Q-learning discretizes the continuous queue state into coarse bins (1,764 combinations), introducing quantization loss compared to fine-grained continuous priority sorting.
> - **Contribution:** The Q-learning agent learns an action policy under a human-designed composite reward function with manually configured weights. It achieves the lowest variance across evaluation seeds (2.69 std), avoids severe starvation (2.4 periods vs 6.6 periods for pure Risk-First), and outperforms static Round-Robin (31.55 ± 3.85).

### Precise Definition of "Retrospective True Long-Duration Capture"
> **Capture Rate Formula:**  
> `Capture Rate = (Handled Delayed Tickets / Total Delayed Arrival Cohort Tickets) * 100%`  
> *(where Delayed Tickets are those with historical resolution duration > 124.925 hours)*

- Evaluated strictly over closed historical tickets with ground-truth durations.
- Measures the fraction of true prolonged cases serviced during the simulation horizon.
- **It is a retrospective diagnostic metric, not an intervention success rate or causal proof of delay reduction.**

---

## 6. Audit of Municipal Backlog & Filtering Definitions

To eliminate confusion between different counts across the application:
- **Canonical Sample:** Exactly 10,000 records sampled across 2023–2024.
- **Recorded Categorical Status 'Closed':** Exactly 9,846 records have recorded status `Closed` (154 records have active non-Closed statuses).
- **Closed with Valid Timestamp:** Exactly 9,778 records have recorded status `Closed` and a non-null closing timestamp (68 records marked `Closed` have missing closing timestamps).
- **Service-Eligible Closed Records:** Exactly 9,777 records (closed with valid timestamp and non-negative duration, with 1 record having a negative duration logging error). Sample median resolution duration is 23.9 hours, and 2,444 records exceed 124.925 hours (exactly 25.0% of the eligible closed records).
- **Active Unclosed Backlog Metric (N = 222):** The administrative dashboard backlog KPI combines the 154 active non-Closed status records with the 68 records marked `Closed` that lack completion timestamps (154 + 68 = 222).
- **Missing `closed_date` Total:** Exactly 196 records lack a closing timestamp (128 non-closed + 68 closed).
- **Supervised Modeling Baseline (N = 9,783):** Exactly 9,783 records possess non-null closing timestamps and non-negative durations (9,777 closed + 6 non-closed with valid timestamps).
- **Supervised Modeling Exclusions (N = 217):** Exactly 196 null closing dates plus 21 negative-duration errors (10,000 - 217 = 9,783).
- **Sample Notice:** Historical counts from the 10,000 sample must never be represented as live municipal backlogs for New York City or Indian municipalities.

---

## 7. Automated Test Suite

To run all 45 automated tests across all project phases:

```bash
pytest tests/ -v
```

### Test Suite Breakdown
- **`tests/test_phase1.py` (9 Tests):** Ingestion retry resilience, 24-month distribution, duplicate prevention, boundary leakage, monthly quotas, negative duration exclusion, strata offset reproducibility, and NYC timestamp parsing.
- **`tests/test_phase2.py` (10 Tests):** Disjoint temporal partitions, training bounds (≤ 2023), non-negative regression targets, open ticket exclusion, training-only threshold derivation, leakage prevention, adversarial missing category handling, pipeline reproducibility, and finite predictions.
- **`tests/test_phase3.py` (12 Tests):** Flask application startup, health check, dashboard route rendering, dynamic filter updates, single complaint scoring, missing input rejection, batch CSV validation, quarantine of prohibited fields, K-Means clustering execution, and ReportLab PDF `%PDF-` verification.
- **`tests/test_phase4.py` (14 Tests):** Q-learning MDP state discretization, action masking, capacity constraint enforcement, reward calculation, policy convergence, 5-policy benchmarking, simulation export integrity, and UI route execution.

---

## 8. Defensible Academic Disclaimers

1. **Statistical Dependence, Not Causation:** Feature importance scores measure statistical association within tabular models. High importance for `complaint_type` reflects departmental workflows and triage practices, not physical causality.
2. **Project-Defined Baseline vs. Municipal SLA:** The 124.925-hour threshold is an empirical 75th percentile derived from 2023 training data. It is an internal benchmark, not an official municipal SLA violation.
3. **Geographic & Administrative Specificity:** Models were trained strictly on NYC 311 data. They are not calibrated, validated, or representative for Indian municipal corporations (e.g., PCMC/PMC) without local data ingestion and workflow re-engineering.
4. **Simulation Scope:** All capacity planning and dispatch outcomes are simulated experiments; they do not guarantee real-world resolution time reductions without field trials.

---

## 9. Academic Project Deliverables

- **Final Academic Report (PDF):** [`report/CivicResolve_AI_Final_Report.pdf`](report/CivicResolve_AI_Final_Report.pdf) (17 pages, publication-grade formatting with embedded screenshot evidence).
- **Final Academic Report (Markdown):** [`report/CivicResolve_AI_Final_Report.md`](report/CivicResolve_AI_Final_Report.md).
- **Examiner Demo Guide:** [`report/DEMO_GUIDE.md`](report/DEMO_GUIDE.md) (Step-by-step viva presentation flow and defense Q&A).
- **Verified UI Screenshots:** Located in `report/screenshots/` (`01_dashboard.png` through `07_pdf_report-3.png`).
- **Phase 5 Handoff Audit:** [`outputs/phase5_handoff.md`](outputs/phase5_handoff.md).
