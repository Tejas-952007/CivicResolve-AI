# CivicResolve AI — Phase 3 Handoff Report: Working Flask Application, Analytics Dashboard, Clustering, and Automated PDF Reports

**Author:** Senior Python Full-Stack & Machine Learning Partner  
**Status:** **PASS** (Ready for Phase 4 Q-Learning Resource Allocation Simulation)  
**Date:** October 2026  
**Repository:** `CivicResolveAI_Starter`  

---

## 1. Executive Summary

Phase 3 successfully translated the leakage-free Phase 2 machine learning pipelines and the Phase 1 temporal dataset into a fully functional, production-styled local web application. The application does not use mockups or fabricated telemetry; all charts, predictions, clusters, and reports are computed dynamically from actual saved models and the 10,000-record NYC 311 dataset.

### Core Deliverables Completed:
1. **Flask Application Architecture (`app.py`):** Structured around a modular service layer (`src/services/`) with preloaded frozen models, Jinja filters, error resilience, and RESTful export endpoints.
2. **Module A — Interactive Overview Dashboard (`/dashboard`):** Real-time KPI cards, dual-axis monthly volume & resolution trends, horizontal bar chart of top 10 categories, borough demand vs. delay rate comparison, and duration distribution histogram rendered via Plotly.js. Responds dynamically to Borough, Category, and Date range filters.
3. **Module B — Individual Complaint Analysis (`/predict`):** Verified intake-time input form with 4 interactive one-click presets. Computes expected resolution duration in hours and human-readable units (e.g. "3 days, 9.9 hours") using Random Forest (Log1p target), and calculates long-duration risk probability with operational review badges using Logistic Regression (Balanced).
4. **Module C — Batch CSV Scoring (`/batch`):** High-throughput complaint scoring (up to 5,000 rows). Automatically detects, quarantines, and excludes outcome columns (`closed_date`, `status`, `resolution_description`, `resolution_hours`, etc.) to prevent data leakage. Ranks complaints by delay risk and provides downloadable scored CSVs. Includes downloadable sample template (`/batch/sample`).
5. **Module D — Model Evaluation Centre (`/evaluation`):** Displays side-by-side validation (2024-H1) and single-evaluation final test (2024-H2) benchmarks, test confusion matrix, precision-recall and ROC curves, actual vs. predicted scatter, residual error distribution, and permutation feature importance charts. Includes an educational metric interpretation guide.
6. **Module E — K-Means Complaint Pattern Discovery (`/clustering`):** Unsupervised clustering across 274 municipal segments (Borough $\times$ Category) evaluated across $k=2..5$. Optimal $k=5$ (silhouette score **$0.3836$**) isolates 5 clear operational archetypes (Rapid Dispatch, Routine Maintenance, Multi-Month Forestry/Capital, High Backlog, and Complex Code Enforcement) with 2D PCA projection ($55.7\%$ variance explained) and CSV export.
7. **Module F — Automated Professional PDF Reporting (`/report/download`):** Built with ReportLab. Generates publication-grade, audit-ready PDF reports with running headers, "Page X of Y" canvas footers, executive summaries, KPI tables, top complaint profiles, model test metrics, cluster summaries, and ethical disclaimers.
8. **Automated Verification:** 31 out of 31 unit and integration tests passed ($9$ Phase 1, $10$ Phase 2, and $12$ Phase 3 tests) in 17.66 seconds.
9. **Real Browser Inspection:** Verified via Chrome browser subagent across all five views with zero JavaScript console errors or layout breaks.

---

## 2. Web Application Modules & Endpoints

| Endpoint | Method | Template / Service | Functional Scope |
| :--- | :---: | :--- | :--- |
| `/` | GET | Redirect to `/dashboard` | Application entry point. |
| `/health` | GET | JSON Response | Service health check and model loading status. |
| `/dashboard` | GET | `templates/dashboard.html` | Overview KPIs, Plotly trends, borough/category filters, CSV export. |
| `/predict` | GET, POST | `templates/single_prediction.html` | Intake-time complaint scoring, presets, uncertainty disclaimers. |
| `/batch` | GET, POST | `templates/batch_prediction.html` | CSV upload, automated leakage quarantine, risk-ranked triage queue. |
| `/batch/sample` | GET | File Download | Download valid CSV intake template for immediate testing. |
| `/batch/export` | POST | File Download | Download full scored batch results as CSV. |
| `/evaluation` | GET | `templates/evaluation.html` | Validation vs test metrics, diagnostic figures, interpretation guide. |
| `/clustering` | GET | `templates/clustering.html` | K-Means pattern discovery, silhouette table, 2D PCA plot, profiles. |
| `/clustering/export` | GET | File Download | Download 274 municipal segments with assigned cluster labels as CSV. |
| `/report/download` | GET | `src/services/report_service.py` | Generate and download publication-grade ReportLab PDF report. |
| `/export/filtered-csv` | GET | File Download | Download filtered dataset slice as CSV. |
| `/figures/<filename>` | GET | `outputs/figures/` | Static server for diagnostic evaluation plots. |

---

## 3. Empirical Model Results Presented in UI

To adhere strictly to research integrity, all metrics displayed in the application are derived directly from the Phase 2 held-out evaluation on 2024-H2 ($N=2,635$):

### Classification (Logistic Regression Balanced):
- **Test Precision:** $0.5908$
- **Test Recall:** $0.8873$ ($488$ of $550$ delayed requests caught)
- **Test F1-Score:** $0.7093$
- **Test PR-AUC:** $0.7227$ (vs. $0.209$ majority baseline)
- **Test ROC-AUC:** $0.9262$
- **Test Confusion Matrix:** $\text{TN}=1747, \text{FP}=338, \text{FN}=62, \text{TP}=488$
- **Operational Presentation:** Clearly presented as an early screening and triage filter rather than an automated decision maker. The 338 false positives are transparently disclosed.

### Regression (Random Forest Log1p Target):
- **Test Median Absolute Error:** $15.587\text{ hours}$ (~$0.65\text{ days}$)
- **Test MAE:** $251.986\text{ hours}$ (~$10.5\text{ days}$)
- **Test RMSE:** $947.665\text{ hours}$
- **Test $R^2$:** $0.1446$
- **Operational Presentation:** The substantial divergence between Median Absolute Error ($15.6\text{h}$) and Mean Absolute Error ($252\text{h}$) is prominently explained in the UI as the mathematical consequence of heavy right-tail outliers (extreme multi-month repair projects).

---

## 4. K-Means Pattern Discovery Findings ($k=5$)

Unsupervised clustering on 274 municipal segments (Borough $\times$ Category) evaluated across $k=2..5$:

| Cluster ID | Archetype Profile Label | Segments ($N$) | Avg Volume | Avg Median Duration | Delay Rate (&gt;124.9h) | Open Backlog % | Representative Categories |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **#0** | High-Volume Rapid Municipal Dispatch | 14 (5.1%) | 250.6 | 9.4h (~0.4d) | 0.2% | 0.1% | Illegal Parking, Residential Noise, Heat/Water |
| **#1** | Moderate-Volume Routine Maintenance | 129 (47.1%) | 24.8 | 22.2h (~0.9d) | 7.5% | 0.3% | Abandoned Vehicle, Dirty Condition, Commercial Noise |
| **#2** | Multi-Month Capital & Forestry Programs | 8 (2.9%) | 16.2 | 8,903.9h (~371d) | 100.0% | 5.5% | New Tree Request, Food Establishment Licensing |
| **#3** | Backlog-Prone / Specialized Operations | 16 (5.8%) | 20.6 | 517.3h (~21.6d) | 53.0% | 26.9% | Damaged Tree, Dead/Dying Tree, Helicopter Noise |
| **#4** | Complex Residential Code Enforcement | 107 (39.1%) | 22.0 | 495.8h (~20.7d) | 62.0% | 0.5% | Unsanitary Condition, Appliance, Door/Window repairs |

- **Silhouette Evidence:** $k=2$ (0.3119), $k=3$ (0.3232), $k=4$ (0.3392), **$k=5$ (0.3836 — Optimal)**.
- **2D PCA Projection:** Captures **$55.7\%$** of total operational variance (PC1: 36.3%, PC2: 19.4%).

---

## 5. Automated Verification & Testing

The complete test suite was executed via `pytest tests/ -v`:

```bash
============================== 31 passed in 17.66s ==============================
```

- **Phase 1 Tests (9/9):** Data ingestion, temporal windows, duplicate row detection, boundary leakage, quota allocation, negative duration exclusion, strata offset reproducibility, and local NYC timestamp parsing.
- **Phase 2 Tests (10/10):** Disjoint temporal splits, training date boundaries ($\le 2023$), non-negative regression targets, open case exclusion, 75th percentile threshold derivation, feature allowlist leakage barriers, adversarial missing/unseen category handling, model reproducibility, and finite predictions.
- **Phase 3 Tests (12/12):**
  1. `test_flask_startup_and_health_route`: Health check returns HTTP 200 with model loading confirmation.
  2. `test_dashboard_renders_without_exceptions`: Overview dashboard loads all widgets and charts.
  3. `test_dashboard_filters_change_displayed_aggregates`: Borough/category filters change rendered numbers.
  4. `test_valid_individual_prediction`: Valid single complaint inputs yield valid predictions and labels.
  5. `test_missing_and_invalid_input_handling`: Missing required fields and out-of-bounds hours are rejected.
  6. `test_unknown_categorical_values_handled_safely`: Novel categories/ZIPs do not cause crashes.
  7. `test_batch_csv_success_and_schema_validation`: Valid CSVs score properly; invalid schemas are rejected.
  8. `test_prohibited_target_columns_quarantined_in_batch`: Leakage columns are detected and quarantined.
  9. `test_prediction_outputs_finite_and_reproducible`: Repeated predictions yield identical, finite outputs.
  10. `test_kmeans_clustering_output_and_profiles`: Clustering executes and assigns cluster profiles.
  11. `test_pdf_report_generation_and_validation`: ReportLab builds valid `%PDF-` documents with tables.
  12. `test_empty_report_filters_handled_gracefully`: Empty filter selections do not cause PDF build crashes.

---

## 6. Live Browser Verification Summary

Using the Chrome browser subagent against `http://127.0.0.1:5000`:
- **Dashboard (`/dashboard`):** Rendered 10,000 total volume, 222 open backlog, 8.7h overall median duration, 22.8% long-delay rate, 179 zero-duration tickets. All 4 Plotly charts rendered without clipping.
- **Individual Analysis (`/predict`):** Loaded "Residential Noise (Brooklyn)" preset; computed $0.68\text{ hours}$ expected duration and $0.3\%$ long-duration risk probability with green "LOW RISK" tier.
- **Batch CSV Scoring (`/batch`):** Upload interface verified; sample CSV downloaded; batch scoring executed and rendered risk-ranked queue.
- **Model Evaluation (`/evaluation`):** Rendered validation comparison tables, held-out test cards, confusion matrix, PR/ROC curves, actual vs. predicted plot, and permutation importance.
- **Pattern Discovery (`/clustering`):** Rendered interactive PCA scatter plot, silhouette evidence table, and cluster archetype breakdown.

---

## 7. Artifacts Created & Modified in Phase 3

### Core Application & Services:
- `app.py`: Main Flask application with routes, error handling, preloaded models, and export handlers.
- `src/services/__init__.py`: Package initialization.
- `src/services/analytics_service.py`: Dataset caching, filtering, KPIs, monthly trends, distributions.
- `src/services/prediction_service.py`: Single intake forecasting, batch CSV scoring, leakage quarantine.
- `src/services/clustering_service.py`: K-Means pattern discovery, silhouette scoring, PCA projection, profiles.
- `src/services/report_service.py`: ReportLab automated PDF generation with custom two-pass NumberedCanvas.

### Templates & Styling:
- `templates/base.html`: Responsive base layout with civic navbar, disclaimers, and footer.
- `templates/dashboard.html`: Overview dashboard with filter controls, KPI cards, and Plotly charts.
- `templates/single_prediction.html`: Intake scoring form with presets, duration labels, and risk cards.
- `templates/batch_prediction.html`: Batch CSV upload, leakage notification, and ranked triage queue.
- `templates/evaluation.html`: Benchmark comparison tables, diagnostic plots, and interpretation cards.
- `templates/clustering.html`: K-Means silhouette comparison, PCA scatter plot, and archetype profiles.
- `static/css/style.css`: Professional civic analytics stylesheet (slate, navy, teal, accessible contrast).

### Verification & Documentation:
- `tests/test_phase3.py`: 12 automated unit and integration tests for Phase 3.
- `README.md`: Updated with full setup, local run commands, module guide, and test instructions.
- `outputs/phase3_handoff.md`: This comprehensive engineering handoff document.

---

## 8. Known Limitations & Governance Notes

1. **Academic Sample Nature:** The 10,000-record dataset is a rank-stratified quota sample with contiguous offsets per stratum, not an exhaustive municipal census.
2. **Right-Tail Skew:** Regressors face an extreme right-tail skew (tickets spanning up to 30,000h). Median absolute error is ~15.6 hours, whereas mean absolute error is ~252 hours.
3. **Screening Tool Context:** The classifier captures 88.7% of delayed tickets with 59.1% precision (generating 338 false alarms). It is designed to assist human triage review, not replace municipal discretion.
4. **Non-Causality:** Permutation importance reflects departmental workflows and routing correlations, not physical root causes of delay.
5. **Geographic Specificity:** Models are calibrated strictly on NYC 311 data and must not be treated as representative for Indian municipal corporations (such as PCMC) without local retraining.

---

## 9. Recommended Next Steps for Phase 4

With the local web application, predictive scoring, clustering, and PDF reporting fully operational and verified, Phase 4 should focus on:
1. **Reinforcement Learning / Q-Learning Simulation:** Implement an agent-based municipal depot dispatch simulator that uses predicted complaint delay risk to optimize municipal crew allocation and work-order prioritization under constrained resource budgets.
2. **Comparative Backlog Triage:** Simulate FIFO (first-in-first-out) vs. Risk-Weighted Dispatch to measure simulated backlog clearance velocity.
