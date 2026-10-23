# CivicResolve AI — Examiner & Viva Demo Guide (5–7 Minutes)

This guide provides a structured, step-by-step demonstration sequence for academic examiners, faculty evaluators, and project viva panels. It outlines the exact click path, timing, and theoretical justification for each module.

---

## Quick Reference Summary

| Phase | Duration | Interface / Route | Key Features Demonstrated | What It Proves to Evaluators |
| :--- | :--- | :--- | :--- | :--- |
| **1. Intro** | 0:00–0:45 | Architecture & Health (`/health`) | Server status, artifact integrity, dataset specs | System readiness and dependency resolution |
| **2. Overview** | 0:45–1:45 | Dashboard (`/dashboard`) | Volume KPIs (10,000 total, 222 unclosed backlog), borough filters | Dynamic aggregation and defensible metric definitions |
| **3. Inference** | 1:45–2:45 | Single Intake (`/predict`) | Intake form, dual model inference, risk tier | Zero-leakage inference at intake time |
| **4. Batch** | 2:45–3:30 | Batch Scoring (`/batch`) | CSV upload, schema validation, quarantine banner, scored queue | Operational throughput and leakage prevention |
| **5. ML Audit** | 3:30–4:30 | Model Evaluation (`/evaluation`) | Test metrics (MAE: 251.99h, F1: 0.7093), confusion matrix, feature ranking | Transparent validation on held-out test data |
| **6. Discovery**| 4:30–5:15 | Clustering (`/clustering`) | K-Means archetypes (k = 5, Silhouette: 0.3836), 2D PCA projection | Unsupervised segmentation without label leakage |
| **7. RL Sim** | 5:15–6:15 | Resource Planning (`/resource-planning`)| 5-policy benchmark, Q-learning (37.33 ± 2.69), capacity curves | Scientifically honest simulation comparison |
| **8. Export** | 6:15–7:00 | Diagnostic PDF (`/report/download`) | Multi-page ReportLab PDF report generation | Enterprise-grade administrative reporting |

---

## Detailed Step-by-Step Demonstration

### 1. Introduction & Health Verification (0:00 – 0:45)
- **Action:** Open terminal and verify server is running on `http://127.0.0.1:5000`. Navigate to `http://127.0.0.1:5000/health` in the browser.
- **Visual:** JSON response with `"status": "healthy"`, model versions, and loaded dataset count (10,000 canonical sample records).
- **Viva Talking Point:**
  > *"CivicResolve AI is an end-to-end municipal intelligence system built on public NYC 311 data (10,000 rank-stratified sampled records spanning 2023–2024). Before serving requests, the `/health` endpoint independently verifies that the scikit-learn regression pipeline, classification pipeline, and pre-trained Q-table are fully loaded in memory."*

---

### 2. Municipal Overview Dashboard (0:45 – 1:45)
- **Action:** Navigate to `http://127.0.0.1:5000/dashboard`.
- **Interactions:**
  1. Point out the top summary KPI cards: Total Complaints (10,000), Unclosed Backlog (222), Median Duration (8.7 hours), and Empirical Delay Threshold (124.93 hours).
  2. Filter by Borough (e.g., select `BROOKLYN`) and click **Apply Filters**.
  3. Observe that all charts, volume cards, and average duration figures dynamically update.
- **Viva Talking Point:**
  > *"Notice the metric definitions: 222 complaints represent the active unclosed backlog (comprising 154 non-Closed statuses and 68 Closed records missing closing timestamps). Notice also the 124.93-hour threshold—this is strictly the empirical 75th percentile of 2023 training data, not an arbitrary target. The dashboard computes live slice aggregations without database round-trips."*

---

### 3. Intake-Time Complaint Forecasting (1:45 – 2:45)
- **Action:** Navigate to `http://127.0.0.1:5000/predict`.
- **Interactions:**
  1. Fill the form with typical intake features or click a preset (e.g., **Street Pothole (Bronx)**):
     - **Complaint Category:** `Street Condition`
     - **Borough:** `BRONX`
     - **Descriptor:** `Pothole`
     - **Incident Zip:** `10458`
     - **Creation Hour:** 11:00
  2. Click **Compute Resolution Forecast & Risk**.
  3. Review the dual output:
     - **Expected Resolution Duration:** e.g., 81.89 hours (via Random Forest Regressor, Log1p target).
     - **Delay Risk Score:** e.g., 20.4% (Low Long-Duration Risk badge via Class-Balanced Logistic Regression).
- **Viva Talking Point:**
  > *"Crucially, notice what is NOT in this form. There is no resolution description, no closing date, and no agency action log. The model operates strictly at complaint intake time, eliminating data leakage. Regression and classification pipelines run in parallel: regression predicts expected hours, while classification estimates the probability of exceeding the 124.93-hour empirical delay mark."*

---

### 4. Batch CSV Scoring & Schema Quarantine (2:45 – 3:30)
- **Action:** Navigate to `http://127.0.0.1:5000/batch`.
- **Interactions:**
  1. Click **Download Sample CSV Template** (`/batch/sample`) to show the accepted schema.
  2. Upload a sample batch CSV file and submit.
  3. Point out the scored priority queue with assigned Risk Tiers (**High**, **Moderate**, **Low**) and expected duration columns.
  4. Point out the informational alert banner confirming that post-intake fields were quarantined.
- **Viva Talking Point:**
  > *"In municipal administration, batch intake occurs via overnight dumps or API synchronization. Our batch ingestion module validates CSV headers against a strict schema. If an incoming file contains post-resolution fields such as `closed_date` or `resolution_description`, the preprocessor actively quarantines them, preventing target leakage."*

---

### 5. Scientifically Honest Model Evaluation (3:30 – 4:30)
- **Action:** Navigate to `http://127.0.0.1:5000/evaluation`.
- **Interactions:**
  1. Scroll through the Preferred Regressor Performance Card (Test MAE: 251.99 hours, Median Absolute Error: 15.59 hours, Test R² = 0.1446).
  2. Scroll through the Preferred Classifier Performance Card (Test Recall = 88.7%, Precision = 59.1%, F1-Score = 0.7093, ROC-AUC = 0.9262).
  3. Point out the Confusion Matrix (1,747 TN, 338 FP, 62 FN, 488 TP) and Top Feature Importances bar chart.
- **Viva Talking Point:**
  > *"We practice strict academic honesty. The R² of 0.1446 reflects the real-world complexity of municipal workflows—human field decisions, physical weather emergencies, and tenant access constraints introduce heavy right-tailed variance that tabular intake features cannot fully capture. Furthermore, our feature importance chart is explicitly labelled as 'Statistical Association, Not Physical Causation' to avoid false operational claims."*

---

### 6. Unsupervised Operational Clustering (4:30 – 5:15)
- **Action:** Navigate to `http://127.0.0.1:5000/clustering`.
- **Interactions:**
  1. View the 2D PCA cluster projection scatter plot (k = 5, 55.7% variance explained).
  2. Walk through the five operational archetypes table:
     - **Cluster 0:** High-Volume Rapid Municipal Dispatch (Avg 250.6 requests, median 9.4h, delay rate 0.2%).
     - **Cluster 1:** Moderate-Volume Routine Maintenance (Avg 24.8 requests, median 22.2h, delay rate 7.5%).
     - **Cluster 2:** Multi-Month Capital & Forestry Programs (Avg 16.2 requests, median 8,903.9h, delay rate 100.0%).
     - **Cluster 3:** Backlog-Prone / Specialized Operations (Avg 20.6 requests, median 517.3h, open backlog share 26.9%).
     - **Cluster 4:** Complex Residential Code Enforcement (Avg 22.0 requests, median 495.8h, delay rate 62.0%).
  3. Click **Download Cluster Segments CSV**.
- **Viva Talking Point:**
  > *"Unsupervised K-Means clustering segments 274 municipality-category segments into five distinct operational archetypes using the maximum silhouette coefficient criterion (k = 5, silhouette 0.3836 vs. k = 2..4: 0.31–0.34). While the algorithm itself is unsupervised, the input features contain historical outcome-derived aggregates (median resolution hours, long-delay share, open-case share). We explicitly label these as retrospective descriptive profiles of administrative performance, not intake-time predictions or causal findings."*

---

### 7. Reinforcement Learning Resource Planning (5:15 – 6:15)
- **Action:** Navigate to `http://127.0.0.1:5000/resource-planning`.
- **Interactions:**
  1. Inspect the simulator parameters (Horizon Periods T = 10, Capacity C = 5 items/period, Arrival Batch B = 6 items/period, 15 RL episodes, 5 evaluation seeds: 99–103).
  2. Point out the 5-policy benchmark comparison table and bar chart:
     - **Risk-Plus-Wait:** Mean Reward = 39.93 ± 4.61, Utilization = 100.0%, True Long Capture = 86.0%.
     - **FIFO:** Mean Reward = 38.31 ± 4.08, Utilization = 100.0%, True Long Capture = 83.4%.
     - **Risk-First:** Mean Reward = 38.31 ± 4.82, Utilization = 100.0%, True Long Capture = 98.8%, Starvation = 6.6 periods.
     - **Q-Learning:** Mean Reward = 37.33 ± 2.69, Utilization = 100.0%, True Long Capture = 82.2%, Starvation = 2.4 periods.
     - **Round-Robin:** Mean Reward = 31.55 ± 3.85, Utilization = 100.0%, True Long Capture = 72.6%, Starvation = 1.0 period.
  3. Expand the **Detailed Simulation Audit & Metric Formulas** accordion.
- **Viva Talking Point (CRITICAL):**
  > *"Here is the most critical scientific finding of our project: **Q-learning is NOT the top-performing policy on mean reward**. In our canonical multi-seed evaluation, the heuristic `Risk-Plus-Wait` policy achieves a mean reward of 39.93 ± 4.61, outperforming both FIFO (38.31 ± 4.08) and Q-Learning (37.33 ± 2.69). Why? Because tabular Q-learning discretizes the queue state into coarse bins (1,764 combinations), creating quantization loss compared to continuous priority sorting. However, Q-learning successfully learns an action policy under a human-designed composite reward function, discovering a stable policy with the lowest standard deviation (2.69), avoiding the severe queue starvation of Risk-First (2.4 periods vs. 6.6 periods), and outperforming static Round-Robin (31.55 ± 3.85)."*
  >
  > *"Notice also our definition of 'Retrospective True Long-Duration Capture': it measures the percentage of all historically prolonged cases (>124.93 hours) in the arrival cohort that were successfully processed within the simulation horizon. It is a retrospective diagnostic metric, not an intervention success rate."*

---

### 8. Multi-Page Diagnostic PDF Report Export (6:15 – 7:00)
- **Action:** Click **Export PDF Report** on the navigation bar or go to `http://127.0.0.1:5000/report/download`.
- **Interactions:**
  1. Open the downloaded PDF in the system viewer or Chrome.
  2. Show that the generated document is multi-page, professionally typeset with ReportLab, containing volume summary tables, borough breakdowns, ML test benchmarks, and a dedicated Reinforcement Learning Simulation Appendix.
- **Viva Talking Point:**
  > *"Finally, administrative users do not need to read raw logs or code. The `/report/download` route programmatically generates a publication-quality diagnostic PDF using ReportLab, complete with page numbering, visual charts, and a full simulation appendix for municipal leadership review."*

---

## Examiner Q&A Defense Sheet

### Q1: "Why is your regression R² only 0.1446?"
> **Answer:** *"Municipal complaint duration is heavily right-tailed (durations range from 0.05 hours to over 8,000 hours with a median of 8.7 hours and extreme multi-month capital outliers). At intake time, we only observe categorical features (agency, complaint type, borough, location type). Unobserved field realities—such as contractor availability, weather disruptions, physical access delays, or legal disputes—account for the remaining variance. An R² of 0.1446 without leakage is realistic and defensible for real-world public administration data, whereas an R² > 0.85 would indicate target leakage (such as inadvertently including resolution timestamps or status updates)."*

### Q2: "Can this system be deployed tomorrow in Pune Municipal Corporation (PMC) or PCMC?"
> **Answer:** *"No, and we explicitly document this limitation in Chapter 8 of our report. CivicResolve AI was trained on NYC 311 data. Indian municipal corporations operate with different departmental structures, complaint taxonomy, localized monsoon patterns, and informal reporting channels. Deploying in India would require re-ingesting local municipal data, retraining the pipeline with regional geographic features, and adapting intake forms for multilingual input (e.g., Marathi/Hindi)."*

### Q3: "Why did Q-Learning fail to beat Risk-Plus-Wait?"
> **Answer:** *"In tabular reinforcement learning, the continuous queue state had to be discretized into bins (queue size bins × age bins × risk tier bins = 1,764 states). In contrast, `Risk-Plus-Wait` computes a continuous priority score for every individual item. This continuous priority function is mathematically near-optimal for single-server queuing systems under linear delay penalties. The Q-learning agent demonstrates how an agent learns an action policy under a human-designed composite reward function, discovering an effective balance between risk and wait time, but coarse state discretization prevents it from matching fine-grained continuous priority sorting."*

### Q4: "What ensures there is no data leakage in your evaluation?"
> **Answer:** *"Three strict controls: First, strict temporal splitting: 2023 data was used for training and cross-validation, while 2024 data was partitioned into held-out validation and test sets. Second, intake-time feature filtering: closing dates, resolution notes, and status fields are quarantined before preprocessing. Third, all transformations (imputation, target encoding, standard scaling) were fit strictly on the training partition within a unified scikit-learn Pipeline."*
