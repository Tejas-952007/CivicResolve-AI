# CivicResolve AI — Phase 6: Quality Assurance, Adversarial Testing & Defect Hunt Test Plan

**Version:** 1.0.0  
**Date:** October 2026  
**Role:** Senior QA Engineer, ML Validation Engineer & Application Security Reviewer  
**Target Application:** CivicResolve AI (Flask, scikit-learn, Pandas, ReportLab, Plotly, Q-Learning)  
**Workspace:** `~/Downloads/CivicResolveAI_Starter`  

---

## 1. Executive Summary & Objective

The objective of Phase 6 is to perform an exhaustive, evidence-based quality assurance campaign, adversarial penetration audit, and defect remediation process on CivicResolve AI. CivicResolve AI is an academic research prototype and municipal decision-support system designed to forecast NYC 311 service resolution velocities, flag long-duration backlog risks, discover operational complaint archetypes via K-Means, and simulate capacity-constrained dispatch policies via tabular Q-learning.

This test plan defines the testing methodology, adversarial test vectors, boundary conditions, and acceptance criteria across 10 specialized stages.

---

## 2. Non-Negotiable Safety Rules & Constraints

1. **Preserve Canonical Data & Frozen Models:** Never delete, overwrite, or retrain the canonical frozen models (`preferred_regressor.joblib`, `preferred_classifier.joblib`), metadata (`model_metadata.json`), or temporal data partitions.
2. **Dedicated Synthetic Test Directory:** All adversarial and malformed inputs are generated purely in-memory or in isolated temporary directories (`scratch/` or pytest fixtures).
3. **No External Network Calls:** Testing is strictly localized to the internal Flask application and its synthetic test inputs.
4. **Minimal, Safe Fixes:** Fixes must be surgical, preserving all documented contracts and without weakening validation assertions.
5. **Honest Reporting:** No fabrication of test results, screenshots, or release status. The final release recommendation must reflect genuine local readiness (PASS, CONDITIONAL PASS, or FAIL).

---

## 3. Test Stages & Scope

### Stage 1: Baseline Verification
- Python environment verification (`.venv/bin/python --version`).
- Execution of baseline test suite (`.venv/bin/pytest tests/ -v`).
- Record baseline test count (45 tests), runtime (27.14s), and zero failures.

### Stage 2: Adversarial Input Testing
- **Missing & Empty Fields:** Missing mandatory fields (`complaint_type`, `borough`), empty strings, whitespace-only strings.
- **Malformed & Extreme Values:** Non-finite numeric inputs (`NaN`, `+Inf`, `-Inf`), impossible timestamps, extreme out-of-bounds coordinates (latitude `> 90` or `< -90`, longitude `> 180` or `< -180`).
- **Unicode & Injection Payloads:** Multilingual text (Marathi/Hindi: "रस्त्यावर कचरा साचला आहे", Emoji: "🚨🔥"), SQL injection strings (`' OR '1'='1`), XSS/HTML tags (`<script>alert('XSS')</script>`), and newline attacks.
- **CSV Formula Injection (CWE-1236):** Uploaded cells starting with spreadsheet command characters (`=`, `+`, `-`, `@`) must be safely sanitized on export without corrupting legitimate negative numeric numbers.
- **Target Leakage Quarantine:** Uploaded batch files containing prohibited post-resolution fields (`closed_date`, `status`, `resolution_hours`, `target_long_duration`, `resolution_description`) must be strictly stripped from model matrices and final output.

### Stage 3: ML Correctness & Leakage Audit
- **Temporal Partition Disjointness:** Ensure training (2023), validation (2024-H1), and test (2024-H2) sets are mutually exclusive with zero record overlap.
- **Zero Future Leakage:** Verify no record created in 2024 ever enters the 2023 training partition.
- **Threshold Integrity:** Verify the 124.925-hour long-duration threshold is derived solely from the 2023 training set distribution.
- **Reproducibility:** Verify that identical valid inputs to frozen pipelines yield bitwise reproducible predictions.
- **Prediction Bounds:** Verify all regression outputs satisfy $\hat{y} \ge 0.0$ and classification probabilities satisfy $0.0 \le \hat{p} \le 1.0$.

### Stage 4: Flask Routes & Application Security
- **HTTP Method Enforcement:** Verify GET-only routes reject POST/PUT/DELETE with HTTP 405 Method Not Allowed.
- **Path Traversal Protection:** Test `/figures/<path:filename>` with traversal vectors (`../../app.py`, `%2e%2e%2f`).
- **File Upload Guardrails:** Test missing file uploads, invalid file extensions (`.exe`, `.sh`), empty CSV files, and oversized uploads (`MAX_CONTENT_LENGTH`).
- **Error Handling:** Verify application handles 404, 405, and 500 errors gracefully with clean user feedback without exposing stack traces.

### Stage 5: K-Means & Analytics Edge Cases
- **Degenerate Sample Handling:** Behavior under empty data, single-row data, and fewer segments than candidate cluster counts ($N < k$).
- **Zero Variance:** Handling of duplicate or identical segment profiles.
- **PCA Dimensionality Safety:** Mathematical validity of 2D PCA projection across degenerate inputs.

### Stage 6: Q-Learning & Simulation Stress Tests
- **Queue Boundary Conditions:** All queues empty, one non-empty queue, capacity of 1, capacity exceeding arrival volume.
- **Parameter Validation:** Out-of-bounds capacity, horizon periods, batch sizes, learning rates ($\alpha$), discount factors ($\gamma$).
- **Fair Policy Cohort:** Ensure all 5 policies receive the exact identical arrival sequence under fixed seeds.
- **Toy Hand-Calculated Cases:** Verification of step transitions, backlog accumulation, and reward computation against deterministic mathematical expectations.

### Stage 7: PDF & Report Generation
- **Signature & Header Validation:** Ensure generated reports start with valid `%PDF-1.4` magic bytes.
- **Filter Edge Cases:** Reports generated with empty dates, Unicode text, non-existent categories, or single-ticket scopes.
- **Division-by-Zero Safety:** Graceful handling when denominator for percentages (e.g., closed tickets) is 0.

### Stage 8: Integration & Usability Smoke Tests
- End-to-end flow from dashboard load -> single prediction -> batch upload -> evaluation -> clustering -> resource planning -> PDF export.

### Stage 9: Performance & Dependency Integrity
- Batch processing latency for 100, 500, and 1,000 synthetic rows.
- Dependency audit via `pip freeze` to ensure no unexpected packages or security advisories.

### Stage 10: Fix, Retest & Defect Reporting
- Systematically reproduce, fix, add regression tests, and rerun the full test suite.
