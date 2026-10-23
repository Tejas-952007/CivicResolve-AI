# CivicResolve AI — Scope Lock

## Project objective
Build a reproducible, explainable, software-only decision-support prototype using public municipal service-request data.

## MVP feature-to-technique map
| Feature | Method | Output | Evaluation |
|---|---|---|---|
| Service-demand analytics | Descriptive statistics | Trends, category and area views | Data-quality and reproducibility checks |
| Resolution-time estimate | Linear Regression and Decision Tree Regressor; optional Random Forest | Predicted elapsed hours/days | MAE, RMSE, R², median baseline |
| Long-duration risk | Logistic Regression and Decision Tree Classifier | Project-defined binary risk | Precision, Recall, F1, PR-AUC, confusion matrix |
| Pattern discovery | K-Means on area-category-period aggregates | Explainable historical clusters | Silhouette plus cluster profiling/stability |
| Report generation | ReportLab | PDF report with filters, charts, model metrics and caveats | Report-content tests and reproducibility |
| Q-learning (Phase 3) | Tabular Q-learning in a simulated environment | Simulated resource-allocation policy | Repeated-episode reward and backlog against simple baselines |

## Guardrails
- Keep the project focused on municipal service requests, not all government grievance types.
- Use temporally separated evaluation; do not leak closure-time information into prediction features.
- Cluster aggregate complaint patterns rather than attempting to cluster all raw rows first.
- A project-defined delay threshold must not be described as an official SLA breach unless it is validated against an official deadline definition.
- A model score is not causal evidence and not a guarantee of resolution time.
- Q-learning is simulated unless valid state/action/reward/transition data and operational authorization become available.
- Report plots and metrics must be generated from actual code runs, not fabricated.
- Do not claim local Indian government deployment without local data, permissions, and validation.
