"""Evaluation suite, permutation feature importance, and diagnostic visualizations for CivicResolve AI."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import (
    compute_training_threshold,
    filter_eligible_supervised,
    load_dataset,
    prepare_feature_matrix,
    split_temporal_data,
)
from src.train_classification import run_classification_experiments
from src.train_regression import run_regression_experiments
DEFAULT_DATA_PATH = ROOT / "data" / "processed" / "nyc_311_phase1_profiled.csv"
FIGURES_DIR = ROOT / "outputs" / "figures"
MODELS_DIR = ROOT / "models"
METRICS_JSON = ROOT / "outputs" / "phase2_metrics.json"
COMPARISON_CSV = ROOT / "outputs" / "phase2_model_comparison.csv"
MODEL_METADATA_JSON = ROOT / "models" / "model_metadata.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate")


def generate_evaluation_plots(
    reg_results: dict[str, Any],
    cls_results: dict[str, Any],
    reg_importance: pd.Series,
    cls_importance: pd.Series,
    output_dir: Path,
) -> None:
    """Generate all Phase 2 diagnostic visualization artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Regression model comparison bar chart
    reg_val_df = pd.DataFrame(reg_results["validation_comparison"]).sort_values("mae_hours")
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(reg_val_df["model_name"], reg_val_df["mae_hours"], color="#1f77b4", edgecolor="black")
    ax.set_title("Phase 2 Regression Benchmark: Validation MAE (Lower is Better)")
    ax.set_xlabel("Mean Absolute Error (Hours)")
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.1f}h", xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "phase2_regression_comparison.png", dpi=160)
    plt.close(fig)

    # 2. Actual vs Predicted Resolution Duration
    y_true_reg = reg_results["predictions_for_evaluation"]["val_y_true"]
    y_pred_reg = reg_results["predictions_for_evaluation"]["val_y_pred"]

    fig, ax = plt.subplots(figsize=(7, 7))
    # Log scale scatter plot to visualize multi-order-of-magnitude spread
    ax.scatter(y_true_reg + 1, y_pred_reg + 1, alpha=0.35, edgecolors="none", color="#2ca02c", s=20)
    max_val = max(float(y_true_reg.max()), float(y_pred_reg.max())) + 1
    ax.plot([1, max_val], [1, max_val], color="red", linestyle="--", linewidth=1.5, label="Perfect Agreement (y = x)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Actual vs Predicted Resolution Duration (Validation Set)")
    ax.set_xlabel("Actual Resolution Hours (log scale, +1h offset)")
    ax.set_ylabel("Predicted Resolution Hours (log scale, +1h offset)")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "phase2_actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    # 3. Regression Residual Distribution
    residuals = y_true_reg - y_pred_reg
    # Clip extreme residuals for readable distribution view
    p01, p99 = float(np.percentile(residuals, 1)), float(np.percentile(residuals, 99))
    clipped_res = residuals[(residuals >= p01) & (residuals <= p99)]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(clipped_res, bins=45, color="#d62728", edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, label="Zero Error")
    ax.axvline(float(np.median(residuals)), color="blue", linestyle=":", linewidth=1.5,
               label=f"Median Residual: {np.median(residuals):.1f}h")
    ax.set_title("Regression Residual Distribution (Trimmed 1st–99th Percentile)")
    ax.set_xlabel("Residual (Actual - Predicted Hours)")
    ax.set_ylabel("Validation Record Count")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "phase2_regression_residuals.png", dpi=160)
    plt.close(fig)

    # 4. Classification Confusion Matrix Heatmap
    cm = np.array(cls_results["final_test_metrics"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax)
    classes = ["Normal (<125h)", "Long-Duration (>=125h)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, rotation=15)
    ax.set_yticklabels(classes)
    ax.set_title(f"Test Confusion Matrix: {cls_results['preferred_model_name']}")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "phase2_classification_confusion_matrix.png", dpi=160)
    plt.close(fig)

    # 5. Precision-Recall and ROC Curves
    y_true_cls = cls_results["predictions_for_evaluation"]["val_y_true"]
    y_prob_cls = cls_results["predictions_for_evaluation"]["val_y_prob"]

    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(12, 5))

    # Precision-Recall
    precisions, recalls, _ = precision_recall_curve(y_true_cls, y_prob_cls)
    ap = cls_results["validation_comparison"][2]["pr_auc"]  # Logistic Regression (Balanced)
    ax_pr.plot(recalls, precisions, color="#ff7f0e", linewidth=2, label=f"PR curve (AP = {ap:.3f})")
    ax_pr.axhline(float(y_true_cls.mean()), color="grey", linestyle="--", label=f"Baseline ({y_true_cls.mean():.2f})")
    ax_pr.set_title("Validation Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend()
    ax_pr.grid(True, linestyle="--", alpha=0.3)

    # ROC
    fpr, tpr, _ = roc_curve(y_true_cls, y_prob_cls)
    roc_auc = cls_results["validation_comparison"][2]["roc_auc"]
    ax_roc.plot(fpr, tpr, color="#9467bd", linewidth=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random Chance")
    ax_roc.set_title("Validation ROC Curve")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()
    ax_roc.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "phase2_pr_roc_curves.png", dpi=160)
    plt.close(fig)

    # 6. Permutation Feature Importance
    fig, (ax_reg_imp, ax_cls_imp) = plt.subplots(1, 2, figsize=(14, 6))

    top_reg = reg_importance.head(10).sort_values()
    ax_reg_imp.barh(top_reg.index, top_reg.values, color="#17becf", edgecolor="black")
    ax_reg_imp.set_title(f"Regression Feature Importance (Validation MAE Drop)\nModel: {reg_results['preferred_model_name']}")
    ax_reg_imp.set_xlabel("Increase in MAE (Hours) when Permuted")

    top_cls = cls_importance.head(10).sort_values()
    ax_cls_imp.barh(top_cls.index, top_cls.values, color="#e377c2", edgecolor="black")
    ax_cls_imp.set_title(f"Classification Feature Importance (Validation F1 Drop)\nModel: {cls_results['preferred_model_name']}")
    ax_cls_imp.set_xlabel("Decrease in F1-score when Permuted")

    fig.tight_layout()
    fig.savefig(output_dir / "phase2_feature_importance.png", dpi=160)
    plt.close(fig)


def run_full_evaluation(
    data_path: str | Path = DEFAULT_DATA_PATH,
    figures_dir: str | Path = FIGURES_DIR,
    models_dir: str | Path = MODELS_DIR,
) -> dict[str, Any]:
    """Execute regression and classification benchmarks, feature importance, and artifacts."""
    logger.info("Executing Phase 2 Regression Experiments...")
    reg_results = run_regression_experiments(data_path=data_path, models_dir=models_dir)

    logger.info("Executing Phase 2 Classification Experiments...")
    cls_results = run_classification_experiments(data_path=data_path, models_dir=models_dir)

    # Load validation dataset for Permutation Feature Importance
    df = load_dataset(data_path)
    train_df, val_df, _ = split_temporal_data(df)
    train_elig, _ = filter_eligible_supervised(train_df)
    val_elig, _ = filter_eligible_supervised(val_df)
    threshold = cls_results["threshold_hours"]

    X_val, y_val_reg, y_val_cls = prepare_feature_matrix(val_elig, threshold=threshold)
    assert y_val_reg is not None and y_val_cls is not None

    preferred_reg = joblib.load(reg_results["preferred_model_file"])
    preferred_cls = joblib.load(cls_results["preferred_model_file"])

    logger.info("Computing Validation Permutation Feature Importance for Regressor...")
    perm_reg = permutation_importance(
        preferred_reg, X_val, y_val_reg.values,
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=42,
    )
    reg_importance = pd.Series(
        -perm_reg.importances_mean,
        index=X_val.columns,
    ).sort_values(ascending=False)

    logger.info("Computing Validation Permutation Feature Importance for Classifier...")
    perm_cls = permutation_importance(
        preferred_cls, X_val, y_val_cls.values,
        scoring="f1",
        n_repeats=5,
        random_state=42,
    )
    cls_importance = pd.Series(
        perm_cls.importances_mean,
        index=X_val.columns,
    ).sort_values(ascending=False)

    # Save Comparison CSV
    all_comparisons = []
    for r in reg_results["validation_comparison"]:
        all_comparisons.append({
            "task": "Regression",
            "model_name": r["model_name"],
            "val_mae_hours": r["mae_hours"],
            "val_rmse_hours": r["rmse_hours"],
            "val_r2": r["r2_score"],
            "val_f1": np.nan,
            "val_pr_auc": np.nan,
            "val_roc_auc": np.nan,
        })
    for c in cls_results["validation_comparison"]:
        all_comparisons.append({
            "task": "Classification",
            "model_name": c["model_name"],
            "val_mae_hours": np.nan,
            "val_rmse_hours": np.nan,
            "val_r2": np.nan,
            "val_f1": c["f1_score"],
            "val_pr_auc": c["pr_auc"],
            "val_roc_auc": c["roc_auc"],
        })
    comparison_df = pd.DataFrame(all_comparisons)
    comparison_df.to_csv(COMPARISON_CSV, index=False)

    # Generate plots
    generate_evaluation_plots(
        reg_results=reg_results,
        cls_results=cls_results,
        reg_importance=reg_importance,
        cls_importance=cls_importance,
        output_dir=Path(figures_dir),
    )

    # Save Master Metrics JSON
    metrics_report = {
        "dataset_used": str(data_path),
        "partition_observation_counts": {
            "training_2023": reg_results["partition_counts"]["train"],
            "validation_2024_H1": reg_results["partition_counts"]["val"],
            "test_2024_H2": reg_results["partition_counts"]["test"],
        },
        "regression": {
            "validation_comparison": reg_results["validation_comparison"],
            "preferred_model": reg_results["preferred_model_name"],
            "final_test_metrics": reg_results["final_test_metrics"],
            "zero_duration_sensitivity": reg_results["zero_duration_sensitivity"],
            "permutation_importance_mae_hours": reg_importance.round(3).to_dict(),
            "importance_interpretation_note": (
                "Permutation importance measures how much validation MAE degrades when feature values are randomly shuffled. "
                "It indicates statistical model dependence, not physical causation."
            ),
        },
        "classification": {
            "threshold_hours": threshold,
            "threshold_derivation": "75th percentile of eligible 2023 training records",
            "validation_comparison": cls_results["validation_comparison"],
            "preferred_model": cls_results["preferred_model_name"],
            "final_test_metrics": cls_results["final_test_metrics"],
            "permutation_importance_f1_drop": cls_importance.round(4).to_dict(),
            "importance_interpretation_note": (
                "Permutation importance measures drop in validation F1-score when a feature is shuffled. "
                "It represents model association rather than causal municipal driver."
            ),
        },
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)

    # Save Model Metadata JSON
    model_metadata = {
        "project": "CivicResolve AI",
        "phase": "Phase 2: Leakage-Free ML Modeling",
        "saved_regressor": str(reg_results["preferred_model_file"]),
        "saved_classifier": str(cls_results["preferred_model_file"]),
        "input_features": list(X_val.columns),
        "long_duration_threshold_hours": threshold,
        "regression_test_mae_hours": reg_results["final_test_metrics"]["mae_hours"],
        "classification_test_f1": cls_results["final_test_metrics"]["f1_score"],
        "classification_test_recall": cls_results["final_test_metrics"]["recall"],
    }
    with open(MODEL_METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)

    logger.info(f"Phase 2 Evaluation Complete. Saved metrics to {METRICS_JSON} and plots to {figures_dir}.")
    return metrics_report


if __name__ == "__main__":
    run_full_evaluation()
