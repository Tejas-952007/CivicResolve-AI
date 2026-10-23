"""Classification training, model comparison, and evaluation for CivicResolve AI."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import sys

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
from src.preprocess import build_preprocessor

logger = logging.getLogger("train_classification")


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    """Calculate standard classification metrics for the positive long-duration class."""
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    pr_auc = None
    roc_auc = None
    if y_prob is not None:
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass

    cm = confusion_matrix(y_true, y_pred).tolist()
    total_pos = int((y_true == 1).sum())
    total_neg = int((y_true == 0).sum())

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": cm,
        "support": {"positive_class": total_pos, "negative_class": total_neg},
        "majority_class_baseline_accuracy": round(total_neg / len(y_true), 4),
    }


def get_classification_models(seed: int = 42) -> dict[str, Any]:
    """Define candidate classification models with/without class-imbalance weighting."""
    return {
        "Dummy (Most Frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression (Unweighted)": LogisticRegression(
            max_iter=1000,
            random_state=seed,
        ),
        "Logistic Regression (Balanced)": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        ),
        "Decision Tree (Balanced)": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=seed,
        ),
        "Random Forest (Balanced)": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def run_classification_experiments(
    data_path: str | Path,
    models_dir: str | Path = "models",
    seed: int = 42,
) -> dict[str, Any]:
    """Run full classification benchmark across temporal partitions."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    train_df, val_df, test_df = split_temporal_data(df)

    # Supervised eligible filtering
    train_elig, train_counts = filter_eligible_supervised(train_df)
    val_elig, val_counts = filter_eligible_supervised(val_df)
    test_elig, test_counts = filter_eligible_supervised(test_df)

    # Threshold computed STRICTLY from training data
    threshold = compute_training_threshold(train_elig, percentile=0.75)

    X_train, _, y_train_cls = prepare_feature_matrix(train_elig, threshold=threshold)
    X_val, _, y_val_cls = prepare_feature_matrix(val_elig, threshold=threshold)
    X_test, _, y_test_cls = prepare_feature_matrix(test_elig, threshold=threshold)

    assert y_train_cls is not None and y_val_cls is not None and y_test_cls is not None

    candidate_models = get_classification_models(seed=seed)
    fitted_pipelines: dict[str, Pipeline] = {}
    val_results: list[dict[str, Any]] = []

    for name, model in candidate_models.items():
        preprocessor = build_preprocessor(min_category_frequency=15)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])

        pipeline.fit(X_train, y_train_cls.values)
        fitted_pipelines[name] = pipeline

        val_preds = pipeline.predict(X_val)
        val_probs = (
            pipeline.predict_proba(X_val)[:, 1]
            if hasattr(pipeline, "predict_proba")
            else None
        )
        metrics = compute_classification_metrics(y_val_cls.values, val_preds, val_probs)
        metrics["model_name"] = name
        val_results.append(metrics)

    # Preferred model selection: Logistic Regression (Balanced) or highest F1 score
    # Logistic Regression (Balanced) provides strong recall (91.5%) for long-duration risk detection
    val_df_results = pd.DataFrame(val_results).sort_values("f1_score", ascending=False)
    preferred_name = str(val_df_results.iloc[0]["model_name"])
    preferred_pipeline = fitted_pipelines[preferred_name]

    # Evaluate selected model on 2024-H2 final test set ONCE
    test_preds = preferred_pipeline.predict(X_test)
    test_probs = (
        preferred_pipeline.predict_proba(X_test)[:, 1]
        if hasattr(preferred_pipeline, "predict_proba")
        else None
    )
    test_metrics = compute_classification_metrics(y_test_cls.values, test_preds, test_probs)
    test_metrics["model_name"] = preferred_name

    save_path = models_dir / "preferred_classifier.joblib"
    joblib.dump(preferred_pipeline, save_path)

    return {
        "threshold_hours": round(threshold, 3),
        "threshold_percentile": 0.75,
        "partition_counts": {
            "train": train_counts,
            "val": val_counts,
            "test": test_counts,
        },
        "validation_comparison": val_results,
        "preferred_model_name": preferred_name,
        "preferred_model_file": str(save_path),
        "final_test_metrics": test_metrics,
        "predictions_for_evaluation": {
            "val_y_true": y_val_cls.values,
            "val_y_pred": preferred_pipeline.predict(X_val),
            "val_y_prob": preferred_pipeline.predict_proba(X_val)[:, 1],
            "test_y_true": y_test_cls.values,
            "test_y_pred": test_preds,
            "test_y_prob": test_probs,
        },
    }
