"""Regression training, model comparison, and evaluation for CivicResolve AI."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
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

logger = logging.getLogger("train_regression")


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate standard regression metrics in original hours."""
    y_pred_clipped = np.clip(y_pred, 0.0, None)
    mae = float(mean_absolute_error(y_true, y_pred_clipped))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_clipped)))
    r2 = float(r2_score(y_true, y_pred_clipped))
    medae = float(median_absolute_error(y_true, y_pred_clipped))

    return {
        "mae_hours": round(mae, 3),
        "rmse_hours": round(rmse, 3),
        "r2_score": round(r2, 4),
        "median_ae_hours": round(medae, 3),
    }


def get_regression_models(seed: int = 42) -> dict[str, Any]:
    """Define candidate regression models comparing raw duration vs log-target."""
    return {
        "Dummy (Median)": DummyRegressor(strategy="median"),
        "Linear Regression (Raw)": LinearRegression(),
        "Linear Regression (Log1p)": TransformedTargetRegressor(
            regressor=Ridge(alpha=10.0, random_state=seed),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "Decision Tree (Raw)": DecisionTreeRegressor(max_depth=6, random_state=seed),
        "Decision Tree (Log1p)": TransformedTargetRegressor(
            regressor=DecisionTreeRegressor(max_depth=6, random_state=seed),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "Random Forest (Log1p)": TransformedTargetRegressor(
            regressor=RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=seed,
                n_jobs=-1,
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
    }


def run_regression_experiments(
    data_path: str | Path,
    models_dir: str | Path = "models",
    seed: int = 42,
) -> dict[str, Any]:
    """Run full regression benchmark across temporal partitions."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    train_df, val_df, test_df = split_temporal_data(df)

    # Supervised eligible filtering
    train_elig, train_counts = filter_eligible_supervised(train_df)
    val_elig, val_counts = filter_eligible_supervised(val_df)
    test_elig, test_counts = filter_eligible_supervised(test_df)

    # Compute training threshold for contract verification
    threshold = compute_training_threshold(train_elig, percentile=0.75)

    X_train, y_train, _ = prepare_feature_matrix(train_elig, threshold=threshold)
    X_val, y_val, _ = prepare_feature_matrix(val_elig, threshold=threshold)
    X_test, y_test, _ = prepare_feature_matrix(test_elig, threshold=threshold)

    assert y_train is not None and y_val is not None and y_test is not None

    candidate_models = get_regression_models(seed=seed)
    fitted_pipelines: dict[str, Pipeline] = {}
    val_results: list[dict[str, Any]] = []

    for name, model in candidate_models.items():
        preprocessor = build_preprocessor(min_category_frequency=15)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model),
        ])

        # Fit strictly on 2023 training data
        pipeline.fit(X_train, y_train.values)
        fitted_pipelines[name] = pipeline

        # Validation evaluation
        val_preds = np.clip(pipeline.predict(X_val), 0.0, None)
        metrics = compute_regression_metrics(y_val.values, val_preds)
        metrics["model_name"] = name
        val_results.append(metrics)

    val_df_results = pd.DataFrame(val_results).sort_values("mae_hours")

    # Select preferred model based on lowest Validation MAE
    preferred_name = str(val_df_results.iloc[0]["model_name"])
    preferred_pipeline = fitted_pipelines[preferred_name]

    # Evaluate selected model on 2024-H2 final test set ONCE
    test_preds = np.clip(preferred_pipeline.predict(X_test), 0.0, None)
    test_metrics = compute_regression_metrics(y_test.values, test_preds)
    test_metrics["model_name"] = preferred_name

    # Save preferred model pipeline
    save_path = models_dir / "preferred_regressor.joblib"
    joblib.dump(preferred_pipeline, save_path)

    # Sensitivity Analysis: excluding zero-duration records
    val_nonzero_mask = y_val.values > 0
    test_nonzero_mask = y_test.values > 0

    val_nonzero_preds = np.clip(preferred_pipeline.predict(X_val[val_nonzero_mask]), 0.0, None)
    val_nonzero_metrics = compute_regression_metrics(y_val.values[val_nonzero_mask], val_nonzero_preds)

    test_nonzero_preds = np.clip(preferred_pipeline.predict(X_test[test_nonzero_mask]), 0.0, None)
    test_nonzero_metrics = compute_regression_metrics(y_test.values[test_nonzero_mask], test_nonzero_preds)

    return {
        "partition_counts": {
            "train": train_counts,
            "val": val_counts,
            "test": test_counts,
        },
        "validation_comparison": val_results,
        "preferred_model_name": preferred_name,
        "preferred_model_file": str(save_path),
        "final_test_metrics": test_metrics,
        "zero_duration_sensitivity": {
            "val_nonzero_metrics": val_nonzero_metrics,
            "test_nonzero_metrics": test_nonzero_metrics,
            "val_zero_duration_count": int((y_val.values == 0).sum()),
            "test_zero_duration_count": int((y_test.values == 0).sum()),
        },
        "predictions_for_evaluation": {
            "val_y_true": y_val.values,
            "val_y_pred": np.clip(preferred_pipeline.predict(X_val), 0.0, None),
            "test_y_true": y_test.values,
            "test_y_pred": test_preds,
        },
    }
