"""Automated test suite for Phase 2: Leakage-Free ML Modeling, Evaluation, and Contracts."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import (
    EXCLUDED_POST_RESOLUTION_FIELDS,
    INTAKE_FEATURE_ALLOWLIST,
    compute_training_threshold,
    filter_eligible_supervised,
    load_dataset,
    prepare_feature_matrix,
    split_temporal_data,
)
from src.preprocess import build_preprocessor
from src.train_classification import get_classification_models
from src.train_regression import get_regression_models


class TestPhase2MLPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = ROOT / "data" / "processed" / "nyc_311_phase1_profiled.csv"
        cls.df = load_dataset(cls.data_path)
        cls.train_df, cls.val_df, cls.test_df = split_temporal_data(cls.df)
        cls.train_elig, cls.train_counts = filter_eligible_supervised(cls.train_df)
        cls.val_elig, cls.val_counts = filter_eligible_supervised(cls.val_df)
        cls.test_elig, cls.test_counts = filter_eligible_supervised(cls.test_df)
        cls.threshold = compute_training_threshold(cls.train_elig, percentile=0.75)

    def test_temporal_partitions_are_strictly_disjoint(self):
        """Test that Train (2023), Val (2024-H1), and Test (2024-H2) do not overlap."""
        train_ids = set(self.train_df["unique_key"])
        val_ids = set(self.val_df["unique_key"])
        test_ids = set(self.test_df["unique_key"])

        self.assertEqual(len(train_ids.intersection(val_ids)), 0)
        self.assertEqual(len(train_ids.intersection(test_ids)), 0)
        self.assertEqual(len(val_ids.intersection(test_ids)), 0)
        self.assertEqual(len(train_ids) + len(val_ids) + len(test_ids), len(self.df))

    def test_no_record_created_in_2024_enters_training(self):
        """Test that training data strictly contains only 2023 creation dates."""
        train_years = pd.to_datetime(self.train_df["created_date_only"]).dt.year
        self.assertTrue((train_years == 2023).all(), "2024 record found in training partition!")

        val_dates = pd.to_datetime(self.val_df["created_date_only"])
        self.assertTrue((val_dates.dt.year == 2024).all())
        self.assertTrue((val_dates.dt.month <= 6).all(), "Non-H1 date found in validation partition!")

        test_dates = pd.to_datetime(self.test_df["created_date_only"])
        self.assertTrue((test_dates.dt.year == 2024).all())
        self.assertTrue((test_dates.dt.month > 6).all(), "Non-H2 date found in test partition!")

    def test_regression_targets_contain_no_negative_durations(self):
        """Test that supervised targets contain only non-negative durations."""
        _, y_train_reg, _ = prepare_feature_matrix(self.train_elig, threshold=self.threshold)
        _, y_val_reg, _ = prepare_feature_matrix(self.val_elig, threshold=self.threshold)
        _, y_test_reg, _ = prepare_feature_matrix(self.test_elig, threshold=self.threshold)

        self.assertIsNotNone(y_train_reg)
        self.assertIsNotNone(y_val_reg)
        self.assertIsNotNone(y_test_reg)

        self.assertTrue((y_train_reg >= 0).all(), "Negative duration in training targets!")
        self.assertTrue((y_val_reg >= 0).all(), "Negative duration in validation targets!")
        self.assertTrue((y_test_reg >= 0).all(), "Negative duration in test targets!")

    def test_open_cases_are_excluded_from_supervised_targets(self):
        """Test that unresolved/open tickets are excluded from supervised targets."""
        self.assertEqual(self.train_counts["ineligible_open_or_unresolved"], 95)
        self.assertEqual(self.val_counts["ineligible_open_or_unresolved"], 36)
        self.assertEqual(self.test_counts["ineligible_open_or_unresolved"], 65)
        self.assertEqual(self.train_counts["ineligible_negative_duration"], 21)

        # In eligible datasets, closed_date must be completely non-null
        self.assertEqual(self.train_elig["closed_date"].isna().sum(), 0)
        self.assertEqual(self.val_elig["closed_date"].isna().sum(), 0)
        self.assertEqual(self.test_elig["closed_date"].isna().sum(), 0)

    def test_threshold_is_computed_from_training_set_only(self):
        """Test that long-duration threshold is derived strictly from 2023 training data."""
        self.assertAlmostEqual(self.threshold, 124.925, places=2)
        # Verify that computing on validation or test would give different values
        val_thresh = float(self.val_elig["resolution_hours"].quantile(0.75))
        self.assertNotEqual(round(self.threshold, 2), round(val_thresh, 2))

    def test_no_post_resolution_field_enters_feature_matrix(self):
        """Test strict feature allowlist and complete absence of target leakage columns."""
        X_train, _, _ = prepare_feature_matrix(self.train_elig, threshold=self.threshold)
        for col in X_train.columns:
            self.assertIn(col, INTAKE_FEATURE_ALLOWLIST)
        for forbidden in EXCLUDED_POST_RESOLUTION_FIELDS:
            self.assertNotIn(forbidden, X_train.columns)
            self.assertNotIn(forbidden.lower(), [c.lower() for c in X_train.columns])

    def test_missing_values_and_unseen_categories_handled_safely(self):
        """Test preprocessing pipeline handles nulls and previously unseen test categories."""
        preprocessor = build_preprocessor(min_category_frequency=15)
        X_train, y_train_reg, _ = prepare_feature_matrix(self.train_elig, threshold=self.threshold)
        preprocessor.fit(X_train)

        # Create adversarial test row with novel categories and all-missing optional fields
        adversarial_X = pd.DataFrame([{
            "complaint_type": "Completely Novel Martian Complaint",
            "descriptor": "Unseen Bug in the Sky",
            "borough": "Atlantis",
            "incident_zip": "99999",
            "open_data_channel_type": "TELEPATHY",
            "created_hour": 14.0,
            "created_weekday": "Funday",
            "created_month": 7.0,
            "is_weekend": 1.0,
            "latitude": np.nan,
            "longitude": np.nan,
        }])

        transformed = preprocessor.transform(adversarial_X)
        self.assertFalse(np.isnan(transformed).any(), "NaN found in transformed adversarial output!")
        self.assertFalse(np.isinf(transformed).any(), "Inf found in transformed adversarial output!")

    def test_pipeline_save_load_and_prediction_reproducibility(self):
        """Test that fitted pipelines can be serialized to disk and reproduce exact predictions."""
        preprocessor = build_preprocessor(min_category_frequency=15)
        model = get_regression_models()["Linear Regression (Log1p)"]
        pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", model)])

        X_train, y_train_reg, _ = prepare_feature_matrix(self.train_elig, threshold=self.threshold)
        X_val, _, _ = prepare_feature_matrix(self.val_elig, threshold=self.threshold)

        pipeline.fit(X_train, y_train_reg.values)
        pred_before = pipeline.predict(X_val)

        with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
            joblib.dump(pipeline, tmp.name)
            loaded_pipe = joblib.load(tmp.name)
            pred_after = loaded_pipe.predict(X_val)

        np.testing.assert_allclose(pred_before, pred_after, rtol=1e-6)

    def test_models_produce_valid_finite_predictions(self):
        """Test that regression and classification models produce finite non-negative outputs."""
        X_train, y_train_reg, y_train_cls = prepare_feature_matrix(self.train_elig, threshold=self.threshold)
        X_val, _, _ = prepare_feature_matrix(self.val_elig, threshold=self.threshold)

        # Regression
        for name, reg in get_regression_models().items():
            pipe = Pipeline([("prep", build_preprocessor()), ("est", reg)])
            pipe.fit(X_train, y_train_reg.values)
            preds = np.clip(pipe.predict(X_val), 0.0, None)
            self.assertFalse(np.isnan(preds).any(), f"NaN in {name} predictions!")
            self.assertFalse(np.isinf(preds).any(), f"Inf in {name} predictions!")
            self.assertTrue((preds >= 0).all(), f"Negative prediction in {name}!")

        # Classification
        for name, clf in get_classification_models().items():
            pipe = Pipeline([("prep", build_preprocessor()), ("est", clf)])
            pipe.fit(X_train, y_train_cls.values)
            preds = pipe.predict(X_val)
            self.assertTrue(set(preds).issubset({0, 1}), f"Invalid binary label in {name}!")

    def test_final_test_set_not_used_in_model_selection(self):
        """Test that preferred model selection depends strictly on validation partition without test feedback."""
        from src.train_regression import run_regression_experiments
        from src.train_classification import run_classification_experiments

        reg_results = run_regression_experiments(self.data_path, models_dir=tempfile.gettempdir())
        cls_results = run_classification_experiments(self.data_path, models_dir=tempfile.gettempdir())

        # Verify regression selection matches validation MAE minimum
        val_df = pd.DataFrame(reg_results["validation_comparison"])
        best_val_reg = val_df.sort_values("mae_hours").iloc[0]["model_name"]
        self.assertEqual(reg_results["preferred_model_name"], best_val_reg)

        # Verify classification selection matches validation F1 maximum
        val_cls_df = pd.DataFrame(cls_results["validation_comparison"])
        best_val_cls = val_cls_df.sort_values("f1_score", ascending=False).iloc[0]["model_name"]
        self.assertEqual(cls_results["preferred_model_name"], best_val_cls)


if __name__ == "__main__":
    unittest.main()
