"""Reusable Scikit-learn preprocessing pipelines for CivicResolve AI.

Guarantees:
- Preprocessor is fitted strictly on training data.
- Handles missing coordinates via median imputation.
- Handles rare and unseen categories gracefully via OneHotEncoder(min_frequency=15, handle_unknown='ignore').
- Encodes cyclical temporal features (hour, month) using smooth sine and cosine transformations.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class CyclicFeatureTransformer(BaseEstimator, TransformerMixin):
    """Transform cyclical features (hour of day, calendar month) into sine/cosine pairs."""

    def __init__(self, periods: dict[str, float] | None = None) -> None:
        # Default: hour has period 24, month has period 12
        self.periods = periods or {"created_hour": 24.0, "created_month": 12.0}

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> CyclicFeatureTransformer:
        self.is_fitted_ = True
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "is_fitted_") and self.is_fitted_

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=list(self.periods.keys()))

        outputs = []
        for col, period in self.periods.items():
            vals = df[col].astype(float).values
            sin_vals = np.sin(2.0 * np.pi * vals / period)
            cos_vals = np.cos(2.0 * np.pi * vals / period)
            outputs.append(sin_vals.reshape(-1, 1))
            outputs.append(cos_vals.reshape(-1, 1))

        return np.hstack(outputs)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        names = []
        for col in self.periods.keys():
            names.extend([f"{col}_sin", f"{col}_cos"])
        return np.array(names)


def build_preprocessor(min_category_frequency: int = 15) -> ColumnTransformer:
    """Build a robust, leakage-free ColumnTransformer for CivicResolve AI.
    
    Transforms:
    - Numerical coordinates & weekend flag: Median Imputer + Standard Scaler
    - Cyclical time features (hour, month): Sine/Cosine Cyclic Transformer
    - Categorical features: OneHotEncoder with infrequent category grouping and ignore for unseen
    """
    cat_cols = [
        "complaint_type",
        "descriptor",
        "borough",
        "incident_zip",
        "open_data_channel_type",
        "created_weekday",
    ]

    coord_cols = ["latitude", "longitude", "is_weekend"]
    cyclic_cols = ["created_hour", "created_month"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cyclic_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("cyclic", CyclicFeatureTransformer()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OneHotEncoder(
            min_frequency=min_category_frequency,
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, coord_cols),
            ("cyclic", cyclic_pipeline, cyclic_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor
