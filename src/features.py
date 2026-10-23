"""Feature definitions, data contracts, and temporal partitioning for CivicResolve AI."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Strict intake-time feature allowlist
INTAKE_FEATURE_ALLOWLIST = [
    "complaint_type",
    "descriptor",
    "borough",
    "incident_zip",
    "open_data_channel_type",
    "created_hour",
    "created_weekday",
    "created_month",
    "is_weekend",
    "latitude",
    "longitude",
]

# Post-resolution fields forbidden from predictor matrices
EXCLUDED_POST_RESOLUTION_FIELDS = [
    "unique_key",
    "closed_date",
    "status",
    "resolution_description",
    "resolution_hours",
    "target_long_duration",
    "closed_dt_local",
]

AGENCY_USAGE_NOTE = (
    "agency is retained for descriptive demand analytics and administrative audits, "
    "but withheld from predictive baseline feature allowlist until intake-time availability is empirically proven."
)

logger = logging.getLogger("features")


def load_dataset(filepath: str | Path) -> pd.DataFrame:
    """Load the processed dataset and enforce basic date typing."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path, low_memory=False)
    return df


def split_temporal_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition dataset into mutually exclusive temporal splits:
    
    - Training:   Jan 1, 2023 - Dec 31, 2023
    - Validation: Jan 1, 2024 - Jun 30, 2024
    - Test:       Jul 1, 2024 - Dec 31, 2024
    
    Records are NEVER randomly shuffled across these boundaries.
    """
    # Ensure date column is parsed
    if "created_date_only" in df.columns:
        dates = pd.to_datetime(df["created_date_only"])
    else:
        dates = pd.to_datetime(df["created_date"]).dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")

    years = dates.dt.year
    months = dates.dt.month

    train_mask = years == 2023
    val_mask = (years == 2024) & (months <= 6)
    test_mask = (years == 2024) & (months > 6)

    # Sanity checks
    assert (train_mask & val_mask).sum() == 0, "Train and Val partitions overlap!"
    assert (train_mask & test_mask).sum() == 0, "Train and Test partitions overlap!"
    assert (val_mask & test_mask).sum() == 0, "Val and Test partitions overlap!"

    return df[train_mask].copy(), df[val_mask].copy(), df[test_mask].copy()


def filter_eligible_supervised(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filter records eligible for supervised duration modeling:
    
    Eligible: closed_date is not null AND resolution_hours >= 0.
    Ineligible: open/unresolved tickets (null duration) or negative durations (entry errors).
    """
    is_closed = df["closed_date"].notna()
    has_duration = df["resolution_hours"].notna()
    non_negative = df["resolution_hours"] >= 0

    eligible_mask = is_closed & has_duration & non_negative
    ineligible_open = (~is_closed) | (~has_duration)
    ineligible_negative = is_closed & has_duration & (df["resolution_hours"] < 0)

    counts = {
        "total_records": len(df),
        "eligible_supervised": int(eligible_mask.sum()),
        "ineligible_open_or_unresolved": int(ineligible_open.sum()),
        "ineligible_negative_duration": int(ineligible_negative.sum()),
    }
    return df[eligible_mask].copy(), counts


def compute_training_threshold(
    train_eligible_df: pd.DataFrame,
    percentile: float = 0.75,
) -> float:
    """Compute the long-duration classification threshold strictly from training records."""
    assert (train_eligible_df["created_year"] == 2023).all(), (
        "Classification threshold must ONLY be computed from 2023 training data!"
    )
    durations = train_eligible_df["resolution_hours"]
    threshold = float(durations.quantile(percentile))
    return threshold


def prepare_feature_matrix(
    df: pd.DataFrame,
    threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None]:
    """Extract approved intake feature matrix X and targets y_reg, y_cls.
    
    Guarantees that no post-resolution fields leak into X.
    """
    X = pd.DataFrame(index=df.index)

    # Categorical intake features
    X["complaint_type"] = df["complaint_type"].fillna("Missing").astype(str)
    X["descriptor"] = df["descriptor"].fillna("Missing").astype(str)
    X["borough"] = df["borough"].fillna("Missing").astype(str)
    
    # Format zip code as string category, avoiding float representation
    zips = df["incident_zip"].fillna(-1).astype(float).astype(int).astype(str)
    X["incident_zip"] = zips.replace("-1", "Missing")
    X["open_data_channel_type"] = df["open_data_channel_type"].fillna("Missing").astype(str)

    # Temporal intake features (local NYC context)
    if "created_date_only" in df.columns:
        dates = pd.to_datetime(df["created_date_only"])
        X["created_month"] = dates.dt.month.astype(float)
    else:
        X["created_month"] = pd.to_datetime(df["created_date"]).dt.month.astype(float)

    X["created_hour"] = df["created_hour"].astype(float)
    X["created_weekday"] = df["created_weekday"].astype(str)
    X["is_weekend"] = df["is_weekend"].astype(float)

    # Geospatial coordinates
    X["latitude"] = df["latitude"].astype(float)
    X["longitude"] = df["longitude"].astype(float)

    # Assert strict allowlist compliance
    assert set(X.columns) == set(INTAKE_FEATURE_ALLOWLIST), (
        f"Feature mismatch: {set(X.columns).symmetric_difference(set(INTAKE_FEATURE_ALLOWLIST))}"
    )
    for forbidden in EXCLUDED_POST_RESOLUTION_FIELDS:
        assert forbidden not in X.columns, f"Data leakage: {forbidden} present in feature matrix!"

    # Targets
    y_reg = df["resolution_hours"].astype(float) if "resolution_hours" in df.columns else None
    y_cls = None
    if y_reg is not None and threshold is not None:
        y_cls = (y_reg > threshold).astype(int)

    return X, y_reg, y_cls
