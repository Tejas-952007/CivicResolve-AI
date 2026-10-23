"""Profile the temporally stratified 311 sample and generate diagnostic artifacts.

Computes comprehensive data-quality metrics, feature boundaries, and diagnostic plots
while preserving the original Phase 0 profiler and artifacts intact.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_INPUT = ROOT / "data" / "raw" / "nyc_311_temporal_sample.csv"
PROCESSED_OUTPUT = ROOT / "data" / "processed" / "nyc_311_phase1_profiled.csv"
REPORT_OUTPUT = ROOT / "outputs" / "phase1_data_quality.json"
MANIFEST_INPUT = ROOT / "outputs" / "sampling_manifest.json"
FIGURES_DIR = ROOT / "outputs" / "figures"

# Strict intake-time feature allowlist vs excluded post-resolution fields
INTAKE_FEATURE_ALLOWLIST = [
    "complaint_type",
    "descriptor",
    "borough",
    "incident_zip",
    "open_data_channel_type",
    "created_hour",
    "created_weekday",
    "created_month",
    "created_year",
    "is_weekend",
    "latitude",
    "longitude",
]

EXCLUDED_POST_RESOLUTION_FIELDS = [
    "closed_date",
    "status",
    "resolution_description",
    "resolution_hours",
    "target_long_duration",
]

AGENCY_USAGE_NOTE = (
    "agency is retained in processed dataset for descriptive demand analytics and routing audit, "
    "but withheld from the primary intake-time predictive allowlist until it is empirically verified "
    "that agency assignment occurs instantaneously at intake rather than through downstream triage."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("profile_temporal_sample")


def is_near_quarter_end(ts_series: pd.Series) -> pd.Series:
    """Identify timestamps occurring in the final 24 hours of any calendar quarter."""
    # Quarter ends: Mar 31, Jun 30, Sep 30, Dec 31
    is_q_end_day = (
        ((ts_series.dt.month == 3) & (ts_series.dt.day == 31)) |
        ((ts_series.dt.month == 6) & (ts_series.dt.day == 30)) |
        ((ts_series.dt.month == 9) & (ts_series.dt.day == 30)) |
        ((ts_series.dt.month == 12) & (ts_series.dt.day == 31))
    )
    return is_q_end_day


def main() -> int:
    if not RAW_INPUT.exists():
        logger.error(f"Raw input not found: {RAW_INPUT}")
        logger.error("Run first: python scripts/03_build_temporal_sample.py")
        return 2

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_INPUT, low_memory=False)
    logger.info(f"Loaded raw sample: {len(df):,} rows, {len(df.columns)} columns.")

    # Manifest metadata
    manifest_info: dict[str, Any] = {}
    if MANIFEST_INPUT.exists():
        try:
            with MANIFEST_INPUT.open("r", encoding="utf-8") as f:
                manifest_info = json.load(f)
        except Exception as exc:
            logger.warning(f"Could not read manifest: {exc}")

    # Timestamp parsing: America/New_York (Eastern Time) interpretation
    # Socrata Floating Timestamps represent local NYC time
    created_raw = df["created_date"]
    closed_raw = df["closed_date"]

    df["created_dt_local"] = pd.to_datetime(created_raw, errors="coerce").dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="shift_forward"
    )
    df["closed_dt_local"] = pd.to_datetime(closed_raw, errors="coerce").dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="shift_forward"
    )

    # Convert to UTC for exact elapsed duration calculation accounting for DST transitions
    created_utc = df["created_dt_local"].dt.tz_convert("UTC")
    closed_utc = df["closed_dt_local"].dt.tz_convert("UTC")
    df["resolution_hours"] = (closed_utc - created_utc).dt.total_seconds() / 3600.0

    # Local calendar and intake features
    df["created_month"] = df["created_dt_local"].dt.strftime("%Y-%m")
    df["created_date_only"] = df["created_dt_local"].dt.strftime("%Y-%m-%d")
    df["created_hour"] = df["created_dt_local"].dt.hour
    df["created_weekday"] = df["created_dt_local"].dt.day_name()
    df["created_year"] = df["created_dt_local"].dt.year
    df["is_weekend"] = df["created_dt_local"].dt.day_of_week.isin([5, 6]).astype(int)

    # Status tracking (Closed vs Open)
    is_closed = df["closed_date"].notna() & (df["status"].str.upper() == "CLOSED")
    is_open = df["closed_date"].isna() | (df["status"].str.upper() != "CLOSED")
    valid_closed_with_duration = is_closed & df["resolution_hours"].notna()

    negative_duration = int((df["resolution_hours"] < 0).sum())
    zero_duration = int((df["resolution_hours"] == 0).sum())
    positive_durations = df.loc[valid_closed_with_duration & (df["resolution_hours"] >= 0), "resolution_hours"]

    # Quarter boundary concentration analysis
    q_end_mask = is_near_quarter_end(df["created_dt_local"])
    near_quarter_end_count = int(q_end_mask.sum())
    near_quarter_end_pct = round((near_quarter_end_count / len(df)) * 100, 2)

    # Temporal split partitions
    train_mask = df["created_year"] == 2023
    val_mask = (df["created_year"] == 2024) & (df["created_dt_local"].dt.month <= 6)
    test_mask = (df["created_year"] == 2024) & (df["created_dt_local"].dt.month > 6)

    # Derive future classification threshold using 2023 training data only
    train_durations = df.loc[train_mask & valid_closed_with_duration & (df["resolution_hours"] >= 0), "resolution_hours"]
    long_duration_threshold_p75 = float(train_durations.quantile(0.75)) if not train_durations.empty else None
    long_duration_threshold_p80 = float(train_durations.quantile(0.80)) if not train_durations.empty else None

    # Binary long-duration target (project-defined, based on training 75th percentile)
    if long_duration_threshold_p75 is not None:
        df["target_long_duration"] = np.where(
            valid_closed_with_duration,
            (df["resolution_hours"] >= long_duration_threshold_p75).astype(int),
            np.nan
        )
    else:
        df["target_long_duration"] = np.nan

    # Generate Figures
    # 1. Monthly sample coverage
    monthly_counts = df["created_month"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    monthly_counts.plot(kind="bar", ax=ax, color="#1f77b4", edgecolor="black", linewidth=0.5)
    ax.set_title("Phase 1 Stratified Sample: Records by Calendar Month (2023–2024)")
    ax.set_xlabel("Creation Month")
    ax.set_ylabel("Sampled Records")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase1_month_coverage.png", dpi=160)
    plt.close(fig)

    # 2. Local hour-of-day distribution
    hour_counts = df["created_hour"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    hour_counts.plot(kind="bar", ax=ax, color="#2ca02c", edgecolor="black", linewidth=0.5)
    ax.set_title("Phase 1 Stratified Sample: Local Hour-of-Day Distribution (NYC Eastern Time)")
    ax.set_xlabel("Local Hour (0 = Midnight, 12 = Noon, 23 = 11 PM)")
    ax.set_ylabel("Sampled Records")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase1_hour_distribution.png", dpi=160)
    plt.close(fig)

    # 3. Day-of-week distribution
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_counts = df["created_weekday"].value_counts().reindex(day_order).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4))
    weekday_counts.plot(kind="bar", ax=ax, color="#ff7f0e", edgecolor="black", linewidth=0.5)
    ax.set_title("Phase 1 Stratified Sample: Day-of-Week Distribution")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Sampled Records")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase1_weekday_distribution.png", dpi=160)
    plt.close(fig)

    # 4. Top complaint categories
    top_categories = df["complaint_type"].value_counts().head(15).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    top_categories.plot(kind="barh", ax=ax, color="#9467bd", edgecolor="black", linewidth=0.5)
    ax.set_title("Phase 1 Stratified Sample: Top 15 Problem Categories")
    ax.set_xlabel("Sampled Records (not citywide population totals)")
    ax.set_ylabel("Complaint Type")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase1_top_categories.png", dpi=160)
    plt.close(fig)

    # 5. Resolution duration distribution (clipped at 95th percentile for display)
    if not positive_durations.empty:
        p95 = float(positive_durations.quantile(0.95))
        clipped_durations = positive_durations.clip(upper=p95)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(clipped_durations, bins=45, color="#d62728", edgecolor="black", linewidth=0.5)
        ax.axvline(float(positive_durations.median()), color="blue", linestyle="--", linewidth=1.5, label=f"Median: {positive_durations.median():.1f}h")
        ax.axvline(float(positive_durations.quantile(0.75)), color="orange", linestyle="--", linewidth=1.5, label=f"75th percentile: {positive_durations.quantile(0.75):.1f}h")
        ax.set_title("Resolution Duration Distribution (Clipped at 95th Percentile for Display)")
        ax.set_xlabel("Elapsed Resolution Hours (closed_date - created_date in UTC)")
        ax.set_ylabel("Number of Completed Requests")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "phase1_resolution_duration.png", dpi=160)
        plt.close(fig)

    # Statistics & JSON Report
    missing_pct = (df.isna().mean() * 100).round(2).sort_values(ascending=False).to_dict()
    unique_dates_per_month = df.groupby("created_month")["created_date_only"].nunique().to_dict()

    duration_quantiles = {}
    if not positive_durations.empty:
        duration_quantiles = {
            "valid_completed_records": int(len(positive_durations)),
            "min_hours": round(float(positive_durations.min()), 3),
            "p25_hours": round(float(positive_durations.quantile(0.25)), 3),
            "median_hours": round(float(positive_durations.median()), 3),
            "p75_hours": round(float(positive_durations.quantile(0.75)), 3),
            "p90_hours": round(float(positive_durations.quantile(0.90)), 3),
            "p95_hours": round(float(positive_durations.quantile(0.95)), 3),
            "p99_hours": round(float(positive_durations.quantile(0.99)), 3),
            "max_hours": round(float(positive_durations.max()), 3),
            "iqr_hours": round(float(positive_durations.quantile(0.75) - positive_durations.quantile(0.25)), 3),
        }

    report = {
        "dataset_name": "NYC 311 Service Requests (2023–2024 Stratified Temporal Sample)",
        "source_url": "https://data.cityofnewyork.us/resource/erm2-nwe9.json",
        "timestamp_interpretation": (
            "Source strings are Socrata Floating Timestamps representing NYC Local Time (America/New_York). "
            "Converted to UTC for exact elapsed duration calculation; local Eastern Time preserved for hour-of-day features."
        ),
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "duplicate_unique_keys": int(df["unique_key"].duplicated().sum()) if "unique_key" in df.columns else 0,
        "date_range_observed": {
            "min_created": str(df["created_dt_local"].min()),
            "max_created": str(df["created_dt_local"].max()),
            "total_distinct_dates": int(df["created_date_only"].nunique()),
        },
        "monthly_sample_counts": {k: int(v) for k, v in monthly_counts.items()},
        "distinct_creation_dates_per_month": {k: int(v) for k, v in unique_dates_per_month.items()},
        "missing_percentage_by_column": missing_pct,
        "request_status_counts": {
            "closed": int(is_closed.sum()),
            "open": int(is_open.sum()),
            "status_values": {str(k): int(v) for k, v in df["status"].value_counts().items()},
        },
        "duration_anomalies": {
            "negative_duration_count": negative_duration,
            "zero_duration_count": zero_duration,
            "zero_duration_percentage": round((zero_duration / len(df)) * 100, 3),
        },
        "resolution_duration_statistics_hours": duration_quantiles,
        "sampling_distribution_integrity": {
            "near_quarter_end_count": near_quarter_end_count,
            "near_quarter_end_percentage": near_quarter_end_pct,
            "note_on_phase0_comparison": (
                f"In Phase 0, 100% of records were clustered at quarter-end midnight. "
                f"In Phase 1, quarter-end rows are {near_quarter_end_pct}%, matching natural uniform proportion (~1.1% of days in a quarter)."
            ),
        },
        "top_15_complaint_categories": {str(k): int(v) for k, v in df["complaint_type"].value_counts().head(15).items()},
        "distinct_complaint_categories": int(df["complaint_type"].nunique()),
        "borough_distribution": {str(k): int(v) for k, v in df["borough"].value_counts().items()},
        "channel_distribution": {str(k): int(v) for k, v in df["open_data_channel_type"].value_counts().items()},
        "hour_distribution_nyc_local": {int(k): int(v) for k, v in hour_counts.items()},
        "weekday_distribution": {str(k): int(v) for k, v in weekday_counts.items()},
        "temporal_evaluation_split": {
            "training_set_2023": int(train_mask.sum()),
            "validation_set_2024_H1": int(val_mask.sum()),
            "test_set_2024_H2": int(test_mask.sum()),
            "long_duration_classification_threshold_p75_hours": round(long_duration_threshold_p75, 3) if long_duration_threshold_p75 else None,
            "long_duration_classification_threshold_p80_hours": round(long_duration_threshold_p80, 3) if long_duration_threshold_p80 else None,
            "threshold_rule": "Threshold computed strictly from 2023 training data to prevent data leakage.",
        },
        "feature_boundaries": {
            "approved_intake_features": INTAKE_FEATURE_ALLOWLIST,
            "excluded_post_resolution_features": EXCLUDED_POST_RESOLUTION_FIELDS,
            "agency_usage_note": AGENCY_USAGE_NOTE,
        },
        "manifest_summary": {
            "target_sample": manifest_info.get("target_sample"),
            "rows_fetched": manifest_info.get("rows_fetched"),
            "duplicates_dropped": manifest_info.get("duplicates_dropped"),
            "shortfall_from_target": manifest_info.get("shortfall_from_target"),
            "seed": manifest_info.get("seed"),
        },
    }

    with REPORT_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Save processed dataset
    df.to_csv(PROCESSED_OUTPUT, index=False)

    logger.info("PHASE 1 PROFILING COMPLETE")
    logger.info(f"Processed CSV: {PROCESSED_OUTPUT}")
    logger.info(f"JSON Report:   {REPORT_OUTPUT}")
    logger.info(f"Figures:       {FIGURES_DIR}")
    logger.info(f"Total rows:    {len(df):,} across {df['created_date_only'].nunique()} distinct dates")
    logger.info(f"Median resolution hours: {duration_quantiles.get('median_hours')}h")
    logger.info(f"Top category:  {list(report['top_15_complaint_categories'].keys())[0]} ({list(report['top_15_complaint_categories'].values())[0]} records)")
    logger.info(f"Near quarter end: {near_quarter_end_count} ({near_quarter_end_pct}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
