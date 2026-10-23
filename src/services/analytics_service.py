"""Analytics service for descriptive municipal complaint intelligence.

Calculates KPIs, temporal trends, categorical distributions, and borough comparisons
strictly from the local sampled NYC 311 dataset.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "nyc_311_phase1_profiled.csv"
LONG_DURATION_THRESHOLD_HOURS = 124.925

_DATASET_CACHE: pd.DataFrame | None = None


def get_dataset() -> pd.DataFrame:
    """Load and cache the profiled Phase 1 dataset."""
    global _DATASET_CACHE
    if _DATASET_CACHE is None:
        if not DATASET_PATH.exists():
            raise FileNotFoundError(f"Processed dataset not found at {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH, low_memory=False)
        df["created_dt"] = pd.to_datetime(df["created_date_only"])
        _DATASET_CACHE = df
    return _DATASET_CACHE


def get_filter_options() -> dict[str, Any]:
    """Retrieve unique boroughs, top complaint types, and date bounds for filter controls."""
    df = get_dataset()
    boroughs = sorted([b for b in df["borough"].dropna().unique().tolist() if b != "Unspecified"])
    boroughs.append("Unspecified")
    
    top_complaint_types = df["complaint_type"].value_counts().head(25).index.tolist()
    
    min_date = str(df["created_date_only"].min())
    max_date = str(df["created_date_only"].max())
    
    return {
        "boroughs": boroughs,
        "complaint_types": top_complaint_types,
        "min_date": min_date,
        "max_date": max_date,
    }


def filter_dataset(
    df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    borough: str | None = None,
    complaint_type: str | None = None,
) -> pd.DataFrame:
    """Apply user filters safely to the dataset."""
    filtered = df.copy()
    
    if start_date:
        filtered = filtered[filtered["created_date_only"] >= start_date]
    if end_date:
        filtered = filtered[filtered["created_date_only"] <= end_date]
    if borough and borough != "ALL":
        filtered = filtered[filtered["borough"] == borough]
    if complaint_type and complaint_type != "ALL":
        filtered = filtered[filtered["complaint_type"] == complaint_type]
        
    return filtered


def compute_kpis(df: pd.DataFrame) -> dict[str, Any]:
    """Compute summary KPIs strictly from the filtered slice."""
    total_records = len(df)
    if total_records == 0:
        return {
            "total_records": 0,
            "open_count": 0,
            "closed_count": 0,
            "eligible_count": 0,
            "median_resolution_hours": 0.0,
            "median_resolution_days": 0.0,
            "long_duration_count": 0,
            "long_duration_share": 0.0,
            "zero_duration_count": 0,
            "date_range": "No records in filter",
        }
        
    is_closed = (df["status"] == "Closed") & df["closed_date"].notna()
    closed_count = int(is_closed.sum())
    open_count = int(total_records - closed_count)
    
    # Eligible closed records with valid non-negative resolution hours
    eligible_mask = is_closed & df["resolution_hours"].notna() & (df["resolution_hours"] >= 0)
    eligible_df = df[eligible_mask]
    eligible_count = len(eligible_df)
    
    if eligible_count > 0:
        median_hours = float(eligible_df["resolution_hours"].median())
        median_days = round(median_hours / 24.0, 1)
        long_count = int((eligible_df["resolution_hours"] > LONG_DURATION_THRESHOLD_HOURS).sum())
        long_share = round((long_count / eligible_count) * 100.0, 1)
        zero_count = int((eligible_df["resolution_hours"] == 0.0).sum())
    else:
        median_hours = 0.0
        median_days = 0.0
        long_count = 0
        long_share = 0.0
        zero_count = 0
        
    min_date = str(df["created_date_only"].min())
    max_date = str(df["created_date_only"].max())
    
    return {
        "total_records": total_records,
        "open_count": open_count,
        "closed_count": closed_count,
        "eligible_count": eligible_count,
        "median_resolution_hours": round(median_hours, 1),
        "median_resolution_days": median_days,
        "long_duration_count": long_count,
        "long_duration_share": long_share,
        "zero_duration_count": zero_count,
        "date_range": f"{min_date} to {max_date}",
    }


def get_monthly_trends(df: pd.DataFrame) -> dict[str, list]:
    """Calculate monthly complaint volumes and median resolution times."""
    if len(df) == 0:
        return {"months": [], "counts": [], "median_durations": []}
        
    temp = df.copy()
    temp["month_period"] = pd.to_datetime(temp["created_date_only"]).dt.to_period("M").astype(str)
    
    # Monthly volume
    vol = temp.groupby("month_period")["unique_key"].count()
    
    # Monthly median resolution duration for eligible records
    eligible = temp[(temp["status"] == "Closed") & (temp["resolution_hours"] >= 0)]
    if len(eligible) > 0:
        dur = eligible.groupby("month_period")["resolution_hours"].median()
    else:
        dur = pd.Series(dtype=float)
        
    months = sorted(vol.index.tolist())
    counts = [int(vol.get(m, 0)) for m in months]
    median_durations = [round(float(dur.get(m, 0.0)), 1) if pd.notna(dur.get(m)) else 0.0 for m in months]
    
    return {
        "months": months,
        "counts": counts,
        "median_durations": median_durations,
    }


def get_top_categories(df: pd.DataFrame, limit: int = 10) -> dict[str, list]:
    """Retrieve top complaint categories by volume with associated resolution duration."""
    if len(df) == 0:
        return {"categories": [], "counts": [], "median_durations": []}
        
    vol = df["complaint_type"].value_counts().head(limit)
    categories = vol.index.tolist()
    counts = [int(v) for v in vol.values]
    
    eligible = df[(df["status"] == "Closed") & (df["resolution_hours"] >= 0)]
    dur_map = eligible.groupby("complaint_type")["resolution_hours"].median().to_dict()
    median_durations = [round(float(dur_map.get(cat, 0.0)), 1) for cat in categories]
    
    return {
        "categories": categories,
        "counts": counts,
        "median_durations": median_durations,
    }


def get_borough_breakdown(df: pd.DataFrame) -> dict[str, list]:
    """Aggregate complaint volume and long-duration delay rate across boroughs."""
    if len(df) == 0:
        return {"boroughs": [], "counts": [], "long_rate_pct": [], "median_hours": []}
        
    vol = df["borough"].value_counts()
    boroughs = vol.index.tolist()
    counts = [int(v) for v in vol.values]
    
    eligible = df[(df["status"] == "Closed") & (df["resolution_hours"] >= 0)]
    
    long_rate = []
    median_dur = []
    for b in boroughs:
        b_el = eligible[eligible["borough"] == b]
        if len(b_el) > 0:
            rate = (b_el["resolution_hours"] > LONG_DURATION_THRESHOLD_HOURS).mean() * 100.0
            med = b_el["resolution_hours"].median()
            long_rate.append(round(float(rate), 1))
            median_dur.append(round(float(med), 1))
        else:
            long_rate.append(0.0)
            median_dur.append(0.0)
            
    return {
        "boroughs": boroughs,
        "counts": counts,
        "long_rate_pct": long_rate,
        "median_hours": median_dur,
    }


def get_duration_distribution(df: pd.DataFrame) -> dict[str, list]:
    """Categorize elapsed resolution times into intuitive operational bins."""
    eligible = df[(df["status"] == "Closed") & (df["resolution_hours"] >= 0)]
    if len(eligible) == 0:
        return {"bins": [], "counts": []}
        
    hours = eligible["resolution_hours"]
    
    bins = [
        "0h (Immediate)",
        ">0 to 12h",
        "12 to 24h",
        "24 to 72h (1-3d)",
        "72 to 125h (3-5d)",
        "125 to 500h (~3w)",
        "> 500h (> 3w)",
    ]
    
    counts = [
        int((hours == 0).sum()),
        int(((hours > 0) & (hours <= 12)).sum()),
        int(((hours > 12) & (hours <= 24)).sum()),
        int(((hours > 24) & (hours <= 72)).sum()),
        int(((hours > 72) & (hours <= 124.925)).sum()),
        int(((hours > 124.925) & (hours <= 500)).sum()),
        int((hours > 500).sum()),
    ]
    
    return {"bins": bins, "counts": counts}
