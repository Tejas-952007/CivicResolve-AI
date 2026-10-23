"""Profile the preliminary 311 sample and create first diagnostic charts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "raw" / "nyc_311_preliminary_sample.csv"
PROCESSED = ROOT / "data" / "processed" / "nyc_311_preliminary_profiled.csv"
REPORT = ROOT / "outputs" / "data_quality_report.json"
FIGURES = ROOT / "outputs" / "figures"


def main() -> int:
    if not INPUT.exists():
        print(f"Dataset sample not found: {INPUT}")
        print("Run first: python scripts/01_fetch_sample.py")
        return 2

    df = pd.read_csv(INPUT, low_memory=False)
    required = {"created_date", "closed_date", "complaint_type"}
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        print(f"Required columns missing from source response: {missing_columns}")
        return 3

    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce", utc=True)
    df["closed_date"] = pd.to_datetime(df["closed_date"], errors="coerce", utc=True)
    df["resolution_hours"] = (df["closed_date"] - df["created_date"]).dt.total_seconds() / 3600
    df["created_month"] = df["created_date"].dt.strftime("%Y-%m")

    valid_dates = df["created_date"].notna() & df["closed_date"].notna()
    negative_duration = int((df["resolution_hours"] < 0).sum())
    zero_duration = int((df["resolution_hours"] == 0).sum())
    positive = df.loc[df["resolution_hours"] >= 0, "resolution_hours"].dropna()

    FIGURES.mkdir(parents=True, exist_ok=True)
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    # Chart 1: top complaint/problem categories
    top = df["complaint_type"].fillna("Unknown").value_counts().head(15).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    top.plot(kind="barh", ax=ax)
    ax.set_title("Preliminary Sample: Top Problem Categories")
    ax.set_xlabel("Records in downloaded sample (not citywide counts)")
    ax.set_ylabel("Problem category")
    fig.tight_layout()
    fig.savefig(FIGURES / "01_top_problem_categories.png", dpi=160)
    plt.close(fig)

    # Chart 2: duration histogram with the far tail clipped for readability.
    if not positive.empty:
        cap = float(positive.quantile(0.95))
        plot_values = positive.clip(upper=cap)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(plot_values, bins=40)
        ax.set_title("Resolution Time Distribution (95th percentile clipped for display)")
        ax.set_xlabel("Resolution hours")
        ax.set_ylabel("Records")
        fig.tight_layout()
        fig.savefig(FIGURES / "02_resolution_hours.png", dpi=160)
        plt.close(fig)

    # Chart 3: quarterly sample counts; this indicates what the fetcher returned,
    # not actual request volume, because each quarter has a fixed fetch cap.
    if df["created_date"].notna().any():
        monthly = df.dropna(subset=["created_date"]).set_index("created_date").resample("MS").size()
        fig, ax = plt.subplots(figsize=(11, 4))
        monthly.plot(ax=ax)
        ax.set_title("Rows in Preliminary Download by Month (sample diagnostics only)")
        ax.set_xlabel("Created month")
        ax.set_ylabel("Downloaded sample rows")
        fig.tight_layout()
        fig.savefig(FIGURES / "03_sample_rows_by_month.png", dpi=160)
        plt.close(fig)

    missing_pct = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
    duration_stats = {}
    if not positive.empty:
        duration_stats = {
            "count_nonnegative": int(positive.shape[0]),
            "min_hours": round(float(positive.min()), 3),
            "median_hours": round(float(positive.median()), 3),
            "p90_hours": round(float(positive.quantile(0.90)), 3),
            "p95_hours": round(float(positive.quantile(0.95)), 3),
            "max_hours": round(float(positive.max()), 3),
        }

    report = {
        "dataset_source": "NYC Open Data: 311 Service Requests from 2020 to Present",
        "source_url": "https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9/about_data",
        "purpose": "Preliminary schema and data-quality feasibility check only; not a representative model dataset.",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_unique_key_rows": int(df["unique_key"].duplicated().sum()) if "unique_key" in df.columns else None,
        "created_date_min": str(df["created_date"].min()),
        "created_date_max": str(df["created_date"].max()),
        "valid_created_and_closed_dates": int(valid_dates.sum()),
        "missing_percent_by_column": {str(k): float(v) for k, v in missing_pct.items()},
        "negative_duration_rows": negative_duration,
        "zero_duration_rows": zero_duration,
        "duration_statistics_hours_excluding_negative": duration_stats,
        "top_15_problem_categories_in_sample": {str(k): int(v) for k, v in df["complaint_type"].value_counts().head(15).items()},
        "distinct_problem_categories_in_sample": int(df["complaint_type"].nunique(dropna=True)),
        "notes": [
            "Sample download is capped by quarter and ordered by created_date; it is not a random sample.",
            "Do not interpret sample category counts as citywide demand shares.",
            "Resolution duration is an observed historical outcome, not a guaranteed SLA or a causal measure of service quality.",
            "Decide the final feature set and temporally separated evaluation strategy after inspecting this report.",
        ],
    }
    with REPORT.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    df.to_csv(PROCESSED, index=False)

    print("DATA PROFILE COMPLETE")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    print(f"Created range: {df['created_date'].min()} to {df['created_date'].max()}")
    print(f"Negative-duration rows: {negative_duration} | Zero-duration rows: {zero_duration}")
    if duration_stats:
        print(f"Median resolution hours (non-negative durations): {duration_stats['median_hours']}")
        print(f"90th percentile hours: {duration_stats['p90_hours']}")
    print(f"Report: {REPORT}")
    print(f"Processed sample: {PROCESSED}")
    print(f"Charts: {FIGURES}")
    print("IMPORTANT: downloaded counts are sample counts, not official citywide demand estimates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
