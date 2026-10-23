"""Fetch a bounded feasibility sample from NYC Open Data's public 311 API.

This is a preliminary dataset check, not the final modelling dataset. The sample
is bounded by quarter to avoid downloading the full multi-million-row table.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUTFILE = ROOT / "data" / "raw" / "nyc_311_preliminary_sample.csv"
API_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
FIELDS = [
    "unique_key",
    "created_date",
    "closed_date",
    "agency",
    "agency_name",
    "complaint_type",
    "descriptor",
    "status",
    "due_date",
    "resolution_description",
    "borough",
    "incident_zip",
    "latitude",
    "longitude",
    "open_data_channel_type",
]


def quarter_windows(start_year: int, end_year: int):
    for year in range(start_year, end_year + 1):
        for month in (1, 4, 7, 10):
            start = pd.Timestamp(year=year, month=month, day=1)
            if month == 10:
                end = pd.Timestamp(year=year + 1, month=1, day=1)
            else:
                end = pd.Timestamp(year=year, month=month + 3, day=1)
            yield start, end


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--per-quarter", type=int, default=1000,
                        help="Maximum rows fetched per quarter; default 1000.")
    parser.add_argument("--output", type=Path, default=OUTFILE)
    args = parser.parse_args()

    if args.start_year > args.end_year:
        parser.error("--start-year must be <= --end-year")
    if args.per_quarter < 1 or args.per_quarter > 5000:
        parser.error("--per-quarter must be between 1 and 5000")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_frames: list[pd.DataFrame] = []
    total_windows = (args.end_year - args.start_year + 1) * 4
    print(f"Fetching a bounded sample from NYC 311 public data ({total_windows} quarters).")
    print("This does NOT download the full dataset.")

    with requests.Session() as session:
        session.headers.update({"User-Agent": "CivicResolveAI-student-project/0.1 (public-data research)"})
        for i, (start, end) in enumerate(quarter_windows(args.start_year, args.end_year), start=1):
            where = (
                f"created_date >= '{start.strftime('%Y-%m-%dT%H:%M:%S')}' "
                f"AND created_date < '{end.strftime('%Y-%m-%dT%H:%M:%S')}' "
                "AND closed_date IS NOT NULL AND complaint_type IS NOT NULL"
            )
            params = {
                "$select": ",".join(FIELDS),
                "$where": where,
                "$order": "created_date DESC",
                "$limit": args.per_quarter,
            }
            try:
                response = session.get(API_URL, params=params, timeout=90)
                response.raise_for_status()
                rows = response.json()
                if not isinstance(rows, list):
                    raise ValueError(f"Expected a list of records from Socrata API, got {type(rows).__name__}: {str(rows)[:200]}")
            except (requests.RequestException, ValueError) as exc:
                print(f"ERROR in window {start.date()} to {end.date()}: {exc}", file=sys.stderr)
                print("No model work has been done. Check internet/API availability and rerun.", file=sys.stderr)
                return 2

            frame = pd.DataFrame(rows)
            if not frame.empty:
                all_frames.append(frame)
            print(f"[{i:02d}/{total_windows}] {start:%Y-%m} to {end:%Y-%m}: {len(frame):,} rows")
            time.sleep(0.15)  # be polite to the public API

    if not all_frames:
        print("No rows were returned. Check the API and date window.", file=sys.stderr)
        return 3

    data = pd.concat(all_frames, ignore_index=True)
    for col in FIELDS:
        if col not in data.columns:
            data[col] = pd.NA
    data = data[FIELDS]
    before = len(data)
    if "unique_key" in data.columns:
        data = data.drop_duplicates(subset=["unique_key"])
    data.to_csv(args.output, index=False)
    print("\nPRELIMINARY SAMPLE SAVED")
    print(f"Rows fetched:       {before:,}")
    print(f"Rows after dedupe:  {len(data):,}")
    print(f"Output:             {args.output}")
    print("Important: quarterly rows are ordered API samples, not a random/representative sample.")
    print("Use them for schema/data-quality feasibility only; design a better temporal sampling plan before final model claims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
