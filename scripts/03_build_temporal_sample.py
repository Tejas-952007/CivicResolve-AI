"""Reproducible stratified temporal sampling from NYC 311 Open Data API.

Builds a temporally distributed dataset across 2023-2024 by:
1. Querying exact monthly population counts across 24 calendar months.
2. Allocating sampling quotas proportionally using the Largest Remainder (Hamilton-Hare) method.
3. Generating stratified offsets across ordered row ranks within each month using a fixed seed.
4. Fetching small bounded chunks via Socrata SODA 2 pagination ($limit, $offset, $order).
5. Handling transient network errors and rate limits with exponential backoff and request checkpointing.
6. Preserving open and closed complaints without pre-filtering on outcome fields.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_OUTPUT = ROOT / "data" / "raw" / "nyc_311_temporal_sample.csv"
DEFAULT_MANIFEST = ROOT / "outputs" / "sampling_manifest.json"
CHECKPOINT_CSV = ROOT / "data" / "raw" / "nyc_311_temporal_sample_checkpoint.csv"

API_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"

# Strict intake and observation fields (excluding sparse due_date and personal address lines)
FIELDS = [
    "unique_key",
    "created_date",
    "closed_date",
    "complaint_type",
    "descriptor",
    "agency",
    "agency_name",
    "borough",
    "open_data_channel_type",
    "incident_zip",
    "latitude",
    "longitude",
    "status",
    "resolution_description",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("build_temporal_sample")


def get_monthly_windows(start_date: str, end_date: str) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Generate calendar month boundaries (inclusive start, exclusive end)."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    windows: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    
    current = pd.Timestamp(year=start.year, month=start.month, day=1)
    end_boundary = end + pd.Timedelta(days=1)
    
    while current < end_boundary:
        label = current.strftime("%Y-%m")
        # Next month start
        if current.month == 12:
            next_month = pd.Timestamp(year=current.year + 1, month=1, day=1)
        else:
            next_month = pd.Timestamp(year=current.year, month=current.month + 1, day=1)
        windows.append((label, current, next_month))
        current = next_month
    return windows


def fetch_population_count(
    session: requests.Session,
    api_url: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    timeout: int = 45,
    max_retries: int = 4,
) -> int:
    """Fetch exact total count of records created in the date window."""
    start_str = start_ts.strftime("%Y-%m-%dT00:00:00")
    end_str = end_ts.strftime("%Y-%m-%dT00:00:00")
    where_clause = f"created_date >= '{start_str}' AND created_date < '{end_str}'"
    params = {"$select": "count(*)", "$where": where_clause}

    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(api_url, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait_time = 2.0 ** attempt
                logger.warning(f"Rate limited (429) fetching count for {start_ts.strftime('%Y-%m')}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, list) and len(payload) > 0 and "count" in payload[0]:
                return int(payload[0]["count"])
            raise ValueError(f"Unexpected count payload: {payload}")
        except (requests.RequestException, ValueError) as exc:
            if attempt == max_retries:
                logger.error(f"Failed to fetch count for window {start_str} to {end_str}: {exc}")
                raise
            wait_time = 1.5 ** attempt
            time.sleep(wait_time)
    raise RuntimeError("Exhausted retries in fetch_population_count")


def allocate_quotas(populations: dict[str, int], total_target: int) -> dict[str, int]:
    """Allocate integer sample quotas proportionally using Largest Remainder (Hamilton-Hare).
    
    Guarantees sum(quotas.values()) == total_target exactly and deterministically.
    """
    total_pop = sum(populations.values())
    if total_pop == 0 or total_target <= 0:
        return {k: 0 for k in populations}

    exact_quotas = {k: (v / total_pop) * total_target for k, v in populations.items()}
    base_quotas = {k: math.floor(q) for k, q in exact_quotas.items()}
    remainders = {k: exact_quotas[k] - base_quotas[k] for k in populations}

    unallocated = total_target - sum(base_quotas.values())
    # Sort by remainder descending, tie-break by month label ascending
    sorted_by_remainder = sorted(remainders.items(), key=lambda item: (-item[1], item[0]))
    for i in range(unallocated):
        month = sorted_by_remainder[i][0]
        base_quotas[month] += 1

    assert sum(base_quotas.values()) == total_target, "Allocated quota sum mismatch!"
    return base_quotas


def generate_strata_offsets(
    population: int,
    quota: int,
    chunk_size: int,
    seed: int,
    month_index: int,
) -> list[tuple[int, int]]:
    """Divide monthly population into strata and select reproducible offset positions.
    
    Returns list of (offset, limit) tuples spanning the monthly rank population.
    """
    if quota <= 0 or population <= 0:
        return []

    num_chunks = math.ceil(quota / chunk_size)
    max_rank = max(0, population - chunk_size)
    rng = np.random.RandomState(seed + month_index * 1000)

    offsets: list[tuple[int, int]] = []
    remaining_quota = quota

    for k in range(num_chunks):
        limit = min(chunk_size, remaining_quota)
        if num_chunks == 1 or max_rank == 0:
            offset = 0
        else:
            stratum_start = int((k / num_chunks) * max_rank)
            stratum_end = int(((k + 1) / num_chunks) * max_rank)
            if stratum_end > stratum_start:
                offset = int(rng.randint(stratum_start, stratum_end))
            else:
                offset = stratum_start
        offsets.append((offset, limit))
        remaining_quota -= limit

    return offsets


def fetch_chunk(
    session: requests.Session,
    api_url: str,
    fields: list[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    offset: int,
    limit: int,
    timeout: int = 45,
    max_retries: int = 4,
) -> list[dict[str, Any]]:
    """Fetch a bounded chunk of records using SODA 2 pagination with exponential backoff."""
    start_str = start_ts.strftime("%Y-%m-%dT00:00:00")
    end_str = end_ts.strftime("%Y-%m-%dT00:00:00")
    where_clause = f"created_date >= '{start_str}' AND created_date < '{end_str}'"
    params = {
        "$select": ",".join(fields),
        "$where": where_clause,
        "$order": "created_date ASC, unique_key ASC",
        "$limit": limit,
        "$offset": offset,
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(api_url, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait_time = 2.0 ** attempt
                logger.warning(f"Rate limited (429) at offset {offset}. Backing off {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, list):
                raise ValueError(f"Expected list of records, got {type(payload).__name__}: {str(payload)[:200]}")
            return payload
        except (requests.RequestException, ValueError) as exc:
            if attempt == max_retries:
                logger.error(f"Error fetching chunk (offset {offset}, limit {limit}): {exc}")
                raise
            wait_time = 1.5 ** attempt
            time.sleep(wait_time)
    return []


def save_manifest(manifest_path: Path, data: dict[str, Any]) -> None:
    """Save sampling metadata manifest atomically."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp_path.replace(manifest_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=str, default="2023-01-01", help="Inclusive start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default="2024-12-31", help="Inclusive end date YYYY-MM-DD")
    parser.add_argument("--target-sample", type=int, default=10000, help="Total target records to sample")
    parser.add_argument("--chunk-size", type=int, default=25, help="Rows per API request chunk (default: 25)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible stratum offset selection")
    parser.add_argument("--output", type=Path, default=DEFAULT_RAW_OUTPUT, help="Path for output raw CSV")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Path for output JSON manifest")
    parser.add_argument("--timeout", type=int, default=45, help="HTTP request timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=4, help="Max retries for HTTP requests")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    windows = get_monthly_windows(args.start_date, args.end_date)
    logger.info(f"Targeting {len(windows)} months from {args.start_date} to {args.end_date}.")
    logger.info(f"Target total sample: {args.target_sample:,} records | Chunk size: {args.chunk_size} | Seed: {args.seed}")

    session = requests.Session()
    session.headers.update({"User-Agent": "CivicResolveAI-student-project/1.0 (academic-research)"})

    # Step 1: Monthly population counts
    logger.info("Step 1: Inquiring monthly population counts...")
    populations: dict[str, int] = {}
    for month_label, start_ts, end_ts in windows:
        count = fetch_population_count(session, API_URL, start_ts, end_ts, timeout=args.timeout, max_retries=args.max_retries)
        populations[month_label] = count
        time.sleep(0.05)
    total_pop = sum(populations.values())
    logger.info(f"Observed total 2-year population: {total_pop:,} records across {len(populations)} months.")

    # Step 2: Quota allocation
    quotas = allocate_quotas(populations, args.target_sample)
    logger.info(f"Deterministic quotas allocated across 24 months. Min quota: {min(quotas.values())}, Max: {max(quotas.values())}.")

    # Step 3: Plan strata offsets
    planned_requests: list[dict[str, Any]] = []
    for month_idx, (month_label, start_ts, end_ts) in enumerate(windows):
        m_pop = populations[month_label]
        m_quota = quotas[month_label]
        offsets = generate_strata_offsets(m_pop, m_quota, args.chunk_size, args.seed, month_idx)
        for offset, limit in offsets:
            planned_requests.append({
                "month": month_label,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "offset": offset,
                "limit": limit,
            })
    logger.info(f"Total planned API requests: {len(planned_requests)} across {len(windows)} months.")

    # Checkpoint / resume handling
    completed_chunks = 0
    all_rows: list[dict[str, Any]] = []
    resumed = False

    if args.resume and CHECKPOINT_CSV.exists() and args.manifest.exists():
        try:
            with args.manifest.open("r", encoding="utf-8") as f:
                prev_manifest = json.load(f)
            if prev_manifest.get("seed") == args.seed and prev_manifest.get("target_sample") == args.target_sample:
                checkpoint_df = pd.read_csv(CHECKPOINT_CSV, low_memory=False)
                all_rows = checkpoint_df.to_dict(orient="records")
                completed_chunks = prev_manifest.get("completed_chunks", 0)
                resumed = True
                logger.info(f"Resumed from checkpoint: {len(all_rows):,} rows loaded ({completed_chunks} chunks).")
        except Exception as exc:
            logger.warning(f"Could not load checkpoint: {exc}. Starting fresh.")
            all_rows = []
            completed_chunks = 0

    manifest_data: dict[str, Any] = {
        "dataset_source": "NYC Open Data: 311 Service Requests from 2020 to Present",
        "api_endpoint": API_URL,
        "execution_timestamp": datetime.now().isoformat(),
        "date_range": {"start": args.start_date, "end": args.end_date},
        "target_sample": args.target_sample,
        "chunk_size": args.chunk_size,
        "seed": args.seed,
        "total_population": total_pop,
        "monthly_populations": populations,
        "monthly_quotas": quotas,
        "total_planned_requests": len(planned_requests),
        "timestamp_interpretation": (
            "Source timestamps are Socrata Floating Timestamps in publisher local time "
            "(America/New_York / US Eastern Time). Raw strings are preserved in the raw CSV. "
            "For duration calculations, local times are converted to UTC to account for DST shifts. "
            "For calendar and hour-of-day features, local Eastern Time is maintained."
        ),
        "fields_requested": FIELDS,
        "completed_chunks": completed_chunks,
        "status": "IN_PROGRESS",
    }
    save_manifest(args.manifest, manifest_data)

    # Step 4: Execute chunk requests
    logger.info("Step 4: Executing stratified chunk sampling...")
    t_start = time.time()
    
    for idx in range(completed_chunks, len(planned_requests)):
        req = planned_requests[idx]
        rows = fetch_chunk(
            session=session,
            api_url=API_URL,
            fields=FIELDS,
            start_ts=req["start_ts"],
            end_ts=req["end_ts"],
            offset=req["offset"],
            limit=req["limit"],
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        all_rows.extend(rows)
        completed_chunks += 1

        if (idx + 1) % 25 == 0 or (idx + 1) == len(planned_requests):
            elapsed = time.time() - t_start
            logger.info(f"[{idx + 1:03d}/{len(planned_requests)}] Month {req['month']} | offset {req['offset']:,} | Total rows: {len(all_rows):,} ({elapsed:.1f}s)")
            # Save periodic checkpoint
            manifest_data["completed_chunks"] = completed_chunks
            manifest_data["rows_fetched_so_far"] = len(all_rows)
            save_manifest(args.manifest, manifest_data)
            pd.DataFrame(all_rows).to_csv(CHECKPOINT_CSV, index=False)

        time.sleep(0.08)  # Polite pacing

    # Step 5: Final Deduplication and Output
    raw_df = pd.DataFrame(all_rows)
    for col in FIELDS:
        if col not in raw_df.columns:
            raw_df[col] = pd.NA
    raw_df = raw_df[FIELDS]

    fetched_count = len(raw_df)
    unique_df = raw_df.drop_duplicates(subset=["unique_key"]) if "unique_key" in raw_df.columns else raw_df
    deduped_count = len(unique_df)
    duplicates_dropped = fetched_count - deduped_count

    unique_df.to_csv(args.output, index=False)
    # Remove checkpoint file upon successful completion
    if CHECKPOINT_CSV.exists():
        CHECKPOINT_CSV.unlink(missing_ok=True)

    manifest_data["completed_chunks"] = completed_chunks
    manifest_data["rows_fetched"] = fetched_count
    manifest_data["rows_after_dedupe"] = deduped_count
    manifest_data["duplicates_dropped"] = duplicates_dropped
    manifest_data["shortfall_from_target"] = max(0, args.target_sample - deduped_count)
    manifest_data["status"] = "COMPLETED"
    manifest_data["output_file"] = str(args.output)
    save_manifest(args.manifest, manifest_data)

    logger.info("TEMPORAL SAMPLING COMPLETE")
    logger.info(f"Rows fetched:           {fetched_count:,}")
    logger.info(f"Rows after dedupe:      {deduped_count:,}")
    logger.info(f"Duplicates dropped:     {duplicates_dropped}")
    logger.info(f"Target sample:          {args.target_sample:,}")
    logger.info(f"Shortfall from target:  {manifest_data['shortfall_from_target']}")
    logger.info(f"Saved raw CSV:          {args.output}")
    logger.info(f"Saved manifest:         {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
