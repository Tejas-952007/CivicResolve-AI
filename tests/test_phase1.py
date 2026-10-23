import importlib
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

build_sample_mod = importlib.import_module("scripts.03_build_temporal_sample")
allocate_quotas = build_sample_mod.allocate_quotas
generate_strata_offsets = build_sample_mod.generate_strata_offsets
get_monthly_windows = build_sample_mod.get_monthly_windows
fetch_chunk = build_sample_mod.fetch_chunk

profile_mod = importlib.import_module("scripts.04_profile_temporal_sample")
INTAKE_FEATURE_ALLOWLIST = profile_mod.INTAKE_FEATURE_ALLOWLIST
EXCLUDED_POST_RESOLUTION_FIELDS = profile_mod.EXCLUDED_POST_RESOLUTION_FIELDS


class TestPhase1SamplingAndIntegrity(unittest.TestCase):
    def test_monthly_quota_allocation_exact_sum(self):
        """Test that Hamilton-Hare quota allocation matches target exactly and deterministically."""
        populations = {
            "2023-01": 240000,
            "2023-02": 220000,
            "2023-03": 250000,
            "2023-04": 280000,
        }
        for target in [100, 1000, 10000, 10003]:
            quotas = allocate_quotas(populations, target)
            self.assertEqual(sum(quotas.values()), target)
            # Check proportions correlate closely (allowing for integer quantization on small target)
            pop_prop = populations["2023-04"] / sum(populations.values())
            quota_prop = quotas["2023-04"] / target
            self.assertAlmostEqual(pop_prop, quota_prop, delta=0.015)

    def test_date_range_windows_24_months(self):
        """Test that 2023-01-01 to 2024-12-31 yields exactly 24 contiguous calendar months."""
        windows = get_monthly_windows("2023-01-01", "2024-12-31")
        self.assertEqual(len(windows), 24)
        self.assertEqual(windows[0][0], "2023-01")
        self.assertEqual(windows[-1][0], "2024-12")
        # Ensure continuity: each window start equals previous window end
        for i in range(len(windows) - 1):
            self.assertEqual(windows[i][2], windows[i + 1][1])

    def test_strata_offsets_reproducibility_and_bounds(self):
        """Test reproducible generation of stratified offsets within population ranks."""
        population = 250000
        quota = 417
        chunk_size = 25
        seed = 42

        offsets_run1 = generate_strata_offsets(population, quota, chunk_size, seed, month_index=0)
        offsets_run2 = generate_strata_offsets(population, quota, chunk_size, seed, month_index=0)
        offsets_different_seed = generate_strata_offsets(population, quota, chunk_size, seed=999, month_index=0)

        # Exact reproducibility
        self.assertEqual(offsets_run1, offsets_run2)
        self.assertNotEqual(offsets_run1, offsets_different_seed)

        # Quota sum check
        total_rows_planned = sum(limit for offset, limit in offsets_run1)
        self.assertEqual(total_rows_planned, quota)

        # Boundary check
        for offset, limit in offsets_run1:
            self.assertGreaterEqual(offset, 0)
            self.assertLessEqual(offset + limit, population)

    def test_malformed_api_response_handling(self):
        """Test that unexpected JSON response formats raise ValueError."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"error": "Server error", "message": "Failed"}
        mock_session.get.return_value = mock_response

        with self.assertRaises(ValueError):
            fetch_chunk(
                session=mock_session,
                api_url="http://fake-api",
                fields=["unique_key"],
                start_ts=pd.Timestamp("2023-01-01"),
                end_ts=pd.Timestamp("2023-02-01"),
                offset=0,
                limit=10,
                max_retries=1,
            )

    def test_duplicate_row_detection(self):
        """Test deduplication drops redundant unique_key records."""
        sample_rows = pd.DataFrame([
            {"unique_key": "1001", "created_date": "2023-01-01T10:00:00.000", "status": "Closed"},
            {"unique_key": "1002", "created_date": "2023-01-01T11:00:00.000", "status": "Closed"},
            {"unique_key": "1001", "created_date": "2023-01-01T10:00:00.000", "status": "Closed"},
        ])
        deduped = sample_rows.drop_duplicates(subset=["unique_key"])
        self.assertEqual(len(sample_rows), 3)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(set(deduped["unique_key"]), {"1001", "1002"})

    def test_timestamp_handling_nyc_local_semantics(self):
        """Test parsing of naive floating timestamps in America/New_York context."""
        raw_strings = pd.Series(["2023-01-15T14:30:00.000", "2023-07-15T09:15:00.000"])
        local_ts = pd.to_datetime(raw_strings).dt.tz_localize("America/New_York")

        # Local hour preservation
        self.assertEqual(local_ts.iloc[0].hour, 14)
        self.assertEqual(local_ts.iloc[1].hour, 9)

        # UTC conversion: EST (UTC-5 in Jan) vs EDT (UTC-4 in July)
        utc_ts = local_ts.dt.tz_convert("UTC")
        self.assertEqual(utc_ts.iloc[0].hour, 19)  # 14:30 EST -> 19:30 UTC
        self.assertEqual(utc_ts.iloc[1].hour, 13)  # 09:15 EDT -> 13:15 UTC

    def test_resolution_duration_and_invalid_pairs(self):
        """Test valid duration, zero duration, and negative duration detection."""
        df = pd.DataFrame({
            "created": ["2023-01-01T10:00:00.000", "2023-01-01T10:00:00.000", "2023-01-01T12:00:00.000"],
            "closed":  ["2023-01-01T12:30:00.000", "2023-01-01T10:00:00.000", "2023-01-01T10:00:00.000"],
        })
        c_utc = pd.to_datetime(df["created"]).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
        cl_utc = pd.to_datetime(df["closed"]).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
        hours = (cl_utc - c_utc).dt.total_seconds() / 3600.0

        self.assertAlmostEqual(hours.iloc[0], 2.5, places=3)
        self.assertEqual(hours.iloc[1], 0.0)
        self.assertLess(hours.iloc[2], 0.0)

        # Verification of anomalies
        negative_count = int((hours < 0).sum())
        zero_count = int((hours == 0).sum())
        self.assertEqual(negative_count, 1)
        self.assertEqual(zero_count, 1)

    def test_feature_boundary_leakage_prevention(self):
        """Verify post-resolution fields are strictly banned from intake feature allowlist."""
        for forbidden in EXCLUDED_POST_RESOLUTION_FIELDS:
            self.assertNotIn(
                forbidden,
                INTAKE_FEATURE_ALLOWLIST,
                f"Data leakage risk: {forbidden} must not be in INTAKE_FEATURE_ALLOWLIST!",
            )
        # Verify approved intake features do not include outcome columns
        self.assertIn("complaint_type", INTAKE_FEATURE_ALLOWLIST)
        self.assertIn("created_hour", INTAKE_FEATURE_ALLOWLIST)
        self.assertIn("borough", INTAKE_FEATURE_ALLOWLIST)

    def test_api_retry_on_transient_failure(self):
        """Test that transient HTTP errors trigger retry and backoff before succeeding."""
        mock_session = MagicMock()
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = [{"unique_key": "999", "created_date": "2023-01-01T00:00:00.000"}]

        # First call 429, second call 200
        mock_session.get.side_effect = [resp_429, resp_200]

        with patch("time.sleep", return_value=None):
            result = fetch_chunk(
                session=mock_session,
                api_url="http://fake-api",
                fields=["unique_key", "created_date"],
                start_ts=pd.Timestamp("2023-01-01"),
                end_ts=pd.Timestamp("2023-02-01"),
                offset=0,
                limit=1,
                max_retries=3,
            )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["unique_key"], "999")
        self.assertEqual(mock_session.get.call_count, 2)


if __name__ == "__main__":
    unittest.main()
