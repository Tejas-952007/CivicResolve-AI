"""CivicResolve AI — Phase 6: Killer QA, Adversarial Testing & Defect Hunt Test Suite.

Exhaustive, evidence-based adversarial testing covering:
- Stage 2: Adversarial input testing (missing fields, malformed CSV, Unicode, formula injection, quarantine).
- Stage 3: ML correctness & zero-leakage verification.
- Stage 4: Flask route security, method enforcement, path traversal, and upload guardrails.
- Stage 5: K-Means & Analytics edge cases.
- Stage 6: Q-learning simulation boundary conditions and toy verification.
- Stage 7: PDF report generation stress tests and zero-division protection.
- Stage 8: Integration and usability smoke tests.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app import create_app
from src.features import (
    EXCLUDED_POST_RESOLUTION_FIELDS,
    INTAKE_FEATURE_ALLOWLIST,
    load_dataset,
    split_temporal_data,
)
from src.services.analytics_service import (
    compute_kpis,
    filter_dataset,
    get_borough_breakdown,
    get_dataset,
    get_monthly_trends,
    get_top_categories,
    LONG_DURATION_THRESHOLD_HOURS,
)
from src.services.clustering_service import run_complaint_clustering
from src.services.prediction_service import (
    load_models,
    predict_batch,
    predict_single,
    sanitize_formula_injection,
)
from src.services.report_service import build_pdf_report
from src.services.rl_simulation_service import (
    FROZEN_QUEUES,
    SimulationConfig,
    encode_state,
    run_policy_comparison,
)


@pytest.fixture(scope="module")
def app():
    """Create and configure Flask test application."""
    test_app = create_app()
    test_app.config.update({
        "TESTING": True,
        "WTF_CSRF_ENABLED": False,
    })
    return test_app


@pytest.fixture(scope="module")
def client(app):
    """Test client fixture."""
    return app.test_client()


# =====================================================================
# STAGE 2: ADVERSARIAL INPUT TESTING
# =====================================================================
class TestStage2AdversarialInputs:
    """Rigorous input validation, fuzzing, and security payload testing."""

    def test_missing_and_empty_mandatory_fields(self):
        """Verify missing or whitespace-only mandatory fields are rejected with clean ValueErrors."""
        with pytest.raises(ValueError, match="Complaint Type is required"):
            predict_single({"complaint_type": "", "borough": "BROOKLYN"})

        with pytest.raises(ValueError, match="Complaint Type is required"):
            predict_single({"complaint_type": "   ", "borough": "BROOKLYN"})

        with pytest.raises(ValueError, match="Borough is required"):
            predict_single({"complaint_type": "Illegal Parking", "borough": ""})

        with pytest.raises(ValueError, match="Borough is required"):
            predict_single({"complaint_type": "Illegal Parking", "borough": "   "})

    def test_extreme_and_out_of_bounds_coordinates(self):
        """Verify impossible coordinates (lat > 90, lon > 180) are rejected."""
        with pytest.raises(ValueError, match="Latitude must be a valid number between -90 and 90"):
            predict_single({
                "complaint_type": "Illegal Parking",
                "borough": "BROOKLYN",
                "latitude": 999.0,
                "longitude": -73.98,
            })

        with pytest.raises(ValueError, match="Longitude must be a valid number between -180 and 180"):
            predict_single({
                "complaint_type": "Illegal Parking",
                "borough": "BROOKLYN",
                "latitude": 40.71,
                "longitude": -999.0,
            })

    def test_non_finite_temporal_inputs(self):
        """Verify non-finite hour and month values raise explicit ValueErrors."""
        with pytest.raises(ValueError, match="Hour must be between 0 and 23"):
            predict_single({
                "complaint_type": "Illegal Parking",
                "borough": "BROOKLYN",
                "created_hour": float("inf"),
            })

        with pytest.raises(ValueError, match="Hour must be between 0 and 23"):
            predict_single({
                "complaint_type": "Illegal Parking",
                "borough": "BROOKLYN",
                "created_hour": float("nan"),
            })

        with pytest.raises(ValueError, match="Month must be a valid finite number"):
            predict_single({
                "complaint_type": "Illegal Parking",
                "borough": "BROOKLYN",
                "created_month": float("nan"),
            })

    def test_multilingual_unicode_and_special_strings(self):
        """Verify model handles Marathi, Hindi, emoji, SQL, and HTML tags gracefully without crashing."""
        payloads = [
            "रस्त्यावर कचरा साचला आहे",  # Marathi
            "पानी की समस्या और जाम",  # Hindi
            "🚨🔥💥 Overflow Alert!",  # Emoji
            "' OR '1'='1' --",  # SQL injection attempt
            "<script>alert('XSS')</script>",  # XSS attempt
            "Very " * 500 + "Long String",  # Long string (>2500 chars)
            'Quotes "with" commas, and\nnewlines\r\ntab\t',
        ]
        for payload in payloads:
            res = predict_single({
                "complaint_type": payload,
                "borough": "QUEENS",
                "descriptor": payload,
            })
            assert np.isfinite(res["predicted_resolution_hours"])
            assert 0.0 <= res["predicted_risk_probability"] <= 1.0
            assert res["risk_tier"] in ["LOW", "MODERATE", "ELEVATED", "HIGH"]

    def test_csv_formula_injection_sanitization(self):
        """Verify spreadsheet formula injection prefixes (=, +, -, @) are sanitized, while signed numbers remain intact."""
        # Dangerous formula payloads
        assert sanitize_formula_injection("=cmd|' /C calc'!A0").startswith("'=")
        assert sanitize_formula_injection("@SUM(A1:A10)").startswith("'@")
        assert sanitize_formula_injection("+2+5+cmd").startswith("'+")
        assert sanitize_formula_injection("-2+3+cmd").startswith("'-")
        assert sanitize_formula_injection("=1+1").startswith("'=")

        # Legitimate signed numbers MUST NOT be corrupted
        assert sanitize_formula_injection("-73.987") == "-73.987"
        assert sanitize_formula_injection("+40.712") == "+40.712"
        assert sanitize_formula_injection("-10") == "-10"
        assert sanitize_formula_injection(123.45) == 123.45
        assert sanitize_formula_injection(None) is None

    def test_batch_csv_formula_injection_neutralization(self):
        """Verify batch CSV scoring sanitizes formula injection cells in results."""
        csv_data = (
            "complaint_type,borough,descriptor\n"
            "=cmd|' /C calc'!A0,BROOKLYN,Normal Descriptor\n"
            "Noise - Residential,QUEENS,@SUM(1+1)\n"
            "Blocked Driveway,BRONX,-73.987\n"  # Valid numeric string
        )
        res = predict_batch(csv_data.encode("utf-8"))
        df_out = res["full_results_df"]

        # Cell starting with '=' must be escaped
        assert str(df_out.loc[df_out["borough"] == "BROOKLYN", "complaint_type"].values[0]).startswith("'=")
        # Cell starting with '@' must be escaped
        assert str(df_out.loc[df_out["borough"] == "QUEENS", "descriptor"].values[0]).startswith("'@")
        # Legitimate signed number should remain unescaped
        assert str(df_out.loc[df_out["borough"] == "BRONX", "descriptor"].values[0]) == "-73.987"

    def test_batch_export_csv_formula_injection_end_to_end(self, client):
        """Verify the full batch export HTTP path neutralizes =, +, -, @ formula cells while preserving signed numbers."""
        csv_payload = (
            "complaint_type,borough,descriptor\n"
            "=cmd|' /C calc'!A0,BROOKLYN,+40.712\n"
            "@SUM(A1:A10),QUEENS,+cmd|' /C calc'!A0\n"
            "-cmd|' /C calc'!A0,MANHATTAN,-73.987\n"
        )
        post_resp = client.post(
            "/batch",
            data={"batch_file": (io.BytesIO(csv_payload.encode("utf-8")), "adversarial_intake.csv")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert post_resp.status_code == 200
        assert b"Successfully processed 3 complaint records" in post_resp.data

        from app import BATCH_RESULTS_CACHE
        session_id = list(BATCH_RESULTS_CACHE.keys())[-1]

        export_resp = client.post("/batch/export", data={"session_id": session_id})
        assert export_resp.status_code == 200
        assert export_resp.headers["Content-Type"].startswith("text/csv")
        csv_text = export_resp.data.decode("utf-8")

        # Assert formula prefixes (=, @, +, -) are neutralized with single quote prefix
        assert "'=cmd|' /C calc'!A0" in csv_text
        assert "'@SUM(A1:A10)" in csv_text
        assert "'+cmd|' /C calc'!A0" in csv_text
        assert "'-cmd|' /C calc'!A0" in csv_text

        # Assert legitimate signed numeric values remain unescaped
        assert "-73.987" in csv_text
        assert "'-73.987" not in csv_text
        assert "+40.712" in csv_text
        assert "'+40.712" not in csv_text

    def test_batch_quarantine_of_all_prohibited_post_resolution_columns(self):
        """Verify that every prohibited post-resolution and target column is quarantined and dropped."""
        adversarial_csv = (
            "complaint_type,borough,closed_date,status,resolution_hours,target_long_duration,resolution_description,agency\n"
            "Illegal Parking,BROOKLYN,2024-05-01,Closed,15.5,0.0,Case resolved,NYPD\n"
            "Noise - Residential,QUEENS,2024-05-02,Closed,250.0,1.0,Action taken,DEP\n"
        )
        res = predict_batch(adversarial_csv.encode("utf-8"))
        df_out = res["full_results_df"]

        # Prohibited fields must be recorded in excluded list
        for forbidden in ["closed_date", "status", "resolution_hours", "target_long_duration", "resolution_description", "agency"]:
            assert forbidden in res["leakage_columns_excluded"]
            assert forbidden not in df_out.columns

        # Verify predictions were calculated and finite
        assert len(df_out) == 2
        assert "predicted_resolution_hours" in df_out.columns
        assert "risk_probability" in df_out.columns

    def test_quarantined_values_do_not_affect_predictions(self):
        """Verify identical intake features produce identical predictions regardless of malicious values in quarantined columns."""
        clean_csv = (
            "complaint_type,borough,incident_zip\n"
            "Illegal Parking,BROOKLYN,11201\n"
        )
        poisoned_csv = (
            "complaint_type,borough,incident_zip,resolution_hours,target_long_duration\n"
            "Illegal Parking,BROOKLYN,11201,99999.0,1.0\n"
        )
        res_clean = predict_batch(clean_csv.encode("utf-8"))
        res_poisoned = predict_batch(poisoned_csv.encode("utf-8"))

        clean_hours = res_clean["full_results_df"]["predicted_resolution_hours"].values[0]
        poison_hours = res_poisoned["full_results_df"]["predicted_resolution_hours"].values[0]
        assert clean_hours == poison_hours

        clean_prob = res_clean["full_results_df"]["risk_probability"].values[0]
        poison_prob = res_poisoned["full_results_df"]["risk_probability"].values[0]
        assert clean_prob == poison_prob

    def test_batch_empty_and_oversized_csv_handling(self):
        """Verify 0-row and >5000-row batch files are rejected cleanly with ValueErrors."""
        empty_csv = "complaint_type,borough\n"
        with pytest.raises(ValueError, match="contains 0 rows"):
            predict_batch(empty_csv.encode("utf-8"))

        # Generate synthetic batch with 5,001 rows
        large_rows = ["complaint_type,borough"] + ["Illegal Parking,BROOKLYN"] * 5001
        large_csv = "\n".join(large_rows).encode("utf-8")
        with pytest.raises(ValueError, match="exceeds maximum batch limit of 5,000 rows"):
            predict_batch(large_csv)


# =====================================================================
# STAGE 3: ML CORRECTNESS & LEAKAGE AUDIT
# =====================================================================
class TestStage3MLCorrectnessAndLeakage:
    """Verify temporal partitions, frozen model integrity, threshold derivation, and reproducibility."""

    def test_temporal_partitions_are_strictly_disjoint(self):
        """Verify train (2023), validation (2024-H1), and test (2024-H2) sets are disjoint with zero overlap."""
        df = get_dataset()
        train_df, val_df, test_df = split_temporal_data(df)

        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0

        train_indices = set(train_df.index)
        val_indices = set(val_df.index)
        test_indices = set(test_df.index)

        assert len(train_indices & val_indices) == 0, "Train and Val overlap!"
        assert len(train_indices & test_indices) == 0, "Train and Test overlap!"
        assert len(val_indices & test_indices) == 0, "Val and Test overlap!"

    def test_no_2024_record_enters_training_split(self):
        """Verify no record created in 2024 enters the 2023 training partition."""
        df = get_dataset()
        train_df, _, _ = split_temporal_data(df)

        dates = pd.to_datetime(train_df["created_date_only"])
        years = dates.dt.year.unique()
        assert list(years) == [2023], f"Found non-2023 records in training split: {years}"

    def test_long_duration_threshold_constant_and_training_derived(self):
        """Verify the 124.925h threshold matches the documented frozen training derivation."""
        assert LONG_DURATION_THRESHOLD_HOURS == 124.925
        reg, cls, meta = load_models()
        if meta and "threshold_hours" in meta:
            assert meta["threshold_hours"] == 124.925

    def test_model_inference_reproducibility(self):
        """Verify repeated inference on identical inputs yields bitwise identical outputs."""
        sample_input = {
            "complaint_type": "Illegal Parking",
            "borough": "QUEENS",
            "descriptor": "Blocked Hydrant",
            "incident_zip": "11375",
            "created_date": "2024-04-15",
            "created_hour": 14,
            "latitude": 40.718,
            "longitude": -73.844,
        }
        res1 = predict_single(sample_input)
        res2 = predict_single(sample_input)

        assert res1["predicted_resolution_hours"] == res2["predicted_resolution_hours"]
        assert res1["predicted_risk_probability"] == res2["predicted_risk_probability"]
        assert res1["predicted_class"] == res2["predicted_class"]


# =====================================================================
# STAGE 4: FLASK ROUTES & APPLICATION SECURITY
# =====================================================================
class TestStage4FlaskSecurityAndRoutes:
    """Verify HTTP methods, query params, path traversal, and upload guardrails."""

    def test_method_not_allowed_enforcement(self, client):
        """Verify GET-only routes reject POST with HTTP 405 Method Not Allowed."""
        resp = client.post("/dashboard")
        assert resp.status_code == 405

        resp = client.post("/evaluation")
        assert resp.status_code == 405

        resp = client.post("/clustering")
        assert resp.status_code == 405

    def test_path_traversal_on_figures_blocked(self, client):
        """Verify directory traversal attempts on /figures/ are blocked with 404."""
        resp = client.get("/figures/../../app.py")
        assert resp.status_code in [404, 400]

        resp = client.get("/figures/%2e%2e%2f%2e%2e%2fapp.py")
        assert resp.status_code in [404, 400]

    def test_non_existent_route_404_handling(self, client):
        """Verify 404 routes redirect cleanly or return 404 without unhandled crashes."""
        resp = client.get("/non-existent-endpoint-12345")
        assert resp.status_code == 404

    def test_custom_404_error_handler(self, client):
        """Verify unknown route triggers custom 404 handler without tracebacks or path disclosure."""
        resp = client.get("/unknown-municipal-route-qa-audit")
        assert resp.status_code == 404
        assert b"Traceback" not in resp.data
        assert b"/home/" not in resp.data

    def test_custom_405_method_not_allowed_handler(self, client):
        """Verify invalid HTTP methods trigger custom 405 handler returning clean JSON error response."""
        resp = client.post("/dashboard")
        assert resp.status_code == 405
        assert b"Method Not Allowed" in resp.data
        assert b"Traceback" not in resp.data
        assert b"/home/" not in resp.data

    def test_batch_upload_file_format_validation(self, client):
        """Verify uploading non-CSV files is rejected with warning."""
        data = {
            "batch_file": (io.BytesIO(b"fake binary executable content"), "payload.exe"),
        }
        resp = client.post("/batch", data=data, content_type="multipart/form-data", follow_redirects=True)
        assert resp.status_code == 200
        assert b"Invalid file format" in resp.data

    def test_batch_cache_session_expiration(self, client):
        """Verify exporting a non-existent or expired session ID is handled gracefully."""
        resp = client.post("/batch/export", data={"session_id": "non-existent-uuid"}, follow_redirects=True)
        assert resp.status_code == 200
        assert b"expired or not found" in resp.data

    def test_batch_upload_exceeding_max_content_length_returns_413(self, client):
        """Verify upload exceeding MAX_CONTENT_LENGTH (10MB) returns HTTP 413 without leaking tracebacks or paths."""
        oversized_bytes = b"0" * (10 * 1024 * 1024 + 1024)
        data = {
            "batch_file": (io.BytesIO(oversized_bytes), "oversized_intake.csv"),
        }
        resp = client.post("/batch", data=data, content_type="multipart/form-data")
        assert resp.status_code == 413
        assert b"Traceback" not in resp.data
        assert b"/home/" not in resp.data
        assert b"secret" not in resp.data

    def test_custom_500_internal_server_error_handler(self):
        """Verify intentionally induced exception triggers custom 500 handler without leaking traceback, secrets, or paths."""
        crash_app = create_app()
        crash_app.config["TESTING"] = True
        crash_app.config["PROPAGATE_EXCEPTIONS"] = False

        @crash_app.route("/test-induced-internal-500")
        def induced_crash():
            raise RuntimeError("Database connection timed out at /home/tejas/secret_credentials.env with key=secret98765")

        client = crash_app.test_client()
        resp = client.get("/test-induced-internal-500")
        assert resp.status_code == 500
        assert b"secret98765" not in resp.data
        assert b"secret_credentials" not in resp.data
        assert b"Traceback" not in resp.data
        assert b"/home/tejas" not in resp.data


# =====================================================================
# STAGE 5: K-MEANS & ANALYTICS EDGE CASES
# =====================================================================
class TestStage5KMeansAndAnalyticsEdgeCases:
    """Verify clustering under boundary and degenerate conditions."""

    def test_clustering_produces_consistent_segment_counts(self):
        """Verify clusters sum to total eligible grouped segments."""
        res = run_complaint_clustering()
        grouped_df = res["grouped_df"]
        total_grouped = len(grouped_df)
        profile_counts = sum(p["size"] for p in res["cluster_profiles"])
        assert profile_counts == total_grouped

    def test_insufficient_segments_raises_clean_error(self):
        """Verify clustering fails gracefully when min_group_size leaves fewer than 2 segments."""
        with pytest.raises(ValueError, match="Insufficient distinct segments"):
            run_complaint_clustering(min_group_size=999999)

    def test_clustering_cache_key_respects_min_group_size(self):
        """Verify cached clustering results for default min_group_size=5 are not incorrectly returned for min_group_size=20."""
        from src.services.clustering_service import clear_clustering_cache
        clear_clustering_cache()

        # 1. First call with default threshold (min_group_size=5)
        res_default = run_complaint_clustering(min_group_size=5)
        count_default = res_default["total_segments_analyzed"]
        assert count_default == 274

        # 2. Second call with more stringent threshold (min_group_size=20)
        res_custom = run_complaint_clustering(min_group_size=20)
        count_custom = res_custom["total_segments_analyzed"]
        assert count_custom == 118

        # 3. Assert the cache did not return the default result for the custom threshold
        assert count_default != count_custom
        assert count_custom < count_default

        # 4. Third call with min_group_size=5 should return cached default cleanly
        res_cached = run_complaint_clustering(min_group_size=5)
        assert res_cached["total_segments_analyzed"] == count_default

        # 5. Verify extreme threshold raises clean ValueError without poisoning cache
        with pytest.raises(ValueError, match="Insufficient distinct segments"):
            run_complaint_clustering(min_group_size=999999)

    def test_analytics_kpis_on_empty_filter(self):
        """Verify compute_kpis handles 0-row datasets gracefully without division-by-zero."""
        empty_df = pd.DataFrame(columns=["status", "closed_date", "resolution_hours", "created_date_only"])
        kpis = compute_kpis(empty_df)
        assert kpis["total_records"] == 0
        assert kpis["eligible_count"] == 0
        assert kpis["long_duration_share"] == 0.0
        assert kpis["median_resolution_hours"] == 0.0

    def test_borough_breakdown_on_empty_filter(self):
        """Verify get_borough_breakdown handles 0-row slices without zero division."""
        empty_df = pd.DataFrame(columns=["borough", "status", "resolution_hours"])
        b_data = get_borough_breakdown(empty_df)
        assert b_data["boroughs"] == []
        assert b_data["counts"] == []


# =====================================================================
# STAGE 6: Q-LEARNING & SIMULATION STRESS TESTS
# =====================================================================
class TestStage6SimulationStressTests:
    """Verify queue allocation simulation under edge cases, bounds, and toy settings."""

    def test_simulation_config_boundary_validation(self):
        """Verify invalid config values (capacity, negative seed, non-finite weights, invalid alpha/gamma) raise ValueError."""
        # 1. Capacity boundaries (1 <= capacity <= 50)
        with pytest.raises(ValueError, match="capacity_per_period"):
            SimulationConfig(capacity_per_period=0).validate()
        with pytest.raises(ValueError, match="capacity_per_period"):
            SimulationConfig(capacity_per_period=-1).validate()
        with pytest.raises(ValueError, match="capacity_per_period"):
            SimulationConfig(capacity_per_period=51).validate()

        # 2. Negative seeds (must be non-negative)
        with pytest.raises(ValueError, match="seed"):
            SimulationConfig(seed=-1).validate()
        with pytest.raises(ValueError, match="seed"):
            SimulationConfig(seed=-999).validate()

        # 3. Non-finite reward weights and penalties (NaN, +inf, -inf)
        for non_finite in [float("nan"), float("inf"), float("-inf")]:
            with pytest.raises(ValueError, match="reward_weight_risk"):
                SimulationConfig(reward_weight_risk=non_finite).validate()
            with pytest.raises(ValueError, match="reward_weight_wait"):
                SimulationConfig(reward_weight_wait=non_finite).validate()
            with pytest.raises(ValueError, match="reward_weight_backlog"):
                SimulationConfig(reward_weight_backlog=non_finite).validate()
            with pytest.raises(ValueError, match="reward_penalty_empty"):
                SimulationConfig(reward_penalty_empty=non_finite).validate()
            with pytest.raises(ValueError, match="reward_penalty_starve"):
                SimulationConfig(reward_penalty_starve=non_finite).validate()
            with pytest.raises(ValueError, match="reward_penalty_term"):
                SimulationConfig(reward_penalty_term=non_finite).validate()

        # Negative reward weights must be rejected
        with pytest.raises(ValueError, match="reward_weight_risk"):
            SimulationConfig(reward_weight_risk=-2.0).validate()
        with pytest.raises(ValueError, match="reward_penalty_empty"):
            SimulationConfig(reward_penalty_empty=-0.5).validate()

        # Legitimate zero-valued weights must be accepted
        zero_weight_cfg = SimulationConfig(reward_weight_risk=0.0, reward_weight_wait=0.0)
        zero_weight_cfg.validate()  # Must not raise

        # 4. Learning rate alpha boundaries (0.01 <= alpha <= 1.0)
        with pytest.raises(ValueError, match="alpha"):
            SimulationConfig(alpha=0.0).validate()
        with pytest.raises(ValueError, match="alpha"):
            SimulationConfig(alpha=1.5).validate()

        # 5. Discount factor gamma boundaries (0.0 <= gamma <= 1.0)
        with pytest.raises(ValueError, match="gamma"):
            SimulationConfig(gamma=-0.1).validate()
        with pytest.raises(ValueError, match="gamma"):
            SimulationConfig(gamma=1.1).validate()

        # 6. Horizon periods (5 <= horizon <= 100)
        with pytest.raises(ValueError, match="horizon_periods"):
            SimulationConfig(horizon_periods=4).validate()
        with pytest.raises(ValueError, match="horizon_periods"):
            SimulationConfig(horizon_periods=101).validate()

        # 7. Arrival batch size (1 <= arrival <= 50)
        with pytest.raises(ValueError, match="arrival_batch_size"):
            SimulationConfig(arrival_batch_size=0).validate()
        with pytest.raises(ValueError, match="arrival_batch_size"):
            SimulationConfig(arrival_batch_size=51).validate()

        # 8. Training episodes (1 <= episodes <= 500)
        with pytest.raises(ValueError, match="training_episodes"):
            SimulationConfig(training_episodes=0).validate()
        with pytest.raises(ValueError, match="training_episodes"):
            SimulationConfig(training_episodes=501).validate()

    def test_state_encoding_determinism_and_uniqueness(self):
        """Verify state encoding produces identical strings for identical queues."""
        queues1 = [[] for _ in range(len(FROZEN_QUEUES))]
        queues2 = [[] for _ in range(len(FROZEN_QUEUES))]
        s1 = encode_state(queues1, capacity_remaining=8)
        s2 = encode_state(queues2, capacity_remaining=8)
        assert s1 == s2

        # Change capacity to a different tier (tier 0: <= 1 slot, tier 2: 4+ slots)
        s3 = encode_state(queues1, capacity_remaining=1)
        assert s1 != s3

    def test_policy_comparison_runs_with_small_valid_config(self):
        """Verify policy comparison runs cleanly with minimal toy configuration."""
        cfg = SimulationConfig(
            capacity_per_period=4,
            horizon_periods=5,
            arrival_batch_size=5,
            training_episodes=10,
            seed=42,
        )
        sim_res = run_policy_comparison(cfg, eval_split="val")
        assert "summary" in sim_res
        assert len(sim_res["summary"]) == 5  # FIFO, Risk-First, Risk+Wait, Round-Robin, Q-Learning

        for policy_stat in sim_res["summary"]:
            assert np.isfinite(policy_stat["mean_reward"])
            assert 0.0 <= policy_stat["mean_slot_utilization_pct"] <= 100.0


# =====================================================================
# STAGE 7: PDF & REPORT GENERATION STRESS TESTS
# =====================================================================
class TestStage7PDFReportGeneration:
    """Verify automated PDF report compilation under edge filters and content constraints."""

    def test_pdf_starts_with_valid_pdf_signature(self):
        """Verify generated PDF starts with standard %PDF- magic bytes."""
        pdf_bytes = build_pdf_report(borough="ALL", complaint_type="ALL")
        assert len(pdf_bytes) > 1000
        assert pdf_bytes.startswith(b"%PDF-")

    def test_pdf_handles_filtered_single_category(self):
        """Verify PDF generation succeeds for filtered single borough and category."""
        pdf_bytes = build_pdf_report(borough="BROOKLYN", complaint_type="Illegal Parking")
        assert len(pdf_bytes) > 1000
        assert pdf_bytes.startswith(b"%PDF-")

    def test_pdf_handles_empty_slice_gracefully(self):
        """Verify PDF generation succeeds even if filtered slice yields zero records."""
        # Non-existent date window
        pdf_bytes = build_pdf_report(start_date="1900-01-01", end_date="1900-01-02")
        assert len(pdf_bytes) > 1000
        assert pdf_bytes.startswith(b"%PDF-")


# =====================================================================
# STAGE 8: INTEGRATION & USABILITY SMOKE TESTS
# =====================================================================
class TestStage8IntegrationSmokeTests:
    """Verify entire workflow across Flask application routes."""

    def test_end_to_end_user_journey(self, client):
        """Simulate realistic user workflow: dashboard -> predict -> batch -> evaluation -> clustering -> simulation."""
        # 1. Dashboard
        r1 = client.get("/dashboard")
        assert r1.status_code == 200

        # 2. Individual prediction
        r2 = client.post("/predict", data={
            "complaint_type": "Blocked Driveway",
            "borough": "QUEENS",
            "descriptor": "No Access",
            "incident_zip": "11375",
            "created_date": "2024-05-10",
            "created_hour": "10",
        }, follow_redirects=True)
        assert r2.status_code == 200

        # 3. Batch scoring sample download
        r3 = client.get("/batch/sample")
        assert r3.status_code == 200
        assert b"complaint_type" in r3.data

        # 4. Batch scoring upload
        r4 = client.post("/batch", data={
            "batch_file": (io.BytesIO(r3.data), "test_intake.csv"),
        }, content_type="multipart/form-data", follow_redirects=True)
        assert r4.status_code == 200
        assert b"Successfully processed" in r4.data

        # 5. Model evaluation
        r5 = client.get("/evaluation")
        assert r5.status_code == 200

        # 6. Pattern discovery
        r6 = client.get("/clustering")
        assert r6.status_code == 200

        # 7. Resource planning
        r7 = client.get("/resource-planning")
        assert r7.status_code == 200

        # 8. CSV and PDF exports
        r8 = client.get("/report/download")
        assert r8.status_code == 200
        assert r8.data.startswith(b"%PDF-")

        r9 = client.get("/export/filtered-csv")
        assert r9.status_code == 200
        assert b"unique_key" in r9.data
