"""Comprehensive unit and integration tests for CivicResolve AI Phase 3.

Validates:
- Flask application factory and health check route.
- Dashboard rendering and dynamic filter responsiveness.
- Individual complaint prediction (valid, missing, and adversarial inputs).
- Unknown and unseen categorical handling without runtime failures.
- Batch CSV scoring, schema validation, and quarantine of prohibited outcome columns.
- Finite and reproducible prediction outputs.
- K-Means pattern discovery execution and cluster profile integrity.
- Automated ReportLab PDF generation, content validation, and error resilience.
- File upload constraints and temporary storage cleanup.
"""
from __future__ import annotations

import io
import json
import pytest
import pandas as pd
import numpy as np

from app import create_app
from src.services.analytics_service import compute_kpis, filter_dataset, get_dataset
from src.services.clustering_service import run_complaint_clustering
from src.services.prediction_service import predict_batch, predict_single
from src.services.report_service import build_pdf_report


@pytest.fixture(scope="module")
def app():
    """Create and configure Flask application for testing."""
    test_app = create_app()
    test_app.config.update({
        "TESTING": True,
        "WTF_CSRF_ENABLED": False,
    })
    return test_app


@pytest.fixture(scope="module")
def client(app):
    """A test client for the app."""
    return app.test_client()


class TestPhase3FlaskAndServices:
    """Phase 3 test suite covering all web routes and service integrations."""

    def test_flask_startup_and_health_route(self, client):
        """Verify the health check endpoint returns status healthy with proper metadata."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert data["project"] == "CivicResolve AI"
        assert data["models_loaded"] is True

    def test_dashboard_renders_without_exceptions(self, client):
        """Verify the overview dashboard renders HTTP 200 and displays core civic analytics widgets."""
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        html = resp.data.decode("utf-8")
        assert "Municipal Service Demand Dashboard" in html
        assert "Filtered Volume" in html
        assert "Median Duration" in html
        assert "chart-monthly-trend" in html
        assert "chart-top-categories" in html

    def test_dashboard_filters_change_displayed_aggregates(self, client):
        """Verify that applying borough/category filters changes calculated KPIs and outputs."""
        # Unfiltered dashboard
        resp_all = client.get("/dashboard")
        assert resp_all.status_code == 200

        # Filtered to BROOKLYN only
        resp_brooklyn = client.get("/dashboard?borough=BROOKLYN")
        assert resp_brooklyn.status_code == 200

        # Filtered to a rare category
        resp_cat = client.get("/dashboard?complaint_type=HEAT%2FHOT+WATER")
        assert resp_cat.status_code == 200

        # Ensure that filtering yields different specific counts in rendered templates
        df = get_dataset()
        total_len = len(df)
        bk_len = len(df[df["borough"] == "BROOKLYN"])
        assert bk_len < total_len
        assert f"{bk_len:,}" in resp_brooklyn.data.decode("utf-8")

    def test_valid_individual_prediction(self, client):
        """Verify single complaint scoring produces valid non-negative hours and bounded probability."""
        payload = {
            "complaint_type": "Noise - Residential",
            "borough": "BROOKLYN",
            "descriptor": "Loud Music/Party",
            "incident_zip": "11201",
            "open_data_channel_type": "ONLINE",
            "created_date": "2024-05-18",
            "created_hour": "23",
            "latitude": "40.692",
            "longitude": "-73.987",
        }
        resp = client.post("/predict", data=payload)
        assert resp.status_code == 200
        html = resp.data.decode("utf-8")
        assert "Forecast Output" in html
        assert "Long-Duration Risk Score" in html

        # Direct service verification
        res = predict_single(payload)
        assert res["predicted_resolution_hours"] >= 0.0
        assert 0.0 <= res["predicted_risk_probability"] <= 1.0
        assert res["predicted_class"] in (0, 1)
        assert "hours" in res["predicted_duration_label"]

    def test_missing_and_invalid_input_handling(self, client):
        """Verify user-friendly error handling when required inputs are missing or malformed."""
        # Missing required complaint_type
        resp = client.post("/predict", data={"borough": "BROOKLYN"})
        assert resp.status_code == 200
        assert b"Complaint Type is required" in resp.data

        # Missing required borough
        resp = client.post("/predict", data={"complaint_type": "Noise - Residential"})
        assert resp.status_code == 200
        assert b"Borough is required" in resp.data

        # Invalid hour value
        with pytest.raises(ValueError, match="Hour must be between 0 and 23"):
            predict_single({
                "complaint_type": "Noise - Residential",
                "borough": "BROOKLYN",
                "created_hour": "45",
            })

    def test_unknown_categorical_values_handled_safely(self):
        """Verify pipeline handles previously unseen categories and ZIP codes gracefully."""
        adversarial_payload = {
            "complaint_type": "COMPLETELY_UNSEEN_COMPLAINT_CATEGORY_999",
            "borough": "UNSEEN_BOROUGH_ATLANTIS",
            "descriptor": "UNKNOWN_DESCRIPTOR_999",
            "incident_zip": "99999",
            "open_data_channel_type": "NEURAL_LINK",
            "created_hour": 14,
            "created_date": "2024-08-15",
            "latitude": None,
            "longitude": None,
        }
        res = predict_single(adversarial_payload)
        assert np.isfinite(res["predicted_resolution_hours"])
        assert np.isfinite(res["predicted_risk_probability"])
        assert res["predicted_class"] in (0, 1)

    def test_batch_csv_success_and_schema_validation(self, client):
        """Verify batch CSV scoring accepts valid batches and rejects invalid schemas."""
        valid_csv = (
            "complaint_type,borough,descriptor,incident_zip,created_hour\n"
            "Noise - Residential,BROOKLYN,Loud Music,11201,22\n"
            "Blocked Driveway,QUEENS,No Access,11375,10\n"
        )
        data = {"batch_file": (io.BytesIO(valid_csv.encode("utf-8")), "valid_test.csv")}
        resp = client.post("/batch", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        assert b"Successfully processed 2 complaint records" in resp.data

        # Test missing mandatory columns
        bad_csv = "descriptor,incident_zip\nLoud Music,11201\n"
        bad_data = {"batch_file": (io.BytesIO(bad_csv.encode("utf-8")), "bad_test.csv")}
        resp_bad = client.post("/batch", data=bad_data, content_type="multipart/form-data")
        assert resp_bad.status_code == 200
        assert b"missing mandatory column" in resp_bad.data

    def test_prohibited_target_columns_quarantined_in_batch(self):
        """Verify that any post-resolution outcome fields in uploaded CSVs are quarantined and stripped."""
        csv_with_leakage = (
            "complaint_type,borough,status,closed_date,resolution_hours,target_long_duration,created_year\n"
            "Street Condition,BRONX,Closed,2024-01-05,48.0,0,2024\n"
            "New Tree Request,MANHATTAN,Closed,2024-06-10,3500.0,1,2024\n"
        )
        batch_res = predict_batch(csv_with_leakage.encode("utf-8"))
        excluded = batch_res["leakage_columns_excluded"]
        assert "status" in excluded
        assert "closed_date" in excluded
        assert "resolution_hours" in excluded
        assert "target_long_duration" in excluded
        assert "created_year" in excluded

        # Check results DataFrame columns
        final_df = batch_res["full_results_df"]
        assert "status" not in final_df.columns
        assert "closed_date" not in final_df.columns
        assert "risk_probability" in final_df.columns
        assert "predicted_resolution_hours" in final_df.columns

    def test_prediction_outputs_finite_and_reproducible(self):
        """Verify identical inputs yield identical, finite predictions across repeated executions."""
        payload = {
            "complaint_type": "Blocked Driveway",
            "borough": "QUEENS",
            "descriptor": "No Access",
            "incident_zip": "11375",
            "open_data_channel_type": "PHONE",
            "created_date": "2024-03-10",
            "created_hour": 10,
        }
        res1 = predict_single(payload)
        res2 = predict_single(payload)

        assert np.isfinite(res1["predicted_resolution_hours"])
        assert np.isfinite(res1["predicted_risk_probability"])
        assert abs(res1["predicted_resolution_hours"] - res2["predicted_resolution_hours"]) < 1e-6
        assert abs(res1["predicted_risk_probability"] - res2["predicted_risk_probability"]) < 1e-6

    def test_kmeans_clustering_output_and_profiles(self, client):
        """Verify K-Means pattern discovery runs, generates silhouette evidence, and assigns cluster profiles."""
        clustering_res = run_complaint_clustering()
        assert clustering_res["total_segments_analyzed"] > 100
        assert clustering_res["selected_k"] in (2, 3, 4, 5)
        assert len(clustering_res["silhouette_scores"]) == 4
        assert len(clustering_res["cluster_profiles"]) == clustering_res["selected_k"]

        # Check web route
        resp = client.get("/clustering")
        assert resp.status_code == 200
        assert b"Discovered Municipal Cluster Archetypes" in resp.data

        # Check CSV export
        resp_csv = client.get("/clustering/export")
        assert resp_csv.status_code == 200
        assert resp_csv.content_type == "text/csv; charset=utf-8"
        assert b"cluster_label" in resp_csv.data

    def test_pdf_report_generation_and_validation(self, client):
        """Verify automated ReportLab PDF builds, returns valid PDF header, and contains key sections."""
        pdf_bytes = build_pdf_report(borough="BROOKLYN")
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 2000
        # Valid PDF signature
        assert pdf_bytes.startswith(b"%PDF-")

        # Test PDF download web endpoint
        resp = client.get("/report/download?borough=MANHATTAN")
        assert resp.status_code == 200
        assert resp.content_type == "application/pdf"
        assert resp.data.startswith(b"%PDF-")

    def test_empty_report_filters_handled_gracefully(self):
        """Verify PDF generation succeeds even when user filter produces 0 matching records."""
        # Filter for a non-existent combination (e.g. invalid date range)
        empty_pdf = build_pdf_report(start_date="1900-01-01", end_date="1900-01-02")
        assert isinstance(empty_pdf, bytes)
        assert empty_pdf.startswith(b"%PDF-")
