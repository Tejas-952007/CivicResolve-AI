"""CivicResolve AI — Phase 3 Flask Web Application.

Provides a professional local municipal intelligence dashboard, intake-time resolution forecasting,
batch CSV scoring, model evaluation reporting, K-Means complaint pattern discovery, and automated PDF export.
"""
from __future__ import annotations

import io
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for,
)

from src.services.analytics_service import (
    compute_kpis,
    filter_dataset,
    get_borough_breakdown,
    get_dataset,
    get_duration_distribution,
    get_filter_options,
    get_monthly_trends,
    get_top_categories,
)
from src.services.clustering_service import run_complaint_clustering
from src.services.prediction_service import (
    load_models,
    predict_batch,
    predict_single,
    LONG_DURATION_THRESHOLD_HOURS,
)
from src.services.report_service import build_pdf_report
from src.services.rl_simulation_service import (
    FROZEN_QUEUES,
    SimulationConfig,
    run_policy_comparison,
    save_phase4_artifacts,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("app")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_JSON_PATH = OUTPUTS_DIR / "phase2_metrics.json"
MODEL_COMPARISON_CSV = OUTPUTS_DIR / "phase2_model_comparison.csv"
PHASE4_METRICS_PATH = OUTPUTS_DIR / "phase4_simulation_metrics.json"
PHASE4_POLICY_CSV = OUTPUTS_DIR / "phase4_policy_comparison.csv"

# In-memory batch results cache for CSV export
BATCH_RESULTS_CACHE: dict[str, pd.DataFrame] = {}


def create_app() -> Flask:
    """Application factory for CivicResolve AI."""
    app = Flask(__name__)
    app.secret_key = os.environ.get("SECRET_KEY", "civicresolve-ai-research-prototype-secret-key-2026")
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload guardrail
    
    # Preload ML models once at startup
    try:
        load_models()
        logger.info("Preferred ML pipelines preloaded successfully.")
    except Exception as e:
        logger.warning("Could not preload ML models at startup: %s", e)

    # Custom Jinja template filters
    @app.template_filter("format_number")
    def format_number_filter(val: Any) -> str:
        try:
            return f"{int(val):,}"
        except (ValueError, TypeError):
            return str(val)

    # ------------------ ROUTES ------------------

    @app.route("/")
    def index():
        return redirect(url_for("dashboard_view"))

    @app.route("/health")
    def health_check():
        return jsonify({
            "status": "healthy",
            "project": "CivicResolve AI",
            "phase": "Phase 3: Web Application & Analytics Dashboard",
            "models_loaded": True,
        })

    # Module A: Overview Dashboard
    @app.route("/dashboard")
    def dashboard_view():
        df_raw = get_dataset()
        filter_opts = get_filter_options()

        borough = request.args.get("borough", "ALL")
        complaint_type = request.args.get("complaint_type", "ALL")
        start_date = request.args.get("start_date", filter_opts["min_date"])
        end_date = request.args.get("end_date", filter_opts["max_date"])

        filtered_df = filter_dataset(
            df_raw,
            start_date=start_date,
            end_date=end_date,
            borough=borough,
            complaint_type=complaint_type,
        )

        kpis = compute_kpis(filtered_df)
        monthly_trends = get_monthly_trends(filtered_df)
        top_cats = get_top_categories(filtered_df, limit=10)
        borough_breakdown = get_borough_breakdown(filtered_df)
        duration_dist = get_duration_distribution(filtered_df)

        return render_template(
            "dashboard.html",
            active_page="dashboard",
            filter_options=filter_opts,
            selected_borough=borough,
            selected_complaint_type=complaint_type,
            selected_start_date=start_date,
            selected_end_date=end_date,
            kpis=kpis,
            monthly_trends=monthly_trends,
            top_categories=top_cats,
            borough_breakdown=borough_breakdown,
            duration_distribution=duration_dist,
            active_borough=borough,
            active_complaint_type=complaint_type,
            active_start_date=start_date,
            active_end_date=end_date,
        )

    # Module B: Individual Complaint Analysis
    @app.route("/predict", methods=["GET", "POST"])
    def predict_view():
        filter_opts = get_filter_options()
        result = None
        form_data = None

        if request.method == "POST":
            form_data = request.form.to_dict()
            try:
                result = predict_single(form_data)
                flash("Resolution forecast and risk probability computed successfully.", "success")
            except Exception as e:
                logger.error("Error during single prediction: %s", e)
                flash(f"Prediction Error: {e}", "danger")

        return render_template(
            "single_prediction.html",
            active_page="predict",
            complaint_types=filter_opts["complaint_types"],
            boroughs=filter_opts["boroughs"],
            form_data=form_data,
            result=result,
        )

    # Module C: Batch Complaint Scoring
    @app.route("/batch", methods=["GET", "POST"])
    def batch_view():
        results = None
        batch_session_id = None

        if request.method == "POST":
            file = request.files.get("batch_file")
            if not file or not file.filename:
                flash("Please select a valid CSV file to upload.", "danger")
                return redirect(url_for("batch_view"))

            if not file.filename.lower().endswith(".csv"):
                flash("Invalid file format. Only CSV files (.csv) are accepted.", "danger")
                return redirect(url_for("batch_view"))

            try:
                results = predict_batch(file)
                batch_session_id = str(uuid.uuid4())
                BATCH_RESULTS_CACHE[batch_session_id] = results["full_results_df"]

                # Memory management: keep cache capped at 10 items
                if len(BATCH_RESULTS_CACHE) > 10:
                    oldest = next(iter(BATCH_RESULTS_CACHE))
                    del BATCH_RESULTS_CACHE[oldest]

                flash(f"Successfully processed {results['total_rows']} complaint records.", "success")
            except Exception as e:
                logger.error("Error during batch scoring: %s", e)
                flash(f"Batch Scoring Error: {e}", "danger")

        return render_template(
            "batch_prediction.html",
            active_page="batch",
            results=results,
            batch_session_id=batch_session_id,
        )

    @app.route("/batch/sample")
    def download_batch_sample():
        """Download an example CSV containing valid intake-time complaints for testing."""
        sample_rows = [
            {
                "complaint_type": "Noise - Residential",
                "descriptor": "Loud Music/Party",
                "borough": "BROOKLYN",
                "incident_zip": "11201",
                "open_data_channel_type": "ONLINE",
                "created_date": "2024-05-18",
                "created_hour": 23,
                "latitude": 40.692,
                "longitude": -73.987,
            },
            {
                "complaint_type": "Blocked Driveway",
                "descriptor": "No Access",
                "borough": "QUEENS",
                "incident_zip": "11375",
                "open_data_channel_type": "MOBILE",
                "created_date": "2024-04-10",
                "created_hour": 9,
                "latitude": 40.718,
                "longitude": -73.844,
            },
            {
                "complaint_type": "Street Condition",
                "descriptor": "Pothole",
                "borough": "BRONX",
                "incident_zip": "10458",
                "open_data_channel_type": "PHONE",
                "created_date": "2024-02-20",
                "created_hour": 11,
                "latitude": 40.855,
                "longitude": -73.883,
            },
            {
                "complaint_type": "HEAT/HOT WATER",
                "descriptor": "ENTIRE BUILDING",
                "borough": "MANHATTAN",
                "incident_zip": "10031",
                "open_data_channel_type": "PHONE",
                "created_date": "2024-01-12",
                "created_hour": 7,
                "latitude": 40.825,
                "longitude": -73.948,
            },
            {
                "complaint_type": "New Tree Request",
                "descriptor": "For One Address",
                "borough": "STATEN ISLAND",
                "incident_zip": "10314",
                "open_data_channel_type": "ONLINE",
                "created_date": "2024-06-01",
                "created_hour": 14,
                "latitude": 40.598,
                "longitude": -74.152,
            },
        ]
        sample_df = pd.DataFrame(sample_rows)
        csv_bytes = sample_df.to_csv(index=False).encode("utf-8")
        return send_file(
            io.BytesIO(csv_bytes),
            mimetype="text/csv",
            as_attachment=True,
            download_name="civicresolve_batch_intake_sample.csv",
        )

    @app.route("/batch/export", methods=["POST"])
    def export_batch_results():
        """Export scored batch records as a downloadable CSV."""
        session_id = request.form.get("session_id")
        if not session_id or session_id not in BATCH_RESULTS_CACHE:
            flash("Batch session expired or not found. Please re-upload.", "warning")
            return redirect(url_for("batch_view"))

        results_df = BATCH_RESULTS_CACHE[session_id]
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        return send_file(
            io.BytesIO(csv_bytes),
            mimetype="text/csv",
            as_attachment=True,
            download_name="civicresolve_batch_scored_results.csv",
        )

    # Module D: Model Evaluation Centre
    @app.route("/evaluation")
    def evaluation_view():
        comp_df = pd.read_csv(MODEL_COMPARISON_CSV) if MODEL_COMPARISON_CSV.exists() else pd.DataFrame()
        reg_models = comp_df[comp_df["task"] == "Regression"].to_dict(orient="records") if len(comp_df) > 0 else []
        cls_models = comp_df[comp_df["task"] == "Classification"].to_dict(orient="records") if len(comp_df) > 0 else []

        metrics_json = {}
        if METRICS_JSON_PATH.exists():
            with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
                metrics_json = json.load(f)

        return render_template(
            "evaluation.html",
            active_page="evaluation",
            reg_models=reg_models,
            cls_models=cls_models,
            metrics=metrics_json,
        )

    # Module E: Pattern Discovery & Clustering
    @app.route("/clustering")
    def clustering_view():
        clustering_res = run_complaint_clustering()
        return render_template(
            "clustering.html",
            active_page="clustering",
            clustering=clustering_res,
        )

    @app.route("/clustering/export")
    def export_clustering_csv():
        clustering_res = run_complaint_clustering()
        grouped_df = clustering_res["grouped_df"]
        csv_bytes = grouped_df.to_csv(index=False).encode("utf-8")
        return send_file(
            io.BytesIO(csv_bytes),
            mimetype="text/csv",
            as_attachment=True,
            download_name="civicresolve_complaint_cluster_archetypes.csv",
        )

    # Module F: ReportLab PDF Report Generation
    @app.route("/report/view")
    @app.route("/report/CivicResolve_AI_Diagnostic_Report.pdf")
    def view_report():
        borough = request.args.get("borough", "ALL")
        complaint_type = request.args.get("complaint_type", "ALL")
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")

        try:
            pdf_bytes = build_pdf_report(
                start_date=start_date,
                end_date=end_date,
                borough=borough,
                complaint_type=complaint_type,
            )
            # Also save to disk so user has a local copy
            local_report_path = BASE_DIR / "static" / "reports" / "CivicResolve_AI_Diagnostic_Report.pdf"
            local_report_path.parent.mkdir(parents=True, exist_ok=True)
            local_report_path.write_bytes(pdf_bytes)

            response = send_file(
                io.BytesIO(pdf_bytes),
                mimetype="application/pdf",
                as_attachment=False,
                download_name="CivicResolve_AI_Diagnostic_Report.pdf",
            )
            response.headers["Content-Disposition"] = 'inline; filename="CivicResolve_AI_Diagnostic_Report.pdf"'
            return response
        except Exception as e:
            logger.error("Error generating PDF report: %s", e)
            flash(f"Report generation error: {e}", "danger")
            return redirect(url_for("dashboard_view"))

    @app.route("/report/download")
    def download_report():
        borough = request.args.get("borough", "ALL")
        complaint_type = request.args.get("complaint_type", "ALL")
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")

        try:
            pdf_bytes = build_pdf_report(
                start_date=start_date,
                end_date=end_date,
                borough=borough,
                complaint_type=complaint_type,
            )
            local_report_path = BASE_DIR / "static" / "reports" / "CivicResolve_AI_Diagnostic_Report.pdf"
            local_report_path.parent.mkdir(parents=True, exist_ok=True)
            local_report_path.write_bytes(pdf_bytes)

            user_downloads = Path.home() / "Downloads"
            if user_downloads.exists():
                (user_downloads / "CivicResolve_AI_Diagnostic_Report.pdf").write_bytes(pdf_bytes)

            response = send_file(
                io.BytesIO(pdf_bytes),
                mimetype="application/pdf",
                as_attachment=True,
                download_name="CivicResolve_AI_Diagnostic_Report.pdf",
            )
            response.headers["Content-Disposition"] = 'attachment; filename="CivicResolve_AI_Diagnostic_Report.pdf"'
            return response
        except Exception as e:
            logger.error("Error generating PDF report: %s", e)
            flash(f"Report generation error: {e}", "danger")
            return redirect(url_for("dashboard_view"))

    # Filtered Data CSV Export
    @app.route("/export/filtered-csv")
    @app.route("/export/civicresolve_filtered_sample.csv")
    def export_filtered_csv():
        df_raw = get_dataset()
        borough = request.args.get("borough", "ALL")
        complaint_type = request.args.get("complaint_type", "ALL")
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")

        filtered_df = filter_dataset(
            df_raw,
            start_date=start_date,
            end_date=end_date,
            borough=borough,
            complaint_type=complaint_type,
        )

        csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
        local_csv_path = BASE_DIR / "static" / "reports" / "civicresolve_filtered_sample.csv"
        local_csv_path.parent.mkdir(parents=True, exist_ok=True)
        local_csv_path.write_bytes(csv_bytes)

        user_downloads = Path.home() / "Downloads"
        if user_downloads.exists():
            (user_downloads / "civicresolve_filtered_sample.csv").write_bytes(csv_bytes)

        response = send_file(
            io.BytesIO(csv_bytes),
            mimetype="text/csv",
            as_attachment=True,
            download_name="civicresolve_filtered_sample.csv",
        )
        response.headers["Content-Disposition"] = 'attachment; filename="civicresolve_filtered_sample.csv"'
        return response

    # Figure server for diagnostic plots
    @app.route("/figures/<path:filename>")
    def figure_file(filename: str):
        return send_from_directory(FIGURES_DIR, filename)

    # Module F: Resource Planning & Q-Learning Simulation
    @app.route("/resource-planning", methods=["GET"])
    def resource_planning_view():
        if not PHASE4_METRICS_PATH.exists():
            config = SimulationConfig(
                capacity_per_period=8,
                horizon_periods=25,
                arrival_batch_size=10,
                training_episodes=100,
                seed=42,
            )
            sim_res = run_policy_comparison(config, eval_split="val")
            save_phase4_artifacts(sim_res)

        with open(PHASE4_METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        config_data = metrics_data.get("metadata", {}).get("config", {})
        summary = metrics_data.get("summary", [])

        # Chart 1: Policy Comparison Bar
        chart_policy_comparison = {
            "policies": [s["policy"] for s in summary],
            "rewards": [s["mean_reward"] for s in summary],
            "stds": [s["std_reward"] for s in summary],
            "utilization": [s["mean_slot_utilization_pct"] for s in summary],
            "capture_pct": [s["mean_true_long_capture_pct"] for s in summary],
            "starvation": [s["mean_max_starvation"] for s in summary],
            "mean_wait": [s["mean_waiting_periods"] for s in summary],
        }

        # Chart 4: Queue Allocation Share per policy
        chart_queue_shares = {
            "policies": [s["policy"] for s in summary],
            "queues": FROZEN_QUEUES,
            "shares": {q: [s.get("queue_distribution_pct", {}).get(q, 0.0) for s in summary] for q in FROZEN_QUEUES},
        }

        # Training curve from saved Q-learning policy
        q_policy_file = MODELS_DIR / "q_learning_policy.json"
        training_rewards = []
        if q_policy_file.exists():
            with open(q_policy_file, "r", encoding="utf-8") as f:
                q_data = json.load(f)
                training_rewards = q_data.get("training_rewards", [])

        smooth_rewards = []
        if training_rewards:
            s_series = pd.Series(training_rewards).rolling(window=10, min_periods=1).mean()
            smooth_rewards = [round(float(v), 2) for v in s_series]

        return render_template(
            "resource_planning.html",
            active_page="resource_planning",
            config=config_data,
            summary=summary,
            queues=FROZEN_QUEUES,
            chart_policy_comparison=chart_policy_comparison,
            chart_queue_shares=chart_queue_shares,
            training_rewards=training_rewards,
            smooth_rewards=smooth_rewards,
        )

    @app.route("/resource-planning/run", methods=["POST"])
    def resource_planning_run():
        try:
            capacity = int(request.form.get("capacity_per_period", 8))
            horizon = int(request.form.get("horizon_periods", 25))
            batch_size = int(request.form.get("arrival_batch_size", 10))
            episodes = int(request.form.get("training_episodes", 100))
            seed = int(request.form.get("seed", 42))
            alpha = float(request.form.get("alpha", 0.1))
            gamma = float(request.form.get("gamma", 0.95))
            eval_split = request.form.get("eval_split", "val")

            # Validate parameter bounds
            capacity = max(1, min(capacity, 50))
            horizon = max(5, min(horizon, 100))
            batch_size = max(1, min(batch_size, 50))
            episodes = max(10, min(episodes, 500))
            alpha = max(0.01, min(alpha, 1.0))
            gamma = max(0.0, min(gamma, 1.0))
            if eval_split not in ["val", "test", "train"]:
                eval_split = "val"

            cfg = SimulationConfig(
                capacity_per_period=capacity,
                horizon_periods=horizon,
                arrival_batch_size=batch_size,
                training_episodes=episodes,
                seed=seed,
                alpha=alpha,
                gamma=gamma,
            )

            results = run_policy_comparison(cfg, eval_split=eval_split)
            save_phase4_artifacts(results)
            flash(f"Simulation completed successfully across 5 policies on '{eval_split}' partition!", "success")
        except Exception as e:
            logger.error("Simulation run failed: %s", e)
            flash(f"Simulation error: {e}", "danger")

        return redirect(url_for("resource_planning_view"))

    @app.route("/resource-planning/export-csv")
    def resource_planning_export_csv():
        if not PHASE4_POLICY_CSV.exists():
            flash("No simulation results available to export.", "warning")
            return redirect(url_for("resource_planning_view"))
        return send_file(
            PHASE4_POLICY_CSV,
            mimetype="text/csv",
            as_attachment=True,
            download_name="phase4_policy_comparison.csv",
        )

    # ------------------ ERROR HANDLERS ------------------
    @app.errorhandler(404)
    def handle_404(err):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Resource not found", "status": 404}), 404
        flash("The requested page or route was not found.", "warning")
        return redirect(url_for("dashboard_view")), 404

    @app.errorhandler(405)
    def handle_405(err):
        return jsonify({"error": "Method Not Allowed", "status": 405}), 405

    @app.errorhandler(413)
    def handle_413(err):
        flash("Uploaded file exceeds the maximum allowed limit of 10 MB.", "danger")
        return redirect(url_for("batch_view")), 413

    @app.errorhandler(500)
    def handle_500(err):
        logger.error("Internal Server Error: %s", err)
        flash("An unexpected internal municipal processing error occurred.", "danger")
        return redirect(url_for("dashboard_view")), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
