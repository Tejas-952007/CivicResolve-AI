"""Prediction service for individual complaint analysis and batch CSV scoring.

Loads frozen Phase 2 regression and classification pipelines once, guarantees zero leakage,
validates intake schemas, and provides transparent, non-causal municipal intelligence.
"""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features import INTAKE_FEATURE_ALLOWLIST, EXCLUDED_POST_RESOLUTION_FIELDS
from src.preprocess import CyclicFeatureTransformer  # Required for unpickling custom transformer

logger = logging.getLogger("prediction_service")

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
REGRESSOR_PATH = MODELS_DIR / "preferred_regressor.joblib"
CLASSIFIER_PATH = MODELS_DIR / "preferred_classifier.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

LONG_DURATION_THRESHOLD_HOURS = 124.925

_REGRESSOR = None
_CLASSIFIER = None
_METADATA = None


def load_models() -> tuple[Any, Any, dict[str, Any]]:
    """Load and cache the frozen Phase 2 regression and classification models."""
    global _REGRESSOR, _CLASSIFIER, _METADATA
    if _REGRESSOR is None or _CLASSIFIER is None:
        if not REGRESSOR_PATH.exists() or not CLASSIFIER_PATH.exists():
            raise FileNotFoundError("Trained model pipelines not found in models/ directory.")
        logger.info("Loading preferred regressor from %s", REGRESSOR_PATH)
        _REGRESSOR = joblib.load(REGRESSOR_PATH)
        logger.info("Loading preferred classifier from %s", CLASSIFIER_PATH)
        _CLASSIFIER = joblib.load(CLASSIFIER_PATH)
        
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                _METADATA = json.load(f)
        else:
            _METADATA = {}
            
    return _REGRESSOR, _CLASSIFIER, _METADATA


def format_duration(hours: float) -> str:
    """Format elapsed hours into an intuitive human-readable duration."""
    if hours < 0:
        return "0.0 hours"
    if hours < 24.0:
        return f"{hours:.1f} hours"
    days = int(hours // 24)
    rem_hours = hours % 24
    if days == 1:
        return f"1 day, {rem_hours:.1f} hours"
    return f"{days} days, {rem_hours:.1f} hours"


def get_risk_tier(prob: float) -> dict[str, str]:
    """Map probability score to intuitive operational review tier."""
    if prob < 0.25:
        return {"tier": "LOW", "badge": "bg-success", "label": "Low Long-Duration Risk"}
    elif prob < 0.50:
        return {"tier": "MODERATE", "badge": "bg-info", "label": "Moderate Review Risk"}
    elif prob < 0.75:
        return {"tier": "ELEVATED", "badge": "bg-warning text-dark", "label": "Elevated Delay Risk"}
    else:
        return {"tier": "HIGH", "badge": "bg-danger", "label": "High Long-Duration Risk"}


def sanitize_formula_injection(val: Any) -> Any:
    """Neutralize spreadsheet formula injection characters (=, +, -, @) without altering legitimate signed numbers."""
    if isinstance(val, str) and len(val) > 0:
        if val[0] in ("=", "+", "-", "@"):
            try:
                float(val)
                return val
            except ValueError:
                return f"'{val}"
    return val


def predict_single(data: dict[str, Any]) -> dict[str, Any]:
    """Validate intake features and compute predictions using frozen Phase 2 pipelines."""
    regressor, classifier, metadata = load_models()
    
    # 1. Validate required fields
    complaint_type = str(data.get("complaint_type", "")).strip()
    borough = str(data.get("borough", "")).strip()
    if not complaint_type:
        raise ValueError("Complaint Type is required.")
    if not borough:
        raise ValueError("Borough is required.")
        
    descriptor = str(data.get("descriptor", "Missing")).strip() or "Missing"
    incident_zip = str(data.get("incident_zip", "Missing")).strip() or "Missing"
    channel = str(data.get("open_data_channel_type", "ONLINE")).strip() or "ONLINE"
    
    # 2. Parse temporal context
    created_date_str = str(data.get("created_date", "")).strip()
    try:
        if created_date_str:
            dt = pd.to_datetime(created_date_str)
            month = float(dt.month)
            weekday = str(dt.weekday())  # 0=Monday, 6=Sunday
            is_weekend = 1.0 if dt.weekday() in (5, 6) else 0.0
        else:
            month = float(data.get("created_month", 6.0))
            weekday = str(data.get("created_weekday", "2"))
            is_weekend = float(data.get("is_weekend", 0.0))
    except Exception as e:
        raise ValueError(f"Invalid created_date format '{created_date_str}': {e}")
        
    if not np.isfinite(month) or not (1.0 <= month <= 12.0):
        raise ValueError("Month must be a valid finite number between 1 and 12.")

    try:
        hour = float(data.get("created_hour", 12.0))
        if not np.isfinite(hour) or not (0.0 <= hour <= 23.0):
            raise ValueError("Hour must be between 0 and 23.")
    except Exception as e:
        raise ValueError(f"Invalid created_hour: {e}")
        
    # 3. Parse coordinates safely with strict geographic bounds
    lat_val = data.get("latitude")
    lon_val = data.get("longitude")
    
    latitude = np.nan
    longitude = np.nan
    if lat_val is not None and str(lat_val).strip() != "":
        try:
            latitude = float(lat_val)
            if not np.isfinite(latitude) or not (-90.0 <= latitude <= 90.0):
                raise ValueError(f"Latitude must be a valid number between -90 and 90 (got '{lat_val}')")
        except ValueError as e:
            raise ValueError(f"Invalid latitude '{lat_val}': {e}")
    if lon_val is not None and str(lon_val).strip() != "":
        try:
            longitude = float(lon_val)
            if not np.isfinite(longitude) or not (-180.0 <= longitude <= 180.0):
                raise ValueError(f"Longitude must be a valid number between -180 and 180 (got '{lon_val}')")
        except ValueError as e:
            raise ValueError(f"Invalid longitude '{lon_val}': {e}")
            
    # 4. Construct single-row DataFrame adhering strictly to allowlist
    row_df = pd.DataFrame([{
        "complaint_type": complaint_type,
        "descriptor": descriptor,
        "borough": borough,
        "incident_zip": incident_zip,
        "open_data_channel_type": channel,
        "created_hour": hour,
        "created_weekday": weekday,
        "created_month": month,
        "is_weekend": is_weekend,
        "latitude": latitude,
        "longitude": longitude,
    }])[INTAKE_FEATURE_ALLOWLIST]
    
    # 5. Model Inference
    pred_raw_hours = float(regressor.predict(row_df)[0])
    pred_hours = max(0.0, pred_raw_hours)
    
    pred_prob = float(classifier.predict_proba(row_df)[0, 1])
    pred_class = int(pred_prob >= 0.5)
    risk_info = get_risk_tier(pred_prob)
    
    return {
        "complaint_type": complaint_type,
        "descriptor": descriptor,
        "borough": borough,
        "incident_zip": incident_zip,
        "open_data_channel_type": channel,
        "created_hour": hour,
        "created_weekday": weekday,
        "created_month": month,
        "is_weekend": is_weekend,
        "latitude": latitude if not np.isnan(latitude) else None,
        "longitude": longitude if not np.isnan(longitude) else None,
        "predicted_resolution_hours": round(pred_hours, 2),
        "predicted_duration_label": format_duration(pred_hours),
        "predicted_risk_probability": round(pred_prob, 4),
        "predicted_risk_percentage": round(pred_prob * 100.0, 1),
        "predicted_class": pred_class,
        "risk_tier": risk_info["tier"],
        "risk_badge": risk_info["badge"],
        "risk_label": risk_info["label"],
        "threshold_hours": LONG_DURATION_THRESHOLD_HOURS,
        "exceeds_threshold": bool(pred_hours > LONG_DURATION_THRESHOLD_HOURS),
    }


def predict_batch(csv_file_or_path: Any) -> dict[str, Any]:
    """Process an uploaded CSV of complaints, enforce leakage constraints, and return risk-scored outputs."""
    regressor, classifier, metadata = load_models()
    
    # 1. Read CSV safely
    if isinstance(csv_file_or_path, pd.DataFrame):
        df_raw = csv_file_or_path.copy()
    elif isinstance(csv_file_or_path, (str, Path)):
        df_raw = pd.read_csv(csv_file_or_path, low_memory=False)
    elif isinstance(csv_file_or_path, (bytes, bytearray)):
        df_raw = pd.read_csv(io.BytesIO(csv_file_or_path), low_memory=False)
    elif hasattr(csv_file_or_path, "read"):
        content = csv_file_or_path.read()
        if isinstance(content, str):
            content = content.encode("utf-8")
        if not content:
            raise ValueError("Uploaded file is empty.")
        df_raw = pd.read_csv(io.BytesIO(content), low_memory=False)
    else:
        raise ValueError(f"Unsupported input type for batch scoring: {type(csv_file_or_path)}")
        
    total_input_rows = len(df_raw)
    if total_input_rows == 0:
        raise ValueError("Uploaded CSV contains 0 rows.")
    if total_input_rows > 5000:
        raise ValueError(f"File exceeds maximum batch limit of 5,000 rows (got {total_input_rows}).")
        
    # 2. Detect and strip potential data leakage columns
    leakage_detected = []
    for col in EXCLUDED_POST_RESOLUTION_FIELDS + ["resolution_hours", "target_long_duration", "created_year", "agency"]:
        if col in df_raw.columns:
            leakage_detected.append(col)
            
    # 3. Validate required columns
    missing_required = []
    for req in ["complaint_type", "borough"]:
        if req not in df_raw.columns:
            missing_required.append(req)
    if missing_required:
        raise ValueError(f"Uploaded CSV is missing mandatory column(s): {', '.join(missing_required)}")
        
    # 4. Construct intake feature matrix safely
    X = pd.DataFrame(index=df_raw.index)
    X["complaint_type"] = df_raw["complaint_type"].fillna("Missing").astype(str)
    X["borough"] = df_raw["borough"].fillna("Missing").astype(str)
    
    X["descriptor"] = df_raw["descriptor"].fillna("Missing").astype(str) if "descriptor" in df_raw.columns else "Missing"
    
    if "incident_zip" in df_raw.columns:
        zips = df_raw["incident_zip"].fillna(-1).astype(str).str.replace(r"\.0$", "", regex=True)
        X["incident_zip"] = zips.replace("-1", "Missing").replace("nan", "Missing")
    else:
        X["incident_zip"] = "Missing"
        
    X["open_data_channel_type"] = (
        df_raw["open_data_channel_type"].fillna("ONLINE").astype(str)
        if "open_data_channel_type" in df_raw.columns
        else "ONLINE"
    )
    
    # Derive or extract temporal features
    if "created_date" in df_raw.columns or "created_date_only" in df_raw.columns:
        date_col = "created_date_only" if "created_date_only" in df_raw.columns else "created_date"
        parsed_dates = pd.to_datetime(df_raw[date_col], errors="coerce")
        X["created_month"] = parsed_dates.dt.month.fillna(6.0).astype(float)
        X["created_weekday"] = parsed_dates.dt.day_of_week.fillna(2).astype(int).astype(str)
        X["is_weekend"] = parsed_dates.dt.day_of_week.isin([5, 6]).astype(float)
    else:
        X["created_month"] = df_raw["created_month"].astype(float) if "created_month" in df_raw.columns else 6.0
        X["created_weekday"] = df_raw["created_weekday"].astype(str) if "created_weekday" in df_raw.columns else "2"
        X["is_weekend"] = df_raw["is_weekend"].astype(float) if "is_weekend" in df_raw.columns else 0.0
        
    X["created_hour"] = df_raw["created_hour"].astype(float) if "created_hour" in df_raw.columns else 12.0
    
    # Geospatial coordinates
    X["latitude"] = pd.to_numeric(df_raw["latitude"], errors="coerce") if "latitude" in df_raw.columns else np.nan
    X["longitude"] = pd.to_numeric(df_raw["longitude"], errors="coerce") if "longitude" in df_raw.columns else np.nan
    
    # Guarantee strict feature allowlist ordering
    X_clean = X[INTAKE_FEATURE_ALLOWLIST]
    
    # 5. Batch Inference
    pred_raw_hours = regressor.predict(X_clean)
    pred_hours = np.clip(pred_raw_hours, a_min=0.0, a_max=None)
    pred_probs = classifier.predict_proba(X_clean)[:, 1]
    pred_classes = (pred_probs >= 0.5).astype(int)
    
    # 6. Build Results DataFrame
    results_df = df_raw.copy()
    
    # Drop forbidden post-resolution columns from final output if user included them
    for col in leakage_detected:
        if col in results_df.columns:
            results_df = results_df.drop(columns=[col])
            
    results_df["predicted_resolution_hours"] = np.round(pred_hours, 2)
    results_df["predicted_duration_label"] = [format_duration(h) for h in pred_hours]
    results_df["risk_probability"] = np.round(pred_probs, 4)
    results_df["predicted_long_duration_class"] = pred_classes
    results_df["risk_tier"] = [get_risk_tier(p)["tier"] for p in pred_probs]
    
    # Rank by risk probability descending
    results_df = results_df.sort_values(by="risk_probability", ascending=False).reset_index(drop=True)
    results_df["review_priority_rank"] = np.arange(1, len(results_df) + 1)
    
    # Sanitize spreadsheet formula injection in exported string columns
    for c in results_df.columns:
        if pd.api.types.is_object_dtype(results_df[c]) or pd.api.types.is_string_dtype(results_df[c]):
            results_df[c] = results_df[c].apply(sanitize_formula_injection)
    
    # Summary telemetry
    high_risk_count = int((results_df["risk_probability"] >= 0.5).sum())
    avg_est_duration = float(np.mean(pred_hours))
    
    return {
        "total_rows": total_input_rows,
        "valid_rows": len(results_df),
        "high_risk_alerts": high_risk_count,
        "high_risk_percentage": round((high_risk_count / len(results_df)) * 100.0, 1),
        "mean_estimated_hours": round(avg_est_duration, 1),
        "leakage_columns_excluded": leakage_detected,
        "preview_records": results_df.head(50).to_dict(orient="records"),
        "full_results_df": results_df,
    }
