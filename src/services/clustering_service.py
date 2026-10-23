"""K-Means clustering service for civic complaint pattern discovery.

Discovers latent municipal service-demand archetypes by aggregating historical complaint data
into interpretable operational profiles without using supervised labels to manufacture clusters.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.services.analytics_service import get_dataset, LONG_DURATION_THRESHOLD_HOURS

logger = logging.getLogger("clustering_service")

_CLUSTERING_CACHE: dict[tuple[int, int | None], dict[str, Any]] = {}


def clear_clustering_cache() -> None:
    """Clear in-memory clustering cache for testing."""
    global _CLUSTERING_CACHE
    _CLUSTERING_CACHE.clear()


def generate_cluster_label(profile: dict[str, float], top_cats: list[str]) -> str:
    """Generate an evidence-based, human-readable operational archetype label."""
    med_hours = profile.get("avg_median_hours", 0.0)
    delay_rate = profile.get("avg_long_delay_rate_pct", 0.0)
    open_rate = profile.get("avg_open_share_pct", 0.0)
    req_count = profile.get("avg_requests", 0.0)
    
    if med_hours > 2000.0 or "New Tree Request" in top_cats:
        return "Multi-Month Capital & Forestry Programs"
    elif open_rate > 15.0 or "Damaged Tree" in top_cats:
        return "Backlog-Prone / Specialized Operations"
    elif delay_rate > 40.0:
        return "Complex Residential Code Enforcement"
    elif req_count > 100.0 or med_hours < 15.0:
        return "High-Volume Rapid Municipal Dispatch"
    else:
        return "Moderate-Volume Routine Maintenance"


def run_complaint_clustering(min_group_size: int = 5, n_clusters: int | None = None) -> dict[str, Any]:
    """Execute K-Means pattern discovery on borough x complaint_type groupings."""
    global _CLUSTERING_CACHE
    cache_key = (min_group_size, n_clusters)
    if cache_key in _CLUSTERING_CACHE:
        return _CLUSTERING_CACHE[cache_key]
        
    df = get_dataset()
    
    # 1. Aggregate historical complaint records by complaint_type and borough
    grouped = df.groupby(["complaint_type", "borough"]).agg(
        request_count=("unique_key", "count"),
        median_resolution_hours=("resolution_hours", lambda s: float(s.dropna().median()) if len(s.dropna()) > 0 else 0.0),
        long_duration_share=("resolution_hours", lambda s: float((s.dropna() > LONG_DURATION_THRESHOLD_HOURS).mean()) if len(s.dropna()) > 0 else 0.0),
        open_case_share=("status", lambda s: float((s != "Closed").mean())),
        weekend_share=("is_weekend", "mean"),
    ).reset_index()
    
    # Filter small groups to ensure statistical stability
    grouped = grouped[grouped["request_count"] >= min_group_size].copy()
    total_segments = len(grouped)
    if total_segments < 2:
        raise ValueError(f"Insufficient distinct segments ({total_segments} found, minimum 2 required) for K-Means clustering.")
    
    # 2. Prepare feature matrix
    feature_cols = [
        "request_count",
        "median_resolution_hours",
        "long_duration_share",
        "open_case_share",
        "weekend_share",
    ]
    X = grouped[feature_cols].fillna(0.0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Model comparison across candidate k values
    candidate_k = [k for k in range(2, min(6, total_segments + 1))]
    if not candidate_k:
        raise ValueError(f"Segment count ({total_segments}) is too small to evaluate candidate clusters.")

    silhouette_scores = {}
    for k in candidate_k:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        if len(set(labels)) < 2:
            silhouette_scores[k] = 0.0
        else:
            score = float(silhouette_score(X_scaled, labels))
            silhouette_scores[k] = round(score, 4)
        
    best_k = n_clusters if n_clusters is not None else max(silhouette_scores, key=silhouette_scores.get)
    if best_k > total_segments:
        raise ValueError(f"Requested n_clusters ({best_k}) exceeds available segment count ({total_segments}).")
    
    # 4. Fit chosen K-Means model
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    grouped["cluster"] = km_final.fit_predict(X_scaled)
    
    # 5. PCA Projection for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    grouped["pca_x"] = np.round(pca_coords[:, 0], 3)
    grouped["pca_y"] = np.round(pca_coords[:, 1], 3)
    
    explained_variance = [round(float(v) * 100.0, 1) for v in pca.explained_variance_ratio_]
    
    # 6. Profile each discovered cluster
    cluster_profiles = []
    for c in range(best_k):
        sub = grouped[grouped["cluster"] == c]
        top_cats = sub["complaint_type"].value_counts().head(3).index.tolist()
        
        prof_stats = {
            "avg_requests": round(float(sub["request_count"].mean()), 1),
            "avg_median_hours": round(float(sub["median_resolution_hours"].mean()), 1),
            "avg_long_delay_rate_pct": round(float(sub["long_duration_share"].mean() * 100.0), 1),
            "avg_open_share_pct": round(float(sub["open_case_share"].mean() * 100.0), 1),
            "avg_weekend_pct": round(float(sub["weekend_share"].mean() * 100.0), 1),
        }
        
        label = generate_cluster_label(prof_stats, top_cats)
        
        cluster_profiles.append({
            "cluster_id": c,
            "label": label,
            "size": len(sub),
            "share_of_segments_pct": round((len(sub) / total_segments) * 100.0, 1),
            "avg_requests": prof_stats["avg_requests"],
            "avg_median_hours": prof_stats["avg_median_hours"],
            "avg_median_days": round(prof_stats["avg_median_hours"] / 24.0, 1),
            "avg_long_delay_rate_pct": prof_stats["avg_long_delay_rate_pct"],
            "avg_open_share_pct": prof_stats["avg_open_share_pct"],
            "avg_weekend_pct": prof_stats["avg_weekend_pct"],
            "representative_categories": ", ".join(top_cats),
        })
        
    # Map profile labels onto the dataframe
    label_map = {p["cluster_id"]: p["label"] for p in cluster_profiles}
    grouped["cluster_label"] = grouped["cluster"].map(label_map)
    
    result = {
        "total_segments_analyzed": total_segments,
        "selected_k": best_k,
        "silhouette_scores": silhouette_scores,
        "selected_silhouette_score": silhouette_scores.get(best_k, 0.0),
        "pca_variance_explained_pct": explained_variance,
        "cluster_profiles": cluster_profiles,
        "segments_data": grouped.to_dict(orient="records"),
        "grouped_df": grouped,
    }
    
    _CLUSTERING_CACHE[cache_key] = result
        
    return result
