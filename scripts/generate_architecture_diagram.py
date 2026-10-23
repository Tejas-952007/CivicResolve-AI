#!/usr/bin/env python3
"""
generate_architecture_diagram.py
Generates a publication-grade vector/raster architecture block diagram for CivicResolve AI.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_diagram(output_path="report/screenshots/architecture_diagram.png"):
    fig, ax = plt.subplots(figsize=(12, 10), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Colors
    c_blue_bg = "#eff6ff"
    c_blue_border = "#2563eb"
    c_dark_text = "#0f172a"
    c_sub_text = "#334155"
    c_green_bg = "#f0fdf4"
    c_green_border = "#16a34a"
    c_purple_bg = "#faf5ff"
    c_purple_border = "#9333ea"
    c_orange_bg = "#fff7ed"
    c_orange_border = "#ea580c"

    # Box Drawing Helper
    def draw_box(x, y, w, h, title, subtitle="", bg=c_blue_bg, border=c_blue_border, title_size=10, sub_size=8):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.5,rounding_size=1.5",
            facecolor=bg, edgecolor=border, linewidth=1.5
        )
        ax.add_patch(rect)
        if subtitle:
            ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=c_dark_text)
            ax.text(x + w / 2, y + h * 0.32, subtitle, ha="center", va="center", fontsize=sub_size, color=c_sub_text)
        else:
            ax.text(x + w / 2, y + h * 0.5, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=c_dark_text)

    # Arrow Helper
    def draw_arrow(x1, y1, x2, y2, color="#475569"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, mutation_scale=12)
        )

    # Title
    ax.text(50, 97.5, "CivicResolve AI: End-to-End System Architecture", ha="center", va="center", fontsize=15, fontweight="bold", color="#1e3a8a")

    # Layer 1: Data Ingestion & Partitioning
    draw_box(20, 88, 60, 6, "NYC 311 Socrata Open Data API (2023–2024 Population)", "Historical Service Requests Telemetry", bg="#f8fafc", border="#64748b")
    draw_arrow(50, 88, 50, 82)

    draw_box(15, 76, 70, 6, "Rank-Stratified Temporal Sampler & Integrity Auditor", "Monthly Quotas (416-417/mo) • Seed 42 • Exact N = 10,000 Canonical Sample", bg="#f8fafc", border="#64748b")
    draw_arrow(50, 76, 50, 70)

    draw_box(10, 64, 80, 6, "Strict Disjoint Calendar Partitioning", "Training 2023 (N=4,711 eligible)  |  Validation 2024-H1 (N=2,437)  |  Final Test 2024-H2 (N=2,635)", bg=c_blue_bg, border=c_blue_border)

    # Arrows from Layer 1 to Layer 2
    draw_arrow(28, 64, 28, 56)
    draw_arrow(72, 64, 72, 56)

    # Layer 2: Feature Matrix & Core Analytics Services
    draw_box(8, 48, 40, 8, "Leakage-Free Feature Pipeline (X)", "Agency, Type, Descriptor, Borough, Zip, Cyclical Time\n[Strictly Barring closed_date, status, resolution_description]", bg=c_orange_bg, border=c_orange_border, title_size=9.5, sub_size=7.5)

    draw_box(52, 48, 40, 8, "Tabular Q-Learning Capacity Simulator", "Discrete MDP (1,764 State Combinations across 6 Queues)\nEvaluated vs. FIFO, Risk-Plus-Wait, Risk-First & Round-Robin", bg=c_purple_bg, border=c_purple_border, title_size=9.5, sub_size=7.5)

    # Arrows between feature matrix and models
    draw_arrow(18, 48, 18, 40)
    draw_arrow(38, 48, 38, 40)
    draw_arrow(48, 52, 52, 52) # Connection to Q-learning

    # Layer 3: Supervised & Unsupervised Models
    draw_box(4, 32, 28, 8, "Random Forest (Log1p)", "Expected Duration (Hours)\nTest MAE: 251.99h | MedAE: 15.59h", bg=c_green_bg, border=c_green_border, title_size=9, sub_size=7.5)

    draw_box(36, 32, 28, 8, "Balanced Logistic Regression", "Delay Risk Classifier (>124.9h)\nTest Recall: 88.7% | F1: 0.7093", bg=c_green_bg, border=c_green_border, title_size=9, sub_size=7.5)

    draw_box(68, 32, 28, 8, "K-Means Cluster Profiler", "k = 5 Demand Archetypes (0.3836 Sil)\n274 Municipality-Category Segments", bg=c_blue_bg, border=c_blue_border, title_size=9, sub_size=7.5)

    # Arrow to RL from Classifier (Out-of-Fold Risk Score)
    draw_arrow(50, 40, 65, 48)

    # Arrows down to Flask UI Layer
    draw_arrow(18, 32, 18, 22)
    draw_arrow(50, 32, 50, 22)
    draw_arrow(82, 32, 82, 22)

    # Layer 4: Interactive Web Application (Flask app.py)
    draw_box(4, 16, 92, 6, "Interactive Flask Web Application Layer (app.py)", "Service Layer Routing • Security Quarantine • Plotly Visualizations • Dynamic ReportLab PDF", bg="#f1f5f9", border="#475569", title_size=10.5, sub_size=8)

    # Layer 5: Concrete Routes & Subsystems
    routes = [
        ("Dashboard", "/dashboard", 4),
        ("Single Intake", "/predict", 17.5),
        ("Batch Scoring", "/batch", 31),
        ("Model Evaluation", "/evaluation", 44.5),
        ("Clustering", "/clustering", 58),
        ("Resource Plan", "/resource-planning", 71.5),
        ("Diagnostic PDF", "/report/download", 85)
    ]
    for name, r_path, x_pos in routes:
        draw_arrow(x_pos + 5.5, 16, x_pos + 5.5, 9)
        draw_box(x_pos, 2, 11, 7, name, r_path, bg="#ffffff", border="#94a3b8", title_size=7.5, sub_size=6.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Generated clean architecture diagram: {output_path}")

if __name__ == "__main__":
    create_diagram()
