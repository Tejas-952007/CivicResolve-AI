"""Automated professional PDF report generator for CivicResolve AI using ReportLab.

Compiles executive summaries, descriptive statistics, measured model evaluation metrics,
and K-Means operational clusters into an audit-ready municipal intelligence report.
"""
from __future__ import annotations

import datetime
import io
import json
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.services.analytics_service import (
    compute_kpis,
    filter_dataset,
    get_borough_breakdown,
    get_dataset,
    get_top_categories,
    LONG_DURATION_THRESHOLD_HOURS,
)
from src.services.clustering_service import run_complaint_clustering

# Palette: Civic Navy, Slate Blue, Teal, Charcoal, Off-White
PRIMARY_COLOR = colors.HexColor("#1e293b")  # Slate 800
ACCENT_COLOR = colors.HexColor("#0284c7")   # Sky 600
SECONDARY_COLOR = colors.HexColor("#0f766e") # Teal 700
LIGHT_BG = colors.HexColor("#f8fafc")      # Slate 50
BORDER_COLOR = colors.HexColor("#cbd5e1")  # Slate 300
TEXT_MUTED = colors.HexColor("#64748b")    # Slate 500


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute total pages and render running headers and footers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: list[Any] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count: int) -> None:
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(TEXT_MUTED)
        
        # Suppress running header on cover/first page
        if self._pageNumber > 1:
            self.drawString(36, 756, "CivicResolve AI — Municipal Intelligence Diagnostic Report")
            self.drawRightString(576, 756, "Confidential / Research Prototype")
            self.setStrokeColor(BORDER_COLOR)
            self.setLineWidth(0.5)
            self.line(36, 750, 576, 750)
            
        # Running footer on all pages
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 30, page_str)
        self.drawString(36, 30, "Source: NYC 311 Sample (2023–2024) | Strictly Evaluated Research Layer")
        self.setStrokeColor(BORDER_COLOR)
        self.setLineWidth(0.5)
        self.line(36, 42, 576, 42)
        
        self.restoreState()


def build_pdf_report(
    start_date: str | None = None,
    end_date: str | None = None,
    borough: str | None = None,
    complaint_type: str | None = None,
) -> bytes:
    """Generate a publication-grade PDF report based on current filters and measured ML metrics."""
    df_raw = get_dataset()
    df_filtered = filter_dataset(
        df_raw,
        start_date=start_date,
        end_date=end_date,
        borough=borough,
        complaint_type=complaint_type,
    )
    
    kpis = compute_kpis(df_filtered)
    top_cats = get_top_categories(df_filtered, limit=5)
    boroughs_data = get_borough_breakdown(df_filtered)
    clustering_res = run_complaint_clustering()
    
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=54,
        bottomMargin=54,
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=PRIMARY_COLOR,
        spaceAfter=6,
    )
    
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=ACCENT_COLOR,
        spaceAfter=15,
    )
    
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=PRIMARY_COLOR,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )
    
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=PRIMARY_COLOR,
        spaceAfter=6,
    )
    
    meta_style = ParagraphStyle(
        "Meta_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11,
        textColor=TEXT_MUTED,
        spaceAfter=8,
    )
    
    cell_style = ParagraphStyle(
        "Cell_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=PRIMARY_COLOR,
    )
    
    cell_bold_style = ParagraphStyle(
        "Cell_Bold_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=PRIMARY_COLOR,
    )

    story: list[Any] = []
    
    # ------------------ COVER / HEADER ------------------
    story.append(Paragraph("CivicResolve AI", subtitle_style))
    story.append(Paragraph("Municipal Operational Intelligence & Delay Risk Diagnostic", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT_COLOR, spaceAfter=10))
    
    gen_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filter_desc = f"Borough: <b>{borough or 'ALL'}</b> | Category: <b>{complaint_type or 'ALL'}</b> | Window: <b>{kpis['date_range']}</b>"
    story.append(Paragraph(f"Generated on <b>{gen_time}</b> | Filter Scope: {filter_desc}", meta_style))
    story.append(Spacer(1, 10))
    
    # ------------------ SECTION 1: EXECUTIVE SUMMARY ------------------
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        f"This diagnostic synthesizes <b>{kpis['total_records']:,}</b> municipal service requests within the selected "
        f"filter scope. Of these records, <b>{kpis['closed_count']:,}</b> have been resolved, and <b>{kpis['open_count']:,}</b> "
        f"remain active or in-progress. The historical median resolution duration across eligible closed cases is "
        f"<b>{kpis['median_resolution_hours']} hours</b> ({kpis['median_resolution_days']} days). A total of "
        f"<b>{kpis['long_duration_count']:,}</b> cases ({kpis['long_duration_share']}%) exceeded the project-defined "
        f"long-duration threshold of <b>{LONG_DURATION_THRESHOLD_HOURS} hours</b> (~5.2 days).",
        body_style,
    ))
    
    # Summary KPI Table
    kpi_table_data = [
        [
            Paragraph("Total Intake Records", cell_bold_style),
            Paragraph(f"{kpis['total_records']:,}", cell_style),
            Paragraph("Median Duration (Eligible)", cell_bold_style),
            Paragraph(f"{kpis['median_resolution_hours']}h ({kpis['median_resolution_days']}d)", cell_style),
        ],
        [
            Paragraph("Closed (Valid Timestamp)", cell_bold_style),
            Paragraph(f"{kpis['closed_count']:,}", cell_style),
            Paragraph("Unresolved / Unclosed Backlog", cell_bold_style),
            Paragraph(f"{kpis['open_count']:,} (active + null date)", cell_style),
        ],
        [
            Paragraph("Service-Eligible Closed", cell_bold_style),
            Paragraph(f"{kpis['eligible_count']:,}", cell_style),
            Paragraph("Exceeding 124.9h Benchmark", cell_bold_style),
            Paragraph(f"{kpis['long_duration_count']:,} ({kpis['long_duration_share']}%)", cell_style),
        ],
    ]
    kpi_tbl = Table(kpi_table_data, colWidths=[130, 140, 150, 120])
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(kpi_tbl)
    story.append(Spacer(1, 12))
    
    # ------------------ SECTION 2: DESCRIPTIVE SERVICE PROFILE ------------------
    story.append(Paragraph("2. Top Complaint Categories & Borough Distribution", h1_style))
    story.append(Paragraph(
        "Historical demand patterns demonstrate pronounced heterogeneity by complaint category and municipal jurisdiction. "
        "Top complaint types represent the primary service volume drivers:",
        body_style,
    ))
    
    top_cat_data = [
        [
            Paragraph("Complaint Category", cell_bold_style),
            Paragraph("Volume", cell_bold_style),
            Paragraph("Share of Filter", cell_bold_style),
            Paragraph("Median Duration", cell_bold_style),
        ]
    ]
    for cat, count, med_dur in zip(top_cats["categories"], top_cats["counts"], top_cats["median_durations"]):
        share_pct = round((count / max(1, kpis["total_records"])) * 100.0, 1)
        top_cat_data.append([
            Paragraph(str(cat), cell_style),
            Paragraph(f"{count:,}", cell_style),
            Paragraph(f"{share_pct}%", cell_style),
            Paragraph(f"{med_dur}h", cell_style),
        ])
        
    top_tbl = Table(top_cat_data, colWidths=[240, 90, 100, 110])
    top_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(top_tbl)
    story.append(Spacer(1, 10))
    
    # Borough Breakdown Table
    borough_table_data = [
        [
            Paragraph("Borough", cell_bold_style),
            Paragraph("Complaints", cell_bold_style),
            Paragraph("Median Duration", cell_bold_style),
            Paragraph("Delay Rate (>124.9h)", cell_bold_style),
        ]
    ]
    for b, c, lr, mh in zip(
        boroughs_data["boroughs"][:5],
        boroughs_data["counts"][:5],
        boroughs_data["long_rate_pct"][:5],
        boroughs_data["median_hours"][:5],
    ):
        borough_table_data.append([
            Paragraph(str(b), cell_style),
            Paragraph(f"{c:,}", cell_style),
            Paragraph(f"{mh}h", cell_style),
            Paragraph(f"{lr}%", cell_style),
        ])
    b_tbl = Table(borough_table_data, colWidths=[180, 110, 120, 130])
    b_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(b_tbl)
    story.append(Spacer(1, 14))
    
    # ------------------ SECTION 3: MACHINE LEARNING EVALUATION ------------------
    story.append(Paragraph("3. Leakage-Free Machine Learning Performance", h1_style))
    story.append(Paragraph(
        "All models were trained exclusively on 2023 intake data ($N=4,711$) using verified intake-time features only. "
        "Hyperparameters and preferred pipelines were chosen strictly on 2024-H1 validation data ($N=2,437$). "
        "The metrics below reflect the <b>unbiased final test evaluation</b> on held-out 2024-H2 records ($N=2,635$):",
        body_style,
    ))
    
    ml_metrics_data = [
        [
            Paragraph("Task & Pipeline", cell_bold_style),
            Paragraph("Primary Metric", cell_bold_style),
            Paragraph("Secondary Metrics", cell_bold_style),
            Paragraph("Operational Assessment", cell_bold_style),
        ],
        [
            Paragraph("<b>Classification</b><br/>Logistic Regression (Class-Balanced)", cell_style),
            Paragraph("<b>F1: 0.7093</b><br/>Recall: 88.7%", cell_style),
            Paragraph("Precision: 59.1%<br/>ROC-AUC: 0.9262<br/>PR-AUC: 0.7227", cell_style),
            Paragraph("Captured 488 of 550 delayed requests; generates 338 false alarms. Suited for screening and review triage.", cell_style),
        ],
        [
            Paragraph("<b>Regression</b><br/>Random Forest (Log1p Target)", cell_style),
            Paragraph("<b>MAE: 251.99h</b><br/>MedAE: 15.59h", cell_style),
            Paragraph("RMSE: 947.67h<br/>R²: 0.1446", cell_style),
            Paragraph("Typical absolute error is ~15.6 hours; heavy right-skew extreme outliers expand mean absolute error.", cell_style),
        ],
    ]
    ml_tbl = Table(ml_metrics_data, colWidths=[150, 100, 130, 160])
    ml_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(ml_tbl)
    story.append(Spacer(1, 12))
    
    # ------------------ SECTION 4: K-MEANS PATTERN DISCOVERY ------------------
    story.append(Paragraph("4. K-Means Operational Pattern Archetypes", h1_style))
    story.append(Paragraph(
        f"K-Means clustering ($k={clustering_res['selected_k']}$, Silhouette score: <b>{clustering_res['selected_silhouette_score']}</b>) "
        "grouped 274 municipal complaint segments into distinct operational demand archetypes:",
        body_style,
    ))
    
    cluster_table_data = [
        [
            Paragraph("Cluster Archetype", cell_bold_style),
            Paragraph("Segments", cell_bold_style),
            Paragraph("Avg Median Hours", cell_bold_style),
            Paragraph("Delay Rate %", cell_bold_style),
            Paragraph("Representative Categories", cell_bold_style),
        ]
    ]
    for p in clustering_res["cluster_profiles"]:
        cluster_table_data.append([
            Paragraph(f"<b>Cluster {p['cluster_id']}</b>: {p['label']}", cell_style),
            Paragraph(f"{p['size']} ({p['share_of_segments_pct']}%)", cell_style),
            Paragraph(f"{p['avg_median_hours']}h ({p['avg_median_days']}d)", cell_style),
            Paragraph(f"{p['avg_long_delay_rate_pct']}%", cell_style),
            Paragraph(p["representative_categories"], cell_style),
        ])
    cl_tbl = Table(cluster_table_data, colWidths=[150, 70, 100, 75, 145])
    cl_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(cl_tbl)
    story.append(Spacer(1, 14))
    
    # ------------------ SECTION 5: GOVERNANCE & LIMITATIONS ------------------
    story.append(Paragraph("5. Scientific Limitations & Governance Disclaimers", h1_style))
    story.append(Paragraph(
        "<b>1. Statistical Dependence, Not Causation:</b> High feature importance scores indicate mathematical correlation "
        "within historical ticket routing, not physical root causes of delay.<br/>"
        "<b>2. Project-Defined Benchmark vs. Official SLA:</b> The 124.925-hour cutoff is an empirical 75th percentile "
        "from the 2023 training partition, not a regulatory standard or official agency SLA violation.<br/>"
        "<b>3. Geographic & Operational Specificity:</b> Models are trained on NYC 311 municipal data. They are not calibrated, "
        "validated, or representative for Indian municipal corporations (e.g., PCMC).<br/>"
        "<b>4. Screening Tool Notice:</b> The classifier exhibits high recall (88.7%) with moderate precision (59.1%). "
        "It should be deployed solely for supervisor screening and review prioritization, not automated punitive actions.",
        body_style,
    ))
    
    # ------------------ SECTION 6: RESOURCE PLANNING & Q-LEARNING SIMULATION (APPENDIX) ------------------
    sim_metrics_path = Path(__file__).resolve().parent.parent.parent / "outputs" / "phase4_simulation_metrics.json"
    if sim_metrics_path.exists():
        try:
            with open(sim_metrics_path, "r", encoding="utf-8") as f:
                sim_data = json.load(f)
            
            story.append(Spacer(1, 14))
            story.append(Paragraph("6. Appendix: Reinforcement Learning & Queue Allocation Simulation", h1_style))
            
            cfg = sim_data.get("metadata", {}).get("config", {})
            story.append(Paragraph(
                f"<b>Simulation Parameters:</b> Horizon: <b>{cfg.get('horizon_periods', 25)} periods</b> | "
                f"Capacity: <b>{cfg.get('capacity_per_period', 8)} slots/period</b> | "
                f"Arrival Batch: <b>{cfg.get('arrival_batch_size', 10)} items/period</b> | "
                f"Training: <b>{cfg.get('training_episodes', 100)} episodes</b> (&alpha;={cfg.get('alpha', 0.1)}, &gamma;={cfg.get('gamma', 0.95)}).<br/>"
                "The multi-queue simulator evaluated 5 policies across identical complaint arrival streams over 5 random seeds:",
                body_style,
            ))
            
            sim_summary = sim_data.get("summary", [])
            sim_table_data = [
                [
                    Paragraph("Policy", cell_bold_style),
                    Paragraph("Mean Reward", cell_bold_style),
                    Paragraph("Slot Utilization", cell_bold_style),
                    Paragraph("Items Handled", cell_bold_style),
                    Paragraph("Mean Wait", cell_bold_style),
                    Paragraph("Max Starve", cell_bold_style),
                    Paragraph("True Long Capture", cell_bold_style),
                ]
            ]
            for s in sim_summary:
                sim_table_data.append([
                    Paragraph(f"<b>{s['policy']}</b>", cell_style),
                    Paragraph(f"{s['mean_reward']} &plusmn; {s['std_reward']}", cell_style),
                    Paragraph(f"{s['mean_slot_utilization_pct']}%", cell_style),
                    Paragraph(str(int(round(s['mean_items_handled']))), cell_style),
                    Paragraph(f"{s['mean_waiting_periods']}p", cell_style),
                    Paragraph(f"{s['mean_max_starvation']}p", cell_style),
                    Paragraph(f"<b>{s['mean_true_long_capture_pct']}%</b>", cell_style),
                ])
                
            sim_tbl = Table(sim_table_data, colWidths=[100, 85, 75, 65, 65, 65, 85])
            sim_tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(sim_tbl)
            story.append(Spacer(1, 8))
            story.append(Paragraph(
                "<i>Disclaimer: Simulated waiting periods and retrospective outcome captures demonstrate policy focus under synthetic resource constraints. "
                "They do not prove that municipal resolution times would have been causally reduced in live operations.</i>",
                meta_style,
            ))
        except Exception:
            pass

    doc.build(story, canvasmaker=NumberedCanvas)
    return buf.getvalue()
