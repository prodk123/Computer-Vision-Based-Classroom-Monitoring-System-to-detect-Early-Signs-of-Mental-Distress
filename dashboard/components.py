"""
Dashboard UI Components.

Reusable Streamlit components for the classroom monitoring dashboard.
Uses Plotly for interactive charts and provides clean, teacher-friendly UI elements.

IMPORTANT: No medical or clinical terminology is used anywhere in the UI.
All labels are neutral and focused on behavioral observations.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


# ============================================================
#  Color Palette (professional, non-alarming)
# ============================================================
COLORS = {
    "low": "#22c55e",          # Green
    "moderate": "#f59e0b",     # Amber
    "elevated": "#f97316",     # Orange
    "high": "#ef4444",         # Red
    "primary": "#3b82f6",      # Blue
    "secondary": "#8b5cf6",    # Purple
    "bg_dark": "#0f172a",      # Slate 900
    "bg_card": "#1e293b",      # Slate 800
    "text": "#f1f5f9",         # Slate 100
    "text_muted": "#94a3b8",   # Slate 400
}

RISK_COLORS = {
    "low": COLORS["low"],
    "moderate": COLORS["moderate"],
    "elevated": COLORS["elevated"],
    "high": COLORS["high"],
}

ENGAGEMENT_LABELS = {
    0: "Very Low",
    1: "Low",
    2: "High",
    3: "Very High",
}


def render_custom_css():
    """Inject custom CSS for a polished, premium glassmorphic dashboard appearance."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Main background */
    .stApp {
        background: radial-gradient(circle at top right, #0d1b2a, #050a10);
        color: #f1f5f9;
        font-family: 'Inter', sans-serif !important;
    }

    h1, h2, h3, h4, h5, h6, span, p {
        font-family: 'Inter', sans-serif !important;
    }

    /* Premium Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 24px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: -100%; width: 50%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent);
        transform: skewX(-30deg);
        transition: 0.6s;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .metric-card:hover::before {
        left: 200%;
    }

    .metric-card h3 {
        color: #94a3b8 !important;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-card .value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }

    /* Dramatic Pulse Alert Banner */
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    .alert-banner {
        background: linear-gradient(135deg, rgba(153, 27, 27, 0.8) 0%, rgba(127, 29, 29, 0.9) 100%);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 18px 24px;
        margin: 16px 0;
        border-left: 5px solid #ef4444;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        display: flex;
        align-items: center;
        gap: 16px;
        animation: pulse-red 2s infinite;
    }

    .alert-banner .alert-icon {
        font-size: 1.8rem;
        filter: drop-shadow(0 0 8px rgba(255,255,255,0.5));
    }

    .alert-banner .alert-text {
        color: #fef2f2;
        font-size: 1rem;
        font-weight: 500;
        letter-spacing: 0.3px;
    }

    /* Glossy Status badge */
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        box-shadow: inset 0 1px 1px rgba(255,255,255,0.2);
    }

    /* Student row upgrades */
    .student-row {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255,255,255,0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .student-row:hover {
        background: rgba(30, 41, 59, 0.8);
        border-color: #3b82f6;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.15);
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent !important;}
    
    /* Header upgrades */
    .dashboard-title-container {
        background: linear-gradient(135deg, rgba(29, 78, 216, 0.4) 0%, rgba(15, 23, 42, 0.8) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 32px 40px;
        margin-bottom: 32px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 40px rgba(0,0,0,0.4);
    }
    
    .dashboard-title-text {
        background: linear-gradient(90deg, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    </style>
    """, unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: str,
    color: str = "#3b82f6",
    subtitle: Optional[str] = None,
):
    """Render a styled metric card."""
    subtitle_html = (
        f'<p style="color: #94a3b8; font-size: 0.8rem; margin: 4px 0 0 0;">'
        f'{subtitle}</p>'
        if subtitle else ""
    )
    st.markdown(f"""
    <div class="metric-card">
        <h3>{label}</h3>
        <p class="value" style="color: {color};">{value}</p>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)


def render_alert_banner(message: str, student_id: Optional[int] = None):
    """Render an alert notification banner."""
    prefix = f"Student {student_id}: " if student_id is not None else ""
    st.markdown(f"""
    <div class="alert-banner">
        <span class="alert-icon">⚠️</span>
        <span class="alert-text"><strong>{prefix}</strong>{message}</span>
    </div>
    """, unsafe_allow_html=True)


def render_risk_gauge(risk_score: float, size: int = 200) -> go.Figure:
    """
    Create a gauge chart for risk score visualization.

    Args:
        risk_score: Risk score between 0 and 1.
        size: Chart size in pixels.

    Returns:
        Plotly Figure with gauge chart.
    """
    # Determine color and label
    if risk_score < 0.25:
        color = COLORS["low"]
        label = "Low"
    elif risk_score < 0.50:
        color = COLORS["moderate"]
        label = "Moderate"
    elif risk_score < 0.75:
        color = COLORS["elevated"]
        label = "Elevated"
    else:
        color = COLORS["high"]
        label = "High"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score * 100,
        number={"suffix": "%", "font": {"size": 24, "color": COLORS["text"]}},
        title={"text": "Observation Level", "font": {"size": 14, "color": COLORS["text_muted"]}},
        gauge={
            "axis": {"range": [0, 100], "dtick": 25, "tickcolor": COLORS["text_muted"]},
            "bar": {"color": color, "thickness": 0.8},
            "bgcolor": COLORS["bg_card"],
            "bordercolor": "#475569",
            "steps": [
                {"range": [0, 25], "color": "rgba(34, 197, 94, 0.15)"},
                {"range": [25, 50], "color": "rgba(245, 158, 11, 0.15)"},
                {"range": [50, 75], "color": "rgba(249, 115, 22, 0.15)"},
                {"range": [75, 100], "color": "rgba(239, 68, 68, 0.15)"},
            ],
            "threshold": {
                "line": {"color": "#ef4444", "width": 2},
                "thickness": 0.8,
                "value": 65,  # Alert threshold
            },
        },
    ))

    fig.update_layout(
        height=size,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text"]},
    )

    return fig


def render_risk_trend(
    history: List[Dict],
    max_points: int = 300,
) -> go.Figure:
    """
    Create a time-series chart of risk score history.

    Args:
        history: List of risk history dictionaries with 'timestamp' and 'smoothed_score'.
        max_points: Maximum number of data points to display.

    Returns:
        Plotly Figure with trend lines.
    """
    if not history:
        # Return empty chart
        fig = go.Figure()
        fig.update_layout(
            title="Risk Trend (No data yet)",
            height=250,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=COLORS["bg_card"],
        )
        return fig

    # Limit data points
    if len(history) > max_points:
        history = history[-max_points:]

    x = list(range(len(history)))
    scores = [h["smoothed_score"] for h in history]

    fig = go.Figure()

    # Risk score line
    fig.add_trace(go.Scatter(
        x=x, y=scores,
        mode="lines",
        name="Observation Score",
        line=dict(color=COLORS["primary"], width=2),
        fill="tozeroy",
        fillcolor="rgba(59, 130, 246, 0.1)",
    ))

    # Alert threshold line
    fig.add_hline(
        y=0.65,
        line_dash="dash",
        line_color=COLORS["high"],
        annotation_text="Alert Threshold",
        annotation_position="top right",
        annotation_font_color=COLORS["text_muted"],
    )

    # Risk zone backgrounds
    fig.add_hrect(y0=0, y1=0.25, fillcolor="rgba(34,197,94,0.05)", line_width=0)
    fig.add_hrect(y0=0.25, y1=0.5, fillcolor="rgba(245,158,11,0.05)", line_width=0)
    fig.add_hrect(y0=0.5, y1=0.75, fillcolor="rgba(249,115,22,0.05)", line_width=0)
    fig.add_hrect(y0=0.75, y1=1.0, fillcolor="rgba(239,68,68,0.05)", line_width=0)

    fig.update_layout(
        title=dict(text="Behavioral Observation Trend", font=dict(size=14, color=COLORS["text"])),
        xaxis=dict(
            title="Time (frames)",
            showgrid=True,
            gridcolor="rgba(71, 85, 105, 0.3)",
            color=COLORS["text_muted"],
        ),
        yaxis=dict(
            title="Score",
            range=[0, 1],
            showgrid=True,
            gridcolor="rgba(71, 85, 105, 0.3)",
            color=COLORS["text_muted"],
        ),
        height=250,
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["bg_card"],
        font=dict(color=COLORS["text"]),
        showlegend=False,
    )

    return fig


def render_engagement_bar(levels: Dict[str, int]) -> go.Figure:
    """
    Create a horizontal bar chart showing engagement levels.

    Args:
        levels: Dictionary mapping task names to level values (0-3).

    Returns:
        Plotly Figure with bar chart.
    """
    tasks = list(levels.keys())
    values = [levels[t] for t in tasks]

    colors = []
    for v in values:
        if v <= 0:
            colors.append(COLORS["high"])
        elif v == 1:
            colors.append(COLORS["elevated"])
        elif v == 2:
            colors.append(COLORS["moderate"])
        else:
            colors.append(COLORS["low"])

    labels = [ENGAGEMENT_LABELS.get(v, "?") for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=[t.capitalize() for t in tasks],
        orientation="h",
        marker_color=colors,
        text=labels,
        textposition="inside",
        textfont=dict(color="white", size=12),
    ))

    fig.update_layout(
        title=dict(text="Behavioral Levels", font=dict(size=14, color=COLORS["text"])),
        xaxis=dict(
            range=[0, 3],
            tickvals=[0, 1, 2, 3],
            ticktext=["Very Low", "Low", "High", "Very High"],
            color=COLORS["text_muted"],
            gridcolor="rgba(71, 85, 105, 0.3)",
        ),
        yaxis=dict(color=COLORS["text"]),
        height=200,
        margin=dict(l=100, r=20, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["bg_card"],
        font=dict(color=COLORS["text"]),
        bargap=0.3,
    )

    return fig


def render_attention_indicator(attention_score: Optional[float]) -> str:
    """
    Generate HTML for attention score indicator.

    Args:
        attention_score: Score between 0 and 1, or None.

    Returns:
        HTML string for the indicator.
    """
    if attention_score is None:
        return '<span style="color: #94a3b8;">N/A</span>'

    pct = int(attention_score * 100)

    if attention_score >= 0.7:
        color = COLORS["low"]
        label = "Focused"
    elif attention_score >= 0.4:
        color = COLORS["moderate"]
        label = "Partial"
    else:
        color = COLORS["high"]
        label = "Distracted"

    return f"""
    <div style="display: flex; align-items: center; gap: 10px;">
        <div style="
            width: 100%;
            background: #334155;
            border-radius: 999px;
            height: 8px;
            overflow: hidden;
        ">
            <div style="
                width: {pct}%;
                background: {color};
                height: 100%;
                border-radius: 999px;
                transition: width 0.3s ease;
            "></div>
        </div>
        <span style="color: {color}; font-weight: 600; white-space: nowrap;">
            {pct}% ({label})
        </span>
    </div>
    """


def render_student_card(state: Dict[str, Any]):
    """
    Render a complete student status card.

    Args:
        state: Student state dictionary from InferencePipeline.
    """
    sid = state.get("student_id", "?")
    risk = state.get("risk", {})
    predictions = state.get("predictions", {})
    attention = state.get("attention", {})

    risk_score = risk.get("risk_score", 0)
    risk_level = risk.get("risk_level", "low")
    risk_color = RISK_COLORS.get(risk_level, COLORS["primary"])
    alert_active = risk.get("alert_active", False)

    with st.container():
        st.markdown(f"""
        <div class="student-row" style="border-left: 4px solid {risk_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0; color: {COLORS['text']};">
                    Student {sid}
                </h4>
                <span class="status-badge" style="
                    background: {risk_color}20;
                    color: {risk_color};
                    border: 1px solid {risk_color};
                ">
                    {risk_level.upper()}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show alert if active
        if alert_active:
            alert_msg = risk.get("alert_message", "Sustained behavioral pattern change detected.")
            render_alert_banner(alert_msg, sid)
