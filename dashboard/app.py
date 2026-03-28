"""
Streamlit Dashboard Application.

Teacher-facing classroom monitoring dashboard displaying:
- Live video feed with face bounding boxes
- Engagement level indicators
- Attention scores
- Risk trend graph over time
- Alert status notifications

IMPORTANT: No medical/clinical terminology is used.
All language is neutral, observational, and supportive.

Usage:
    streamlit run dashboard/app.py -- --config config/config.yaml
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Optional

import cv2
import numpy as np
import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.components import (
    render_custom_css,
    render_metric_card,
    render_alert_banner,
    render_risk_gauge,
    render_risk_trend,
    render_engagement_bar,
    render_attention_indicator,
    render_student_card,
    COLORS,
)


# ============================================================
#  Page Configuration
# ============================================================
st.set_page_config(
    page_title="Classroom Observation Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_custom_css()


# ============================================================
#  Session State Initialization
# ============================================================
def init_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        "pipeline": None,
        "running": False,
        "frame_count": 0,
        "student_states": [],
        "risk_histories": {},
        "alerts": [],
        "config": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ============================================================
#  Sidebar Configuration
# ============================================================
def render_sidebar():
    """Render sidebar configuration panel."""
    with st.sidebar:
        st.markdown(
            '<h2 style="color: #f1f5f9;">⚙️ Configuration</h2>',
            unsafe_allow_html=True,
        )

        st.divider()

        # Input source
        st.subheader("📹 Input Source")
        source_type = st.radio(
            "Select input",
            ["Camera", "Video File", "Demo Mode"],
            index=2,
            label_visibility="collapsed",
        )

        camera_id = 0
        video_path = None

        if source_type == "Camera":
            camera_id = st.number_input("Camera ID", 0, 10, 0)
        elif source_type == "Video File":
            video_path = st.text_input("Video file path")

        st.divider()

        # Model settings
        st.subheader("🧠 Model Settings")
        checkpoint_path = st.text_input(
            "Checkpoint path",
            value="checkpoints/best_model.pth",
        )

        confidence = st.slider(
            "Face Detection Confidence",
            0.1, 1.0, 0.5,
            step=0.05,
        )

        st.divider()

        # Alert settings
        st.subheader("🔔 Alert Settings")
        alert_threshold = st.slider(
            "Alert Threshold",
            0.0, 1.0, 0.65,
            step=0.05,
            help="Observation score above this triggers attention",
        )

        persistence = st.slider(
            "Persistence Duration (seconds)",
            5, 120, 30,
            step=5,
            help="How long a pattern must persist before notification",
        )

        st.divider()

        # Controls
        col1, col2 = st.columns(2)
        with col1:
            start_btn = st.button(
                "▶ Start",
                use_container_width=True,
                type="primary",
            )
        with col2:
            stop_btn = st.button(
                "⏹ Stop",
                use_container_width=True,
            )

        return {
            "source_type": source_type,
            "camera_id": camera_id,
            "video_path": video_path,
            "checkpoint_path": checkpoint_path,
            "confidence": confidence,
            "alert_threshold": alert_threshold,
            "persistence": persistence,
            "start": start_btn,
            "stop": stop_btn,
        }


# ============================================================
#  Demo Mode Data Generator
# ============================================================
class DemoDataGenerator:
    """Generates simulated student data for demo mode."""

    def __init__(self, num_students: int = 4):
        self.num_students = num_students
        self.frame_idx = 0
        self._risk_histories = {i: [] for i in range(num_students)}

    def generate_frame_data(self) -> List[Dict]:
        """Generate simulated student states for one frame."""
        self.frame_idx += 1
        states = []

        for sid in range(self.num_students):
            # Simulate varying behavioral patterns
            t = self.frame_idx / 30.0  # Time in "seconds"
            phase = sid * 1.5  # Phase offset per student

            # Engagement varies sinusoidally with noise
            eng_base = 1.5 + 1.2 * np.sin(t * 0.3 + phase)
            engagement = int(np.clip(eng_base + np.random.normal(0, 0.3), 0, 3))

            # Boredom inversely correlated with engagement
            boredom = int(np.clip(3 - engagement + np.random.randint(-1, 2), 0, 3))

            # Confusion with occasional spikes
            confusion_spike = 2 if (self.frame_idx + sid * 50) % 200 < 30 else 0
            confusion = int(np.clip(
                np.random.choice([0, 0, 1]) + confusion_spike, 0, 3
            ))

            # Frustration builds slowly
            frustration = int(np.clip(
                np.random.choice([0, 0, 0, 1]) +
                (1 if engagement == 0 and confusion >= 2 else 0),
                0, 3,
            ))

            # Attention score
            attention = np.clip(0.5 + 0.4 * np.cos(t * 0.2 + phase) + np.random.normal(0, 0.1), 0, 1)

            # Risk score computation (simplified)
            low_eng = 1.0 if engagement < 2 else 0.0
            conf_pers = confusion / 3.0
            frus_pers = frustration / 3.0
            attn_dev = 1.0 - attention

            risk_score = 0.35 * low_eng + 0.25 * conf_pers + 0.25 * frus_pers + 0.15 * attn_dev
            risk_score = np.clip(risk_score, 0, 1)

            risk_level = (
                "low" if risk_score < 0.25
                else "moderate" if risk_score < 0.5
                else "elevated" if risk_score < 0.75
                else "high"
            )

            alert_active = risk_score > 0.65 and self.frame_idx % 100 > 70

            history_entry = {
                "timestamp": time.time(),
                "smoothed_score": risk_score,
            }
            self._risk_histories[sid].append(history_entry)

            # Keep history bounded
            if len(self._risk_histories[sid]) > 300:
                self._risk_histories[sid] = self._risk_histories[sid][-300:]

            state = {
                "student_id": sid,
                "bbox": (50 + sid * 160, 50, 200 + sid * 160, 250),
                "predictions": {
                    "engagement": engagement,
                    "boredom": boredom,
                    "confusion": confusion,
                    "frustration": frustration,
                },
                "attention": {
                    "yaw": float(np.random.normal(0, 15)),
                    "pitch": float(np.random.normal(0, 10)),
                    "roll": float(np.random.normal(0, 5)),
                    "attention_score": float(attention),
                    "is_off_task": attention < 0.4,
                },
                "risk": {
                    "risk_score": float(risk_score),
                    "risk_level": risk_level,
                    "alert_active": alert_active,
                    "alert_message": (
                        "This student has shown a sustained pattern of low participation. "
                        "Consider checking in or offering support."
                    ) if alert_active else None,
                    "component_scores": {
                        "low_engagement": 0.35 * low_eng,
                        "confusion": 0.25 * conf_pers,
                        "frustration": 0.25 * frus_pers,
                        "attention_deviation": 0.15 * attn_dev,
                    },
                },
                "risk_history": list(self._risk_histories[sid]),
            }
            states.append(state)

        return states


# ============================================================
#  Main Dashboard Layout
# ============================================================
def render_header():
    """Render the dashboard header with new premium CSS."""
    st.markdown("""
    <div class="dashboard-title-container">
        <h1 class="dashboard-title-text">
            🎓 Classroom Intelligence
        </h1>
        <p style="margin: 8px 0 0 0; color: #94a3b8; font-size: 1.05rem; font-weight: 500;">
            Real-time behavioral patterning • Accelerated AI Engine
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_overview_metrics(student_states: List[Dict]):
    """Render top-level classroom overview metrics."""
    n_students = len(student_states)

    if n_students == 0:
        cols = st.columns(4)
        with cols[0]:
            render_metric_card("Students Detected", "0", COLORS["primary"])
        return

    # Compute aggregates
    avg_engagement = np.mean([
        s["predictions"]["engagement"] for s in student_states
    ])
    avg_attention = np.mean([
        s["attention"].get("attention_score", 0.5) or 0.5
        for s in student_states
    ])
    avg_risk = np.mean([
        s["risk"]["risk_score"] for s in student_states
    ])
    n_alerts = sum(1 for s in student_states if s["risk"].get("alert_active"))

    cols = st.columns(4)

    with cols[0]:
        render_metric_card(
            "Students Detected",
            str(n_students),
            COLORS["primary"],
        )
    with cols[1]:
        eng_color = COLORS["low"] if avg_engagement >= 2 else COLORS["elevated"]
        eng_labels = {0: "Very Low", 1: "Low", 2: "High", 3: "Very High"}
        eng_text = eng_labels.get(round(avg_engagement), "—")
        render_metric_card(
            "Avg Engagement",
            eng_text,
            eng_color,
            f"Score: {avg_engagement:.1f}/3",
        )
    with cols[2]:
        attn_color = COLORS["low"] if avg_attention >= 0.6 else COLORS["elevated"]
        render_metric_card(
            "Avg Attention",
            f"{avg_attention:.0%}",
            attn_color,
        )
    with cols[3]:
        alert_color = COLORS["high"] if n_alerts > 0 else COLORS["low"]
        render_metric_card(
            "Active Notifications",
            str(n_alerts),
            alert_color,
            "Students may need check-in" if n_alerts > 0 else "All clear",
        )


def render_student_details(student_states: List[Dict]):
    """Render detailed per-student panels."""
    if not student_states:
        st.info("No students detected. Waiting for data...")
        return

    for state in student_states:
        sid = state["student_id"]
        risk = state["risk"]
        predictions = state["predictions"]
        attention = state["attention"]

        # Student card with expander
        risk_level = risk.get("risk_level", "low")
        risk_score = risk.get("risk_score", 0)

        render_student_card(state)

        with st.expander(f"📊 Details for Student {sid}", expanded=(risk_level in ["elevated", "high"])):
            col1, col2 = st.columns([1, 1])

            with col1:
                # Risk gauge
                fig = render_risk_gauge(risk_score)
                st.plotly_chart(fig, use_container_width=True, key=f"gauge_{sid}_{st.session_state.frame_count}")

                # Attention indicator
                st.markdown("**Attention Level**")
                attn_html = render_attention_indicator(attention.get("attention_score"))
                st.markdown(attn_html, unsafe_allow_html=True)

                # Head pose info
                if attention.get("yaw") is not None:
                    st.caption(
                        f"Head orientation: Yaw {attention['yaw']:.0f}°, "
                        f"Pitch {attention['pitch']:.0f}°, "
                        f"Roll {attention['roll']:.0f}°"
                    )

            with col2:
                # Behavioral levels
                fig = render_engagement_bar(predictions)
                st.plotly_chart(fig, use_container_width=True, key=f"bar_{sid}_{st.session_state.frame_count}")

                # Component breakdown
                components = risk.get("component_scores", {})
                if components:
                    st.markdown("**Observation Breakdown**")
                    for comp, score in components.items():
                        label = comp.replace("_", " ").title()
                        pct = score * 100
                        st.progress(min(score, 1.0), text=f"{label}: {pct:.0f}%")

            # Risk trend
            history = state.get("risk_history", [])
            if history:
                fig = render_risk_trend(history)
                st.plotly_chart(fig, use_container_width=True, key=f"trend_{sid}_{st.session_state.frame_count}")


# ============================================================
#  Main Application
# ============================================================
def main():
    render_header()
    sidebar_config = render_sidebar()

    # Handle start/stop
    if sidebar_config["stop"]:
        st.session_state.running = False

    if sidebar_config["start"]:
        st.session_state.running = True

    # Main content area
    if sidebar_config["source_type"] == "Demo Mode":
        # Demo mode — generate simulated data
        if "demo_gen" not in st.session_state:
            st.session_state.demo_gen = DemoDataGenerator(num_students=4)

        if st.session_state.running:
            student_states = st.session_state.demo_gen.generate_frame_data()
            st.session_state.student_states = student_states
            st.session_state.frame_count += 1
        else:
            student_states = st.session_state.student_states

        # Render dashboard
        render_overview_metrics(student_states)

        st.divider()

        # Alert section
        active_alerts = [s for s in student_states if s["risk"].get("alert_active")]
        if active_alerts:
            st.markdown(
                '<h3 style="color: #f1f5f9;">🔔 Notifications</h3>',
                unsafe_allow_html=True,
            )
            for s in active_alerts:
                render_alert_banner(
                    s["risk"].get("alert_message", "Pattern change detected."),
                    s["student_id"],
                )
            st.divider()

        # Student details
        st.markdown(
            '<h3 style="color: #f1f5f9;">👥 Student Details</h3>',
            unsafe_allow_html=True,
        )
        render_student_details(student_states)

        # Auto-refresh in demo mode
        if st.session_state.running:
            time.sleep(1)
            st.rerun()

    else:
        # Live / Video mode
        if not st.session_state.running:
            st.markdown("""
            <div class="metric-card">
                <h3>Live Monitoring Mode</h3>
                <p style="color: #94a3b8;">
                    To use live monitoring:<br>
                    1. Ensure camera is connected or provide video file path<br>
                    2. Load a trained model checkpoint<br>
                    3. Click <strong>Start</strong> in the sidebar
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Clean up if stopped
            if "cap" in st.session_state:
                st.session_state.cap.release()
                del st.session_state["cap"]
            if "pipeline" in st.session_state:
                del st.session_state["pipeline"]

        if st.session_state.running:
            # Initialize pipeline if not present
            if "pipeline" not in st.session_state or st.session_state.pipeline is None:
                from src.inference.pipeline import InferencePipeline
                from src.utils.helpers import load_config
                try:
                    config = load_config("config/config.yaml")

                    # Apply sidebar overrides
                    config["preprocessing"]["face_min_confidence"] = sidebar_config["confidence"]
                    config["risk_fusion"]["alert_threshold"] = sidebar_config["alert_threshold"]
                    config["risk_fusion"]["persistence_duration"] = sidebar_config["persistence"]

                    st.session_state.pipeline = InferencePipeline(
                        config=config, 
                        checkpoint_path=sidebar_config["checkpoint_path"]
                    )
                except Exception as e:
                    st.error(f"Failed to load model from {sidebar_config['checkpoint_path']}: {e}")
                    st.session_state.running = False
                    st.rerun()

            # Initialize capture
            if "cap" not in st.session_state:
                if sidebar_config["source_type"] == "Camera":
                    st.session_state.cap = cv2.VideoCapture(sidebar_config["camera_id"])
                else:
                    path = sidebar_config["video_path"]
                    if not path:
                        st.error("Please enter a valid video path.")
                        st.session_state.running = False
                        st.rerun()
                    st.session_state.cap = cv2.VideoCapture(path)

            # Flush the OpenCV queue to force absolute real-time (prevents creeping lag)
            if sidebar_config["source_type"] == "Camera":
                for _ in range(4):
                    st.session_state.cap.grab()
            
            ret, frame = st.session_state.cap.read()
            if not ret:
                st.warning("Video stream ended or camera disconnected.")
                st.session_state.running = False
                st.session_state.cap.release()
                del st.session_state["cap"]
                st.rerun()

            # Process frame
            result = st.session_state.pipeline.process_frame(frame)
            
            # 1. Show live video feed
            annotated_rgb = cv2.cvtColor(result["annotated_frame"], cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, channels="RGB", use_container_width=True)
            st.caption(f"Processing FPS: {1000/max(result['frame_info']['processing_time_ms'], 1):.1f}")
            
            # 2. Show metrics
            render_overview_metrics(result["students"])

            st.divider()

            # 3. Alert section
            active_alerts = [s for s in result["students"] if s["risk"].get("alert_active")]
            if active_alerts:
                st.markdown(
                    '<h3 style="color: #f1f5f9;">🔔 Notifications</h3>',
                    unsafe_allow_html=True,
                )
                for s in active_alerts:
                    render_alert_banner(
                        s["risk"].get("alert_message", "Pattern change detected."),
                        s["student_id"],
                    )
                st.divider()

            # 4. Show student details below
            st.markdown(
                '<h3 style="color: #f1f5f9;">👥 Student Details</h3>',
                unsafe_allow_html=True,
            )
            render_student_details(result["students"])

            # Rerun continuously to capture next frame
            time.sleep(0.05)
            st.rerun()

    # Footer
    st.markdown("---")
    st.caption(
        "⚠️ This system provides behavioral observations only. "
        "It does not diagnose any conditions. All observations should be "
        "interpreted by qualified educators in context."
    )


if __name__ == "__main__":
    main()
