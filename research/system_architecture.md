# System Architecture

## Computer Vision–Based Classroom Monitoring System for Early Detection of Distress and Concentration Risk Indicators in Students

---

## 1. System Overview

This system is a **decision-support tool** for educators that monitors behavioral patterns in classroom settings through computer vision analysis. It does **NOT** diagnose mental health conditions — it detects **sustained behavioral risk indicators** and provides interpretable, neutral observations to teachers.

### Core Philosophy
- **Observation, not diagnosis**
- **Neutral, interpretable outputs**
- **Ethically responsible design**
- **Decision-support, not decision-making**

---

## 2. Architecture Components

### Branch 1: Affective Behavioral Model
A multi-task Convolutional Neural Network based on a pretrained backbone (ResNet-18 or EfficientNet-B0) that simultaneously classifies four behavioral dimensions from facial regions:

| Dimension   | Levels (0–3)                                     |
|-------------|--------------------------------------------------|
| Engagement  | Very Low → Low → High → Very High                |
| Boredom     | Very Low → Low → High → Very High                |
| Confusion   | Very Low → Low → High → Very High                |
| Frustration | Very Low → Low → High → Very High                |

The backbone produces a shared feature embedding (256-d) that is passed to the temporal modeling branch.

### Branch 2: Attention Estimation Module
Uses MediaPipe Face Mesh to detect 468+ facial landmarks and estimates head pose (yaw, pitch, roll) through the Perspective-n-Point (PnP) algorithm. Angular deviations from the task direction are converted to a normalized attention score via exponential decay.

### Branch 3: Temporal Behavior Modeling
- **Sliding Window Smoother**: Maintains rolling buffers for each behavioral signal and computes persistence ratios, smoothed values, and trend slopes over a configurable window.
- **LSTM/GRU Model (optional)**: Takes sequences of frame embeddings and applies temporal attention for sequence-level classification.

### Risk Fusion Engine
An interpretable weighted fusion module that combines all branch outputs into a single risk indicator score. Includes EMA smoothing, persistence-based alert triggering, and false-positive suppression via cooldown periods.

---

## 3. Data Flow

```
Input Frame
    │
    ├── Face Detection (MediaPipe)
    │       │
    │       ├── Face Crop → Branch 1 (CNN Multi-Task Model)
    │       │                    ├── Engagement logits
    │       │                    ├── Boredom logits
    │       │                    ├── Confusion logits
    │       │                    ├── Frustration logits
    │       │                    └── 256-d embedding → Branch 3 (Temporal)
    │       │
    │       └── Face Region → Branch 2 (Head Pose via Face Mesh)
    │                              ├── Yaw, Pitch, Roll
    │                              └── Attention Score
    │
    └── Temporal Buffer
            │
            ├── Sliding Window Trends
            │       ├── Low engagement ratio
            │       ├── Confusion persistence
            │       ├── Frustration persistence
            │       └── Off-task head orientation ratio
            │
            └── Risk Fusion Engine
                    ├── Weighted composite score
                    ├── EMA smoothing
                    ├── Persistence check
                    ├── Alert trigger / suppression
                    └── Dashboard output
```

---

## 4. Dashboard

The Streamlit-based teacher dashboard provides:
- **Overview metrics**: Total students, average engagement, average attention, active notifications
- **Per-student panels**: Risk gauge, behavioral level bars, attention indicator, component breakdown
- **Trend visualization**: Time-series plot of observation scores with alert threshold line
- **Notifications**: Neutral alert messages suggesting teacher check-in when sustained patterns are detected
