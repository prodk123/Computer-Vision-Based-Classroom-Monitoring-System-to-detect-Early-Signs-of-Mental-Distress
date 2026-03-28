# 🎓 Computer Vision–Based Classroom Monitoring System

**For Early Detection of Distress and Concentration Risk Indicators in Students**

---

> ⚠️ **Important**: This system does NOT diagnose mental health disorders. It detects sustained behavioral risk indicators and provides decision-support insights to teachers. All outputs are neutral, interpretable, and ethically responsible.

---

## 📋 Overview

A research-grade prototype system that monitors student behavioral patterns in classroom settings using computer vision. The system:

1. **Estimates engagement levels** from facial behavioral cues
2. **Detects sustained confusion, boredom, and frustration** patterns
3. **Estimates attention** using head pose deviation from task direction
4. **Models temporal behavioral trends** over time (not single-frame predictions)
5. **Fuses multiple behavioral signals** into a continuous risk indicator score
6. **Generates alerts** only when abnormal patterns persist over a defined duration
7. **Provides a real-time dashboard** prototype for teachers

---

## 🏗️ Architecture

```
Camera → Face Detection → ┬─ CNN Multi-Task Model (Branch 1)
                           │    ├── Engagement
                           │    ├── Boredom
                           │    ├── Confusion
                           │    └── Frustration
                           │
                           ├─ Head Pose Estimation (Branch 2)
                           │    └── Attention Score
                           │
                           └─ Temporal Modeling (Branch 3)
                                └── Sliding Window + LSTM
                                        │
                                   Risk Fusion Engine
                                        │
                                   Dashboard Output
```

---

## 📁 Project Structure

```
├── config/config.yaml          # All hyperparameters & settings
├── src/
│   ├── preprocessing/          # Frame extraction, face detection, dataset building
│   ├── models/                 # Multi-task CNN, attention estimator, temporal model, risk fusion
│   ├── training/               # Trainer, losses, metrics
│   ├── inference/              # Real-time inference pipeline
│   ├── evaluation/             # Evaluator, ablation studies
│   └── utils/                  # Logging, helpers
├── dashboard/                  # Streamlit teacher dashboard
├── scripts/                    # CLI scripts for preprocessing, training, evaluation, inference
├── research/                   # Architecture docs, math formulations, methodology, ethics
├── checkpoints/                # Saved model weights
├── data/                       # Dataset storage
├── logs/                       # Training logs
├── outputs/                    # Evaluation outputs
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone and install dependencies
cd "SGP 3"
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### 2. Dataset Preparation

Download the [DAiSEE dataset](https://iith.ac.in/~daisee/) and place it at:
```
data/raw/DAiSEE/
├── DataSet/
│   ├── Train/{UserID}/{ClipID}.avi
│   ├── Validation/
│   └── Test/
└── Labels/
    ├── TrainLabels.csv
    ├── ValidationLabels.csv
    └── TestLabels.csv
```

### 3. Preprocessing

```bash
python scripts/preprocess.py --config config/config.yaml
```

This will:
- Extract frames at 5 FPS from all videos
- Detect and crop faces
- Build dataset DataFrame
- Create train/val/test splits

### 4. Training

```bash
# Default training
python scripts/train.py --config config/config.yaml

# With custom settings
python scripts/train.py --config config/config.yaml --backbone resnet18 --epochs 50 --lr 0.001

# Resume from checkpoint
python scripts/train.py --config config/config.yaml --resume checkpoints/checkpoint_epoch_10.pth
```

### 5. Evaluation

```bash
# Standard evaluation
python scripts/evaluate.py --config config/config.yaml --checkpoint checkpoints/best_model.pth

# With ablation study
python scripts/evaluate.py --config config/config.yaml --checkpoint checkpoints/best_model.pth --ablation
```

### 6. Real-Time Inference

```bash
# Live camera
python scripts/run_inference.py --config config/config.yaml --checkpoint checkpoints/best_model.pth --camera 0

# Video file
python scripts/run_inference.py --config config/config.yaml --checkpoint checkpoints/best_model.pth --video path/to/video.mp4
```

### 7. Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard has a **Demo Mode** that works without a trained model, generating simulated student data for UI exploration.

---

## 🧪 Ablation Studies

| Configuration     | Components                          | Purpose                     |
|-------------------|-------------------------------------|-----------------------------|
| `facial_only`     | CNN multi-task model                | Baseline facial features    |
| `facial_headpose` | CNN + Head pose attention           | + Attention estimation      |
| `full_system`     | CNN + Head pose + Temporal + Risk   | Complete pipeline           |

---

## 📊 Risk Score Formula

```
RiskScore(t) = w1 × LowEngagementRatio
             + w2 × ConfusionPersistence
             + w3 × FrustrationPersistence
             + w4 × OffTaskAttentionRatio
```

Default weights: w1=0.35, w2=0.25, w3=0.25, w4=0.15

Alerts trigger when:
- Risk score exceeds threshold (0.65) for ≥30 sustained seconds
- And cooldown period (60s) from last alert has elapsed

---

## ⚖️ Ethics & Responsible Use

This system is designed as a **decision-support tool**, not a diagnosis engine:

- ✅ Uses neutral, non-medical language
- ✅ Only alerts on sustained patterns (never single-frame)
- ✅ Fully interpretable risk score decomposition
- ✅ Configurable thresholds and weights
- ❌ Never diagnoses conditions
- ❌ Never used for grading or discipline
- ❌ Never stores video or personal data permanently

See [research/ethics_limitations.md](research/ethics_limitations.md) for full ethical guidelines.

---

## 📚 Research Documentation

| Document                                                          | Description                        |
|-------------------------------------------------------------------|------------------------------------|
| [System Architecture](research/system_architecture.md)            | Full system design                 |
| [Mathematical Formulation](research/math_formulation.md)          | All equations and derivations      |
| [Experimental Methodology](research/methodology.md)               | Training protocol and benchmarks   |
| [Ethics & Limitations](research/ethics_limitations.md)            | Responsible use guidelines         |

---

## 🔧 Configuration

All hyperparameters are centralized in `config/config.yaml`:

- **Preprocessing**: FPS, face detection settings, crop size
- **Model**: Backbone, embedding dimension, dropout
- **Training**: Epochs, learning rate, scheduler, task weights
- **Temporal**: Window size, LSTM settings
- **Risk Fusion**: Weights, thresholds, persistence, cooldown
- **Inference**: Device, camera settings, max faces
- **Dashboard**: Refresh rate, history length

---

## 📦 Dependencies

- PyTorch ≥ 2.0
- MediaPipe ≥ 0.10
- OpenCV ≥ 4.8
- Streamlit ≥ 1.28
- Plotly ≥ 5.17
- scikit-learn, NumPy, Pandas, Matplotlib, Seaborn

---

## 📄 License

This project is for academic and research purposes only.
