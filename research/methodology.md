# Experimental Methodology

## Computer Vision–Based Classroom Monitoring System

---

## 1. Dataset

### DAiSEE (Dataset for Affective States in E-Environments)
- **Size**: ~9,068 video clips from 112 participants
- **Resolution**: 640×480 pixels, 30 FPS
- **Format**: AVI files (~10 seconds each)
- **Labels**: 4 behavioral dimensions × 4 ordinal levels (0–3)
  - Engagement, Boredom, Confusion, Frustration
- **Annotation**: Crowd-sourced labels via Amazon Mechanical Turk

### Preprocessing Pipeline
1. **Frame Extraction**: Sample at 5 FPS (configurable) for temporal coverage without redundancy
2. **Face Detection**: MediaPipe Face Detection with 0.5 confidence threshold and 20% padding
3. **Face Cropping**: Resize to 224×224 for backbone input
4. **Split Strategy**: User-level GroupShuffleSplit (70/15/15) to prevent data leakage

---

## 2. Model Training

### Architecture
- **Backbone**: ResNet-18 pretrained on ImageNet
- **Embedding**: 256-dimensional shared feature vector
- **Task Heads**: 4 independent classification heads (128 → C)
- **Total Parameters**: ~12M (11.7M backbone + shared + heads)

### Training Protocol
| Parameter         | Value              |
|-------------------|--------------------|
| Optimizer         | AdamW              |
| Learning Rate     | 1e-3               |
| Weight Decay      | 1e-4               |
| Scheduler         | Cosine Annealing   |
| Batch Size        | 32                 |
| Epochs            | 50 (early stop: 7) |
| Gradient Clipping | 1.0                |
| Backbone Freeze   | First 5 epochs     |

### Data Augmentation
- Random horizontal flip (p=0.5)
- Random rotation (±10°)
- Color jitter (brightness=0.2, contrast=0.2, saturation=0.1)
- Random grayscale (p=0.05)
- ImageNet normalization

---

## 3. Evaluation Protocol

### Metrics
- **Per-task**: Accuracy, Weighted F1, Macro F1, Precision, Recall
- **Overall**: Mean accuracy and F1 across all 4 tasks
- **Confusion Matrices**: Normalized per-class visualization

### Ablation Studies

| Configuration     | Components                           |
|-------------------|--------------------------------------|
| Facial Only       | CNN multi-task model                 |
| Facial + HeadPose | CNN + MediaPipe head pose + attention|
| Full System       | CNN + HeadPose + Temporal + Risk     |

### Temporal Evaluation
- Case study visualizations showing behavioral trajectories for individual students over time
- Risk trend plots with alert threshold markers
- Comparison of temporal smoothing vs. per-frame predictions

---

## 4. Real-Time Performance

### Inference Benchmarks (Target)
| Component           | Target Latency (CPU) |
|---------------------|----------------------|
| Face Detection      | < 15 ms/frame        |
| CNN Inference       | < 30 ms/face         |
| Head Pose           | < 10 ms/face         |
| Temporal + Fusion   | < 1 ms/face          |
| **Total (1 face)**  | **< 60 ms/frame**   |

### Scalability
- Tested with up to 30 simultaneous faces
- IoU-based simple tracker for student ID assignment
- Automatic stale tracker cleanup (5-second timeout)

---

## 5. Experimental Variations

### Recommended Experiments
1. **Backbone comparison**: ResNet-18 vs EfficientNet-B0
2. **Loss function**: Standard CE vs Focal Loss (γ=2)
3. **Temporal window size**: 15, 30, 60 frames
4. **Risk weight sensitivity**: Vary w1–w4 and observe alert behavior
5. **Alert threshold sweep**: 0.4, 0.5, 0.6, 0.65, 0.7, 0.8
6. **Persistence duration**: 15s, 30s, 60s, 90s

### Reproducibility
- Seed = 42 for all random operations
- User-level splits prevent train/test contamination
- All hyperparameters centralized in `config/config.yaml`
- Checkpoint saving every 5 epochs + best model tracking
