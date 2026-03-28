# Mathematical Formulations

## Computer Vision–Based Classroom Monitoring System

---

## 1. Multi-Task Loss Function

The model is trained with a weighted multi-task cross-entropy loss:

$$
\mathcal{L}_{total} = \sum_{k \in \mathcal{T}} \lambda_k \cdot \mathcal{L}_{CE}^{(k)}
$$

where:
- $\mathcal{T} = \{\text{engagement}, \text{boredom}, \text{confusion}, \text{frustration}\}$
- $\lambda_k$ are per-task loss weights
- $\mathcal{L}_{CE}^{(k)} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=0}^{C-1} y_{i,c}^{(k)} \log(\hat{p}_{i,c}^{(k)})$

### Optional Focal Loss Variant

For class-imbalanced tasks:

$$
\mathcal{L}_{FL}^{(k)} = -\frac{1}{N} \sum_{i=1}^{N} \alpha_t (1 - \hat{p}_{i,t})^\gamma \log(\hat{p}_{i,t})
$$

where $\hat{p}_{i,t}$ is the predicted probability for the true class and $\gamma$ is the focusing parameter (default: 2.0).

---

## 2. Attention Score from Head Pose

### Head Pose Estimation

Head pose angles $(\text{yaw}, \text{pitch}, \text{roll})$ are estimated by solving the Perspective-n-Point (PnP) problem:

Given 3D model points $\mathbf{P}_i \in \mathbb{R}^3$ and corresponding 2D image points $\mathbf{p}_i \in \mathbb{R}^2$:

$$
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \begin{bmatrix} \mathbf{R} | \mathbf{t} \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
$$

where $\mathbf{K}$ is the camera intrinsic matrix, $\mathbf{R}$ is the rotation matrix, and $\mathbf{t}$ is the translation vector.

### Attention Score Computation

Angular deviation from task direction:

$$
d_{\text{yaw}} = \max\left(0, \frac{|\text{yaw}| - 0.3 \cdot \theta_{\text{yaw}}}{\theta_{\text{yaw}}}\right)
$$

$$
d_{\text{pitch}} = \max\left(0, \frac{|\text{pitch}| - 0.3 \cdot \theta_{\text{pitch}}}{\theta_{\text{pitch}}}\right)
$$

$$
d_{\text{roll}} = \max\left(0, \frac{|\text{roll}| - 0.3 \cdot \theta_{\text{roll}}}{\theta_{\text{roll}}}\right)
$$

Weighted total deviation:

$$
D = 0.5 \cdot d_{\text{yaw}} + 0.3 \cdot d_{\text{pitch}} + 0.2 \cdot d_{\text{roll}}
$$

Attention score with exponential decay:

$$
A(t) = e^{-\beta \cdot 10 \cdot D}
$$

where $\beta$ is the decay rate parameter.

---

## 3. Temporal Trend Computation

### Low Engagement Ratio

$$
E_{\text{low}}(t) = \frac{1}{W} \sum_{i=t-W}^{t} \mathbb{1}[\text{engagement}_i < \tau_e]
$$

### Confusion Persistence

$$
C_{\text{persist}}(t) = \frac{1}{W} \sum_{i=t-W}^{t} \mathbb{1}[\text{confusion}_i \geq 2]
$$

### Frustration Persistence

$$
F_{\text{persist}}(t) = \frac{1}{W} \sum_{i=t-W}^{t} \mathbb{1}[\text{frustration}_i \geq 2]
$$

### Off-Task Head Orientation Ratio

$$
A_{\text{dev}}(t) = \frac{1}{W} \sum_{i=t-W}^{t} \mathbb{1}[A_i < 0.5]
$$

---

## 4. Risk Score Fusion

### Composite Risk Score

$$
R(t) = w_1 \cdot E_{\text{low}}(t) + w_2 \cdot C_{\text{persist}}(t) + w_3 \cdot F_{\text{persist}}(t) + w_4 \cdot A_{\text{dev}}(t)
$$

Default weights: $w_1 = 0.35, w_2 = 0.25, w_3 = 0.25, w_4 = 0.15$

### Normalization

$$
R_{\text{norm}}(t) = \text{clip}\left(\frac{R(t)}{\sum_{i} w_i}, 0, 1\right)
$$

### Exponential Moving Average Smoothing

$$
\hat{R}(t) = \alpha \cdot R_{\text{norm}}(t) + (1 - \alpha) \cdot \hat{R}(t-1)
$$

where $\alpha = 0.3$ is the smoothing factor.

---

## 5. Alert Triggering Logic

### Persistence-Based Alert

$$
\text{Alert}(t) = \begin{cases}
1 & \text{if } \hat{R}(t) > \theta_{\text{alert}} \text{ for } \geq T_{\text{persist}} \text{ consecutive frames} \\
  & \text{and } (t - t_{\text{last\_alert}}) > T_{\text{cooldown}} \\
0 & \text{otherwise}
\end{cases}
$$

### Hysteresis for Alert Clearing

Alert is cleared when $\hat{R}(t) < 0.8 \cdot \theta_{\text{alert}}$.

The consecutive high-risk frame counter decays by 2 per frame when below threshold (for hysteresis):

$$
n_{\text{high}}(t) = \max\left(0, n_{\text{high}}(t-1) - 2\right) \quad \text{if } \hat{R}(t) < \theta_{\text{alert}}
$$

---

## 6. LSTM Temporal Attention

For the optional LSTM branch, temporal attention weights are computed:

$$
\alpha_t = \text{softmax}\left(\mathbf{v}^\top \tanh(\mathbf{W} \mathbf{h}_t + \mathbf{b})\right)
$$

Context vector:

$$
\mathbf{c} = \sum_{t=1}^{T} \alpha_t \mathbf{h}_t
$$

Task predictions from context:

$$
\hat{y}_k = \text{softmax}(\mathbf{W}_k \mathbf{c} + \mathbf{b}_k)
$$
