"""
Temporal Behavior Modeling (Branch 3).

Implements:
1. Sliding window temporal smoothing for behavioral signals.
2. Optional LSTM/GRU temporal model using frame embeddings.
3. Computation of sustained behavioral trend metrics.
"""

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("classroom_monitor.models")


class SlidingWindowSmoother:
    """
    Applies sliding window temporal smoothing to behavioral signals.

    Maintains a buffer of recent observations and computes
    smoothed statistics (mean, ratio of threshold crossings).

    Args:
        window_size: Number of frames in the sliding window.
        stride: Step size for window advancement.
    """

    def __init__(self, window_size: int = 30, stride: int = 1):
        self.window_size = window_size
        self.stride = stride
        self._buffers: Dict[str, deque] = {}

    def update(self, signal_name: str, value: float) -> None:
        """
        Add a new observation to the named signal buffer.

        Args:
            signal_name: Name of the signal (e.g., 'engagement').
            value: New observation value.
        """
        if signal_name not in self._buffers:
            self._buffers[signal_name] = deque(maxlen=self.window_size)
        self._buffers[signal_name].append(value)

    def get_smoothed_value(self, signal_name: str) -> Optional[float]:
        """
        Get the mean value over the current window.

        Args:
            signal_name: Name of the signal.

        Returns:
            Mean value, or None if buffer is empty.
        """
        if signal_name not in self._buffers or len(self._buffers[signal_name]) == 0:
            return None
        return float(np.mean(list(self._buffers[signal_name])))

    def get_threshold_ratio(
        self, signal_name: str, threshold: float, below: bool = True
    ) -> Optional[float]:
        """
        Compute the ratio of observations below (or above) a threshold.

        Args:
            signal_name: Name of the signal.
            threshold: Threshold value.
            below: If True, count values below threshold. If False, above.

        Returns:
            Ratio of threshold crossings in [0, 1], or None if buffer empty.
        """
        if signal_name not in self._buffers or len(self._buffers[signal_name]) == 0:
            return None

        values = list(self._buffers[signal_name])
        if below:
            count = sum(1 for v in values if v < threshold)
        else:
            count = sum(1 for v in values if v > threshold)

        return count / len(values)

    def get_trend(self, signal_name: str) -> Optional[float]:
        """
        Compute a simple linear trend (slope) of the signal over the window.

        Positive slope = increasing trend, negative = decreasing.

        Args:
            signal_name: Name of the signal.

        Returns:
            Slope of linear fit, or None if insufficient data.
        """
        if signal_name not in self._buffers or len(self._buffers[signal_name]) < 3:
            return None

        values = list(self._buffers[signal_name])
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return float(coeffs[0])  # Slope

    def compute_behavioral_trends(
        self,
        engagement_level: int,
        boredom_level: int,
        confusion_level: int,
        frustration_level: int,
        attention_score: float,
    ) -> Dict[str, float]:
        """
        Update all signal buffers and compute behavioral trend metrics.

        Args:
            engagement_level: Predicted engagement level (0–3).
            boredom_level: Predicted boredom level (0–3).
            confusion_level: Predicted confusion level (0–3).
            frustration_level: Predicted frustration level (0–3).
            attention_score: Attention score from head pose (0–1).

        Returns:
            Dictionary of computed trend metrics.
        """
        # Update buffers
        self.update("engagement", engagement_level)
        self.update("boredom", boredom_level)
        self.update("confusion", confusion_level)
        self.update("frustration", frustration_level)
        self.update("attention", attention_score)

        # Compute trend metrics
        trends = {}

        # Low engagement ratio (engagement < 2 means low/very low)
        low_eng_ratio = self.get_threshold_ratio("engagement", 2.0, below=True)
        trends["low_engagement_ratio"] = low_eng_ratio if low_eng_ratio is not None else 0.0

        # Confusion persistence (confusion >= 2 means moderate/high)
        conf_persist = self.get_threshold_ratio("confusion", 2.0, below=False)
        trends["confusion_persistence"] = conf_persist if conf_persist is not None else 0.0

        # Frustration persistence (frustration >= 2)
        frus_persist = self.get_threshold_ratio("frustration", 2.0, below=False)
        trends["frustration_persistence"] = frus_persist if frus_persist is not None else 0.0

        # Off-task attention ratio (attention < 0.5)
        off_task_ratio = self.get_threshold_ratio("attention", 0.5, below=True)
        trends["off_task_ratio"] = off_task_ratio if off_task_ratio is not None else 0.0

        # Smoothed values
        trends["engagement_smoothed"] = self.get_smoothed_value("engagement") or 0.0
        trends["attention_smoothed"] = self.get_smoothed_value("attention") or 0.0

        # Trends (slopes)
        eng_trend = self.get_trend("engagement")
        trends["engagement_trend_slope"] = eng_trend if eng_trend is not None else 0.0

        att_trend = self.get_trend("attention")
        trends["attention_trend_slope"] = att_trend if att_trend is not None else 0.0

        return trends

    def reset(self):
        """Clear all buffers."""
        self._buffers.clear()

    @property
    def buffer_fill_ratio(self) -> float:
        """Fraction of the window that has been filled (for warmup tracking)."""
        if not self._buffers:
            return 0.0
        max_len = max(len(buf) for buf in self._buffers.values())
        return max_len / self.window_size


class TemporalBehaviorModel(nn.Module):
    """
    LSTM/GRU-based temporal model for sequence-level behavioral modeling.

    Takes sequences of frame embeddings and produces temporally-aware
    behavioral predictions.

    Args:
        input_dim: Dimension of input frame embeddings.
        hidden_dim: Hidden state dimension for LSTM/GRU.
        num_layers: Number of recurrent layers.
        num_classes: Number of output classes per task.
        dropout: Dropout probability.
        rnn_type: Type of recurrent unit ('lstm' or 'gru').
    """

    TASKS = ["engagement", "boredom", "confusion", "frustration"]

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.2,
        rnn_type: str = "lstm",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()

        # Recurrent layer
        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Attention mechanism for temporal aggregation
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Task heads on temporal features
        self.task_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes),
            )
            for task in self.TASKS
        })

        logger.info(
            f"TemporalBehaviorModel initialized: rnn={rnn_type}, "
            f"hidden={hidden_dim}, layers={num_layers}"
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input embeddings of shape (B, T, D) where T is sequence length.
            lengths: Optional actual lengths of each sequence in the batch.

        Returns:
            Dictionary mapping task names to logits of shape (B, C).
        """
        # Run through recurrent layer
        rnn_out, _ = self.rnn(x)  # (B, T, hidden_dim)

        # Temporal attention
        attn_weights = self.temporal_attention(rnn_out)  # (B, T, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(rnn_out * attn_weights, dim=1)  # (B, hidden_dim)

        # Task predictions
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(context)

        return outputs

    def predict_from_sequence(
        self, embeddings: torch.Tensor
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get predictions and probabilities from embedding sequence.

        Args:
            embeddings: Tensor of shape (B, T, D) or (T, D).

        Returns:
            Dictionary mapping task names to (predicted_class, probabilities).
        """
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)

        outputs = self.forward(embeddings)
        predictions = {}

        for task_name in self.TASKS:
            logits = outputs[task_name]
            probs = torch.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1)
            predictions[task_name] = (pred_class, probs)

        return predictions
