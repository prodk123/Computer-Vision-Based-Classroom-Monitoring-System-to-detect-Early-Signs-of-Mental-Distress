"""
Risk Fusion Engine.

Combines behavioral trend metrics from multiple branches into a
single interpretable risk indicator score with alert triggering logic.

The risk score is:
    RiskScore(t) = w1 * LowEngagementTrend
                 + w2 * ConfusionTrend
                 + w3 * FrustrationTrend
                 + w4 * AttentionDeviation

Includes normalization, persistence filtering, and false positive suppression.
"""

import logging
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("classroom_monitor.models")


class RiskFusionEngine:
    """
    Fuses multi-branch behavioral signals into a risk indicator score.

    Args:
        weights: Dictionary of signal weights.
        alert_threshold: Risk score threshold for triggering alerts.
        persistence_duration: Seconds of sustained risk before alerting.
        false_positive_cooldown: Seconds of cooldown after an alert.
        smoothing_alpha: EMA smoothing factor for risk score.
        fps: Expected frames per second (for time computation).
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        alert_threshold: float = 0.65,
        persistence_duration: float = 30.0,
        false_positive_cooldown: float = 60.0,
        smoothing_alpha: float = 0.3,
        fps: float = 5.0,
    ):
        if weights is None:
            weights = {
                "low_engagement": 0.35,
                "confusion": 0.25,
                "frustration": 0.25,
                "attention_deviation": 0.15,
            }

        self.weights = weights
        self.alert_threshold = alert_threshold
        self.persistence_duration = persistence_duration
        self.false_positive_cooldown = false_positive_cooldown
        self.smoothing_alpha = smoothing_alpha
        self.fps = fps

        # Internal state
        self._smoothed_risk = 0.0
        self._risk_history: deque = deque(maxlen=1000)
        self._alert_start_time: Optional[float] = None
        self._last_alert_time: Optional[float] = None
        self._alert_active = False
        self._consecutive_high_risk_frames = 0

        # For interpretability
        self._component_scores: Dict[str, float] = {}

    def compute_risk_score(
        self, trends: Dict[str, float]
    ) -> float:
        """
        Compute the raw composite risk score from behavioral trends.

        Args:
            trends: Dictionary containing:
                - low_engagement_ratio: fraction of low-engagement frames
                - confusion_persistence: fraction of high-confusion frames
                - frustration_persistence: fraction of high-frustration frames
                - off_task_ratio: fraction of off-task frames

        Returns:
            Raw risk score (not yet normalized).
        """
        components = {
            "low_engagement": trends.get("low_engagement_ratio", 0.0),
            "confusion": trends.get("confusion_persistence", 0.0),
            "frustration": trends.get("frustration_persistence", 0.0),
            "attention_deviation": trends.get("off_task_ratio", 0.0),
        }

        # Weighted sum
        raw_score = sum(
            self.weights.get(k, 0.0) * v for k, v in components.items()
        )

        # Normalize by total weight
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            normalized_score = raw_score / total_weight
        else:
            normalized_score = 0.0

        # Clip to [0, 1]
        normalized_score = max(0.0, min(1.0, normalized_score))

        # Store component scores for interpretability
        self._component_scores = {
            k: v * self.weights.get(k, 0.0) / total_weight
            for k, v in components.items()
        }

        return normalized_score

    def update(
        self,
        trends: Dict[str, float],
        timestamp: Optional[float] = None,
    ) -> Dict[str, object]:
        """
        Full risk assessment update step.

        Args:
            trends: Behavioral trend dictionary from SlidingWindowSmoother.
            timestamp: Optional timestamp (seconds). If None, uses current time.

        Returns:
            Dictionary with:
                - risk_score: Current smoothed risk score [0, 1]
                - raw_risk_score: Unsmoothed risk score
                - alert_active: Whether alert is currently triggered
                - alert_message: Human-readable alert message (if any)
                - component_scores: Individual component contributions
                - risk_level: Categorical risk level string
        """
        if timestamp is None:
            timestamp = time.time()

        # Compute raw risk score
        raw_score = self.compute_risk_score(trends)

        # Apply EMA smoothing
        self._smoothed_risk = (
            self.smoothing_alpha * raw_score
            + (1 - self.smoothing_alpha) * self._smoothed_risk
        )

        # Track history
        self._risk_history.append({
            "timestamp": timestamp,
            "raw_score": raw_score,
            "smoothed_score": self._smoothed_risk,
        })

        # --- Alert Logic ---
        alert_message = None
        alert_triggered = False

        if self._smoothed_risk >= self.alert_threshold:
            self._consecutive_high_risk_frames += 1

            # Check persistence duration
            required_frames = int(self.persistence_duration * self.fps)
            if self._consecutive_high_risk_frames >= required_frames:
                # Check cooldown
                if self._last_alert_time is None or (
                    timestamp - self._last_alert_time > self.false_positive_cooldown
                ):
                    alert_triggered = True
                    self._alert_active = True
                    self._last_alert_time = timestamp
                    alert_message = self._generate_alert_message()
                    logger.info(
                        f"Alert triggered: risk={self._smoothed_risk:.3f}, "
                        f"message={alert_message}"
                    )
        else:
            self._consecutive_high_risk_frames = max(
                0, self._consecutive_high_risk_frames - 2
            )  # Slow decay of counter for hysteresis
            if self._smoothed_risk < self.alert_threshold * 0.8:
                self._alert_active = False

        # Determine risk level category
        risk_level = self._categorize_risk(self._smoothed_risk)

        return {
            "risk_score": self._smoothed_risk,
            "raw_risk_score": raw_score,
            "alert_active": self._alert_active,
            "alert_triggered": alert_triggered,
            "alert_message": alert_message,
            "component_scores": dict(self._component_scores),
            "risk_level": risk_level,
            "consecutive_high_frames": self._consecutive_high_risk_frames,
        }

    def _generate_alert_message(self) -> str:
        """
        Generate a neutral, non-medical alert message based on
        the dominant contributing signal.
        """
        if not self._component_scores:
            return "Sustained pattern of reduced engagement detected."

        # Find dominant component
        dominant = max(self._component_scores, key=self._component_scores.get)

        messages = {
            "low_engagement": (
                "This student has shown a sustained pattern of low participation. "
                "Consider checking in or offering support."
            ),
            "confusion": (
                "This student may be experiencing difficulty with the current material. "
                "They may benefit from additional guidance."
            ),
            "frustration": (
                "This student appears to be struggling. "
                "A brief check-in may help."
            ),
            "attention_deviation": (
                "This student's attention has been directed away from the task area "
                "for an extended period."
            ),
        }

        return messages.get(
            dominant,
            "Sustained behavioral pattern change detected. Consider a check-in."
        )

    @staticmethod
    def _categorize_risk(score: float) -> str:
        """Categorize risk score into a human-readable level."""
        if score < 0.25:
            return "low"
        elif score < 0.50:
            return "moderate"
        elif score < 0.75:
            return "elevated"
        else:
            return "high"

    def get_risk_history(
        self, last_n: Optional[int] = None
    ) -> List[Dict]:
        """
        Get recent risk score history.

        Args:
            last_n: Number of recent entries to return. None = all.

        Returns:
            List of risk history dictionaries.
        """
        history = list(self._risk_history)
        if last_n is not None:
            history = history[-last_n:]
        return history

    def reset(self):
        """Reset all internal state."""
        self._smoothed_risk = 0.0
        self._risk_history.clear()
        self._alert_start_time = None
        self._last_alert_time = None
        self._alert_active = False
        self._consecutive_high_risk_frames = 0
        self._component_scores.clear()

    def get_status_summary(self) -> Dict[str, object]:
        """Get a summary of current risk engine status."""
        return {
            "current_risk_score": self._smoothed_risk,
            "risk_level": self._categorize_risk(self._smoothed_risk),
            "alert_active": self._alert_active,
            "history_length": len(self._risk_history),
            "component_scores": dict(self._component_scores),
            "weights": dict(self.weights),
        }
