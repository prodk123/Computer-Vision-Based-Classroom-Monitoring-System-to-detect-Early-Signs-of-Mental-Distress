"""
Attention Estimation Module (Branch 2).

Estimates student attention level using head pose (yaw, pitch, roll)
derived from MediaPipe Face Mesh landmarks.

Converts head orientation deviation into a normalized attention score.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("classroom_monitor.models")


class AttentionEstimator:
    """
    Estimates attention from head pose using MediaPipe Face Mesh.

    Computes yaw, pitch, roll angles from facial landmarks and converts
    deviations from a reference pose into a normalized attention score.

    Args:
        yaw_threshold: Max yaw deviation (degrees) before attention drops.
        pitch_threshold: Max pitch deviation (degrees).
        roll_threshold: Max roll deviation (degrees).
        decay_rate: Exponential decay rate for attention score computation.
    """

    # MediaPipe Face Mesh indices for head pose estimation
    # Using key landmarks: nose tip, chin, left/right eye corners, forehead
    _POSE_LANDMARKS = {
        "nose_tip": 1,
        "chin": 152,
        "left_eye_outer": 263,
        "right_eye_outer": 33,
        "left_mouth_corner": 287,
        "right_mouth_corner": 57,
    }

    def __init__(
        self,
        yaw_threshold: float = 30.0,
        pitch_threshold: float = 25.0,
        roll_threshold: float = 20.0,
        decay_rate: float = 0.1,
    ):
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.roll_threshold = roll_threshold
        self.decay_rate = decay_rate
        self._face_mesh = None

    def _init_face_mesh(self):
        """Lazily initialize MediaPipe Face Mesh."""
        if self._face_mesh is None:
            try:
                import mediapipe as mp
                import mediapipe.python.solutions.face_mesh as mp_face_mesh
                self._mp_face_mesh = mp_face_mesh
                self._face_mesh = self._mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except (ImportError, AttributeError) as e:
                logger.warning(f"MediaPipe Face Mesh not available on this Python version: {e}. Attention estimation disabled.")
                self._face_mesh = "DISABLED"

    def estimate_head_pose(
        self, image: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """
        Estimate head pose angles from an image.

        Uses the solvePnP approach with 3D face model points and
        2D detected landmarks.

        Args:
            image: BGR image (full frame or cropped face region).

        Returns:
            Dictionary with 'yaw', 'pitch', 'roll' in degrees,
            or None if face landmarks are not detected.
        """
        self._init_face_mesh()

        if self._face_mesh == "DISABLED":
            return None

        h, w = image.shape[:2]

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]

        # 3D model points (generic face model in world coordinates)
        model_points = np.array([
            (0.0, 0.0, 0.0),           # Nose tip
            (0.0, -330.0, -65.0),       # Chin
            (-225.0, 170.0, -135.0),    # Left eye outer corner
            (225.0, 170.0, -135.0),     # Right eye outer corner
            (-150.0, -150.0, -125.0),   # Left mouth corner
            (150.0, -150.0, -125.0),    # Right mouth corner
        ], dtype=np.float64)

        # 2D image points from detected landmarks
        image_points = np.array([
            self._get_landmark_point(landmarks, self._POSE_LANDMARKS["nose_tip"], w, h),
            self._get_landmark_point(landmarks, self._POSE_LANDMARKS["chin"], w, h),
            self._get_landmark_point(landmarks, self._POSE_LANDMARKS["left_eye_outer"], w, h),
            self._get_landmark_point(landmarks, self._POSE_LANDMARKS["right_eye_outer"], w, h),
            self._get_landmark_point(landmarks, self._POSE_LANDMARKS["left_mouth_corner"], w, h),
            self._get_landmark_point(landmarks, self._POSE_LANDMARKS["right_mouth_corner"], w, h),
        ], dtype=np.float64)

        # Camera intrinsics (approximate)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        # Solve PnP to get rotation and translation vectors
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        # Convert rotation vector to rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)

        # Decompose to Euler angles
        proj_matrix = np.hstack((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
            np.vstack((proj_matrix, [0, 0, 0, 1]))[:3]
        )

        pitch = euler_angles[0, 0]
        yaw = euler_angles[1, 0]
        roll = euler_angles[2, 0]

        # Clamp angles to reasonable range
        pitch = max(-90, min(90, pitch))
        yaw = max(-90, min(90, yaw))
        roll = max(-90, min(90, roll))

        return {
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll),
        }

    @staticmethod
    def _get_landmark_point(
        landmarks, idx: int, img_w: int, img_h: int
    ) -> Tuple[float, float]:
        """Extract 2D pixel coordinates from a face mesh landmark."""
        lm = landmarks.landmark[idx]
        return (lm.x * img_w, lm.y * img_h)

    def compute_attention_score(
        self, head_pose: Dict[str, float]
    ) -> float:
        """
        Convert head pose angles into a normalized attention score.

        Attention score = 1.0 means fully on-task (looking at task direction).
        Attention score = 0.0 means completely off-task.

        Uses exponential decay based on angular deviation from thresholds.

        Args:
            head_pose: Dictionary with 'yaw', 'pitch', 'roll' in degrees.

        Returns:
            Attention score in [0.0, 1.0].
        """
        yaw_dev = max(0, abs(head_pose["yaw"]) - self.yaw_threshold * 0.3)
        pitch_dev = max(0, abs(head_pose["pitch"]) - self.pitch_threshold * 0.3)
        roll_dev = max(0, abs(head_pose["roll"]) - self.roll_threshold * 0.3)

        # Weighted angular deviation
        total_deviation = (
            (yaw_dev / self.yaw_threshold) * 0.5
            + (pitch_dev / self.pitch_threshold) * 0.3
            + (roll_dev / self.roll_threshold) * 0.2
        )

        # Exponential decay
        attention_score = math.exp(-self.decay_rate * total_deviation * 10)

        return max(0.0, min(1.0, attention_score))

    def is_off_task(self, head_pose: Dict[str, float]) -> bool:
        """
        Determine if head orientation indicates off-task behavior.

        Args:
            head_pose: Dictionary with 'yaw', 'pitch', 'roll' in degrees.

        Returns:
            True if the student appears to be looking away from task.
        """
        return (
            abs(head_pose["yaw"]) > self.yaw_threshold
            or abs(head_pose["pitch"]) > self.pitch_threshold
            or abs(head_pose["roll"]) > self.roll_threshold
        )

    def process_frame(
        self, image: np.ndarray
    ) -> Dict[str, Optional[float]]:
        """
        Full pipeline: image → head pose → attention score.

        Args:
            image: BGR image.

        Returns:
            Dictionary with 'yaw', 'pitch', 'roll', 'attention_score',
            and 'is_off_task'. Values are None if face not detected.
        """
        head_pose = self.estimate_head_pose(image)

        if head_pose is None:
            return {
                "yaw": None,
                "pitch": None,
                "roll": None,
                "attention_score": 0.5,
                "is_off_task": False,
            }

        attention_score = self.compute_attention_score(head_pose)
        off_task = self.is_off_task(head_pose)

        return {
            "yaw": head_pose["yaw"],
            "pitch": head_pose["pitch"],
            "roll": head_pose["roll"],
            "attention_score": attention_score,
            "is_off_task": off_task,
        }

    def close(self):
        """Release MediaPipe resources."""
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None

    def __del__(self):
        self.close()
