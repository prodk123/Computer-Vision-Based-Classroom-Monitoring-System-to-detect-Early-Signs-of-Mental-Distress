"""
Real-Time Inference Pipeline.

Orchestrates the full inference flow:
    Camera Stream → Frame Capture → Face Detection →
    CNN Behavioral Prediction → Head Pose Estimation →
    Temporal Buffer → Risk Score Computation → Output

Optimized for near real-time CPU inference.
"""

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms

from src.models.affective_model import AffectiveModel
from src.models.attention_estimator import AttentionEstimator
from src.models.temporal_model import SlidingWindowSmoother
from src.models.risk_fusion import RiskFusionEngine
from src.preprocessing.face_detector import FaceDetector

logger = logging.getLogger("classroom_monitor.inference")


class StudentTracker:
    """
    Tracks a single student's behavioral state over time.

    Each detected face gets its own tracker with independent
    temporal smoothing and risk scoring.
    """

    def __init__(
        self,
        student_id: int,
        window_size: int = 30,
        risk_config: Optional[Dict] = None,
        fps: float = 5.0,
    ):
        self.student_id = student_id
        self.last_seen = time.time()
        self.bbox = (0, 0, 0, 0)

        # Temporal smoother
        self.smoother = SlidingWindowSmoother(window_size=window_size)

        # Risk fusion
        risk_cfg = risk_config or {}
        self.risk_engine = RiskFusionEngine(
            weights=risk_cfg.get("weights"),
            alert_threshold=risk_cfg.get("alert_threshold", 0.65),
            persistence_duration=risk_cfg.get("persistence_duration", 30),
            false_positive_cooldown=risk_cfg.get("false_positive_cooldown", 60),
            smoothing_alpha=risk_cfg.get("smoothing_alpha", 0.3),
            fps=fps,
        )

        # Latest state
        self.latest_state: Dict[str, Any] = {}

    def update(
        self,
        bbox: Tuple[int, int, int, int],
        predictions: Dict[str, int],
        attention_result: Dict[str, Optional[float]],
    ) -> Dict[str, Any]:
        """
        Update student state with new frame data.

        Args:
            bbox: Face bounding box (x1, y1, x2, y2).
            predictions: Predicted class indices for each task.
            attention_result: Head pose and attention metrics.

        Returns:
            Complete student state dictionary.
        """
        self.last_seen = time.time()
        self.bbox = bbox

        # Get attention score (default to 0.5 if face not detected for pose)
        attention_score = attention_result.get("attention_score") or 0.5

        # Update temporal smoother
        trends = self.smoother.compute_behavioral_trends(
            engagement_level=predictions.get("engagement", 1),
            boredom_level=predictions.get("boredom", 0),
            confusion_level=predictions.get("confusion", 0),
            frustration_level=predictions.get("frustration", 0),
            attention_score=attention_score,
        )

        # Update risk engine
        risk_result = self.risk_engine.update(trends)

        self.latest_state = {
            "student_id": self.student_id,
            "bbox": bbox,
            "predictions": predictions,
            "attention": attention_result,
            "trends": trends,
            "risk": risk_result,
            "timestamp": self.last_seen,
        }

        return self.latest_state

    @property
    def is_stale(self) -> bool:
        """Check if this tracker hasn't been updated recently."""
        return time.time() - self.last_seen > 5.0  # 5 second timeout


class InferencePipeline:
    """
    End-to-end real-time inference pipeline.

    Processes video frames through face detection, behavioral classification,
    head pose estimation, temporal modeling, and risk scoring.

    Args:
        model: Trained AffectiveModel.
        config: Configuration dictionary.
        device: Torch device.
        checkpoint_path: Optional path to model checkpoint.
    """

    # ImageNet normalization
    _NORMALIZE = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    def __init__(
        self,
        model: Optional[AffectiveModel] = None,
        config: Optional[Dict] = None,
        device: Optional[torch.device] = None,
        checkpoint_path: Optional[str] = None,
    ):
        self.config = config or {}
        self.device = device or torch.device("cpu")

        # Model setup
        if model is not None:
            self.model = model.to(self.device)
        else:
            model_cfg = self.config.get("model", {})
            self.model = AffectiveModel(
                backbone_name=model_cfg.get("backbone", "resnet18"),
                pretrained=False,
                embedding_dim=model_cfg.get("embedding_dim", 256),
                dropout=model_cfg.get("dropout", 0.3),
            ).to(self.device)

        # Load checkpoint if provided
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        self.model.eval()

        # Face detector
        preproc_cfg = self.config.get("preprocessing", {})
        self.face_detector = FaceDetector(
            min_confidence=preproc_cfg.get("face_min_confidence", 0.5),
            crop_size=tuple(preproc_cfg.get("face_crop_size", [224, 224])),
            padding=preproc_cfg.get("face_padding", 0.2),
        )

        # Attention estimator
        attn_cfg = self.config.get("attention", {})
        self.attention_estimator = AttentionEstimator(
            yaw_threshold=attn_cfg.get("yaw_threshold", 30.0),
            pitch_threshold=attn_cfg.get("pitch_threshold", 25.0),
            roll_threshold=attn_cfg.get("roll_threshold", 20.0),
            decay_rate=attn_cfg.get("decay_rate", 0.1),
        )

        # Student trackers
        self.student_trackers: Dict[int, StudentTracker] = {}
        self._next_student_id = 0

        # Pipeline config
        infer_cfg = self.config.get("inference", {})
        self.max_faces = infer_cfg.get("max_faces", 30)
        self.target_fps = self.config.get("preprocessing", {}).get("target_fps", 5)

        # Preprocessing transform (for inference)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self._NORMALIZE,
        ])

        logger.info("InferencePipeline initialized")

    def _load_checkpoint(self, path: str):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        logger.info(f"Model checkpoint loaded: {path}")

    def _assign_student_id(
        self, bbox: Tuple[int, int, int, int]
    ) -> int:
        """
        Locks the student ID to 1 to prevent random numbers (4, 5, 8, etc.) 
        when the face temporarily disappears or moves fast.
        """
        # Always return Student 1 for a single-person dashboard
        return 1

    @staticmethod
    def _compute_iou(
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int],
    ) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    @torch.no_grad()
    def process_frame(
        self, frame: np.ndarray
    ) -> Dict[str, Any]:
        """
        Process a single video frame through the full pipeline.

        Args:
            frame: BGR image from camera/video.

        Returns:
            Dictionary with:
                - students: List of per-student state dicts
                - frame_info: Metadata about the frame processing
                - annotated_frame: Frame with drawn overlays
        """
        start_time = time.time()

        # 1. Face Detection
        detections = self.face_detector.detect_faces(frame)
        face_crops = self.face_detector.crop_faces(frame, detections)

        # Limit faces
        if len(face_crops) > self.max_faces:
            face_crops = face_crops[:self.max_faces]

        student_states = []

        # 🔥 ACTIVE RISK TRIGGER: If OpenCV loses the student's face because they completely turned away 
        # or left the camera frame, we deliberately plummet their Engagement and Attention to spike Risk!
        if not face_crops and 1 in self.student_trackers:
            state = self.student_trackers[1].update(
                self.student_trackers[1].bbox, 
                {"engagement": 0, "boredom": 3, "confusion": 0, "frustration": 0},
                {"attention_score": 0.0, "is_off_task": True}
            )
            student_states.append(state)

        for face_img, bbox in face_crops:
            # 2. Assign student ID (simple IoU-based tracking)
            student_id = self._assign_student_id(bbox)

            # 3. CNN Behavioral Prediction
            input_tensor = self.transform(face_img).unsqueeze(0).to(self.device)
            predictions_raw = self.model.get_predictions(input_tensor)

            predictions = {
                task: pred_class.item()
                for task, (pred_class, probs) in predictions_raw.items()
            }

            # 4. Head Pose Estimation (on face region from full frame)
            x1, y1, x2, y2 = bbox
            face_region = frame[y1:y2, x1:x2]
            attention_result = self.attention_estimator.process_frame(face_region)

            # 🔥 SMART OVERRIDE: DAiSEE models natively bias towards "High Engagement". 
            # If the student's head is completely physically turned away, we force engagement to Very Low.
            if attention_result.get("is_off_task", False):
                predictions["engagement"] = 0

            # 5. Update student tracker
            if student_id not in self.student_trackers:
                risk_cfg = self.config.get("risk_fusion", {})
                temporal_cfg = self.config.get("temporal", {})
                self.student_trackers[student_id] = StudentTracker(
                    student_id=student_id,
                    window_size=temporal_cfg.get("window_size", 30),
                    risk_config=risk_cfg,
                    fps=self.target_fps,
                )

            state = self.student_trackers[student_id].update(
                bbox, predictions, attention_result
            )
            student_states.append(state)

        # Clean up stale trackers
        stale_ids = [
            sid for sid, t in self.student_trackers.items()
            if t.is_stale
        ]
        for sid in stale_ids:
            del self.student_trackers[sid]

        # 6. Annotate frame
        annotated = self._annotate_frame(frame.copy(), student_states)

        elapsed = time.time() - start_time

        return {
            "students": student_states,
            "frame_info": {
                "num_faces": len(face_crops),
                "processing_time_ms": elapsed * 1000,
                "active_trackers": len(self.student_trackers),
            },
            "annotated_frame": annotated,
        }

    def _annotate_frame(
        self,
        frame: np.ndarray,
        student_states: List[Dict],
    ) -> np.ndarray:
        """Draw bounding boxes and labels on the frame."""
        for state in student_states:
            bbox = state["bbox"]
            x1, y1, x2, y2 = bbox
            risk = state["risk"]
            predictions = state["predictions"]

            # Color based on risk level
            risk_score = risk.get("risk_score", 0)
            if risk_score < 0.25:
                color = (0, 200, 0)       # Green — low risk
            elif risk_score < 0.50:
                color = (0, 200, 200)     # Yellow — moderate
            elif risk_score < 0.75:
                color = (0, 130, 255)     # Orange — elevated
            else:
                color = (0, 0, 255)       # Red — high

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Labels
            eng_level = predictions.get("engagement", 0)
            eng_labels = ["Very Low", "Low", "High", "Very High"]
            eng_text = eng_labels[eng_level] if eng_level < len(eng_labels) else "?"

            attn_score = state["attention"].get("attention_score")
            attn_text = f"{attn_score:.0%}" if attn_score is not None else "N/A"

            # Draw text background
            texts = [
                f"ID:{state['student_id']}",
                f"Eng: {eng_text}",
                f"Attn: {attn_text}",
                f"Risk: {risk.get('risk_level', 'N/A')}",
            ]

            y_offset = y1 - 10
            for text in reversed(texts):
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(
                    frame,
                    (x1, y_offset - text_size[1] - 5),
                    (x1 + text_size[0] + 5, y_offset + 5),
                    color,
                    -1,
                )
                cv2.putText(
                    frame, text,
                    (x1 + 2, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )
                y_offset -= text_size[1] + 10

            # Alert indicator
            if risk.get("alert_active"):
                cv2.putText(
                    frame, "⚠ ALERT",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2, cv2.LINE_AA,
                )

        return frame

    def run_on_camera(
        self,
        camera_id: int = 0,
        display: bool = True,
        max_frames: Optional[int] = None,
    ):
        """
        Run real-time inference on camera feed.

        Args:
            camera_id: Camera device index.
            display: Whether to show the annotated frame in a window.
            max_frames: Optional maximum number of frames to process.
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"Cannot open camera {camera_id}")
            return

        logger.info(f"Running inference on camera {camera_id}")
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                result = self.process_frame(frame)
                frame_count += 1

                if display:
                    annotated = result["annotated_frame"]
                    info = result["frame_info"]

                    # Draw FPS
                    fps_text = f"FPS: {1000/max(info['processing_time_ms'], 1):.1f}"
                    cv2.putText(
                        annotated, fps_text,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2,
                    )

                    cv2.imshow("Classroom Monitor", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if max_frames and frame_count >= max_frames:
                    break
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            logger.info(f"Processed {frame_count} frames")

    def run_on_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = False,
    ) -> List[Dict]:
        """
        Run inference on a video file.

        Args:
            video_path: Path to input video.
            output_path: Optional path to save annotated output video.
            display: Whether to display frames.

        Returns:
            List of per-frame results.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_results = []

        from tqdm import tqdm
        for _ in tqdm(range(total_frames), desc="Processing video"):
            ret, frame = cap.read()
            if not ret:
                break

            result = self.process_frame(frame)
            all_results.append(result)

            if writer:
                writer.write(result["annotated_frame"])

            if display:
                cv2.imshow("Classroom Monitor", result["annotated_frame"])
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        logger.info(f"Processed {len(all_results)} frames from {video_path}")
        return all_results

    def close(self):
        """Release all resources."""
        self.face_detector.close()
        self.attention_estimator.close()
