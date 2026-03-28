"""
Face Detector Module.

Detects and crops faces from frames using MediaPipe Face Detection.
Provides bounding boxes with configurable padding for face region extraction.
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("classroom_monitor.preprocessing")


class FaceDetector:
    """
    Detects faces in images using MediaPipe Face Detection.

    Args:
        min_confidence: Minimum detection confidence threshold (0.0–1.0).
        crop_size: (width, height) for resizing cropped face regions.
        padding: Fractional padding around detected face bounding box.
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        crop_size: Tuple[int, int] = (224, 224),
        padding: float = 0.2,
    ):
        self.min_confidence = min_confidence
        self.crop_size = crop_size
        self.padding = padding
        self._detector = None

    def _init_detector(self):
        """Lazily initialize OpenCV face detector."""
        if self._detector is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._detector = cv2.CascadeClassifier(cascade_path)

    def detect_faces(
        self, image: np.ndarray
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in an image using OpenCV.

        Args:
            image: BGR image as numpy array.

        Returns:
            List of (x1, y1, x2, y2, confidence) tuples for each detected face.
        """
        self._init_detector()

        h, w = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # We simulate min_confidence with minNeighbors
        min_neighbors = max(3, int(self.min_confidence * 10))
        faces = self._detector.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=min_neighbors, 
            minSize=(30, 30)
        )

        detections = []
        for (x, y, fw, fh) in faces:
            # OpenCV doesn't give a confidence score easily in Python 
            # so we just assign a fixed high confidence if it passes minNeighbors
            confidence = 0.95 

            # Absolute coordinates
            x1 = int(x)
            y1 = int(y)
            bw = int(fw)
            bh = int(fh)

            # Add padding
            pad_w = int(bw * self.padding)
            pad_h = int(bh * self.padding)

            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x1 + bw + 2 * pad_w)
            y2 = min(h, y1 + bh + 2 * pad_h)

            detections.append((x1, y1, x2, y2, confidence))

        return detections

    def crop_faces(
        self,
        image: np.ndarray,
        detections: Optional[List[Tuple[int, int, int, int, float]]] = None,
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Crop and resize detected face regions from an image.

        Args:
            image: BGR image as numpy array.
            detections: Optional pre-computed detections. If None, runs detection.

        Returns:
            List of (cropped_face, bounding_box) tuples.
        """
        if detections is None:
            detections = self.detect_faces(image)

        crops = []
        for x1, y1, x2, y2, conf in detections:
            face_crop = image[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            face_resized = cv2.resize(face_crop, self.crop_size)
            crops.append((face_resized, (x1, y1, x2, y2)))

        return crops

    def crop_and_save(
        self,
        image_path: str,
        output_dir: str,
        prefix: str = "face",
    ) -> List[str]:
        """
        Detect faces in an image file and save cropped faces.

        Args:
            image_path: Path to input image.
            output_dir: Directory to save cropped face images.
            prefix: Filename prefix for saved crops.

        Returns:
            List of saved face crop file paths.
        """
        import os

        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Cannot read image: {image_path}")
            return []

        crops = self.crop_faces(image)

        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []

        for idx, (face_img, bbox) in enumerate(crops):
            filename = f"{prefix}_{idx:03d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, face_img)
            saved_paths.append(filepath)

        return saved_paths

    def process_frame_batch(
        self, frames: List[np.ndarray]
    ) -> List[List[Tuple[np.ndarray, Tuple[int, int, int, int]]]]:
        """
        Process a batch of frames and return detected face crops.

        Args:
            frames: List of BGR images.

        Returns:
            List of crop lists, one per input frame.
        """
        return [self.crop_faces(frame) for frame in frames]

    def close(self):
        """Release resources."""
        if self._detector is not None:
            self._detector = None

    def __del__(self):
        self.close()
