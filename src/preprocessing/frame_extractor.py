"""
Frame Extractor Module.

Extracts frames from video files at a configurable FPS rate.
Designed to work with the DAiSEE dataset video format.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger("classroom_monitor.preprocessing")


class FrameExtractor:
    """
    Extracts frames from video clips at configurable frame rates.

    Supports the DAiSEE dataset structure:
        DAiSEE/DataSet/{Train,Validation,Test}/{UserID}/{ClipID}.avi

    Args:
        target_fps: Number of frames to extract per second of video.
        output_size: Optional (width, height) to resize frames.
    """

    def __init__(
        self,
        target_fps: int = 5,
        output_size: Optional[Tuple[int, int]] = None,
    ):
        self.target_fps = target_fps
        self.output_size = output_size

    def extract_from_video(
        self,
        video_path: str,
        output_dir: str,
        prefix: Optional[str] = None,
    ) -> List[str]:
        """
        Extract frames from a single video file.

        Args:
            video_path: Path to the video file.
            output_dir: Directory to save extracted frames.
            prefix: Optional prefix for frame filenames.

        Returns:
            List of saved frame file paths.
        """
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Cannot open video: {video_path}")
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if video_fps <= 0:
            logger.warning(f"Invalid FPS for video: {video_path}")
            cap.release()
            return []

        # Calculate frame sampling interval
        frame_interval = max(1, int(video_fps / self.target_fps))

        os.makedirs(output_dir, exist_ok=True)

        if prefix is None:
            prefix = Path(video_path).stem

        saved_paths = []
        frame_idx = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                if self.output_size is not None:
                    frame = cv2.resize(frame, self.output_size)

                filename = f"{prefix}_frame_{saved_count:05d}.jpg"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, frame)
                saved_paths.append(filepath)
                saved_count += 1

            frame_idx += 1

        cap.release()

        logger.debug(
            f"Extracted {saved_count} frames from {video_path} "
            f"(total: {total_frames}, interval: {frame_interval})"
        )

        return saved_paths

    def extract_from_directory(
        self,
        input_dir: str,
        output_dir: str,
        video_extensions: Tuple[str, ...] = (".avi", ".mp4", ".mkv"),
    ) -> dict:
        """
        Extract frames from all videos in a directory (recursively).

        Args:
            input_dir: Root directory containing videos.
            output_dir: Root output directory for frames.
            video_extensions: Tuple of video file extensions to process.

        Returns:
            Dictionary mapping video paths to lists of frame paths.
        """
        video_files = []
        for root, _, files in os.walk(input_dir):
            for fname in files:
                if fname.lower().endswith(video_extensions):
                    video_files.append(os.path.join(root, fname))

        logger.info(f"Found {len(video_files)} video files in {input_dir}")

        results = {}
        for video_path in tqdm(video_files, desc="Extracting frames"):
            # Preserve directory structure
            rel_path = os.path.relpath(video_path, input_dir)
            video_name = Path(rel_path).stem
            video_output_dir = os.path.join(
                output_dir, os.path.dirname(rel_path), video_name
            )

            frame_paths = self.extract_from_video(
                video_path, video_output_dir, prefix=video_name
            )
            results[video_path] = frame_paths

        total_frames = sum(len(v) for v in results.values())
        logger.info(
            f"Total frames extracted: {total_frames} from {len(results)} videos"
        )

        return results

    def extract_frames_in_memory(
        self, video_path: str
    ) -> List[np.ndarray]:
        """
        Extract frames and return as numpy arrays (for real-time use).

        Args:
            video_path: Path to the video file.

        Returns:
            List of frames as numpy arrays (BGR format).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Cannot open video: {video_path}")
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / self.target_fps))

        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                if self.output_size is not None:
                    frame = cv2.resize(frame, self.output_size)
                frames.append(frame)

            frame_idx += 1

        cap.release()
        return frames
