"""
CLI Real-Time Inference Script.

Runs the full inference pipeline on camera feed or video file.

Usage:
    python scripts/run_inference.py --config config/config.yaml --checkpoint checkpoints/best_model.pth
    python scripts/run_inference.py --config config/config.yaml --checkpoint checkpoints/best_model.pth --video path/to/video.mp4
    python scripts/run_inference.py --config config/config.yaml --checkpoint checkpoints/best_model.pth --camera 0
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.pipeline import InferencePipeline
from src.utils.helpers import load_config, get_device
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run real-time inference")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--camera", type=int, default=None,
        help="Camera device index for live inference",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to video file for offline inference",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save annotated output video",
    )
    parser.add_argument(
        "--no_display", action="store_true",
        help="Disable display window",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    logger = setup_logger(
        name="classroom_monitor",
        log_dir=config["paths"]["logs"],
    )

    device = get_device(config.get("inference", {}).get("device", "auto"))
    logger.info(f"Using device: {device}")

    # Create inference pipeline
    pipeline = InferencePipeline(
        config=config,
        device=device,
        checkpoint_path=args.checkpoint,
    )

    try:
        if args.video:
            # Run on video file
            logger.info(f"Running inference on video: {args.video}")
            results = pipeline.run_on_video(
                video_path=args.video,
                output_path=args.output,
                display=not args.no_display,
            )
            logger.info(f"Processed {len(results)} frames")
        else:
            # Run on camera
            camera_id = args.camera if args.camera is not None else \
                config.get("inference", {}).get("camera_id", 0)
            logger.info(f"Running inference on camera {camera_id}")
            pipeline.run_on_camera(
                camera_id=camera_id,
                display=not args.no_display,
            )
    finally:
        pipeline.close()

    logger.info("Inference complete!")


if __name__ == "__main__":
    main()
