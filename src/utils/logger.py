"""
Logging utility for the Classroom Monitoring System.

Provides centralized logging configuration with file and console handlers.
"""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "classroom_monitor",
    log_dir: str = "logs",
    level: str = "INFO",
    log_to_file: bool = True,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return a logger instance.

    Args:
        name: Logger name.
        log_dir: Directory for log files.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_to_file: Whether to also write logs to a file.
        log_format: Optional custom log format string.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if log_format is None:
        log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Log file: {log_file}")

    return logger
