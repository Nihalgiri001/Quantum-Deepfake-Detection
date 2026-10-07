"""
Utility Functions
==================
Common helpers used across the pipeline:
- Logging setup
- Random seed setting for reproducibility
- Directory creation
- Progress reporting
"""

import os
import sys
import time
import random
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "deepfake_classical",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Create a formatted logger.

    Args:
        name: Logger name.
        level: Logging level (e.g. logging.INFO).
        log_file: Optional path to also write logs to a file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers when called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic mode (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The same path for chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_video_files(directory: Path, extensions: tuple = (".mp4", ".avi", ".mkv")) -> list:
    """List all video files in a directory.

    Args:
        directory: Path to scan.
        extensions: Accepted video file extensions.

    Returns:
        Sorted list of video file Paths.
    """
    if not directory.exists():
        return []
    videos = [
        f for f in sorted(directory.iterdir())
        if f.is_file() and f.suffix.lower() in extensions
    ]
    return videos


class Timer:
    """Simple context-manager timer for profiling pipeline stages."""

    def __init__(self, description: str = "", logger: Optional[logging.Logger] = None):
        self.description = description
        self.logger = logger
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        msg = f"{self.description} completed in {self.elapsed:.2f}s"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)


def format_count(n: int) -> str:
    """Human-readable number formatting.

    Examples:
        format_count(1234)  -> '1,234'
        format_count(0)     -> '0'
    """
    return f"{n:,}"
