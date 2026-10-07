"""
Dataset Module
===============
Handles discovery and cataloguing of FaceForensics++ videos.

Responsibilities:
- Scan the dataset directory for original and manipulated videos
- Build a unified video registry with labels and metadata
- Provide the video list to downstream scripts (splitting, extraction)
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

from config import Config
from utils import get_video_files, setup_logger

logger = setup_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Label encoding (consistent across ALL scripts)
# ──────────────────────────────────────────────────────────────────────────────
LABEL_MAP = {"original": 0, "fake": 1}
LABEL_NAMES = {0: "REAL", 1: "FAKE"}


def discover_videos(cfg: Config) -> pd.DataFrame:
    """Scan the FaceForensics++ dataset and build a video catalogue.

    Each row contains:
        video_id          – unique identifier (filename stem)
        video_path        – absolute path to the .mp4 file
        label             – 0 (REAL) or 1 (FAKE)
        label_name        – "REAL" or "FAKE"
        manipulation_type – "original" / "Deepfakes" / "Face2Face" / ...
        source_video      – for manipulated videos, the first part of the
                            filename (e.g. "003" from "003_000.mp4")

    Args:
        cfg: Pipeline configuration.

    Returns:
        DataFrame with one row per video.
    """
    records: List[Dict] = []

    # ── Original (REAL) videos ────────────────────────────────────────────────
    orig_dir = cfg.original_videos_dir
    if not orig_dir.exists():
        raise FileNotFoundError(
            f"Original videos directory not found: {orig_dir}\n"
            f"Please check that DATASET_ROOT is correct in config.py.\n"
            f"Current dataset_root = {cfg.dataset_root}"
        )

    orig_videos = get_video_files(orig_dir)
    logger.info(f"Found {len(orig_videos)} original videos in {orig_dir}")

    for vpath in orig_videos:
        vid_id = vpath.stem  # e.g. "000"
        records.append({
            "video_id": vid_id,
            "video_path": str(vpath),
            "label": cfg.REAL_LABEL,
            "label_name": "REAL",
            "manipulation_type": "original",
            "source_video": vid_id,
        })

    # ── Manipulated (FAKE) videos ─────────────────────────────────────────────
    for manip_type in cfg.manipulation_types:
        manip_dir = cfg.get_manipulation_dir(manip_type)
        if not manip_dir.exists():
            logger.warning(
                f"Manipulation directory not found: {manip_dir}  — skipping."
            )
            continue

        manip_videos = get_video_files(manip_dir)
        logger.info(f"Found {len(manip_videos)} {manip_type} videos in {manip_dir}")

        for vpath in manip_videos:
            vid_id = vpath.stem  # e.g. "003_000"
            # The source video ID is the first part before '_'
            source = vid_id.split("_")[0] if "_" in vid_id else vid_id
            records.append({
                "video_id": vid_id,
                "video_path": str(vpath),
                "label": cfg.FAKE_LABEL,
                "label_name": "FAKE",
                "manipulation_type": manip_type,
                "source_video": source,
            })

    if not records:
        raise RuntimeError(
            "No videos found in the dataset. "
            "Check dataset_root in config.py and verify directory structure."
        )

    df = pd.DataFrame(records)
    logger.info(
        f"Total videos catalogued: {len(df)} "
        f"(REAL={len(df[df.label == cfg.REAL_LABEL])}, "
        f"FAKE={len(df[df.label == cfg.FAKE_LABEL])})"
    )
    return df


def get_dataset_stats(df: pd.DataFrame) -> Dict:
    """Compute summary statistics for the video catalogue.

    Args:
        df: Video catalogue DataFrame from discover_videos().

    Returns:
        Dictionary of counts and breakdowns.
    """
    stats = {
        "total_videos": len(df),
        "real_videos": int((df.label == 0).sum()),
        "fake_videos": int((df.label == 1).sum()),
        "manipulation_breakdown": df.groupby("manipulation_type").size().to_dict(),
    }
    return stats
