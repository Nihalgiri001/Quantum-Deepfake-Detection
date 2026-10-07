"""
Video-Level Dataset Splitting
===============================
Splits videos into train / validation / test sets at the VIDEO level
to prevent data leakage.

CRITICAL: Frames from the same video NEVER appear in different splits.

Usage:
    python src/split_dataset.py
    python src/split_dataset.py --max-videos 20     # smoke test

Outputs:
    data/metadata/train_videos.csv
    data/metadata/val_videos.csv
    data/metadata/test_videos.csv
    data/metadata/all_videos.csv
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from dataset import discover_videos, get_dataset_stats
from utils import setup_logger, set_seed, ensure_dir, format_count

logger = setup_logger("split_dataset")


def verify_no_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> bool:
    """Verify that no video appears in more than one split.

    Raises:
        RuntimeError: If any overlap (data leakage) is detected.

    Returns:
        True if verification passes.
    """
    train_ids = set(train_df.video_id)
    val_ids = set(val_df.video_id)
    test_ids = set(test_df.video_id)

    train_val = train_ids & val_ids
    train_test = train_ids & test_ids
    val_test = val_ids & test_ids

    print()
    print("─" * 60)
    print("  Dataset Split Verification")
    print("─" * 60)
    print(f"  Training videos:          {format_count(len(train_ids))}")
    print(f"  Validation videos:        {format_count(len(val_ids))}")
    print(f"  Test videos:              {format_count(len(test_ids))}")
    print()
    print(f"  Train/Validation overlap: {len(train_val)}")
    print(f"  Train/Test overlap:       {len(train_test)}")
    print(f"  Validation/Test overlap:  {len(val_test)}")
    print()

    if train_val or train_test or val_test:
        msg = (
            "❌ DATA LEAKAGE DETECTED!\n"
            f"   Train∩Val = {train_val}\n"
            f"   Train∩Test = {train_test}\n"
            f"   Val∩Test = {val_test}\n"
            "   STOPPING. Fix the split before proceeding."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    print("  ✅ Split verification: PASSED — no data leakage")
    print("─" * 60)
    return True


def split_videos(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> dict:
    """Split a video catalogue into train/val/test at the VIDEO level.

    The split is stratified by manipulation_type so that each split
    has a proportional representation of each category.

    Args:
        df: Video catalogue DataFrame.
        train_ratio: Fraction of videos for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        random_seed: Seed for reproducibility.

    Returns:
        Dictionary with keys 'train', 'val', 'test' mapping to DataFrames.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    set_seed(random_seed)

    train_parts, val_parts, test_parts = [], [], []

    # Stratified split per manipulation type
    for mtype, group in df.groupby("manipulation_type"):
        group = group.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # Remaining go to test (handles rounding)

        train_parts.append(group.iloc[:n_train])
        val_parts.append(group.iloc[n_train:n_train + n_val])
        test_parts.append(group.iloc[n_train + n_val:])

        logger.info(
            f"  {mtype:20s}: train={n_train}, val={n_val}, "
            f"test={n - n_train - n_val}"
        )

    splits = {
        "train": pd.concat(train_parts, ignore_index=True),
        "val": pd.concat(val_parts, ignore_index=True),
        "test": pd.concat(test_parts, ignore_index=True),
    }

    return splits


def main():
    parser = argparse.ArgumentParser(
        description="Split FaceForensics++ videos into train/val/test sets."
    )
    parser.add_argument(
        "--max-videos", type=int, default=None,
        help="Limit total videos per category (for smoke testing)."
    )
    parser.add_argument(
        "--dataset-root", type=str, default=None,
        help="Override dataset root path."
    )
    parser.add_argument(
        "--train-ratio", type=float, default=None,
        help="Training set ratio (default: 0.70)."
    )
    parser.add_argument(
        "--val-ratio", type=float, default=None,
        help="Validation set ratio (default: 0.15)."
    )
    parser.add_argument(
        "--test-ratio", type=float, default=None,
        help="Test set ratio (default: 0.15)."
    )
    args = parser.parse_args()

    cfg = Config()
    if args.dataset_root:
        cfg.dataset_root = args.dataset_root
    if args.max_videos is not None:
        cfg.max_videos = args.max_videos
    if args.train_ratio is not None:
        cfg.train_ratio = args.train_ratio
    if args.val_ratio is not None:
        cfg.val_ratio = args.val_ratio
    if args.test_ratio is not None:
        cfg.test_ratio = args.test_ratio

    print()
    print("=" * 60)
    print("  Video-Level Dataset Splitting")
    print("=" * 60)

    # Discover videos
    df = discover_videos(cfg)

    # Optionally limit videos per category (for smoke testing)
    if cfg.max_videos is not None:
        logger.info(f"Limiting to {cfg.max_videos} videos per category (smoke test)")
        limited_parts = []
        for mtype, group in df.groupby("manipulation_type"):
            limited_parts.append(group.head(cfg.max_videos))
        df = pd.concat(limited_parts, ignore_index=True)
        logger.info(f"Total videos after limiting: {len(df)}")

    # Split
    print()
    print("Splitting by category:")
    splits = split_videos(
        df,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        random_seed=cfg.random_seed,
    )

    # Verify no overlap
    verify_no_overlap(splits["train"], splits["val"], splits["test"])

    # Save metadata
    metadata_dir = ensure_dir(cfg.metadata_dir)

    for split_name, split_df in splits.items():
        path = metadata_dir / f"{split_name}_videos.csv"
        split_df.to_csv(path, index=False)
        logger.info(f"Saved {split_name} metadata: {path} ({len(split_df)} videos)")

    # Save combined catalogue
    all_path = metadata_dir / "all_videos.csv"
    df.to_csv(all_path, index=False)
    logger.info(f"Saved complete catalogue: {all_path}")

    # Summary
    print()
    print("─" * 60)
    print("  Split Summary")
    print("─" * 60)
    for split_name, split_df in splits.items():
        real_count = (split_df.label == 0).sum()
        fake_count = (split_df.label == 1).sum()
        print(
            f"  {split_name:12s}: {format_count(len(split_df)):>6s} videos "
            f"(REAL={format_count(real_count)}, FAKE={format_count(fake_count)})"
        )
    print("─" * 60)
    print()
    print(f"  Metadata saved to: {metadata_dir}")
    print()


if __name__ == "__main__":
    main()
