"""
Dataset Verification Script
=============================
Checks that the FaceForensics++ dataset is present and correctly structured.

Usage:
    python src/verify_dataset.py

What it does:
    1. Validates that the configured dataset_root exists.
    2. Checks for expected sub-directories (original + 4 manipulation types).
    3. Counts videos in each category.
    4. Reports a summary table.

This script is FAST (no video processing) and should be run first.
"""

import sys
import argparse
from pathlib import Path

# ── Ensure src/ is on the Python path ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from dataset import discover_videos, get_dataset_stats
from utils import setup_logger, format_count

logger = setup_logger("verify_dataset")


def verify_directory_structure(cfg: Config) -> bool:
    """Check that all expected directories exist.

    Returns:
        True if all directories are present, False otherwise.
    """
    all_ok = True

    # Dataset root
    if not cfg.dataset_path.exists():
        logger.error(f"Dataset root NOT FOUND: {cfg.dataset_path}")
        logger.error(
            "Please update 'dataset_root' in src/config.py or place the "
            "FaceForensics++ dataset at the expected location."
        )
        return False
    logger.info(f"Dataset root found: {cfg.dataset_path}")

    # Original videos
    if not cfg.original_videos_dir.exists():
        logger.error(f"Original videos directory NOT FOUND: {cfg.original_videos_dir}")
        all_ok = False
    else:
        logger.info(f"Original videos directory: OK")

    # Manipulation directories
    for manip_type in cfg.manipulation_types:
        manip_dir = cfg.get_manipulation_dir(manip_type)
        if not manip_dir.exists():
            logger.warning(f"{manip_type} directory NOT FOUND: {manip_dir}")
            all_ok = False
        else:
            logger.info(f"{manip_type} directory: OK")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Verify FaceForensics++ dataset structure and contents."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Override the dataset root path from config.py",
    )
    args = parser.parse_args()

    cfg = Config()
    if args.dataset_root:
        cfg.dataset_root = args.dataset_root

    print()
    print("=" * 60)
    print("  FaceForensics++ Dataset Verification")
    print("=" * 60)
    print(f"  Dataset root: {cfg.dataset_root}")
    print()

    # Step 1: Directory structure
    print("─" * 60)
    print("  Step 1: Checking directory structure")
    print("─" * 60)
    structure_ok = verify_directory_structure(cfg)
    print()

    if not structure_ok:
        print("❌ Directory structure verification FAILED.")
        print("   Some expected directories are missing.")
        print("   Please check the README for dataset setup instructions.")
        sys.exit(1)

    # Step 2: Discover and count videos
    print("─" * 60)
    print("  Step 2: Counting videos")
    print("─" * 60)
    try:
        df = discover_videos(cfg)
    except Exception as e:
        logger.error(f"Video discovery failed: {e}")
        sys.exit(1)

    stats = get_dataset_stats(df)
    print()
    print(f"  Total videos found:    {format_count(stats['total_videos'])}")
    print(f"  Real (original):       {format_count(stats['real_videos'])}")
    print(f"  Fake (manipulated):    {format_count(stats['fake_videos'])}")
    print()
    print("  Breakdown by manipulation type:")
    for mtype, count in sorted(stats["manipulation_breakdown"].items()):
        print(f"    {mtype:20s}: {format_count(count)}")

    print()
    print("─" * 60)
    print("  ✅ Dataset verification PASSED")
    print("─" * 60)
    print()


if __name__ == "__main__":
    main()
