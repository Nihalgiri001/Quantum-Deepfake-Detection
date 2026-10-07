"""
CNN Feature Extraction Script
===============================
Loads face crops, passes them through pretrained ResNet-18,
and saves the resulting feature vectors to disk.

Usage:
    python src/extract_features.py

Inputs:
    data/metadata/frame_manifest.csv
    data/frames/{split}/{video_id}/frame_*.jpg

Outputs:
    features/train_features.npy    (N_train × 512)
    features/train_labels.npy      (N_train,)
    features/train_video_ids.npy   (N_train,)
    features/val_features.npy
    features/val_labels.npy
    features/val_video_ids.npy
    features/test_features.npy
    features/test_labels.npy
    features/test_video_ids.npy

Computationally EXPENSIVE — run only when ready.
The features are saved so that subsequent stages (PCA, SVM, QSVM)
do not need to re-run ResNet-18.
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from feature_extractor import ResNetFeatureExtractor
from utils import setup_logger, ensure_dir, set_seed, Timer, format_count

logger = setup_logger("extract_features")


def extract_split_features(
    manifest: pd.DataFrame,
    split_name: str,
    extractor: ResNetFeatureExtractor,
    batch_size: int = 32,
) -> tuple:
    """Extract ResNet-18 features for all frames in a split.

    Args:
        manifest: Frame manifest DataFrame.
        split_name: "train", "val", or "test".
        extractor: ResNetFeatureExtractor instance.
        batch_size: Number of images per batch.

    Returns:
        Tuple of (features_array, labels_array, video_ids_array).
    """
    split_data = manifest[manifest.split == split_name].reset_index(drop=True)

    if len(split_data) == 0:
        logger.warning(f"No frames found for split: {split_name}")
        return (
            np.empty((0, extractor.FEATURE_DIM)),
            np.empty(0, dtype=int),
            np.empty(0, dtype=object),
        )

    all_features = []
    all_labels = []
    all_video_ids = []

    # Process in batches
    n_batches = (len(split_data) + batch_size - 1) // batch_size

    for batch_idx in tqdm(
        range(n_batches),
        desc=f"  {split_name}",
        unit="batch",
    ):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(split_data))
        batch_df = split_data.iloc[start:end]

        image_paths = batch_df.frame_path.tolist()
        labels = batch_df.label.values
        video_ids = batch_df.video_id.values

        # Extract features
        features = extractor.extract_batch(image_paths)

        # Handle cases where some images failed to load
        if len(features) < len(image_paths):
            logger.warning(
                f"Batch {batch_idx}: {len(image_paths) - len(features)} "
                f"images failed to load"
            )
            # Only keep labels/ids for successfully processed images
            labels = labels[:len(features)]
            video_ids = video_ids[:len(features)]

        all_features.append(features)
        all_labels.append(labels)
        all_video_ids.append(video_ids)

    features_array = np.vstack(all_features)
    labels_array = np.concatenate(all_labels)
    video_ids_array = np.concatenate(all_video_ids)

    return features_array, labels_array, video_ids_array


def main():
    parser = argparse.ArgumentParser(
        description="Extract ResNet-18 features from face crops."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size for feature extraction (default: 32)."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Compute device: 'auto', 'cpu', 'mps', or 'cuda'."
    )
    args = parser.parse_args()

    cfg = Config()
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.device is not None:
        cfg.device = args.device

    set_seed(cfg.random_seed)

    print()
    print("=" * 60)
    print("  ResNet-18 Feature Extraction")
    print("=" * 60)
    print(f"  Device: {cfg.get_device()}")
    print(f"  Batch size: {cfg.batch_size}")
    print()

    # Load frame manifest
    manifest_path = cfg.metadata_dir / "frame_manifest.csv"
    if not manifest_path.exists():
        logger.error(
            f"Frame manifest not found: {manifest_path}\n"
            "Run 'python src/extract_frames.py' first."
        )
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    logger.info(f"Loaded manifest with {format_count(len(manifest))} frames")

    # Initialise feature extractor
    extractor = ResNetFeatureExtractor(
        device=cfg.get_device(),
        image_size=cfg.image_size,
    )

    # Extract features for each split
    features_dir = ensure_dir(cfg.features_dir)

    for split_name in ["train", "val", "test"]:
        logger.info(f"Extracting features for {split_name} split...")

        with Timer(f"{split_name} feature extraction", logger):
            features, labels, video_ids = extract_split_features(
                manifest=manifest,
                split_name=split_name,
                extractor=extractor,
                batch_size=cfg.batch_size,
            )

        # Save to disk
        np.save(features_dir / f"{split_name}_features.npy", features)
        np.save(features_dir / f"{split_name}_labels.npy", labels)
        np.save(features_dir / f"{split_name}_video_ids.npy", video_ids)

        logger.info(
            f"  {split_name}: {format_count(len(features))} features saved "
            f"(shape: {features.shape})"
        )

    # Summary
    print()
    print("─" * 60)
    print("  Feature Extraction Summary")
    print("─" * 60)
    for split_name in ["train", "val", "test"]:
        feat = np.load(features_dir / f"{split_name}_features.npy")
        print(f"  {split_name:12s}: {format_count(feat.shape[0]):>6s} samples × {feat.shape[1]} dims")
    print()
    print(f"  Features saved to: {features_dir}")
    print("─" * 60)
    print()


if __name__ == "__main__":
    main()
