"""
Frame Extraction Script
========================
Reads videos from each split, samples frames at a configurable interval,
detects faces, crops them, and saves the face crops to disk.

Usage:
    python src/extract_frames.py
    python src/extract_frames.py --max-videos 20     # smoke test
    python src/extract_frames.py --frame-interval 5  # denser sampling

Inputs:
    data/metadata/train_videos.csv
    data/metadata/val_videos.csv
    data/metadata/test_videos.csv

Outputs:
    data/frames/{split}/{video_id}/frame_{N:06d}.jpg
    data/metadata/frame_manifest.csv   (maps every saved frame to its
                                        video, label, split, and path)

Computationally EXPENSIVE — run only when ready.
"""

import sys
import argparse
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from face_detection import FaceDetector
from preprocessing import save_face_crop
from utils import setup_logger, ensure_dir, set_seed, format_count

logger = setup_logger("extract_frames")


def extract_faces_from_video(
    video_path: str,
    video_id: str,
    output_dir: Path,
    face_detector: FaceDetector,
    frame_interval: int = 10,
    image_size: int = 224,
) -> list:
    """Extract face crops from a single video.

    Args:
        video_path: Path to the video file.
        video_id: Unique identifier for this video.
        output_dir: Directory to save cropped face images.
        face_detector: FaceDetector instance.
        frame_interval: Sample every N-th frame.
        image_size: Output face crop resolution.

    Returns:
        List of dicts with frame metadata (path, frame_number, etc.).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return []

    frame_records = []
    frame_idx = 0
    saved_count = 0
    no_face_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Detect face
            face_crop = face_detector.crop_face(frame)

            if face_crop is not None:
                # Save face crop
                frame_filename = f"frame_{frame_idx:06d}.jpg"
                frame_path = output_dir / frame_filename

                if save_face_crop(face_crop, str(frame_path), image_size):
                    frame_records.append({
                        "frame_path": str(frame_path),
                        "frame_number": frame_idx,
                        "video_id": video_id,
                    })
                    saved_count += 1
            else:
                no_face_count += 1

        frame_idx += 1

    cap.release()

    if no_face_count > 0:
        logger.debug(
            f"  {video_id}: {saved_count} faces saved, "
            f"{no_face_count} frames with no face detected"
        )

    return frame_records


def process_split(
    split_name: str,
    split_df: pd.DataFrame,
    cfg: Config,
    face_detector: FaceDetector,
    max_videos: int = None,
) -> pd.DataFrame:
    """Process all videos in a split.

    Args:
        split_name: "train", "val", or "test".
        split_df: Video metadata for this split.
        cfg: Pipeline configuration.
        face_detector: FaceDetector instance.
        max_videos: Limit videos to process (for smoke testing).

    Returns:
        DataFrame with one row per extracted face frame.
    """
    if max_videos is not None:
        split_df = split_df.head(max_videos)

    split_frames_dir = ensure_dir(cfg.frames_dir / split_name)
    all_records = []

    logger.info(f"Processing {split_name} split: {len(split_df)} videos")

    for _, row in tqdm(
        split_df.iterrows(),
        total=len(split_df),
        desc=f"  {split_name}",
        unit="video",
    ):
        video_id = row["video_id"]
        video_path = row["video_path"]
        label = row["label"]
        label_name = row["label_name"]
        manipulation_type = row["manipulation_type"]

        # Create per-video output directory
        video_output_dir = ensure_dir(split_frames_dir / video_id)

        # Extract faces
        frame_records = extract_faces_from_video(
            video_path=video_path,
            video_id=video_id,
            output_dir=video_output_dir,
            face_detector=face_detector,
            frame_interval=cfg.frame_interval,
            image_size=cfg.image_size,
        )

        # Enrich records with label info
        for rec in frame_records:
            rec["label"] = label
            rec["label_name"] = label_name
            rec["manipulation_type"] = manipulation_type
            rec["split"] = split_name

        all_records.extend(frame_records)

    return pd.DataFrame(all_records)


def main():
    parser = argparse.ArgumentParser(
        description="Extract face crops from FaceForensics++ videos."
    )
    parser.add_argument(
        "--max-videos", type=int, default=None,
        help="Max videos to process PER SPLIT (for smoke testing)."
    )
    parser.add_argument(
        "--frame-interval", type=int, default=None,
        help="Sample every N-th frame (default: 10)."
    )
    parser.add_argument(
        "--dataset-root", type=str, default=None,
        help="Override dataset root path."
    )
    args = parser.parse_args()

    cfg = Config()
    if args.dataset_root:
        cfg.dataset_root = args.dataset_root
    if args.max_videos is not None:
        cfg.max_videos = args.max_videos
    if args.frame_interval is not None:
        cfg.frame_interval = args.frame_interval

    set_seed(cfg.random_seed)

    print()
    print("=" * 60)
    print("  Frame Extraction (Face Crops)")
    print("=" * 60)
    print(f"  Frame interval: every {cfg.frame_interval}th frame")
    print(f"  Image size: {cfg.image_size}×{cfg.image_size}")
    if cfg.max_videos:
        print(f"  Max videos per split: {cfg.max_videos}")
    print()

    # Check that split metadata exists
    metadata_dir = cfg.metadata_dir
    splits = {}
    for split_name in ["train", "val", "test"]:
        csv_path = metadata_dir / f"{split_name}_videos.csv"
        if not csv_path.exists():
            logger.error(
                f"Split metadata not found: {csv_path}\n"
                "Run 'python src/split_dataset.py' first."
            )
            sys.exit(1)
        splits[split_name] = pd.read_csv(csv_path)

    # Initialise face detector
    face_detector = FaceDetector()

    # Process each split
    all_frame_dfs = []
    for split_name in ["train", "val", "test"]:
        frame_df = process_split(
            split_name=split_name,
            split_df=splits[split_name],
            cfg=cfg,
            face_detector=face_detector,
            max_videos=cfg.max_videos,
        )
        all_frame_dfs.append(frame_df)

    # Combine and save frame manifest
    manifest = pd.concat(all_frame_dfs, ignore_index=True)
    manifest_path = metadata_dir / "frame_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    # Summary
    print()
    print("─" * 60)
    print("  Frame Extraction Summary")
    print("─" * 60)
    print(f"  Total face crops extracted: {format_count(len(manifest))}")
    for split_name in ["train", "val", "test"]:
        split_data = manifest[manifest.split == split_name]
        real = (split_data.label == 0).sum()
        fake = (split_data.label == 1).sum()
        print(
            f"  {split_name:12s}: {format_count(len(split_data)):>6s} frames "
            f"(REAL={format_count(real)}, FAKE={format_count(fake)})"
        )
    print()
    print("  Breakdown by manipulation type:")
    for mtype, count in manifest.groupby("manipulation_type").size().items():
        print(f"    {mtype:20s}: {format_count(count)}")
    print()
    print(f"  Frames saved to:   {cfg.frames_dir}")
    print(f"  Manifest saved to: {manifest_path}")
    print("─" * 60)
    print()


if __name__ == "__main__":
    main()
