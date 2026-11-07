"""
Video Aggregator
=================
Combine per-frame fake probabilities into per-video predictions.
Pure functions on numpy arrays — no Qiskit or sklearn dependencies.

Classes:
    AggregationStrategy     — enum of supported strategies
    VideoAggregationResult  — dataclass holding per-video aggregation results
    VideoAggregator         — aggregate and tune aggregation strategy

Functions:
    pool_frames_per_video   — reduce frame-level features to video-level by pooling
"""

import sys
import logging
import numpy as np
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from sklearn.metrics import f1_score
except ImportError:
    raise ImportError(
        "scikit-learn is required for VideoAggregator.\n"
        "Install it with:  pip install scikit-learn"
    )

try:
    from utils import setup_logger
    logger = setup_logger("quantum.video_aggregator")
except ImportError:
    logger = logging.getLogger("quantum.video_aggregator")
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


# ── Aggregation strategy enum ─────────────────────────────────────────────────

class AggregationStrategy(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    TOP25_MEAN = "top25_mean"
    EQUAL_WEIGHT_MEAN = "equal_weight_mean"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class VideoAggregationResult:
    """Per-video aggregation results."""
    video_ids: np.ndarray           # unique video IDs, shape (V,)
    aggregated_probs: np.ndarray    # per-video fake probability, shape (V,)
    predicted_labels: np.ndarray    # 0=REAL, 1=FAKE, shape (V,)
    threshold_used: float
    strategy_used: str


# ── Core aggregator class ──────────────────────────────────────────────────────

class VideoAggregator:
    """Aggregate per-frame fake probabilities into per-video predictions."""

    # Threshold sweep grid
    THRESHOLD_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    def aggregate(
        self,
        frame_probs: np.ndarray,
        video_ids: np.ndarray,
        strategy: AggregationStrategy,
        threshold: float = 0.5,
    ) -> VideoAggregationResult:
        """
        Group frame_probs by video_ids and aggregate using strategy.

        For top25_mean: when a video has < 2 frames, the single available
        probability is used directly (no top-25% selection).

        Args:
            frame_probs: Per-frame fake probabilities, shape (N,).
            video_ids:   Video ID per frame, shape (N,).
            strategy:    Aggregation strategy.
            threshold:   Fake probability cutoff for label assignment.

        Returns:
            VideoAggregationResult with one entry per unique video.
        """
        frame_probs = np.asarray(frame_probs, dtype=float)
        video_ids = np.asarray(video_ids)

        unique_vids = []
        # Preserve order of first occurrence for determinism
        seen = {}
        for vid in video_ids:
            key = vid if not isinstance(vid, np.ndarray) else vid.item()
            if key not in seen:
                seen[key] = len(unique_vids)
                unique_vids.append(vid)

        aggregated = np.zeros(len(unique_vids), dtype=float)

        for i, vid in enumerate(unique_vids):
            mask = video_ids == vid
            probs = frame_probs[mask]

            if strategy == AggregationStrategy.MEAN:
                aggregated[i] = float(probs.mean())

            elif strategy == AggregationStrategy.MEDIAN:
                aggregated[i] = float(np.median(probs))

            elif strategy == AggregationStrategy.TOP25_MEAN:
                if len(probs) < 2:
                    # Edge case: single frame, use directly
                    aggregated[i] = float(probs[0])
                else:
                    # Top 25% = highest fake probabilities
                    cutoff_idx = max(1, int(np.ceil(0.25 * len(probs))))
                    top_probs = np.sort(probs)[-cutoff_idx:]
                    aggregated[i] = float(top_probs.mean())

            elif strategy == AggregationStrategy.EQUAL_WEIGHT_MEAN:
                # Equal-weight mean — same as mean for uniform weights
                aggregated[i] = float(probs.mean())

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        unique_vids_arr = np.array(unique_vids)
        predicted = (aggregated >= threshold).astype(int)

        return VideoAggregationResult(
            video_ids=unique_vids_arr,
            aggregated_probs=aggregated,
            predicted_labels=predicted,
            threshold_used=threshold,
            strategy_used=str(strategy.value) if isinstance(strategy, AggregationStrategy) else str(strategy),
        )

    def tune_threshold_and_strategy(
        self,
        val_frame_probs: np.ndarray,
        val_video_ids: np.ndarray,
        val_true_labels: Union[np.ndarray, Dict],
    ) -> Tuple[AggregationStrategy, float]:
        """
        Grid search over all strategies × thresholds on validation data.

        Threshold sweep: {0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70}
        Selection criterion: maximise video-level F1 score.

        Args:
            val_frame_probs:  Per-frame fake probabilities.
            val_video_ids:    Video ID per frame.
            val_true_labels:  Dict or array mapping video_id → true label.
                              If array, must be aligned with unique video_ids
                              in order of first occurrence.

        Returns:
            (best_strategy, best_threshold) tuple.
        """
        val_frame_probs = np.asarray(val_frame_probs, dtype=float)
        val_video_ids = np.asarray(val_video_ids)

        # Get unique video IDs in order of first occurrence
        unique_vids = []
        seen = {}
        for vid in val_video_ids:
            key = vid if not isinstance(vid, np.ndarray) else vid.item()
            if key not in seen:
                seen[key] = len(unique_vids)
                unique_vids.append(vid)
        unique_vids_arr = np.array(unique_vids)

        # Build ground-truth label array
        if isinstance(val_true_labels, dict):
            y_true = np.array([val_true_labels[vid] for vid in unique_vids_arr])
        else:
            y_true = np.asarray(val_true_labels)
            assert len(y_true) == len(unique_vids_arr), (
                f"val_true_labels length {len(y_true)} ≠ unique video count {len(unique_vids_arr)}"
            )

        best_f1 = -1.0
        best_strategy = AggregationStrategy.MEAN
        best_threshold = 0.5

        for strategy in AggregationStrategy:
            for threshold in self.THRESHOLD_GRID:
                result = self.aggregate(
                    val_frame_probs, val_video_ids, strategy, threshold
                )
                # Align result labels to unique_vids_arr order
                pred_dict = {
                    vid: label
                    for vid, label in zip(result.video_ids, result.predicted_labels)
                }
                y_pred = np.array([pred_dict[vid] for vid in unique_vids_arr])

                f1 = f1_score(y_true, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_strategy = strategy
                    best_threshold = threshold

        logger.info(
            f"Best aggregation: strategy={best_strategy.value}, "
            f"threshold={best_threshold:.2f}, F1={best_f1:.4f}"
        )
        return best_strategy, best_threshold


# ── Video-level feature pooling (core architectural fix) ─────────────────────

def pool_frames_per_video(
    frame_features: np.ndarray,
    video_ids: np.ndarray,
    method: str = "mean",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reduce frame-level features to video-level features by pooling.

    This is the core architectural fix: instead of encoding individual
    frames (which introduces noise and dilutes the 500-sample budget),
    mean-pool all frames per video to create one feature vector per video.
    The resulting QSVM operates on ~414 videos rather than 500 frames,
    each represented by a cleaner, averaged embedding.

    Args:
        frame_features: Shape (N, D) — raw frame features.
        video_ids:      Shape (N,) — video ID per frame.
        method:         "mean" (default) or "median".

    Returns:
        (video_features, unique_video_ids, video_labels)
        where video_labels are inferred from majority label per video.

    Note:
        video_labels require that frame_features has no label column; the
        caller is expected to pass labels separately. This function returns
        zero-filled labels as a placeholder when called without frame_labels.
        For proper usage, see the version in ExperimentRunner that accepts
        frame_labels explicitly.
    """
    if method not in ("mean", "median"):
        raise ValueError(f"method must be 'mean' or 'median'; got '{method}'")

    frame_features = np.asarray(frame_features)
    video_ids = np.asarray(video_ids)

    # Maintain insertion order of unique video IDs
    seen = {}
    ordered_unique = []
    for vid in video_ids:
        key = vid if not isinstance(vid, np.ndarray) else vid.item()
        if key not in seen:
            seen[key] = len(ordered_unique)
            ordered_unique.append(vid)

    n_videos = len(ordered_unique)
    n_features = frame_features.shape[1]
    video_features = np.zeros((n_videos, n_features), dtype=frame_features.dtype)
    video_labels = np.zeros(n_videos, dtype=int)  # placeholder

    for i, vid in enumerate(ordered_unique):
        mask = video_ids == vid
        frames = frame_features[mask]
        if method == "mean":
            video_features[i] = frames.mean(axis=0)
        else:
            video_features[i] = np.median(frames, axis=0)

    return video_features, np.array(ordered_unique), video_labels


def pool_frames_per_video_with_labels(
    frame_features: np.ndarray,
    video_ids: np.ndarray,
    frame_labels: np.ndarray,
    method: str = "mean",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reduce frame-level features and labels to video-level.

    Video label is determined by majority vote of frame labels.

    Args:
        frame_features: Shape (N, D).
        video_ids:      Shape (N,).
        frame_labels:   Shape (N,) — integer class labels (0=REAL, 1=FAKE).
        method:         "mean" or "median".

    Returns:
        (video_features, unique_video_ids, video_labels)
    """
    video_features, unique_vids, _ = pool_frames_per_video(
        frame_features, video_ids, method
    )

    frame_labels = np.asarray(frame_labels)
    video_labels = np.zeros(len(unique_vids), dtype=int)

    for i, vid in enumerate(unique_vids):
        mask = video_ids == vid
        labels = frame_labels[mask]
        # Majority vote
        video_labels[i] = int(np.bincount(labels).argmax())

    return video_features, unique_vids, video_labels
