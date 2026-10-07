"""
Evaluation Script — Classical Deepfake Detector
=================================================
Loads saved models (scaler, PCA, SVM) and evaluates on the held-out
test set at BOTH frame level and video level.

Usage:
    python src/evaluate.py

Inputs:
    models/scaler.pkl
    models/pca.pkl
    models/svm.pkl
    features/test_features.npy
    features/test_labels.npy
    features/test_video_ids.npy

Outputs:
    results/confusion_matrix_frame.png
    results/roc_curve_frame.png
    results/confusion_matrix_video.png
    results/roc_curve_video.png
    results/metrics.json
    results/classification_report.txt
    results/experiment_config.json
"""

import sys
import json
import argparse
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from utils import setup_logger, ensure_dir, set_seed, format_count

logger = setup_logger("evaluate")


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["REAL", "FAKE"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str,
    title: str = "ROC Curve",
) -> float:
    """Plot and save an ROC curve. Returns AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")
    return auc


# ──────────────────────────────────────────────────────────────────────────────
# Video-level aggregation
# ──────────────────────────────────────────────────────────────────────────────
def aggregate_video_predictions(
    video_ids: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> tuple:
    """Aggregate frame-level predictions to video level.

    For each video:
    1. Collect all frame fake-probabilities.
    2. Compute the mean probability.
    3. Classify as FAKE if mean probability >= threshold.

    Args:
        video_ids: Array of video IDs per frame.
        labels: Ground truth labels per frame.
        probabilities: Predicted fake probability per frame.
        threshold: Decision threshold.

    Returns:
        (video_labels, video_probs, video_preds, unique_video_ids)
    """
    unique_ids = np.unique(video_ids)
    video_labels = []
    video_probs = []
    video_preds = []

    for vid in unique_ids:
        mask = video_ids == vid
        vid_labels = labels[mask]
        vid_probs = probabilities[mask]

        # All frames in a video should have the same label
        video_label = int(vid_labels[0])
        mean_prob = float(np.mean(vid_probs))
        video_pred = 1 if mean_prob >= threshold else 0

        video_labels.append(video_label)
        video_probs.append(mean_prob)
        video_preds.append(video_pred)

    return (
        np.array(video_labels),
        np.array(video_probs),
        np.array(video_preds),
        unique_ids,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute all evaluation metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted fake probabilities.

    Returns:
        Dictionary of metric name → value.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # ROC-AUC requires both classes present
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
        logger.warning("Only one class present — ROC-AUC cannot be computed.")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the classical deepfake detector on the test set."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Video-level classification threshold (default: 0.5)."
    )
    args = parser.parse_args()

    cfg = Config()
    set_seed(cfg.random_seed)

    print()
    print("=" * 60)
    print("  Classical Deepfake Detector — Evaluation")
    print("=" * 60)
    print()

    # ── Load models ───────────────────────────────────────────────────────────
    models_dir = cfg.models_dir
    for name in ["scaler.pkl", "pca.pkl", "svm.pkl"]:
        if not (models_dir / name).exists():
            logger.error(
                f"Model file not found: {models_dir / name}\n"
                "Run 'python src/train.py' first."
            )
            sys.exit(1)

    scaler = joblib.load(models_dir / "scaler.pkl")
    pca = joblib.load(models_dir / "pca.pkl")
    svm = joblib.load(models_dir / "svm.pkl")
    logger.info("Loaded scaler, PCA, and SVM models.")

    # ── Load test features ────────────────────────────────────────────────────
    features_dir = cfg.features_dir
    for name in ["test_features.npy", "test_labels.npy", "test_video_ids.npy"]:
        if not (features_dir / name).exists():
            logger.error(
                f"Feature file not found: {features_dir / name}\n"
                "Run 'python src/extract_features.py' first."
            )
            sys.exit(1)

    X_test = np.load(features_dir / "test_features.npy")
    y_test = np.load(features_dir / "test_labels.npy")
    video_ids = np.load(features_dir / "test_video_ids.npy", allow_pickle=True)

    logger.info(
        f"Test set: {format_count(X_test.shape[0])} samples × {X_test.shape[1]} dims"
    )

    # ── Apply scaler + PCA ────────────────────────────────────────────────────
    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)

    # ── Frame-level predictions ───────────────────────────────────────────────
    y_pred = svm.predict(X_test_pca)
    y_prob = svm.predict_proba(X_test_pca)[:, 1]  # P(FAKE)

    frame_metrics = compute_metrics(y_test, y_pred, y_prob)

    print()
    print("─" * 60)
    print("  FRAME-LEVEL Results")
    print("─" * 60)
    for metric_name, value in frame_metrics.items():
        if value is not None:
            print(f"  {metric_name:12s}: {value:.4f}")
        else:
            print(f"  {metric_name:12s}: N/A")
    print()

    frame_report = classification_report(
        y_test, y_pred,
        target_names=["REAL", "FAKE"],
        digits=4,
    )
    print(frame_report)

    # ── Video-level predictions ───────────────────────────────────────────────
    vid_labels, vid_probs, vid_preds, unique_vids = aggregate_video_predictions(
        video_ids, y_test, y_prob, threshold=args.threshold,
    )

    video_metrics = compute_metrics(vid_labels, vid_preds, vid_probs)

    print("─" * 60)
    print("  VIDEO-LEVEL Results")
    print("─" * 60)
    print(f"  Total test videos: {format_count(len(unique_vids))}")
    print(f"  Threshold:         {args.threshold}")
    print()
    for metric_name, value in video_metrics.items():
        if value is not None:
            print(f"  {metric_name:12s}: {value:.4f}")
        else:
            print(f"  {metric_name:12s}: N/A")
    print()

    video_report = classification_report(
        vid_labels, vid_preds,
        target_names=["REAL", "FAKE"],
        digits=4,
    )
    print(video_report)

    # ── Generate plots ────────────────────────────────────────────────────────
    results_dir = ensure_dir(cfg.results_dir)

    # Frame-level plots
    plot_confusion_matrix(
        y_test, y_pred,
        save_path=str(results_dir / "confusion_matrix_frame.png"),
        title="Frame-Level Confusion Matrix",
    )

    frame_auc = None
    if frame_metrics["roc_auc"] is not None:
        frame_auc = plot_roc_curve(
            y_test, y_prob,
            save_path=str(results_dir / "roc_curve_frame.png"),
            title="Frame-Level ROC Curve",
        )

    # Video-level plots
    plot_confusion_matrix(
        vid_labels, vid_preds,
        save_path=str(results_dir / "confusion_matrix_video.png"),
        title="Video-Level Confusion Matrix",
    )

    video_auc = None
    if video_metrics["roc_auc"] is not None:
        video_auc = plot_roc_curve(
            vid_labels, vid_probs,
            save_path=str(results_dir / "roc_curve_video.png"),
            title="Video-Level ROC Curve",
        )

    # Safely extract SVM parameters regardless of CalibratedClassifierCV wrapper
    base_est = getattr(svm, "estimator", getattr(svm, "base_estimator", svm))
    svm_kernel = str(getattr(svm, "kernel", getattr(base_est, "kernel", "rbf")))
    svm_c = float(getattr(svm, "C", getattr(base_est, "C", 10.0)))
    svm_gamma = str(getattr(svm, "gamma", getattr(base_est, "gamma", "scale")))

    # ── Save metrics ──────────────────────────────────────────────────────────
    all_metrics = {
        "dataset": "FaceForensics++ (C23)",
        "test_samples_frame": int(len(y_test)),
        "test_samples_video": int(len(unique_vids)),
        "pca_components": int(pca.n_components_),
        "svm_kernel": svm_kernel,
        "svm_c": svm_c,
        "svm_gamma": svm_gamma,
        "video_threshold": args.threshold,
        "frame_level": frame_metrics,
        "video_level": video_metrics,
    }

    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    logger.info(f"Saved: {metrics_path}")

    # Save classification reports
    report_path = results_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("FRAME-LEVEL CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(frame_report + "\n\n")
        f.write("VIDEO-LEVEL CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(video_report + "\n")
    logger.info(f"Saved: {report_path}")

    # Save experiment config
    experiment_config = {
        "project": "Classical Deepfake Detection (Baseline)",
        "dataset": "FaceForensics++ C23",
        "feature_extractor": "ResNet-18 (ImageNet pretrained)",
        "feature_dim_original": int(X_test.shape[1]),
        "feature_dim_pca": int(pca.n_components_),
        "pca_cumulative_variance": float(np.sum(pca.explained_variance_ratio_)),
        "classifier": "RBF-SVM",
        "svm_kernel": svm_kernel,
        "svm_c": svm_c,
        "svm_gamma": svm_gamma,
        "class_weight": "balanced",
        "random_seed": cfg.random_seed,
        "label_encoding": {"REAL": 0, "FAKE": 1},
    }
    exp_config_path = results_dir / "experiment_config.json"
    with open(exp_config_path, "w") as f:
        json.dump(experiment_config, f, indent=4)
    logger.info(f"Saved: {exp_config_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Evaluation Complete")
    print("=" * 60)
    print(f"  Results saved to: {results_dir}")
    print()
    print("  Generated files:")
    print(f"    • {results_dir / 'confusion_matrix_frame.png'}")
    print(f"    • {results_dir / 'roc_curve_frame.png'}")
    print(f"    • {results_dir / 'confusion_matrix_video.png'}")
    print(f"    • {results_dir / 'roc_curve_video.png'}")
    print(f"    • {results_dir / 'metrics.json'}")
    print(f"    • {results_dir / 'classification_report.txt'}")
    print(f"    • {results_dir / 'experiment_config.json'}")
    print()


if __name__ == "__main__":
    main()
