"""
Training Script — Classical Deepfake Detector
================================================
Loads saved ResNet-18 features, applies StandardScaler + PCA,
trains an RBF-SVM classifier, and saves all model artefacts.

Usage:
    python src/train.py
    python src/train.py --pca-components 32
    python src/train.py --svm-c 1.0 --svm-gamma 0.01

Pipeline:
    train_features.npy → StandardScaler → PCA → RBF-SVM

Outputs:
    models/scaler.pkl
    models/pca.pkl
    models/svm.pkl
    models/training_config.json

IMPORTANT — This script must be run by the user manually.
"""

import sys
import json
import argparse
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from utils import setup_logger, ensure_dir, set_seed, Timer, format_count

logger = setup_logger("train")


def load_features(features_dir: Path, split: str) -> tuple:
    """Load saved features, labels, and video IDs for a split.

    Args:
        features_dir: Directory containing .npy files.
        split: Split name ("train", "val", "test").

    Returns:
        (features, labels, video_ids) numpy arrays.

    Raises:
        FileNotFoundError: If feature files are missing.
    """
    feat_path = features_dir / f"{split}_features.npy"
    label_path = features_dir / f"{split}_labels.npy"
    vid_path = features_dir / f"{split}_video_ids.npy"

    for p in [feat_path, label_path, vid_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Feature file not found: {p}\n"
                "Run 'python src/extract_features.py' first."
            )

    features = np.load(feat_path)
    labels = np.load(label_path)
    video_ids = np.load(vid_path, allow_pickle=True)

    return features, labels, video_ids


def main():
    parser = argparse.ArgumentParser(
        description="Train the classical RBF-SVM deepfake detector."
    )
    parser.add_argument("--pca-components", type=int, default=None,
                        help="Number of PCA components (default: 64).")
    parser.add_argument("--svm-c", type=float, default=None,
                        help="SVM regularisation parameter C (default: 10.0).")
    parser.add_argument("--svm-gamma", type=str, default=None,
                        help="SVM gamma parameter (default: 'scale').")
    parser.add_argument("--svm-kernel", type=str, default=None,
                        help="SVM kernel type (default: 'rbf').")
    parser.add_argument("--max-samples", type=int, default=25000,
                        help="Max training samples for SVM (default: 25000 for fast training; 0 for all).")
    args = parser.parse_args()

    cfg = Config()
    if args.pca_components is not None:
        cfg.pca_components = args.pca_components
    if args.svm_c is not None:
        cfg.svm_c = args.svm_c
    if args.svm_gamma is not None:
        cfg.svm_gamma = args.svm_gamma
    if args.svm_kernel is not None:
        cfg.svm_kernel = args.svm_kernel

    set_seed(cfg.random_seed)

    print()
    print("=" * 60)
    print("  Classical Deepfake Detector — Training")
    print("=" * 60)
    print(f"  PCA components: {cfg.pca_components}")
    print(f"  SVM kernel:     {cfg.svm_kernel}")
    print(f"  SVM C:          {cfg.svm_c}")
    print(f"  SVM gamma:      {cfg.svm_gamma}")
    print(f"  Max samples:    {args.max_samples if args.max_samples > 0 else 'All'}")
    print()

    # ── Step 1: Load features ─────────────────────────────────────────────────
    logger.info("Loading training features...")
    X_train, y_train, vid_train = load_features(cfg.features_dir, "train")
    logger.info(f"Training set: {format_count(X_train.shape[0])} samples × {X_train.shape[1]} dims")

    logger.info("Loading validation features...")
    X_val, y_val, vid_val = load_features(cfg.features_dir, "val")
    logger.info(f"Validation set: {format_count(X_val.shape[0])} samples × {X_val.shape[1]} dims")

    # ── Step 2: StandardScaler (fit on TRAINING data only) ────────────────────
    logger.info("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    logger.info("StandardScaler fitted and applied.")

    # ── Step 3: PCA (fit on TRAINING data only) ───────────────────────────────
    n_components = min(cfg.pca_components, X_train_scaled.shape[0], X_train_scaled.shape[1])
    if n_components != cfg.pca_components:
        logger.warning(
            f"Adjusted PCA components from {cfg.pca_components} to {n_components} "
            f"(limited by data dimensions)"
        )

    logger.info(f"Fitting PCA with {n_components} components on training data...")
    pca = PCA(n_components=n_components, random_state=cfg.random_seed)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    # PCA statistics
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    print()
    print("─" * 60)
    print("  PCA Results")
    print("─" * 60)
    print(f"  Components:                {n_components}")
    print(f"  Cumulative explained var:  {cumulative_var[-1] * 100:.2f}%")
    print(f"  Feature dim reduction:     {X_train.shape[1]} → {n_components}")
    print("─" * 60)

    # Subsample if dataset is large for fast RBF-SVM training
    if args.max_samples > 0 and X_train_pca.shape[0] > args.max_samples:
        from sklearn.model_selection import train_test_split
        logger.info(
            f"Subsampling training set from {format_count(X_train_pca.shape[0])} to "
            f"{format_count(args.max_samples)} samples for fast SVM training..."
        )
        X_train_pca, _, y_train, _ = train_test_split(
            X_train_pca,
            y_train,
            train_size=args.max_samples,
            stratify=y_train,
            random_state=cfg.random_seed,
        )
        logger.info(f"Subsampled training set shape: {X_train_pca.shape}")

    # ── Step 4: Train RBF-SVM ─────────────────────────────────────────────────
    logger.info("Training RBF-SVM classifier...")

    # Parse gamma (could be "scale", "auto", or a float)
    gamma = cfg.svm_gamma
    try:
        gamma = float(gamma)
    except (ValueError, TypeError):
        pass  # Keep as string ("scale" or "auto")

    base_svc = SVC(
        kernel=cfg.svm_kernel,
        C=cfg.svm_c,
        gamma=gamma,
        class_weight="balanced",
        random_state=cfg.random_seed,
    )

    try:
        from sklearn.calibration import CalibratedClassifierCV
        svm = CalibratedClassifierCV(base_svc, cv=3, ensemble=False)
    except Exception:
        svm = SVC(
            kernel=cfg.svm_kernel,
            C=cfg.svm_c,
            gamma=gamma,
            class_weight="balanced",
            probability=True,
            random_state=cfg.random_seed,
        )

    with Timer("SVM training", logger):
        svm.fit(X_train_pca, y_train)

    logger.info("SVM training complete.")

    # ── Step 5: Validation performance ────────────────────────────────────────
    y_val_pred = svm.predict(X_val_pca)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    print()
    print("─" * 60)
    print("  Validation Results")
    print("─" * 60)
    print(f"  Validation accuracy: {val_accuracy * 100:.2f}%")
    print()
    print(classification_report(
        y_val, y_val_pred,
        target_names=["REAL", "FAKE"],
        digits=4,
    ))
    print("─" * 60)

    # ── Step 6: Save model artefacts ──────────────────────────────────────────
    models_dir = ensure_dir(cfg.models_dir)

    scaler_path = models_dir / "scaler.pkl"
    pca_path = models_dir / "pca.pkl"
    svm_path = models_dir / "svm.pkl"
    config_path = models_dir / "training_config.json"

    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved scaler:  {scaler_path}")

    joblib.dump(pca, pca_path)
    logger.info(f"Saved PCA:     {pca_path}")

    joblib.dump(svm, svm_path)
    logger.info(f"Saved SVM:     {svm_path}")

    # Save training configuration
    training_config = {
        "pca_components": n_components,
        "cumulative_explained_variance": float(cumulative_var[-1]),
        "svm_kernel": cfg.svm_kernel,
        "svm_c": cfg.svm_c,
        "svm_gamma": str(cfg.svm_gamma),
        "class_weight": "balanced",
        "train_samples": int(X_train_pca.shape[0]),
        "val_samples": int(X_val_pca.shape[0]),
        "val_accuracy": float(val_accuracy),
        "feature_dim_original": int(X_train.shape[1]),
        "feature_dim_after_pca": int(n_components),
        "random_seed": cfg.random_seed,
    }
    with open(config_path, "w") as f:
        json.dump(training_config, f, indent=4)
    logger.info(f"Saved config:  {config_path}")

    print()
    print("=" * 60)
    print("  Training Complete")
    print("=" * 60)
    print(f"  Models saved to: {models_dir}")
    print()
    print("  Next step: Run evaluation on the held-out test set:")
    print("    python src/evaluate.py")
    print()


if __name__ == "__main__":
    main()
