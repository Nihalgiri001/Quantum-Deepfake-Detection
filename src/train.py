"""
Training Script — Classical Deepfake Detector
================================================
Loads saved ResNet-18 features, applies StandardScaler + PCA,
trains specified classical classifier(s) (Logistic Regression, RBF-SVM,
Random Forest, XGBoost, MLP), and saves all model artefacts.

Usage:
    python src/train.py --model svm
    python src/train.py --model logistic_regression
    python src/train.py --model random_forest
    python src/train.py --model xgboost
    python src/train.py --model mlp
    python src/train.py --model all
    python src/train.py --model xgboost --tune

Outputs:
    models/scaler.pkl
    models/pca.pkl
    models/<model_name>.pkl
    models/training_config.json
"""

import sys
import json
import time
import argparse
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from models import get_model, get_hyperparameter_grid, save_model, AVAILABLE_MODELS
from utils import setup_logger, ensure_dir, set_seed, Timer, format_count

logger = setup_logger("train")


def load_features(features_dir: Path, split: str) -> tuple:
    """Load saved features, labels, and video IDs for a split.

    Args:
        features_dir: Directory containing .npy files.
        split: Split name ("train", "val", "test").

    Returns:
        (features, labels, video_ids) numpy arrays.
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


def train_single_model(
    model_name: str,
    X_train_pca: np.ndarray,
    y_train: np.ndarray,
    X_val_pca: np.ndarray,
    y_val: np.ndarray,
    models_dir: Path,
    cfg: Config,
    tune: bool = False,
) -> dict:
    """Train, optionally tune, evaluate on validation set, and save a single model."""
    logger.info(f"Initialising classifier: '{model_name}'...")
    model_inst = get_model(model_name, random_seed=cfg.random_seed)

    start_time = time.time()
    if tune:
        logger.info(f"Running hyperparameter tuning (GridSearchCV) for '{model_name}'...")
        param_grid = get_hyperparameter_grid(model_name, estimator=model_inst.model)
        if param_grid:
            grid_search = GridSearchCV(
                model_inst.model,
                param_grid,
                cv=3,
                scoring="accuracy",
                n_jobs=-1,
                verbose=1,
            )
            grid_search.fit(X_train_pca, y_train)
            model_inst.model = grid_search.best_estimator_
            logger.info(f"Best hyperparameters for '{model_name}': {grid_search.best_params_}")
        else:
            model_inst.fit(X_train_pca, y_train)
    else:
        model_inst.fit(X_train_pca, y_train)

    train_time = time.time() - start_time
    logger.info(f"Training for '{model_name}' complete in {train_time:.2f}s.")

    # Validation evaluation
    y_val_pred = model_inst.predict(X_val_pca)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    print()
    print("─" * 60)
    print(f"  Validation Results — [{model_inst.display_name}]")
    print("─" * 60)
    print(f"  Validation accuracy: {val_accuracy * 100:.2f}%")
    print()
    print(classification_report(
        y_val, y_val_pred,
        target_names=["REAL", "FAKE"],
        digits=4,
    ))
    print("─" * 60)

    # Save model object
    model_path = models_dir / f"{model_name}.pkl"
    save_model(model_inst, model_path)
    logger.info(f"Saved model: {model_path}")

    # For backward compatibility, if svm is trained, also save to svm.pkl
    if model_name == "svm":
        legacy_svm_path = models_dir / "svm.pkl"
        save_model(model_inst, legacy_svm_path)

    return {
        "model_name": model_name,
        "display_name": model_inst.display_name,
        "train_time_seconds": float(train_time),
        "val_accuracy": float(val_accuracy),
        "model_path": str(model_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train classical deepfake detection classifiers."
    )
    parser.add_argument(
        "--model", "-m", type=str, default="svm",
        help=f"Classifier to train: {AVAILABLE_MODELS + ['all']} (default: svm)."
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run lightweight hyperparameter tuning using GridSearchCV."
    )
    parser.add_argument(
        "--pca-components", type=int, default=None,
        help="Number of PCA components (default: 64)."
    )
    parser.add_argument(
        "--max-samples", type=int, default=25000,
        help="Max training samples for classifier (default: 25000; 0 for all)."
    )
    args = parser.parse_args()

    cfg = Config()
    if args.pca_components is not None:
        cfg.pca_components = args.pca_components

    set_seed(cfg.random_seed)

    models_to_train = (
        AVAILABLE_MODELS if args.model.lower() == "all"
        else [args.model.lower()]
    )

    print()
    print("=" * 60)
    print("  Classical Deepfake Detector — Multi-Classifier Training")
    print("=" * 60)
    print(f"  Classifiers:    {', '.join(models_to_train)}")
    print(f"  PCA components: {cfg.pca_components}")
    print(f"  Max samples:    {args.max_samples if args.max_samples > 0 else 'All'}")
    print(f"  Hyper-tuning:   {args.tune}")
    print()

    # 1. Load features
    logger.info("Loading training features...")
    X_train, y_train, vid_train = load_features(cfg.features_dir, "train")
    logger.info(f"Training set: {format_count(X_train.shape[0])} samples × {X_train.shape[1]} dims")

    logger.info("Loading validation features...")
    X_val, y_val, vid_val = load_features(cfg.features_dir, "val")
    logger.info(f"Validation set: {format_count(X_val.shape[0])} samples × {X_val.shape[1]} dims")

    # 2. Fit StandardScaler on training features only
    logger.info("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 3. Fit PCA on training features only
    n_components = min(cfg.pca_components, X_train_scaled.shape[0], X_train_scaled.shape[1])
    logger.info(f"Fitting PCA with {n_components} components on training data...")
    pca = PCA(n_components=n_components, random_state=cfg.random_seed)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    print()
    print("─" * 60)
    print("  PCA Statistics")
    print("─" * 60)
    print(f"  Components:                {n_components}")
    print(f"  Cumulative explained var:  {cumulative_var[-1] * 100:.2f}%")
    print(f"  Feature dim reduction:     {X_train.shape[1]} → {n_components}")
    print("─" * 60)

    # Subsample training data if specified
    if args.max_samples > 0 and X_train_pca.shape[0] > args.max_samples:
        logger.info(
            f"Subsampling training set from {format_count(X_train_pca.shape[0])} to "
            f"{format_count(args.max_samples)} samples for fast classifier training..."
        )
        X_train_pca, _, y_train, _ = train_test_split(
            X_train_pca,
            y_train,
            train_size=args.max_samples,
            stratify=y_train,
            random_state=cfg.random_seed,
        )
        logger.info(f"Subsampled training set shape: {X_train_pca.shape}")

    models_dir = ensure_dir(cfg.models_dir)

    # Save StandardScaler and PCA
    scaler_path = models_dir / "scaler.pkl"
    pca_path = models_dir / "pca.pkl"
    joblib.dump(scaler, scaler_path)
    joblib.dump(pca, pca_path)
    logger.info(f"Saved scaler: {scaler_path}")
    logger.info(f"Saved PCA:    {pca_path}")

    # 4. Train models
    summary_results = []
    for model_name in models_to_train:
        res = train_single_model(
            model_name=model_name,
            X_train_pca=X_train_pca,
            y_train=y_train,
            X_val_pca=X_val_pca,
            y_val=y_val,
            models_dir=models_dir,
            cfg=cfg,
            tune=args.tune,
        )
        summary_results.append(res)

    # Save training config
    training_config = {
        "pca_components": n_components,
        "cumulative_explained_variance": float(cumulative_var[-1]),
        "trained_models": models_to_train,
        "max_samples_used": args.max_samples,
        "train_samples": int(X_train_pca.shape[0]),
        "val_samples": int(X_val_pca.shape[0]),
        "random_seed": cfg.random_seed,
        "summary": summary_results,
    }
    config_path = models_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(training_config, f, indent=4)
    logger.info(f"Saved config: {config_path}")

    print()
    print("=" * 60)
    print("  Training Complete")
    print("=" * 60)
    print(f"  All models saved to: {models_dir}")
    print()
    print("  Next step: Run evaluation on test set:")
    print("    python src/evaluate.py --model all")
    print()


if __name__ == "__main__":
    main()
