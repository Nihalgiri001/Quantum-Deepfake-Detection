"""
Benchmark Script — Classical Deepfake Detection Classifier Suite
===================================================================
Automates training, timing, evaluation, and comparative benchmarking of all 5 classical
machine learning classifiers (Logistic Regression, RBF-SVM, Random Forest, XGBoost, MLP)
on identical ResNet-18 + PCA feature representations.

Usage:
    python src/benchmark.py --models all
    python src/benchmark.py --models logistic_regression,svm,random_forest,xgboost,mlp
    python src/benchmark.py --max-samples 25000

Outputs:
    results/<model_name>/...
    results/comparison/model_comparison.csv
    results/comparison/comparison_summary.json
    results/comparison/metrics_comparison_frame.png
    results/comparison/metrics_comparison_video.png

IMPORTANT: This script must be executed manually by the user.
"""

import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from models import get_model, save_model, AVAILABLE_MODELS
from evaluate import aggregate_video_predictions, compute_metrics, evaluate_classifier
from utils import setup_logger, ensure_dir, set_seed, Timer, format_count

logger = setup_logger("benchmark")


def load_all_features(features_dir: Path) -> tuple:
    """Load train, val, and test feature arrays."""
    splits = {}
    for s in ["train", "val", "test"]:
        f = features_dir / f"{s}_features.npy"
        l = features_dir / f"{s}_labels.npy"
        v = features_dir / f"{s}_video_ids.npy"
        for p in [f, l, v]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Feature file missing: {p}\n"
                    "Run 'python src/extract_features.py' first."
                )
        splits[s] = (np.load(f), np.load(l), np.load(v, allow_pickle=True))
    return splits["train"], splits["val"], splits["test"]


def generate_comparison_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate and save comparative bar plots for frame and video metrics."""
    output_dir = ensure_dir(output_dir)

    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    model_names = df["display_name"].tolist()

    # 1. Frame-Level Comparison Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    frame_cols = [f"frame_{m}" for m in metrics]
    frame_df = df.set_index("display_name")[frame_cols]
    frame_df.columns = [m.upper().replace("_", "-") for m in metrics]

    frame_df.plot(kind="bar", ax=ax, width=0.8, colormap="viridis")
    ax.set_title("Frame-Level Performance Comparison Across Classifiers", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=12)
    ax.set_xlabel("Classifier Model", fontsize=12)
    ax.set_ylim([0.0, 1.05])
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    plt.legend(loc="lower right")
    plt.tight_layout()

    frame_plot_path = output_dir / "metrics_comparison_frame.png"
    plt.savefig(frame_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {frame_plot_path}")

    # 2. Video-Level Comparison Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    video_cols = [f"video_{m}" for m in metrics]
    video_df = df.set_index("display_name")[video_cols]
    video_df.columns = [m.upper().replace("_", "-") for m in metrics]

    video_df.plot(kind="bar", ax=ax, width=0.8, colormap="magma")
    ax.set_title("Video-Level Performance Comparison Across Classifiers", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=12)
    ax.set_xlabel("Classifier Model", fontsize=12)
    ax.set_ylim([0.0, 1.05])
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    plt.legend(loc="lower right")
    plt.tight_layout()

    video_plot_path = output_dir / "metrics_comparison_video.png"
    plt.savefig(video_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {video_plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark all classical classifiers on identical features."
    )
    parser.add_argument(
        "--models", type=str, default="all",
        help="Comma-separated list of models or 'all'."
    )
    parser.add_argument(
        "--max-samples", type=int, default=25000,
        help="Max training samples for classifiers (default: 25000; 0 for all)."
    )
    parser.add_argument(
        "--pca-components", type=int, default=64,
        help="Number of PCA components (default: 64)."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Video classification threshold (default: 0.5)."
    )
    args = parser.parse_args()

    cfg = Config()
    cfg.pca_components = args.pca_components
    set_seed(cfg.random_seed)

    if args.models.lower() == "all":
        models_to_run = AVAILABLE_MODELS
    else:
        models_to_run = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    print()
    print("=" * 70)
    print("  Classical Deepfake Detector — Benchmark Suite")
    print("=" * 70)
    print(f"  Classifiers to Benchmark: {', '.join(models_to_run)}")
    print(f"  PCA Components:           {cfg.pca_components}")
    print(f"  Max Training Samples:     {args.max_samples if args.max_samples > 0 else 'All'}")
    print("=" * 70)
    print()

    # 1. Load features once
    logger.info("Loading train, val, and test feature arrays...")
    (X_train, y_train, vid_train), (X_val, y_val, vid_val), (X_test, y_test, vid_test) = load_all_features(cfg.features_dir)

    logger.info(f"Train set: {format_count(X_train.shape[0])} samples × {X_train.shape[1]} dims")
    logger.info(f"Val set:   {format_count(X_val.shape[0])} samples × {X_val.shape[1]} dims")
    logger.info(f"Test set:  {format_count(X_test.shape[0])} samples × {X_test.shape[1]} dims")

    # 2. Fit StandardScaler + PCA once on training data
    logger.info("Fitting StandardScaler on training features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    logger.info(f"Fitting PCA ({cfg.pca_components} components) on training features...")
    pca = PCA(n_components=cfg.pca_components, random_state=cfg.random_seed)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Subsample training features if requested
    if args.max_samples > 0 and X_train_pca.shape[0] > args.max_samples:
        logger.info(f"Subsampling training set to {format_count(args.max_samples)} samples...")
        X_train_pca, _, y_train, _ = train_test_split(
            X_train_pca, y_train,
            train_size=args.max_samples,
            stratify=y_train,
            random_state=cfg.random_seed,
        )

    # Save fitted scaler and PCA
    models_dir = ensure_dir(cfg.models_dir)
    save_model(scaler, models_dir / "scaler.pkl")
    save_model(pca, models_dir / "pca.pkl")

    results_base_dir = ensure_dir(cfg.results_dir)
    benchmark_records = []

    # 3. Iterate through each classifier
    for model_name in models_to_run:
        print()
        print("=" * 70)
        print(f"  BENCHMARKING MODEL: [{model_name.upper()}]")
        print("=" * 70)

        model_inst = get_model(model_name, random_seed=cfg.random_seed)

        # Train model
        t_start = time.time()
        model_inst.fit(X_train_pca, y_train)
        t_train = time.time() - t_start
        logger.info(f"[{model_inst.display_name}] Training time: {t_train:.2f}s")

        # Measure inference latency on test set
        t_inf_start = time.time()
        _ = model_inst.predict(X_test_pca)
        t_infer = time.time() - t_inf_start
        infer_latency_ms = (t_infer / len(X_test_pca)) * 1000.0
        logger.info(f"[{model_inst.display_name}] Inference time: {t_infer:.2f}s ({infer_latency_ms:.4f} ms/sample)")

        # Save model
        save_model(model_inst, models_dir / f"{model_name}.pkl")
        if model_name == "svm":
            save_model(model_inst, models_dir / "svm.pkl")

        # Full Evaluation
        eval_metrics = evaluate_classifier(
            model_name=model_name,
            model_obj=model_inst,
            X_test_pca=X_test_pca,
            y_test=y_test,
            video_ids=vid_test,
            pca_n_components=pca.n_components_,
            X_test_orig_dim=X_test.shape[1],
            results_base_dir=results_base_dir,
            cfg=cfg,
            threshold=args.threshold,
        )

        record = {
            "model_name": model_name,
            "display_name": model_inst.display_name,
            "train_time_sec": round(t_train, 2),
            "infer_time_sec": round(t_infer, 2),
            "infer_latency_ms": round(infer_latency_ms, 4),
            "frame_accuracy": eval_metrics["frame_level"]["accuracy"],
            "frame_precision": eval_metrics["frame_level"]["precision"],
            "frame_recall": eval_metrics["frame_level"]["recall"],
            "frame_f1_score": eval_metrics["frame_level"]["f1_score"],
            "frame_roc_auc": eval_metrics["frame_level"]["roc_auc"],
            "video_accuracy": eval_metrics["video_level"]["accuracy"],
            "video_precision": eval_metrics["video_level"]["precision"],
            "video_recall": eval_metrics["video_level"]["recall"],
            "video_f1_score": eval_metrics["video_level"]["f1_score"],
            "video_roc_auc": eval_metrics["video_level"]["roc_auc"],
        }
        benchmark_records.append(record)

    # 4. Generate comparison table & summary
    df_comp = pd.DataFrame(benchmark_records)
    comp_dir = ensure_dir(results_base_dir / "comparison")

    csv_path = comp_dir / "model_comparison.csv"
    df_comp.to_csv(csv_path, index=False)
    logger.info(f"Saved comparison CSV: {csv_path}")

    json_path = comp_dir / "comparison_summary.json"
    with open(json_path, "w") as f:
        json.dump(benchmark_records, f, indent=4)
    logger.info(f"Saved comparison JSON: {json_path}")

    # Generate comparison plots
    generate_comparison_plots(df_comp, comp_dir)

    # 5. Print final comparative terminal summary table
    print()
    print("=" * 80)
    print("  FINAL CLASSIFIER BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    header = (
        f"{'Model':<20} | {'Frame Acc':<9} | {'Frame F1':<8} | {'Frame AUC':<9} | "
        f"{'Vid Acc':<7} | {'Vid F1':<6} | {'Train(s)':<8}"
    )
    print(header)
    print("-" * 80)
    for r in benchmark_records:
        row = (
            f"{r['display_name']:<20} | "
            f"{r['frame_accuracy']*100:8.2f}% | "
            f"{r['frame_f1_score']:8.4f} | "
            f"{(r['frame_roc_auc'] or 0.0):9.4f} | "
            f"{r['video_accuracy']*100:6.2f}% | "
            f"{r['video_f1_score']:6.4f} | "
            f"{r['train_time_sec']:8.2f}s"
        )
        print(row)
    print("=" * 80)
    print()
    print(f"  Full comparison report and plots saved to: {comp_dir}")
    print()


if __name__ == "__main__":
    main()
