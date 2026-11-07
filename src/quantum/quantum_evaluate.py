"""
Quantum QSVM Evaluation Module
================================
Evaluates a trained QuantumSVM at both frame level and video level,
generates all result artefacts, and produces a classical-vs-quantum
comparison CSV.

Design decisions:
- Imports ``aggregate_video_predictions``, ``compute_metrics``,
  ``plot_confusion_matrix``, and ``plot_roc_curve`` directly from the
  existing ``src/evaluate.py`` — no duplication.
- The QSVM model, QuantumFeatureReducer, and cached kernel matrices are
  all loaded from ``models_dir/quantum/`` and ``features/quantum/``.
- ``classical_vs_quantum.csv`` is always written even when no classical
  SVM results exist; in that case only the QSVM row is present.
- Video IDs and labels are loaded from ``features/quantum/`` where
  ``train_full_pipeline`` saved the subsampled split metadata.  If those
  files are absent (e.g. after a manual partial run), they fall back to
  the original ``features/`` arrays.

Output artefacts:
    results/quantum/metrics.json
    results/quantum/classification_report.txt
    results/quantum/confusion_matrix.png
    results/quantum/roc_curve.png
    results/quantum/video_level_metrics.json
    results/quantum/classical_vs_quantum.csv

CLI:
    python -m src.quantum.quantum_evaluate
    python -m src.quantum.quantum_evaluate --backend simulator
    python -m src.quantum.quantum_evaluate --split test
    python -m src.quantum.quantum_evaluate --threshold 0.4
"""

import sys
import json
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Classical helpers (import once, reuse) ────────────────────────────────────
try:
    from evaluate import (
        aggregate_video_predictions,
        compute_metrics,
        plot_confusion_matrix,
        plot_roc_curve,
    )
except ImportError as exc:
    raise ImportError(
        "Could not import evaluation helpers from src/evaluate.py. "
        "Ensure src/ is on sys.path."
    ) from exc

from quantum.quantum_config import QuantumConfig
from quantum.quantum_svm import QuantumSVM
from quantum.feature_reducer import QuantumFeatureReducer
from quantum.quantum_kernel import QuantumKernelComputer
from quantum.simulator_backend import SimulatorBackend

try:
    from utils import setup_logger, ensure_dir
    logger = setup_logger("quantum.evaluate")
except ImportError:
    logger = logging.getLogger("quantum.evaluate")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO,
                            format="[%(asctime)s] %(levelname)-8s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    def ensure_dir(p):
        Path(p).mkdir(parents=True, exist_ok=True)
        return Path(p)


# ── Expected metric keys ──────────────────────────────────────────────────────
EXPECTED_METRIC_KEYS = {"accuracy", "precision", "recall", "f1_score", "roc_auc"}


# ── Core evaluation function ──────────────────────────────────────────────────

def evaluate_qsvm(
    config: Optional[QuantumConfig] = None,
    models_dir: Optional[str] = None,
    features_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
    backend_type: str = "simulator",
    split: str = "test",
    threshold: float = 0.5,
    kernel_cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a trained QSVM at frame and video level.

    Args:
        config:           QuantumConfig.  Loaded from ``models_dir`` if None.
        models_dir:       Directory containing ``quantum_svm.pkl`` etc.
        features_dir:     Directory containing raw ResNet-18 feature arrays.
        results_dir:      Output directory for result artefacts.
        backend_type:     "simulator" (the only local option).
        split:            Which split to evaluate: "test" (default) or "val".
        threshold:        Video-level fake probability threshold.
        kernel_cache_dir: Override kernel cache location.

    Returns:
        Dictionary with frame_metrics, video_metrics, and file paths.
    """
    # ── Resolve paths ─────────────────────────────────────────────────────────
    _root = Path(__file__).resolve().parent.parent.parent
    if models_dir is None:
        models_dir = str(_root / "models" / "quantum")
    if features_dir is None:
        features_dir = str(_root / "features")
    if results_dir is None:
        results_dir = str(_root / "results" / "quantum")
    if kernel_cache_dir is None:
        kernel_cache_dir = str(_root / "features" / "quantum")

    md = Path(models_dir)
    fd = Path(features_dir)
    rd = ensure_dir(Path(results_dir))
    kd = Path(kernel_cache_dir)

    # ── Load config ───────────────────────────────────────────────────────────
    if config is None:
        cfg_path = md / "quantum_config.json"
        if cfg_path.exists():
            config = QuantumConfig.load_json(str(cfg_path))
        else:
            config = QuantumConfig()
            logger.warning("quantum_config.json not found; using defaults.")

    # ── Load trained QSVM ─────────────────────────────────────────────────────
    logger.info(f"Loading QuantumSVM from {md}")
    qsvm = QuantumSVM.load(str(md))

    # ── Load saved training features (for cross-kernel) ───────────────────────
    X_train_q = qsvm._train_features_reduced
    if X_train_q is None:
        raise RuntimeError(
            "train_features_reduced.npy not found in model artefacts. "
            "Re-run '--train' to regenerate."
        )

    # ── Load evaluation split features ────────────────────────────────────────
    # Prefer the subsampled split arrays saved by train_full_pipeline
    # (in features/quantum/); fall back to full arrays in features/.
    def _load_split(split_name: str):
        q_feat  = kd / f"{split_name}_features_q.npy"
        q_label = kd / f"{split_name}_labels.npy"
        q_vid   = kd / f"{split_name}_video_ids.npy"

        if q_feat.exists() and q_label.exists() and q_vid.exists():
            X   = np.load(str(q_feat))
            y   = np.load(str(q_label))
            vid = np.load(str(q_vid), allow_pickle=True)
            logger.info(f"Loaded pre-reduced {split_name} features from {kd}")
            return X, y, vid, True   # already_reduced=True

        # Fall back: raw 512-dim features
        feat_path  = fd / f"{split_name}_features.npy"
        label_path = fd / f"{split_name}_labels.npy"
        vid_path   = fd / f"{split_name}_video_ids.npy"

        # Also check subsampled labels/video_ids in kd
        if (kd / f"{split_name}_labels.npy").exists():
            label_path = kd / f"{split_name}_labels.npy"
        if (kd / f"{split_name}_video_ids.npy").exists():
            vid_path = kd / f"{split_name}_video_ids.npy"

        for p in [feat_path, label_path, vid_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Feature file not found: {p}\n"
                    "Run 'python -m src.quantum.quantum_svm --train' first."
                )
        X   = np.load(str(feat_path))
        y   = np.load(str(label_path))
        vid = np.load(str(vid_path), allow_pickle=True)
        logger.info(f"Loaded raw {split_name} features from {fd} ({X.shape[0]} samples)")
        return X, y, vid, False  # already_reduced=False

    X_eval, y_eval, vid_eval, already_reduced = _load_split(split)

    # ── Apply QuantumFeatureReducer if needed ─────────────────────────────────
    if not already_reduced:
        reducer = QuantumFeatureReducer.load(str(md))
        X_eval_q = reducer.transform(X_eval)
        logger.info(f"Reduced {split} features: {X_eval.shape} → {X_eval_q.shape}")
    else:
        X_eval_q = X_eval

    # ── Subsample eval set if too large ──────────────────────────────────────
    max_eval = config.max_test_samples if split == "test" else config.max_val_samples
    if X_eval_q.shape[0] > max_eval:
        from quantum.kernel_utils import subsample_balanced
        X_eval_q, y_eval, idx = subsample_balanced(
            X_eval_q, y_eval, max_eval, seed=config.random_seed
        )
        vid_eval = vid_eval[idx]
        logger.info(f"Subsampled {split} to {X_eval_q.shape[0]} samples")

    n_eval   = X_eval_q.shape[0]
    n_train  = X_train_q.shape[0]

    # ── Build or load cross-kernel ─────────────────────────────────────────────
    backend  = SimulatorBackend(shots=config.shots, seed=config.random_seed)
    computer = QuantumKernelComputer(
        config=config, backend=backend, cache_dir=str(kd)
    )
    K_name = f"K_{split}"
    logger.info(f"Loading or computing {K_name} ({n_eval}×{n_train})…")
    K_eval = computer.compute_cross_kernel(
        X_eval_q, X_train_q, name=K_name, use_cache=True
    )

    # ── Frame-level inference ─────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        y_prob = qsvm.predict_proba(K_eval)[:, 1]   # P(FAKE)
        y_pred = qsvm.predict(K_eval)

    frame_metrics = compute_metrics(y_eval, y_pred, y_prob)

    # ── Video-level aggregation ───────────────────────────────────────────────
    vid_labels, vid_probs, vid_preds, unique_vids = aggregate_video_predictions(
        vid_eval, y_eval, y_prob, threshold=threshold
    )
    video_metrics = compute_metrics(vid_labels, vid_preds, vid_probs)

    # ── Print to stdout ───────────────────────────────────────────────────────
    from sklearn.metrics import classification_report as skl_cls_report
    print()
    print("─" * 60)
    print(f"  FRAME-LEVEL Results — [Quantum Kernel SVM]")
    print("─" * 60)
    for k, v in frame_metrics.items():
        print(f"  {k:12s}: {v:.4f}" if v is not None else f"  {k:12s}: N/A")
    print()
    frame_report = skl_cls_report(
        y_eval, y_pred, target_names=["REAL", "FAKE"], digits=4
    )
    print(frame_report)

    print("─" * 60)
    print(f"  VIDEO-LEVEL Results — [Quantum Kernel SVM]")
    print("─" * 60)
    print(f"  Test videos : {len(unique_vids)}")
    print(f"  Threshold   : {threshold}")
    for k, v in video_metrics.items():
        print(f"  {k:12s}: {v:.4f}" if v is not None else f"  {k:12s}: N/A")
    print()
    video_report = skl_cls_report(
        vid_labels, vid_preds, target_names=["REAL", "FAKE"], digits=4
    )
    print(video_report)

    # ── Save artefacts ────────────────────────────────────────────────────────

    # metrics.json
    all_metrics = {
        "model_name": "quantum_svm",
        "display_name": "Quantum Kernel SVM",
        "dataset": "FaceForensics++ (C23)",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "threshold": threshold,
        "n_eval_frames": int(n_eval),
        "n_eval_videos": int(len(unique_vids)),
        "num_qubits": config.num_qubits,
        "feature_dimension": config.feature_dimension,
        "reps": config.reps,
        "entanglement": config.entanglement,
        "frame_level": {k: (float(v) if v is not None else None)
                        for k, v in frame_metrics.items()},
        "video_level": {k: (float(v) if v is not None else None)
                        for k, v in video_metrics.items()},
    }
    with open(rd / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)
    logger.info(f"Saved: {rd / 'metrics.json'}")

    # classification_report.txt
    with open(rd / "classification_report.txt", "w") as f:
        f.write("FRAME-LEVEL CLASSIFICATION REPORT — [Quantum Kernel SVM]\n")
        f.write("=" * 60 + "\n")
        f.write(frame_report + "\n\n")
        f.write("VIDEO-LEVEL CLASSIFICATION REPORT — [Quantum Kernel SVM]\n")
        f.write("=" * 60 + "\n")
        f.write(video_report + "\n")
    logger.info(f"Saved: {rd / 'classification_report.txt'}")

    # confusion_matrix.png (frame level)
    plot_confusion_matrix(
        y_eval, y_pred,
        save_path=str(rd / "confusion_matrix.png"),
        title="Frame-Level Confusion Matrix (Quantum Kernel SVM)",
    )

    # roc_curve.png (frame level)
    if frame_metrics.get("roc_auc") is not None:
        plot_roc_curve(
            y_eval, y_prob,
            save_path=str(rd / "roc_curve.png"),
            title="Frame-Level ROC Curve (Quantum Kernel SVM)",
        )

    # video_level_metrics.json
    with open(rd / "video_level_metrics.json", "w") as f:
        json.dump({
            "n_videos": int(len(unique_vids)),
            "threshold": threshold,
            "metrics": {k: (float(v) if v is not None else None)
                        for k, v in video_metrics.items()},
        }, f, indent=4)
    logger.info(f"Saved: {rd / 'video_level_metrics.json'}")

    # classical_vs_quantum.csv
    _write_comparison_csv(all_metrics, rd)

    return {
        "frame_metrics": frame_metrics,
        "video_metrics": video_metrics,
        "results_dir": str(rd),
        "n_eval_frames": int(n_eval),
        "n_eval_videos": int(len(unique_vids)),
    }


def _write_comparison_csv(qsvm_metrics: Dict[str, Any], results_dir: Path) -> None:
    """Write classical_vs_quantum.csv comparing SVM and QSVM results.

    Gracefully handles missing classical SVM results: writes QSVM-only row.

    Args:
        qsvm_metrics: Metrics dict from evaluate_qsvm().
        results_dir:  Output directory (results/quantum/).
    """
    rows = []

    # Try to load classical SVM results
    _root = results_dir.parent.parent  # results/ → project root
    svm_path = _root / "results" / "svm" / "metrics.json"
    if not svm_path.exists():
        # Also try top-level results/metrics.json
        svm_path = _root / "results" / "metrics.json"

    if svm_path.exists():
        try:
            with open(svm_path) as f:
                svm_data = json.load(f)
            fl = svm_data.get("frame_level", {})
            vl = svm_data.get("video_level", {})
            rows.append({
                "model": "Classical RBF-SVM",
                "frame_accuracy":  fl.get("accuracy"),
                "frame_precision": fl.get("precision"),
                "frame_recall":    fl.get("recall"),
                "frame_f1":        fl.get("f1_score"),
                "frame_roc_auc":   fl.get("roc_auc"),
                "video_accuracy":  vl.get("accuracy"),
                "video_precision": vl.get("precision"),
                "video_recall":    vl.get("recall"),
                "video_f1":        vl.get("f1_score"),
                "video_roc_auc":   vl.get("roc_auc"),
            })
            logger.info(f"Loaded classical SVM metrics from {svm_path}")
        except Exception as e:
            logger.warning(f"Could not parse classical SVM metrics: {e}")

    # QSVM row
    qfl = qsvm_metrics.get("frame_level", {})
    qvl = qsvm_metrics.get("video_level", {})
    rows.append({
        "model": f"Quantum Kernel SVM ({qsvm_metrics.get('num_qubits', '?')}q)",
        "frame_accuracy":  qfl.get("accuracy"),
        "frame_precision": qfl.get("precision"),
        "frame_recall":    qfl.get("recall"),
        "frame_f1":        qfl.get("f1_score"),
        "frame_roc_auc":   qfl.get("roc_auc"),
        "video_accuracy":  qvl.get("accuracy"),
        "video_precision": qvl.get("precision"),
        "video_recall":    qvl.get("recall"),
        "video_f1":        qvl.get("f1_score"),
        "video_roc_auc":   qvl.get("roc_auc"),
    })

    df = pd.DataFrame(rows)
    csv_path = results_dir / "classical_vs_quantum.csv"
    df.to_csv(str(csv_path), index=False)
    logger.info(f"Saved: {csv_path}")

    # Pretty-print comparison to stdout
    print()
    print("─" * 60)
    print("  Classical vs Quantum Comparison")
    print("─" * 60)
    header = f"  {'Model':<30} | {'Frm Acc':>7} | {'Frm F1':>6} | {'Vid Acc':>7} | {'Vid F1':>6}"
    print(header)
    print("  " + "─" * 56)
    for _, row in df.iterrows():
        fa  = f"{row['frame_accuracy']:.4f}"  if row['frame_accuracy']  is not None else "  N/A "
        ff  = f"{row['frame_f1']:.4f}"        if row['frame_f1']        is not None else "  N/A "
        va  = f"{row['video_accuracy']:.4f}"  if row['video_accuracy']  is not None else "  N/A "
        vf  = f"{row['video_f1']:.4f}"        if row['video_f1']        is not None else "  N/A "
        print(f"  {str(row['model']):<30} | {fa:>7} | {ff:>6} | {va:>7} | {vf:>6}")
    print("─" * 60)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the trained Quantum Kernel SVM."
    )
    parser.add_argument(
        "--backend", type=str, default="simulator",
        choices=["simulator"],
        help="Execution backend (default: simulator)."
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["test", "val"],
        help="Dataset split to evaluate (default: test)."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Video-level fake probability threshold (default: 0.5)."
    )
    parser.add_argument(
        "--models-dir", type=str, default=None,
        help="Override models directory."
    )
    parser.add_argument(
        "--features-dir", type=str, default=None,
        help="Override features directory."
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Override results output directory."
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  TrustLens-Q — Quantum Kernel SVM Evaluation")
    print("=" * 60)
    print(f"  Split     : {args.split}")
    print(f"  Threshold : {args.threshold}")
    print(f"  Backend   : {args.backend}")
    print()

    results = evaluate_qsvm(
        models_dir=args.models_dir,
        features_dir=args.features_dir,
        results_dir=args.results_dir,
        backend_type=args.backend,
        split=args.split,
        threshold=args.threshold,
    )

    print()
    print("=" * 60)
    print("  Evaluation Complete")
    print("=" * 60)
    print(f"  Results saved to: {results['results_dir']}")
    print()


if __name__ == "__main__":
    main()
