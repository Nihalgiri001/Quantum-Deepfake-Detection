"""
Quantum Experiment Runner
==========================
Single entry point for running any QSVM experiment configuration end-to-end,
with reproducible artefact management.

Classes:
    ExperimentConfig  — dataclass of all experiment parameters
    ExperimentResult  — dataclass of all experiment outputs
    ExperimentRunner  — orchestrates experiment execution

CLI:
    python -m src.quantum.quantum_experiment --feature-dim 8
    python -m src.quantum.quantum_experiment --feature-dim 8 --reps 2 --svm-c 10.0
    python -m src.quantum.quantum_experiment --config-json path/to/config.json
    python -m src.quantum.quantum_experiment --dry-run
"""

import csv
import hashlib
import json
import logging
import random
import sys
import time
import warnings
import argparse
import math
import numpy as np
import joblib

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantum.quantum_config import QuantumConfig
from quantum.feature_reducer import QuantumFeatureReducer
from quantum.quantum_kernel import QuantumKernelComputer, KernelNumericalError
from quantum.quantum_svm import QuantumSVM
from quantum.simulator_backend import SimulatorBackend
from quantum.kernel_utils import hash_kernel_config, subsample_balanced
from quantum.quantum_diagnostics import DiagnosticsEngine, DiagnosticsReport
from quantum.video_aggregator import (
    VideoAggregator,
    AggregationStrategy,
    pool_frames_per_video_with_labels,
)
from quantum.classical_baseline import ClassicalBaselineSVM

try:
    from utils import setup_logger
    logger = setup_logger("quantum.experiment")
except ImportError:
    logger = logging.getLogger("quantum.experiment")
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


# ── ExperimentConfig ──────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """All parameters for a single QSVM experiment."""

    # Feature reduction
    feature_dimension: int = 4
    feature_range_low: float = -math.pi
    feature_range_high: float = math.pi
    use_video_pooling: bool = False

    # Feature map
    feature_map_type: str = "zz"          # "zz" | "pauli" | "reuploading"
    reps: int = 1
    entanglement: str = "linear"
    reuploading_reps: int = 2

    # Quantum simulation
    shots: int = 1024
    random_seed: int = 42
    backend: str = "simulator"

    # Sample budgets
    max_train_samples: int = 500
    max_val_samples: int = 200
    max_test_samples: int = 200

    # SVM
    svm_c: float = 1.0
    svm_class_weight: Optional[str] = "balanced"
    probability_calibration: bool = True

    # Aggregation (None = auto-tune on validation)
    aggregation_strategy: Optional[str] = None
    threshold: Optional[float] = None

    def to_json(self) -> str:
        """Stable JSON serialisation (sorted keys)."""
        d = {
            "feature_dimension": self.feature_dimension,
            "feature_range_low": self.feature_range_low,
            "feature_range_high": self.feature_range_high,
            "use_video_pooling": self.use_video_pooling,
            "feature_map_type": self.feature_map_type,
            "reps": self.reps,
            "entanglement": self.entanglement,
            "reuploading_reps": self.reuploading_reps,
            "shots": self.shots,
            "random_seed": self.random_seed,
            "backend": self.backend,
            "max_train_samples": self.max_train_samples,
            "max_val_samples": self.max_val_samples,
            "max_test_samples": self.max_test_samples,
            "svm_c": self.svm_c,
            "svm_class_weight": self.svm_class_weight,
            "probability_calibration": self.probability_calibration,
            "aggregation_strategy": self.aggregation_strategy,
            "threshold": self.threshold,
        }
        return json.dumps(d, sort_keys=True)

    @property
    def experiment_id(self) -> str:
        """MD5 hash of the stable JSON representation (8-char hex)."""
        return hashlib.md5(self.to_json().encode()).hexdigest()[:8]

    def to_dict(self) -> dict:
        d = json.loads(self.to_json())
        d["experiment_id"] = self.experiment_id
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_json_file(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)


# ── ExperimentResult ──────────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    """All outputs from a completed experiment run."""
    experiment_id: str
    config: ExperimentConfig
    train_val_metrics: Dict[str, Any]
    test_metrics: Optional[Dict[str, Any]]
    kernel_diagnostics: Dict[str, Any]
    timing: Dict[str, float]
    classical_baseline_metrics: Optional[Dict[str, float]]
    best_aggregation_strategy: str
    best_threshold: float
    output_dir: str


# ── Feature map builder ────────────────────────────────────────────────────────

def _build_experiment_feature_map(config: ExperimentConfig):
    """
    Build the feature map circuit for the given config.

    - "zz": uses build_feature_map() from quantum_feature_map (existing)
    - "pauli": uses pauli_feature_map() if available; raises ImportError otherwise
    - "reuploading": builds a shallow re-uploading circuit
    """
    from qiskit.circuit import QuantumCircuit, ParameterVector

    if config.feature_map_type == "zz":
        from quantum.quantum_feature_map import build_feature_map
        cfg = QuantumConfig(
            num_qubits=config.feature_dimension,
            feature_dimension=config.feature_dimension,
            reps=config.reps,
            entanglement=config.entanglement,
        )
        return build_feature_map(cfg)

    elif config.feature_map_type == "pauli":
        try:
            from qiskit.circuit.library import pauli_feature_map
            fm = pauli_feature_map(
                feature_dimension=config.feature_dimension,
                reps=config.reps,
                entanglement=config.entanglement,
            )
            return fm
        except ImportError:
            raise ImportError(
                "pauli_feature_map is not available in your version of Qiskit. "
                "Install qiskit>=2.6 or use feature_map_type='zz' instead."
            )

    elif config.feature_map_type == "reuploading":
        # Build a shallow re-uploading circuit
        # For each rep: Ry(x[i]) for each qubit, then CX(i, i+1) for each pair
        n = config.feature_dimension
        x = ParameterVector("x", length=n)
        qc = QuantumCircuit(n)
        for _ in range(config.reuploading_reps):
            for i in range(n):
                qc.ry(x[i], i)
            # Linear entanglement
            for i in range(n - 1):
                qc.cx(i, i + 1)
        return qc

    else:
        raise ValueError(
            f"Unknown feature_map_type: '{config.feature_map_type}'. "
            "Choose 'zz', 'pauli', or 'reuploading'."
        )


# ── Experiment runner ─────────────────────────────────────────────────────────

class ExperimentRunner:
    """Orchestrate QSVM experiments with reproducible artefact management."""

    SUMMARY_CSV = "summary.csv"
    SUMMARY_COLUMNS = [
        "experiment_id", "model_name", "feature_dim", "reps", "entanglement",
        "svm_c", "frame_accuracy", "frame_f1", "video_accuracy", "video_f1",
        "kta", "use_video_pooling", "feature_map_type", "exploratory",
    ]

    def __init__(
        self,
        features_dir: str,
        results_dir: str = "results/quantum",
        kernel_cache_dir: Optional[str] = None,
    ):
        self.features_dir = Path(features_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        if kernel_cache_dir is not None:
            self.kernel_cache_dir = Path(kernel_cache_dir)
        else:
            self.kernel_cache_dir = self.features_dir / "quantum"
        self.kernel_cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Main run entry point ──────────────────────────────────────────────────

    def run(
        self,
        config: ExperimentConfig,
        evaluate_test: bool = False,
        small_val: bool = False,
        dry_run: bool = False,
    ) -> ExperimentResult:
        """
        Execute a complete experiment from an ExperimentConfig.

        Args:
            config:        Experiment configuration.
            evaluate_test: If True, also compute test metrics.
            small_val:     Cap val to 50 samples; mark outputs "exploratory".
            dry_run:       Validate config + check caches only; no circuits.

        Returns:
            ExperimentResult.
        """
        t_total_start = time.time()
        timing: Dict[str, float] = {}
        exp_id = config.experiment_id
        exploratory = small_val

        exp_dir = self.results_dir / "experiments" / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting experiment {exp_id}: {config.to_json()}")

        # ── Step 1: Set all random seeds ──────────────────────────────────────
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

        # ── Step 2: Load features ─────────────────────────────────────────────
        fd = self.features_dir
        X_train_raw = np.load(str(fd / "train_features.npy"))
        y_train_all = np.load(str(fd / "train_labels.npy"))
        v_train_all = np.load(str(fd / "train_video_ids.npy"), allow_pickle=True)

        X_val_raw = np.load(str(fd / "val_features.npy"))
        y_val_all = np.load(str(fd / "val_labels.npy"))
        v_val_all = np.load(str(fd / "val_video_ids.npy"), allow_pickle=True)

        # Apply subsampling
        if X_train_raw.shape[0] > config.max_train_samples:
            X_train_raw, y_train_all, tr_idx = subsample_balanced(
                X_train_raw, y_train_all, config.max_train_samples, seed=config.random_seed
            )
            v_train_all = v_train_all[tr_idx]

        val_cap = 50 if small_val else config.max_val_samples
        if X_val_raw.shape[0] > val_cap:
            X_val_raw, y_val_all, va_idx = subsample_balanced(
                X_val_raw, y_val_all, val_cap, seed=config.random_seed
            )
            v_val_all = v_val_all[va_idx]

        if evaluate_test:
            X_test_raw = np.load(str(fd / "test_features.npy"))
            y_test_all = np.load(str(fd / "test_labels.npy"))
            v_test_all = np.load(str(fd / "test_video_ids.npy"), allow_pickle=True)
            if X_test_raw.shape[0] > config.max_test_samples:
                X_test_raw, y_test_all, te_idx = subsample_balanced(
                    X_test_raw, y_test_all, config.max_test_samples, seed=config.random_seed
                )
                v_test_all = v_test_all[te_idx]

        # ── Step 4 (dry_run): validate config + check caches ─────────────────
        if dry_run:
            logger.info(f"Dry run: config validated, caches checked for {exp_id}")
            return ExperimentResult(
                experiment_id=exp_id,
                config=config,
                train_val_metrics={"status": "dry_run"},
                test_metrics=None,
                kernel_diagnostics={},
                timing={},
                classical_baseline_metrics=None,
                best_aggregation_strategy="mean",
                best_threshold=0.5,
                output_dir=str(exp_dir),
            )

        # ── Step 5: Fit QuantumFeatureReducer ─────────────────────────────────
        qcfg = QuantumConfig(
            num_qubits=config.feature_dimension,
            feature_dimension=config.feature_dimension,
            reps=config.reps,
            entanglement=config.entanglement,
            shots=config.shots,
            random_seed=config.random_seed,
            feature_range_low=config.feature_range_low,
            feature_range_high=config.feature_range_high,
            svm_c=config.svm_c,
            svm_class_weight=config.svm_class_weight,
            max_train_samples=config.max_train_samples,
            max_val_samples=config.max_val_samples,
            max_test_samples=config.max_test_samples,
        )

        t0 = time.time()
        reducer = QuantumFeatureReducer(config=qcfg)
        X_train_q = reducer.fit_transform(X_train_raw)
        X_val_q = reducer.transform(X_val_raw)
        reducer.save(str(exp_dir))
        timing["feature_reduction_seconds"] = round(time.time() - t0, 3)

        # ── Step 6: Optional video pooling ────────────────────────────────────
        if config.use_video_pooling:
            X_train_q, v_train_pool, y_train_all = pool_frames_per_video_with_labels(
                X_train_q, v_train_all, y_train_all, method="mean"
            )
            v_train_all = v_train_pool
            X_val_q_pool, v_val_pool, y_val_all = pool_frames_per_video_with_labels(
                X_val_q, v_val_all, y_val_all, method="mean"
            )
            X_val_q = X_val_q_pool
            v_val_all = v_val_pool
            logger.info(
                f"Video pooling: {X_train_q.shape[0]} videos (from frames) in train"
            )

        # ── Step 7: Build feature map ─────────────────────────────────────────
        feature_map = _build_experiment_feature_map(config)

        # ── Step 8: Check circuit depth ───────────────────────────────────────
        depth = feature_map.depth()
        circuit_depth_warning = depth > 200
        if circuit_depth_warning:
            logger.warning(
                f"Circuit depth {depth} > 200 — simulation may be slow."
            )

        # ── Step 9: Compute K_train (cache-first) ─────────────────────────────
        backend = SimulatorBackend(shots=config.shots, seed=config.random_seed)

        # Build a kernel computer using the experiment-specific cache dir
        exp_cache_dir = self.kernel_cache_dir / exp_id
        exp_cache_dir.mkdir(parents=True, exist_ok=True)

        computer = QuantumKernelComputer(
            config=qcfg,
            backend=backend,
            cache_dir=str(exp_cache_dir),
        )

        t0 = time.time()
        try:
            K_train = computer.compute_train_kernel(X_train_q, use_cache=True)
        except KernelNumericalError as e:
            _write_error_record(exp_dir, exp_id, config, str(e), exploratory)
            raise

        timing["k_train_seconds"] = round(time.time() - t0, 3)

        # ── Step 10: Classical baseline on identical reduced features ─────────
        classical_metrics: Optional[Dict[str, float]] = None
        try:
            baseline = ClassicalBaselineSVM(
                feature_dimension=config.feature_dimension,
                svm_c=config.svm_c,
                class_weight=config.svm_class_weight,
                random_seed=config.random_seed,
            )
            baseline.fit(X_train_q, y_train_all)
            classical_metrics = baseline.evaluate(X_val_q, y_val_all)
            baseline.save(str(exp_dir))
            logger.info(
                f"Classical baseline: accuracy={classical_metrics['accuracy']:.4f}, "
                f"F1={classical_metrics['f1']:.4f}"
            )
        except Exception as e:
            logger.warning(f"Classical baseline failed: {e}")

        # ── Step 11: Fit QuantumSVM on K_train ───────────────────────────────
        t0 = time.time()
        qsvm = QuantumSVM(config=qcfg)
        qsvm.fit(K_train, y_train_all)
        qsvm.store_train_features(X_train_q)
        timing["svm_fit_seconds"] = round(time.time() - t0, 3)

        # ── Step 12: Compute K_val and run diagnostics ────────────────────────
        t0 = time.time()
        K_val = computer.compute_cross_kernel(
            X_val_q, X_train_q, name="K_val", use_cache=True
        )
        timing["k_val_seconds"] = round(time.time() - t0, 3)

        diag_engine = DiagnosticsEngine()
        try:
            diag_report = diag_engine.analyze(K_train, y_train_all)
        except KernelNumericalError:
            diag_report = None

        # ── Step 13: Tune aggregation strategy + threshold ────────────────────
        val_probs = qsvm.predict_proba(K_val)[:, 1]
        val_preds = qsvm.predict(K_val)

        aggregator = VideoAggregator()
        best_strategy: AggregationStrategy
        best_threshold: float

        if config.aggregation_strategy is not None and config.threshold is not None:
            # Use provided values
            best_strategy = AggregationStrategy(config.aggregation_strategy)
            best_threshold = config.threshold
        else:
            # Auto-tune on val
            # Build video-level ground truth for unique val video IDs
            unique_val_vids = []
            seen = {}
            for vid in v_val_all:
                key = str(vid)
                if key not in seen:
                    seen[key] = len(unique_val_vids)
                    unique_val_vids.append(vid)
            # Majority-vote video labels
            unique_val_vids_arr = np.array(unique_val_vids)
            vid_labels_dict = {}
            for vid, lbl in zip(v_val_all, y_val_all):
                key = str(vid)
                if key not in vid_labels_dict:
                    vid_labels_dict[key] = []
                vid_labels_dict[key].append(int(lbl))
            y_vid_true_dict = {
                vid: int(np.bincount(vid_labels_dict[str(vid)]).argmax())
                for vid in unique_val_vids_arr
            }

            best_strategy, best_threshold = aggregator.tune_threshold_and_strategy(
                val_probs, v_val_all, y_vid_true_dict
            )

        # Compute video-level val metrics using best config
        val_agg = aggregator.aggregate(val_probs, v_val_all, best_strategy, best_threshold)

        # Build per-video val ground truth for final metrics
        unique_result_vids = val_agg.video_ids
        y_vid_val = np.array([
            int(np.bincount(
                [int(y) for y, v in zip(y_val_all, v_val_all) if v == vid]
            ).argmax())
            for vid in unique_result_vids
        ])

        from sklearn.metrics import accuracy_score, f1_score
        video_val_acc = float(accuracy_score(y_vid_val, val_agg.predicted_labels))
        video_val_f1 = float(f1_score(y_vid_val, val_agg.predicted_labels, zero_division=0))

        from sklearn.metrics import precision_score, recall_score
        frame_acc = float(accuracy_score(y_val_all, val_preds))
        frame_f1 = float(f1_score(y_val_all, val_preds, zero_division=0))
        frame_prec = float(precision_score(y_val_all, val_preds, zero_division=0))
        frame_rec = float(recall_score(y_val_all, val_preds, zero_division=0))

        try:
            from sklearn.metrics import roc_auc_score
            frame_auc_proba = float(roc_auc_score(y_val_all, val_probs))
            frame_auc_dec = float(roc_auc_score(
                y_val_all, qsvm.decision_function(K_val)
            ))
        except ValueError:
            frame_auc_proba = float("nan")
            frame_auc_dec = float("nan")

        # ── Step 14: Save train_val_metrics.json and kernel_diagnostics.json ──
        kta_val = diag_report.kta if diag_report else None
        kta_verdict = diag_report.kta_verdict if diag_report else None

        train_val_metrics = {
            "experiment_id": exp_id,
            "exploratory": exploratory,
            "frame_level": {
                "accuracy": round(frame_acc, 6),
                "precision": round(frame_prec, 6),
                "recall": round(frame_rec, 6),
                "f1": round(frame_f1, 6),
                "roc_auc_decision": round(frame_auc_dec, 6) if not math.isnan(frame_auc_dec) else None,
                "roc_auc_proba": round(frame_auc_proba, 6) if not math.isnan(frame_auc_proba) else None,
            },
            "video_level": {
                "best_strategy": best_strategy.value if isinstance(best_strategy, AggregationStrategy) else str(best_strategy),
                "best_threshold": best_threshold,
                "accuracy": round(video_val_acc, 6),
                "f1": round(video_val_f1, 6),
            },
            "kta": round(kta_val, 6) if kta_val is not None else None,
            "kta_verdict": kta_verdict,
        }
        with open(exp_dir / "train_val_metrics.json", "w") as fp:
            json.dump(train_val_metrics, fp, indent=4)

        # Kernel diagnostics
        kernel_diagnostics_dict = diag_report.to_dict() if diag_report else {}
        diag_engine.save_report(diag_report, str(exp_dir / "kernel_diagnostics.json")) if diag_report else None

        # Per-video val predictions
        per_video_val = {
            str(vid): {
                "prob": float(p),
                "predicted": int(pred),
            }
            for vid, p, pred in zip(
                val_agg.video_ids, val_agg.aggregated_probs, val_agg.predicted_labels
            )
        }
        with open(exp_dir / "per_video_val.json", "w") as fp:
            json.dump(per_video_val, fp, indent=4)

        # ── Step 15: Optional test evaluation ────────────────────────────────
        test_metrics: Optional[Dict[str, Any]] = None
        if evaluate_test:
            if config.use_video_pooling:
                X_test_q_raw = reducer.transform(X_test_raw)
                X_test_q_pool, v_test_pool, y_test_all = pool_frames_per_video_with_labels(
                    X_test_q_raw, v_test_all, y_test_all, method="mean"
                )
                X_test_q = X_test_q_pool
                v_test_all = v_test_pool
            else:
                X_test_q = reducer.transform(X_test_raw)

            t0 = time.time()
            K_test = computer.compute_cross_kernel(
                X_test_q, X_train_q, name="K_test", use_cache=True
            )
            timing["k_test_seconds"] = round(time.time() - t0, 3)

            test_probs = qsvm.predict_proba(K_test)[:, 1]
            test_preds = qsvm.predict(K_test)

            test_frame_acc = float(accuracy_score(y_test_all, test_preds))
            test_frame_f1 = float(f1_score(y_test_all, test_preds, zero_division=0))

            test_agg = aggregator.aggregate(
                test_probs, v_test_all, best_strategy, best_threshold
            )
            unique_test_vids = test_agg.video_ids
            y_vid_test = np.array([
                int(np.bincount(
                    [int(y) for y, v in zip(y_test_all, v_test_all) if v == vid]
                ).argmax())
                for vid in unique_test_vids
            ])

            test_video_acc = float(accuracy_score(y_vid_test, test_agg.predicted_labels))
            test_video_f1 = float(f1_score(y_vid_test, test_agg.predicted_labels, zero_division=0))

            test_metrics = {
                "experiment_id": exp_id,
                "exploratory": exploratory,
                "frame_level": {
                    "accuracy": round(test_frame_acc, 6),
                    "f1": round(test_frame_f1, 6),
                },
                "video_level": {
                    "strategy": best_strategy.value if isinstance(best_strategy, AggregationStrategy) else str(best_strategy),
                    "threshold": best_threshold,
                    "accuracy": round(test_video_acc, 6),
                    "f1": round(test_video_f1, 6),
                },
            }
            with open(exp_dir / "test_metrics.json", "w") as fp:
                json.dump(test_metrics, fp, indent=4)

            per_video_test = {
                str(vid): {
                    "prob": float(p),
                    "predicted": int(pred),
                }
                for vid, p, pred in zip(
                    test_agg.video_ids, test_agg.aggregated_probs, test_agg.predicted_labels
                )
            }
            with open(exp_dir / "per_video_test.json", "w") as fp:
                json.dump(per_video_test, fp, indent=4)

        # ── Save config.json ──────────────────────────────────────────────────
        config_dict = config.to_dict()
        config_dict["circuit_depth_warning"] = circuit_depth_warning
        with open(exp_dir / "config.json", "w") as fp:
            json.dump(config_dict, fp, indent=4)

        # ── Save timing.json ──────────────────────────────────────────────────
        timing["total_seconds"] = round(time.time() - t_total_start, 3)
        if "k_test_seconds" not in timing:
            timing["k_test_seconds"] = 0.0
        with open(exp_dir / "timing.json", "w") as fp:
            json.dump(timing, fp, indent=4)

        # ── Save model artefacts ──────────────────────────────────────────────
        qsvm.save(str(exp_dir))
        np.save(str(exp_dir / "train_features_reduced.npy"), X_train_q)

        # ── Save classical_vs_quantum.csv ─────────────────────────────────────
        strategy_str = best_strategy.value if isinstance(best_strategy, AggregationStrategy) else str(best_strategy)
        csv_rows = [
            {
                "experiment_id": exp_id,
                "model_name": "Quantum Kernel SVM",
                "feature_dim": config.feature_dimension,
                "reps": config.reps,
                "entanglement": config.entanglement,
                "svm_c": config.svm_c,
                "frame_accuracy": round(frame_acc, 6),
                "frame_f1": round(frame_f1, 6),
                "video_accuracy": round(video_val_acc, 6),
                "video_f1": round(video_val_f1, 6),
                "kta": round(kta_val, 6) if kta_val is not None else "",
                "exploratory": exploratory,
            }
        ]
        if classical_metrics:
            csv_rows.append({
                "experiment_id": exp_id,
                "model_name": f"Classical RBF-SVM ({config.feature_dimension}-dim)",
                "feature_dim": config.feature_dimension,
                "reps": "—",
                "entanglement": "—",
                "svm_c": config.svm_c,
                "frame_accuracy": round(classical_metrics.get("accuracy", 0), 6),
                "frame_f1": round(classical_metrics.get("f1", 0), 6),
                "video_accuracy": "",
                "video_f1": "",
                "kta": "—",
                "exploratory": exploratory,
            })

        csv_path = exp_dir / "classical_vs_quantum.csv"
        _write_csv(str(csv_path), csv_rows)

        # ── Step 16: Append row to summary.csv ───────────────────────────────
        summary_row = {
            "experiment_id": exp_id,
            "model_name": "Quantum Kernel SVM",
            "feature_dim": config.feature_dimension,
            "reps": config.reps,
            "entanglement": config.entanglement,
            "svm_c": config.svm_c,
            "frame_accuracy": round(frame_acc, 6),
            "frame_f1": round(frame_f1, 6),
            "video_accuracy": round(video_val_acc, 6),
            "video_f1": round(video_val_f1, 6),
            "kta": round(kta_val, 6) if kta_val is not None else "",
            "use_video_pooling": config.use_video_pooling,
            "feature_map_type": config.feature_map_type,
            "exploratory": exploratory,
        }
        self._append_summary_row(summary_row)

        logger.info(
            f"Experiment {exp_id} complete in {timing['total_seconds']:.1f}s. "
            f"Frame acc={frame_acc:.4f}, Video acc={video_val_acc:.4f}"
        )

        return ExperimentResult(
            experiment_id=exp_id,
            config=config,
            train_val_metrics=train_val_metrics,
            test_metrics=test_metrics,
            kernel_diagnostics=kernel_diagnostics_dict,
            timing=timing,
            classical_baseline_metrics=classical_metrics,
            best_aggregation_strategy=strategy_str,
            best_threshold=best_threshold,
            output_dir=str(exp_dir),
        )

    # ── Summary helpers ───────────────────────────────────────────────────────

    def _append_summary_row(self, row: dict) -> None:
        """Append a row to the summary CSV, creating it if needed."""
        exp_dir = self.results_dir / "experiments"
        exp_dir.mkdir(parents=True, exist_ok=True)
        csv_path = exp_dir / self.SUMMARY_CSV

        write_header = not csv_path.exists()
        with open(str(csv_path), "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.SUMMARY_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def summarize(self):
        """Return summary.csv as a DataFrame (requires pandas) or list of dicts."""
        csv_path = self.results_dir / "experiments" / self.SUMMARY_CSV
        if not csv_path.exists():
            return [] if not _PANDAS_AVAILABLE else None

        if _PANDAS_AVAILABLE:
            return pd.read_csv(str(csv_path))
        else:
            rows = []
            with open(str(csv_path)) as f:
                for r in csv.DictReader(f):
                    rows.append(r)
            return rows

    def load_from_json(self, config_path: str) -> ExperimentResult:
        """Load ExperimentConfig from JSON file and run."""
        config = ExperimentConfig.from_json_file(config_path)
        return self.run(config)


# ── Helper utilities ──────────────────────────────────────────────────────────

def _write_csv(path: str, rows: List[dict]) -> None:
    """Write a list of dicts to a CSV file."""
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _write_error_record(
    exp_dir: Path,
    exp_id: str,
    config: ExperimentConfig,
    error_msg: str,
    exploratory: bool,
) -> None:
    """Write a failure record to train_val_metrics.json."""
    record = {
        "experiment_id": exp_id,
        "status": "failed",
        "error": error_msg,
        "exploratory": exploratory,
    }
    with open(exp_dir / "train_val_metrics.json", "w") as f:
        json.dump(record, f, indent=4)
    config_dict = config.to_dict()
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4)



# ── Named experiment presets ──────────────────────────────────────────────────

NAMED_EXPERIMENTS = {
    "exp-baseline": ExperimentConfig(
        feature_dimension=4, reps=1, entanglement="linear", svm_c=1.0,
        use_video_pooling=False, max_train_samples=500,
    ),
    "exp-8dim": ExperimentConfig(
        feature_dimension=8, reps=1, entanglement="linear", svm_c=1.0,
        use_video_pooling=False, max_train_samples=500,
    ),
    "exp-8dim-reps2": ExperimentConfig(
        feature_dimension=8, reps=2, entanglement="linear", svm_c=10.0,
        use_video_pooling=False, max_train_samples=500,
    ),
    "exp-8dim-full": ExperimentConfig(
        feature_dimension=8, reps=1, entanglement="full", svm_c=1.0,
        use_video_pooling=False, max_train_samples=500,
    ),
    "exp-video-pool": ExperimentConfig(
        feature_dimension=8, reps=1, entanglement="linear", svm_c=1.0,
        use_video_pooling=True, max_train_samples=500,
    ),
    "exp-best": ExperimentConfig(
        feature_dimension=8, reps=1, entanglement="linear", svm_c=10.0,
        svm_class_weight="balanced", use_video_pooling=True, max_train_samples=500,
    ),
}

# ── CLI ────────────────────────────────────────────────────────────────────────

def _default_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Run a QSVM experiment with the given configuration."
    )
    root = _default_root()

    parser.add_argument("--features-dir", default=str(root / "features"),
                        help="Directory with {split}_features/labels/video_ids.npy.")
    parser.add_argument("--results-dir", default=str(root / "results" / "quantum"),
                        help="Output directory for experiment artefacts.")
    parser.add_argument("--kernel-cache-dir", default=None,
                        help="Override kernel cache directory.")
    parser.add_argument("--config-json", default=None,
                        help="Path to an ExperimentConfig JSON file.")
    parser.add_argument("--feature-dim", type=int, default=4, choices=[4, 6, 8],
                        help="Feature dimension (default: 4).")
    parser.add_argument("--reps", type=int, default=1,
                        help="ZZFeatureMap reps (default: 1).")
    parser.add_argument("--entanglement", default="linear",
                        choices=["linear", "full", "circular"],
                        help="Entanglement pattern (default: linear).")
    parser.add_argument("--svm-c", type=float, default=1.0,
                        help="SVM regularisation C (default: 1.0).")
    parser.add_argument("--feature-map", default="zz",
                        choices=["zz", "pauli", "reuploading"],
                        help="Feature map type (default: zz).")
    parser.add_argument("--use-video-pooling", action="store_true",
                        help="Pool frames per video before QSVM.")
    parser.add_argument("--max-train", type=int, default=500,
                        help="Max training samples (default: 500).")
    parser.add_argument("--shots", type=int, default=1024,
                        help="Shots per circuit (default: 1024).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Also evaluate on the test split.")
    parser.add_argument("--small-val", action="store_true",
                        help="Cap val to 50 samples (exploratory mode).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config only; no kernel computation.")
    parser.add_argument("--experiment", default=None,
                        choices=list(NAMED_EXPERIMENTS.keys()),
                        help="Use a named experiment preset. "
                             "Overrides individual --feature-dim etc. flags.")
    parser.add_argument("--summarize", action="store_true",
                        help="Print the summary CSV of all completed experiments and exit.")

    args = parser.parse_args()

    # --summarize: print summary CSV and exit
    if args.summarize:
        runner_tmp = ExperimentRunner(
            features_dir=args.features_dir,
            results_dir=args.results_dir,
        )
        rows = runner_tmp.summarize()
        if not rows:
            print("  No experiments found in summary CSV.")
        else:
            if hasattr(rows, "to_string"):   # pandas DataFrame
                print(rows.to_string(index=False))
            else:                            # list of dicts
                import csv as _csv, sys as _sys
                if rows:
                    writer = _csv.DictWriter(_sys.stdout, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
        sys.exit(0)

    if args.experiment:
        config = NAMED_EXPERIMENTS[args.experiment]
        # Allow overrides from other CLI flags when combined with --experiment
        if args.reps != 1:
            config = ExperimentConfig(**{**vars(config), "reps": args.reps})
        if args.svm_c != 1.0:
            config = ExperimentConfig(**{**vars(config), "svm_c": args.svm_c})
        if args.use_video_pooling:
            config = ExperimentConfig(**{**vars(config), "use_video_pooling": True})
        if args.max_train != 500:
            config = ExperimentConfig(**{**vars(config), "max_train_samples": args.max_train})
        if args.shots != 1024:
            config = ExperimentConfig(**{**vars(config), "shots": args.shots})
        if args.seed != 42:
            config = ExperimentConfig(**{**vars(config), "random_seed": args.seed})
    elif args.config_json:
        config = ExperimentConfig.from_json_file(args.config_json)
    else:
        config = ExperimentConfig(
            feature_dimension=args.feature_dim,
            feature_map_type=args.feature_map,
            reps=args.reps,
            entanglement=args.entanglement,
            svm_c=args.svm_c,
            use_video_pooling=args.use_video_pooling,
            max_train_samples=args.max_train,
            shots=args.shots,
            random_seed=args.seed,
        )

    print()
    print("=" * 65)
    print("  TrustLens-Q — Experiment Runner")
    print("=" * 65)
    print(f"  Experiment ID   : {config.experiment_id}")
    print(f"  Feature dim     : {config.feature_dimension}")
    print(f"  Feature map     : {config.feature_map_type}")
    print(f"  Reps            : {config.reps}")
    print(f"  Entanglement    : {config.entanglement}")
    print(f"  SVM C           : {config.svm_c}")
    print(f"  Video pooling   : {config.use_video_pooling}")
    print(f"  Max train       : {config.max_train_samples}")
    print(f"  Random seed     : {config.random_seed}")
    print()

    runner = ExperimentRunner(
        features_dir=args.features_dir,
        results_dir=args.results_dir,
        kernel_cache_dir=args.kernel_cache_dir,
    )

    result = runner.run(
        config=config,
        evaluate_test=args.evaluate_test,
        small_val=args.small_val,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("  Dry run complete — no kernel computation performed.")
    else:
        print(f"  Results saved to : {result.output_dir}")
        tv = result.train_val_metrics
        if "frame_level" in tv:
            print(f"  Frame accuracy   : {tv['frame_level']['accuracy']:.4f}")
            print(f"  Video accuracy   : {tv['video_level']['accuracy']:.4f}")
            print(f"  KTA              : {tv.get('kta', 'N/A')}")

    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
