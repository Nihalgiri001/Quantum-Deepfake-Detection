"""
Quantum Kernel SVM
====================
Core classifier and end-to-end training pipeline for the TrustLens-Q
Quantum Kernel Support Vector Machine.

The classifier uses a classical SVC with ``kernel="precomputed"``, fed a
quantum fidelity kernel matrix produced by ``QuantumKernelComputer``.
The full pipeline is:

    ResNet-18 (512d)
        → QuantumFeatureReducer  (512 → 4/6/8d, fitted on train only)
        → QuantumKernelComputer  (K_train (n×n), K_val (m×n), K_test (p×n))
        → SVC(kernel="precomputed")
        → frame-level predictions → video-level aggregation

Design decisions:
- ``QuantumSVM`` extends ``BaseClassifier`` from ``src/models/base_model.py``
  for interface compatibility with the existing benchmark harness.
- ``fit(K_train, y_train)`` accepts a pre-computed kernel matrix, not raw
  features.  This deviates from the ``BaseClassifier`` contract (which
  passes raw feature arrays) — the deviation is documented here and in the
  class docstring.
- ``SVC(probability=True)`` is used rather than ``CalibratedClassifierCV``
  because ``CalibratedClassifierCV`` does not expose ``decision_function``,
  which is needed for consistent scoring.  A ``FutureWarning`` is suppressed
  (sklearn 1.9.1; ``probability=True`` is not removed until 1.11).
- All artefacts needed for inference are saved together so that a single
  ``QuantumSVM.load(models_dir)`` call reconstructs the full pipeline.

CLI:
    python -m src.quantum.quantum_svm --smoke-test
    python -m src.quantum.quantum_svm --train --backend simulator
    python -m src.quantum.quantum_svm --train --num-qubits 6
    python -m src.quantum.quantum_svm --train --max-samples 100
"""

import sys
import json
import time
import warnings
import argparse
import logging
import numpy as np
import joblib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantum.quantum_config import QuantumConfig
from quantum.feature_reducer import QuantumFeatureReducer
from quantum.quantum_kernel import QuantumKernelComputer
from quantum.simulator_backend import SimulatorBackend
from quantum.kernel_utils import subsample_balanced

try:
    from models.base_model import BaseClassifier
except ImportError:
    # Fallback when running from within src/quantum/
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models.base_model import BaseClassifier

try:
    from utils import setup_logger
    logger = setup_logger("quantum.svm")
except ImportError:
    logger = logging.getLogger("quantum.svm")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO,
                            format="[%(asctime)s] %(levelname)-8s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")


# ── QuantumSVM classifier ─────────────────────────────────────────────────────

class QuantumSVM(BaseClassifier):
    """SVC with a precomputed quantum fidelity kernel.

    IMPORTANT — interface deviation from BaseClassifier:
        ``fit(K_train, y_train)`` expects a pre-computed (n×n) kernel
        matrix as its first argument, NOT raw feature vectors.
        Similarly, ``predict``, ``predict_proba``, and
        ``decision_function`` all expect cross-kernel matrices of shape
        (m, n_train), not feature arrays.

        This deviation is intentional and is required by
        ``SVC(kernel="precomputed")``.

    Attributes:
        name:          "quantum_svm"
        display_name:  "Quantum Kernel SVM"
        config:        QuantumConfig governing circuit and training params.
        model:         Fitted SVC instance (None until fit() is called).
        _train_features_reduced: Reduced training features stored for
                                 cross-kernel computation at inference time.
    """

    name: str = "quantum_svm"
    display_name: str = "Quantum Kernel SVM"

    def __init__(
        self,
        config: Optional[QuantumConfig] = None,
        random_seed: int = 42,
        **kwargs,
    ):
        super().__init__(random_seed=random_seed)
        self.config = config or QuantumConfig()
        self.model: Optional[SVC] = None
        self._train_features_reduced: Optional[np.ndarray] = None
        self._training_labels: Optional[np.ndarray] = None

    # ── Core interface ────────────────────────────────────────────────────────

    def fit(self, K_train: np.ndarray, y_train: np.ndarray) -> "QuantumSVM":
        """Fit the SVC on the precomputed training kernel matrix.

        Args:
            K_train: Square kernel matrix, shape (n_train, n_train).
                     MUST be the quantum fidelity kernel; passing raw
                     features will produce incorrect results.
            y_train: Integer label array, shape (n_train,).
                     0 = REAL, 1 = FAKE.

        Returns:
            self (for chaining).
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.model = SVC(
                kernel="precomputed",
                probability=True,
                C=self.config.svm_c,
                class_weight=self.config.svm_class_weight,
                random_state=self.random_seed,
            )
            self.model.fit(K_train, y_train)
        self._training_labels = y_train.copy()
        self.is_fitted = True
        logger.info(
            f"QuantumSVM fitted: {K_train.shape[0]} samples, "
            f"C={self.config.svm_c}, "
            f"support vectors={self.model.n_support_.sum()}"
        )
        return self

    def predict(self, K_test: np.ndarray) -> np.ndarray:
        """Predict class labels from a precomputed cross-kernel matrix.

        Args:
            K_test: Cross-kernel matrix, shape (m, n_train).

        Returns:
            Label array of shape (m,) with values 0 (REAL) or 1 (FAKE).
        """
        self._check_fitted()
        return self.model.predict(K_test)

    def predict_proba(self, K_test: np.ndarray) -> np.ndarray:
        """Predict class probabilities from a precomputed cross-kernel.

        Args:
            K_test: Cross-kernel matrix, shape (m, n_train).

        Returns:
            Probability matrix of shape (m, 2): [:, 0] = P(REAL),
            [:, 1] = P(FAKE).
        """
        self._check_fitted()
        return self.model.predict_proba(K_test)

    def decision_function(self, K_test: np.ndarray) -> np.ndarray:
        """Signed distance to the decision boundary.

        Args:
            K_test: Cross-kernel matrix, shape (m, n_train).

        Returns:
            1-D array of shape (m,): positive = FAKE, negative = REAL.
        """
        self._check_fitted()
        return self.model.decision_function(K_test)

    def get_support_vector_features(self) -> Optional[np.ndarray]:
        """Return the reduced training features for support vectors only.

        Needed to compute a minimal cross-kernel at inference time when
        the full training set is too large.

        Returns:
            Array of shape (n_sv, feature_dimension), or None if training
            features have not been stored via store_train_features().
        """
        if self._train_features_reduced is None or not self.is_fitted:
            return None
        sv_idx = self.model.support_
        return self._train_features_reduced[sv_idx]

    def store_train_features(self, X_train_reduced: np.ndarray) -> None:
        """Store the reduced training feature vectors.

        Called by ``train_full_pipeline`` before ``save`` so that
        cross-kernel computation at inference time has access to the
        original training points.

        Args:
            X_train_reduced: Quantum-ready features, shape (n_train, d).
        """
        self._train_features_reduced = X_train_reduced.copy()

    def _check_fitted(self) -> None:
        if not self.is_fitted or self.model is None:
            raise RuntimeError(
                "QuantumSVM must be fitted before inference. "
                "Call fit(K_train, y_train) first."
            )

    # ── Artefact persistence ──────────────────────────────────────────────────

    def save(self, models_dir: str) -> None:
        """Save all artefacts required for inference.

        Creates:
            <models_dir>/quantum_svm.pkl           — fitted SVC
            <models_dir>/train_features_reduced.npy — reduced train features
            <models_dir>/quantum_config.json        — QuantumConfig snapshot
            <models_dir>/training_metadata.json     — training run record

        Args:
            models_dir: Directory to write artefacts into.
        """
        self._check_fitted()
        d = Path(models_dir)
        d.mkdir(parents=True, exist_ok=True)

        # Save the SVC model
        joblib.dump(self.model, d / "quantum_svm.pkl")

        # Save reduced training features for cross-kernel at inference time
        if self._train_features_reduced is not None:
            np.save(str(d / "train_features_reduced.npy"),
                    self._train_features_reduced)

        # Save config snapshot
        self.config.save_json(str(d / "quantum_config.json"))

        # Save training metadata
        meta = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "classifier": self.display_name,
            "num_qubits": self.config.num_qubits,
            "feature_dimension": self.config.feature_dimension,
            "reps": self.config.reps,
            "entanglement": self.config.entanglement,
            "svm_c": self.config.svm_c,
            "svm_class_weight": self.config.svm_class_weight,
            "random_seed": self.random_seed,
            "n_train_samples": (
                int(self._train_features_reduced.shape[0])
                if self._train_features_reduced is not None else None
            ),
            "n_support_vectors": int(self.model.n_support_.sum()),
            "classes": self.model.classes_.tolist(),
        }
        with open(d / "training_metadata.json", "w") as f:
            json.dump(meta, f, indent=4)

        logger.info(f"QuantumSVM artefacts saved to: {d}")

    @staticmethod
    def load(models_dir: str) -> "QuantumSVM":
        """Load a previously saved QuantumSVM from artefact directory.

        Args:
            models_dir: Directory containing quantum_svm.pkl and friends.

        Returns:
            Fitted QuantumSVM ready for inference.

        Raises:
            FileNotFoundError: If quantum_svm.pkl is missing.
        """
        d = Path(models_dir)
        svc_path = d / "quantum_svm.pkl"
        if not svc_path.exists():
            raise FileNotFoundError(
                f"quantum_svm.pkl not found in {d}. "
                "Run '--train' first to generate the model."
            )

        # Load config
        cfg_path = d / "quantum_config.json"
        config = QuantumConfig.load_json(str(cfg_path)) if cfg_path.exists() else QuantumConfig()

        qsvm = QuantumSVM(config=config)
        qsvm.model = joblib.load(svc_path)
        qsvm.is_fitted = True

        # Load reduced training features (used for cross-kernel at inference)
        feat_path = d / "train_features_reduced.npy"
        if feat_path.exists():
            qsvm._train_features_reduced = np.load(str(feat_path))

        logger.info(f"QuantumSVM loaded from: {d}")
        return qsvm


# ── Full training pipeline ────────────────────────────────────────────────────

def train_full_pipeline(
    config: QuantumConfig,
    features_dir: str,
    models_dir: str,
    backend_type: str = "simulator",
    kernel_cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """End-to-end QSVM training pipeline.

    Steps:
        1.  Load ResNet-18 features for all three splits.
        2.  Subsample training set (stratified, ≤ config.max_train_samples).
        3.  Fit QuantumFeatureReducer on training data only.
        4.  Transform all three splits.
        5.  Compute K_train.
        6.  Fit QuantumSVM.
        7.  Compute K_val, K_test.
        8.  Evaluate on val (accuracy + F1) and print summary.
        9.  Save all artefacts.

    Args:
        config:           QuantumConfig.
        features_dir:     Directory with {split}_features/labels/video_ids.npy.
        models_dir:       Output directory for model artefacts.
        backend_type:     "simulator" (only option currently).
        kernel_cache_dir: Override cache directory for kernel matrices.

    Returns:
        Dictionary of results including metrics, timing, and paths.
    """
    fd = Path(features_dir)
    md = Path(models_dir)
    md.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "config": config.to_dict(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    # ── 1. Load features ──────────────────────────────────────────────────────
    logger.info("Loading ResNet-18 features…")
    X_train_raw = np.load(fd / "train_features.npy")
    y_train_all = np.load(fd / "train_labels.npy")
    v_train_all = np.load(fd / "train_video_ids.npy", allow_pickle=True)

    X_val_raw   = np.load(fd / "val_features.npy")
    y_val_all   = np.load(fd / "val_labels.npy")
    v_val_all   = np.load(fd / "val_video_ids.npy", allow_pickle=True)

    X_test_raw  = np.load(fd / "test_features.npy")
    y_test_all  = np.load(fd / "test_labels.npy")
    v_test_all  = np.load(fd / "test_video_ids.npy", allow_pickle=True)

    logger.info(
        f"Raw sizes — train: {X_train_raw.shape[0]}, "
        f"val: {X_val_raw.shape[0]}, test: {X_test_raw.shape[0]}"
    )

    # ── 2. Subsample training set ─────────────────────────────────────────────
    n_train_cap = config.max_train_samples
    if X_train_raw.shape[0] > n_train_cap:
        logger.info(
            f"Subsampling training set: {X_train_raw.shape[0]} → {n_train_cap} "
            f"(stratified, seed={config.random_seed})"
        )
        X_train_raw, y_train_all, tr_idx = subsample_balanced(
            X_train_raw, y_train_all, n_train_cap, seed=config.random_seed
        )
        v_train_all = v_train_all[tr_idx]

    n_val_cap = config.max_val_samples
    if X_val_raw.shape[0] > n_val_cap:
        X_val_raw, y_val_all, va_idx = subsample_balanced(
            X_val_raw, y_val_all, n_val_cap, seed=config.random_seed
        )
        v_val_all = v_val_all[va_idx]

    n_test_cap = config.max_test_samples
    if X_test_raw.shape[0] > n_test_cap:
        X_test_raw, y_test_all, te_idx = subsample_balanced(
            X_test_raw, y_test_all, n_test_cap, seed=config.random_seed
        )
        v_test_all = v_test_all[te_idx]

    logger.info(
        f"Working sizes — train: {X_train_raw.shape[0]}, "
        f"val: {X_val_raw.shape[0]}, test: {X_test_raw.shape[0]}"
    )
    results["n_train"] = int(X_train_raw.shape[0])
    results["n_val"]   = int(X_val_raw.shape[0])
    results["n_test"]  = int(X_test_raw.shape[0])

    # ── 3. Fit QuantumFeatureReducer on training data only ───────────────────
    logger.info("Fitting QuantumFeatureReducer…")
    reducer = QuantumFeatureReducer(config=config)
    t0 = time.time()
    X_train_q = reducer.fit_transform(X_train_raw)
    X_val_q   = reducer.transform(X_val_raw)
    X_test_q  = reducer.transform(X_test_raw)
    reducer_time = time.time() - t0
    reducer.save(str(md))

    logger.info(
        f"Feature reduction: {X_train_raw.shape[1]}d → {X_train_q.shape[1]}d "
        f"in {reducer_time:.1f}s"
    )
    results["feature_reduction_time_sec"] = round(reducer_time, 2)

    # ── 4. Build simulator backend ────────────────────────────────────────────
    if backend_type != "simulator":
        raise ValueError(
            f"backend_type='{backend_type}' is not supported by "
            "train_full_pipeline.  Use 'simulator' for local training; "
            "use ibm_hardware.py for hardware experiments."
        )
    backend = SimulatorBackend(shots=config.shots, seed=config.random_seed)

    # Resolve kernel cache directory
    if kernel_cache_dir is None:
        _root = Path(__file__).resolve().parent.parent.parent
        kernel_cache_dir = str(_root / "features" / "quantum")

    computer = QuantumKernelComputer(
        config=config,
        backend=backend,
        cache_dir=kernel_cache_dir,
    )

    # ── 5. Compute K_train ────────────────────────────────────────────────────
    logger.info("Computing training kernel…")
    t0 = time.time()
    K_train = computer.compute_train_kernel(X_train_q, use_cache=True)
    k_train_time = time.time() - t0
    logger.info(f"K_train computed in {k_train_time:.1f}s")
    results["k_train_time_sec"] = round(k_train_time, 2)

    # ── 6. Fit QuantumSVM ─────────────────────────────────────────────────────
    logger.info("Fitting QuantumSVM…")
    qsvm = QuantumSVM(config=config)
    t0 = time.time()
    qsvm.fit(K_train, y_train_all)
    svm_fit_time = time.time() - t0
    qsvm.store_train_features(X_train_q)
    logger.info(f"SVM fit in {svm_fit_time:.1f}s")
    results["svm_fit_time_sec"] = round(svm_fit_time, 2)

    # ── 7. Compute K_val, K_test ──────────────────────────────────────────────
    logger.info("Computing validation kernel…")
    t0 = time.time()
    K_val = computer.compute_cross_kernel(
        X_val_q, X_train_q, name="K_val", use_cache=True
    )
    k_val_time = time.time() - t0

    logger.info("Computing test kernel…")
    t0 = time.time()
    K_test = computer.compute_cross_kernel(
        X_test_q, X_train_q, name="K_test", use_cache=True
    )
    k_test_time = time.time() - t0
    results["k_val_time_sec"]  = round(k_val_time, 2)
    results["k_test_time_sec"] = round(k_test_time, 2)

    # ── 8. Evaluate on validation set ─────────────────────────────────────────
    y_val_pred = qsvm.predict(K_val)
    y_val_prob = qsvm.predict_proba(K_val)[:, 1]
    val_acc = float(accuracy_score(y_val_all, y_val_pred))
    val_f1  = float(f1_score(y_val_all, y_val_pred, zero_division=0))

    y_test_pred = qsvm.predict(K_test)
    y_test_prob = qsvm.predict_proba(K_test)[:, 1]
    test_acc = float(accuracy_score(y_test_all, y_test_pred))
    test_f1  = float(f1_score(y_test_all, y_test_pred, zero_division=0))

    results["val_accuracy"]  = round(val_acc, 4)
    results["val_f1"]        = round(val_f1, 4)
    results["test_accuracy"] = round(test_acc, 4)
    results["test_f1"]       = round(test_f1, 4)

    # Print validation summary
    print()
    print("─" * 60)
    print("  QSVM Validation Results")
    print("─" * 60)
    print(f"  Train samples : {X_train_raw.shape[0]}")
    print(f"  Val samples   : {X_val_raw.shape[0]}")
    print(f"  Val accuracy  : {val_acc * 100:.2f}%")
    print(f"  Val F1        : {val_f1:.4f}")
    print()
    print(classification_report(y_val_all, y_val_pred,
                                target_names=["REAL", "FAKE"], digits=4))

    # ── 9. Save all artefacts ─────────────────────────────────────────────────
    qsvm.save(str(md))

    # Save video IDs alongside kernel matrices for downstream evaluation
    np.save(str(Path(kernel_cache_dir) / "train_video_ids.npy"), v_train_all)
    np.save(str(Path(kernel_cache_dir) / "val_video_ids.npy"),   v_val_all)
    np.save(str(Path(kernel_cache_dir) / "test_video_ids.npy"),  v_test_all)
    np.save(str(Path(kernel_cache_dir) / "train_labels.npy"),    y_train_all)
    np.save(str(Path(kernel_cache_dir) / "val_labels.npy"),      y_val_all)
    np.save(str(Path(kernel_cache_dir) / "test_labels.npy"),     y_test_all)

    results["finished_at"] = datetime.now(timezone.utc).isoformat()
    results["models_dir"]  = str(md)

    with open(md / "training_metadata.json", "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Training complete. Artefacts saved to: {md}")
    return results


# ── Smoke test ────────────────────────────────────────────────────────────────

def _run_smoke_test():
    """End-to-end pipeline on synthetic 512-dim data. No real features needed."""
    import tempfile

    print()
    print("=" * 60)
    print("  QuantumSVM — Smoke Test (synthetic data)")
    print("=" * 60)

    rng = np.random.default_rng(0)

    # Synthetic 512-dim data (mimics ResNet-18 output)
    n_train, n_val, n_test = 30, 10, 10
    raw_dim = 512
    X_train_raw = rng.standard_normal((n_train, raw_dim)).astype(np.float32)
    y_train     = np.array([0]*15 + [1]*15)
    X_val_raw   = rng.standard_normal((n_val, raw_dim)).astype(np.float32)
    y_val       = np.array([0]*5 + [1]*5)
    X_test_raw  = rng.standard_normal((n_test, raw_dim)).astype(np.float32)
    y_test      = np.array([0]*5 + [1]*5)

    cfg = QuantumConfig(
        num_qubits=4, feature_dimension=4,
        shots=1024, random_seed=42,
        max_train_samples=30, max_val_samples=10, max_test_samples=10,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        features_dir = tmp_path / "features"
        models_dir   = tmp_path / "models" / "quantum"
        kernel_dir   = tmp_path / "features" / "quantum"
        features_dir.mkdir(parents=True)

        np.save(str(features_dir / "train_features.npy"),  X_train_raw)
        np.save(str(features_dir / "train_labels.npy"),    y_train)
        np.save(str(features_dir / "train_video_ids.npy"), np.array([f"v{i}" for i in range(n_train)]))
        np.save(str(features_dir / "val_features.npy"),    X_val_raw)
        np.save(str(features_dir / "val_labels.npy"),      y_val)
        np.save(str(features_dir / "val_video_ids.npy"),   np.array([f"v{i}" for i in range(n_val)]))
        np.save(str(features_dir / "test_features.npy"),   X_test_raw)
        np.save(str(features_dir / "test_labels.npy"),     y_test)
        np.save(str(features_dir / "test_video_ids.npy"),  np.array([f"v{i}" for i in range(n_test)]))

        t_start = time.time()
        results = train_full_pipeline(
            config=cfg,
            features_dir=str(features_dir),
            models_dir=str(models_dir),
            backend_type="simulator",
            kernel_cache_dir=str(kernel_dir),
        )
        elapsed = time.time() - t_start

        # ── Acceptance checks ──────────────────────────────────────────────
        print()
        print("─" * 60)
        print("  Acceptance Checks")
        print("─" * 60)

        # 1. Fit on 30×30 kernel → predict on 10×30 shapes
        qsvm = QuantumSVM.load(str(models_dir))
        reducer = QuantumFeatureReducer.load(str(models_dir))
        X_test_q  = reducer.transform(X_test_raw)
        X_train_q = qsvm._train_features_reduced
        backend   = SimulatorBackend(shots=1024, seed=42)
        computer  = QuantumKernelComputer(config=cfg, backend=backend,
                                          cache_dir=str(kernel_dir))
        K_test_reload = computer.compute_cross_kernel(
            X_test_q, X_train_q, name="K_test_smoke", use_cache=False
        )
        preds  = qsvm.predict(K_test_reload)
        probas = qsvm.predict_proba(K_test_reload)

        assert preds.shape == (n_test,), f"predict shape {preds.shape}"
        print(f"  ✅  predict shape ({n_test},)")

        assert probas.shape == (n_test, 2), f"predict_proba shape {probas.shape}"
        print(f"  ✅  predict_proba shape ({n_test}, 2)")

        # 2. Probabilities in [0,1] and sum to 1
        assert np.all((probas >= 0) & (probas <= 1)), "proba out of [0,1]"
        assert np.allclose(probas.sum(axis=1), 1.0), "proba rows don't sum to 1"
        print(f"  ✅  predict_proba values in [0,1], sums to 1")

        # 3. Save + reload → identical predictions
        preds_original = qsvm.predict(K_test_reload)
        qsvm2 = QuantumSVM.load(str(models_dir))
        preds_reloaded = qsvm2.predict(K_test_reload)
        assert np.array_equal(preds_original, preds_reloaded), "save/load mismatch"
        print(f"  ✅  Save + reload → identical predictions")

        # 4. Class labels: 0=REAL, 1=FAKE
        assert set(qsvm.model.classes_.tolist()) == {0, 1}, "wrong class labels"
        print(f"  ✅  Classes: 0=REAL, 1=FAKE")

        # 5. Smoke test timing
        print(f"  ✅  Smoke test completed in {elapsed:.1f}s (limit: 120s)")
        assert elapsed < 120, f"Smoke test too slow: {elapsed:.1f}s"

        print()
        print("  ✅  All smoke test acceptance criteria PASSED")
        print(f"  Val accuracy: {results['val_accuracy'] * 100:.1f}%")
        print("=" * 60)
        print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train or smoke-test the Quantum Kernel SVM pipeline."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--smoke-test", action="store_true",
        help="Run the full pipeline on 30 synthetic 512-dim samples."
    )
    mode.add_argument(
        "--train", action="store_true",
        help="Train on real extracted ResNet-18 features."
    )

    parser.add_argument(
        "--backend", type=str, default="simulator",
        choices=["simulator"],
        help="Execution backend (default: simulator)."
    )
    parser.add_argument(
        "--num-qubits", type=int, default=4, choices=[4, 6, 8],
        help="Number of qubits / quantum feature dimension (default: 4)."
    )
    parser.add_argument(
        "--reps", type=int, default=1,
        help="ZZFeatureMap circuit repetitions (default: 1)."
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Override max_train_samples in QuantumConfig."
    )
    parser.add_argument(
        "--shots", type=int, default=1024,
        help="Shots per circuit execution (default: 1024)."
    )
    parser.add_argument(
        "--features-dir", type=str, default=None,
        help="Override features directory."
    )
    parser.add_argument(
        "--models-dir", type=str, default=None,
        help="Override models output directory."
    )
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
        return

    # ── Full training on real data ─────────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import Config
    classical_cfg = Config()

    cfg = QuantumConfig(
        num_qubits=args.num_qubits,
        feature_dimension=args.num_qubits,
        reps=args.reps,
        shots=args.shots,
        max_train_samples=args.max_samples or 500,
        max_val_samples=200,
        max_test_samples=200,
    )

    features_dir = args.features_dir or str(classical_cfg.features_dir)
    models_dir   = args.models_dir   or str(classical_cfg.models_dir / "quantum")

    print()
    print("=" * 60)
    print("  QuantumSVM — Full Training")
    print("=" * 60)
    print(cfg)
    print()

    results = train_full_pipeline(
        config=cfg,
        features_dir=features_dir,
        models_dir=models_dir,
        backend_type=args.backend,
    )

    print()
    print("─" * 60)
    print("  Training Summary")
    print("─" * 60)
    print(f"  Train samples     : {results['n_train']}")
    print(f"  Val accuracy      : {results['val_accuracy'] * 100:.2f}%")
    print(f"  Val F1            : {results['val_f1']:.4f}")
    print(f"  K_train time      : {results['k_train_time_sec']:.1f}s")
    print(f"  SVM fit time      : {results['svm_fit_time_sec']:.2f}s")
    print(f"  Models saved to   : {results['models_dir']}")
    print("─" * 60)
    print()


if __name__ == "__main__":
    main()
