"""
Classical Baseline SVM
========================
Train a classical RBF-SVM on the same low-dimensional features used by
the QSVM, enabling a fair kernel-type comparison.

Uses the same QuantumFeatureReducer artefacts as the QSVM so input
representations are byte-for-byte identical. The only difference is the
kernel: classical RBF vs. quantum fidelity.

Usage:
    from quantum.classical_baseline import ClassicalBaselineSVM
    clf = ClassicalBaselineSVM(feature_dimension=8)
    clf.fit(X_train, y_train)
    metrics = clf.evaluate(X_val, y_val)
"""

import sys
import json
import logging
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Optional

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from utils import setup_logger
    logger = setup_logger("quantum.classical_baseline")
except ImportError:
    logger = logging.getLogger("quantum.classical_baseline")
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


class ClassicalBaselineSVM:
    """RBF-SVM on identical low-dimensional features as QSVM.

    Uses the same QuantumFeatureReducer artefacts as the QSVM so input
    representations are byte-for-byte identical. The only difference
    is the kernel: classical RBF vs. quantum fidelity.

    Model name in any CSV: "Classical RBF-SVM (N-dim)" where N is feature_dimension.
    """

    MODEL_FILE = "classical_baseline_svc.pkl"
    METRICS_FILE = "classical_baseline_metrics.json"

    def __init__(
        self,
        feature_dimension: int,
        svm_c: float = 1.0,
        class_weight: Optional[str] = "balanced",
        random_seed: int = 42,
    ):
        self.feature_dimension = feature_dimension
        self.svm_c = svm_c
        self.class_weight = class_weight
        self.random_seed = random_seed

        self._svc: Optional[SVC] = None
        self._is_fitted: bool = False
        self._metrics: Dict[str, float] = {}

    @property
    def model_name(self) -> str:
        return f"Classical RBF-SVM ({self.feature_dimension}-dim)"

    # ── Core interface ─────────────────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "ClassicalBaselineSVM":
        """Fit RBF-SVM on low-dimensional features (not a kernel matrix).

        Args:
            X_train: Reduced features, shape (N, feature_dimension).
            y_train: Integer labels, shape (N,). 0=REAL, 1=FAKE.

        Returns:
            self (for chaining).
        """
        assert X_train.shape[1] == self.feature_dimension, (
            f"Expected {self.feature_dimension}-dim features; "
            f"got {X_train.shape[1]}-dim"
        )
        self._svc = SVC(
            kernel="rbf",
            C=self.svm_c,
            class_weight=self.class_weight,
            probability=True,
            random_state=self.random_seed,
        )
        self._svc.fit(X_train, y_train)
        self._is_fitted = True
        logger.info(
            f"ClassicalBaselineSVM fitted: {X_train.shape[0]} samples, "
            f"feature_dim={self.feature_dimension}, C={self.svm_c}, "
            f"support_vectors={self._svc.n_support_.sum()}"
        )
        return self

    def evaluate(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        video_ids_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Return accuracy, precision, recall, F1, ROC-AUC.

        Args:
            X_val:         Reduced val features, shape (M, feature_dimension).
            y_val:         True labels, shape (M,).
            video_ids_val: Unused (kept for API consistency). Reserved for
                           future video-level aggregation support.

        Returns:
            Dict with keys: accuracy, precision, recall, f1, roc_auc.
        """
        self._check_fitted()

        y_pred = self._svc.predict(X_val)
        y_prob = self._svc.predict_proba(X_val)[:, 1]

        try:
            auc = float(roc_auc_score(y_val, y_prob))
        except ValueError:
            auc = float("nan")

        metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "f1": float(f1_score(y_val, y_pred, zero_division=0)),
            "roc_auc": auc,
            "model_name": self.model_name,
            "feature_dimension": self.feature_dimension,
            "svm_c": self.svm_c,
        }
        self._metrics = metrics
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        self._check_fitted()
        return self._svc.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        self._check_fitted()
        return self._svc.predict_proba(X)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, output_dir: str) -> None:
        """Save SVC pkl + metrics JSON to output_dir.

        Creates:
            <output_dir>/classical_baseline_svc.pkl
            <output_dir>/classical_baseline_metrics.json
        """
        self._check_fitted()
        d = Path(output_dir)
        d.mkdir(parents=True, exist_ok=True)

        joblib.dump(self._svc, d / self.MODEL_FILE)

        meta = {
            "model_name": self.model_name,
            "feature_dimension": self.feature_dimension,
            "svm_c": self.svm_c,
            "class_weight": self.class_weight,
            "random_seed": self.random_seed,
            "metrics": self._metrics,
        }
        with open(d / self.METRICS_FILE, "w") as f:
            json.dump(meta, f, indent=4)

        logger.info(f"ClassicalBaselineSVM saved to: {d}")

    @staticmethod
    def load(models_dir: str) -> "ClassicalBaselineSVM":
        """Load a previously saved ClassicalBaselineSVM.

        Args:
            models_dir: Directory containing classical_baseline_svc.pkl
                        and classical_baseline_metrics.json.

        Returns:
            Fitted ClassicalBaselineSVM instance.

        Raises:
            FileNotFoundError: If pkl is missing.
        """
        d = Path(models_dir)
        pkl_path = d / ClassicalBaselineSVM.MODEL_FILE
        meta_path = d / ClassicalBaselineSVM.METRICS_FILE

        if not pkl_path.exists():
            raise FileNotFoundError(
                f"{ClassicalBaselineSVM.MODEL_FILE} not found in {d}"
            )

        meta: dict = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        feature_dimension = meta.get("feature_dimension", 4)
        svm_c = meta.get("svm_c", 1.0)
        class_weight = meta.get("class_weight", "balanced")
        random_seed = meta.get("random_seed", 42)

        clf = ClassicalBaselineSVM(
            feature_dimension=feature_dimension,
            svm_c=svm_c,
            class_weight=class_weight,
            random_seed=random_seed,
        )
        clf._svc = joblib.load(pkl_path)
        clf._is_fitted = True
        clf._metrics = meta.get("metrics", {})

        logger.info(f"ClassicalBaselineSVM loaded from: {d}")
        return clf

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._is_fitted or self._svc is None:
            raise RuntimeError(
                "ClassicalBaselineSVM must be fitted before inference. "
                "Call fit(X_train, y_train) first."
            )
