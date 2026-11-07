"""
Quantum Feature Reducer
========================
Reduces raw 512-dimensional ResNet-18 features to a small quantum-ready
feature space (4, 6, or 8 dimensions) suitable for encoding into a
quantum circuit via ZZFeatureMap.

Pipeline per sample:
    512-dim ResNet-18  →  StandardScaler  →  PCA(n)  →  MinMaxScaler([-π, π])

Design decisions:
- ALL transformations are fitted on training data ONLY to prevent leakage.
- After fitting, calling fit() again without force=True raises RuntimeError.
- The reducer preserves video_ids / labels arrays unchanged (pass-through).
- Three artefacts are saved: scaler.pkl, pca.pkl, feature_scaler.pkl.
- A JSON sidecar records configuration for reproducibility.

Usage:
    python -m src.quantum.feature_reducer --fit
"""

import sys
import json
import argparse
import numpy as np
import joblib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantum.quantum_config import QuantumConfig

try:
    from utils import setup_logger
    logger = setup_logger("quantum.feature_reducer")
except ImportError:
    import logging
    logger = logging.getLogger("quantum.feature_reducer")


class QuantumFeatureReducer:
    """512-dim ResNet-18 features → N-dim quantum-ready features.

    Attributes:
        config:   QuantumConfig instance (uses feature_dimension, feature_range).
        _fitted:  Whether the reducer has been fitted on training data.
        scaler:   StandardScaler for initial normalisation.
        pca:      PCA for dimensionality reduction.
        feature_scaler: MinMaxScaler for mapping to quantum feature range.
    """

    ARTEFACT_NAMES = ("scaler.pkl", "pca.pkl", "feature_scaler.pkl")

    def __init__(self, config: Optional[QuantumConfig] = None):
        self.config = config or QuantumConfig()
        self._fitted = False
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.feature_scaler: Optional[MinMaxScaler] = None
        self._fit_metadata: dict = {}

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, force: bool = False) -> "QuantumFeatureReducer":
        """Fit all preprocessing transformers on training data only.

        Args:
            X_train: Raw ResNet-18 features, shape (N, 512).
            force:   Allow re-fitting. Default False (guards against leakage).

        Returns:
            self (for chaining).

        Raises:
            RuntimeError: If already fitted and force=False.
            ValueError:   If X_train has wrong feature dimension.
        """
        if self._fitted and not force:
            raise RuntimeError(
                "QuantumFeatureReducer is already fitted. "
                "Pass force=True to refit (e.g. when changing config)."
            )

        n_samples, n_features = X_train.shape
        n_components = self.config.feature_dimension

        if n_features < n_components:
            raise ValueError(
                f"X_train has {n_features} features but config requests "
                f"{n_components} PCA components."
            )

        logger.info(
            f"Fitting QuantumFeatureReducer on {n_samples} training samples "
            f"({n_features} → {n_components} dims)"
        )

        # Step 1: StandardScaler (zero mean, unit variance)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # Step 2: PCA (512 → n_components)
        n_components_actual = min(n_components, n_samples - 1, n_features)
        self.pca = PCA(n_components=n_components_actual, random_state=self.config.random_seed)
        X_pca = self.pca.fit_transform(X_scaled)

        # Step 3: MinMaxScaler (→ [feature_range_low, feature_range_high])
        self.feature_scaler = MinMaxScaler(
            feature_range=self.config.feature_range
        )
        self.feature_scaler.fit(X_pca)

        self._fitted = True

        # Record fitting metadata for reproducibility
        self._fit_metadata = {
            "original_feature_dim": int(n_features),
            "pca_components": int(n_components_actual),
            "feature_range": list(self.config.feature_range),
            "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
            "cumulative_explained_variance": float(
                self.pca.explained_variance_ratio_.sum()
            ),
            "n_train_samples": int(n_samples),
            "scaler_mean_norm": float(np.linalg.norm(self.scaler.mean_)),
        }

        logger.info(
            f"PCA cumulative explained variance: "
            f"{self._fit_metadata['cumulative_explained_variance'] * 100:.2f}%"
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted pipeline to feature data.

        Args:
            X: Raw ResNet-18 features, shape (N, 512).

        Returns:
            Quantum-ready features, shape (N, feature_dimension).

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._fitted:
            raise RuntimeError(
                "QuantumFeatureReducer must be fitted before transform(). "
                "Call fit(X_train) first."
            )
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        X_quantum = self.feature_scaler.transform(X_pca)
        return X_quantum.astype(np.float64)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Convenience: fit then transform training data.

        Args:
            X_train: Raw ResNet-18 training features.

        Returns:
            Transformed training features.
        """
        self.fit(X_train)
        return self.transform(X_train)

    # ── Artefact I/O ──────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save all fitted transformers and configuration to directory.

        Creates:
            <dir>/scaler.pkl
            <dir>/pca.pkl
            <dir>/feature_scaler.pkl
            <dir>/feature_scaling.json

        Args:
            directory: Path to output directory.
        """
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted QuantumFeatureReducer.")

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.scaler, d / "scaler.pkl")
        joblib.dump(self.pca, d / "pca.pkl")
        joblib.dump(self.feature_scaler, d / "feature_scaler.pkl")

        meta = {**self._fit_metadata}
        meta["config"] = self.config.to_dict()
        with open(d / "feature_scaling.json", "w") as f:
            json.dump(meta, f, indent=4)

        logger.info(f"QuantumFeatureReducer saved to: {d}")

    @classmethod
    def load(cls, directory: str, config: Optional[QuantumConfig] = None) -> "QuantumFeatureReducer":
        """Load a previously saved QuantumFeatureReducer.

        Args:
            directory: Directory containing the .pkl artefacts.
            config:    Optional QuantumConfig override. If None, loaded from JSON.

        Returns:
            Fitted QuantumFeatureReducer.
        """
        d = Path(directory)

        meta_path = d / "feature_scaling.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            if config is None:
                config = QuantumConfig.load_json(str(meta_path))
                # feature_scaling.json has nested "config" dict
                if "config" in meta:
                    config = QuantumConfig(**{
                        k: v for k, v in meta["config"].items()
                        if k in QuantumConfig.__dataclass_fields__
                    })

        reducer = cls(config=config)
        reducer.scaler = joblib.load(d / "scaler.pkl")
        reducer.pca = joblib.load(d / "pca.pkl")
        reducer.feature_scaler = joblib.load(d / "feature_scaler.pkl")
        reducer._fitted = True

        if meta_path.exists():
            reducer._fit_metadata = {k: v for k, v in meta.items() if k != "config"}

        logger.info(f"QuantumFeatureReducer loaded from: {d}")
        return reducer


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fit quantum feature reducer on extracted ResNet-18 features."
    )
    parser.add_argument(
        "--fit", action="store_true",
        help="Fit the reducer on training features and save artefacts."
    )
    parser.add_argument(
        "--num-qubits", type=int, default=4, choices=[4, 6, 8],
        help="Number of quantum features / qubits (default: 4)."
    )
    parser.add_argument(
        "--features-dir", type=str, default=None,
        help="Directory containing {split}_features.npy files."
    )
    parser.add_argument(
        "--models-dir", type=str, default=None,
        help="Output directory for quantum artefacts."
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import Config
    cfg_classical = Config()

    features_dir = Path(args.features_dir) if args.features_dir else cfg_classical.features_dir
    models_dir = Path(args.models_dir) if args.models_dir else cfg_classical.models_dir / "quantum"

    cfg_quantum = QuantumConfig(
        num_qubits=args.num_qubits,
        feature_dimension=args.num_qubits,
    )

    if args.fit:
        print()
        print("=" * 55)
        print("  QuantumFeatureReducer — Fitting")
        print("=" * 55)

        # Load features
        X_train = np.load(features_dir / "train_features.npy")
        X_val   = np.load(features_dir / "val_features.npy")
        X_test  = np.load(features_dir / "test_features.npy")

        reducer = QuantumFeatureReducer(config=cfg_quantum)
        X_train_q = reducer.fit_transform(X_train)
        X_val_q   = reducer.transform(X_val)
        X_test_q  = reducer.transform(X_test)

        reducer.save(str(models_dir))

        print(f"\n  Train: {X_train.shape} → {X_train_q.shape}")
        print(f"  Val:   {X_val.shape}   → {X_val_q.shape}")
        print(f"  Test:  {X_test.shape}  → {X_test_q.shape}")
        print(f"\n  Range check — min: {X_train_q.min():.4f}, max: {X_train_q.max():.4f}")
        print(f"\n  Artefacts saved to: {models_dir}")
        print()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
