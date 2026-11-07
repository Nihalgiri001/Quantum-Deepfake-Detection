"""
Quantum Kernel Computer
=========================
Computes fidelity-based quantum kernel matrices using a ZZFeatureMap
circuit and the FidelityQuantumKernel from Qiskit Machine Learning.

The quantum kernel between two data points x and x' is:
    K(x, x') = |⟨φ(x)|φ(x')⟩|²

where |φ(x)⟩ is the quantum state produced by the ZZFeatureMap applied
to x.  This fidelity kernel implicitly embeds data in an exponentially
large Hilbert space and can capture correlations that are classically
hard to compute at scale.

Design decisions:
- Training kernel:  K_train[i,j] = K(x_i, x_j)  — shape (n, n), symmetric
- Cross kernel:     K_cross[i,j] = K(x_new_i, x_train_j)  — shape (m, n)
- Caching:  kernel matrices are saved as .npy + sidecar JSON keyed by an
  MD5 hash of (feature_dimension, reps, entanglement, n_train_samples).
  A cache hit avoids recomputing O(n²) circuit evaluations.
- Validation:  checks shape, finiteness, symmetry, diagonal, and value
  range.  Violations are logged as warnings.  NaN/Inf raise
  KernelNumericalError — the matrix is NEVER silently patched.
- Diagnostics:  symmetry error, diagonal stats, eigenvalue bounds, and
  condition number are always available via diagnose_kernel().

Usage:
    backend  = SimulatorBackend(shots=1024, seed=42)
    computer = QuantumKernelComputer(config=QuantumConfig(), backend=backend)
    K_train  = computer.compute_train_kernel(X_train)
    K_test   = computer.compute_cross_kernel(X_test, X_train)

CLI:
    python -m src.quantum.quantum_kernel
    python -m src.quantum.quantum_kernel --n-samples 20
"""

import sys
import json
import time
import logging
import argparse
import warnings
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
except ImportError as exc:
    raise ImportError(
        "qiskit-machine-learning is required.\n"
        "Install it with:  pip install 'qiskit-machine-learning==0.9.1'"
    ) from exc

from quantum.quantum_config import QuantumConfig
from quantum.quantum_feature_map import build_feature_map, validate_feature_map
from quantum.simulator_backend import SimulatorBackend
from quantum.kernel_utils import (
    hash_kernel_config,
    save_kernel_matrix,
    load_kernel_matrix,
)

try:
    from utils import setup_logger
    logger = setup_logger("quantum.kernel")
except ImportError:
    logger = logging.getLogger("quantum.kernel")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO,
                            format="[%(asctime)s] %(levelname)-8s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")


# ── Custom exception ──────────────────────────────────────────────────────────

class KernelNumericalError(Exception):
    """Raised when a kernel matrix contains NaN or Inf values.

    The matrix is not corrected; computation should be re-attempted with
    different parameters or fewer shots.
    """


# ── Main class ────────────────────────────────────────────────────────────────

class QuantumKernelComputer:
    """Compute and cache fidelity-based quantum kernel matrices.

    Attributes:
        config:    QuantumConfig controlling the circuit and sampler.
        backend:   SimulatorBackend (or compatible) providing the sampler.
        cache_dir: Directory for cached .npy kernel matrices.
        _kernel:   Lazily-built FidelityQuantumKernel instance.
    """

    def __init__(
        self,
        config: Optional[QuantumConfig] = None,
        backend: Optional[SimulatorBackend] = None,
        cache_dir: Optional[str] = None,
    ):
        """Initialise the kernel computer.

        Args:
            config:    QuantumConfig.  Defaults to QuantumConfig().
            backend:   SimulatorBackend.  Defaults to SimulatorBackend().
            cache_dir: Directory for cached kernel matrices.  Defaults to
                       features/quantum/ relative to project root.
        """
        self.config = config or QuantumConfig()
        self.backend = backend or SimulatorBackend(
            shots=self.config.shots,
            seed=self.config.random_seed,
        )

        # Resolve cache directory
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
        else:
            # Walk up from src/quantum/ to project root, then into features/quantum/
            _pkg_root = Path(__file__).resolve().parent.parent.parent
            self.cache_dir = _pkg_root / "features" / "quantum"

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazily initialised
        self._feature_map = None
        self._kernel: Optional[FidelityQuantumKernel] = None

        # Cached training data for cross-kernel reuse
        self._X_train_cache: Optional[np.ndarray] = None

    # ── Kernel construction ────────────────────────────────────────────────────

    def _get_kernel(self) -> FidelityQuantumKernel:
        """Return the cached FidelityQuantumKernel, building it on first call."""
        if self._kernel is None:
            self._feature_map = build_feature_map(self.config)
            validate_feature_map(self._feature_map, self.config)
            self._kernel = self.backend.build_kernel(self._feature_map)
            logger.info(
                f"Built FidelityQuantumKernel: {self.config.feature_dimension} qubits, "
                f"reps={self.config.reps}, entanglement={self.config.entanglement}"
            )
        return self._kernel

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _cache_key(self, n_train: int) -> str:
        """Build the MD5 cache key for the current config + dataset size."""
        return hash_kernel_config({
            "feature_dimension": self.config.feature_dimension,
            "reps": self.config.reps,
            "entanglement": self.config.entanglement,
            "n_train_samples": n_train,
            "shots": self.config.shots,
            "random_seed": self.config.random_seed,
        })

    def _try_load_cache(self, name: str, expected_shape: tuple, config_hash: str):
        """Attempt to load a cached kernel matrix.

        Returns the matrix if the cache is valid, None otherwise.
        """
        npy_path = self.cache_dir / f"{name}.npy"
        if not npy_path.exists():
            return None
        try:
            K, meta = load_kernel_matrix(str(npy_path))
            if meta is None or meta.get("config_hash") != config_hash:
                logger.info(f"Cache miss ({name}): config hash changed — recomputing.")
                return None
            if K.shape != expected_shape:
                logger.info(f"Cache miss ({name}): shape {K.shape} ≠ {expected_shape} — recomputing.")
                return None
            logger.info(f"Cache hit: loaded {name} from {npy_path}")
            return K
        except Exception as e:
            logger.warning(f"Cache load failed for {name}: {e} — recomputing.")
            return None

    def _save_cache(self, K: np.ndarray, name: str, extra_meta: dict):
        """Save a kernel matrix to the cache directory."""
        npy_path = self.cache_dir / f"{name}.npy"
        meta = {
            "config_hash": self._cache_key(
                extra_meta.get("n_train_samples", K.shape[0])
            ),
            "shape": list(K.shape),
            "feature_dimension": self.config.feature_dimension,
            "reps": self.config.reps,
            "entanglement": self.config.entanglement,
            "shots": self.config.shots,
            "random_seed": self.config.random_seed,
            **extra_meta,
        }
        save_kernel_matrix(K, str(npy_path), metadata=meta)
        logger.info(f"Cached kernel matrix: {npy_path}")

    # ── Core computation ──────────────────────────────────────────────────────

    def compute_train_kernel(
        self,
        X_train: np.ndarray,
        use_cache: bool = True,
    ) -> np.ndarray:
        """Compute the (n, n) symmetric training kernel matrix.

        K_train[i,j] = |⟨φ(x_i)|φ(x_j)⟩|²

        Args:
            X_train:   Reduced feature vectors, shape (n, feature_dimension).
            use_cache: If True, load from cache when config hash matches.

        Returns:
            Symmetric kernel matrix of shape (n, n).

        Raises:
            KernelNumericalError: If the matrix contains NaN or Inf.
        """
        n = X_train.shape[0]
        config_hash = self._cache_key(n)

        if use_cache:
            cached = self._try_load_cache("K_train", (n, n), config_hash)
            if cached is not None:
                self._X_train_cache = X_train.copy()
                return cached

        logger.info(f"Computing training kernel ({n}×{n}) — this may take a while…")
        t0 = time.time()

        kernel = self._get_kernel()
        K = kernel.evaluate(x_vec=X_train)

        elapsed = time.time() - t0
        logger.info(f"Training kernel computed in {elapsed:.1f}s")

        self.validate_kernel(K, name="K_train", is_train=True)
        self._X_train_cache = X_train.copy()

        if use_cache:
            self._save_cache(K, "K_train", {
                "n_train_samples": n,
                "compute_time_sec": round(elapsed, 2),
            })

        return K

    def compute_cross_kernel(
        self,
        X_new: np.ndarray,
        X_train: np.ndarray,
        name: str = "K_cross",
        use_cache: bool = False,
    ) -> np.ndarray:
        """Compute the (m, n) cross kernel matrix between new and training data.

        K_cross[i,j] = |⟨φ(x_new_i)|φ(x_train_j)⟩|²

        Args:
            X_new:     New feature vectors, shape (m, feature_dimension).
            X_train:   Training feature vectors, shape (n, feature_dimension).
            name:      Cache name (e.g. "K_val", "K_test").
            use_cache: Whether to attempt a cache load/save.

        Returns:
            Cross kernel matrix of shape (m, n).

        Raises:
            KernelNumericalError: If the matrix contains NaN or Inf.
        """
        m, n = X_new.shape[0], X_train.shape[0]
        config_hash = self._cache_key(n)

        if use_cache:
            cached = self._try_load_cache(name, (m, n), config_hash)
            if cached is not None:
                return cached

        logger.info(f"Computing cross kernel ({m}×{n}) for '{name}'…")
        t0 = time.time()

        kernel = self._get_kernel()
        K = kernel.evaluate(x_vec=X_new, y_vec=X_train)

        elapsed = time.time() - t0
        logger.info(f"Cross kernel '{name}' computed in {elapsed:.1f}s")

        self.validate_kernel(K, name=name, is_train=False)

        if use_cache:
            self._save_cache(K, name, {
                "n_train_samples": n,
                "n_new_samples": m,
                "compute_time_sec": round(elapsed, 2),
            })

        return K

    # ── Validation ────────────────────────────────────────────────────────────

    def validate_kernel(
        self,
        K: np.ndarray,
        name: str = "K",
        is_train: bool = True,
    ) -> None:
        """Validate a kernel matrix and report all violations.

        Raises KernelNumericalError on NaN/Inf.  All other violations are
        logged as warnings — the matrix is NOT modified.

        Checks:
            1. Finite values (no NaN or Inf) — hard error.
            2. Value range [0, 1] — warning.
            3. Shape is 2-D — warning.
            4. Symmetry (train only) — warning.
            5. Diagonal ≈ 1 (train only, tol 1e-3) — warning.

        Args:
            K:        Kernel matrix to validate.
            name:     Descriptive name for log messages.
            is_train: If True, apply symmetry and diagonal checks.

        Raises:
            KernelNumericalError: If K contains NaN or Inf.
        """
        # ── Hard check: finiteness ─────────────────────────────────────────
        if np.isnan(K).any() or np.isinf(K).any():
            n_bad = int(np.isnan(K).sum() + np.isinf(K).sum())
            raise KernelNumericalError(
                f"Kernel '{name}' contains {n_bad} NaN/Inf entries. "
                "This indicates a numerical failure in circuit simulation. "
                "Try reducing feature_dimension, reps, or increasing shots."
            )

        # ── Soft checks ────────────────────────────────────────────────────
        if K.ndim != 2:
            logger.warning(f"[{name}] Expected 2-D matrix; got {K.ndim}-D.")

        out_of_range = ((K < -1e-6) | (K > 1.0 + 1e-6)).sum()
        if out_of_range > 0:
            logger.warning(
                f"[{name}] {out_of_range} entries outside [0, 1] "
                f"(min={K.min():.6f}, max={K.max():.6f})."
            )

        if is_train:
            sym_error = float(np.abs(K - K.T).max())
            if sym_error > 1e-6:
                logger.warning(
                    f"[{name}] Symmetry violation: max|K−Kᵀ| = {sym_error:.2e}. "
                    "For statevector mode this should be < 1e-14."
                )

            diag = np.diag(K)
            diag_err = float(np.abs(diag - 1.0).max())
            if diag_err > 1e-3:
                logger.warning(
                    f"[{name}] Diagonal deviation: max|diag−1| = {diag_err:.4f}. "
                    "Fidelity of a state with itself should be 1."
                )

        logger.debug(
            f"[{name}] shape={K.shape}, "
            f"min={K.min():.4f}, max={K.max():.4f}, "
            f"mean={K.mean():.4f}"
        )

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def diagnose_kernel(self, K: np.ndarray) -> Dict[str, Any]:
        """Return a diagnostic summary of a kernel matrix.

        Used for logging, debugging, and inclusion in training metadata.

        Args:
            K: Kernel matrix (square for train, rectangular for cross).

        Returns:
            Dictionary with keys:
                shape              — tuple (rows, cols)
                is_square          — bool
                symmetry_error     — max|K−Kᵀ| (None if not square)
                diag_mean          — mean of diagonal (None if not square)
                diag_std           — std of diagonal (None if not square)
                min_value          — minimum entry
                max_value          — maximum entry
                has_nan            — bool
                has_inf            — bool
                min_eigenvalue     — smallest eigenvalue (None if not square)
                max_eigenvalue     — largest eigenvalue (None if not square)
                condition_number   — matrix condition number (None if not square)
                is_psd             — bool, eigenvalue-based (None if not square)
        """
        is_square = K.ndim == 2 and K.shape[0] == K.shape[1]

        result: Dict[str, Any] = {
            "shape": list(K.shape),
            "is_square": is_square,
            "symmetry_error": None,
            "diag_mean": None,
            "diag_std": None,
            "min_value": float(K.min()) if K.size > 0 else None,
            "max_value": float(K.max()) if K.size > 0 else None,
            "has_nan": bool(np.isnan(K).any()),
            "has_inf": bool(np.isinf(K).any()),
            "min_eigenvalue": None,
            "max_eigenvalue": None,
            "condition_number": None,
            "is_psd": None,
        }

        if is_square and not result["has_nan"] and not result["has_inf"]:
            result["symmetry_error"] = float(np.abs(K - K.T).max())

            diag = np.diag(K)
            result["diag_mean"] = float(diag.mean())
            result["diag_std"] = float(diag.std())

            try:
                eigvals = np.linalg.eigvalsh(K)
                result["min_eigenvalue"] = float(eigvals.min())
                result["max_eigenvalue"] = float(eigvals.max())
                result["is_psd"] = bool(eigvals.min() >= -1e-8)
                result["condition_number"] = float(np.linalg.cond(K))
            except np.linalg.LinAlgError:
                pass

        return result


# ── CLI / demo ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute and inspect a quantum kernel matrix."
    )
    parser.add_argument(
        "--n-samples", type=int, default=10,
        help="Number of training samples for the demo (default: 10)."
    )
    parser.add_argument(
        "--n-test", type=int, default=4,
        help="Number of test samples for the cross-kernel demo (default: 4)."
    )
    parser.add_argument(
        "--num-qubits", type=int, default=4, choices=[4, 6, 8],
        help="Number of qubits (default: 4)."
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Skip kernel matrix cache."
    )
    args = parser.parse_args()

    import tempfile, os

    cfg = QuantumConfig(
        num_qubits=args.num_qubits,
        feature_dimension=args.num_qubits,
        shots=1024,
        random_seed=42,
    )
    backend = SimulatorBackend(shots=cfg.shots, seed=cfg.random_seed)

    # Use a temp dir for demo cache so we don't pollute features/quantum/
    with tempfile.TemporaryDirectory() as tmp_cache:
        computer = QuantumKernelComputer(
            config=cfg,
            backend=backend,
            cache_dir=tmp_cache,
        )

        rng = np.random.default_rng(42)
        X_train = rng.uniform(-np.pi, np.pi, (args.n_samples, cfg.feature_dimension))
        X_test  = rng.uniform(-np.pi, np.pi, (args.n_test,    cfg.feature_dimension))

        print()
        print("=" * 60)
        print("  QuantumKernelComputer — Demo")
        print("=" * 60)
        print(f"  num_qubits   : {cfg.num_qubits}")
        print(f"  n_train      : {args.n_samples}")
        print(f"  n_test       : {args.n_test}")
        print()

        # Training kernel
        print("  Computing training kernel…")
        t0 = time.time()
        K_train = computer.compute_train_kernel(X_train, use_cache=not args.no_cache)
        t_train = time.time() - t0

        # Cross kernel
        print("  Computing cross kernel…")
        t0 = time.time()
        K_test = computer.compute_cross_kernel(X_test, X_train, name="K_test",
                                               use_cache=not args.no_cache)
        t_cross = time.time() - t0

        # Diagnostics
        diag_train = computer.diagnose_kernel(K_train)
        diag_test  = computer.diagnose_kernel(K_test)

        print()
        print("─" * 60)
        print("  Training Kernel Diagnostics")
        print("─" * 60)
        print(f"  Shape          : {diag_train['shape']}")
        print(f"  Symmetric      : {diag_train['symmetry_error']:.2e}")
        print(f"  Diag mean±std  : {diag_train['diag_mean']:.6f} ± {diag_train['diag_std']:.2e}")
        print(f"  Min/Max value  : {diag_train['min_value']:.6f} / {diag_train['max_value']:.6f}")
        print(f"  Min eigenvalue : {diag_train['min_eigenvalue']:.6f}")
        print(f"  Condition no.  : {diag_train['condition_number']:.2f}")
        print(f"  PSD            : {diag_train['is_psd']}")
        print(f"  Compute time   : {t_train:.2f}s")
        print()
        print("─" * 60)
        print("  Cross Kernel Diagnostics")
        print("─" * 60)
        print(f"  Shape          : {diag_test['shape']}")
        print(f"  Min/Max value  : {diag_test['min_value']:.6f} / {diag_test['max_value']:.6f}")
        print(f"  Has NaN/Inf    : {diag_test['has_nan']} / {diag_test['has_inf']}")
        print(f"  Compute time   : {t_cross:.2f}s")
        print()

        # Acceptance criteria summary
        print("─" * 60)
        print("  Acceptance Checks")
        print("─" * 60)

        n, m = args.n_samples, args.n_test
        assert K_train.shape == (n, n), f"Train shape {K_train.shape} ≠ ({n},{n})"
        print(f"  ✅  K_train shape == ({n},{n})")

        assert K_test.shape == (m, n), f"Cross shape {K_test.shape} ≠ ({m},{n})"
        print(f"  ✅  K_test shape == ({m},{n})")

        assert diag_train['symmetry_error'] < 1e-6, \
            f"Symmetry error {diag_train['symmetry_error']:.2e} ≥ 1e-6"
        print(f"  ✅  |K_train − K_trainᵀ|.max() < 1e-6")

        assert abs(diag_train['diag_mean'] - 1.0) < 1e-3, \
            f"Diag mean {diag_train['diag_mean']:.6f} not ≈ 1"
        print(f"  ✅  Diagonal ≈ 1 (within 1e-3)")

        assert not diag_train['has_nan'] and not diag_train['has_inf'], "NaN/Inf in K_train"
        assert not diag_test['has_nan']  and not diag_test['has_inf'],  "NaN/Inf in K_test"
        print(f"  ✅  All values finite")

        assert diag_train['min_value'] >= -1e-6, f"Min value {diag_train['min_value']:.6f} < 0"
        assert diag_train['max_value'] <= 1.0 + 1e-6
        print(f"  ✅  Values in [0, 1]")

        # NaN injection test
        K_bad = K_train.copy()
        K_bad[0, 0] = float("nan")
        try:
            computer.validate_kernel(K_bad, name="K_bad")
            raise AssertionError("Should have raised KernelNumericalError")
        except KernelNumericalError:
            pass
        print(f"  ✅  KernelNumericalError raised on injected NaN")

        # Diagnose dict keys
        required_keys = {
            "shape", "is_square", "symmetry_error", "diag_mean", "diag_std",
            "min_value", "max_value", "has_nan", "has_inf",
            "min_eigenvalue", "max_eigenvalue", "condition_number", "is_psd",
        }
        missing = required_keys - set(diag_train.keys())
        assert not missing, f"diagnose_kernel missing keys: {missing}"
        print(f"  ✅  diagnose_kernel returns all expected keys")

        # Cache test: recompute should hit cache (same hash)
        K_train2 = computer.compute_train_kernel(X_train, use_cache=True)
        assert np.array_equal(K_train, K_train2), "Cache hit should return identical matrix"
        print(f"  ✅  Cache hit returns identical matrix")

        print()
        print("  ✅  All acceptance criteria passed")
        print("=" * 60)
        print()


if __name__ == "__main__":
    main()
