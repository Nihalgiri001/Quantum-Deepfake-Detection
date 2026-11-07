"""
Quantum Kernel Utilities
=========================
Shared utility functions for kernel matrix operations used across the
quantum package. Pure functions — no state.

Functions:
    symmetrize              — force symmetry via (K + K.T) / 2
    check_psd               — test positive semi-definiteness
    kernel_alignment        — normalised Frobenius similarity between kernels
    subsample_balanced      — stratified subsampling preserving class balance
    hash_kernel_config      — MD5 cache key for kernel configurations
    save_kernel_matrix      — save .npy + sidecar JSON metadata
    load_kernel_matrix      — reload .npy + sidecar JSON metadata
"""

import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional


# ── Kernel matrix operations ──────────────────────────────────────────────────

def symmetrize(K: np.ndarray) -> np.ndarray:
    """Return (K + K.T) / 2.

    Only call this when asymmetry is documented and within tolerance.
    Does NOT silently hide large asymmetry.

    Args:
        K: Square kernel matrix.

    Returns:
        Symmetrised copy of K.
    """
    if K.shape[0] != K.shape[1]:
        raise ValueError(f"symmetrize requires square matrix; got shape {K.shape}")
    sym = (K + K.T) / 2.0
    asym_error = float(np.abs(K - K.T).max())
    if asym_error > 1e-4:
        import warnings
        warnings.warn(
            f"symmetrize: large asymmetry detected (max |K-K.T| = {asym_error:.6e}). "
            "This may indicate a numerical issue in kernel computation.",
            RuntimeWarning,
        )
    return sym


def check_psd(K: np.ndarray, tol: float = 1e-8) -> bool:
    """Check whether a matrix is positive semi-definite.

    Args:
        K: Square symmetric matrix.
        tol: Tolerance; eigenvalues >= -tol are considered non-negative.

    Returns:
        True if K is PSD within tolerance.
    """
    if K.shape[0] != K.shape[1]:
        return False
    try:
        eigvals = np.linalg.eigvalsh(K)
        return bool(np.all(eigvals >= -tol))
    except np.linalg.LinAlgError:
        return False


def kernel_alignment(K1: np.ndarray, K2: np.ndarray) -> float:
    """Compute normalised kernel alignment (Frobenius inner product).

    Used for comparing hardware-derived vs simulator-derived kernels.
    Value is in [0, 1] for PSD kernels with non-negative entries.

    Args:
        K1: First kernel matrix.
        K2: Second kernel matrix (same shape).

    Returns:
        Alignment score in [0, 1].
    """
    if K1.shape != K2.shape:
        raise ValueError(
            f"Kernel shapes must match; got {K1.shape} vs {K2.shape}"
        )
    inner = float(np.sum(K1 * K2))
    norm1 = float(np.sqrt(np.sum(K1 * K1)))
    norm2 = float(np.sqrt(np.sum(K2 * K2)))
    denom = norm1 * norm2
    if denom < 1e-12:
        return 0.0
    return inner / denom


# ── Subsampling ───────────────────────────────────────────────────────────────

def subsample_balanced(
    X: np.ndarray,
    y: np.ndarray,
    n: int,
    seed: int = 42,
    video_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified subsampling preserving class balance.

    Selects up to n samples total, approximately balanced between classes.
    If a class has fewer than n//2 samples, all of that class are kept.

    Args:
        X: Feature array of shape (N, D).
        y: Label array of shape (N,).
        n: Target total sample count.
        seed: Random seed.
        video_ids: Optional video IDs array; subsampled consistently with X.

    Returns:
        (X_sub, y_sub, indices) where indices are the selected row indices
        into the original arrays.
    """
    rng = np.random.default_rng(seed)
    unique_labels = np.unique(y)
    n_per_class = n // len(unique_labels)

    selected_indices = []
    for label in unique_labels:
        class_idx = np.where(y == label)[0]
        take = min(n_per_class, len(class_idx))
        chosen = rng.choice(class_idx, size=take, replace=False)
        selected_indices.append(chosen)

    indices = np.concatenate(selected_indices)
    # Shuffle the combined selection
    indices = rng.permutation(indices)

    return X[indices], y[indices], indices


# ── Cache key ─────────────────────────────────────────────────────────────────

def hash_kernel_config(config_dict: Dict[str, Any]) -> str:
    """Compute an MD5 hash of a kernel configuration dictionary.

    Used for cache invalidation: when configuration changes, the cached
    kernel matrices are considered stale and are recomputed.

    Args:
        config_dict: Dictionary of configuration parameters.

    Returns:
        8-character hex digest.
    """
    # Stable serialisation: sort keys, stringify values
    canonical = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(canonical.encode()).hexdigest()[:8]


# ── Kernel matrix I/O ─────────────────────────────────────────────────────────

def save_kernel_matrix(
    K: np.ndarray,
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a kernel matrix as .npy with a JSON sidecar for metadata.

    Args:
        K: Kernel matrix to save.
        path: Destination path (e.g. "features/quantum/K_train.npy").
        metadata: Optional dictionary to save as <path>.json.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(p), K)

    if metadata is not None:
        meta_path = p.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)


def load_kernel_matrix(
    path: str,
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """Load a kernel matrix and its optional JSON sidecar.

    Args:
        path: Path to the .npy file.

    Returns:
        (K, metadata) where metadata is None if no sidecar exists.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Kernel matrix not found: {p}")
    K = np.load(str(p))

    meta_path = p.with_suffix(".json")
    metadata = None
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    return K, metadata
