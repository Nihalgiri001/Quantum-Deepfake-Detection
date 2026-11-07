"""
Shared pytest fixtures for the TrustLens-Q test suite.

Provides:
    rng             — seeded numpy Generator
    small_cfg       — 4-qubit QuantumConfig with small sample budgets
    synthetic_512   — factory for (n, 512) float32 arrays + balanced labels
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ is on sys.path so all quantum modules are importable
_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def small_cfg():
    from quantum.quantum_config import QuantumConfig
    return QuantumConfig(
        num_qubits=4,
        feature_dimension=4,
        shots=512,
        random_seed=42,
        max_train_samples=20,
        max_val_samples=8,
        max_test_samples=8,
    )


@pytest.fixture
def synthetic_512(rng):
    """Return a factory: synthetic_512(n) → (X, y) with balanced labels."""
    def _make(n: int):
        X = rng.standard_normal((n, 512)).astype(np.float32)
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        return X, y
    return _make


@pytest.fixture
def synthetic_quantum(rng, small_cfg):
    """Return reduced (4-dim) features in [-π, π] with balanced labels."""
    def _make(n: int):
        X = rng.uniform(-np.pi, np.pi, (n, small_cfg.feature_dimension))
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        return X, y
    return _make


@pytest.fixture
def precomputed_kernel(rng):
    """Return a valid (n×n) symmetric PSD kernel + labels."""
    def _make(n: int):
        A = rng.uniform(0, 1, (n, n))
        K = (A @ A.T) / n + np.eye(n) * 0.5   # guaranteed PSD
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        return K, y
    return _make
