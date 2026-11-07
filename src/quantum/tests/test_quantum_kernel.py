"""
Tests for src/quantum/quantum_kernel.py

Covers: train/cross shapes, symmetry, diagonal, finite values,
        cache hit, KernelNumericalError on NaN, diagnose_kernel keys.

Uses ≤ 8 training samples so each test completes in < 10 s.
"""
import numpy as np
import pytest

from quantum.quantum_config import QuantumConfig
from quantum.quantum_kernel import QuantumKernelComputer, KernelNumericalError
from quantum.simulator_backend import SimulatorBackend


@pytest.fixture(scope="module")
def kernel_setup(tmp_path_factory):
    """Build a kernel computer with a small config and temp cache dir."""
    tmp = tmp_path_factory.mktemp("kernel_cache")
    cfg = QuantumConfig(num_qubits=4, feature_dimension=4,
                        shots=512, random_seed=42)
    backend  = SimulatorBackend(shots=512, seed=42)
    computer = QuantumKernelComputer(config=cfg, backend=backend,
                                     cache_dir=str(tmp))
    rng = np.random.default_rng(0)
    X_train = rng.uniform(-np.pi, np.pi, (6, 4))
    X_test  = rng.uniform(-np.pi, np.pi, (3, 4))
    return computer, X_train, X_test, cfg, tmp


class TestKernelShapes:
    def test_train_kernel_square(self, kernel_setup):
        computer, X_train, _, _, _ = kernel_setup
        K = computer.compute_train_kernel(X_train, use_cache=False)
        assert K.shape == (6, 6)

    def test_cross_kernel_shape(self, kernel_setup):
        computer, X_train, X_test, _, _ = kernel_setup
        K = computer.compute_cross_kernel(X_test, X_train,
                                          name="K_test_shape", use_cache=False)
        assert K.shape == (3, 6)


class TestKernelProperties:
    def test_symmetric(self, kernel_setup):
        computer, X_train, _, _, _ = kernel_setup
        K = computer.compute_train_kernel(X_train, use_cache=False)
        assert np.abs(K - K.T).max() < 1e-6

    def test_diagonal_approx_one(self, kernel_setup):
        computer, X_train, _, _, _ = kernel_setup
        K = computer.compute_train_kernel(X_train, use_cache=False)
        assert np.abs(np.diag(K) - 1.0).max() < 1e-3

    def test_values_in_range(self, kernel_setup):
        computer, X_train, _, _, _ = kernel_setup
        K = computer.compute_train_kernel(X_train, use_cache=False)
        assert K.min() >= -1e-6
        assert K.max() <=  1.0 + 1e-6

    def test_no_nan_or_inf(self, kernel_setup):
        computer, X_train, _, _, _ = kernel_setup
        K = computer.compute_train_kernel(X_train, use_cache=False)
        assert not np.isnan(K).any()
        assert not np.isinf(K).any()

    def test_reproducible(self, kernel_setup, tmp_path):
        _, X_train, _, cfg, _ = kernel_setup
        b1 = SimulatorBackend(shots=512, seed=42)
        b2 = SimulatorBackend(shots=512, seed=42)
        c1 = QuantumKernelComputer(config=cfg, backend=b1, cache_dir=str(tmp_path / "c1"))
        c2 = QuantumKernelComputer(config=cfg, backend=b2, cache_dir=str(tmp_path / "c2"))
        K1 = c1.compute_train_kernel(X_train, use_cache=False)
        K2 = c2.compute_train_kernel(X_train, use_cache=False)
        assert np.array_equal(K1, K2)


class TestKernelCache:
    def test_cache_hit_returns_identical(self, kernel_setup):
        computer, X_train, _, _, _ = kernel_setup
        K1 = computer.compute_train_kernel(X_train, use_cache=True)
        K2 = computer.compute_train_kernel(X_train, use_cache=True)
        assert np.array_equal(K1, K2)

    def test_cache_file_created(self, kernel_setup):
        _, X_train, _, _, tmp = kernel_setup
        import os
        assert os.path.exists(str(tmp / "K_train.npy"))


class TestValidateKernel:
    def test_nan_raises_hard_error(self, kernel_setup):
        computer, X_train, _, _, _ = kernel_setup
        K = np.eye(4)
        K[0, 1] = float("nan")
        with pytest.raises(KernelNumericalError):
            computer.validate_kernel(K, name="bad", is_train=True)

    def test_inf_raises_hard_error(self, kernel_setup):
        computer, _, _, _, _ = kernel_setup
        K = np.eye(4)
        K[2, 2] = float("inf")
        with pytest.raises(KernelNumericalError):
            computer.validate_kernel(K, name="bad_inf", is_train=True)

    def test_soft_violation_no_modify(self, kernel_setup):
        computer, X_train, _, _, _ = kernel_setup
        K = computer.compute_train_kernel(X_train, use_cache=True)
        K_asym = K.copy()
        K_asym[0, 1] += 0.001   # tiny asymmetry
        K_before = K_asym.copy()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            computer.validate_kernel(K_asym, name="asym", is_train=True)
        assert np.array_equal(K_asym, K_before)


class TestDiagnoseKernel:
    EXPECTED_KEYS = {
        "shape", "is_square", "symmetry_error", "diag_mean", "diag_std",
        "min_value", "max_value", "has_nan", "has_inf",
        "min_eigenvalue", "max_eigenvalue", "condition_number", "is_psd",
    }

    def test_all_keys_present(self, kernel_setup):
        computer, X_train, _, _, _ = kernel_setup
        K = computer.compute_train_kernel(X_train, use_cache=True)
        d = computer.diagnose_kernel(K)
        assert self.EXPECTED_KEYS <= set(d.keys())

    def test_square_detection(self, kernel_setup):
        computer, X_train, X_test, _, _ = kernel_setup
        K_sq   = computer.compute_train_kernel(X_train, use_cache=True)
        K_rect = computer.compute_cross_kernel(X_test, X_train,
                                               name="K_diag", use_cache=False)
        assert computer.diagnose_kernel(K_sq)["is_square"]
        assert not computer.diagnose_kernel(K_rect)["is_square"]

    def test_psd_identity(self, kernel_setup):
        computer, _, _, _, _ = kernel_setup
        d = computer.diagnose_kernel(np.eye(4))
        assert d["is_psd"]
