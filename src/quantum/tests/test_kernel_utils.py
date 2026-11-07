"""
Tests for src/quantum/kernel_utils.py

Covers: symmetrize, check_psd, kernel_alignment, subsample_balanced,
        hash_kernel_config, save_kernel_matrix, load_kernel_matrix.
"""
import warnings
import numpy as np
import pytest

from quantum.kernel_utils import (
    symmetrize,
    check_psd,
    kernel_alignment,
    subsample_balanced,
    hash_kernel_config,
    save_kernel_matrix,
    load_kernel_matrix,
)


class TestSymmetrize:
    def test_symmetric_result(self):
        K = np.array([[1.0, 0.2, 0.5],
                      [0.3, 1.0, 0.4],
                      [0.6, 0.4, 1.0]])
        S = symmetrize(K)
        assert np.allclose(S, S.T), "Result should be symmetric"

    def test_already_symmetric_unchanged(self):
        K = np.eye(4)
        S = symmetrize(K)
        assert np.allclose(K, S)

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            symmetrize(np.ones((3, 4)))

    def test_large_asymmetry_warns(self):
        K = np.eye(3)
        K[0, 1] = 0.9   # large asymmetry
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            symmetrize(K)
        assert any(issubclass(x.category, RuntimeWarning) for x in w)

    def test_small_asymmetry_no_warn(self):
        K = np.eye(3)
        K[0, 1] = 1e-6; K[1, 0] = 0.0   # below 1e-4 threshold
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            symmetrize(K)
        assert not any(issubclass(x.category, RuntimeWarning) for x in w)


class TestCheckPSD:
    def test_identity_is_psd(self):
        assert check_psd(np.eye(5))

    def test_negative_eigenvalue_not_psd(self):
        K = np.array([[1.0, 2.0], [2.0, 1.0]])   # eigenvalues: -1, 3
        assert not check_psd(K)

    def test_non_square_returns_false(self):
        assert not check_psd(np.ones((3, 4)))

    def test_diagonal_psd(self):
        K = np.diag([1.0, 2.0, 3.0])
        assert check_psd(K)

    def test_tolerance(self):
        K = np.eye(3) * 1e-9 - np.ones((3, 3)) * 1e-10
        assert check_psd(K, tol=1e-8)


class TestKernelAlignment:
    def test_identical_kernels(self):
        K = np.eye(4)
        assert abs(kernel_alignment(K, K) - 1.0) < 1e-9

    def test_orthogonal_like(self):
        K1 = np.diag([1.0, 0.0, 0.0, 0.0])
        K2 = np.diag([0.0, 1.0, 0.0, 0.0])
        assert kernel_alignment(K1, K2) == 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            kernel_alignment(np.eye(3), np.eye(4))

    def test_zero_matrix_returns_zero(self):
        assert kernel_alignment(np.zeros((3, 3)), np.zeros((3, 3))) == 0.0


class TestSubsampleBalanced:
    def test_output_shapes(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 10))
        y = np.array([0] * 50 + [1] * 50)
        X_s, y_s, idx = subsample_balanced(X, y, n=20, seed=7)
        assert X_s.shape == (20, 10)
        assert y_s.shape == (20,)
        assert idx.shape == (20,)

    def test_class_balance(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((80, 5))
        y = np.array([0] * 40 + [1] * 40)
        _, y_s, _ = subsample_balanced(X, y, n=20, seed=0)
        assert abs((y_s == 0).sum() - (y_s == 1).sum()) <= 1

    def test_indices_valid(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 4))
        y = np.array([0] * 25 + [1] * 25)
        X_s, y_s, idx = subsample_balanced(X, y, n=10, seed=1)
        assert np.all(idx >= 0) and np.all(idx < 50)
        assert np.allclose(X_s, X[idx])
        assert np.array_equal(y_s, y[idx])

    def test_cap_at_available(self):
        X = np.eye(6)
        y = np.array([0, 0, 0, 1, 1, 1])
        X_s, y_s, idx = subsample_balanced(X, y, n=100, seed=0)
        assert len(y_s) <= 6


class TestHashKernelConfig:
    def test_same_config_same_hash(self):
        cfg = {"a": 1, "b": "linear"}
        assert hash_kernel_config(cfg) == hash_kernel_config(cfg)

    def test_different_config_different_hash(self):
        assert hash_kernel_config({"a": 1}) != hash_kernel_config({"a": 2})

    def test_key_order_irrelevant(self):
        h1 = hash_kernel_config({"a": 1, "b": 2})
        h2 = hash_kernel_config({"b": 2, "a": 1})
        assert h1 == h2

    def test_returns_8_char_hex(self):
        h = hash_kernel_config({"x": 42})
        assert len(h) == 8
        assert all(c in "0123456789abcdef" for c in h)


class TestSaveLoadKernelMatrix:
    def test_round_trip(self, tmp_path):
        K = np.eye(5) * 2.5
        path = str(tmp_path / "K.npy")
        save_kernel_matrix(K, path, metadata={"test": True})
        K2, meta = load_kernel_matrix(path)
        assert np.array_equal(K, K2)
        assert meta["test"] is True

    def test_no_metadata(self, tmp_path):
        K = np.ones((3, 3))
        path = str(tmp_path / "K_no_meta.npy")
        save_kernel_matrix(K, path)
        K2, meta = load_kernel_matrix(path)
        assert np.array_equal(K, K2)
        assert meta is None

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_kernel_matrix(str(tmp_path / "nonexistent.npy"))

    def test_creates_parent_dir(self, tmp_path):
        path = str(tmp_path / "subdir" / "deep" / "K.npy")
        save_kernel_matrix(np.eye(2), path)
        import os; assert os.path.exists(path)
