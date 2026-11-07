"""
Tests for src/quantum/feature_reducer.py

Covers: fit/transform shapes, range clipping, save/load identity,
        double-fit guard, transform-before-fit guard.
"""
import math
import numpy as np
import pytest

from quantum.feature_reducer import QuantumFeatureReducer
from quantum.quantum_config import QuantumConfig


@pytest.fixture
def cfg4():
    return QuantumConfig(num_qubits=4, feature_dimension=4, random_seed=42)


@pytest.fixture
def raw_data():
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((40, 512)).astype(np.float32)
    X_test  = rng.standard_normal((10, 512)).astype(np.float32)
    return X_train, X_test


class TestFitTransform:
    def test_output_shape(self, cfg4, raw_data):
        X_train, X_test = raw_data
        r = QuantumFeatureReducer(config=cfg4)
        r.fit(X_train)
        out = r.transform(X_test)
        assert out.shape == (10, 4)

    def test_output_range(self, cfg4, raw_data):
        X_train, _ = raw_data
        r = QuantumFeatureReducer(config=cfg4)
        X_q = r.fit_transform(X_train)
        # MinMaxScaler can produce values ε outside [-π, π] due to float32
        # rounding; allow a generous tolerance (values are ≪ 1e-4 outside bound).
        tol = 1e-4
        assert X_q.min() >= -math.pi - tol, f"min {X_q.min()} < -pi - {tol}"
        assert X_q.max() <=  math.pi + tol, f"max {X_q.max()} > pi + {tol}"

    def test_fit_transform_equivalent(self, cfg4, raw_data):
        X_train, _ = raw_data
        r1 = QuantumFeatureReducer(config=cfg4)
        X1 = r1.fit_transform(X_train)

        r2 = QuantumFeatureReducer(config=cfg4)
        r2.fit(X_train)
        X2 = r2.transform(X_train)
        assert np.allclose(X1, X2)

    def test_dtype_float64(self, cfg4, raw_data):
        X_train, _ = raw_data
        r = QuantumFeatureReducer(config=cfg4)
        X_q = r.fit_transform(X_train)
        assert X_q.dtype == np.float64


class TestGuards:
    def test_transform_before_fit_raises(self, cfg4, raw_data):
        _, X_test = raw_data
        r = QuantumFeatureReducer(config=cfg4)
        with pytest.raises(RuntimeError, match="fitted"):
            r.transform(X_test)

    def test_double_fit_raises_without_force(self, cfg4, raw_data):
        X_train, _ = raw_data
        r = QuantumFeatureReducer(config=cfg4)
        r.fit(X_train)
        with pytest.raises(RuntimeError, match="already fitted"):
            r.fit(X_train)

    def test_double_fit_allowed_with_force(self, cfg4, raw_data):
        X_train, _ = raw_data
        r = QuantumFeatureReducer(config=cfg4)
        r.fit(X_train)
        r.fit(X_train, force=True)   # should not raise
        assert r._fitted

    def test_save_before_fit_raises(self, cfg4, tmp_path):
        r = QuantumFeatureReducer(config=cfg4)
        with pytest.raises(RuntimeError):
            r.save(str(tmp_path))


class TestSaveLoad:
    def test_reload_identical_output(self, cfg4, raw_data, tmp_path):
        X_train, X_test = raw_data
        r = QuantumFeatureReducer(config=cfg4)
        X_q1 = r.fit_transform(X_train)
        r.save(str(tmp_path))

        r2 = QuantumFeatureReducer.load(str(tmp_path))
        X_q2 = r2.transform(X_train)
        assert np.allclose(X_q1, X_q2, atol=1e-9)

    def test_reload_transform_test_identical(self, cfg4, raw_data, tmp_path):
        X_train, X_test = raw_data
        r = QuantumFeatureReducer(config=cfg4)
        r.fit(X_train)
        out1 = r.transform(X_test)
        r.save(str(tmp_path))

        r2 = QuantumFeatureReducer.load(str(tmp_path))
        out2 = r2.transform(X_test)
        assert np.allclose(out1, out2, atol=1e-9)

    def test_saved_artefacts_exist(self, cfg4, raw_data, tmp_path):
        import os
        X_train, _ = raw_data
        r = QuantumFeatureReducer(config=cfg4)
        r.fit(X_train)
        r.save(str(tmp_path))
        for fname in ("scaler.pkl", "pca.pkl", "feature_scaler.pkl",
                      "feature_scaling.json"):
            assert os.path.exists(str(tmp_path / fname))

    def test_fit_metadata_populated(self, cfg4, raw_data):
        X_train, _ = raw_data
        r = QuantumFeatureReducer(config=cfg4)
        r.fit(X_train)
        meta = r._fit_metadata
        assert meta["n_train_samples"] == len(X_train)
        assert meta["pca_components"] == 4
        assert "cumulative_explained_variance" in meta
