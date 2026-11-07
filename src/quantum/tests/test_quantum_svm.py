"""
Tests for src/quantum/quantum_svm.py

Covers: fit/predict shapes, proba sums, decision_function, save/load
        identity, class labels, RuntimeError guards, store_train_features.
"""
import warnings
import numpy as np
import pytest

from quantum.quantum_svm import QuantumSVM
from quantum.quantum_config import QuantumConfig


@pytest.fixture
def cfg():
    return QuantumConfig(num_qubits=4, feature_dimension=4, random_seed=42)


@pytest.fixture
def kernels():
    """Pre-built synthetic kernel matrices."""
    rng = np.random.default_rng(7)
    n_tr, n_te = 30, 10
    A = rng.uniform(0, 1, (n_tr, n_tr))
    K_train = (A @ A.T) / n_tr + np.eye(n_tr) * 0.5
    B = rng.uniform(0, 1, (n_te, n_tr))
    K_test  = B / (B.sum(axis=1, keepdims=True) + 1e-8)
    y_train = np.array([0] * 15 + [1] * 15)
    y_test  = np.array([0] * 5  + [1] * 5)
    return K_train, y_train, K_test, y_test


class TestFitPredict:
    def test_predict_shape(self, cfg, kernels):
        K_train, y_train, K_test, _ = kernels
        qsvm = QuantumSVM(config=cfg)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            qsvm.fit(K_train, y_train)
        preds = qsvm.predict(K_test)
        assert preds.shape == (10,)

    def test_predict_proba_shape(self, cfg, kernels):
        K_train, y_train, K_test, _ = kernels
        qsvm = QuantumSVM(config=cfg)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            qsvm.fit(K_train, y_train)
        proba = qsvm.predict_proba(K_test)
        assert proba.shape == (10, 2)

    def test_proba_in_range(self, cfg, kernels):
        K_train, y_train, K_test, _ = kernels
        qsvm = QuantumSVM(config=cfg)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            qsvm.fit(K_train, y_train)
        proba = qsvm.predict_proba(K_test)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_proba_sums_to_one(self, cfg, kernels):
        K_train, y_train, K_test, _ = kernels
        qsvm = QuantumSVM(config=cfg)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            qsvm.fit(K_train, y_train)
        proba = qsvm.predict_proba(K_test)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_decision_function_shape(self, cfg, kernels):
        K_train, y_train, K_test, _ = kernels
        qsvm = QuantumSVM(config=cfg)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            qsvm.fit(K_train, y_train)
        df = qsvm.decision_function(K_test)
        assert df.shape == (10,)

    def test_class_labels(self, cfg, kernels):
        K_train, y_train, _, _ = kernels
        qsvm = QuantumSVM(config=cfg)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            qsvm.fit(K_train, y_train)
        assert list(qsvm.model.classes_) == [0, 1]

    def test_predict_binary_output(self, cfg, kernels):
        K_train, y_train, K_test, _ = kernels
        qsvm = QuantumSVM(config=cfg)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            qsvm.fit(K_train, y_train)
        preds = qsvm.predict(K_test)
        assert set(preds).issubset({0, 1})


class TestGuards:
    def test_predict_before_fit_raises(self, cfg, kernels):
        _, _, K_test, _ = kernels
        qsvm = QuantumSVM(config=cfg)
        with pytest.raises(RuntimeError, match="fitted"):
            qsvm.predict(K_test)

    def test_predict_proba_before_fit_raises(self, cfg, kernels):
        _, _, K_test, _ = kernels
        qsvm = QuantumSVM(config=cfg)
        with pytest.raises(RuntimeError, match="fitted"):
            qsvm.predict_proba(K_test)


class TestSaveLoad:
    def test_reload_identical_predictions(self, cfg, kernels, tmp_path):
        K_train, y_train, K_test, _ = kernels
        qsvm = QuantumSVM(config=cfg)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            qsvm.fit(K_train, y_train)
        qsvm.store_train_features(
            np.random.default_rng(0).uniform(-np.pi, np.pi, (30, 4))
        )
        qsvm.save(str(tmp_path))

        qsvm2 = QuantumSVM.load(str(tmp_path))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            p1 = qsvm.predict(K_test)
            p2 = qsvm2.predict(K_test)
        assert np.array_equal(p1, p2)

    def test_reload_classes_preserved(self, cfg, kernels, tmp_path):
        K_train, y_train, _, _ = kernels
        qsvm = QuantumSVM(config=cfg)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            qsvm.fit(K_train, y_train)
        qsvm.store_train_features(np.zeros((30, 4)))
        qsvm.save(str(tmp_path))

        qsvm2 = QuantumSVM.load(str(tmp_path))
        assert list(qsvm2.model.classes_) == [0, 1]

    def test_artefacts_created(self, cfg, kernels, tmp_path):
        import os
        K_train, y_train, _, _ = kernels
        qsvm = QuantumSVM(config=cfg)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            qsvm.fit(K_train, y_train)
        qsvm.store_train_features(np.zeros((30, 4)))
        qsvm.save(str(tmp_path))
        for fname in ("quantum_svm.pkl", "quantum_config.json",
                      "training_metadata.json", "train_features_reduced.npy"):
            assert os.path.exists(str(tmp_path / fname))

    def test_missing_pkl_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            QuantumSVM.load(str(tmp_path))


class TestSupportVectors:
    def test_get_support_vector_features(self, cfg, kernels):
        K_train, y_train, _, _ = kernels
        X_train_q = np.random.default_rng(0).uniform(-np.pi, np.pi, (30, 4))
        qsvm = QuantumSVM(config=cfg)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            qsvm.fit(K_train, y_train)
        qsvm.store_train_features(X_train_q)
        sv_feats = qsvm.get_support_vector_features()
        assert sv_feats is not None
        assert sv_feats.ndim == 2
        assert sv_feats.shape[1] == 4

    def test_none_before_store(self, cfg, kernels):
        K_train, y_train, _, _ = kernels
        qsvm = QuantumSVM(config=cfg)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            qsvm.fit(K_train, y_train)
        assert qsvm.get_support_vector_features() is None
