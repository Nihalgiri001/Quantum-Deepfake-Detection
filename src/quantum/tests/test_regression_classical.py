"""
Regression tests — confirm the existing classical pipeline is untouched.

These tests import the classical pipeline modules and verify that:
    - All 5 classifiers still instantiate, train, and predict correctly.
    - The saved svm.pkl on disk still loads and produces predictions.
    - evaluation helpers compute_metrics and aggregate_video_predictions work.

No quantum code is imported here.
"""
import sys
import numpy as np
import pytest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture(scope="module")
def dummy_data():
    rng = np.random.default_rng(99)
    X_train = rng.standard_normal((60, 64)).astype(np.float32)
    y_train = np.array([0] * 30 + [1] * 30)
    X_test  = rng.standard_normal((20, 64)).astype(np.float32)
    y_test  = np.array([0] * 10 + [1] * 10)
    return X_train, y_train, X_test, y_test


class TestClassicalModels:
    MODELS = ["logistic_regression", "svm", "random_forest", "xgboost", "mlp"]

    @pytest.mark.parametrize("model_name", MODELS)
    def test_instantiate(self, model_name):
        from models import get_model
        m = get_model(model_name, random_seed=42)
        assert m is not None
        assert hasattr(m, "fit")
        assert hasattr(m, "predict")
        assert hasattr(m, "predict_proba")

    @pytest.mark.parametrize("model_name", MODELS)
    def test_fit_predict(self, model_name, dummy_data):
        from models import get_model
        X_train, y_train, X_test, y_test = dummy_data
        m = get_model(model_name, random_seed=42)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        proba = m.predict_proba(X_test)
        assert preds.shape == (20,)
        assert proba.shape == (20, 2)
        assert m.is_fitted

    @pytest.mark.parametrize("model_name", MODELS)
    def test_save_load_round_trip(self, model_name, dummy_data, tmp_path):
        from models import get_model, save_model, load_model
        X_train, y_train, X_test, _ = dummy_data
        m = get_model(model_name, random_seed=42)
        m.fit(X_train, y_train)
        preds_before = m.predict(X_test)

        path = tmp_path / f"{model_name}.pkl"
        save_model(m, path)
        m2 = load_model(path)
        preds_after = m2.predict(X_test)
        assert np.array_equal(preds_before, preds_after)


class TestSavedSVMOnDisk:
    """Verify the SVM trained against the real dataset is still loadable."""

    def test_svm_pkl_loads(self):
        svm_path = _ROOT / "models" / "svm.pkl"
        if not svm_path.exists():
            pytest.skip("models/svm.pkl not present — run train.py first")
        from models import load_model
        m = load_model(svm_path)
        assert m is not None
        assert hasattr(m, "predict")

    def test_svm_predicts_on_synthetic(self):
        svm_path = _ROOT / "models" / "svm.pkl"
        if not svm_path.exists():
            pytest.skip("models/svm.pkl not present — run train.py first")
        from models import load_model
        m = load_model(svm_path)
        pca_path = _ROOT / "models" / "pca.pkl"
        if not pca_path.exists():
            pytest.skip("models/pca.pkl not present")
        import joblib
        pca = joblib.load(pca_path)
        n_components = pca.n_components_
        X_fake = np.random.default_rng(0).standard_normal((5, n_components))
        preds = m.predict(X_fake)
        assert preds.shape == (5,)
        assert set(preds).issubset({0, 1})


class TestEvaluationHelpers:
    def test_compute_metrics(self):
        from evaluate import compute_metrics
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        y_prob = np.array([0.1, 0.6, 0.9, 0.8, 0.3])
        m = compute_metrics(y_true, y_pred, y_prob)
        assert "accuracy" in m
        assert "f1_score" in m
        assert "roc_auc"  in m
        assert 0.0 <= m["accuracy"] <= 1.0

    def test_aggregate_video_predictions(self):
        from evaluate import aggregate_video_predictions
        video_ids = np.array(["v0", "v0", "v1", "v1", "v2"])
        labels    = np.array([0, 0, 1, 1, 1])
        probs     = np.array([0.1, 0.2, 0.8, 0.9, 0.7])
        vid_labels, vid_probs, vid_preds, unique_vids = \
            aggregate_video_predictions(video_ids, labels, probs, threshold=0.5)
        assert len(unique_vids) == 3
        assert vid_labels.shape == (3,)
        assert vid_preds.shape  == (3,)

    def test_available_models_list(self):
        from models import AVAILABLE_MODELS
        assert "svm" in AVAILABLE_MODELS
        assert len(AVAILABLE_MODELS) >= 5
