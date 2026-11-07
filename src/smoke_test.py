"""
Lightweight Smoke Test Script
=============================
Verifies that all 5 classical classifiers (Logistic Regression, RBF-SVM,
Random Forest, XGBoost, MLP) can be instantiated, trained on a small synthetic dataset,
predict probabilities, save/load models, and compute evaluation metrics without error.

Runs in ~2–3 seconds and DOES NOT execute on the full FF++ dataset.
"""

import sys
import argparse
import tempfile
import warnings
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import (
    get_model,
    save_model,
    load_model,
    AVAILABLE_MODELS,
)
from evaluate import compute_metrics
from utils import setup_logger

logger = setup_logger("smoke_test")


def run_smoke_test():
    print()
    print("=" * 60)
    print("  Classical Deepfake Detector — Lightweight Smoke Test")
    print("=" * 60)

    # 1. Generate small dummy synthetic feature arrays (50 samples, 64 features)
    np.random.seed(42)
    X_dummy = np.random.randn(50, 64).astype(np.float32)
    y_dummy = np.random.choice([0, 1], size=50)

    X_test_dummy = np.random.randn(20, 64).astype(np.float32)
    y_test_dummy = np.random.choice([0, 1], size=20)

    print(f"  Dummy Synthetic Train Set: {X_dummy.shape}")
    print(f"  Dummy Synthetic Test Set:  {X_test_dummy.shape}")
    print("─" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        for model_name in AVAILABLE_MODELS:
            print(f"  Testing classifier: [{model_name}] ... ", end="", flush=True)

            # Instantiation
            model = get_model(model_name, random_seed=42)

            # Training
            model.fit(X_dummy, y_dummy)
            assert model.is_fitted, f"[{model_name}] Model failed fit flag."

            # Prediction
            preds = model.predict(X_test_dummy)
            probs = model.predict_proba(X_test_dummy)

            assert preds.shape == (20,), f"[{model_name}] Invalid predict shape: {preds.shape}"
            assert probs.shape == (20, 2), f"[{model_name}] Invalid predict_proba shape: {probs.shape}"

            # Metrics
            metrics = compute_metrics(y_test_dummy, preds, probs[:, 1])
            assert "accuracy" in metrics, f"[{model_name}] Missing accuracy metric."

            # Serialization (Save & Load)
            save_file = tmp_path / f"{model_name}.pkl"
            save_model(model, save_file)
            assert save_file.exists(), f"[{model_name}] Save file not found."

            reloaded_model = load_model(save_file)
            reloaded_preds = reloaded_model.predict(X_test_dummy)
            assert np.array_equal(preds, reloaded_preds), f"[{model_name}] Reloaded predictions mismatch."

            print("PASSED ✅")

    print("─" * 60)
    print("  ALL 5 CLASSIFIERS PASSED SMOKE TEST SUCCESSFULLY! ✅")
    print("=" * 60)
    print()



# ── Quantum smoke test (additive — only runs with --quantum flag) ─────────────

def run_quantum_smoke_test():
    """Run the full quantum pipeline on synthetic 512-dim data.

    10 steps verified:
      1. check_install assertions
      2. QuantumFeatureReducer.fit_transform (60 samples, 512-dim)
      3. 30×30 kernel matrix via simulator
      4. QuantumSVM.fit()
      5. QuantumSVM.predict() / predict_proba()
      6. Frame-level metrics (compute_metrics)
      7. Video-level aggregation (aggregate_video_predictions)
      8. Save all artefacts to tempdir
      9. Reload artefacts; verify identical predictions
     10. Classical regression check: get_model("svm") still works
    """
    import time
    warnings.filterwarnings("ignore", category=FutureWarning)

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    print()
    print("=" * 60)
    print("  TrustLens-Q — Quantum Smoke Test")
    print("=" * 60)

    def _step(n, desc):
        print(f"  Step {n:2d}: {desc} ... ", end="", flush=True)

    def _ok():
        print("PASSED ✅")

    def _fail(exc):
        print(f"FAILED ❌\n  Error: {exc}")
        raise

    t_start = time.time()

    # ── Step 1: check_install ─────────────────────────────────────────────────
    _step(1, "Qiskit installation check")
    try:
        from quantum.check_install import run_checks
        assert run_checks(verbose=False), "check_install returned False"
        _ok()
    except Exception as exc:
        _fail(exc)

    # ── Step 2: QuantumFeatureReducer ─────────────────────────────────────────
    _step(2, "QuantumFeatureReducer.fit_transform (60 samples, 512-dim)")
    try:
        from quantum.feature_reducer import QuantumFeatureReducer
        from quantum.quantum_config import QuantumConfig

        rng = np.random.default_rng(42)
        n_tr, n_v, n_te = 30, 15, 15
        cfg = QuantumConfig(num_qubits=4, feature_dimension=4, shots=1024,
                            random_seed=42, max_train_samples=30,
                            max_val_samples=15, max_test_samples=15)

        X_train = rng.standard_normal((n_tr, 512)).astype(np.float32)
        y_train = np.array([0] * 15 + [1] * 15)
        X_val   = rng.standard_normal((n_v, 512)).astype(np.float32)
        y_val   = np.array([0] * 7 + [1] * 8)
        X_test  = rng.standard_normal((n_te, 512)).astype(np.float32)
        y_test  = np.array([0] * 7 + [1] * 8)

        reducer = QuantumFeatureReducer(config=cfg)
        X_train_q = reducer.fit_transform(X_train)
        X_val_q   = reducer.transform(X_val)
        X_test_q  = reducer.transform(X_test)

        assert X_train_q.shape == (n_tr, 4)
        assert X_val_q.shape   == (n_v,  4)
        assert X_test_q.shape  == (n_te, 4)
        _ok()
    except Exception as exc:
        _fail(exc)

    # ── Step 3: 30×30 kernel matrix ───────────────────────────────────────────
    _step(3, "30×30 training kernel via simulator")
    try:
        from quantum.quantum_kernel import QuantumKernelComputer
        from quantum.simulator_backend import SimulatorBackend

        with tempfile.TemporaryDirectory() as kd:
            backend  = SimulatorBackend(shots=1024, seed=42)
            computer = QuantumKernelComputer(config=cfg, backend=backend,
                                             cache_dir=kd)
            K_train = computer.compute_train_kernel(X_train_q, use_cache=True)
            K_val   = computer.compute_cross_kernel(X_val_q,   X_train_q,
                                                    name="K_val",   use_cache=True)
            K_test  = computer.compute_cross_kernel(X_test_q,  X_train_q,
                                                    name="K_test",  use_cache=True)

        assert K_train.shape == (n_tr, n_tr)
        assert K_val.shape   == (n_v,  n_tr)
        assert K_test.shape  == (n_te, n_tr)
        assert not np.isnan(K_train).any()
        _ok()
    except Exception as exc:
        _fail(exc)

    # ── Step 4: QuantumSVM.fit() ──────────────────────────────────────────────
    _step(4, "QuantumSVM.fit(K_train, y_train)")
    try:
        from quantum.quantum_svm import QuantumSVM
        qsvm = QuantumSVM(config=cfg)
        qsvm.fit(K_train, y_train)
        assert qsvm.is_fitted
        _ok()
    except Exception as exc:
        _fail(exc)

    # ── Step 5: predict() + predict_proba() ───────────────────────────────────
    _step(5, "QuantumSVM.predict() + predict_proba()")
    try:
        preds  = qsvm.predict(K_test)
        probas = qsvm.predict_proba(K_test)
        assert preds.shape  == (n_te,)
        assert probas.shape == (n_te, 2)
        assert set(preds).issubset({0, 1})
        assert np.all((probas >= 0) & (probas <= 1))
        assert np.allclose(probas.sum(axis=1), 1.0)
        _ok()
    except Exception as exc:
        _fail(exc)

    # ── Step 6: frame-level metrics ───────────────────────────────────────────
    _step(6, "Frame-level metrics (compute_metrics)")
    try:
        from evaluate import compute_metrics
        fake_probs = probas[:, 1]
        metrics = compute_metrics(y_test, preds, fake_probs)
        for key in ("accuracy", "precision", "recall", "f1_score"):
            assert key in metrics
            assert metrics[key] is None or 0.0 <= metrics[key] <= 1.0
        _ok()
    except Exception as exc:
        _fail(exc)

    # ── Step 7: video-level aggregation ───────────────────────────────────────
    _step(7, "Video-level aggregation (aggregate_video_predictions)")
    try:
        from evaluate import aggregate_video_predictions
        vid_ids = np.array([f"v{i}" for i in range(n_te)])
        vid_labels, vid_probs, vid_preds, unique_vids =             aggregate_video_predictions(vid_ids, y_test, fake_probs, threshold=0.5)
        assert len(unique_vids) == n_te          # 1 frame per video
        assert vid_labels.shape == (n_te,)
        _ok()
    except Exception as exc:
        _fail(exc)

    # ── Step 8: save all artefacts ────────────────────────────────────────────
    _step(8, "Save all artefacts to tempdir")
    import os
    tmp_dir = tempfile.mkdtemp(prefix="trustlens_q_smoke_")
    try:
        md = Path(tmp_dir) / "models" / "quantum"
        reducer.save(str(md))
        qsvm.store_train_features(X_train_q)
        qsvm.save(str(md))
        cfg.save_json(str(md / "quantum_config.json"))

        for fname in ("scaler.pkl", "pca.pkl", "feature_scaler.pkl",
                      "quantum_svm.pkl", "train_features_reduced.npy"):
            assert os.path.exists(str(md / fname)), f"Missing: {fname}"
        _ok()
    except Exception as exc:
        import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)
        _fail(exc)

    # ── Step 9: reload artefacts; verify identical predictions ────────────────
    _step(9, "Reload artefacts + verify identical predictions")
    try:
        qsvm2    = QuantumSVM.load(str(md))
        reducer2 = QuantumFeatureReducer.load(str(md))
        X_test_q2 = reducer2.transform(X_test)
        assert np.allclose(X_test_q, X_test_q2, atol=1e-9)
        preds2 = qsvm2.predict(K_test)
        assert np.array_equal(preds, preds2), "Reloaded predictions differ"
        _ok()
    except Exception as exc:
        _fail(exc)
    finally:
        import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Step 10: classical regression — get_model("svm") still works ──────────
    _step(10, "Classical regression: get_model('svm') still works")
    try:
        from models import get_model, AVAILABLE_MODELS
        assert "svm" in AVAILABLE_MODELS
        classical_svm = get_model("svm", random_seed=42)
        X_cls = np.random.default_rng(0).standard_normal((40, 64)).astype(np.float32)
        y_cls = np.array([0] * 20 + [1] * 20)
        classical_svm.fit(X_cls, y_cls)
        p = classical_svm.predict(X_cls[:5])
        assert p.shape == (5,)
        _ok()
    except Exception as exc:
        _fail(exc)

    elapsed = time.time() - t_start
    print()
    print(f"  Total time: {elapsed:.1f}s  (limit: 120s)")
    assert elapsed < 120, f"Quantum smoke test too slow: {elapsed:.1f}s"
    print()
    print("  ✅  ALL QUANTUM STEPS PASSED")
    print("=" * 60)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TrustLens-Q smoke tests."
    )
    parser.add_argument(
        "--quantum", action="store_true",
        help="Also run the quantum pipeline smoke test."
    )
    args = parser.parse_args()

    run_smoke_test()

    if args.quantum:
        run_quantum_smoke_test()

