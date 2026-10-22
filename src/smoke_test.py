"""
Lightweight Smoke Test Script
=============================
Verifies that all 5 classical classifiers (Logistic Regression, RBF-SVM,
Random Forest, XGBoost, MLP) can be instantiated, trained on a small synthetic dataset,
predict probabilities, save/load models, and compute evaluation metrics without error.

Runs in ~2–3 seconds and DOES NOT execute on the full FF++ dataset.
"""

import sys
import tempfile
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


if __name__ == "__main__":
    run_smoke_test()
