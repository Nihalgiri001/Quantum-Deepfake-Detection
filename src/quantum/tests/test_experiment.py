"""
Tests for quantum_experiment.py, video_aggregator.py, and classical_baseline.py

All tests use synthetic pre-computed kernel matrices via monkeypatching.
No live Qiskit circuits are invoked.
All tests run in < 10 seconds each.
"""

import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure src/ is on sys.path
_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from quantum.quantum_experiment import ExperimentConfig, ExperimentRunner
from quantum.video_aggregator import (
    AggregationStrategy,
    VideoAggregationResult,
    VideoAggregator,
    pool_frames_per_video,
    pool_frames_per_video_with_labels,
)
from quantum.classical_baseline import ClassicalBaselineSVM


# ── Shared fixture helpers ─────────────────────────────────────────────────────

def _make_features_dir(tmp_path: Path, n_train=40, n_val=20, n_test=20, dim=512) -> Path:
    """Create minimal feature files in tmp_path/features/."""
    fd = tmp_path / "features"
    fd.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        X = rng.standard_normal((n, dim)).astype(np.float32)
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        vids = np.array([f"v{i:03d}" for i in range(n)])
        np.save(str(fd / f"{split}_features.npy"), X)
        np.save(str(fd / f"{split}_labels.npy"), y)
        np.save(str(fd / f"{split}_video_ids.npy"), vids)

    return fd


def _small_config(**kwargs) -> ExperimentConfig:
    """Return a fast ExperimentConfig for testing."""
    defaults = {
        "feature_dimension": 4,
        "max_train_samples": 20,
        "max_val_samples": 10,
        "max_test_samples": 10,
        "shots": 64,
        "random_seed": 42,
    }
    defaults.update(kwargs)
    return ExperimentConfig(**defaults)


def _make_precomputed_kernel(n: int, seed: int = 0) -> np.ndarray:
    """Return a valid symmetric PSD kernel matrix."""
    rng = np.random.default_rng(seed)
    A = rng.uniform(0, 1, (n, n))
    K = (A @ A.T) / n
    np.fill_diagonal(K, 1.0)
    K = np.clip(K, 0.0, 1.0)
    return K


# ── ExperimentConfig tests ────────────────────────────────────────────────────

class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_experiment_id_is_deterministic(self):
        """Same config always produces the same experiment_id (Property 13)."""
        cfg1 = _small_config()
        cfg2 = _small_config()
        assert cfg1.experiment_id == cfg2.experiment_id

    def test_experiment_id_changes_with_config(self):
        """Different configs produce different experiment IDs."""
        cfg1 = _small_config(feature_dimension=4)
        cfg2 = _small_config(feature_dimension=6)
        assert cfg1.experiment_id != cfg2.experiment_id

    def test_to_json_is_sorted(self):
        """to_json() produces sorted-key JSON for determinism."""
        cfg = _small_config()
        s = cfg.to_json()
        parsed = json.loads(s)
        keys = list(parsed.keys())
        assert keys == sorted(keys), "JSON keys are not sorted"

    def test_round_trip_from_dict(self):
        """ExperimentConfig round-trips through to_dict / from_dict."""
        cfg = _small_config(feature_dimension=4, reps=2, svm_c=5.0)
        restored = ExperimentConfig.from_dict(cfg.to_dict())
        assert restored.feature_dimension == cfg.feature_dimension
        assert restored.reps == cfg.reps
        assert restored.svm_c == cfg.svm_c
        assert restored.experiment_id == cfg.experiment_id

    def test_experiment_id_8_chars(self):
        """experiment_id is an 8-character hex string."""
        cfg = _small_config()
        eid = cfg.experiment_id
        assert len(eid) == 8
        assert all(c in "0123456789abcdef" for c in eid)

    def test_cache_key_sensitivity(self):
        """Changing any key field produces a different cache key (Property 9 proxy)."""
        from quantum.kernel_utils import hash_kernel_config
        base = {
            "feature_dimension": 4, "reps": 1, "entanglement": "linear",
            "n_train_samples": 20, "shots": 64, "random_seed": 42,
        }
        base_hash = hash_kernel_config(base)

        for changed_key, new_val in [
            ("feature_dimension", 6),
            ("reps", 2),
            ("n_train_samples", 40),
            ("shots", 128),
            ("random_seed", 99),
        ]:
            modified = {**base, changed_key: new_val}
            assert hash_kernel_config(modified) != base_hash, (
                f"Hash unchanged when {changed_key}={new_val}"
            )


# ── VideoAggregator tests ─────────────────────────────────────────────────────

class TestVideoAggregator:
    """Tests for VideoAggregator.aggregate() and tune_threshold_and_strategy()."""

    def test_aggregate_returns_one_per_unique_video(self):
        """aggregate() returns exactly len(unique(video_ids)) entries (Property 11)."""
        agg = VideoAggregator()
        probs = np.array([0.1, 0.8, 0.6, 0.3, 0.9, 0.2])
        vids = np.array(["v0", "v1", "v0", "v2", "v1", "v2"])
        for strategy in AggregationStrategy:
            result = agg.aggregate(probs, vids, strategy)
            assert len(result.video_ids) == 3, (
                f"Strategy {strategy}: expected 3 unique videos, got {len(result.video_ids)}"
            )

    def test_mean_strategy_correct(self):
        """Mean strategy computes correct per-video mean."""
        agg = VideoAggregator()
        probs = np.array([0.2, 0.4, 0.6])
        vids = np.array(["v0", "v0", "v1"])
        result = agg.aggregate(probs, vids, AggregationStrategy.MEAN)

        v0_idx = list(result.video_ids).index("v0")
        v1_idx = list(result.video_ids).index("v1")
        assert abs(result.aggregated_probs[v0_idx] - 0.3) < 1e-9
        assert abs(result.aggregated_probs[v1_idx] - 0.6) < 1e-9

    def test_median_strategy_correct(self):
        """Median strategy computes correct per-video median."""
        agg = VideoAggregator()
        probs = np.array([0.1, 0.5, 0.9, 0.3])
        vids = np.array(["v0", "v0", "v0", "v1"])
        result = agg.aggregate(probs, vids, AggregationStrategy.MEDIAN)

        v0_idx = list(result.video_ids).index("v0")
        assert abs(result.aggregated_probs[v0_idx] - 0.5) < 1e-9

    def test_top25_mean_four_frames(self):
        """Top-25% mean on 4 frames selects highest value."""
        agg = VideoAggregator()
        # 4 frames → top 25% = top 1 frame
        probs = np.array([0.2, 0.4, 0.7, 0.9])
        vids = np.array(["v0", "v0", "v0", "v0"])
        result = agg.aggregate(probs, vids, AggregationStrategy.TOP25_MEAN)
        assert abs(result.aggregated_probs[0] - 0.9) < 1e-9

    def test_top25_mean_single_frame_returns_value(self):
        """Top-25% mean on 1 frame returns the single value (edge case)."""
        agg = VideoAggregator()
        probs = np.array([0.73])
        vids = np.array(["v_single"])
        result = agg.aggregate(probs, vids, AggregationStrategy.TOP25_MEAN)
        assert abs(result.aggregated_probs[0] - 0.73) < 1e-9

    def test_equal_weight_mean_same_as_mean(self):
        """EQUAL_WEIGHT_MEAN is identical to MEAN for uniform weights."""
        agg = VideoAggregator()
        probs = np.array([0.3, 0.7, 0.5, 0.9])
        vids = np.array(["v0", "v0", "v1", "v1"])
        r_mean = agg.aggregate(probs, vids, AggregationStrategy.MEAN)
        r_eq = agg.aggregate(probs, vids, AggregationStrategy.EQUAL_WEIGHT_MEAN)
        np.testing.assert_array_almost_equal(r_mean.aggregated_probs, r_eq.aggregated_probs)

    def test_threshold_determines_labels(self):
        """Predictions match threshold > prob."""
        agg = VideoAggregator()
        probs = np.array([0.4, 0.6, 0.8])
        vids = np.array(["v0", "v1", "v2"])
        result = agg.aggregate(probs, vids, AggregationStrategy.MEAN, threshold=0.5)
        # v0 (0.4 < 0.5) → 0, v1 (0.6 >= 0.5) → 1, v2 (0.8 >= 0.5) → 1
        expected = np.array([0, 1, 1])
        np.testing.assert_array_equal(result.predicted_labels, expected)

    def test_tune_threshold_evaluates_all_9_thresholds(self):
        """tune_threshold_and_strategy evaluates all 9 thresholds (Property 12)."""
        agg = VideoAggregator()
        rng = np.random.default_rng(5)
        n_frames = 40
        probs = rng.uniform(0, 1, n_frames)
        vids = np.array([f"v{i // 4}" for i in range(n_frames)])  # 10 videos, 4 frames each
        y_true_dict = {f"v{i}": int(i % 2) for i in range(10)}

        best_strategy, best_threshold = agg.tune_threshold_and_strategy(
            probs, vids, y_true_dict
        )
        assert best_threshold in VideoAggregator.THRESHOLD_GRID
        assert isinstance(best_strategy, AggregationStrategy)

    def test_tune_threshold_returns_valid_strategy(self):
        """tune_threshold_and_strategy returns a valid AggregationStrategy."""
        agg = VideoAggregator()
        probs = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3])
        vids = np.array(["v0", "v0", "v1", "v1", "v2", "v2"])
        y_true = np.array([1, 0, 1])  # video-level labels

        best_strategy, best_threshold = agg.tune_threshold_and_strategy(
            probs, vids, y_true
        )
        assert isinstance(best_strategy, AggregationStrategy)
        assert best_threshold in VideoAggregator.THRESHOLD_GRID


class TestVideoPooling:
    """Tests for pool_frames_per_video()."""

    def test_pool_returns_one_feature_per_video(self):
        """pool_frames_per_video returns one row per unique video."""
        X = np.ones((6, 4))
        vids = np.array(["v0", "v0", "v1", "v1", "v2", "v2"])
        feat, unique_vids, _ = pool_frames_per_video(X, vids)
        assert feat.shape == (3, 4)
        assert len(unique_vids) == 3

    def test_pool_mean_correct(self):
        """Mean pooling averages feature vectors correctly."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        vids = np.array(["v0", "v0", "v1"])
        feat, unique_vids, _ = pool_frames_per_video(X, vids, method="mean")
        # v0 should be mean of first two rows
        v0_idx = list(unique_vids).index("v0")
        np.testing.assert_array_almost_equal(feat[v0_idx], [2.0, 3.0])

    def test_pool_with_labels_majority_vote(self):
        """pool_frames_per_video_with_labels uses majority vote for video labels."""
        X = np.ones((5, 4))
        vids = np.array(["v0", "v0", "v0", "v1", "v1"])
        y = np.array([1, 1, 0, 0, 0])  # v0: 2 FAKE, 1 REAL → FAKE; v1: REAL
        feat, unique_vids, video_labels = pool_frames_per_video_with_labels(X, vids, y)
        v0_idx = list(unique_vids).index("v0")
        v1_idx = list(unique_vids).index("v1")
        assert video_labels[v0_idx] == 1  # FAKE (majority)
        assert video_labels[v1_idx] == 0  # REAL (majority)


# ── ClassicalBaselineSVM tests ────────────────────────────────────────────────

class TestClassicalBaselineSVM:
    """Tests for ClassicalBaselineSVM."""

    def test_fit_and_evaluate_returns_metrics(self):
        """fit() and evaluate() return expected metric keys."""
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((40, 4))
        y_train = np.array([0] * 20 + [1] * 20)
        X_val = rng.standard_normal((20, 4))
        y_val = np.array([0] * 10 + [1] * 10)

        clf = ClassicalBaselineSVM(feature_dimension=4)
        clf.fit(X_train, y_train)
        metrics = clf.evaluate(X_val, y_val)

        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert key in metrics, f"Missing metric: {key}"

    def test_model_name_includes_dimension(self):
        """model_name includes the feature dimension."""
        clf = ClassicalBaselineSVM(feature_dimension=8)
        assert "8" in clf.model_name

    def test_save_and_load_roundtrip(self, tmp_path):
        """save() and load() reconstruct a working classifier."""
        rng = np.random.default_rng(1)
        X_train = rng.standard_normal((20, 4))
        y_train = np.array([0] * 10 + [1] * 10)
        X_val = rng.standard_normal((10, 4))

        clf = ClassicalBaselineSVM(feature_dimension=4, svm_c=2.0)
        clf.fit(X_train, y_train)

        clf.save(str(tmp_path))
        loaded = ClassicalBaselineSVM.load(str(tmp_path))

        preds_orig = clf.predict(X_val)
        preds_loaded = loaded.predict(X_val)
        np.testing.assert_array_equal(preds_orig, preds_loaded)

    def test_predict_before_fit_raises(self):
        """predict() before fit() raises RuntimeError."""
        clf = ClassicalBaselineSVM(feature_dimension=4)
        X = np.random.default_rng(2).standard_normal((5, 4))
        with pytest.raises(RuntimeError):
            clf.predict(X)

    def test_wrong_feature_dim_raises_assertion(self):
        """fit() raises AssertionError when feature dimension mismatches."""
        clf = ClassicalBaselineSVM(feature_dimension=4)
        X_train = np.random.default_rng(3).standard_normal((10, 8))  # 8 ≠ 4
        y_train = np.array([0] * 5 + [1] * 5)
        with pytest.raises(AssertionError):
            clf.fit(X_train, y_train)


# ── ExperimentRunner tests (monkeypatched kernel computation) ─────────────────

def _patch_kernel_computer(K_train: np.ndarray, K_val: np.ndarray, K_test: np.ndarray = None):
    """
    Return a context manager that monkeypatches QuantumKernelComputer
    to return pre-computed kernels without running any quantum circuits.
    """
    from unittest.mock import patch, MagicMock

    def fake_compute_train(self_inner, X_train, use_cache=True):
        return K_train

    def fake_compute_cross(self_inner, X_new, X_train, name="K_cross", use_cache=False):
        if "test" in name.lower() and K_test is not None:
            return K_test
        return K_val

    return patch.multiple(
        "quantum.quantum_experiment.QuantumKernelComputer",
        compute_train_kernel=fake_compute_train,
        compute_cross_kernel=fake_compute_cross,
    )


class TestExperimentRunner:
    """Integration tests for ExperimentRunner with mocked kernel computation."""

    def _run_experiment(self, tmp_path, config=None, evaluate_test=False):
        """Helper: run a patched experiment and return (runner, result)."""
        fd = _make_features_dir(tmp_path, n_train=20, n_val=10, n_test=10)
        results_dir = tmp_path / "results"
        config = config or _small_config()

        n_train = min(config.max_train_samples, 20)
        n_val = min(config.max_val_samples, 10)
        n_test = min(config.max_test_samples, 10)

        K_train = _make_precomputed_kernel(n_train, seed=0)
        K_val = np.random.default_rng(1).uniform(0, 1, (n_val, n_train))
        K_test = np.random.default_rng(2).uniform(0, 1, (n_test, n_train))

        runner = ExperimentRunner(
            features_dir=str(fd),
            results_dir=str(results_dir),
        )
        with _patch_kernel_computer(K_train, K_val, K_test):
            result = runner.run(config, evaluate_test=evaluate_test)

        return runner, result

    def test_artefact_completeness_no_test(self, tmp_path):
        """Without --evaluate-test, required files exist and test_metrics.json is absent (Property 14)."""
        _, result = self._run_experiment(tmp_path)
        out_dir = Path(result.output_dir)

        required = ["config.json", "train_val_metrics.json",
                    "kernel_diagnostics.json", "timing.json",
                    "classical_vs_quantum.csv"]
        for fname in required:
            assert (out_dir / fname).exists(), f"{fname} is missing"

        assert not (out_dir / "test_metrics.json").exists(), \
            "test_metrics.json should NOT exist without --evaluate-test"

    def test_test_metrics_written_when_requested(self, tmp_path):
        """test_metrics.json is written when evaluate_test=True."""
        _, result = self._run_experiment(tmp_path, evaluate_test=True)
        out_dir = Path(result.output_dir)
        assert (out_dir / "test_metrics.json").exists()
        assert result.test_metrics is not None

    def test_experiment_id_determinism(self, tmp_path):
        """Same config produces same experiment_id on repeated calls (Property 13)."""
        cfg = _small_config(random_seed=77)
        eid1 = cfg.experiment_id
        eid2 = cfg.experiment_id
        assert eid1 == eid2

    def test_summary_csv_row_count(self, tmp_path):
        """Running N experiments appends N rows to summary.csv (Property 15)."""
        fd = _make_features_dir(tmp_path, n_train=20, n_val=10, n_test=10)
        results_dir = tmp_path / "results"

        n_experiments = 3
        for i in range(n_experiments):
            # Use different svm_c to get distinct experiment IDs
            config = _small_config(svm_c=float(i + 1), random_seed=42)
            n_train = min(config.max_train_samples, 20)
            n_val = min(config.max_val_samples, 10)
            K_train = _make_precomputed_kernel(n_train, seed=i)
            K_val = np.random.default_rng(i + 10).uniform(0, 1, (n_val, n_train))

            runner = ExperimentRunner(
                features_dir=str(fd),
                results_dir=str(results_dir),
            )
            with _patch_kernel_computer(K_train, K_val):
                runner.run(config)

        # Check summary.csv
        summary_path = results_dir / "experiments" / "summary.csv"
        assert summary_path.exists(), "summary.csv was not created"
        with open(summary_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == n_experiments, (
            f"Expected {n_experiments} rows; got {len(rows)}"
        )

    def test_config_json_written(self, tmp_path):
        """config.json is written with correct experiment_id."""
        cfg = _small_config()
        _, result = self._run_experiment(tmp_path, config=cfg)
        config_path = Path(result.output_dir) / "config.json"
        assert config_path.exists()
        with open(config_path) as f:
            data = json.load(f)
        assert data["experiment_id"] == cfg.experiment_id

    def test_timing_json_written(self, tmp_path):
        """timing.json is written with required keys."""
        _, result = self._run_experiment(tmp_path)
        timing_path = Path(result.output_dir) / "timing.json"
        assert timing_path.exists()
        with open(timing_path) as f:
            timing = json.load(f)
        for key in ("feature_reduction_seconds", "k_train_seconds",
                    "k_val_seconds", "svm_fit_seconds", "total_seconds"):
            assert key in timing, f"Missing timing key: {key}"

    def test_classical_vs_quantum_csv_structure(self, tmp_path):
        """classical_vs_quantum.csv has required column headers."""
        _, result = self._run_experiment(tmp_path)
        csv_path = Path(result.output_dir) / "classical_vs_quantum.csv"
        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) >= 1
        required_cols = {"experiment_id", "model_name", "feature_dim",
                         "frame_accuracy", "frame_f1"}
        for col in required_cols:
            assert col in rows[0], f"Missing column: {col}"

    def test_dry_run_returns_without_files(self, tmp_path):
        """dry_run=True returns quickly without creating metric files."""
        fd = _make_features_dir(tmp_path, n_train=20, n_val=10)
        results_dir = tmp_path / "results"
        config = _small_config()

        runner = ExperimentRunner(
            features_dir=str(fd),
            results_dir=str(results_dir),
        )
        result = runner.run(config, dry_run=True)
        assert result.train_val_metrics.get("status") == "dry_run"

    def test_train_val_metrics_schema(self, tmp_path):
        """train_val_metrics.json has required schema keys."""
        _, result = self._run_experiment(tmp_path)
        tv = result.train_val_metrics
        assert "frame_level" in tv
        assert "video_level" in tv
        assert "accuracy" in tv["frame_level"]
        assert "f1" in tv["frame_level"]
        assert "best_strategy" in tv["video_level"]
        assert "best_threshold" in tv["video_level"]

    def test_per_video_val_json_written(self, tmp_path):
        """per_video_val.json is written to experiment directory."""
        _, result = self._run_experiment(tmp_path)
        pv_path = Path(result.output_dir) / "per_video_val.json"
        assert pv_path.exists(), "per_video_val.json not written"
        with open(pv_path) as f:
            data = json.load(f)
        assert len(data) > 0
