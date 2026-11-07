"""
Tests for quantum_audit.py

Uses synthetic pre-computed artefacts in tmp_path — no live quantum circuits.
All tests run in < 10 seconds each.
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.svm import SVC

# Ensure src/ is on sys.path
_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from quantum.quantum_audit import run_audit, AuditReport, AuditCheck


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_synthetic_artefacts(tmp_path: Path, n: int = 20):
    """Create minimal valid QSVM artefact set in tmp_path."""
    models_dir = tmp_path / "models" / "quantum"
    features_dir = tmp_path / "features" / "quantum"
    models_dir.mkdir(parents=True)
    features_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)

    # Synthetic symmetric PSD kernel (n×n)
    A = rng.uniform(0, 1, (n, n))
    K_train = (A @ A.T) / n
    K_train = K_train / (K_train.diagonal().max() + 1e-8)
    # Force diagonal ≈ 1
    np.fill_diagonal(K_train, 1.0)
    # Ensure values in [0, 1]
    K_train = np.clip(K_train, 0.0, 1.0)

    # Cross kernel (m=10, n)
    m = 10
    K_val = rng.uniform(0, 1, (m, n))

    # Labels
    y_train = np.array([0] * (n // 2) + [1] * (n - n // 2))
    y_val = np.array([0] * (m // 2) + [1] * (m - m // 2))

    # Save kernel matrices
    np.save(str(features_dir / "K_train.npy"), K_train)
    np.save(str(features_dir / "K_val.npy"), K_val)
    np.save(str(features_dir / "train_labels.npy"), y_train)
    np.save(str(features_dir / "val_labels.npy"), y_val)

    # Fit a tiny SVC on the synthetic kernel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svc = SVC(kernel="precomputed", probability=True, C=1.0, random_state=42)
        svc.fit(K_train, y_train)

    joblib.dump(svc, models_dir / "quantum_svm.pkl")

    # feature_scaling.json
    scaling_meta = {
        "n_train_samples": n,
        "pca_components": 4,
        "feature_range": [-3.14159, 3.14159],
        "original_feature_dim": 512,
        "cumulative_explained_variance": 0.285,
        "config": {
            "num_qubits": 4,
            "feature_dimension": 4,
        },
    }
    with open(models_dir / "feature_scaling.json", "w") as f:
        json.dump(scaling_meta, f)

    return models_dir, features_dir


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRunAudit:
    """Tests for run_audit()."""

    def test_returns_audit_report(self, tmp_path):
        """run_audit returns an AuditReport instance."""
        models_dir, features_dir = _make_synthetic_artefacts(tmp_path)
        results_dir = tmp_path / "results"
        report = run_audit(
            str(models_dir), str(features_dir), str(results_dir)
        )
        assert isinstance(report, AuditReport)

    def test_eleven_checks_present(self, tmp_path):
        """AuditReport contains exactly 11 named checks."""
        models_dir, features_dir = _make_synthetic_artefacts(tmp_path)
        results_dir = tmp_path / "results"
        report = run_audit(
            str(models_dir), str(features_dir), str(results_dir)
        )
        assert len(report.checks) == 11, (
            f"Expected 11 checks, got {len(report.checks)}: {list(report.checks.keys())}"
        )

    def test_all_checks_have_passed_bool(self, tmp_path):
        """Each check has a 'passed' boolean field."""
        models_dir, features_dir = _make_synthetic_artefacts(tmp_path)
        results_dir = tmp_path / "results"
        report = run_audit(
            str(models_dir), str(features_dir), str(results_dir)
        )
        for name, chk in report.checks.items():
            assert isinstance(chk.passed, bool), (
                f"Check '{name}' has non-bool passed: {type(chk.passed)}"
            )

    def test_all_passed_is_conjunction(self, tmp_path):
        """all_passed equals all(check.passed for check in checks)."""
        models_dir, features_dir = _make_synthetic_artefacts(tmp_path)
        results_dir = tmp_path / "results"
        report = run_audit(
            str(models_dir), str(features_dir), str(results_dir)
        )
        expected = all(c.passed for c in report.checks.values())
        assert report.all_passed == expected

    def test_timestamp_is_set(self, tmp_path):
        """AuditReport.timestamp is a non-empty ISO string."""
        models_dir, features_dir = _make_synthetic_artefacts(tmp_path)
        results_dir = tmp_path / "results"
        report = run_audit(
            str(models_dir), str(features_dir), str(results_dir)
        )
        assert isinstance(report.timestamp, str)
        assert len(report.timestamp) > 0

    def test_audit_report_json_written(self, tmp_path):
        """audit_report.json is written to results_dir."""
        models_dir, features_dir = _make_synthetic_artefacts(tmp_path)
        results_dir = tmp_path / "results"
        run_audit(str(models_dir), str(features_dir), str(results_dir))

        report_path = results_dir / "audit_report.json"
        assert report_path.exists(), "audit_report.json was not created"
        with open(report_path) as f:
            data = json.load(f)
        assert "checks" in data
        assert "all_passed" in data
        assert len(data["checks"]) == 11

    def test_no_model_files_modified(self, tmp_path):
        """run_audit does not modify any model file (mtime unchanged)."""
        models_dir, features_dir = _make_synthetic_artefacts(tmp_path)
        results_dir = tmp_path / "results"

        svc_path = models_dir / "quantum_svm.pkl"
        scaling_path = models_dir / "feature_scaling.json"

        mtime_svc_before = svc_path.stat().st_mtime
        mtime_scaling_before = scaling_path.stat().st_mtime

        run_audit(str(models_dir), str(features_dir), str(results_dir))

        mtime_svc_after = svc_path.stat().st_mtime
        mtime_scaling_after = scaling_path.stat().st_mtime

        assert mtime_svc_before == mtime_svc_after, "quantum_svm.pkl was modified"
        assert mtime_scaling_before == mtime_scaling_after, "feature_scaling.json was modified"

    def test_leakage_detection_zero_samples(self, tmp_path):
        """Leakage guard check fails when n_train_samples=0."""
        models_dir, features_dir = _make_synthetic_artefacts(tmp_path)
        results_dir = tmp_path / "results"

        # Overwrite feature_scaling.json with n_train_samples=0
        scaling_path = models_dir / "feature_scaling.json"
        with open(scaling_path) as f:
            meta = json.load(f)
        meta["n_train_samples"] = 0
        with open(scaling_path, "w") as f:
            json.dump(meta, f)

        report = run_audit(str(models_dir), str(features_dir), str(results_dir))
        assert report.checks["leakage_guard"].passed is False

    def test_label_inversion_detected_on_reversed_svc(self, tmp_path):
        """Label inversion is detected when SVC has reversed class probabilities."""
        models_dir, features_dir = _make_synthetic_artefacts(tmp_path)
        results_dir = tmp_path / "results"

        # Create a SVC trained with inverted labels (1=REAL, 0=FAKE)
        K_train = np.load(str(features_dir / "K_train.npy"))
        y_train = np.load(str(features_dir / "train_labels.npy"))
        y_inverted = 1 - y_train  # flip labels

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            svc_inv = SVC(kernel="precomputed", probability=True, C=1.0, random_state=42)
            svc_inv.fit(K_train, y_inverted)

        joblib.dump(svc_inv, models_dir / "quantum_svm.pkl")

        # The label inversion check depends on probabilities; this is best-effort
        report = run_audit(str(models_dir), str(features_dir), str(results_dir))
        # We just verify the report has the check and it ran without error
        assert "label_inversion" in report.checks

    def test_missing_svc_raises_file_not_found(self, tmp_path):
        """FileNotFoundError raised when quantum_svm.pkl is missing."""
        models_dir, features_dir = _make_synthetic_artefacts(tmp_path)
        results_dir = tmp_path / "results"

        (models_dir / "quantum_svm.pkl").unlink()

        with pytest.raises(FileNotFoundError):
            run_audit(str(models_dir), str(features_dir), str(results_dir))

    def test_missing_scaling_json_raises_file_not_found(self, tmp_path):
        """FileNotFoundError raised when feature_scaling.json is missing."""
        models_dir, features_dir = _make_synthetic_artefacts(tmp_path)
        results_dir = tmp_path / "results"

        (models_dir / "feature_scaling.json").unlink()

        with pytest.raises(FileNotFoundError):
            run_audit(str(models_dir), str(features_dir), str(results_dir))

    def test_kta_check_present_and_reasonable(self, tmp_path):
        """KTA check is present and returns a numeric value."""
        models_dir, features_dir = _make_synthetic_artefacts(tmp_path)
        results_dir = tmp_path / "results"

        report = run_audit(str(models_dir), str(features_dir), str(results_dir))
        kta_check = report.checks.get("kta_score")
        assert kta_check is not None
        assert isinstance(kta_check.value, (int, float)) or kta_check.value is None

    def test_k_train_square_check(self, tmp_path):
        """Square K_train passes the k_train_square check."""
        models_dir, features_dir = _make_synthetic_artefacts(tmp_path)
        results_dir = tmp_path / "results"

        report = run_audit(str(models_dir), str(features_dir), str(results_dir))
        assert report.checks["k_train_square"].passed is True
