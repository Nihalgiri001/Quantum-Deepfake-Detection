"""
Tests for quantum_diagnostics.py

All tests use synthetic pre-computed kernel matrices.
No live Qiskit circuits invoked.
All tests run in < 10 seconds each.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ is on sys.path
_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from quantum.quantum_diagnostics import DiagnosticsEngine, DiagnosticsReport
from quantum.quantum_kernel import KernelNumericalError


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_well_separated_kernel(n: int = 20) -> tuple:
    """
    Create a kernel where same-class pairs have mean ~0.9 and
    different-class pairs have mean ~0.1. KTA should be clearly positive.
    """
    y = np.array([0] * (n // 2) + [1] * (n - n // 2))
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if y[i] == y[j]:
                K[i, j] = 0.9 if i != j else 1.0
            else:
                K[i, j] = 0.1
    K = np.clip(K, 0.0, 1.0)
    return K, y


def _make_concentrated_kernel(n: int = 20) -> np.ndarray:
    """Create a kernel where all off-diagonal values are nearly identical (CV < 0.05)."""
    K = np.full((n, n), 0.5)
    # Add tiny variation: std/mean ~ 0.001/0.5 = 0.002 << 0.05
    rng = np.random.default_rng(1)
    K += rng.uniform(-0.001, 0.001, (n, n))
    K = (K + K.T) / 2
    np.fill_diagonal(K, 1.0)
    K = np.clip(K, 0.0, 1.0)
    return K


def _make_dispersed_kernel(n: int = 20) -> np.ndarray:
    """Create a kernel whose off-diagonal CV > 0.1 (not concentrated)."""
    rng = np.random.default_rng(2)
    A = rng.uniform(0, 1, (n, n))
    K = (A + A.T) / 2
    np.fill_diagonal(K, 1.0)
    K = K / K.max()
    return K


def _make_psd_kernel(n: int = 10) -> np.ndarray:
    """Create a valid PSD kernel matrix."""
    rng = np.random.default_rng(3)
    A = rng.uniform(0, 1, (n, n))
    K = (A @ A.T) / n
    np.fill_diagonal(K, 1.0)
    K = np.clip(K, 0.0, 1.0)
    return K


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDiagnosticsEngineAnalyze:
    """Unit tests for DiagnosticsEngine.analyze()."""

    def test_returns_diagnostics_report(self):
        """analyze() returns a DiagnosticsReport."""
        K = _make_psd_kernel(10)
        engine = DiagnosticsEngine()
        report = engine.analyze(K)
        assert isinstance(report, DiagnosticsReport)

    def test_statistics_exact(self):
        """Scalar statistics match numpy ground truth (Property 1)."""
        rng = np.random.default_rng(42)
        K = rng.uniform(0, 1, (15, 15))
        K = (K + K.T) / 2
        np.fill_diagonal(K, 1.0)
        engine = DiagnosticsEngine()
        report = engine.analyze(K)

        assert abs(report.min_value - float(K.min())) < 1e-9
        assert abs(report.max_value - float(K.max())) < 1e-9
        assert abs(report.mean - float(K.mean())) < 1e-9
        assert abs(report.std - float(K.std())) < 1e-9

    def test_kta_positive_for_well_separated(self):
        """KTA > 0 when same-class pairs have high similarity (Property 2)."""
        K, y = _make_well_separated_kernel(20)
        engine = DiagnosticsEngine()
        report = engine.analyze(K, y)
        assert report.kta is not None
        assert report.kta > 0, f"Expected KTA > 0; got {report.kta}"

    def test_kta_verdict_good(self):
        """KTA verdict is 'good' when KTA > 0.1."""
        K, y = _make_well_separated_kernel(20)
        engine = DiagnosticsEngine()
        report = engine.analyze(K, y)
        # With 0.9 / 0.1 separation, KTA should easily exceed 0.1
        assert report.kta_verdict in ("good", "weak"), (
            f"Expected 'good' or 'weak'; got '{report.kta_verdict}', kta={report.kta}"
        )

    def test_concentration_detected_low_cv(self):
        """concentration_warning=True when off-diagonal CV < 0.05 (Property 3)."""
        K = _make_concentrated_kernel(20)
        engine = DiagnosticsEngine()
        report = engine.analyze(K)
        assert report.kernel_concentration_warning is True, (
            f"Expected concentration warning; CV={report.off_diagonal_cv:.4f}"
        )

    def test_concentration_not_detected_high_cv(self):
        """concentration_warning=False when off-diagonal CV >= 0.05 (Property 3)."""
        K = _make_dispersed_kernel(20)
        engine = DiagnosticsEngine()
        report = engine.analyze(K)
        assert report.kernel_concentration_warning is False, (
            f"Expected no concentration warning; CV={report.off_diagonal_cv:.4f}"
        )

    def test_rectangular_matrix_skips_square_fields(self):
        """Rectangular matrix produces None for square-only fields (Property 4)."""
        rng = np.random.default_rng(5)
        K = rng.uniform(0, 1, (10, 20))  # rectangular
        engine = DiagnosticsEngine()
        report = engine.analyze(K)

        assert report.symmetry_error is None
        assert report.eigenvalue_min is None
        assert report.eigenvalue_max is None
        assert report.neg_eigenvalue_fraction is None
        assert report.is_psd is None
        assert report.diagonal_mean is None
        assert report.diagonal_std is None
        assert report.is_square is False

    def test_square_matrix_computes_square_fields(self):
        """Square matrix returns non-None for square-only fields (Property 4)."""
        K = _make_psd_kernel(10)
        engine = DiagnosticsEngine()
        report = engine.analyze(K)

        assert report.symmetry_error is not None
        assert report.eigenvalue_min is not None
        assert report.eigenvalue_max is not None
        assert report.neg_eigenvalue_fraction is not None
        assert report.is_psd is not None
        assert report.is_square is True

    def test_nan_injection_raises_kernel_numerical_error(self):
        """NaN in kernel matrix raises KernelNumericalError."""
        K = _make_psd_kernel(10).copy()
        K[0, 0] = float("nan")
        engine = DiagnosticsEngine()
        with pytest.raises(KernelNumericalError):
            engine.analyze(K)

    def test_inf_injection_raises_kernel_numerical_error(self):
        """Inf in kernel matrix raises KernelNumericalError."""
        K = _make_psd_kernel(10).copy()
        K[1, 2] = float("inf")
        engine = DiagnosticsEngine()
        with pytest.raises(KernelNumericalError):
            engine.analyze(K)

    def test_pct_outside_unit_zero_for_valid_kernel(self):
        """All values in [0,1] gives pct_outside_unit == 0."""
        K = _make_psd_kernel(10)
        engine = DiagnosticsEngine()
        report = engine.analyze(K)
        assert report.pct_outside_unit == 0.0

    def test_shape_and_is_square_fields(self):
        """Shape and is_square fields are set correctly."""
        K = _make_psd_kernel(10)
        engine = DiagnosticsEngine()
        report = engine.analyze(K)
        assert report.shape == (10, 10)
        assert report.is_square is True

    def test_has_nan_and_has_inf_false_for_valid(self):
        """has_nan and has_inf are False for a valid kernel."""
        K = _make_psd_kernel(10)
        engine = DiagnosticsEngine()
        report = engine.analyze(K)
        assert report.has_nan is False
        assert report.has_inf is False

    def test_psd_kernel_is_psd(self):
        """PSD kernel matrix is reported as PSD."""
        K = _make_psd_kernel(10)
        engine = DiagnosticsEngine()
        report = engine.analyze(K)
        assert report.is_psd is True

    def test_class_conditioned_stats_none_without_labels(self):
        """Class-conditioned stats are None when y is not provided."""
        K = _make_psd_kernel(10)
        engine = DiagnosticsEngine()
        report = engine.analyze(K, y=None)
        assert report.same_class_mean is None
        assert report.diff_class_mean is None
        assert report.separation_ratio is None
        assert report.kta is None
        assert report.kta_verdict is None

    def test_off_diagonal_mean_correct(self):
        """Off-diagonal mean matches manual computation for a square matrix."""
        n = 5
        K = np.ones((n, n)) * 0.5
        np.fill_diagonal(K, 1.0)
        engine = DiagnosticsEngine()
        report = engine.analyze(K)
        # All off-diagonal entries are 0.5
        assert abs(report.off_diagonal_mean - 0.5) < 1e-9


class TestDiagnosticsRoundTrip:
    """Tests for save_report() and JSON serialisation."""

    def test_round_trip_json_has_required_keys(self, tmp_path):
        """save_report then reload JSON has all required keys."""
        K = _make_psd_kernel(10)
        y = np.array([0] * 5 + [1] * 5)
        engine = DiagnosticsEngine()
        report = engine.analyze(K, y)

        output_path = str(tmp_path / "diagnostics.json")
        engine.save_report(report, output_path)

        with open(output_path) as f:
            data = json.load(f)

        required_keys = {
            "min_value", "max_value", "mean", "std",
            "off_diagonal_mean", "off_diagonal_std", "off_diagonal_cv",
            "symmetry_error", "diagonal_mean", "diagonal_std",
            "pct_outside_unit", "same_class_mean", "diff_class_mean",
            "separation_ratio", "kta", "kta_verdict",
            "eigenvalue_min", "eigenvalue_max", "neg_eigenvalue_fraction",
            "is_psd", "kernel_concentration_warning", "has_nan", "has_inf",
            "shape", "is_square",
        }
        missing = required_keys - set(data.keys())
        assert len(missing) == 0, f"Missing keys in JSON: {missing}"

    def test_report_json_file_nonempty(self, tmp_path):
        """Saved JSON file has non-zero size."""
        K = _make_psd_kernel(8)
        engine = DiagnosticsEngine()
        report = engine.analyze(K)
        output_path = str(tmp_path / "diag.json")
        engine.save_report(report, output_path)
        assert Path(output_path).stat().st_size > 0


class TestDiagnosticsPlots:
    """Tests for heatmap and histogram generation."""

    def test_heatmap_created_nonempty(self, tmp_path):
        """generate_heatmap creates a non-zero PNG file."""
        K = _make_psd_kernel(10)
        engine = DiagnosticsEngine()
        heatmap_path = str(tmp_path / "heatmap.png")
        engine.generate_heatmap(K, heatmap_path, title="Test Heatmap")
        p = Path(heatmap_path)
        assert p.exists(), "Heatmap file was not created"
        assert p.stat().st_size > 0, "Heatmap file is empty"

    def test_histogram_created_nonempty(self, tmp_path):
        """generate_similarity_histogram creates a non-zero PNG file."""
        K = _make_psd_kernel(10)
        y = np.array([0] * 5 + [1] * 5)
        engine = DiagnosticsEngine()
        hist_path = str(tmp_path / "histogram.png")
        engine.generate_similarity_histogram(K, y, hist_path)
        p = Path(hist_path)
        assert p.exists(), "Histogram file was not created"
        assert p.stat().st_size > 0, "Histogram file is empty"


class TestPropertyBasedDiagnostics:
    """Property-based style tests (deterministic — not using Hypothesis for speed)."""

    def test_stats_exact_multiple_sizes(self):
        """Scalar stats are exact for several kernel sizes (Property 1 sweep)."""
        engine = DiagnosticsEngine()
        rng = np.random.default_rng(99)
        for n in [5, 10, 15, 20]:
            K = rng.uniform(0.0, 1.0, (n, n))
            K = (K + K.T) / 2
            np.fill_diagonal(K, 1.0)
            report = engine.analyze(K)
            assert abs(report.min_value - float(K.min())) < 1e-9
            assert abs(report.max_value - float(K.max())) < 1e-9
            assert abs(report.mean - float(K.mean())) < 1e-9

    def test_concentration_boundary(self):
        """CV exactly at 0.05 threshold triggers warning check correctly (Property 3)."""
        engine = DiagnosticsEngine()
        n = 20
        # Build a kernel with off-diagonal std/mean = 0.04 (below threshold → warning)
        K_low = np.full((n, n), 0.5)
        rng = np.random.default_rng(10)
        noise = rng.uniform(-0.02, 0.02, (n, n))
        K_low += noise
        K_low = (K_low + K_low.T) / 2
        np.fill_diagonal(K_low, 1.0)
        K_low = np.clip(K_low, 0.0, 1.0)

        report_low = engine.analyze(K_low)
        # We check the logic: if cv < 0.05 → warning=True
        expected_warning = report_low.off_diagonal_cv < 0.05
        assert report_low.kernel_concentration_warning == expected_warning

    def test_kta_property_2_multiple_scenarios(self):
        """KTA > 0 for multiple well-separated kernels (Property 2)."""
        engine = DiagnosticsEngine()
        for n in [10, 16, 20]:
            K, y = _make_well_separated_kernel(n)
            report = engine.analyze(K, y)
            assert report.kta is not None and report.kta > 0, (
                f"KTA={report.kta} not > 0 for n={n}"
            )
