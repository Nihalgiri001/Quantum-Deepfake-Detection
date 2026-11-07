"""
Tests for src/quantum/ibm_hardware.py

ALL tests run without a real IBM Quantum account.
No jobs are submitted in any code path.

Covers: IBMCredentialsError, SafetyAbortError, PreflightRequiredError,
        preflight dict keys, train_hardware_svm without QPU.
"""
import os
import numpy as np
import pytest

from quantum.ibm_hardware import (
    IBMHardwareBackend,
    IBMCredentialsError,
    SafetyAbortError,
    PreflightRequiredError,
)
from quantum.quantum_config import QuantumConfig


@pytest.fixture(autouse=True)
def clear_token(monkeypatch):
    """Ensure IBM_QUANTUM_TOKEN is unset for all tests by default."""
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)


@pytest.fixture
def cfg():
    return QuantumConfig(
        num_qubits=4, feature_dimension=4, shots=64, random_seed=42
    )


@pytest.fixture
def with_token(monkeypatch):
    monkeypatch.setenv("IBM_QUANTUM_TOKEN", "FAKE_TEST_TOKEN_DO_NOT_USE")


@pytest.fixture
def backend_with_token(with_token):
    return IBMHardwareBackend(instance="ibm-q/open/main")


@pytest.fixture
def backend_after_preflight(backend_with_token, cfg):
    rng = np.random.default_rng(0)
    X = rng.uniform(-np.pi, np.pi, (6, 4))
    backend_with_token.preflight(cfg, X)
    return backend_with_token


class TestCredentials:
    def test_no_token_raises(self):
        with pytest.raises(IBMCredentialsError, match="IBM_QUANTUM_TOKEN"):
            IBMHardwareBackend()

    def test_empty_token_raises(self, monkeypatch):
        monkeypatch.setenv("IBM_QUANTUM_TOKEN", "")
        with pytest.raises(IBMCredentialsError):
            IBMHardwareBackend()

    def test_token_not_logged(self, with_token, capsys):
        IBMHardwareBackend(instance="ibm-q/open/main")
        captured = capsys.readouterr()
        assert "FAKE_TEST_TOKEN" not in captured.out
        assert "FAKE_TEST_TOKEN" not in captured.err


class TestPreflightGuard:
    def test_compute_without_preflight_raises(self, backend_with_token, cfg):
        rng = np.random.default_rng(0)
        X = rng.uniform(-np.pi, np.pi, (6, 4))
        with pytest.raises(PreflightRequiredError, match="preflight"):
            backend_with_token.compute_hardware_kernel(
                X, X[:3], cfg, confirmed=True
            )


class TestSafetyAbortGuard:
    def test_confirmed_false_raises(self, backend_after_preflight, cfg):
        rng = np.random.default_rng(0)
        X = rng.uniform(-np.pi, np.pi, (6, 4))
        with pytest.raises(SafetyAbortError, match="confirmed"):
            backend_after_preflight.compute_hardware_kernel(
                X, X[:3], cfg, confirmed=False
            )


class TestPreflightReport:
    REQUIRED_KEYS = {
        "n_train_samples", "n_test_samples", "n_kernel_pairs",
        "n_circuits", "circuit_depth", "transpiled_depth",
        "num_qubits", "shots", "estimated_qpu_time_sec",
        "backend_name", "warning",
    }

    def test_all_required_keys(self, backend_with_token, cfg):
        rng = np.random.default_rng(0)
        X = rng.uniform(-np.pi, np.pi, (6, 4))
        report = backend_with_token.preflight(cfg, X)
        assert self.REQUIRED_KEYS <= set(report.keys())

    def test_preflight_done_flag(self, backend_with_token, cfg):
        assert not backend_with_token._preflight_done
        rng = np.random.default_rng(0)
        X = rng.uniform(-np.pi, np.pi, (6, 4))
        backend_with_token.preflight(cfg, X)
        assert backend_with_token._preflight_done

    def test_n_circuits_formula(self, backend_with_token, cfg):
        n = 6
        rng = np.random.default_rng(0)
        X = rng.uniform(-np.pi, np.pi, (n, 4))
        report = backend_with_token.preflight(cfg, X)
        expected_pairs    = n * (n - 1) // 2 + n
        expected_circuits = expected_pairs * 2
        assert report["n_kernel_pairs"] == expected_pairs
        assert report["n_circuits"] == expected_circuits

    def test_estimated_time_positive(self, backend_with_token, cfg):
        rng = np.random.default_rng(0)
        X = rng.uniform(-np.pi, np.pi, (6, 4))
        report = backend_with_token.preflight(cfg, X)
        assert report["estimated_qpu_time_sec"] > 0

    def test_warning_contains_budget_note(self, backend_with_token, cfg):
        rng = np.random.default_rng(0)
        X = rng.uniform(-np.pi, np.pi, (6, 4))
        report = backend_with_token.preflight(cfg, X)
        assert "600" in report["warning"] or "month" in report["warning"]


class TestTrainHardwareSVM:
    def test_trains_and_saves(self, backend_with_token, cfg, tmp_path):
        rng = np.random.default_rng(0)
        n_tr, n_te = 8, 4
        A = rng.uniform(0, 1, (n_tr, n_tr))
        K_train = (A @ A.T) / n_tr + np.eye(n_tr) * 0.5
        K_test  = rng.uniform(0, 1, (n_te, n_tr))
        y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_test  = np.array([0, 0, 1, 1])
        meta = backend_with_token.train_hardware_svm(
            K_train, y_train, K_test, y_test,
            config=cfg, output_dir=str(tmp_path)
        )
        assert 0.0 <= meta["test_accuracy"] <= 1.0
        assert 0.0 <= meta["test_f1"] <= 1.0
        assert os.path.exists(str(tmp_path / "quantum_svm_hw.pkl"))
        assert os.path.exists(str(tmp_path / "hardware_training_metadata.json"))

    def test_metadata_fields(self, backend_with_token, cfg, tmp_path):
        rng = np.random.default_rng(1)
        n = 6
        A = rng.uniform(0, 1, (n, n))
        K = (A @ A.T) / n + np.eye(n) * 0.5
        y = np.array([0, 0, 0, 1, 1, 1])
        meta = backend_with_token.train_hardware_svm(
            K, y, K[:3], y[:3], config=cfg, output_dir=str(tmp_path)
        )
        for key in ("model", "test_accuracy", "test_f1", "n_train", "n_test"):
            assert key in meta


class TestNoJobSubmission:
    def test_preflight_no_submit(self, backend_with_token, cfg, monkeypatch):
        """preflight() must never call SamplerV2.run."""
        submitted = []

        class FakeSamplerV2:
            def run(self, *a, **kw):
                submitted.append(True)

        monkeypatch.setattr(
            "qiskit_ibm_runtime.SamplerV2",
            lambda *a, **kw: FakeSamplerV2()
        )
        rng = np.random.default_rng(0)
        backend_with_token.preflight(cfg, rng.uniform(-np.pi, np.pi, (6, 4)))
        assert not submitted, "preflight() must not submit any QPU jobs"
