"""
Tests for src/quantum/quantum_feature_map.py

Covers: qubit counts, parameter counts, binding, validate_feature_map,
        describe_feature_map, depth limits.
"""
import numpy as np
import pytest

from quantum.quantum_config import QuantumConfig
from quantum.quantum_feature_map import (
    build_feature_map,
    validate_feature_map,
    describe_feature_map,
    bind_parameters,
)


@pytest.fixture
def cfg4():
    return QuantumConfig(num_qubits=4, feature_dimension=4)


@pytest.fixture
def cfg6():
    return QuantumConfig(num_qubits=6, feature_dimension=6)


class TestBuildFeatureMap:
    def test_4qubit_qubit_count(self, cfg4):
        fm = build_feature_map(cfg4)
        assert fm.num_qubits == 4

    def test_4qubit_parameter_count(self, cfg4):
        fm = build_feature_map(cfg4)
        assert fm.num_parameters == 4

    def test_6qubit_qubit_count(self, cfg6):
        fm = build_feature_map(cfg6)
        assert fm.num_qubits == 6

    def test_6qubit_parameter_count(self, cfg6):
        fm = build_feature_map(cfg6)
        assert fm.num_parameters == 6

    def test_8qubit(self):
        cfg8 = QuantumConfig(num_qubits=8, feature_dimension=8)
        fm = build_feature_map(cfg8)
        assert fm.num_qubits == 8
        assert fm.num_parameters == 8

    def test_depth_positive(self, cfg4):
        fm = build_feature_map(cfg4)
        assert fm.depth() > 0

    def test_returns_quantum_circuit(self, cfg4):
        from qiskit.circuit import QuantumCircuit
        fm = build_feature_map(cfg4)
        assert isinstance(fm, QuantumCircuit)


class TestValidateFeatureMap:
    def test_valid_config_does_not_raise(self, cfg4):
        fm = build_feature_map(cfg4)
        validate_feature_map(fm, cfg4)   # should not raise

    def test_qubit_mismatch_raises(self, cfg4):
        from qiskit.circuit.library import zz_feature_map
        bad_fm = zz_feature_map(feature_dimension=2, reps=1)
        with pytest.raises(ValueError, match="qubit count mismatch"):
            validate_feature_map(bad_fm, cfg4)

    def test_depth_limit_respected(self, cfg4):
        import quantum.quantum_feature_map as qfm_mod
        original = qfm_mod._MAX_DEPTH
        qfm_mod._MAX_DEPTH = 5
        try:
            fm = build_feature_map(cfg4)
            with pytest.raises(ValueError, match="depth"):
                validate_feature_map(fm, cfg4)
        finally:
            qfm_mod._MAX_DEPTH = original


class TestDescribeFeatureMap:
    def test_required_keys(self, cfg4):
        fm = build_feature_map(cfg4)
        info = describe_feature_map(fm)
        for key in ("num_qubits", "num_parameters", "depth",
                    "gate_counts", "parameter_names"):
            assert key in info

    def test_correct_values(self, cfg4):
        fm = build_feature_map(cfg4)
        info = describe_feature_map(fm)
        assert info["num_qubits"] == 4
        assert info["num_parameters"] == 4
        assert info["depth"] > 0
        assert isinstance(info["gate_counts"], dict)

    def test_parameter_names_sorted(self, cfg4):
        fm = build_feature_map(cfg4)
        info = describe_feature_map(fm)
        names = info["parameter_names"]
        assert names == sorted(names)


class TestBindParameters:
    def test_binding_removes_free_params(self, cfg4):
        fm = build_feature_map(cfg4)
        vals = np.full(fm.num_parameters, 0.5)
        bound = bind_parameters(fm, vals)
        assert bound.num_parameters == 0

    def test_binding_does_not_mutate_original(self, cfg4):
        fm = build_feature_map(cfg4)
        n_params_before = fm.num_parameters
        bind_parameters(fm, np.zeros(fm.num_parameters))
        assert fm.num_parameters == n_params_before

    def test_wrong_length_raises(self, cfg4):
        fm = build_feature_map(cfg4)
        with pytest.raises(ValueError):
            bind_parameters(fm, np.array([0.1, 0.2]))   # too short

    def test_accepts_list_input(self, cfg4):
        fm = build_feature_map(cfg4)
        bound = bind_parameters(fm, [0.1, 0.2, 0.3, 0.4])
        assert bound.num_parameters == 0
