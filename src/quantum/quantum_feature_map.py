"""
Quantum Feature Map
=====================
Builds and validates the ZZFeatureMap quantum circuit used to encode
reduced feature vectors into quantum states for kernel computation.

Design decisions:
- Uses the ``zz_feature_map()`` *function* from ``qiskit.circuit.library``
  (Qiskit 2.x API).  The ``ZZFeatureMap`` *class* was deprecated in
  Qiskit 2.1 and scheduled for removal in 3.0; we avoid it entirely.
- Pure functions only — no stateful class.  The feature map is cheap to
  build and is recreated wherever needed rather than cached globally.
- ``num_parameters`` always equals ``feature_dimension`` for ZZFeatureMap
  regardless of ``reps`` or ``entanglement`` — confirmed for Qiskit 2.6.2.
- The depth limit in ``validate_feature_map`` is deliberately generous
  (≤ 200) to accommodate reps > 1 and full entanglement without false
  failures, while still catching runaway configurations.

Public API:
    build_feature_map(config)         → QuantumCircuit
    validate_feature_map(fm, config)  → None  (raises ValueError on failure)
    describe_feature_map(fm)          → dict
    bind_parameters(fm, values)       → QuantumCircuit  (bound copy)

CLI:
    python -m src.quantum.quantum_feature_map
    python -m src.quantum.quantum_feature_map --num-qubits 6 --reps 2
"""

import sys
import argparse
from collections import Counter
from pathlib import Path
from typing import Union

import numpy as np

try:
    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit.library import zz_feature_map
except ImportError as exc:
    raise ImportError(
        "qiskit is required for the quantum feature map.\n"
        "Install it with:  pip install 'qiskit==2.6.2'"
    ) from exc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quantum.quantum_config import QuantumConfig


# ── Maximum circuit depth considered safe for local simulation ────────────────
_MAX_DEPTH = 200


# ── Core functions ────────────────────────────────────────────────────────────

def build_feature_map(config: QuantumConfig) -> QuantumCircuit:
    """Build a ZZFeatureMap circuit from a QuantumConfig.

    Uses the ``zz_feature_map()`` function API introduced in Qiskit 2.1
    to avoid the deprecation warning from the ``ZZFeatureMap`` class.

    The ZZFeatureMap encodes an n-dimensional feature vector x into an
    n-qubit quantum state via:
        1. A layer of Hadamard gates (uniform superposition).
        2. Single-qubit Z-rotations:   exp(i · alpha · x_j · Z_j).
        3. Two-qubit ZZ-interactions:  exp(i · alpha · x_j · x_k · ZZ_jk)
           for all entangled pairs (j, k).
    Steps 1–3 are repeated ``reps`` times.

    The resulting kernel K(x, x') = |⟨φ(x)|φ(x')⟩|² captures non-linear
    feature interactions that are hard to replicate classically for large n.

    Args:
        config: QuantumConfig specifying num_qubits, reps, entanglement.

    Returns:
        An unbound QuantumCircuit with ``num_parameters == feature_dimension``.
    """
    fm = zz_feature_map(
        feature_dimension=config.feature_dimension,
        reps=config.reps,
        entanglement=config.entanglement,
    )
    return fm


def validate_feature_map(fm: QuantumCircuit, config: QuantumConfig) -> None:
    """Validate that a feature map circuit matches the expected configuration.

    Checks:
        1. Qubit count matches ``config.feature_dimension``.
        2. Parameter count matches ``config.feature_dimension``.
        3. Circuit depth is within the safe simulation limit.

    Args:
        fm:     QuantumCircuit to validate.
        config: QuantumConfig to validate against.

    Raises:
        ValueError: If any check fails.  The message names the specific
                    violation so the caller can act on it.
    """
    errors = []

    if fm.num_qubits != config.feature_dimension:
        errors.append(
            f"qubit count mismatch: circuit has {fm.num_qubits} qubits "
            f"but config.feature_dimension = {config.feature_dimension}"
        )

    if fm.num_parameters != config.feature_dimension:
        errors.append(
            f"parameter count mismatch: circuit has {fm.num_parameters} parameters "
            f"but config.feature_dimension = {config.feature_dimension}"
        )

    depth = fm.depth()
    if depth > _MAX_DEPTH:
        errors.append(
            f"circuit depth {depth} exceeds safe limit {_MAX_DEPTH}; "
            f"reduce reps or feature_dimension"
        )

    if errors:
        raise ValueError(
            "Feature map validation failed:\n"
            + "\n".join(f"  • {e}" for e in errors)
        )


def describe_feature_map(fm: QuantumCircuit) -> dict:
    """Return a metadata dictionary describing the feature map circuit.

    The returned dict is included verbatim in ``training_metadata.json``
    so that every saved model records the exact circuit that produced it.

    Args:
        fm: QuantumCircuit (bound or unbound).

    Returns:
        Dictionary with keys:
            num_qubits      — number of qubits
            num_parameters  — number of free parameters (0 if bound)
            depth           — circuit depth
            gate_counts     — dict mapping gate name → count
            parameter_names — sorted list of parameter name strings
    """
    gate_counts = dict(Counter(inst.operation.name for inst in fm.data))
    param_names = sorted(str(p) for p in fm.parameters)

    return {
        "num_qubits": fm.num_qubits,
        "num_parameters": fm.num_parameters,
        "depth": fm.depth(),
        "gate_counts": gate_counts,
        "parameter_names": param_names,
    }


def bind_parameters(
    fm: QuantumCircuit,
    values: Union[np.ndarray, list],
) -> QuantumCircuit:
    """Return a fully-bound copy of the feature map circuit.

    Parameters are bound in the order they appear in ``fm.parameters``
    (alphabetical for ZZFeatureMap: x[0], x[1], …).

    Args:
        fm:     Unbound feature map QuantumCircuit.
        values: 1-D array or list of length ``fm.num_parameters``.

    Returns:
        A new QuantumCircuit with all parameters replaced by the given values.

    Raises:
        ValueError: If ``len(values) != fm.num_parameters``.
    """
    values = np.asarray(values, dtype=float)
    if values.shape != (fm.num_parameters,):
        raise ValueError(
            f"values length {len(values)} does not match "
            f"num_parameters {fm.num_parameters}"
        )
    # assign_parameters accepts an ordered sequence; parameters are sorted
    # alphabetically in fm.parameters, matching the natural x[0]…x[n-1] order.
    return fm.assign_parameters(values)


# ── CLI / demo ────────────────────────────────────────────────────────────────

def _run_demo(num_qubits: int = 4, reps: int = 1, entanglement: str = "linear"):
    """Print circuit summary and run all acceptance checks."""
    cfg = QuantumConfig(
        num_qubits=num_qubits,
        feature_dimension=num_qubits,
        reps=reps,
        entanglement=entanglement,
    )

    print()
    print("=" * 60)
    print("  quantum_feature_map — Demo")
    print("=" * 60)
    print(f"  num_qubits   : {cfg.num_qubits}")
    print(f"  reps         : {cfg.reps}")
    print(f"  entanglement : {cfg.entanglement}")
    print()

    fm = build_feature_map(cfg)

    # Validation
    validate_feature_map(fm, cfg)

    # Description
    info = describe_feature_map(fm)
    print("  Circuit properties:")
    print(f"    num_qubits    : {info['num_qubits']}")
    print(f"    num_parameters: {info['num_parameters']}")
    print(f"    depth         : {info['depth']}")
    print(f"    gate_counts   : {info['gate_counts']}")
    print(f"    parameters    : {info['parameter_names']}")
    print()

    # Parameter binding with random values
    rng = np.random.default_rng(42)
    vals = rng.uniform(-np.pi, np.pi, fm.num_parameters)
    bound = bind_parameters(fm, vals)
    assert bound.num_parameters == 0, "Bound circuit should have 0 free parameters"

    # ASCII circuit diagram
    print("  Circuit diagram (unbound):")
    print()
    try:
        diagram = fm.draw(output="text", fold=80)
        for line in str(diagram).splitlines():
            print(f"    {line}")
    except Exception:
        print("    (circuit diagram unavailable in this environment)")
    print()

    print("  Acceptance checks:")
    assert info["num_qubits"] == num_qubits,   "qubit count"
    assert info["num_parameters"] == num_qubits, "parameter count"
    print(f"  ✅  num_qubits == {num_qubits}")
    print(f"  ✅  num_parameters == {num_qubits}")
    print(f"  ✅  validate_feature_map passed")
    print(f"  ✅  bind_parameters produces 0 free parameters")

    # validate_feature_map should raise on mismatched config
    bad_cfg = QuantumConfig(num_qubits=4, feature_dimension=4)
    bad_fm  = zz_feature_map(feature_dimension=2, reps=1)  # deliberate mismatch
    try:
        validate_feature_map(bad_fm, bad_cfg)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        print(f"  ✅  validate_feature_map raises ValueError on mismatch")

    print()
    print("  ✅  All acceptance criteria passed")
    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Build and inspect a ZZFeatureMap quantum circuit."
    )
    parser.add_argument(
        "--num-qubits", type=int, default=4, choices=[4, 6, 8],
        help="Number of qubits / feature dimension (default: 4)."
    )
    parser.add_argument(
        "--reps", type=int, default=1,
        help="ZZFeatureMap circuit repetitions (default: 1)."
    )
    parser.add_argument(
        "--entanglement", type=str, default="linear",
        choices=["linear", "full", "circular"],
        help="Entanglement pattern (default: linear)."
    )
    args = parser.parse_args()
    _run_demo(
        num_qubits=args.num_qubits,
        reps=args.reps,
        entanglement=args.entanglement,
    )


if __name__ == "__main__":
    main()
