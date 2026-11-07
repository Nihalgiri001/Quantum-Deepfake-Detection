"""
Local Quantum Simulator Backend
=================================
Wraps Qiskit's StatevectorSampler into a clean, reusable backend that
drives all local simulator-based kernel computations.

Design decisions:
- Uses ``qiskit.primitives.StatevectorSampler`` (Qiskit 2.x core), NOT
  ``qiskit_aer.primitives.StatevectorSampler``.  The former performs exact
  statevector simulation in pure Python/NumPy and needs no Aer binary,
  which makes it the most portable choice on Apple Silicon.
- Qiskit Aer is still listed as a dependency (for SamplerV2 / noise models
  in future work), but the default simulator path does not require it.
- The sampler and fidelity objects are lazy-initialised on first use and
  then cached — each SimulatorBackend instance owns exactly one sampler.
- Shot count is passed as ``default_shots`` to StatevectorSampler.  In
  statevector mode the shot count only affects the *sampling* of the
  probability distribution; the underlying fidelity values are exact.
- Setting ``seed`` makes every circuit execution deterministic, which is
  critical for reproducible kernel matrices.

Usage:
    backend = SimulatorBackend(shots=1024, seed=42)
    sampler = backend.get_sampler()
    fidelity = backend.get_fidelity(feature_map)
    kernel   = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

CLI:
    python -m src.quantum.simulator_backend
"""

import sys
from pathlib import Path
from typing import Optional

# ── Qiskit imports with helpful error messages ───────────────────────────────
try:
    from qiskit.primitives import StatevectorSampler
except ImportError as exc:
    raise ImportError(
        "qiskit is required for the quantum pipeline.\n"
        "Install it with:  pip install 'qiskit==2.6.2'"
    ) from exc

try:
    from qiskit_machine_learning.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
except ImportError as exc:
    raise ImportError(
        "qiskit-machine-learning is required for quantum kernel computation.\n"
        "Install it with:  pip install 'qiskit-machine-learning==0.9.1'"
    ) from exc


class SimulatorBackend:
    """Local statevector simulator backend for quantum kernel computation.

    Provides a ``StatevectorSampler`` and a ``ComputeUncompute`` fidelity
    object, both seeded for reproducibility and cached for efficiency.

    Attributes:
        shots:   Default shot count per circuit.
        seed:    Integer random seed for reproducible simulation.
    """

    def __init__(self, shots: int = 1024, seed: int = 42):
        """Initialise the simulator backend.

        Args:
            shots: Number of shots per circuit execution.  In statevector
                   mode this controls sampling granularity; fidelity values
                   are exact regardless.
            seed:  Random seed for reproducible output.
        """
        self.shots = shots
        self.seed = seed

        # Lazy-initialised; created on first call to get_sampler()
        self._sampler: Optional[StatevectorSampler] = None

    # ── Core accessors ────────────────────────────────────────────────────────

    def get_sampler(self) -> StatevectorSampler:
        """Return the cached StatevectorSampler instance.

        Creates the sampler on first call and reuses it on subsequent calls
        so that circuit execution options remain consistent throughout a
        kernel computation run.

        Returns:
            A seeded ``StatevectorSampler`` with ``default_shots`` set.
        """
        if self._sampler is None:
            self._sampler = StatevectorSampler(
                default_shots=self.shots,
                seed=self.seed,
            )
        return self._sampler

    def get_fidelity(self, feature_map) -> ComputeUncompute:
        """Build a ComputeUncompute fidelity object for the given feature map.

        ``ComputeUncompute`` estimates the squared overlap
        |⟨ψ(x)|ψ(x')⟩|² between two quantum states by appending the
        inverse circuit and measuring in the computational basis.

        A new ``ComputeUncompute`` instance is created each time because it
        is stateless (it wraps the sampler) and cheap to construct.

        Args:
            feature_map: A ``QuantumCircuit`` whose parameters encode the
                         input feature vector.

        Returns:
            A ``ComputeUncompute`` fidelity primitive backed by this
            backend's cached sampler.
        """
        return ComputeUncompute(sampler=self.get_sampler())

    def build_kernel(self, feature_map) -> FidelityQuantumKernel:
        """Convenience: build a FidelityQuantumKernel ready for evaluation.

        Args:
            feature_map: Parameterised ``QuantumCircuit`` (e.g. from
                         ``quantum_feature_map.build_feature_map``).

        Returns:
            ``FidelityQuantumKernel`` wired to this backend.
        """
        fidelity = self.get_fidelity(feature_map)
        return FidelityQuantumKernel(
            feature_map=feature_map,
            fidelity=fidelity,
        )

    # ── Class-level availability check ────────────────────────────────────────

    @classmethod
    def is_available(cls) -> bool:
        """Check whether all required packages are importable.

        Returns:
            True if qiskit, qiskit-machine-learning, and qiskit-aer are
            all importable; False otherwise.
        """
        try:
            import qiskit                                     # noqa: F401
            import qiskit_machine_learning                    # noqa: F401
            import qiskit_aer                                 # noqa: F401
            from qiskit.primitives import StatevectorSampler  # noqa: F401
            from qiskit_machine_learning.state_fidelities import ComputeUncompute  # noqa: F401
            return True
        except ImportError:
            return False

    # ── String representation ─────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"SimulatorBackend("
            f"shots={self.shots}, seed={self.seed}, "
            f"sampler_cached={self._sampler is not None})"
        )


# ── Module-level CLI / demo ───────────────────────────────────────────────────

def _run_demo():
    """Print backend info and run a 2-qubit statevector fidelity sanity check."""
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    try:
        import qiskit
        import qiskit_machine_learning
        import qiskit_aer
    except ImportError:
        pass  # versions printed below where available

    print()
    print("=" * 55)
    print("  SimulatorBackend — Environment Check")
    print("=" * 55)

    # Package versions
    def _ver(pkg_name):
        try:
            import importlib
            m = importlib.import_module(pkg_name)
            return getattr(m, "__version__", "unknown")
        except ImportError:
            return "NOT INSTALLED"

    print(f"  qiskit                  : {_ver('qiskit')}")
    print(f"  qiskit-machine-learning : {_ver('qiskit_machine_learning')}")
    print(f"  qiskit-aer              : {_ver('qiskit_aer')}")

    if not SimulatorBackend.is_available():
        print()
        print("  ❌  One or more required packages are missing.")
        print("  Run:  pip install qiskit==2.6.2 qiskit-machine-learning==0.9.1 qiskit-aer==0.18.2")
        sys.exit(1)

    print()
    print("  ✅  All required packages available.")
    print()

    # ── Reproducibility check ─────────────────────────────────────────────────
    from qiskit.circuit.library import zz_feature_map

    fm = zz_feature_map(feature_dimension=2, reps=1)
    X = np.array([[0.5, -0.3], [1.2, 0.8], [-0.9, 1.5]])

    results = []
    for run in range(2):
        backend = SimulatorBackend(shots=1024, seed=42)
        kernel = backend.build_kernel(fm)
        K = kernel.evaluate(x_vec=X)
        results.append(K.copy())

    # Structural checks
    assert results[0].shape == (3, 3), f"Expected (3,3), got {results[0].shape}"
    assert np.allclose(results[0], results[0].T), "Kernel not symmetric"
    assert np.allclose(np.diag(results[0]), 1.0), "Diagonal not ≈ 1"
    assert np.array_equal(results[0], results[1]), "Results differ between runs with same seed"

    print("─" * 55)
    print("  2-qubit ZZFeatureMap kernel (3×3 on random inputs)")
    print("─" * 55)
    print(f"  Shape:      {results[0].shape}")
    print(f"  Symmetric:  {np.allclose(results[0], results[0].T)}")
    print(f"  Diagonal:   {np.diag(results[0]).tolist()}")
    print(f"  Off-diag sample K[0,1] = {results[0][0,1]:.6f}")
    print()
    print("  Reproducibility (same seed, two runs):")
    print(f"  Run 1 K[0,1] = {results[0][0,1]:.8f}")
    print(f"  Run 2 K[0,1] = {results[1][0,1]:.8f}")
    print(f"  Match: {np.array_equal(results[0], results[1])}")
    print()
    print("  ✅  SimulatorBackend demo PASSED")
    print("=" * 55)
    print()


if __name__ == "__main__":
    _run_demo()
