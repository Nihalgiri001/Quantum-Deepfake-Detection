"""
Qiskit Installation Check
===========================
Validates that the entire Qiskit stack required by TrustLens-Q is
installed correctly, at the right versions, and that end-to-end kernel
computation works.

Checks performed:
    1. All four required packages can be imported.
    2. Qiskit version is ≥ 2.0 (confirms the 2.x API is in use).
    3. The ``zz_feature_map`` function API is available (not the deprecated
       ``ZZFeatureMap`` class).
    4. ``StatevectorSampler`` lives in ``qiskit.primitives`` (2.x location).
    5. ``FidelityQuantumKernel`` and ``ComputeUncompute`` are importable.
    6. End-to-end smoke: a 2-qubit ZZFeatureMap kernel is evaluated on 3
       random samples, producing a valid (3,3) symmetric matrix.

Exit codes:
    0 — all checks passed.
    1 — one or more checks failed (message printed to stderr).

CLI:
    python -m src.quantum.check_install
"""

import sys
import textwrap
from pathlib import Path


def _version_table() -> list:
    """Collect (package_display_name, version_str, status) tuples."""
    rows = []

    packages = [
        ("qiskit",                   "qiskit"),
        ("qiskit-machine-learning",  "qiskit_machine_learning"),
        ("qiskit-aer",               "qiskit_aer"),
        ("qiskit-ibm-runtime",       "qiskit_ibm_runtime"),
    ]

    for display, import_name in packages:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "unknown")
            rows.append((display, ver, "✅"))
        except ImportError:
            rows.append((display, "NOT INSTALLED", "❌"))

    return rows


def _check_qiskit_version() -> tuple:
    """Return (major, minor, patch) or raise ImportError."""
    import qiskit
    raw = qiskit.__version__          # e.g. "2.6.2"
    parts = raw.split(".")
    major = int(parts[0]) if parts else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    patch = int(parts[2]) if len(parts) > 2 else 0
    return major, minor, patch


def _run_kernel_smoke_test() -> dict:
    """2-qubit ZZFeatureMap kernel on 3 random samples.

    Returns a dict with keys: shape, symmetric, diagonal_ok, values_ok.
    """
    import numpy as np
    from qiskit.circuit.library import zz_feature_map
    from qiskit.primitives import StatevectorSampler
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.state_fidelities import ComputeUncompute

    fm       = zz_feature_map(feature_dimension=2, reps=1)
    sampler  = StatevectorSampler(seed=42)
    fidelity = ComputeUncompute(sampler=sampler)
    kernel   = FidelityQuantumKernel(feature_map=fm, fidelity=fidelity)

    rng = np.random.default_rng(0)
    X   = rng.uniform(-3.14159, 3.14159, (3, 2))
    K   = kernel.evaluate(x_vec=X)

    return {
        "shape":        K.shape,
        "symmetric":    bool(np.allclose(K, K.T, atol=1e-6)),
        "diagonal_ok":  bool(np.allclose(np.diag(K), 1.0, atol=1e-3)),
        "values_ok":    bool((K >= -1e-6).all() and (K <= 1.0 + 1e-6).all()),
        "has_nan":      bool(np.isnan(K).any()),
    }


def run_checks(verbose: bool = True) -> bool:
    """Run all installation checks.

    Args:
        verbose: Print results to stdout.

    Returns:
        True if all checks passed, False otherwise.
    """
    failures = []

    # ── 1. Version table ──────────────────────────────────────────────────────
    rows = _version_table()
    if verbose:
        print()
        print("=" * 55)
        print("  TrustLens-Q — Qiskit Installation Check")
        print("=" * 55)
        print()
        print(f"  {'Package':<28} {'Version':<15} Status")
        print("  " + "─" * 50)
        for display, ver, status in rows:
            print(f"  {display:<28} {ver:<15} {status}")
        print()

    for display, ver, status in rows:
        if status == "❌":
            failures.append(
                f"Package not installed: {display}\n"
                f"  Fix:  pip install '{display}'"
            )

    if failures:
        # Can't proceed with API checks if imports failed
        _report_failures(failures, verbose)
        return False

    # ── 2. Qiskit version ≥ 2.0 ──────────────────────────────────────────────
    major, minor, patch = _check_qiskit_version()
    version_str = f"{major}.{minor}.{patch}"
    version_ok  = major >= 2

    if verbose:
        status = "✅" if version_ok else "❌"
        print(f"  Qiskit version ≥ 2.0 : {version_str}  {status}")

    if not version_ok:
        failures.append(
            f"Qiskit {version_str} is too old — TrustLens-Q requires ≥ 2.0.\n"
            f"  Fix:  pip install 'qiskit==2.6.2'"
        )

    # ── 3. zz_feature_map function (not deprecated class) ────────────────────
    try:
        from qiskit.circuit.library import zz_feature_map
        fm   = zz_feature_map(feature_dimension=2, reps=1)
        ok   = fm.num_qubits == 2 and fm.num_parameters == 2
        if verbose:
            status = "✅" if ok else "❌"
            print(f"  zz_feature_map() function API  {status}")
        if not ok:
            failures.append("zz_feature_map() returned unexpected circuit.")
    except Exception as e:
        if verbose:
            print(f"  zz_feature_map() function API  ❌")
        failures.append(f"zz_feature_map import failed: {e}")

    # ── 4. StatevectorSampler in qiskit.primitives ────────────────────────────
    try:
        from qiskit.primitives import StatevectorSampler
        ok = hasattr(StatevectorSampler(seed=0), "run")
        if verbose:
            status = "✅" if ok else "❌"
            print(f"  StatevectorSampler (qiskit.primitives)  {status}")
        if not ok:
            failures.append("StatevectorSampler has no .run method.")
    except Exception as e:
        if verbose:
            print(f"  StatevectorSampler (qiskit.primitives)  ❌")
        failures.append(f"StatevectorSampler import failed: {e}")

    # ── 5. FidelityQuantumKernel + ComputeUncompute ───────────────────────────
    try:
        from qiskit_machine_learning.kernels import FidelityQuantumKernel  # noqa
        from qiskit_machine_learning.state_fidelities import ComputeUncompute  # noqa
        if verbose:
            print(f"  FidelityQuantumKernel + ComputeUncompute  ✅")
    except Exception as e:
        if verbose:
            print(f"  FidelityQuantumKernel + ComputeUncompute  ❌")
        failures.append(f"FidelityQuantumKernel/ComputeUncompute import failed: {e}")

    # ── 6. End-to-end kernel smoke test ───────────────────────────────────────
    if not failures:   # only run if imports all passed
        try:
            result = _run_kernel_smoke_test()
            all_ok = (
                result["shape"] == (3, 3)
                and result["symmetric"]
                and result["diagonal_ok"]
                and result["values_ok"]
                and not result["has_nan"]
            )
            if verbose:
                status = "✅" if all_ok else "❌"
                print(f"  End-to-end kernel smoke (3×3)  {status}")
                if all_ok:
                    print(f"    shape={result['shape']}, "
                          f"symmetric={result['symmetric']}, "
                          f"diagonal≈1={result['diagonal_ok']}")
            if not all_ok:
                failures.append(
                    f"Kernel smoke test failed: {result}"
                )
        except Exception as e:
            if verbose:
                print(f"  End-to-end kernel smoke (3×3)  ❌")
            failures.append(f"Kernel smoke test raised: {e}")
    else:
        if verbose:
            print(f"  End-to-end kernel smoke (3×3)  ⏭  (skipped — fix imports first)")

    # ── Summary ───────────────────────────────────────────────────────────────
    if verbose:
        print()
        if not failures:
            print("  ✅  All checks passed — TrustLens-Q quantum stack is ready.")
        else:
            print(f"  ❌  {len(failures)} check(s) failed — see details below.")
        print("=" * 55)
        print()

    _report_failures(failures, verbose)
    return len(failures) == 0


def _report_failures(failures: list, verbose: bool) -> None:
    if failures and verbose:
        print("  FAILURES:")
        print("  " + "─" * 50)
        for i, msg in enumerate(failures, 1):
            indented = textwrap.indent(msg, "    ")
            print(f"  {i}. {indented.strip()}")
        print()
        print("  Quick fix:")
        print("    pip install qiskit==2.6.2 qiskit-machine-learning==0.9.1 \\")
        print("                qiskit-aer==0.18.2 qiskit-ibm-runtime==0.51.0")
        print()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Verify the TrustLens-Q Qiskit installation."
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress output; use exit code only."
    )
    args = parser.parse_args()

    passed = run_checks(verbose=not args.quiet)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
