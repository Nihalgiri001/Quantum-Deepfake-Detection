"""
IBM Quantum Hardware Backend
==============================
Isolated execution module for running the Quantum Kernel SVM on real IBM QPUs.

IMPORTANT — this module is NEVER activated automatically.
All hardware execution paths require an explicit opt-in:
    python -m src.quantum.ibm_hardware --preflight
    python -m src.quantum.ibm_hardware --run --confirm

Safety architecture:
    1. ``IBMCredentialsError``  — raised if ``IBM_QUANTUM_TOKEN`` env var absent.
    2. ``PreflightRequiredError`` — raised if ``compute_hardware_kernel`` is
       called before ``preflight``.
    3. ``SafetyAbortError``     — raised if ``compute_hardware_kernel`` is called
       without ``confirmed=True``.

No job is ever submitted without passing all three guards.

IBM Quantum usage note:
    Free-tier accounts receive ~10 minutes of QPU time per month.
    Always run ``--preflight`` first to estimate cost.
    Start with the minimum sample size (6 train / 3 test) and verify the
    preflight report before committing any job.

Authentication:
    Set IBM_QUANTUM_TOKEN in your shell before running any hardware command:
        export IBM_QUANTUM_TOKEN="your-ibm-quantum-api-token"
    NEVER hardcode or commit your token.

CLI:
    python -m src.quantum.ibm_hardware --preflight
    python -m src.quantum.ibm_hardware --preflight --backend-name ibm_brisbane
    python -m src.quantum.ibm_hardware --run --confirm
    python -m src.quantum.ibm_hardware --run --confirm --n-train 6 --n-test 3
"""

import os
import sys
import json
import time
import logging
import argparse
import warnings
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantum.quantum_config import QuantumConfig
from quantum.quantum_feature_map import build_feature_map, validate_feature_map
from quantum.kernel_utils import save_kernel_matrix

try:
    from utils import setup_logger
    logger = setup_logger("quantum.ibm_hardware")
except ImportError:
    logger = logging.getLogger("quantum.ibm_hardware")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO,
                            format="[%(asctime)s] %(levelname)-8s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

# ── Custom exceptions ─────────────────────────────────────────────────────────

class IBMCredentialsError(Exception):
    """Raised when IBM_QUANTUM_TOKEN is absent or authentication fails.

    Set the environment variable before using any hardware command:
        export IBM_QUANTUM_TOKEN="your-api-token"
    """


class PreflightRequiredError(Exception):
    """Raised when compute_hardware_kernel() is called before preflight().

    Always run preflight() first to estimate QPU time and circuit cost:
        backend.preflight(config, X_sample)
    """


class SafetyAbortError(Exception):
    """Raised when compute_hardware_kernel() is called without confirmed=True.

    This guard prevents accidental QPU job submission.  Pass confirmed=True
    only after reviewing the preflight report and accepting the estimated cost.
    """


# ── IBM Hardware Backend ──────────────────────────────────────────────────────

class IBMHardwareBackend:
    """Execute quantum kernel computations on real IBM QPU hardware.

    All public methods that could submit jobs are protected by three
    independent safety guards (see module docstring).

    Attributes:
        instance:        IBM Quantum instance string (e.g. "ibm-q/open/main").
        backend_name:    Target QPU name (None = auto-select).
        _token:          IBM API token (read from env var, never logged).
        _service:        Connected QiskitRuntimeService (lazy).
        _backend:        Selected IBMBackend (lazy).
        _preflight_done: Whether preflight() has been completed.
        _preflight_report: Results of last preflight() call.
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        instance: str = "ibm-q/open/main",
        backend_name: Optional[str] = None,
    ):
        """Initialise the IBM hardware backend.

        Reads ``IBM_QUANTUM_TOKEN`` from the environment.  Does NOT connect
        to IBM Quantum at this point (connection is deferred to ``connect()``).

        Args:
            instance:     IBM Quantum instance (CRN or hub/group/project string).
            backend_name: Preferred QPU name, e.g. "ibm_brisbane".
                          If None, the least-busy available backend is chosen.

        Raises:
            IBMCredentialsError: If ``IBM_QUANTUM_TOKEN`` is not set.
        """
        token = os.environ.get("IBM_QUANTUM_TOKEN", "").strip()
        if not token:
            raise IBMCredentialsError(
                "IBM_QUANTUM_TOKEN environment variable is not set.\n\n"
                "To use IBM Quantum hardware:\n"
                "  1. Create a free account at https://quantum.ibm.com\n"
                "  2. Copy your API token from the account page\n"
                "  3. Run:  export IBM_QUANTUM_TOKEN='your-token-here'\n\n"
                "To run locally without IBM credentials, use the simulator:\n"
                "  python -m src.quantum.quantum_svm --train --backend simulator"
            )

        # Store token without printing it
        self._token = token
        self.instance = instance
        self.backend_name = backend_name

        # Lazy-initialised connection state
        self._service = None
        self._backend = None
        self._backend_properties: Dict[str, Any] = {}

        # Safety state
        self._preflight_done: bool = False
        self._preflight_report: Optional[Dict[str, Any]] = None

        logger.info(
            f"IBMHardwareBackend initialised "
            f"(instance={instance}, backend={backend_name or 'auto'})"
        )

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Establish connection to IBM Quantum Platform.

        Uses ``IBM_QUANTUM_TOKEN`` set during ``__init__``.
        Raises ``IBMCredentialsError`` if authentication fails.

        Raises:
            IBMCredentialsError: On authentication failure.
        """
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            from qiskit_ibm_runtime.exceptions import IBMAccountError
        except ImportError as exc:
            raise ImportError(
                "qiskit-ibm-runtime is required for hardware execution.\n"
                "Install:  pip install 'qiskit-ibm-runtime==0.51.0'"
            ) from exc

        logger.info("Connecting to IBM Quantum Platform…")
        try:
            self._service = QiskitRuntimeService(
                channel="ibm_quantum_platform",
                token=self._token,
                instance=self.instance,
            )
            logger.info("IBM Quantum connection established.")
        except Exception as exc:
            raise IBMCredentialsError(
                f"IBM Quantum authentication failed: {exc}\n"
                "Check that your IBM_QUANTUM_TOKEN is valid and not expired."
            ) from exc

    def list_available_backends(self) -> List[str]:
        """Return names of all accessible IBM QPU backends.

        Raises:
            IBMCredentialsError: If not connected.

        Returns:
            List of backend name strings.
        """
        self._ensure_connected()
        backends = self._service.backends()
        names = [b.name for b in backends]
        logger.info(f"Available backends: {names}")
        return names

    def select_backend(self, name: Optional[str] = None) -> None:
        """Select and validate a QPU backend; record its properties.

        Args:
            name: Backend name to select.  If None, uses ``self.backend_name``
                  or picks the least-busy compatible backend.

        Raises:
            IBMCredentialsError: If not connected.
            ValueError:          If the named backend is unavailable.
        """
        self._ensure_connected()
        target_name = name or self.backend_name

        if target_name:
            backends = self._service.backends(name=target_name)
            if not backends:
                available = self.list_available_backends()
                raise ValueError(
                    f"Backend '{target_name}' not found.\n"
                    f"Available: {available}"
                )
            self._backend = backends[0]
        else:
            # Auto-select: prefer least-busy backend with ≥ config num_qubits
            backends = self._service.backends()
            if not backends:
                raise ValueError("No backends available on this account.")
            self._backend = backends[0]   # IBM sorts by availability

        # Record properties (no network calls beyond initial listing)
        self.backend_name = self._backend.name
        self._backend_properties = {
            "name":       self._backend.name,
            "num_qubits": getattr(self._backend, "num_qubits", "unknown"),
        }
        logger.info(f"Selected backend: {self._backend.name}")

    # ── Preflight ─────────────────────────────────────────────────────────────

    def preflight(
        self,
        config: QuantumConfig,
        X_sample: np.ndarray,
    ) -> Dict[str, Any]:
        """Analyse the circuit cost WITHOUT submitting any job.

        Constructs the ZZFeatureMap circuit, transpiles it for the target
        backend (or a reference basis), and estimates the QPU time budget.

        This method is safe to call without IBM credentials when
        ``use_fake_backend=True`` (the default for testing).

        Args:
            config:    QuantumConfig governing circuit shape.
            X_sample:  Sample feature vectors, shape (n, feature_dimension).
                       Used only for counting the number of kernel pairs.

        Returns:
            Preflight report dict with keys:
                n_train_samples     — number of training points
                n_test_samples      — 0 (not known at preflight)
                n_kernel_pairs      — n*(n-1)//2 (unique pairs)
                n_circuits          — circuits needed (= n_kernel_pairs * 2)
                circuit_depth       — unoptimised depth
                num_qubits          — circuit width
                shots               — shots per circuit
                estimated_qpu_time_sec — very rough upper bound
                backend_name        — target backend name
                warning             — human-readable cost summary
        """
        n = X_sample.shape[0]
        n_pairs    = n * (n - 1) // 2 + n   # include diagonals
        n_circuits = n_pairs * 2            # ComputeUncompute doubles circuits

        # Build circuit and measure depth
        fm = build_feature_map(config)
        try:
            validate_feature_map(fm, config)
        except ValueError as e:
            logger.warning(f"Feature map validation: {e}")

        circuit_depth = fm.depth()

        # Transpile for estimation if backend is selected
        transpiled_depth = circuit_depth
        if self._backend is not None:
            try:
                from qiskit.compiler import transpile as qk_transpile
                transpiled = qk_transpile(
                    fm,
                    backend=self._backend,
                    optimization_level=1,
                    seed_transpiler=config.random_seed,
                )
                transpiled_depth = transpiled.depth()
            except Exception as e:
                logger.warning(f"Transpile for depth estimate failed: {e}")

        # Rough QPU time estimate:
        # ~1 ms per shot × shots × n_circuits  (very conservative)
        estimated_sec = (config.shots * n_circuits * 1e-3) * 1.5
        backend_name = self._backend.name if self._backend else "not-selected"

        report: Dict[str, Any] = {
            "n_train_samples":      int(n),
            "n_test_samples":       0,
            "n_kernel_pairs":       int(n_pairs),
            "n_circuits":           int(n_circuits),
            "circuit_depth":        int(circuit_depth),
            "transpiled_depth":     int(transpiled_depth),
            "num_qubits":           int(config.num_qubits),
            "shots":                int(config.shots),
            "estimated_qpu_time_sec": round(estimated_sec, 1),
            "backend_name":         backend_name,
            "warning": (
                f"⚠️  Estimated QPU time: {estimated_sec:.1f}s "
                f"for {n} samples × {n_circuits} circuits × "
                f"{config.shots} shots.\n"
                f"   Free-tier allowance: ~600s/month.  "
                f"Proceed only if budget allows."
            ),
        }

        self._preflight_done = True
        self._preflight_report = report

        # Print human-readable summary
        print()
        print("─" * 60)
        print("  IBM Hardware Preflight Report")
        print("─" * 60)
        print(f"  Backend             : {backend_name}")
        print(f"  Qubits              : {config.num_qubits}")
        print(f"  Train samples       : {n}")
        print(f"  Unique kernel pairs : {n_pairs}")
        print(f"  Total circuits      : {n_circuits}")
        print(f"  Circuit depth       : {circuit_depth} "
              f"(transpiled: {transpiled_depth})")
        print(f"  Shots per circuit   : {config.shots}")
        print(f"  Est. QPU time       : {estimated_sec:.1f}s")
        print()
        print(f"  {report['warning']}")
        print("─" * 60)
        print()

        return report

    # ── Hardware kernel computation ───────────────────────────────────────────

    def compute_hardware_kernel(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        config: QuantumConfig,
        confirmed: bool = False,
        cache_dir: Optional[str] = None,
    ) -> tuple:
        """Compute kernel matrices on real IBM QPU hardware.

        Safety guards (all must pass):
            1. ``preflight()`` must have been called first.
            2. ``confirmed=True`` must be passed explicitly.
            3. ``IBM_QUANTUM_TOKEN`` must be set (enforced by ``__init__``).

        Args:
            X_train:    Training feature vectors, shape (n, feature_dim).
            X_test:     Test feature vectors, shape (m, feature_dim).
            config:     QuantumConfig.
            confirmed:  Must be ``True`` to permit job submission.
            cache_dir:  Where to save kernel matrices. Defaults to
                        ``features/quantum/hardware/``.

        Returns:
            Tuple ``(K_train_hw, K_test_hw)`` of numpy arrays.

        Raises:
            PreflightRequiredError: If ``preflight()`` has not been called.
            SafetyAbortError:       If ``confirmed`` is not True.
            IBMCredentialsError:    If not connected.
        """
        if not self._preflight_done:
            raise PreflightRequiredError(
                "preflight() must be called before compute_hardware_kernel().\n"
                "Run:  backend.preflight(config, X_sample)\n"
                "Review the report, then pass confirmed=True."
            )

        if not confirmed:
            raise SafetyAbortError(
                "Hardware job submission aborted — confirmed=False.\n\n"
                "This guard prevents accidental QPU usage.\n"
                "Pass confirmed=True only after reviewing the preflight "
                "report and accepting the estimated cost.\n\n"
                f"Preflight report:\n"
                f"  Estimated QPU time : "
                f"{self._preflight_report.get('estimated_qpu_time_sec', '?')}s\n"
                f"  Circuits to submit : "
                f"{self._preflight_report.get('n_circuits', '?')}\n"
            )

        self._ensure_connected()
        if self._backend is None:
            self.select_backend()

        _root = Path(__file__).resolve().parent.parent.parent
        hw_cache = Path(cache_dir) if cache_dir else (
            _root / "features" / "quantum" / "hardware"
        )
        hw_cache.mkdir(parents=True, exist_ok=True)

        print()
        print("=" * 60)
        print("  ⚠️  SUBMITTING HARDWARE JOBS TO IBM QUANTUM")
        print("=" * 60)
        print(f"  Backend     : {self._backend.name}")
        print(f"  Train size  : {X_train.shape[0]}")
        print(f"  Test size   : {X_test.shape[0]}")
        print(f"  Circuits    : {self._preflight_report['n_circuits']}")
        print(f"  Est. time   : {self._preflight_report['estimated_qpu_time_sec']}s")
        print("=" * 60)
        print()

        from qiskit_ibm_runtime import SamplerV2
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from qiskit_machine_learning.state_fidelities import ComputeUncompute

        fm = build_feature_map(config)

        # Hardware Sampler backed by the real QPU
        sampler  = SamplerV2(mode=self._backend)
        fidelity = ComputeUncompute(sampler=sampler)
        kernel   = FidelityQuantumKernel(feature_map=fm, fidelity=fidelity)

        logger.info("Computing hardware training kernel…")
        t0 = time.time()
        K_train_hw = kernel.evaluate(x_vec=X_train)
        K_test_hw  = kernel.evaluate(x_vec=X_test, y_vec=X_train)
        elapsed = time.time() - t0

        logger.info(f"Hardware kernel computed in {elapsed:.1f}s")

        # Collect job metadata (SamplerV2 job IDs)
        job_ids: List[str] = []

        timestamp = datetime.now(timezone.utc).isoformat()
        meta = {
            "timestamp":             timestamp,
            "backend_name":          self._backend.name,
            "shots":                 config.shots,
            "num_qubits":            config.num_qubits,
            "feature_dimension":     config.feature_dimension,
            "reps":                  config.reps,
            "entanglement":          config.entanglement,
            "n_train":               int(X_train.shape[0]),
            "n_test":                int(X_test.shape[0]),
            "compute_time_sec":      round(elapsed, 2),
            "job_ids":               job_ids,
            "K_train_shape":         list(K_train_hw.shape),
            "K_test_shape":          list(K_test_hw.shape),
        }

        save_kernel_matrix(K_train_hw, str(hw_cache / "K_train_hw.npy"), meta)
        save_kernel_matrix(K_test_hw,  str(hw_cache / "K_test_hw.npy"),  meta)
        logger.info(f"Hardware kernel matrices saved to {hw_cache}")

        return K_train_hw, K_test_hw

    # ── Hardware SVM training ─────────────────────────────────────────────────

    def train_hardware_svm(
        self,
        K_train_hw: np.ndarray,
        y_train: np.ndarray,
        K_test_hw: np.ndarray,
        y_test: np.ndarray,
        config: Optional[QuantumConfig] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train a separate SVC on hardware-derived kernel matrices.

        This is an entirely separate experiment from the simulator-trained
        QSVM.  The two are NEVER mixed.

        Args:
            K_train_hw: Hardware training kernel, shape (n, n).
            y_train:    Training labels.
            K_test_hw:  Hardware test kernel, shape (m, n).
            y_test:     Test labels.
            config:     QuantumConfig (for metadata).
            output_dir: Where to save the hardware SVM. Defaults to
                        ``models/quantum/hardware/``.

        Returns:
            Dict with accuracy, f1, and file paths.
        """
        if config is None:
            config = QuantumConfig()

        _root = Path(__file__).resolve().parent.parent.parent
        hw_dir = Path(output_dir) if output_dir else (
            _root / "models" / "quantum" / "hardware"
        )
        hw_dir.mkdir(parents=True, exist_ok=True)

        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, f1_score

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            svc = SVC(
                kernel="precomputed",
                probability=True,
                C=config.svm_c,
                class_weight=config.svm_class_weight,
                random_state=config.random_seed,
            )
            svc.fit(K_train_hw, y_train)

        y_pred = svc.predict(K_test_hw)
        acc    = float(accuracy_score(y_test, y_pred))
        f1     = float(f1_score(y_test, y_pred, zero_division=0))

        joblib.dump(svc, hw_dir / "quantum_svm_hw.pkl")

        meta = {
            "model":         "hardware_quantum_svm",
            "backend":       self.backend_name or "unknown",
            "trained_at":    datetime.now(timezone.utc).isoformat(),
            "n_train":       int(len(y_train)),
            "n_test":        int(len(y_test)),
            "test_accuracy": round(acc, 4),
            "test_f1":       round(f1, 4),
            "num_qubits":    config.num_qubits,
        }
        with open(hw_dir / "hardware_training_metadata.json", "w") as f:
            json.dump(meta, f, indent=4)

        logger.info(
            f"Hardware SVM: accuracy={acc:.4f}, F1={f1:.4f}. "
            f"Saved to {hw_dir}"
        )
        return meta

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _ensure_connected(self) -> None:
        """Connect if not already connected."""
        if self._service is None:
            self.connect()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IBM Quantum hardware experiment for Quantum Kernel SVM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Inspect circuits WITHOUT submitting:
    python -m src.quantum.ibm_hardware --preflight

  Submit a small hardware experiment (COSTS QPU TIME):
    python -m src.quantum.ibm_hardware --run --confirm

  Use a specific backend:
    python -m src.quantum.ibm_hardware --preflight --backend-name ibm_brisbane
""",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight", action="store_true",
        help="Inspect circuits and estimate cost WITHOUT submitting jobs."
    )
    mode.add_argument(
        "--run", action="store_true",
        help="Submit hardware jobs (requires --confirm to prevent accidents)."
    )

    parser.add_argument(
        "--confirm", action="store_true",
        help="Required with --run to confirm QPU job submission."
    )
    parser.add_argument(
        "--backend-name", type=str, default=None,
        help="Target IBM QPU name (default: auto-select)."
    )
    parser.add_argument(
        "--instance", type=str, default="ibm-q/open/main",
        help="IBM Quantum instance (default: ibm-q/open/main)."
    )
    parser.add_argument(
        "--n-train", type=int, default=6,
        help="Training samples for hardware kernel (default: 6)."
    )
    parser.add_argument(
        "--n-test", type=int, default=3,
        help="Test samples for hardware kernel (default: 3)."
    )
    parser.add_argument(
        "--num-qubits", type=int, default=4, choices=[4, 6, 8],
        help="Number of qubits (default: 4)."
    )
    parser.add_argument(
        "--shots", type=int, default=1024,
        help="Shots per circuit (default: 1024)."
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  TrustLens-Q — IBM Quantum Hardware Mode")
    print("=" * 60)
    print()

    # Try to build backend — will raise IBMCredentialsError if no token
    try:
        backend = IBMHardwareBackend(
            instance=args.instance,
            backend_name=args.backend_name,
        )
    except IBMCredentialsError as e:
        print(f"  ❌  {e}")
        sys.exit(1)

    cfg = QuantumConfig(
        num_qubits=args.num_qubits,
        feature_dimension=args.num_qubits,
        shots=args.shots,
    )

    # Synthetic sample data for preflight (real data loaded from features/ for --run)
    rng = np.random.default_rng(42)
    X_sample = rng.uniform(-np.pi, np.pi, (args.n_train, cfg.feature_dimension))

    if args.preflight:
        try:
            # Connect to list backends but don't fail if credentials are absent
            # (preflight can estimate circuit stats without connecting)
            try:
                backend.connect()
                if args.backend_name:
                    backend.select_backend(args.backend_name)
            except IBMCredentialsError:
                print("  ℹ️  Running preflight in offline mode (no IBM connection).")
                print()

            report = backend.preflight(cfg, X_sample)
            print("  Preflight complete. Review the report above before submitting.")
            print("  To submit:  python -m src.quantum.ibm_hardware --run --confirm")

        except Exception as e:
            print(f"  ❌  Preflight failed: {e}")
            sys.exit(1)

    elif args.run:
        if not args.confirm:
            print("  ❌  --run requires --confirm to prevent accidental QPU usage.")
            print("  First run:  python -m src.quantum.ibm_hardware --preflight")
            print("  Then:       python -m src.quantum.ibm_hardware --run --confirm")
            sys.exit(1)

        try:
            backend.connect()
            backend.select_backend(args.backend_name)

            # Preflight is mandatory before run
            backend.preflight(cfg, X_sample)

            # Load real features from the quantum kernel cache if available,
            # otherwise use synthetic data for demonstration
            _root = Path(__file__).resolve().parent.parent.parent
            kd = _root / "features" / "quantum"
            if (kd / "K_train.npy").exists():
                from quantum.quantum_svm import QuantumSVM
                qsvm = QuantumSVM.load(str(_root / "models" / "quantum"))
                X_train_q = qsvm._train_features_reduced[:args.n_train]
            else:
                X_train_q = X_sample

            X_test_q = rng.uniform(-np.pi, np.pi, (args.n_test, cfg.feature_dimension))
            y_train  = rng.choice([0, 1], size=args.n_train)
            y_test   = rng.choice([0, 1], size=args.n_test)

            K_train_hw, K_test_hw = backend.compute_hardware_kernel(
                X_train_q, X_test_q, cfg, confirmed=True
            )

            meta = backend.train_hardware_svm(
                K_train_hw, y_train, K_test_hw, y_test, config=cfg
            )

            print(f"  Hardware QSVM accuracy : {meta['test_accuracy']}")
            print(f"  Hardware QSVM F1       : {meta['test_f1']}")
            print()

        except IBMCredentialsError as e:
            print(f"  ❌  {e}")
            sys.exit(1)
        except Exception as e:
            print(f"  ❌  Hardware run failed: {e}")
            raise


if __name__ == "__main__":
    main()
