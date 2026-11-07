"""
Quantum Audit Module
=====================
Read-only inspection of saved QSVM artefacts. Produces a structured JSON
report without writing any model file or modifying any artefact.

Checks performed:
  1.  K_train shape is (n, n) — square.
  2.  K_cross shape is (m, n) — n matches K_train.
  3.  Symmetry error max|K_train − K_train^T| — warn if > 1e-6.
  4.  Diagonal deviation max|diag(K_train) − 1| — warn if > 1e-3.
  5.  Value range — warn if any entry outside [0, 1].
  6.  SVC.classes_[1] == 1 (label ordering).
  7.  Mini cross-check: predict_proba[:, 1] higher for FAKE.
  8.  n_train_samples > 0 in feature_scaling.json (leakage guard).
  9.  Label inversion detection.
  10. KTA on loaded K_train + labels (if labels available).
  11. Writes audit_report.json to results_dir.

Usage:
    python -m src.quantum.quantum_audit
    python -m src.quantum.quantum_audit --models-dir models/quantum
                                        --features-dir features/quantum
"""

import sys
import json
import argparse
import logging
import numpy as np
import joblib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from utils import setup_logger
    logger = setup_logger("quantum.audit")
except ImportError:
    logger = logging.getLogger("quantum.audit")
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class AuditCheck:
    """Result of a single audit check."""
    name: str
    passed: bool
    value: Any
    warning_threshold: Optional[Any]
    message: str


@dataclass
class AuditReport:
    """Structured result from a single audit run."""
    checks: Dict[str, AuditCheck] = field(default_factory=dict)
    all_passed: bool = False
    label_inversion_detected: bool = False
    timestamp: str = ""

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON output."""
        checks_serialised = {}
        for name, chk in self.checks.items():
            checks_serialised[name] = {
                "name": chk.name,
                "passed": chk.passed,
                "value": _to_json_safe(chk.value),
                "warning_threshold": _to_json_safe(chk.warning_threshold),
                "message": chk.message,
            }
        return {
            "all_passed": self.all_passed,
            "label_inversion_detected": self.label_inversion_detected,
            "timestamp": self.timestamp,
            "checks": checks_serialised,
        }


def _to_json_safe(v: Any) -> Any:
    """Convert numpy types to plain Python types for JSON serialisation."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, bool):
        return v
    return v


# ── KTA helper ────────────────────────────────────────────────────────────────

def _compute_kta(K: np.ndarray, y: np.ndarray) -> float:
    """Kernel-Target Alignment: <K, yy^T>_F / (||K||_F * ||yy^T||_F)."""
    y_col = y.reshape(-1, 1).astype(float)
    # Convert 0/1 labels to -1/+1 for KTA
    y_signed = np.where(y_col == 0, -1.0, 1.0)
    T = y_signed @ y_signed.T  # ideal kernel
    inner = float(np.sum(K * T))
    norm_K = float(np.sqrt(np.sum(K * K)))
    norm_T = float(np.sqrt(np.sum(T * T)))
    denom = norm_K * norm_T
    if denom < 1e-12:
        return 0.0
    return inner / denom


# ── Core audit function ────────────────────────────────────────────────────────

def run_audit(
    models_dir: str,
    features_quantum_dir: str,
    results_dir: str = "results/quantum",
) -> AuditReport:
    """
    Inspect saved QSVM artefacts without modifying any file.

    Args:
        models_dir:            Directory with quantum_svm.pkl, feature_scaling.json.
        features_quantum_dir:  Directory with K_train.npy, train_labels.npy.
        results_dir:           Output directory for audit_report.json.

    Returns:
        AuditReport with all findings.

    Raises:
        FileNotFoundError: When quantum_svm.pkl or feature_scaling.json are missing.
    """
    md = Path(models_dir)
    fd = Path(features_quantum_dir)
    rd = Path(results_dir)

    # ── Pre-flight: required files must exist ─────────────────────────────────
    svc_path = md / "quantum_svm.pkl"
    scaling_path = md / "feature_scaling.json"

    if not svc_path.exists():
        raise FileNotFoundError(
            f"quantum_svm.pkl not found in {md}. "
            "Run the QSVM training pipeline first."
        )
    if not scaling_path.exists():
        raise FileNotFoundError(
            f"feature_scaling.json not found in {md}. "
            "Run the QSVM training pipeline first."
        )

    report = AuditReport(
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    checks: Dict[str, AuditCheck] = {}

    # ── Load artefacts (read-only) ─────────────────────────────────────────────
    logger.info(f"Loading artefacts from {md} and {fd}")

    svc = joblib.load(str(svc_path))

    with open(str(scaling_path), "r") as f:
        scaling_meta = json.load(f)

    # Load kernel matrices if available
    K_train: Optional[np.ndarray] = None
    K_cross: Optional[np.ndarray] = None
    train_labels: Optional[np.ndarray] = None

    k_train_path = fd / "K_train.npy"
    k_cross_path = fd / "K_val.npy"  # K_val serves as the cross-kernel for audit
    labels_path = fd / "train_labels.npy"

    if k_train_path.exists():
        K_train = np.load(str(k_train_path))
    if k_cross_path.exists():
        K_cross = np.load(str(k_cross_path))
    if labels_path.exists():
        train_labels = np.load(str(labels_path))

    # ── Check 1: K_train shape is square ─────────────────────────────────────
    if K_train is not None:
        is_square = K_train.ndim == 2 and K_train.shape[0] == K_train.shape[1]
        checks["k_train_square"] = AuditCheck(
            name="k_train_square",
            passed=is_square,
            value=list(K_train.shape),
            warning_threshold=None,
            message=(
                f"K_train is square ({K_train.shape[0]}×{K_train.shape[1]})"
                if is_square
                else f"K_train is NOT square: shape={K_train.shape}"
            ),
        )
    else:
        checks["k_train_square"] = AuditCheck(
            name="k_train_square",
            passed=False,
            value=None,
            warning_threshold=None,
            message=f"K_train.npy not found at {k_train_path}",
        )

    # ── Check 2: K_cross shape (m, n) — n matches K_train ────────────────────
    if K_cross is not None and K_train is not None:
        n_train = K_train.shape[0]
        n_cross_cols = K_cross.shape[1] if K_cross.ndim == 2 else -1
        cross_ok = K_cross.ndim == 2 and n_cross_cols == n_train
        checks["k_cross_shape"] = AuditCheck(
            name="k_cross_shape",
            passed=cross_ok,
            value=list(K_cross.shape),
            warning_threshold=n_train,
            message=(
                f"K_cross shape {K_cross.shape} consistent with K_train cols={n_train}"
                if cross_ok
                else f"K_cross shape {K_cross.shape} — cols ({n_cross_cols}) ≠ K_train n ({n_train})"
            ),
        )
    else:
        cross_found = K_cross is not None
        checks["k_cross_shape"] = AuditCheck(
            name="k_cross_shape",
            passed=False,
            value=None,
            warning_threshold=None,
            message=(
                "K_val.npy found but K_train.npy missing — cannot compare shapes"
                if cross_found else
                f"K_val.npy not found at {k_cross_path}"
            ),
        )

    # ── Check 3: Symmetry error ───────────────────────────────────────────────
    if K_train is not None and K_train.ndim == 2 and K_train.shape[0] == K_train.shape[1]:
        sym_err = float(np.abs(K_train - K_train.T).max())
        sym_ok = sym_err <= 1e-6
        checks["symmetry_error"] = AuditCheck(
            name="symmetry_error",
            passed=sym_ok,
            value=sym_err,
            warning_threshold=1e-6,
            message=(
                f"Symmetry error {sym_err:.2e} ≤ 1e-6 ✓"
                if sym_ok else
                f"Symmetry error {sym_err:.2e} > 1e-6 — WARNING"
            ),
        )
    else:
        checks["symmetry_error"] = AuditCheck(
            name="symmetry_error",
            passed=False,
            value=None,
            warning_threshold=1e-6,
            message="Cannot compute symmetry error — K_train not square or missing",
        )

    # ── Check 4: Diagonal deviation ──────────────────────────────────────────
    if K_train is not None and K_train.ndim == 2 and K_train.shape[0] == K_train.shape[1]:
        diag = np.diag(K_train)
        diag_err = float(np.abs(diag - 1.0).max())
        diag_ok = diag_err <= 1e-3
        checks["diagonal_deviation"] = AuditCheck(
            name="diagonal_deviation",
            passed=diag_ok,
            value=diag_err,
            warning_threshold=1e-3,
            message=(
                f"Diagonal deviation {diag_err:.4f} ≤ 1e-3 ✓"
                if diag_ok else
                f"Diagonal deviation {diag_err:.4f} > 1e-3 — WARNING"
            ),
        )
    else:
        checks["diagonal_deviation"] = AuditCheck(
            name="diagonal_deviation",
            passed=False,
            value=None,
            warning_threshold=1e-3,
            message="Cannot compute diagonal deviation — K_train not square or missing",
        )

    # ── Check 5: Value range [0, 1] ───────────────────────────────────────────
    if K_train is not None:
        n_out = int(((K_train < -1e-6) | (K_train > 1.0 + 1e-6)).sum())
        range_ok = n_out == 0
        checks["value_range"] = AuditCheck(
            name="value_range",
            passed=range_ok,
            value={"entries_outside_unit": n_out, "min": float(K_train.min()), "max": float(K_train.max())},
            warning_threshold=0,
            message=(
                f"All K_train entries in [0,1] ✓"
                if range_ok else
                f"{n_out} entries outside [0,1] — WARNING (min={K_train.min():.4f}, max={K_train.max():.4f})"
            ),
        )
    else:
        checks["value_range"] = AuditCheck(
            name="value_range",
            passed=False,
            value=None,
            warning_threshold=0,
            message="Cannot check value range — K_train missing",
        )

    # ── Check 6: SVC.classes_[1] == 1 ────────────────────────────────────────
    try:
        classes = svc.classes_.tolist()
        label_order_ok = len(classes) >= 2 and classes[1] == 1
        checks["label_ordering"] = AuditCheck(
            name="label_ordering",
            passed=label_order_ok,
            value=classes,
            warning_threshold=None,
            message=(
                f"SVC.classes_ = {classes}, classes_[1] == 1 ✓"
                if label_order_ok else
                f"SVC.classes_ = {classes} — WARNING: classes_[1] ≠ 1"
            ),
        )
    except Exception as e:
        checks["label_ordering"] = AuditCheck(
            name="label_ordering",
            passed=False,
            value=None,
            warning_threshold=None,
            message=f"Could not inspect SVC.classes_: {e}",
        )

    # ── Check 7: Mini cross-check (predict_proba[:, 1] higher for FAKE) ──────
    if K_train is not None and train_labels is not None:
        try:
            # Use a small subset of K_train rows as "cross-kernel" proxy
            n_check = min(20, K_train.shape[0])
            K_mini = K_train[:n_check, :]
            y_mini = train_labels[:n_check]
            proba_mini = svc.predict_proba(K_mini)[:, 1]  # P(FAKE)

            fake_mask = y_mini == 1
            real_mask = y_mini == 0

            if fake_mask.sum() > 0 and real_mask.sum() > 0:
                fake_proba_mean = float(proba_mini[fake_mask].mean())
                real_proba_mean = float(proba_mini[real_mask].mean())
                cross_check_ok = fake_proba_mean > real_proba_mean
                checks["proba_direction"] = AuditCheck(
                    name="proba_direction",
                    passed=cross_check_ok,
                    value={"fake_proba_mean": fake_proba_mean, "real_proba_mean": real_proba_mean},
                    warning_threshold=None,
                    message=(
                        f"FAKE proba ({fake_proba_mean:.3f}) > REAL proba ({real_proba_mean:.3f}) ✓"
                        if cross_check_ok else
                        f"FAKE proba ({fake_proba_mean:.3f}) ≤ REAL proba ({real_proba_mean:.3f}) — possible label inversion"
                    ),
                )
                report.label_inversion_detected = not cross_check_ok
            else:
                checks["proba_direction"] = AuditCheck(
                    name="proba_direction",
                    passed=True,
                    value={"note": "insufficient class diversity in mini-subset"},
                    warning_threshold=None,
                    message="Skipped — mini-subset has only one class",
                )
        except Exception as e:
            checks["proba_direction"] = AuditCheck(
                name="proba_direction",
                passed=False,
                value=None,
                warning_threshold=None,
                message=f"predict_proba failed: {e}",
            )
    else:
        checks["proba_direction"] = AuditCheck(
            name="proba_direction",
            passed=False,
            value=None,
            warning_threshold=None,
            message="K_train or train_labels missing — cannot perform cross-check",
        )

    # ── Check 8: n_train_samples > 0 in feature_scaling.json ────────────────
    n_train_samples = scaling_meta.get("n_train_samples", 0)
    leakage_ok = int(n_train_samples) > 0
    checks["leakage_guard"] = AuditCheck(
        name="leakage_guard",
        passed=leakage_ok,
        value=n_train_samples,
        warning_threshold=0,
        message=(
            f"n_train_samples={n_train_samples} in feature_scaling.json ✓"
            if leakage_ok else
            f"n_train_samples={n_train_samples} in feature_scaling.json — leakage risk"
        ),
    )

    # ── Check 9: Label inversion detection ───────────────────────────────────
    # Already done in check 7; consolidate here
    label_inv = report.label_inversion_detected
    checks["label_inversion"] = AuditCheck(
        name="label_inversion",
        passed=not label_inv,
        value=label_inv,
        warning_threshold=None,
        message=(
            "No label inversion detected ✓"
            if not label_inv else
            "Label inversion detected — FAKE/REAL predictions may be swapped"
        ),
    )

    # ── Check 10: KTA score ───────────────────────────────────────────────────
    if K_train is not None and train_labels is not None:
        try:
            kta = _compute_kta(K_train, train_labels)
            kta_ok = kta > 0
            checks["kta_score"] = AuditCheck(
                name="kta_score",
                passed=kta_ok,
                value=round(kta, 6),
                warning_threshold=0.1,
                message=(
                    f"KTA = {kta:.4f} > 0 ✓"
                    + (" (good alignment)" if kta > 0.1 else " (weak — near-random)")
                    if kta_ok else
                    f"KTA = {kta:.4f} ≤ 0 — poor kernel alignment"
                ),
            )
        except Exception as e:
            checks["kta_score"] = AuditCheck(
                name="kta_score",
                passed=False,
                value=None,
                warning_threshold=0.1,
                message=f"KTA computation failed: {e}",
            )
    else:
        checks["kta_score"] = AuditCheck(
            name="kta_score",
            passed=False,
            value=None,
            warning_threshold=0.1,
            message="K_train or train_labels missing — cannot compute KTA",
        )

    # ── Check 11: Completeness (all required artefact files present) ──────────
    required_files = [
        md / "quantum_svm.pkl",
        md / "feature_scaling.json",
        fd / "K_train.npy",
        fd / "train_labels.npy",
    ]
    missing_files = [str(f) for f in required_files if not f.exists()]
    artefacts_ok = len(missing_files) == 0
    checks["artefact_completeness"] = AuditCheck(
        name="artefact_completeness",
        passed=artefacts_ok,
        value={"missing": missing_files},
        warning_threshold=None,
        message=(
            "All required artefact files present ✓"
            if artefacts_ok else
            f"Missing artefacts: {missing_files}"
        ),
    )

    # ── Finalise report ───────────────────────────────────────────────────────
    assert len(checks) == 11, f"Expected 11 checks, got {len(checks)}"

    report.checks = checks
    report.all_passed = all(c.passed for c in checks.values())

    # ── Write audit_report.json ───────────────────────────────────────────────
    rd.mkdir(parents=True, exist_ok=True)
    report_path = rd / "audit_report.json"
    with open(str(report_path), "w") as f:
        json.dump(report.to_dict(), f, indent=4)

    logger.info(f"Audit report written to: {report_path}")
    logger.info(f"All checks passed: {report.all_passed}")

    return report


# ── CLI ────────────────────────────────────────────────────────────────────────

def _default_project_root() -> Path:
    """Walk up from src/quantum/ to project root."""
    return Path(__file__).resolve().parent.parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Read-only audit of saved QSVM artefacts."
    )
    root = _default_project_root()
    parser.add_argument(
        "--models-dir",
        default=str(root / "models" / "quantum"),
        help="Directory containing quantum_svm.pkl and feature_scaling.json.",
    )
    parser.add_argument(
        "--features-dir",
        default=str(root / "features" / "quantum"),
        help="Directory containing K_train.npy, train_labels.npy.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(root / "results" / "quantum"),
        help="Output directory for audit_report.json.",
    )
    args = parser.parse_args()

    print()
    print("=" * 65)
    print("  TrustLens-Q — QSVM Artefact Audit")
    print("=" * 65)
    print(f"  Models dir   : {args.models_dir}")
    print(f"  Features dir : {args.features_dir}")
    print(f"  Results dir  : {args.results_dir}")
    print()

    try:
        report = run_audit(
            models_dir=args.models_dir,
            features_quantum_dir=args.features_dir,
            results_dir=args.results_dir,
        )
    except FileNotFoundError as e:
        print(f"  ❌  FileNotFoundError: {e}")
        sys.exit(1)

    # Print summary
    print(f"  {'CHECK':<30} {'PASSED':>7}  VALUE")
    print("  " + "-" * 55)
    for name, chk in report.checks.items():
        status = "✅ yes" if chk.passed else "❌  no"
        print(f"  {name:<30} {status:>7}  {chk.message[:60]}")

    print()
    print(f"  Label inversion detected : {report.label_inversion_detected}")
    print(f"  All checks passed        : {report.all_passed}")
    print(f"  Timestamp                : {report.timestamp}")
    print()
    print(f"  Report written to: {Path(args.results_dir) / 'audit_report.json'}")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
