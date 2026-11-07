"""
Quantum Kernel Diagnostics
============================
Compute rich kernel statistics and generate diagnostic plots. Operates on
any kernel matrix passed directly (not loaded from disk).

Classes:
    DiagnosticsReport   — dataclass holding all scalar and structural stats
    DiagnosticsEngine   — computes stats, saves reports, generates plots

Re-exports:
    KernelNumericalError — from quantum_kernel, so callers of this module
                           only need to import from here.

CLI:
    python -m src.quantum.quantum_diagnostics
    python -m src.quantum.quantum_diagnostics \\
        --kernel features/quantum/K_train.npy \\
        --labels features/quantum/train_labels.npy \\
        --output results/quantum/kernel_diagnostics.json \\
        --heatmap results/quantum/K_train_heatmap.png
"""

import sys
import json
import argparse
import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Re-export KernelNumericalError so callers only import from here
from quantum.quantum_kernel import KernelNumericalError  # noqa: F401

try:
    from utils import setup_logger
    logger = setup_logger("quantum.diagnostics")
except ImportError:
    logger = logging.getLogger("quantum.diagnostics")
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


# ── Diagnostics data class ─────────────────────────────────────────────────────

@dataclass
class DiagnosticsReport:
    """All diagnostic statistics for a kernel matrix."""

    # Scalar statistics
    min_value: float
    max_value: float
    mean: float
    std: float
    off_diagonal_mean: float
    off_diagonal_std: float
    off_diagonal_cv: float          # std / mean — concentration indicator

    # Square-only statistics
    symmetry_error: Optional[float]   # None for rectangular matrices
    diagonal_mean: Optional[float]
    diagonal_std: Optional[float]

    # Range check
    pct_outside_unit: float           # % of entries outside [0, 1]

    # Class-conditioned statistics (None if y not provided)
    same_class_mean: Optional[float]
    diff_class_mean: Optional[float]
    separation_ratio: Optional[float]
    kta: Optional[float]
    kta_verdict: Optional[str]        # "good" | "weak" | "poor"

    # Eigenvalue analysis (square matrices only)
    eigenvalue_min: Optional[float]
    eigenvalue_max: Optional[float]
    neg_eigenvalue_fraction: Optional[float]
    is_psd: Optional[bool]

    # Concentration and validity flags
    kernel_concentration_warning: bool
    has_nan: bool
    has_inf: bool

    # Shape metadata
    shape: Tuple[int, int]
    is_square: bool

    def to_dict(self) -> dict:
        """Convert to a plain dict suitable for JSON serialisation."""
        d = {}
        for attr in self.__dataclass_fields__:  # type: ignore[attr-defined]
            v = getattr(self, attr)
            if isinstance(v, tuple):
                d[attr] = list(v)
            elif isinstance(v, (np.integer,)):
                d[attr] = int(v)
            elif isinstance(v, (np.floating,)):
                d[attr] = float(v)
            elif isinstance(v, np.bool_):
                d[attr] = bool(v)
            else:
                d[attr] = v
        return d


# ── Diagnostics engine ────────────────────────────────────────────────────────

class DiagnosticsEngine:
    """Compute kernel diagnostics, save reports, and generate plots."""

    # KTA thresholds
    KTA_GOOD_THRESHOLD = 0.1
    KTA_WEAK_MIN = 0.0
    # Concentration: CV below this triggers warning
    CONCENTRATION_CV_THRESHOLD = 0.05

    def analyze(
        self,
        K: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> DiagnosticsReport:
        """
        Compute all diagnostics for kernel matrix K.

        Raises KernelNumericalError if K contains NaN or Inf.
        Eigenvalue and symmetry checks are skipped for rectangular K.
        KTA verdict: "good" if KTA > 0.1, "weak" if 0 < KTA <= 0.1,
                     "poor" if KTA <= 0.
        Concentration warning: True if off_diagonal_cv < 0.05.

        Args:
            K: Kernel matrix, square or rectangular.
            y: Optional binary label array (0=REAL, 1=FAKE), len == K.shape[0].

        Returns:
            DiagnosticsReport dataclass.

        Raises:
            KernelNumericalError: If K contains NaN or Inf.
        """
        has_nan = bool(np.isnan(K).any())
        has_inf = bool(np.isinf(K).any())

        if has_nan or has_inf:
            n_bad = int(np.isnan(K).sum() + np.isinf(K).sum())
            raise KernelNumericalError(
                f"Kernel matrix contains {n_bad} NaN/Inf entries. "
                "Reduce feature_dimension, reps, or increase shots."
            )

        n_rows, n_cols = K.shape
        is_square = (n_rows == n_cols)
        shape: Tuple[int, int] = (n_rows, n_cols)

        # ── Scalar statistics ─────────────────────────────────────────────────
        min_value = float(K.min())
        max_value = float(K.max())
        mean = float(K.mean())
        std = float(K.std())

        # Off-diagonal statistics
        if is_square and n_rows > 1:
            mask = ~np.eye(n_rows, dtype=bool)
            off_diag_vals = K[mask]
        else:
            # For rectangular: treat all elements as "off-diagonal"
            off_diag_vals = K.flatten()

        off_diagonal_mean = float(off_diag_vals.mean())
        off_diagonal_std = float(off_diag_vals.std())

        # CV = std / mean; guard against division by zero
        if abs(off_diagonal_mean) < 1e-12:
            off_diagonal_cv = 0.0
        else:
            off_diagonal_cv = off_diagonal_std / abs(off_diagonal_mean)

        # Concentration: low CV means all off-diagonal values are similar → no discriminability
        kernel_concentration_warning = off_diagonal_cv < self.CONCENTRATION_CV_THRESHOLD

        # ── Percentage outside [0, 1] ─────────────────────────────────────────
        n_outside = int(((K < 0.0) | (K > 1.0)).sum())
        pct_outside_unit = 100.0 * n_outside / K.size

        # ── Square-only: symmetry, diagonal, eigenvalues ──────────────────────
        symmetry_error: Optional[float] = None
        diagonal_mean: Optional[float] = None
        diagonal_std: Optional[float] = None
        eigenvalue_min: Optional[float] = None
        eigenvalue_max: Optional[float] = None
        neg_eigenvalue_fraction: Optional[float] = None
        is_psd: Optional[bool] = None

        if is_square:
            symmetry_error = float(np.abs(K - K.T).max())
            diag = np.diag(K)
            diagonal_mean = float(diag.mean())
            diagonal_std = float(diag.std())

            if n_rows >= 2:
                try:
                    eigvals = np.linalg.eigvalsh(K)
                    eigenvalue_min = float(eigvals.min())
                    eigenvalue_max = float(eigvals.max())
                    n_neg = int((eigvals < -1e-8).sum())
                    neg_eigenvalue_fraction = n_neg / len(eigvals)
                    is_psd = eigenvalue_min >= -1e-8
                except np.linalg.LinAlgError:
                    pass

        # ── Class-conditioned statistics ──────────────────────────────────────
        same_class_mean: Optional[float] = None
        diff_class_mean: Optional[float] = None
        separation_ratio: Optional[float] = None
        kta: Optional[float] = None
        kta_verdict: Optional[str] = None

        if y is not None and is_square:
            y_arr = np.asarray(y)
            assert len(y_arr) == n_rows, (
                f"Label array length {len(y_arr)} ≠ K.shape[0] ({n_rows})"
            )

            # Build same/different-class masks
            same_mask = (y_arr[:, None] == y_arr[None, :]) & ~np.eye(n_rows, dtype=bool)
            diff_mask = (y_arr[:, None] != y_arr[None, :])

            same_vals = K[same_mask]
            diff_vals = K[diff_mask]

            if len(same_vals) > 0:
                same_class_mean = float(same_vals.mean())
            if len(diff_vals) > 0:
                diff_class_mean = float(diff_vals.mean())

            if same_class_mean is not None and diff_class_mean is not None:
                denom = max(abs(diff_class_mean), 1e-12)
                separation_ratio = same_class_mean / denom

            # KTA: <K, yy^T>_F / (||K||_F * ||yy^T||_F)
            y_signed = np.where(y_arr == 0, -1.0, 1.0)
            T = np.outer(y_signed, y_signed)
            inner = float(np.sum(K * T))
            norm_K = float(np.sqrt(np.sum(K * K)))
            norm_T = float(np.sqrt(np.sum(T * T)))
            denom_kta = norm_K * norm_T
            if denom_kta < 1e-12:
                kta = 0.0
            else:
                kta = inner / denom_kta

            if kta > self.KTA_GOOD_THRESHOLD:
                kta_verdict = "good"
            elif kta > self.KTA_WEAK_MIN:
                kta_verdict = "weak"
            else:
                kta_verdict = "poor"

        return DiagnosticsReport(
            min_value=min_value,
            max_value=max_value,
            mean=mean,
            std=std,
            off_diagonal_mean=off_diagonal_mean,
            off_diagonal_std=off_diagonal_std,
            off_diagonal_cv=off_diagonal_cv,
            symmetry_error=symmetry_error,
            diagonal_mean=diagonal_mean,
            diagonal_std=diagonal_std,
            pct_outside_unit=pct_outside_unit,
            same_class_mean=same_class_mean,
            diff_class_mean=diff_class_mean,
            separation_ratio=separation_ratio,
            kta=kta,
            kta_verdict=kta_verdict,
            eigenvalue_min=eigenvalue_min,
            eigenvalue_max=eigenvalue_max,
            neg_eigenvalue_fraction=neg_eigenvalue_fraction,
            is_psd=is_psd,
            kernel_concentration_warning=kernel_concentration_warning,
            has_nan=has_nan,
            has_inf=has_inf,
            shape=shape,
            is_square=is_square,
        )

    def generate_heatmap(
        self,
        K: np.ndarray,
        output_path: str,
        title: str = "Kernel Matrix",
    ) -> None:
        """Save a kernel heatmap PNG at output_path."""
        try:
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(K, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_title(title, fontsize=13)
            ax.set_xlabel("Sample index")
            ax.set_ylabel("Sample index")
            plt.colorbar(im, ax=ax, label="Kernel value")
            plt.tight_layout()
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=100)
            plt.close(fig)
            logger.info(f"Heatmap saved to: {output_path}")
        except ImportError:
            logger.warning("matplotlib not available — heatmap not generated.")

    def generate_similarity_histogram(
        self,
        K: np.ndarray,
        y: np.ndarray,
        output_path: str,
    ) -> None:
        """Save same-class vs. different-class similarity histogram PNG."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            n = K.shape[0]
            y_arr = np.asarray(y)
            same_mask = (y_arr[:, None] == y_arr[None, :]) & ~np.eye(n, dtype=bool)
            diff_mask = y_arr[:, None] != y_arr[None, :]

            same_vals = K[same_mask]
            diff_vals = K[diff_mask]

            fig, ax = plt.subplots(figsize=(8, 5))
            bins = np.linspace(0, 1, 51)
            ax.hist(same_vals, bins=bins, alpha=0.6, label="Same class", color="steelblue")
            ax.hist(diff_vals, bins=bins, alpha=0.6, label="Different class", color="coral")
            ax.set_xlabel("Kernel similarity")
            ax.set_ylabel("Count")
            ax.set_title("Kernel Similarity Distribution by Class Pair")
            ax.legend()
            plt.tight_layout()
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=100)
            plt.close(fig)
            logger.info(f"Similarity histogram saved to: {output_path}")
        except ImportError:
            logger.warning("matplotlib not available — histogram not generated.")

    def save_report(
        self,
        report: DiagnosticsReport,
        output_path: str,
    ) -> None:
        """Serialize DiagnosticsReport to JSON at output_path."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=4)
        logger.info(f"Diagnostics report saved to: {output_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _default_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Compute kernel diagnostics from a saved .npy kernel matrix."
    )
    root = _default_root()
    parser.add_argument(
        "--kernel",
        default=str(root / "features" / "quantum" / "K_train.npy"),
        help="Path to kernel matrix .npy file.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Path to label array .npy file (optional).",
    )
    parser.add_argument(
        "--output",
        default=str(root / "results" / "quantum" / "kernel_diagnostics.json"),
        help="Output path for diagnostics JSON.",
    )
    parser.add_argument(
        "--heatmap",
        default=None,
        help="Optional path to save a kernel heatmap PNG.",
    )
    parser.add_argument(
        "--histogram",
        default=None,
        help="Optional path to save a similarity histogram PNG (requires --labels).",
    )
    args = parser.parse_args()

    print()
    print("=" * 65)
    print("  TrustLens-Q — Quantum Kernel Diagnostics")
    print("=" * 65)

    # Load kernel
    if not Path(args.kernel).exists():
        print(f"  ❌  Kernel file not found: {args.kernel}")
        sys.exit(1)

    K = np.load(args.kernel)
    print(f"  Kernel shape  : {K.shape}")

    # Load labels
    y = None
    if args.labels and Path(args.labels).exists():
        y = np.load(args.labels)
        print(f"  Labels shape  : {y.shape}")
    elif args.labels:
        print(f"  ⚠️  Labels file not found: {args.labels}")

    engine = DiagnosticsEngine()

    try:
        report = engine.analyze(K, y)
    except KernelNumericalError as e:
        print(f"  ❌  KernelNumericalError: {e}")
        sys.exit(1)

    # Print summary
    print()
    print(f"  Shape                       : {report.shape}")
    print(f"  Is square                   : {report.is_square}")
    print(f"  Min / Max                   : {report.min_value:.4f} / {report.max_value:.4f}")
    print(f"  Mean / Std                  : {report.mean:.4f} / {report.std:.4f}")
    print(f"  Off-diag mean / std / CV    : {report.off_diagonal_mean:.4f} / {report.off_diagonal_std:.4f} / {report.off_diagonal_cv:.4f}")
    print(f"  Symmetry error              : {report.symmetry_error}")
    print(f"  Diagonal mean ± std         : {report.diagonal_mean} ± {report.diagonal_std}")
    print(f"  Pct outside [0,1]           : {report.pct_outside_unit:.2f}%")
    print(f"  KTA                         : {report.kta}")
    print(f"  KTA verdict                 : {report.kta_verdict}")
    print(f"  Eigenvalue min/max          : {report.eigenvalue_min} / {report.eigenvalue_max}")
    print(f"  Is PSD                      : {report.is_psd}")
    print(f"  Neg eigenvalue fraction     : {report.neg_eigenvalue_fraction}")
    print(f"  Concentration warning       : {report.kernel_concentration_warning}")
    print(f"  Same-class mean             : {report.same_class_mean}")
    print(f"  Diff-class mean             : {report.diff_class_mean}")
    print(f"  Separation ratio            : {report.separation_ratio}")
    print()

    # Save report
    engine.save_report(report, args.output)
    print(f"  Report saved to: {args.output}")

    # Optional plots
    if args.heatmap:
        engine.generate_heatmap(K, args.heatmap, title="K_train")
        print(f"  Heatmap saved to: {args.heatmap}")

    if args.histogram and y is not None:
        engine.generate_similarity_histogram(K, y, args.histogram)
        print(f"  Histogram saved to: {args.histogram}")
    elif args.histogram and y is None:
        print("  ⚠️  Cannot generate histogram without labels.")

    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
