"""
Quantum Configuration Module
==============================
Single source of truth for all quantum pipeline parameters.

All tunable quantum parameters — qubit count, circuit depth, shots,
sample budgets, and preprocessing ranges — live here so that no magic
numbers are scattered across the quantum package.

Usage:
    from quantum.quantum_config import QuantumConfig
    cfg = QuantumConfig()
    cfg.save_json("models/quantum/quantum_config.json")
"""

import json
import math
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Tuple, Optional


@dataclass
class QuantumConfig:
    """Central configuration for the quantum kernel SVM pipeline.

    Attributes:
        num_qubits:          Number of qubits (= feature dimension after PCA).
        feature_dimension:   Must equal num_qubits. PCA reduces 512 → this.
        reps:                ZZFeatureMap circuit repetitions.
        entanglement:        Circuit entanglement pattern ("linear", "full", "circular").
        shots:               Shot count per circuit execution on simulator.
        random_seed:         Global random seed for reproducibility.
        backend:             Execution backend ("simulator" or "ibm_hardware").
        max_train_samples:   Max training samples (QSVM kernel is O(n^2)).
        max_val_samples:     Max validation samples for evaluation.
        max_test_samples:    Max test samples for evaluation.
        feature_range:       Target range for post-PCA feature scaling.
        video_threshold:     Fake probability threshold for video-level verdict.
        svm_c:               SVC regularisation parameter.
        svm_class_weight:    SVC class weight ("balanced" or None).
    """

    num_qubits: int = 4
    feature_dimension: int = 4
    reps: int = 1
    entanglement: str = "linear"
    shots: int = 1024
    random_seed: int = 42
    backend: str = "simulator"
    max_train_samples: int = 500
    max_val_samples: int = 200
    max_test_samples: int = 200
    feature_range_low: float = -math.pi
    feature_range_high: float = math.pi
    video_threshold: float = 0.5
    svm_c: float = 1.0
    svm_class_weight: Optional[str] = "balanced"

    def __post_init__(self):
        if self.num_qubits != self.feature_dimension:
            raise ValueError(
                f"num_qubits ({self.num_qubits}) must equal "
                f"feature_dimension ({self.feature_dimension})"
            )
        if self.feature_dimension not in (4, 6, 8):
            raise ValueError(
                f"feature_dimension must be 4, 6, or 8; got {self.feature_dimension}"
            )
        if self.backend not in ("simulator", "ibm_hardware"):
            raise ValueError(
                f"backend must be 'simulator' or 'ibm_hardware'; got {self.backend}"
            )
        if self.entanglement not in ("linear", "full", "circular", "sca"):
            raise ValueError(
                f"entanglement must be one of 'linear', 'full', 'circular'; "
                f"got {self.entanglement}"
            )

    @property
    def feature_range(self) -> Tuple[float, float]:
        return (self.feature_range_low, self.feature_range_high)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["feature_range"] = list(self.feature_range)
        return d

    def save_json(self, path: str) -> None:
        """Persist configuration to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load_json(cls, path: str) -> "QuantumConfig":
        """Load configuration from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        # Remove derived/computed fields not in __init__
        data.pop("feature_range", None)
        known = {k for k in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def __str__(self) -> str:
        lines = [
            "",
            "=" * 55,
            "  TrustLens-Q — Quantum Configuration",
            "=" * 55,
        ]
        for k, v in self.to_dict().items():
            lines.append(f"  {k:25s}: {v}")
        lines.append("=" * 55)
        return "\n".join(lines)


if __name__ == "__main__":
    cfg = QuantumConfig()
    print(cfg)

    # JSON round-trip test
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    cfg.save_json(tmp)
    reloaded = QuantumConfig.load_json(tmp)
    os.unlink(tmp)
    assert cfg == reloaded, "JSON round-trip failed"
    print("\nJSON round-trip: PASSED ✅")
