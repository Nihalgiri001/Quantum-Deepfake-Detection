"""
Central Configuration Module
=============================
All configurable parameters for the Classical Deepfake Detection pipeline.

This module provides a single source of truth for:
- Dataset paths and structure
- Processing parameters (frame extraction, face detection)
- Model hyperparameters (PCA components, SVM parameters)
- Data split ratios
- Device selection (CPU / MPS / CUDA)

Usage:
    from config import Config
    cfg = Config()
    cfg.frame_interval = 5  # Override defaults as needed
"""

import json
import torch
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List


# ──────────────────────────────────────────────────────────────────────────────
# Project root: one level above src/
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    """Central configuration for the classical deepfake detection pipeline.

    Every parameter that controls the pipeline is defined here.
    Modify defaults directly or override via CLI arguments in individual scripts.
    """

    # ── Dataset Paths ─────────────────────────────────────────────────────────
    # Point this to the top-level FaceForensics++ directory containing
    # 'original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'.
    dataset_root: str = str(PROJECT_ROOT.parent / "FaceForensics++_C23")
    output_root: str = str(PROJECT_ROOT)

    # ── Frame Extraction ──────────────────────────────────────────────────────
    frame_interval: int = 10        # Sample every N-th frame
    image_size: int = 224           # CNN input size (ResNet-18 expects 224×224)

    # ── PCA ───────────────────────────────────────────────────────────────────
    pca_components: int = 64        # Number of principal components to retain

    # ── SVM Hyperparameters ───────────────────────────────────────────────────
    svm_c: float = 10.0
    svm_gamma: str = "scale"
    svm_kernel: str = "rbf"

    # ── Data Split Ratios ─────────────────────────────────────────────────────
    random_seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # ── Processing ────────────────────────────────────────────────────────────
    batch_size: int = 32
    num_workers: int = 4
    max_videos: Optional[int] = None   # None = process all videos

    # ── Device ────────────────────────────────────────────────────────────────
    device: str = "auto"            # "auto", "cpu", "mps", or "cuda"

    # ── Label Encoding (DO NOT CHANGE between scripts) ────────────────────────
    REAL_LABEL: int = 0
    FAKE_LABEL: int = 1

    # ── FaceForensics++ Manipulation Types ────────────────────────────────────
    manipulation_types: List[str] = field(default_factory=lambda: [
        "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"
    ])

    # ── FaceForensics++ Structure ─────────────────────────────────────────────
    # Folder name for original (real) videos inside the dataset root
    original_dir_name: str = "original"

    # ──────────────────────────────────────────────────────────────────────────
    # Device resolution
    # ──────────────────────────────────────────────────────────────────────────
    def get_device(self) -> torch.device:
        """Auto-detect the best available compute device.

        Priority: CUDA > MPS (Apple Silicon) > CPU
        """
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.device)

    # ──────────────────────────────────────────────────────────────────────────
    # Derived dataset paths
    # ──────────────────────────────────────────────────────────────────────────
    @property
    def dataset_path(self) -> Path:
        return Path(self.dataset_root)

    @property
    def original_videos_dir(self) -> Path:
        """Directory containing original (real) videos."""
        return self.dataset_path / self.original_dir_name

    def get_manipulation_dir(self, manipulation_type: str) -> Path:
        """Directory containing videos of a specific manipulation type."""
        return self.dataset_path / manipulation_type

    # ──────────────────────────────────────────────────────────────────────────
    # Derived output paths
    # ──────────────────────────────────────────────────────────────────────────
    @property
    def frames_dir(self) -> Path:
        return Path(self.output_root) / "data" / "frames"

    @property
    def metadata_dir(self) -> Path:
        return Path(self.output_root) / "data" / "metadata"

    @property
    def features_dir(self) -> Path:
        return Path(self.output_root) / "features"

    @property
    def models_dir(self) -> Path:
        return Path(self.output_root) / "models"

    @property
    def results_dir(self) -> Path:
        return Path(self.output_root) / "results"

    # ──────────────────────────────────────────────────────────────────────────
    # Serialisation
    # ──────────────────────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        """Convert configuration to a JSON-serialisable dictionary."""
        d = asdict(self)
        d["device_resolved"] = str(self.get_device())
        return d

    def save(self, filepath: str) -> None:
        """Persist configuration to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, filepath: str) -> "Config":
        """Load configuration from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        # Remove keys that are not constructor parameters
        data.pop("device_resolved", None)
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})

    def __str__(self) -> str:
        lines = [
            "",
            "=" * 60,
            "  Classical Deepfake Detection — Configuration",
            "=" * 60,
        ]
        for k, v in asdict(self).items():
            lines.append(f"  {k:25s}: {v}")
        lines.append(f"  {'device_resolved':25s}: {self.get_device()}")
        lines.append("=" * 60)
        return "\n".join(lines)


def get_config(**overrides) -> Config:
    """Create a Config instance with optional keyword overrides.

    Example:
        cfg = get_config(frame_interval=5, max_videos=20)
    """
    return Config(**overrides)
