"""
ResNet-18 Feature Extractor
=============================
Uses a pretrained ResNet-18 as a fixed feature extractor.

Architecture:
    Face crop  →  ResNet-18 (minus final FC)  →  512-dim feature vector

Key design decisions:
- The final classification layer is removed; we extract from the average
  pooling layer, yielding a 512-dimensional feature vector per image.
- All ResNet-18 parameters are frozen (no fine-tuning).
- The same extractor will be reused by the future quantum pipeline,
  ensuring a fair comparison between RBF-SVM and QSVM.

This module is INDEPENDENT from the classifier (SVM / future QSVM).
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from torchvision.models import ResNet18_Weights
from PIL import Image
from pathlib import Path
from typing import List, Optional

from preprocessing import get_preprocessing_transform
from utils import setup_logger

logger = setup_logger(__name__)


class ResNetFeatureExtractor:
    """Extract deep features using a pretrained ResNet-18.

    Attributes:
        model: The modified ResNet-18 (final FC replaced with Identity).
        device: Compute device (CPU / MPS / CUDA).
        transform: Preprocessing pipeline matching ImageNet normalisation.
        feature_dim: Dimensionality of the output feature vector (512).
    """

    FEATURE_DIM = 512  # ResNet-18 avgpool output

    def __init__(
        self,
        device: Optional[torch.device] = None,
        image_size: int = 224,
    ):
        """Initialise the feature extractor.

        Args:
            device: Compute device. Defaults to CPU.
            image_size: Input image resolution (must match training preprocessing).
        """
        self.device = device or torch.device("cpu")
        self.image_size = image_size

        # Load pretrained ResNet-18 with ImageNet weights
        logger.info("Loading pretrained ResNet-18 (ImageNet weights)...")
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Remove the final fully-connected classification layer
        # Replace with Identity so forward() returns the 512-d feature vector
        self.model.fc = nn.Identity()

        # Freeze all parameters — we are NOT training the CNN
        for param in self.model.parameters():
            param.requires_grad = False

        self.model = self.model.to(self.device)
        self.model.eval()

        # Preprocessing transform
        self.transform = get_preprocessing_transform(image_size)

        logger.info(
            f"ResNet-18 feature extractor ready on {self.device} "
            f"(output dim = {self.FEATURE_DIM})"
        )

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess a single image from disk.

        Args:
            image_path: Path to the image file.

        Returns:
            Tensor of shape (3, H, W).
        """
        img = Image.open(image_path).convert("RGB")
        return self.transform(img)

    def extract_single(self, image_path: str) -> np.ndarray:
        """Extract features from a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            1-D numpy array of shape (512,).
        """
        tensor = self.preprocess_image(image_path).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(tensor)
        return features.cpu().numpy().flatten()

    def extract_batch(self, image_paths: List[str]) -> np.ndarray:
        """Extract features from a batch of images.

        Args:
            image_paths: List of image file paths.

        Returns:
            2-D numpy array of shape (N, 512).
        """
        tensors = []
        valid_indices = []
        for i, path in enumerate(image_paths):
            try:
                t = self.preprocess_image(path)
                tensors.append(t)
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"Failed to load image {path}: {e}")

        if not tensors:
            return np.empty((0, self.FEATURE_DIM))

        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            features = self.model(batch)
        return features.cpu().numpy()
