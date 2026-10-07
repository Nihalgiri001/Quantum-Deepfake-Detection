"""
Image Preprocessing Module
============================
Prepares face crops for ResNet-18 feature extraction.

Pipeline per image:
1. Convert BGR → RGB
2. Resize to 224 × 224
3. Normalise with ImageNet mean/std
4. Convert to PyTorch tensor

No aggressive augmentation is applied to preserve deepfake artefacts.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# ImageNet normalisation statistics (used by all torchvision pretrained models)
# ──────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_preprocessing_transform(image_size: int = 224) -> transforms.Compose:
    """Build the standard preprocessing pipeline for inference / feature extraction.

    This matches the preprocessing expected by torchvision ResNet-18
    with pretrained ImageNet weights.

    Args:
        image_size: Target spatial resolution.

    Returns:
        A torchvision Compose transform.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_training_transform(image_size: int = 224) -> transforms.Compose:
    """Lightweight augmentation for training data only.

    Kept minimal to avoid destroying deepfake artefacts:
    - Small random horizontal flip
    - Slight colour jitter

    Args:
        image_size: Target spatial resolution.

    Returns:
        A torchvision Compose transform.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_face_crop(
    image_bgr: np.ndarray,
    image_size: int = 224,
) -> torch.Tensor:
    """Convert a single BGR face crop to a normalised tensor.

    Args:
        image_bgr: Face crop in BGR format (from OpenCV).
        image_size: Target spatial resolution.

    Returns:
        Tensor of shape (3, image_size, image_size).
    """
    # BGR → RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    transform = get_preprocessing_transform(image_size)
    return transform(pil_image)


def save_face_crop(
    image_bgr: np.ndarray,
    save_path: str,
    image_size: int = 224,
) -> bool:
    """Resize and save a face crop to disk as a JPEG.

    The image is saved in its original colour space (BGR→RGB conversion
    happens later during feature extraction). This avoids double
    conversion and keeps stored images human-viewable.

    Args:
        image_bgr: Face crop from the detector.
        save_path: Destination file path.
        image_size: Target spatial resolution.

    Returns:
        True if saved successfully, False otherwise.
    """
    try:
        resized = cv2.resize(image_bgr, (image_size, image_size))
        cv2.imwrite(save_path, resized)
        return True
    except Exception:
        return False
