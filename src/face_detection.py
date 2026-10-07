"""
Face Detection Module
======================
Face detection using OpenCV's FaceDetectorYN (DNN-based, OpenCV 5.0+).

Design decisions:
- Uses OpenCV's built-in FaceDetectorYN with the YuNet ONNX model.
- The model file is automatically downloaded from the official OpenCV
  Zoo repository on first use and cached locally.
- When multiple faces are detected, selects the LARGEST by bounding-box area.
- Returns None when no face is found so callers can log and skip gracefully.
- Compatible with OpenCV 5.0+ (which removed the legacy CascadeClassifier).
"""

import cv2
import numpy as np
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

from utils import setup_logger

logger = setup_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# YuNet face detection model (official OpenCV model zoo)
# ──────────────────────────────────────────────────────────────────────────────
_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODEL_CACHE_DIR = _PROJECT_ROOT / "models" / "face_detector"
_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"


def _resolve_model_path() -> str:
    """Find or download the YuNet face detection ONNX model.

    Returns:
        Absolute path to the ONNX model file.
    """
    cached_path = _MODEL_CACHE_DIR / _MODEL_FILENAME
    if cached_path.exists():
        return str(cached_path)

    # Download from OpenCV Zoo
    logger.info("Downloading YuNet face detection model from OpenCV Zoo...")
    _MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(_MODEL_URL, str(cached_path))
        logger.info(f"YuNet model saved to: {cached_path}")
        return str(cached_path)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not download the YuNet face detection model.\n"
            f"  URL: {_MODEL_URL}\n"
            f"  Error: {e}\n\n"
            f"Manual fix: download the ONNX file from the URL above and "
            f"place it at:\n  {cached_path}"
        ) from e


class FaceDetector:
    """Detect and crop faces from images using OpenCV FaceDetectorYN.

    This uses the YuNet DNN model, which is more accurate than the
    legacy Haar Cascade and is the standard face detector in OpenCV 5.0+.

    Attributes:
        detector: The FaceDetectorYN instance.
        score_threshold: Minimum confidence for a detection.
        nms_threshold: Non-maximum suppression threshold.
        input_size: Default detection input size (width, height).
    """

    def __init__(
        self,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        input_size: Tuple[int, int] = (320, 320),
    ):
        model_path = _resolve_model_path()
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

        self.detector = cv2.FaceDetectorYN_create(
            model=model_path,
            config="",
            input_size=self.input_size,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
        )
        logger.info(f"Face detector loaded (YuNet DNN, OpenCV {cv2.__version__})")

    def detect_faces(self, image: np.ndarray) -> list:
        """Detect all faces in an image.

        Args:
            image: BGR image (as returned by cv2.imread or cv2.VideoCapture).

        Returns:
            List of (x, y, w, h, confidence) tuples.
            Empty list if no faces are detected.
        """
        h, w = image.shape[:2]
        # FaceDetectorYN requires setting the input size to match the image
        self.detector.setInputSize((w, h))

        _, detections = self.detector.detect(image)

        if detections is None:
            return []

        faces = []
        for det in detections:
            x, y, fw, fh = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            confidence = float(det[-1])
            faces.append((x, y, fw, fh, confidence))

        return faces

    def get_largest_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Return the bounding box of the largest detected face.

        Args:
            image: BGR image.

        Returns:
            (x, y, w, h) of the largest face, or None if no face is detected.
        """
        faces = self.detect_faces(image)
        if not faces:
            return None
        # Select face with the largest area (w * h), ignoring confidence
        best = max(faces, key=lambda f: f[2] * f[3])
        return (best[0], best[1], best[2], best[3])

    def crop_face(
        self,
        image: np.ndarray,
        margin: float = 0.2,
    ) -> Optional[np.ndarray]:
        """Detect the largest face and return the cropped region.

        A configurable margin is added around the bounding box to include
        some context (forehead, chin, etc.), which helps the CNN.

        Args:
            image: BGR image.
            margin: Fractional margin to add around the bounding box
                    (0.2 = 20% on each side).

        Returns:
            Cropped face image (BGR), or None if no face is found.
        """
        bbox = self.get_largest_face(image)
        if bbox is None:
            return None

        x, y, w, h = bbox
        img_h, img_w = image.shape[:2]

        # Add margin
        margin_w = int(w * margin)
        margin_h = int(h * margin)

        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(img_w, x + w + margin_w)
        y2 = min(img_h, y + h + margin_h)

        crop = image[y1:y2, x1:x2]

        # Ensure non-empty crop
        if crop.size == 0:
            return None

        return crop
