"""
Inference Script — Classical Deepfake Detector
===============================================
Extracts face crops from a user-provided video, runs ResNet-18 feature extraction,
applies StandardScaler + PCA, and predicts whether the video is REAL or FAKE using
the trained RBF-SVM model.

Usage:
    python src/predict.py --video "/path/to/video.mp4"
    python src/predict.py --video "../FaceForensics++_C23/DeepFakeDetection/02_15__walking_and_outside_surprised__MZWH8ATN.mp4"
    python src/predict.py --video "path/to/video.mp4" --frame-interval 5

Outputs:
    Terminal summary displaying:
    - Number of frames processed & faces detected
    - Frame-level probability breakdown
    - Final Video Classification: REAL or FAKE
    - Confidence percentage
"""

import sys
import argparse
import cv2
import torch
import joblib
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from face_detection import FaceDetector
from feature_extractor import ResNetFeatureExtractor
from utils import setup_logger, ensure_dir

logger = setup_logger("predict")


def predict_video(
    video_path: str,
    models_dir: Path,
    frame_interval: int = 10,
    threshold: float = 0.5,
    device_str: str = "auto",
) -> dict:
    """Run classical deepfake detection on a single video file.

    Args:
        video_path: Path to the input video file.
        models_dir: Directory containing scaler.pkl, pca.pkl, and svm.pkl.
        frame_interval: Extract every N-th frame from the video.
        threshold: Classification probability threshold (default: 0.5).
        device_str: Device to run PyTorch model ("auto", "cpu", "mps", "cuda").

    Returns:
        Dictionary containing prediction results and per-frame confidence scores.
    """
    video_file = Path(video_path).resolve()
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_file}")

    # 1. Load trained model artefacts
    scaler_path = models_dir / "scaler.pkl"
    pca_path = models_dir / "pca.pkl"
    svm_path = models_dir / "svm.pkl"

    for p in [scaler_path, pca_path, svm_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Model file not found at {p}.\n"
                "Please run 'python src/train.py' first to train and save the model."
            )

    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    svm = joblib.load(svm_path)

    # 2. Initialise Face Detector & Feature Extractor
    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available())
        else "mps" if (device_str == "auto" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        else "cpu"
    )

    detector = FaceDetector()
    extractor = ResNetFeatureExtractor(device=device, image_size=224)

    # 3. Read video and extract face crops
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_file}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_idx = 0
    processed_count = 0
    face_tensors = []
    frame_numbers = []

    logger.info(f"Opening video: {video_file.name}")
    logger.info(f"Total video frames: {total_video_frames} | FPS: {fps:.2f} | Frame Interval: every {frame_interval}th frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            processed_count += 1
            # Detect largest face crop
            crop_bgr = detector.crop_face(frame, margin=0.2)
            if crop_bgr is not None:
                # Convert BGR (OpenCV) to RGB (PIL)
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(crop_rgb)
                tensor = extractor.transform(pil_img)
                face_tensors.append(tensor)
                frame_numbers.append(frame_idx)

        frame_idx += 1

    cap.release()

    if not face_tensors:
        logger.warning("No faces detected in the extracted video frames.")
        return {
            "video_name": video_file.name,
            "verdict": "UNKNOWN",
            "confidence": 0.0,
            "mean_fake_prob": 0.5,
            "total_frames_sampled": processed_count,
            "faces_detected": 0,
            "frame_scores": [],
        }

    # 4. Extract ResNet-18 features (512-dim)
    batch = torch.stack(face_tensors).to(device)
    with torch.no_grad():
        features = extractor.model(batch).cpu().numpy()

    # 5. Apply StandardScaler + PCA
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)

    # 6. Predict probabilities with SVM
    # predict_proba returns [P(REAL), P(FAKE)]
    probs = svm.predict_proba(features_pca)
    fake_probs = probs[:, 1]  # P(FAKE)

    mean_fake_prob = float(np.mean(fake_probs))
    is_fake = mean_fake_prob >= threshold
    verdict = "FAKE" if is_fake else "REAL"
    confidence = mean_fake_prob if is_fake else (1.0 - mean_fake_prob)

    frame_scores = [
        {"frame": fn, "fake_probability": float(p), "prediction": "FAKE" if p >= 0.5 else "REAL"}
        for fn, p in zip(frame_numbers, fake_probs)
    ]

    return {
        "video_name": video_file.name,
        "video_path": str(video_file),
        "verdict": verdict,
        "confidence": confidence,
        "mean_fake_prob": mean_fake_prob,
        "total_frames_sampled": processed_count,
        "faces_detected": len(face_tensors),
        "frame_scores": frame_scores,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Classical Deepfake Detection on a single video file."
    )
    parser.add_argument(
        "--video", "-v", type=str, required=True,
        help="Path to the input video file (.mp4, .avi, .mov, etc.)."
    )
    parser.add_argument(
        "--frame-interval", type=int, default=10,
        help="Extract every N-th frame (default: 10)."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Probability threshold for FAKE verdict (default: 0.5)."
    )
    args = parser.parse_args()

    cfg = Config()

    print()
    print("=" * 60)
    print("  Classical Deepfake Detector — Video Inference")
    print("=" * 60)

    results = predict_video(
        video_path=args.video,
        models_dir=cfg.models_dir,
        frame_interval=args.frame_interval,
        threshold=args.threshold,
        device_str=cfg.device,
    )

    print()
    print("─" * 60)
    print("  INFERENCE RESULTS")
    print("─" * 60)
    print(f"  Video File:             {results['video_name']}")
    print(f"  Frames Processed:       {results['total_frames_sampled']}")
    print(f"  Faces Detected:         {results['faces_detected']}")
    print(f"  Mean Fake Probability:  {results['mean_fake_prob'] * 100:.2f}%")
    print("─" * 60)

    verdict_badge = "🚨 FAKE" if results['verdict'] == "FAKE" else "✅ REAL"
    print(f"  VERDICT:                {verdict_badge}")
    print(f"  CONFIDENCE:             {results['confidence'] * 100:.2f}%")
    print("─" * 60)

    if results["frame_scores"]:
        print()
        print("  Sample Frame Breakdown:")
        print("  " + "─" * 45)
        print("  Frame #   | Fake Probability | Frame Verdict")
        print("  " + "─" * 45)
        for s in results["frame_scores"][:10]:
            indicator = "FAKE" if s["fake_probability"] >= 0.5 else "REAL"
            print(f"  Frame {s['frame']:<4}  | {s['fake_probability'] * 100:6.2f}%          | {indicator}")
        if len(results["frame_scores"]) > 10:
            print(f"  ... ({len(results['frame_scores']) - 10} more frames evaluated)")
        print("  " + "─" * 45)

    print()


if __name__ == "__main__":
    main()
