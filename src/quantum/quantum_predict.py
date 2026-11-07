"""
Quantum Video Prediction
=========================
End-to-end deepfake detection on a single video file using the trained
Quantum Kernel SVM pipeline.

Pipeline:
    Video → frame extraction → face detection → ResNet-18 (512d)
         → QuantumFeatureReducer (4/6/8d) → cross-kernel vs training set
         → QuantumSVM.predict_proba → frame aggregation → verdict

The output schema intentionally mirrors ``src/predict.py`` so that the
quantum and classical predictors are drop-in compatible for downstream
callers, with three extra fields added:
    backend              — "simulator" or "ibm_hardware"
    num_qubits           — circuit width used
    kernel_compute_time_sec — time to compute the cross-kernel

Design decisions:
- Face detection and feature extraction reuse ``FaceDetector`` and
  ``ResNetFeatureExtractor`` from the existing classical pipeline — no
  duplication.
- The cross-kernel is computed against the full training set stored in
  ``models_dir/train_features_reduced.npy``.  At inference this is always
  a ``(n_frames, n_train)`` matrix.
- ``--backend ibm_hardware`` requires ``IBMCredentialsError`` to be raised
  cleanly when ``IBM_QUANTUM_TOKEN`` is absent.  Since ``ibm_hardware.py``
  is implemented in Task 11, we import its exception lazily and fall back
  to a local stub so this module works now.
- Frames where no face is detected are silently skipped.  If no frames
  yield a detected face the function returns UNKNOWN with 0.5 probability.

CLI:
    python -m src.quantum.quantum_predict --input sample.mp4
    python -m src.quantum.quantum_predict --input sample.mp4 --backend simulator
    python -m src.quantum.quantum_predict --input sample.mp4 --backend ibm_hardware
"""

import sys
import time
import logging
import argparse
import warnings
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Optional, List, Dict, Any
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantum.quantum_config import QuantumConfig
from quantum.quantum_svm import QuantumSVM
from quantum.feature_reducer import QuantumFeatureReducer
from quantum.quantum_kernel import QuantumKernelComputer
from quantum.simulator_backend import SimulatorBackend

try:
    from face_detection import FaceDetector
    from feature_extractor import ResNetFeatureExtractor
    from utils import setup_logger
    logger = setup_logger("quantum.predict")
except ImportError:
    logger = logging.getLogger("quantum.predict")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO,
                            format="[%(asctime)s] %(levelname)-8s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")


# ── IBM credentials stub (replaced by ibm_hardware.py when Task 11 ships) ─────

class IBMCredentialsError(Exception):
    """IBM Quantum credentials are missing or invalid.

    Set the IBM_QUANTUM_TOKEN environment variable before using
    the ibm_hardware backend:

        export IBM_QUANTUM_TOKEN="your-token-here"

    Then run:
        python -m src.quantum.quantum_predict --input video.mp4 --backend ibm_hardware
    """


def _get_ibm_credentials_error():
    """Return IBMCredentialsError from ibm_hardware.py if available."""
    try:
        from quantum.ibm_hardware import IBMCredentialsError as _E
        return _E
    except ImportError:
        return IBMCredentialsError


# ── Core prediction function ──────────────────────────────────────────────────

def predict_video_quantum(
    video_path: str,
    models_dir: Optional[str] = None,
    frame_interval: int = 10,
    threshold: float = 0.5,
    backend_type: str = "simulator",
    device_str: str = "auto",
) -> Dict[str, Any]:
    """Run quantum deepfake detection on a single video file.

    Args:
        video_path:     Path to the input video (.mp4, .avi, etc.).
        models_dir:     Directory containing ``quantum_svm.pkl`` and friends.
                        Defaults to ``models/quantum/`` relative to project root.
        frame_interval: Extract every N-th frame (default: 10).
        threshold:      Fake-probability threshold for video verdict (default: 0.5).
        backend_type:   "simulator" (default) or "ibm_hardware".
        device_str:     PyTorch device for ResNet-18: "auto", "cpu", "mps", "cuda".

    Returns:
        Dictionary with keys:
            video_name              str
            video_path              str
            verdict                 "REAL" | "FAKE" | "UNKNOWN"
            confidence              float  in [0, 1]
            mean_fake_prob          float  in [0, 1]
            total_frames_sampled    int
            faces_detected          int
            frame_scores            list[dict]
            backend                 str
            num_qubits              int
            kernel_compute_time_sec float
            model_info              dict

    Raises:
        FileNotFoundError:    If video file or model artefacts are missing.
        IBMCredentialsError:  If backend_type=="ibm_hardware" and no token set.
        ValueError:           If backend_type is unrecognised.
    """
    # ── Validate backend ──────────────────────────────────────────────────────
    if backend_type == "ibm_hardware":
        import os
        token = os.environ.get("IBM_QUANTUM_TOKEN", "")
        if not token:
            CredError = _get_ibm_credentials_error()
            raise CredError(
                "IBM_QUANTUM_TOKEN environment variable is not set.\n"
                "Set it with:  export IBM_QUANTUM_TOKEN='your-token-here'\n"
                "Or use --backend simulator for local execution."
            )
        # Delegate to ibm_hardware module when it exists
        try:
            from quantum.ibm_hardware import IBMHardwareBackend
            raise NotImplementedError(
                "IBM hardware inference is not yet implemented in this version. "
                "Use --backend simulator for local prediction."
            )
        except ImportError:
            raise NotImplementedError(
                "ibm_hardware.py is not yet implemented (Task 11). "
                "Use --backend simulator."
            )

    if backend_type != "simulator":
        raise ValueError(
            f"Unknown backend '{backend_type}'. "
            "Supported values: 'simulator', 'ibm_hardware'."
        )

    # ── Resolve paths ─────────────────────────────────────────────────────────
    _root = Path(__file__).resolve().parent.parent.parent
    if models_dir is None:
        models_dir = str(_root / "models" / "quantum")
    md = Path(models_dir)

    video_file = Path(video_path).resolve()
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_file}")

    # ── Load quantum artefacts ────────────────────────────────────────────────
    logger.info(f"Loading QuantumSVM from {md}")
    qsvm    = QuantumSVM.load(str(md))
    reducer = QuantumFeatureReducer.load(str(md))
    config  = qsvm.config

    X_train_q = qsvm._train_features_reduced
    if X_train_q is None:
        raise RuntimeError(
            "train_features_reduced.npy not found in model artefacts. "
            "Re-run '--train' to regenerate."
        )

    # ── Set up compute device ─────────────────────────────────────────────────
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    # ── Initialise face detector and feature extractor ────────────────────────
    detector  = FaceDetector()
    extractor = ResNetFeatureExtractor(device=device, image_size=224)

    # ── Open video and extract frames ─────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_file}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(
        f"Video: {video_file.name} | frames={total_video_frames} | "
        f"fps={fps:.1f} | interval=every {frame_interval}"
    )

    face_tensors: List[torch.Tensor] = []
    frame_numbers: List[int] = []
    frame_idx = 0
    sampled_count = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            sampled_count += 1
            crop = detector.crop_face(frame_bgr, margin=0.2)
            if crop is not None:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img  = Image.fromarray(crop_rgb)
                tensor   = extractor.transform(pil_img)
                face_tensors.append(tensor)
                frame_numbers.append(frame_idx)
        frame_idx += 1

    cap.release()

    # ── Handle no-face case ───────────────────────────────────────────────────
    if not face_tensors:
        logger.warning("No faces detected in the video — returning UNKNOWN.")
        return {
            "video_name":            video_file.name,
            "video_path":            str(video_file),
            "verdict":               "UNKNOWN",
            "confidence":            0.0,
            "mean_fake_prob":        0.5,
            "total_frames_sampled":  sampled_count,
            "faces_detected":        0,
            "frame_scores":          [],
            "backend":               backend_type,
            "num_qubits":            config.num_qubits,
            "kernel_compute_time_sec": 0.0,
            "model_info": {
                "models_dir":      str(md),
                "feature_dimension": config.feature_dimension,
                "reps":            config.reps,
                "entanglement":    config.entanglement,
            },
        }

    # ── Extract ResNet-18 features ────────────────────────────────────────────
    batch = torch.stack(face_tensors).to(device)
    with torch.no_grad():
        raw_features = extractor.model(batch).cpu().numpy()   # (n_faces, 512)

    logger.info(f"Extracted features for {len(face_tensors)} face crops")

    # ── Apply QuantumFeatureReducer ───────────────────────────────────────────
    X_new_q = reducer.transform(raw_features)   # (n_faces, feature_dim)

    # ── Compute cross-kernel ──────────────────────────────────────────────────
    backend  = SimulatorBackend(shots=config.shots, seed=config.random_seed)
    computer = QuantumKernelComputer(
        config=config,
        backend=backend,
        cache_dir=str(_root / "features" / "quantum"),
    )

    t0 = time.time()
    K_cross = computer.compute_cross_kernel(
        X_new_q, X_train_q,
        name="K_predict",
        use_cache=False,          # never cache ad-hoc prediction kernels
    )
    kernel_time = time.time() - t0
    logger.info(f"Cross-kernel ({X_new_q.shape[0]}×{X_train_q.shape[0]}) "
                f"computed in {kernel_time:.2f}s")

    # ── QSVM inference ────────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        proba    = qsvm.predict_proba(K_cross)   # (n_faces, 2): [P(REAL), P(FAKE)]
    fake_probs = proba[:, 1]

    # ── Aggregate to video level ──────────────────────────────────────────────
    mean_fake_prob = float(np.mean(fake_probs))
    is_fake        = mean_fake_prob >= threshold
    verdict        = "FAKE" if is_fake else "REAL"
    confidence     = mean_fake_prob if is_fake else (1.0 - mean_fake_prob)

    frame_scores = [
        {
            "frame":           fn,
            "fake_probability": float(fp),
            "prediction":       "FAKE" if fp >= 0.5 else "REAL",
        }
        for fn, fp in zip(frame_numbers, fake_probs)
    ]

    return {
        "video_name":            video_file.name,
        "video_path":            str(video_file),
        "verdict":               verdict,
        "confidence":            float(confidence),
        "mean_fake_prob":        float(mean_fake_prob),
        "total_frames_sampled":  sampled_count,
        "faces_detected":        len(face_tensors),
        "frame_scores":          frame_scores,
        "backend":               backend_type,
        "num_qubits":            int(config.num_qubits),
        "kernel_compute_time_sec": float(kernel_time),
        "model_info": {
            "models_dir":        str(md),
            "feature_dimension": int(config.feature_dimension),
            "reps":              int(config.reps),
            "entanglement":      config.entanglement,
        },
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run Quantum Kernel SVM deepfake detection on a video file."
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to the input video file."
    )
    parser.add_argument(
        "--backend", type=str, default="simulator",
        choices=["simulator", "ibm_hardware"],
        help="Execution backend (default: simulator)."
    )
    parser.add_argument(
        "--frame-interval", type=int, default=10,
        help="Extract every N-th frame (default: 10)."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Fake-probability threshold for video verdict (default: 0.5)."
    )
    parser.add_argument(
        "--models-dir", type=str, default=None,
        help="Override models directory."
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="PyTorch device: auto, cpu, mps, cuda (default: auto)."
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  TrustLens-Q — Quantum Video Inference")
    print("=" * 60)
    print(f"  Input   : {args.input}")
    print(f"  Backend : {args.backend}")
    print(f"  Interval: every {args.frame_interval} frames")
    print(f"  Device  : {args.device}")
    print()

    try:
        results = predict_video_quantum(
            video_path=args.input,
            models_dir=args.models_dir,
            frame_interval=args.frame_interval,
            threshold=args.threshold,
            backend_type=args.backend,
            device_str=args.device,
        )
    except (IBMCredentialsError, Exception) as exc:
        # Catch IBMCredentialsError from either the stub or the real module
        CredError = _get_ibm_credentials_error()
        if isinstance(exc, CredError):
            print(f"  ❌  IBM Credentials Error: {exc}")
            print()
            sys.exit(1)
        raise

    # ── Print results ──────────────────────────────────────────────────────────
    print("─" * 60)
    print("  INFERENCE RESULTS")
    print("─" * 60)
    print(f"  Video File        : {results['video_name']}")
    print(f"  Frames Sampled    : {results['total_frames_sampled']}")
    print(f"  Faces Detected    : {results['faces_detected']}")
    print(f"  Mean Fake Prob    : {results['mean_fake_prob'] * 100:.2f}%")
    print(f"  Backend           : {results['backend']}")
    print(f"  Qubits            : {results['num_qubits']}")
    print(f"  Kernel Time       : {results['kernel_compute_time_sec']:.2f}s")
    print("─" * 60)

    badge = "🚨 FAKE" if results["verdict"] == "FAKE" else (
        "✅ REAL" if results["verdict"] == "REAL" else "⚠️  UNKNOWN"
    )
    print(f"  VERDICT           : {badge}")
    print(f"  CONFIDENCE        : {results['confidence'] * 100:.2f}%")
    print("─" * 60)

    if results["frame_scores"]:
        print()
        print("  Sample Frame Breakdown (first 10):")
        print("  " + "─" * 45)
        print("  Frame #   | Fake Probability | Prediction")
        print("  " + "─" * 45)
        for s in results["frame_scores"][:10]:
            print(
                f"  Frame {s['frame']:<4}  | "
                f"{s['fake_probability'] * 100:6.2f}%           | "
                f"{s['prediction']}"
            )
        if len(results["frame_scores"]) > 10:
            print(f"  ... ({len(results['frame_scores']) - 10} more frames)")
        print("  " + "─" * 45)
    print()


if __name__ == "__main__":
    main()
