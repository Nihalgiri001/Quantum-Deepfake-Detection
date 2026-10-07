# Deepfake Detection Using Quantum Computing for Identity Fraud Prevention
## Module 1: Classical Deepfake Detection Baseline

---

## 1. Project Overview

### 1.1 Motivation

Deepfake technology uses deep learning and generative models to manipulate or synthesize facial images and videos. Such manipulated media can be weaponised for:

- **Identity impersonation** and fraudulent identity verification
- **Social engineering** attacks
- **Misinformation** campaigns
- **Fake profiles** and digital identity fraud

This project investigates whether **quantum machine learning** can improve upon classical deepfake detection. To make a fair scientific comparison, we first build a **classical baseline** using well-established techniques.

### 1.2 Overall Architecture (Full Project)

```
Input Image / Video
        │
        ▼
Media Preprocessing → Face Detection → ResNet-18 Features
        │
        ▼
StandardScaler → PCA → Dimensionality-Reduced Features
        │
        ├──────────────────────┐
        ▼                      ▼
   RBF-SVM (classical)    QSVM (quantum)
        │                      │
        ▼                      ▼
   REAL / FAKE             REAL / FAKE
        │                      │
        └──────────┬───────────┘
                   ▼
           Comparative Analysis
```

### 1.3 Current Scope — This Module

**This repository implements ONLY the classical baseline:**

```
FaceForensics++ Videos
        │
        ▼
Video-Level Split (70/15/15)
        │
        ▼
Frame Extraction (every 10th frame)
        │
        ▼
Face Detection (Haar Cascade)
        │
        ▼
Face Crop (224×224)
        │
        ▼
ResNet-18 Feature Extraction (512-dim)
        │
        ▼
StandardScaler → PCA (64 components)
        │
        ▼
RBF-SVM Classifier
        │
        ▼
REAL / FAKE
```

### 1.4 Why Classical First?

A classical baseline is essential for:

1. Establishing reproducible benchmark results.
2. Validating the complete data pipeline.
3. Providing a fair comparison target for the future QSVM.
4. Ensuring both systems use identical features.

### 1.5 Future QSVM Extension

The quantum module will **reuse the exact same pipeline** up to PCA:

```
Same ResNet-18 features → Same StandardScaler → Same PCA
        │
        ▼
Quantum Feature Map → Quantum Kernel → QSVM → REAL / FAKE
```

Only the classification mechanism changes. The saved PCA features (`features/`) can be directly loaded by the quantum module.

---

## 2. Dataset — FaceForensics++

### 2.1 About the Dataset

[FaceForensics++](https://github.com/ondyari/FaceForensics) is a large-scale benchmark for face manipulation detection. It contains 1,000 original YouTube videos and their corresponding manipulated versions using four methods.

### 2.2 Manipulation Types

| Category | Label | Description |
|---|---|---|
| Original (YouTube) | `REAL (0)` | Unmanipulated source videos |
| Deepfakes | `FAKE (1)` | Face swap using autoencoders |
| Face2Face | `FAKE (1)` | Facial reenactment |
| FaceSwap | `FAKE (1)` | Computer graphics face swap |
| NeuralTextures | `FAKE (1)` | Neural rendering manipulation |

### 2.3 Compression Level

This implementation uses the **C23** (light compression) version, which is the standard benchmark quality level.

### 2.4 How to Obtain the Dataset

> **IMPORTANT**: FaceForensics++ requires an access request.

1. Visit the official repository: https://github.com/ondyari/FaceForensics
2. Follow their instructions to request dataset access.
3. Download the C23 compressed videos.
4. **Do NOT commit the dataset to Git.**

### 2.5 Expected Dataset Structure

Place the dataset **alongside** the project directory:

```
Major Project/
├── FaceForensics++_C23/          ← Your dataset goes here
│   ├── original/                 ← 1,000 real videos (000.mp4 ... 999.mp4)
│   ├── Deepfakes/                ← 1,000 manipulated videos (000_003.mp4 ...)
│   ├── Face2Face/                ← 1,000 manipulated videos
│   ├── FaceSwap/                 ← 1,000 manipulated videos
│   └── NeuralTextures/           ← 1,000 manipulated videos
│
└── deepfake_classical/           ← This project
    ├── src/
    ├── requirements.txt
    └── ...
```

If your dataset is at a different location, update `dataset_root` in `src/config.py`.

### 2.6 Label Encoding

Consistent across ALL scripts:

```python
REAL = 0
FAKE = 1
```

---

## 3. Project Structure

```
deepfake_classical/
│
├── src/                          ← Source code
│   ├── config.py                 ← Central configuration (all parameters)
│   ├── utils.py                  ← Logging, seeding, directory helpers
│   ├── dataset.py                ← Video discovery and cataloguing
│   ├── verify_dataset.py         ← Dataset verification script
│   ├── split_dataset.py          ← Video-level train/val/test splitting
│   ├── extract_frames.py         ← Frame extraction + face detection
│   ├── face_detection.py         ← Haar Cascade face detector
│   ├── preprocessing.py          ← ImageNet normalisation pipeline
│   ├── feature_extractor.py      ← ResNet-18 feature extractor class
│   ├── extract_features.py       ← Batch CNN feature extraction
│   ├── train.py                  ← StandardScaler + PCA + RBF-SVM training
│   └── evaluate.py               ← Frame-level + video-level evaluation
│
├── data/                         ← Generated data (git-ignored)
│   ├── frames/                   ← Extracted face crops
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── metadata/                 ← Split CSVs and frame manifest
│       ├── train_videos.csv
│       ├── val_videos.csv
│       ├── test_videos.csv
│       └── frame_manifest.csv
│
├── features/                     ← Saved ResNet-18 features (git-ignored)
│   ├── train_features.npy        ← (N_train, 512)
│   ├── train_labels.npy
│   ├── train_video_ids.npy
│   ├── val_features.npy
│   ├── val_labels.npy
│   ├── val_video_ids.npy
│   ├── test_features.npy
│   ├── test_labels.npy
│   └── test_video_ids.npy
│
├── models/                       ← Saved model artefacts (git-ignored)
│   ├── scaler.pkl
│   ├── pca.pkl
│   ├── svm.pkl
│   └── training_config.json
│
├── results/                      ← Evaluation outputs (git-ignored)
│   ├── confusion_matrix_frame.png
│   ├── roc_curve_frame.png
│   ├── confusion_matrix_video.png
│   ├── roc_curve_video.png
│   ├── metrics.json
│   ├── classification_report.txt
│   └── experiment_config.json
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 4. Environment Setup

### 4.1 Create Virtual Environment

```bash
# Navigate to the project directory
cd deepfake_classical

# Create a Python virtual environment
python3 -m venv .venv

# Activate the environment
source .venv/bin/activate
```

### 4.2 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: On Apple Silicon Macs, PyTorch will automatically use the MPS backend when available. No CUDA is required.

---

## 5. Configuration

All parameters are centralised in [`src/config.py`](src/config.py).

### 5.1 Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `dataset_root` | `../FaceForensics++_C23` | Path to the FaceForensics++ dataset |
| `frame_interval` | `10` | Sample every N-th frame |
| `image_size` | `224` | Face crop resolution |
| `pca_components` | `64` | Number of PCA dimensions |
| `svm_c` | `10.0` | SVM regularisation parameter |
| `svm_gamma` | `"scale"` | SVM kernel coefficient |
| `svm_kernel` | `"rbf"` | SVM kernel type |
| `random_seed` | `42` | Seed for reproducibility |
| `train_ratio` | `0.70` | Training set fraction |
| `val_ratio` | `0.15` | Validation set fraction |
| `test_ratio` | `0.15` | Test set fraction |
| `batch_size` | `32` | CNN feature extraction batch size |
| `max_videos` | `None` | Limit videos per category (for testing) |
| `device` | `"auto"` | `"auto"`, `"cpu"`, `"mps"`, or `"cuda"` |

### 5.2 Customising the Dataset Path

If your dataset is in a different location, either:

1. **Edit `src/config.py`**: Change the `dataset_root` default.
2. **Use CLI argument**: `python src/verify_dataset.py --dataset-root /path/to/data`

---

## 6. Step-by-Step Execution Guide

### Overview

```
Step 1: Setup environment
Step 2: Install dependencies
Step 3: Verify dataset
Step 4: Split dataset (video-level)
Step 5: Extract frames (face crops)
Step 6: Extract ResNet-18 features
Step 7: Train RBF-SVM
Step 8: Evaluate on test set
```

---

### STEP 1 — Verify Dataset

**Command:**
```bash
python src/verify_dataset.py
```

**What it does:**
- Checks that the dataset directory exists.
- Verifies all expected sub-directories (original + 4 manipulation types).
- Counts videos in each category.

**Computationally expensive?** No (instant).

**Expected output:**
```
  FaceForensics++ Dataset Verification
  ─────────────────────────────────────
  Original videos directory: OK
  Deepfakes directory: OK
  Face2Face directory: OK
  FaceSwap directory: OK
  NeuralTextures directory: OK

  Total videos found:    5,000
  Real (original):       1,000
  Fake (manipulated):    4,000

  ✅ Dataset verification PASSED
```

---

### STEP 2 — Split Dataset

**Command (full dataset):**
```bash
python src/split_dataset.py
```

**Command (smoke test — 20 videos per category):**
```bash
python src/split_dataset.py --max-videos 20
```

**What it does:**
- Discovers all videos.
- Performs stratified video-level splitting (70/15/15).
- Verifies zero overlap between splits (data leakage check).
- Saves metadata CSVs.

**Computationally expensive?** No (instant).

**Generated files:**
```
data/metadata/
├── train_videos.csv
├── val_videos.csv
├── test_videos.csv
└── all_videos.csv
```

**Expected output includes:**
```
  Dataset Split Verification
  ─────────────────────────
  Train/Validation overlap: 0
  Train/Test overlap:       0
  Validation/Test overlap:  0

  ✅ Split verification: PASSED — no data leakage
```

---

### STEP 3 — Extract Frames

**Command (smoke test):**
```bash
python src/extract_frames.py --max-videos 20
```

**Command (full dataset):**
```bash
python src/extract_frames.py
```

**What it does:**
- Opens each video with OpenCV.
- Samples every 10th frame (configurable with `--frame-interval`).
- Detects the largest face using Haar Cascade.
- Crops and resizes the face to 224×224.
- Saves the face crop as JPEG.
- Creates a frame manifest CSV.

**Computationally expensive?** ⚠️ **YES** — processing all 5,000 videos takes significant time. Use `--max-videos` for initial testing.

**Generated files:**
```
data/frames/{train,val,test}/{video_id}/frame_000000.jpg
data/metadata/frame_manifest.csv
```

---

### STEP 4 — Extract ResNet-18 Features

**Command:**
```bash
python src/extract_features.py
```

**What it does:**
- Loads all face crop images.
- Passes them through pretrained ResNet-18 (with final FC removed).
- Extracts 512-dimensional feature vectors.
- Saves features, labels, and video IDs as `.npy` files.

**Computationally expensive?** ⚠️ **YES** — GPU/MPS acceleration helps significantly. Use `--device cpu` to force CPU if MPS causes issues.

**Generated files:**
```
features/
├── train_features.npy    ← shape: (N_train, 512)
├── train_labels.npy
├── train_video_ids.npy
├── val_features.npy
├── val_labels.npy
├── val_video_ids.npy
├── test_features.npy
├── test_labels.npy
└── test_video_ids.npy
```

---

### STEP 5 — Train Classical Model

**Command:**
```bash
python src/train.py
```

**With custom parameters:**
```bash
python src/train.py --pca-components 32 --svm-c 1.0
```

**What it does:**
1. Loads training features from `features/train_features.npy`.
2. Fits `StandardScaler` on training data (and transforms validation data).
3. Fits `PCA` on training data (reports explained variance).
4. Trains `RBF-SVM` with `class_weight="balanced"`.
5. Reports validation accuracy and classification report.
6. Saves scaler, PCA, and SVM to `models/`.

**Computationally expensive?** Moderate — SVM training on the PCA-reduced features is usually fast (seconds to minutes).

**Generated files:**
```
models/
├── scaler.pkl            ← Fitted StandardScaler
├── pca.pkl               ← Fitted PCA transform
├── svm.pkl               ← Trained RBF-SVM classifier
└── training_config.json  ← Hyperparameters and training metadata
```

---

### STEP 6 — Evaluate on Test Set

**Command:**
```bash
python src/evaluate.py
```

**What it does:**
1. Loads saved scaler, PCA, and SVM.
2. Loads test features.
3. Transforms test features through scaler → PCA.
4. Generates frame-level predictions.
5. Aggregates predictions at video level (mean probability, threshold = 0.5).
6. Computes all metrics (accuracy, precision, recall, F1, ROC-AUC).
7. Generates confusion matrix and ROC curve plots.
8. Saves all results.

**Computationally expensive?** No (instant).

**Generated files:**
```
results/
├── confusion_matrix_frame.png
├── roc_curve_frame.png
├── confusion_matrix_video.png
├── roc_curve_video.png
├── metrics.json              ← All metrics in structured format
├── classification_report.txt ← Human-readable report
└── experiment_config.json    ← Full experiment configuration
```

---

## 7. Smoke Test Workflow

For quick verification that the entire pipeline works, use a small subset:

```bash
# 1. Verify dataset
python src/verify_dataset.py

# 2. Split with limited videos (20 per category = 100 total)
python src/split_dataset.py --max-videos 20

# 3. Extract frames from the limited set
python src/extract_frames.py --max-videos 20

# 4. Extract CNN features
python src/extract_features.py

# 5. Train the model
python src/train.py

# 6. Evaluate
python src/evaluate.py
```

This will process approximately 100 videos instead of 5,000 and should complete in minutes rather than hours.

---

## 8. Full Dataset Execution

```bash
# 1. Verify dataset
python src/verify_dataset.py

# 2. Split ALL videos
python src/split_dataset.py

# 3. Extract frames from ALL videos (⚠️ time-consuming)
python src/extract_frames.py

# 4. Extract CNN features (⚠️ time-consuming — use GPU/MPS if available)
python src/extract_features.py

# 5. Train the model
python src/train.py

# 6. Final evaluation
python src/evaluate.py
```

---

## 9. Technical Details

### 9.1 Data Leakage Prevention

Splitting is performed at the **video level**, not the frame level. All frames from one video remain in the same split. This prevents inflated accuracy from train/test frame overlap within a single video.

The split verification script checks for zero overlap and **raises an error** if any leakage is detected.

### 9.2 Feature Extraction Pipeline

```
Face Crop (224×224×3)
        │
        ▼
ImageNet Normalisation (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        │
        ▼
ResNet-18 (pretrained, frozen, FC removed)
        │
        ▼
512-dimensional Feature Vector
```

### 9.3 StandardScaler → PCA → SVM

- **StandardScaler**: Fit on training data only. Ensures zero mean and unit variance.
- **PCA (64 components)**: Fit on training data only. Reduces 512 → 64 dimensions while retaining most variance.
- **RBF-SVM**: Trained with `class_weight="balanced"` to handle class imbalance. Probability estimates enabled for ROC-AUC.

### 9.4 Video-Level Evaluation

For each test video:
1. Collect fake probabilities from all its frames.
2. Compute the **mean** fake probability.
3. Apply threshold (default: 0.5).
4. One prediction per video.

This is important because real-world deepfake detection operates on complete media, not individual frames.

### 9.5 Apple Silicon Compatibility

The pipeline runs on **CPU** by default. If you have an Apple Silicon Mac with PyTorch MPS support, the code will automatically use MPS for CNN feature extraction. Override with `--device cpu` if needed.

---

## 10. Model Artefact Locations

| Artefact | Path | Description |
|---|---|---|
| StandardScaler | `models/scaler.pkl` | Fitted feature standardiser |
| PCA | `models/pca.pkl` | Fitted PCA transformation |
| RBF-SVM | `models/svm.pkl` | Trained classical classifier |
| Training config | `models/training_config.json` | Hyperparameters used |
| Training features | `features/train_features.npy` | Raw ResNet-18 features |
| PCA features | (computed at runtime) | StandardScaler + PCA applied |

---

## 11. Future Quantum Extension (QSVM)

### 11.1 How to Reuse Features

The quantum module will load the **exact same** PCA features:

```python
# Load saved pipeline components
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca.pkl")

# Load raw features
X = np.load("features/train_features.npy")

# Apply same preprocessing
X_scaled = scaler.transform(X)
X_pca = pca.transform(X_scaled)

# X_pca is now ready for quantum feature mapping
# quantum_kernel = compute_quantum_kernel(X_pca)
# qsvm = QSVM(quantum_kernel)
```

### 11.2 Fair Comparison Guarantee

Both classical and quantum classifiers will use:

| Component | Shared? |
|---|---|
| FaceForensics++ dataset | ✅ Same |
| Video-level split | ✅ Same |
| Frame extraction | ✅ Same |
| Face detection | ✅ Same |
| ResNet-18 features | ✅ Same |
| StandardScaler | ✅ Same |
| PCA | ✅ Same |
| Evaluation metrics | ✅ Same |
| **Classifier** | ❌ RBF-SVM vs QSVM |

### 11.3 Modular Classifier Design

The training script is designed so that the classifier can be swapped:

```python
# Current (classical):
features = load_pca_features()
classifier = SVC(kernel="rbf", ...)
classifier.fit(features, labels)

# Future (quantum):
features = load_pca_features()
classifier = QuantumKernelSVM(...)
classifier.fit(features, labels)
```

---

## 12. Reproducibility

| Mechanism | Value |
|---|---|
| Random seed | `42` (set across all libraries) |
| PyTorch deterministic mode | Enabled |
| Split ratios | 70/15/15 (stratified by manipulation type) |
| Frame interval | Every 10th frame |
| PCA fitted on | Training data only |
| Scaler fitted on | Training data only |
| Test set used for | Final evaluation only (never for model selection) |

---

## 13. Computational Considerations

| Stage | Time Estimate | Storage |
|---|---|---|
| Dataset verification | < 1 second | None |
| Video splitting | < 1 second | ~100 KB (CSVs) |
| Frame extraction (full) | 30–120 minutes | 5–20 GB |
| Feature extraction (full) | 30–90 minutes | ~100 MB |
| Training | 1–10 minutes | ~10 MB |
| Evaluation | < 30 seconds | ~5 MB |

Use `--max-videos 20` to run a quick smoke test before committing to the full dataset.

---

## 14. Troubleshooting

| Error | Solution |
|---|---|
| `Dataset root NOT FOUND` | Update `dataset_root` in `src/config.py` or use `--dataset-root` |
| `Split metadata not found` | Run `python src/split_dataset.py` first |
| `Frame manifest not found` | Run `python src/extract_frames.py` first |
| `Feature file not found` | Run `python src/extract_features.py` first |
| `Model file not found` | Run `python src/train.py` first |
| MPS errors on Apple Silicon | Use `--device cpu` to force CPU mode |
| `Invalid PCA dimensions` | Reduce `--pca-components` (must be ≤ number of samples) |

---

## 15. License & Citation

This project uses FaceForensics++. If you use this work, please cite:

```bibtex
@inproceedings{roessler2019faceforensicspp,
    title={FaceForensics++: Learning to Detect Manipulated Facial Images},
    author={Rössler, Andreas and Cozzolino, Davide and Verdoliva, Luisa
            and Riess, Christian and Thies, Justus and Nießner, Matthias},
    booktitle={International Conference on Computer Vision (ICCV)},
    year={2019}
}
```
