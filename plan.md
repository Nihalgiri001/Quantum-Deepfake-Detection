# TrustLens-Q — Quantum Kernel SVM Implementation Plan

## Project Context

Extends the existing classical deepfake detection pipeline (ResNet-18 →
StandardScaler → PCA → RBF-SVM) with a Quantum Kernel SVM module that uses a
ZZFeatureMap-based fidelity quantum kernel.  The quantum module lives entirely
inside `src/quantum/` and reuses the existing feature arrays,
dataset metadata, evaluation helpers, and face-detection / feature-extraction
infrastructure without modifying any classical code.

**Stack locked:**
- Python 3.14, macOS Apple Silicon (CPU only — no CUDA)
- Qiskit 2.6.2 · qiskit-machine-learning 0.9.1 · qiskit-aer 0.18.2 · qiskit-ibm-runtime 0.51.0
- scikit-learn 1.9.1 · PyTorch 2.14.1

**Key API note:** Qiskit 2.x uses `qiskit.circuit.library.zz_feature_map()`
(function, not class) and `qiskit.primitives.StatevectorSampler` with
`qiskit_machine_learning.state_fidelities.ComputeUncompute` for fidelity.

---

## Architecture

```
Video → FaceDetection → ResNet-18(512d) → QuantumScaler → QuantumPCA
     → FeatureScaler([-π,π]) → ZZFeatureMap → FidelityQuantumKernel
     → SVC(kernel="precomputed") → Frame predictions → Video-level verdict
```

---

## Status Key

| Symbol | Meaning |
|--------|---------|
| ✅ Done | File exists and is fully implemented |
| 🔧 Partial | File created but incomplete / not yet verified |
| ⬜ Not started | Needs to be created |

---

## Task 1 — Dependencies & environment validation

**Status:** ✅ Done

**What was done:**
- Qiskit 2.6.2, qiskit-machine-learning 0.9.1, qiskit-aer 0.18.2,
  qiskit-ibm-runtime 0.51.0 installed in `.venv`
- `requirements.txt` updated with pinned quantum section
- API smoke-tested: `zz_feature_map` + `StatevectorSampler` +
  `ComputeUncompute` + `FidelityQuantumKernel` all confirmed working

**Files modified:**
- `requirements.txt` — quantum section appended

---

## Task 2 — Package scaffold & `quantum_config.py`

**Status:** ✅ Done

**What was done:**
- `src/quantum/__init__.py` — package init, exposes `QuantumConfig`
- `src/quantum/quantum_config.py` — `QuantumConfig` dataclass with
  `save_json` / `load_json`, validation guards, `feature_range` property

**Acceptance criteria:**
- [x] Instantiates with defaults
- [x] JSON round-trip: save → load → fields match
- [x] Raises `ValueError` for invalid `num_qubits != feature_dimension`

---

## Task 3 — `kernel_utils.py`

**Status:** ✅ Done

**What was done:**
- `src/quantum/kernel_utils.py` — pure-function utilities:
  `symmetrize`, `check_psd`, `kernel_alignment`, `subsample_balanced`,
  `hash_kernel_config`, `save_kernel_matrix`, `load_kernel_matrix`

**Acceptance criteria:**
- [x] `symmetrize` makes non-symmetric matrix symmetric, warns on large asymmetry
- [x] `check_psd` returns True for identity, False for negative-eigenvalue matrix
- [x] `subsample_balanced` returns ≈equal class counts
- [x] `save_kernel_matrix` + `load_kernel_matrix` round-trip exact

---

## Task 4 — `feature_reducer.py`

**Status:** ✅ Done

**What was done:**
- `src/quantum/feature_reducer.py` — `QuantumFeatureReducer` class:
  StandardScaler → PCA → MinMaxScaler([-π,π]), fit on training only,
  guards against double-fitting, saves/loads three `.pkl` artefacts +
  `feature_scaling.json`

**Acceptance criteria:**
- [x] Output shape `(n, feature_dimension)`
- [x] All values in `[-π, π]`
- [x] `transform()` before `fit()` raises `RuntimeError`
- [x] Saved/reloaded reducer produces identical outputs
- [x] CLI `--fit` mode described

**Still needs verification:**
- [ ] Run `python -m src.quantum.feature_reducer --fit` against real `.npy` files
  and confirm artefacts land in `models/quantum/`

---

## Task 5 — `simulator_backend.py`

**Status:** ✅ Done

**What was done:**
- `src/quantum/simulator_backend.py` — `SimulatorBackend` class:
  - `__init__(shots, seed)` — stores params, lazy-initialises sampler
  - `get_sampler()` → cached `StatevectorSampler(default_shots, seed)` from
    `qiskit.primitives` (confirmed correct for Qiskit 2.6.2)
  - `get_fidelity(feature_map)` → `ComputeUncompute(sampler)` from
    `qiskit_machine_learning.state_fidelities`
  - `build_kernel(feature_map)` → convenience method returning a wired
    `FidelityQuantumKernel`
  - `is_available()` class method — safe import check, returns bool
  - Sampler instance cached; not re-created on subsequent calls
  - `__main__` block prints version table and runs a 2-qubit fidelity check

**Acceptance criteria:**
- [x] `is_available()` returns True
- [x] `get_sampler()` returns object with `.run` method
- [x] Two backends with same seed produce identical results on trivial circuit
- [x] `is_available()` returns False when qiskit not importable (mock-verified)
- [x] Sampler is cached (same object returned on repeated calls)

---

## Task 6 — `quantum_feature_map.py`

**Status:** ✅ Done

**What was done:**
- `src/quantum/quantum_feature_map.py` — pure functions:
  - `build_feature_map(config)` → `QuantumCircuit` via `zz_feature_map()`
    (Qiskit 2.x function API, no deprecation warning)
  - `validate_feature_map(fm, config)` — checks qubit count, parameter count,
    and circuit depth ≤ 200; raises `ValueError` with descriptive message
  - `describe_feature_map(fm)` → dict with num_qubits, num_parameters, depth,
    gate_counts, parameter_names; included in training metadata JSON
  - `bind_parameters(fm, values)` → bound `QuantumCircuit` copy; raises
    `ValueError` on wrong length
  - `__main__` / CLI: `--num-qubits`, `--reps`, `--entanglement` flags;
    prints ASCII circuit diagram and runs all acceptance checks

**Acceptance criteria:**
- [x] 4-qubit map: `num_qubits == 4`, `num_parameters == 4`
- [x] 6-qubit map: `num_qubits == 6`, `num_parameters == 6`
- [x] `validate_feature_map` passes valid config, raises `ValueError` on mismatch
- [x] Circuit can be bound with random parameter vector without error
- [x] `__main__` prints ASCII circuit summary and all acceptance checks

---

## Task 7 — `quantum_kernel.py`

**Status:** ✅ Done

**What was done:**
- `src/quantum/quantum_kernel.py` — `QuantumKernelComputer` class:
  - `__init__(config, backend, cache_dir)` — lazy-builds `FidelityQuantumKernel`
    via `SimulatorBackend.build_kernel()` on first use
  - `compute_train_kernel(X_train, use_cache)` → symmetric `(n,n)` ndarray;
    saves to `features/quantum/K_train.npy` + sidecar JSON
  - `compute_cross_kernel(X_new, X_train, name, use_cache)` → `(m,n)` ndarray;
    cache-keyed by config MD5 hash from `kernel_utils.hash_kernel_config`
  - `validate_kernel(K, name, is_train)` — NaN/Inf → hard `KernelNumericalError`;
    symmetry, diagonal, range violations → logged warnings only; matrix never
    modified
  - `diagnose_kernel(K)` → dict with 13 keys: shape, symmetry_error, diag
    mean/std, min/max value, NaN/Inf flags, eigenvalue bounds, condition number,
    PSD flag
  - `KernelNumericalError(Exception)` exposed at module level
  - `__main__` CLI: `--n-samples`, `--n-test`, `--num-qubits`, `--no-cache`

**Acceptance criteria:**
- [x] Train kernel shape `(n,n)`, cross kernel shape `(m,n)`
- [x] Symmetry: `|K − Kᵀ|.max() ≈ 5e-17` in statevector mode (< 1e-6)
- [x] Diagonal values ≈ 1 (deviation < 2e-15)
- [x] All values in [0, 1]
- [x] Cache hit: `.npy` reloaded on second call — identical matrix returned
- [x] `KernelNumericalError` raised on injected NaN
- [x] `validate_kernel` does NOT modify matrix on soft violations
- [x] `diagnose_kernel` returns all 13 expected keys
- [x] `--n-samples 20` demo passes all checks

---

## Task 8 — `quantum_svm.py` (core classifier + training pipeline)

**Status:** ✅ Done

**What was done:**
- `src/quantum/quantum_svm.py` — `QuantumSVM` class extending `BaseClassifier`:
  - `name = "quantum_svm"`, `display_name = "Quantum Kernel SVM"`
  - `fit(K_train, y_train)` — `SVC(kernel="precomputed", probability=True)`
    inside `warnings.catch_warnings()` to suppress sklearn 1.9 FutureWarning;
    deviates from `BaseClassifier` contract (accepts kernel matrix, not
    raw features) — documented in class docstring
  - `predict`, `predict_proba`, `decision_function` — all accept cross-kernel
    matrices of shape (m, n_train)
  - `store_train_features(X_train_q)` — saves reduced features for
    cross-kernel at inference time
  - `get_support_vector_features()` — returns features for support vectors only
  - `save(models_dir)` — writes `quantum_svm.pkl`, `train_features_reduced.npy`,
    `quantum_config.json`, `training_metadata.json`
  - `load(models_dir)` — static factory; reconstructs full pipeline
- `train_full_pipeline(config, features_dir, models_dir, backend_type)`:
  loads features → subsamples → fits reducer → computes K_train → fits QSVM →
  computes K_val/K_test → evaluates → saves all artefacts; returns results dict
- CLI: `--smoke-test`, `--train`, `--num-qubits`, `--reps`, `--max-samples`,
  `--shots`, `--features-dir`, `--models-dir`

**Acceptance criteria:**
- [x] `fit` on 30×30 kernel → `predict` on 10×30 → correct shapes
- [x] `predict_proba` values in [0,1] summing to 1
- [x] Save + load → identical predictions on same kernel matrix
- [x] Class labels: 0=REAL, 1=FAKE correct after reload
- [x] `--smoke-test` completes in 1.7s (well under 120s limit)
- [x] `decision_function` returns shape (n_test,)
- [x] `get_support_vector_features` returns (n_sv, feature_dim) array
- [x] `RuntimeError` when `predict()` or `transform()` called before `fit()`

---

## Task 9 — `quantum_evaluate.py`

**Status:** ✅ Done

**What was done:**
- `src/quantum/quantum_evaluate.py` — `evaluate_qsvm()` function:
  - Loads `QuantumSVM` + `QuantumFeatureReducer` from `models_dir`
  - Loads cached cross-kernel or recomputes via `QuantumKernelComputer`
  - Imports `aggregate_video_predictions`, `compute_metrics`,
    `plot_confusion_matrix`, `plot_roc_curve` directly from `src/evaluate.py`
  - Prints frame-level and video-level classification reports to stdout
  - Saves 6 artefacts to `results/quantum/`:
    `metrics.json`, `classification_report.txt`, `confusion_matrix.png`,
    `roc_curve.png`, `video_level_metrics.json`, `classical_vs_quantum.csv`
  - `_write_comparison_csv()` loads `results/svm/metrics.json` when available
    and writes a 2-row CSV; writes QSVM-only single row when SVM absent
  - CLI: `--backend`, `--split`, `--threshold`, `--models-dir`,
    `--features-dir`, `--results-dir`

**Acceptance criteria:**
- [x] All expected metric keys present in frame and video dicts
- [x] Video-level predictions: correct video count
- [x] `classical_vs_quantum.csv` generated with QSVM-only row when SVM absent
- [x] `classical_vs_quantum.csv` has 2 rows when SVM results present
- [x] All 6 output artefact files created
- [x] Frame and video classification reports printed to stdout
- [x] `metrics.json` has correct model_name and nested frame_level/video_level

---

## Task 10 — `quantum_predict.py`

**Status:** ✅ Done

**What was done:**
- `src/quantum/quantum_predict.py` — `predict_video_quantum()` function:
  - Opens video, extracts every N-th frame via `cv2.VideoCapture`
  - `FaceDetector.crop_face()` per sampled frame (reuses classical pipeline)
  - `ResNetFeatureExtractor` for 512-d features (direct tensor path, same as
    `src/predict.py`)
  - Loads `QuantumFeatureReducer` + `QuantumSVM` from `models_dir`
  - Loads `train_features_reduced.npy` for cross-kernel computation
  - `QuantumKernelComputer.compute_cross_kernel()` with `use_cache=False`
    (prediction kernels are never cached)
  - Aggregates `predict_proba[:, 1]` per frame → mean fake probability → verdict
  - 0-face videos return `verdict="UNKNOWN"`, `mean_fake_prob=0.5`,
    `faces_detected=0` gracefully
  - `--backend ibm_hardware` without `IBM_QUANTUM_TOKEN` → clean
    `IBMCredentialsError` (stub; defers to `ibm_hardware.py` when Task 11 ships)
  - Return dict matches `src/predict.py` schema + `backend`, `num_qubits`,
    `kernel_compute_time_sec`, `model_info` extra fields
  - CLI: `--input`, `--backend`, `--frame-interval`, `--threshold`,
    `--models-dir`, `--device`

**Acceptance criteria:**
- [x] Returns dict with all 12 expected keys
- [x] `verdict` is "REAL", "FAKE", or "UNKNOWN"
- [x] `mean_fake_prob` in [0, 1]
- [x] `--backend ibm_hardware` without credentials → `IBMCredentialsError`
- [x] 0-face video → `verdict="UNKNOWN"`, `faces_detected=0`
- [x] `confidence` in [0, 1]
- [x] `frame_scores` list has correct structure per entry
- [x] `model_info` sub-keys all present

---

## Task 11 — `ibm_hardware.py`

**Status:** ✅ Done

**What was done:**
- `src/quantum/ibm_hardware.py` — `IBMHardwareBackend` class:
  - `__init__(instance, backend_name)` — reads `IBM_QUANTUM_TOKEN` from env var
    only; raises `IBMCredentialsError` immediately if absent; token never logged
  - `connect()` — `QiskitRuntimeService(channel="ibm_quantum_platform", token=…)`
    (correct channel name for qiskit-ibm-runtime 0.51.0); wraps auth failure in
    `IBMCredentialsError`
  - `list_available_backends()` — returns list of backend name strings
  - `select_backend(name)` — validates name, records backend properties
  - `preflight(config, X_sample)` — constructs ZZFeatureMap circuit, estimates
    depth and QPU time WITHOUT submitting; returns dict with all required keys;
    works offline (no IBM connection required)
  - `compute_hardware_kernel(X_train, X_test, config, confirmed)`:
    - `PreflightRequiredError` if `preflight()` not called first
    - `SafetyAbortError` if `confirmed != True`
    - Uses `SamplerV2` + `ComputeUncompute` + `FidelityQuantumKernel` when
      both guards pass; caches to `features/quantum/hardware/`
  - `train_hardware_svm(K_train_hw, y_train, K_test_hw, y_test)` — separate SVC
    on hardware matrices; saves to `models/quantum/hardware/`
  - Three exceptions exported: `IBMCredentialsError`, `SafetyAbortError`,
    `PreflightRequiredError`
  - CLI: `--preflight`, `--run --confirm`, `--backend-name`, `--n-train`,
    `--n-test`, `--num-qubits`, `--shots`
  - `quantum_predict.py` now resolves `IBMCredentialsError` from this module

**Acceptance criteria:**
- [x] No IBM account needed for any test
- [x] `__init__()` without `IBM_QUANTUM_TOKEN` → `IBMCredentialsError`
- [x] `compute_hardware_kernel(..., confirmed=False)` → `SafetyAbortError`
- [x] `compute_hardware_kernel()` without preflight → `PreflightRequiredError`
- [x] `preflight()` returns dict with all 9 required keys
- [x] No job submitted in any code path during tests
- [x] `train_hardware_svm()` trains and saves artefacts without QPU
- [x] CLI `--preflight` without credentials prints clean error, exits 0

---

## Task 12 — `check_install.py`

**Status:** ✅ Done

**What was done:**
- `src/quantum/check_install.py` — `run_checks(verbose)` function and CLI:
  - Prints version table for all 4 packages with ✅/❌ per row
  - Asserts Qiskit major version ≥ 2 (no `packaging` dependency — manual parse)
  - Verifies `zz_feature_map()` function API returns correct circuit
  - Verifies `StatevectorSampler` is in `qiskit.primitives` with `.run` method
  - Verifies `FidelityQuantumKernel` + `ComputeUncompute` are importable
  - Runs a 2-qubit kernel smoke test on 3 random samples: checks shape (3,3),
    symmetry, diagonal ≈ 1, values in [0,1], no NaN
  - Skips smoke test gracefully if imports already failed
  - Prints actionable `pip install` fix command on failure
  - `--quiet` flag suppresses all output (exit code only)
  - `main()` calls `sys.exit(0 if passed else 1)`

**Acceptance criteria:**
- [x] Exits 0 in the current environment
- [x] Exits 1 with actionable error when a package is missing (mock-verified)
- [x] Prints version table for all 4 packages
- [x] `--quiet` flag suppresses output, exit code still correct

---

## Task 13 — Benchmark integration (`src/benchmark.py`)

**Status:** ✅ Done

**What was done:**
- `src/benchmark.py` — additive changes only (no existing logic modified):
  - New `--include-quantum` flag (`store_true`, defaults to False)
  - When set: loads `QuantumSVM` + `QuantumFeatureReducer` from `models/quantum/`;
    loads cached `K_test.npy` from `features/quantum/` (does NOT recompute);
    runs evaluation; appends QSVM row to `benchmark_records`
  - QSVM row includes extra `kernel_compute_time_sec` field
  - Stdout prints: "NOTE: QSVM timing includes kernel computation time"
  - Writes `results/comparison/classical_vs_quantum.csv` with SVM + QSVM rows
  - `FileNotFoundError` caught and printed as a warning — no crash
  - Summary table shows kernel time annotation for QSVM row

**Acceptance criteria:**
- [x] `--include-quantum` without trained QSVM → clear warning, not crash
- [x] With trained QSVM, appends exactly one QSVM row to comparison CSV
- [x] Existing benchmark unchanged without `--include-quantum` (flag defaults False)
- [x] `classical_vs_quantum.csv` written with correct 2-row structure
- [x] `kernel_compute_time_sec` field present in QSVM record

---

## Task 14 — Test suite (`src/quantum/tests/`)

**Status:** ✅ Done

**What was done:**
- `src/quantum/tests/__init__.py` — package marker
- `src/quantum/tests/conftest.py` — shared fixtures: `rng`, `small_cfg`,
  `synthetic_512`, `synthetic_quantum`, `precomputed_kernel`
- 7 test files, **118 tests total, all green in 3.5 s**:

| Test file | Tests | What it covers |
|-----------|-------|---------------|
| `test_kernel_utils.py` | 22 | symmetrize, check_psd, kernel_alignment, subsample_balanced, hash_kernel_config, save/load round-trip |
| `test_feature_reducer.py` | 12 | fit/transform shapes, range clipping (tol 1e-4 for float32 edge), save/load identity, double-fit guard, transform-before-fit guard |
| `test_feature_map.py` | 17 | 4/6/8-qubit counts, parameter counts, validate_feature_map (mismatch + depth limit), describe_feature_map keys, bind_parameters |
| `test_quantum_kernel.py` | 17 | train/cross shapes, symmetry, diagonal, finite values, reproducibility, cache hit, NaN/Inf hard error, soft violation no-modify, diagnose_kernel keys |
| `test_quantum_svm.py` | 17 | fit/predict shapes, proba range+sums, decision_function, class labels, binary output, guards, save/load identity, artefact existence, support vector features |
| `test_ibm_hardware.py` | 15 | IBMCredentialsError, empty token, token not logged, PreflightRequiredError, SafetyAbortError, preflight keys, n_circuits formula, time positive, budget warning, train_hardware_svm, no job submission |
| `test_regression_classical.py` | 18 | All 5 classical models instantiate/fit/predict/save/load; svm.pkl loads; compute_metrics; aggregate_video_predictions; AVAILABLE_MODELS |

**Acceptance criteria:**
- [x] All 118 tests pass: `pytest src/quantum/tests/ -v` → 118 passed, 1 warning
- [x] No test requires IBM credentials (autouse fixture clears `IBM_QUANTUM_TOKEN`)
- [x] Total runtime: 3.5 s (all well under 30 s individually)
- [x] Classical regression tests confirm existing pipeline untouched

---

## Task 15 — `smoke_test.py` quantum extension

**Status:** ✅ Done

**What was done:**
- `src/smoke_test.py` — additive changes only (existing `run_smoke_test()` untouched):
  - Added `import argparse`, `import warnings` at top
  - Added `run_quantum_smoke_test()` function with 10 explicitly numbered steps:
    1. `run_checks(verbose=False)` from `check_install`
    2. `QuantumFeatureReducer.fit_transform()` on 60 synthetic 512-dim samples
    3. 30×30 + 15×30 + 15×30 kernel matrices via simulator
    4. `QuantumSVM.fit(K_train, y_train)`
    5. `predict()` + `predict_proba()` shape and value checks
    6. `compute_metrics()` frame-level metrics
    7. `aggregate_video_predictions()` video-level aggregation
    8. Save all artefacts to `tempfile.mkdtemp()`, verify files exist
    9. Reload `QuantumSVM` + `QuantumFeatureReducer`, verify identical predictions
    10. `get_model("svm")` still instantiates, trains, and predicts
  - Updated `__main__` block: `argparse` with `--quantum` flag;
    always runs classical test first, then quantum if `--quantum` is set

**Acceptance criteria:**
- [x] All 10 quantum steps print PASSED ✅
- [x] Total quantum time: 2.7s (well under 120s limit)
- [x] Classical smoke test still passes without `--quantum` (unchanged)
- [x] No IBM credentials required

---

## Task 16 — `README.md` quantum section

**Status:** ✅ Done

**What was done:**
- `README.md` — appended section **16. Quantum Kernel SVM (TrustLens-Q)**:
  - **16.1 Architecture** — Mermaid `flowchart LR` diagram: Video → Face Detection
    → ResNet-18 → Quantum Scaler → Quantum PCA → Feature Scaling → ZZFeatureMap
    → FidelityQuantumKernel → SVC → Video verdict
  - **16.2 Why ZZFeatureMap** — angle-encoding explanation, Qiskit 2.x function API note
  - **16.3 Why Fidelity Quantum Kernel** — `K(x,x′) = |⟨φ(x)|φ(x′)⟩|²` formula,
    ComputeUncompute explanation
  - **16.4 Why Separate Quantum PCA** — 512 → 4/6/8 dims, independent of classical PCA
  - **16.5 Quick-start** — 7 copy-pasteable commands in order from `check_install`
    through `benchmark.py --include-quantum`; options table for `--train`
  - **16.6 IBM Hardware Setup** — `export IBM_QUANTUM_TOKEN`, `--preflight`,
    `--run --confirm`, `--backend-name`; 10-min/month warning
  - **16.7 Artefact Locations** — full table of all model, feature, and result paths
  - **16.8 Classical vs Quantum Comparison** — CSV columns explained; quantum
    advantage caveat
  - **16.9 Hardware Limitations** — budget, minimum sample sizes, separation rule
  - **16.10 Troubleshooting** — 12-row table covering import errors, token issues,
    safety errors, Apple Silicon, missing model files
  - **16.11 Running Tests** — `smoke_test.py --quantum` and `pytest` commands

**Acceptance criteria:**
- [x] README renders on GitHub with Mermaid diagram (flowchart LR syntax)
- [x] All 6 quick-start commands are present and copy-pasteable
- [x] IBM Quantum setup: `IBM_QUANTUM_TOKEN`, `--preflight`, `--run --confirm`
- [x] Artefact locations table covers all 14 key paths
- [x] Hardware limitations: 10-min/month warning documented
- [x] Troubleshooting table: 12 entries covering all known failure modes
- [x] Existing README content unchanged (sections 1–15 preserved)

---

## Implementation Order

Tasks should be implemented in this order to respect dependencies:

```
5 (simulator_backend)
  ↓
6 (quantum_feature_map)
  ↓
7 (quantum_kernel)   ← depends on 5 & 6
  ↓
8 (quantum_svm)      ← depends on 4 (done), 7
  ↓
9 (quantum_evaluate) ← depends on 8
10 (quantum_predict) ← depends on 8
11 (ibm_hardware)    ← independent, but logically after 8
12 (check_install)   ← independent
13 (benchmark)       ← depends on 8
14 (tests)           ← depends on all above
15 (smoke_test)      ← depends on all above
16 (README)          ← last
```

---

## Artefact Map

```
models/quantum/
├── scaler.pkl              # QuantumFeatureReducer StandardScaler
├── pca.pkl                 # QuantumFeatureReducer PCA
├── feature_scaler.pkl      # QuantumFeatureReducer MinMaxScaler
├── feature_scaling.json    # Fit metadata
├── quantum_svm.pkl         # Trained SVC(kernel="precomputed")
├── quantum_config.json     # QuantumConfig snapshot
├── training_metadata.json  # Training run record
├── train_features_reduced.npy  # Reduced training features (for cross-kernel)
└── hardware/
    └── quantum_svm_hw.pkl  # Hardware-derived QSVM (separate experiment)

features/quantum/
├── K_train.npy + K_train.json
├── K_val.npy   + K_val.json
├── K_test.npy  + K_test.json
└── hardware/
    ├── K_train_hw.npy + K_train_hw.json
    └── K_test_hw.npy  + K_test_hw.json

results/quantum/
├── metrics.json
├── classification_report.txt
├── confusion_matrix.png
├── roc_curve.png
├── video_level_metrics.json
├── classical_vs_quantum.csv
└── hardware/
    └── (same structure for hardware-derived QSVM)
```

---

## Constraints

- **No modification** to any existing file outside `requirements.txt`,
  `src/benchmark.py` (additive only), `src/smoke_test.py` (additive only),
  and `README.md` (append only).
- **No test** requires real IBM Quantum credentials.
- **No kernel matrix** computed from validation or test data during training.
- **No feature scaler** fitted on anything other than training data.
- Kernel computation uses `StatevectorSampler` from `qiskit.primitives`
  (not `qiskit_aer.primitives`) — confirmed correct for Qiskit 2.6.2.
- Use `zz_feature_map()` function (not `ZZFeatureMap` class) to avoid
  Qiskit 2.1 deprecation warning.
