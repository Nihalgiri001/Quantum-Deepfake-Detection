"""
TrustLens-Q — Quantum Kernel SVM Package
==========================================
Quantum computing extension for the TrustLens deepfake detection system.

This package implements a Quantum Kernel Support Vector Machine (QSVM)
pipeline that extends the classical ResNet-18 feature extraction with
a ZZFeatureMap-based fidelity quantum kernel.

Public API:
    QuantumConfig       — central configuration for all quantum parameters
    QuantumFeatureReducer — 512-dim → N-dim quantum PCA preprocessing
    QuantumSVM          — SVC with precomputed quantum kernel
    SimulatorBackend    — local Qiskit Aer simulator wrapper
"""

from .quantum_config import QuantumConfig

__version__ = "1.0.0"
__all__ = ["QuantumConfig"]
