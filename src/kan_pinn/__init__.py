"""
KAN-PINN Module
===============

Kolmogorov-Arnold Network + Physics-Informed Neural Network for sensorless
position observation in magnetic levitator.

This module implements a two-stage architecture:
1. Flux Observer (Stage 1): (u, i) → φ̂
2. Position Predictor (Stage 2): (u, i, φ̂) → ŷ

Requirements:
- PyTorch >= 1.12
- numpy
- scipy

Note: This module requires PyTorch to be installed. If PyTorch is not available,
the module will not be functional but can still be imported.
"""

import sys
import warnings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch is not installed. KAN-PINN modules will not be functional. "
        "Install PyTorch to use KAN-PINN features: pip install torch",
        ImportWarning
    )

__all__ = []

if TORCH_AVAILABLE:
    try:
        from .hippo_layer import HiPPOLayer
        from .kan_layer import KANLayer
        from .flux_observer import FluxObserver
        from .position_predictor import PositionPredictor
        from .physics_loss import KirchhoffLoss, PINNLoss
        from .trainer import KANPINNTrainer
        
        __all__.extend([
            'HiPPOLayer',
            'KANLayer',
            'FluxObserver',
            'PositionPredictor',
            'KirchhoffLoss',
            'PINNLoss',
            'KANPINNTrainer'
        ])
    except ImportError as e:
        warnings.warn(f"Could not import all KAN-PINN components: {e}", ImportWarning)
