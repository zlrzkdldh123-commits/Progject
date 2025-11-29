"""
Utility modules for H-MTL framework

This package contains:
- metrics: Loss functions (EMD, Hierarchical) and evaluation metrics (ACR)
- dataset: Data loading and preprocessing (optional)
- visualization: Plotting and visualization utilities (optional)
"""

from .metrics import (
    emd_loss,
    hierarchical_loss,
    calculate_acr,
    calculate_metrics
)

__all__ = [
    # Metrics
    'emd_loss',
    'hierarchical_loss',
    'calculate_acr',
    'calculate_metrics',
]

__version__ = "1.0.0"
__description__ = "H-MTL utility functions"
