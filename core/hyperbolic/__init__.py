"""
Hyperspherical geometry modules for open-world object detection.
Uses vMF (von Mises-Fisher) classification on the unit hypersphere.

Legacy Poincare ball utilities are still importable for backward compat
but are NOT used in the active pipeline.
"""

# Legacy Poincare math -- kept for backward compat, not used in pipeline
from .pmath import (
    expmap0,
    logmap0,
    project,
    dist,
    dist_matrix,
    poincare_mean,
    mobius_add,
    RiemannianGradient,
    busemann,
    busemann_batch,
)

# Legacy NN layers -- kept for backward compat
from .nn import ToPoincare, FromPoincare, HyperbolicMLR

# Active pipeline (vMF spherical)
from .projector import (
    HyperbolicProjector,
    vMFClassifier,
    vMFLoss,
    BiLipschitzProjector,
    compute_class_weights,
    stable_log_vmf_normalizer,
    # Backward-compat aliases
    GeodesicPrototypeClassifier,
    GeodesicPrototypeLoss,
    HorosphericalClassifier,
    HorosphericalLoss,
    HorosphericalLossV2,
    SphericalProjector,
)

# Visualization (Poincare-specific, import on demand):
#   from core.hyperbolic.visualization import plot_poincare_disk, ...

__all__ = [
    # Legacy math (not used in pipeline)
    'expmap0', 'logmap0', 'project', 'dist', 'dist_matrix',
    'poincare_mean', 'mobius_add', 'RiemannianGradient',
    'busemann', 'busemann_batch',
    # Legacy NN
    'ToPoincare', 'FromPoincare', 'HyperbolicMLR',
    # Active pipeline
    'HyperbolicProjector', 'SphericalProjector',
    'vMFClassifier', 'vMFLoss',
    'BiLipschitzProjector',
    'compute_class_weights',
    'stable_log_vmf_normalizer',
    # Backward compat aliases
    'GeodesicPrototypeClassifier', 'GeodesicPrototypeLoss',
    'HorosphericalClassifier', 'HorosphericalLoss', 'HorosphericalLossV2',
]
