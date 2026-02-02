"""
Hyperbolic geometry modules for open-world object detection.
Based on HypGCD and Hyp-OW implementations.
"""

from .pmath import (
    expmap0,
    logmap0,
    project,
    dist,
    dist_matrix,
    poincare_mean,
    mobius_add,
    RiemannianGradient,
)

from .nn import ToPoincare, FromPoincare, HyperbolicMLR

from .projector import HyperbolicProjector, HyperbolicContrastiveLoss

from .visualization import (
    visualize_hyperbolic_embeddings,
    plot_poincare_disk,
    hyperbolic_umap_visualization,
)

__all__ = [
    'expmap0',
    'logmap0', 
    'project',
    'dist',
    'dist_matrix',
    'poincare_mean',
    'mobius_add',
    'RiemannianGradient',
    'ToPoincare',
    'FromPoincare',
    'HyperbolicMLR',
    'HyperbolicProjector',
    'HyperbolicContrastiveLoss',
    'visualize_hyperbolic_embeddings',
    'plot_poincare_disk',
    'hyperbolic_umap_visualization',
]
