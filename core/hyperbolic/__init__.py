"""
Hyperbolic geometry modules for open-world object detection.
Uses Horospherical classification with ideal prototypes on boundary.
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
    busemann,
    busemann_batch,
)

from .nn import ToPoincare, FromPoincare, HyperbolicMLR

from .projector import (
    HyperbolicProjector,
    HorosphericalClassifier,
    HorosphericalLoss,
)

from .visualization import (
    visualize_hyperbolic_embeddings,
    plot_poincare_disk,
    hyperbolic_umap_visualization,
)

__all__ = [
    # Math operations
    'expmap0',
    'logmap0', 
    'project',
    'dist',
    'dist_matrix',
    'poincare_mean',
    'mobius_add',
    'RiemannianGradient',
    'busemann',
    'busemann_batch',
    # NN layers
    'ToPoincare',
    'FromPoincare',
    'HyperbolicMLR',
    # Projector and classifier
    'HyperbolicProjector',
    'HorosphericalClassifier',
    'HorosphericalLoss',
    # Visualization
    'visualize_hyperbolic_embeddings',
    'plot_poincare_disk',
    'hyperbolic_umap_visualization',
]
