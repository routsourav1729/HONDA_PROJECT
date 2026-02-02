"""
Visualization utilities for hyperbolic embeddings.
Includes Poincare disk plots and UMAP with hyperbolic output metric.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import warnings

# Import pmath for conversions
from . import pmath


def plot_poincare_disk(
    embeddings,
    labels=None,
    prototypes=None,
    prototype_labels=None,
    curvature=0.1,
    ax=None,
    title="Poincaré Disk Embedding",
    cmap='Spectral',
    point_size=10,
    prototype_size=200,
    alpha=0.6,
    show_boundary=True,
    figsize=(10, 10)
):
    """
    Plot embeddings on the Poincaré disk.
    
    Parameters
    ----------
    embeddings : tensor or ndarray
        Points in Poincaré ball of shape (N, D). Only first 2 dims used.
    labels : tensor or ndarray, optional
        Class labels for coloring
    prototypes : tensor or ndarray, optional
        Prototype points to highlight
    prototype_labels : list, optional
        Labels for prototypes
    curvature : float
        Ball curvature (determines boundary radius)
    ax : matplotlib axis, optional
        Axis to plot on
    title : str
        Plot title
    cmap : str
        Colormap for points
    point_size : int
        Size of embedding points
    prototype_size : int
        Size of prototype markers
    alpha : float
        Point transparency
    show_boundary : bool
        Whether to show the boundary circle
    figsize : tuple
        Figure size
    
    Returns
    -------
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to numpy
    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    if torch.is_tensor(prototypes):
        prototypes = prototypes.detach().cpu().numpy()
    
    # Use first 2 dimensions
    x = embeddings[:, 0]
    y = embeddings[:, 1]
    
    # Plot boundary circle (radius = 1/sqrt(c))
    if show_boundary:
        radius = 1.0 / np.sqrt(curvature)
        boundary = Circle((0, 0), radius, fc='none', ec='black', linewidth=2, linestyle='--')
        ax.add_patch(boundary)
    
    # Plot embeddings
    if labels is not None:
        scatter = ax.scatter(x, y, c=labels, cmap=cmap, s=point_size, alpha=alpha)
        plt.colorbar(scatter, ax=ax, label='Class')
    else:
        ax.scatter(x, y, s=point_size, alpha=alpha)
    
    # Plot prototypes
    if prototypes is not None:
        proto_x = prototypes[:, 0]
        proto_y = prototypes[:, 1]
        ax.scatter(proto_x, proto_y, c='red', s=prototype_size, marker='*', 
                   edgecolors='black', linewidths=1, label='Prototypes', zorder=5)
        
        # Add prototype labels
        if prototype_labels is not None:
            for i, (px, py) in enumerate(zip(proto_x, proto_y)):
                ax.annotate(str(prototype_labels[i]), (px, py), fontsize=8, 
                           ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlim(-1.1 / np.sqrt(curvature), 1.1 / np.sqrt(curvature))
    ax.set_ylim(-1.1 / np.sqrt(curvature), 1.1 / np.sqrt(curvature))
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    
    if prototypes is not None:
        ax.legend()
    
    return ax


def visualize_hyperbolic_embeddings(
    embeddings,
    labels=None,
    prototypes=None,
    prototype_names=None,
    curvature=0.1,
    method='pca',
    title="Hyperbolic Embeddings",
    save_path=None,
    figsize=(12, 10)
):
    """
    Visualize high-dimensional hyperbolic embeddings.
    
    Parameters
    ----------
    embeddings : tensor or ndarray
        Points in Poincaré ball of shape (N, D)
    labels : tensor or ndarray, optional
        Class labels for coloring
    prototypes : tensor or ndarray, optional
        Prototype points
    prototype_names : list, optional
        Names for each prototype class
    curvature : float
        Ball curvature
    method : str
        Dimensionality reduction method: 'pca', 'tsne', or 'first2'
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    
    Returns
    -------
    fig : matplotlib figure
    """
    # Convert to numpy
    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    if torch.is_tensor(prototypes):
        prototypes = prototypes.detach().cpu().numpy()
    
    # Map back to tangent space for dimensionality reduction
    # logmap0 is the inverse of expmap0
    if method != 'first2':
        # Convert to tangent space for better linear dim reduction
        emb_tangent = _logmap0_numpy(embeddings, curvature)
        if prototypes is not None:
            proto_tangent = _logmap0_numpy(prototypes, curvature)
    else:
        emb_tangent = embeddings
        proto_tangent = prototypes
    
    # Dimensionality reduction
    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        emb_2d = reducer.fit_transform(emb_tangent)
        if prototypes is not None:
            proto_2d = reducer.transform(proto_tangent)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        if prototypes is not None:
            combined = np.vstack([emb_tangent, proto_tangent])
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined)-1))
            combined_2d = reducer.fit_transform(combined)
            emb_2d = combined_2d[:len(embeddings)]
            proto_2d = combined_2d[len(embeddings):]
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb_tangent)-1))
            emb_2d = reducer.fit_transform(emb_tangent)
            proto_2d = None
    elif method == 'first2':
        emb_2d = embeddings[:, :2]
        proto_2d = prototypes[:, :2] if prototypes is not None else None
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Project back to Poincaré ball for visualization
    emb_2d_hyp = _expmap0_numpy(emb_2d * 0.5, curvature)  # Scale down to fit in ball
    if proto_2d is not None:
        proto_2d_hyp = _expmap0_numpy(proto_2d * 0.5, curvature)
    else:
        proto_2d_hyp = None
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    plot_poincare_disk(
        emb_2d_hyp,
        labels=labels,
        prototypes=proto_2d_hyp,
        prototype_labels=prototype_names if prototype_names else (
            list(range(len(prototypes))) if prototypes is not None else None
        ),
        curvature=curvature,
        ax=ax,
        title=f"{title} ({method.upper()})"
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    return fig


def hyperbolic_umap_visualization(
    embeddings,
    labels=None,
    prototypes=None,
    prototype_names=None,
    curvature=0.1,
    n_neighbors=15,
    min_dist=0.1,
    title="Hyperbolic UMAP",
    save_path=None,
    figsize=(12, 10)
):
    """
    Visualize embeddings using UMAP with hyperboloid output metric.
    
    This embeds data into hyperbolic space using UMAP's built-in
    hyperboloid metric, then converts to Poincaré disk for visualization.
    
    Parameters
    ----------
    embeddings : tensor or ndarray
        Input embeddings (can be Euclidean or hyperbolic)
    labels : tensor or ndarray, optional
        Class labels
    prototypes : tensor or ndarray, optional
        Prototype points
    prototype_names : list, optional
        Names for prototypes
    curvature : float
        Ball curvature
    n_neighbors : int
        UMAP n_neighbors parameter
    min_dist : float
        UMAP min_dist parameter
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    
    Returns
    -------
    fig : matplotlib figure
    embedding_2d : ndarray
        2D Poincaré disk coordinates
    """
    try:
        import umap
    except ImportError:
        raise ImportError("Please install umap-learn: pip install umap-learn")
    
    # Convert to numpy
    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    if torch.is_tensor(prototypes):
        prototypes = prototypes.detach().cpu().numpy()
    
    # If embeddings are in Poincaré ball, map to tangent space first
    # (UMAP works better on Euclidean input)
    emb_for_umap = _logmap0_numpy(embeddings, curvature)
    
    # Apply UMAP with hyperboloid output metric
    print(f"Running UMAP with hyperboloid output metric on {len(embeddings)} points...")
    
    # Handle prototypes by including them in UMAP
    if prototypes is not None:
        proto_tangent = _logmap0_numpy(prototypes, curvature)
        combined = np.vstack([emb_for_umap, proto_tangent])
        
        reducer = umap.UMAP(
            output_metric='hyperboloid',
            n_neighbors=min(n_neighbors, len(combined) - 1),
            min_dist=min_dist,
            n_components=2,
            random_state=42
        )
        combined_2d = reducer.fit_transform(combined)
        hyp_2d = combined_2d[:len(embeddings)]
        proto_hyp_2d = combined_2d[len(embeddings):]
    else:
        reducer = umap.UMAP(
            output_metric='hyperboloid',
            n_neighbors=min(n_neighbors, len(emb_for_umap) - 1),
            min_dist=min_dist,
            n_components=2,
            random_state=42
        )
        hyp_2d = reducer.fit_transform(emb_for_umap)
        proto_hyp_2d = None
    
    # Convert hyperboloid coordinates to Poincaré disk
    # hyperboloid: (x, y) with z = sqrt(1 + x^2 + y^2)
    # Poincaré disk: (x/(1+z), y/(1+z))
    x = hyp_2d[:, 0]
    y = hyp_2d[:, 1]
    z = np.sqrt(1 + x**2 + y**2)
    disk_x = x / (1 + z)
    disk_y = y / (1 + z)
    disk_2d = np.stack([disk_x, disk_y], axis=1)
    
    if proto_hyp_2d is not None:
        px = proto_hyp_2d[:, 0]
        py = proto_hyp_2d[:, 1]
        pz = np.sqrt(1 + px**2 + py**2)
        proto_disk_x = px / (1 + pz)
        proto_disk_y = py / (1 + pz)
        proto_disk_2d = np.stack([proto_disk_x, proto_disk_y], axis=1)
    else:
        proto_disk_2d = None
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Custom plotting for Poincaré disk (curvature=1 for UMAP output)
    boundary = Circle((0, 0), 1, fc='none', ec='black', linewidth=2, linestyle='--')
    ax.add_patch(boundary)
    
    if labels is not None:
        scatter = ax.scatter(disk_x, disk_y, c=labels, cmap='Spectral', 
                            s=10, alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='Class')
    else:
        ax.scatter(disk_x, disk_y, s=10, alpha=0.6)
    
    if proto_disk_2d is not None:
        ax.scatter(proto_disk_2d[:, 0], proto_disk_2d[:, 1], 
                  c='red', s=200, marker='*', edgecolors='black', 
                  linewidths=1, label='Prototypes', zorder=5)
        
        if prototype_names is not None:
            for i, (px, py) in enumerate(proto_disk_2d):
                ax.annotate(str(prototype_names[i]), (px, py), fontsize=8,
                           ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    
    if proto_disk_2d is not None:
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    return fig, disk_2d


def _expmap0_numpy(u, c):
    """Exponential map from origin (numpy version)."""
    sqrt_c = np.sqrt(c)
    u_norm = np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-5)
    gamma_1 = np.tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return gamma_1


def _logmap0_numpy(y, c):
    """Logarithmic map to origin (numpy version)."""
    sqrt_c = np.sqrt(c)
    y_norm = np.maximum(np.linalg.norm(y, axis=-1, keepdims=True), 1e-5)
    return y / y_norm / sqrt_c * np.arctanh(np.clip(sqrt_c * y_norm, -1 + 1e-5, 1 - 1e-5))


def plot_distance_histogram(
    embeddings,
    prototypes,
    labels,
    curvature=0.1,
    save_path=None,
    figsize=(12, 6)
):
    """
    Plot histogram of distances to prototypes for known vs unknown.
    
    Parameters
    ----------
    embeddings : tensor
        Hyperbolic embeddings
    prototypes : tensor
        Class prototypes
    labels : tensor
        Ground truth labels (-1 for unknown)
    curvature : float
        Ball curvature
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    # Convert to tensor if needed
    if not torch.is_tensor(embeddings):
        embeddings = torch.tensor(embeddings)
    if not torch.is_tensor(prototypes):
        prototypes = torch.tensor(prototypes)
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)
    
    # Compute distances
    distances = pmath.dist_matrix(embeddings, prototypes, c=curvature)
    min_distances = distances.min(dim=-1).values.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    # Separate known and unknown
    known_mask = labels >= 0
    unknown_mask = labels < 0
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if known_mask.sum() > 0:
        ax.hist(min_distances[known_mask], bins=50, alpha=0.6, 
               label=f'Known (n={known_mask.sum()})', color='blue')
    if unknown_mask.sum() > 0:
        ax.hist(min_distances[unknown_mask], bins=50, alpha=0.6, 
               label=f'Unknown (n={unknown_mask.sum()})', color='red')
    
    ax.set_xlabel('Minimum Distance to Prototype')
    ax.set_ylabel('Count')
    ax.set_title('Distance Distribution: Known vs Unknown')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved histogram to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Quick test
    print("Testing visualization utilities...")
    
    # Generate random test data
    np.random.seed(42)
    N, D, K = 500, 256, 5
    
    # Create embeddings in Poincaré ball
    emb_tangent = np.random.randn(N, D) * 0.3
    embeddings = _expmap0_numpy(emb_tangent, c=0.1)
    
    # Create prototypes
    proto_tangent = np.random.randn(K, D) * 0.5
    prototypes = _expmap0_numpy(proto_tangent, c=0.1)
    
    # Random labels
    labels = np.random.randint(0, K, N)
    
    # Test PCA visualization
    print("\n1. Testing PCA visualization...")
    fig1 = visualize_hyperbolic_embeddings(
        embeddings, labels, prototypes,
        prototype_names=[f'Class {i}' for i in range(K)],
        method='pca',
        title='Test Embeddings'
    )
    plt.close(fig1)
    print("   PCA visualization OK")
    
    # Test UMAP visualization
    print("\n2. Testing UMAP visualization...")
    try:
        fig2, _ = hyperbolic_umap_visualization(
            embeddings[:100], labels[:100],  # Subset for speed
            title='Test UMAP'
        )
        plt.close(fig2)
        print("   UMAP visualization OK")
    except ImportError:
        print("   UMAP not installed, skipping")
    
    print("\nAll visualization tests PASSED!")
