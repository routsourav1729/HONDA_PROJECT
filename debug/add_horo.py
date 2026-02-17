"""
POST-PROCESSING: Create horosphere visualization from saved data
NO GPU REQUIRED

This script loads the .npz file created by debug/visualize_simple.py
which contains:
  - embeddings_2d (already projected!)
  - prototypes_2d (already projected!)
  - labels
  - max_scores
  - class_names
  
Then just overlays horospheres using the bias values from your analysis output.

Usage:
    python add_horospheres_to_saved.py \
        --data_npz /path/to/data_model_30.npz \
        --biases 0.2139 0.2893 0.1604 0.1375 0.1111 -0.0203 -0.1021 -0.1281 -0.0381 \
        --output horospheres_final.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def sigmoid(a):
    """σ(a) = 1 / (1 + exp(-a))"""
    return 1.0 / (1.0 + np.exp(-a))


def compute_horosphere_params(prototype_2d, bias):
    """
    Horosphere in 2D Poincaré disk (Paper Proposition 1):
    - Center: (1 - σ(a)) * p
    - Radius: σ(a)
    """
    sigma_a = sigmoid(bias)
    center = (1.0 - sigma_a) * prototype_2d
    radius = sigma_a
    return center, radius


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_npz', type=str, required=True,
                       help='Path to data_model_XX.npz from visualize_simple.py')
    parser.add_argument('--biases', type=float, nargs='+', required=True,
                       help='Bias values for each class (from your analysis output)')
    parser.add_argument('--output', type=str, default='horospheres_overlay.png',
                       help='Output image path')
    parser.add_argument('--known_classes', type=str, nargs='+',
                       default=['car', 'motorcycle', 'rider', 'person', 'autorickshaw',
                               'traffic sign', 'traffic light', 'pole', 'bicycle'],
                       help='Known class names')
    args = parser.parse_args()
    
    # Load saved data
    print(f"\n=== Loading: {args.data_npz} ===")
    data = np.load(args.data_npz, allow_pickle=True)
    
    print(f"Available keys: {list(data.keys())}")
    
    # Extract data
    embeddings_2d = data['embeddings_2d']  # Already projected!
    prototypes_2d = data['prototypes_2d']  # Already projected!
    labels = data['labels']
    max_scores = data['max_scores'] if 'max_scores' in data else None
    class_names = data['class_names'].tolist() if 'class_names' in data else args.known_classes
    
    print(f"  Embeddings 2D: {embeddings_2d.shape}")
    print(f"  Prototypes 2D: {prototypes_2d.shape}")
    print(f"  Labels: {labels.shape}, unique: {np.unique(labels)}")
    print(f"  Class names: {class_names}")
    
    biases = np.array(args.biases)
    if len(biases) != len(prototypes_2d):
        print(f"ERROR: Got {len(biases)} biases but {len(prototypes_2d)} prototypes")
        return
    
    print(f"  Biases: {biases}")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # === LEFT: Poincaré Disk with Horospheres ===
    ax = axes[0]
    
    # Draw boundary
    boundary = Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=3, linestyle='--', label='Ball boundary')
    ax.add_patch(boundary)
    
    # Colors
    colors = plt.cm.tab10(np.arange(len(class_names)))
    
    # Plot embeddings
    known_set = set(args.known_classes)
    for i, cls_name in enumerate(class_names):
        mask = labels == i
        if mask.sum() == 0:
            continue
        
        pts = embeddings_2d[mask]
        
        # Sample if too many
        if len(pts) > 500:
            idx = np.random.choice(len(pts), 500, replace=False)
            pts = pts[idx]
        
        is_known = cls_name in known_set
        alpha = 0.6 if is_known else 0.25
        size = 15 if is_known else 8
        
        ax.scatter(pts[:, 0], pts[:, 1], c=[colors[i]], s=size, alpha=alpha,
                  label=f'{cls_name} ({mask.sum()})', edgecolors='none')
    
    # Draw horospheres for known classes
    for i, cls_name in enumerate(args.known_classes):
        if cls_name not in class_names:
            continue
        
        cls_idx = class_names.index(cls_name)
        center, radius = compute_horosphere_params(prototypes_2d[cls_idx], biases[cls_idx])
        
        # Horosphere circle
        circle = Circle(center, radius, fill=False, edgecolor=colors[cls_idx],
                       linewidth=2.5, linestyle='-', alpha=0.9, zorder=3)
        ax.add_patch(circle)
        
        # Prototype star on boundary
        ax.plot(prototypes_2d[cls_idx, 0], prototypes_2d[cls_idx, 1], '*',
               color=colors[cls_idx], markersize=28, markeredgecolor='black',
               markeredgewidth=1.8, zorder=5)
        
        # Label
        label_pos = prototypes_2d[cls_idx] * 1.18
        ax.text(label_pos[0], label_pos[1], cls_name,
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                        edgecolor=colors[cls_idx], linewidth=2.5, alpha=0.95),
               zorder=6)
    
    # Origin
    ax.plot(0, 0, 'k+', markersize=18, markeredgewidth=3, label='Origin', zorder=4)
    
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect('equal')
    ax.set_title('Poincaré Disk: Horospherical Classifier\n' +
                 '(★ = ideal prototypes on boundary, points inside ball)',
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    
    # Legend outside
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9,
             framealpha=0.95, ncol=1, markerscale=1.5)
    
    # === RIGHT: Horosphere Score Distribution ===
    ax = axes[1]
    
    if max_scores is not None:
        for i, cls_name in enumerate(class_names):
            mask = labels == i
            if mask.sum() < 5:
                continue
            
            scores = max_scores[mask]
            ax.hist(scores, bins=25, alpha=0.6, color=colors[i],
                   label=f'{cls_name}', edgecolor='black', linewidth=0.5)
        
        # Example OOD threshold
        median = np.median(max_scores)
        ax.axvline(median - 0.6, color='red', linestyle='--', linewidth=3,
                  label='Example OOD threshold', zorder=10)
        
        ax.set_xlabel('Max Horosphere Score (ξ = -B + a)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Horosphere Score Distribution by Class\n' +
                    '(Higher score = closer to some prototype → more ID)',
                    fontsize=13, fontweight='bold', pad=20)
        ax.legend(fontsize=9, framealpha=0.95, ncol=2)
        ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {args.output}")
    
    # Print horosphere parameters
    print(f"\n{'='*70}")
    print("HOROSPHERE PARAMETERS")
    print(f"{'='*70}")
    print(f"{'Class':<20s} {'Bias':>8s} {'σ(a)':>8s} {'Center':>18s} {'Radius':>8s}")
    print(f"{'-'*70}")
    for i, cls_name in enumerate(args.known_classes):
        if cls_name not in class_names:
            continue
        cls_idx = class_names.index(cls_name)
        center, radius = compute_horosphere_params(prototypes_2d[cls_idx], biases[cls_idx])
        print(f"{cls_name:<20s} {biases[cls_idx]:>8.4f} {radius:>8.4f} " +
              f"({center[0]:>6.3f}, {center[1]:>6.3f}) {radius:>8.4f}")
    
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()