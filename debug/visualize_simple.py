"""
Poincaré Ball Visualization for Horospherical Classifiers

This script visualizes hyperbolic embeddings with ideal prototypes (on boundary).
Uses UMAP with hyperboloid output metric for proper hyperbolic projection.

Key changes for horospherical approach:
- Prototypes are ON the boundary (||p|| = 1 for c=1)
- OOD detection via Busemann function (max horosphere score < threshold → unknown)
- Curvature c=1.0 (ball radius = 1.0)
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # No display
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from core import add_config
from core.util.model_ema import add_model_ema_configs
from core.pascal_voc import register_pascal_voc, inital_prompts
from core.hyp_customyoloworld import HypCustomYoloWorld, load_hyp_ckpt
from core.hyperbolic import busemann_batch

from mmengine.config import Config
from mmengine.runner import Runner
from detectron2.engine import default_argument_parser, default_setup
from detectron2.config import get_cfg


class Register:
    def __init__(self, dataset_root, split, cfg, dataset_key=None):
        self.dataset_root = dataset_root
        self.super_split = split.split('/')[0]
        self.cfg = cfg
        self.dataset_key = dataset_key if dataset_key is not None else self.super_split

        self.PREDEFINED_SPLITS_DATASET = {
            "my_train": split,
            "my_val": os.path.join(self.super_split, 'test')
        }

    def register_dataset(self):
        for name, split in self.PREDEFINED_SPLITS_DATASET.items():
            register_pascal_voc(name, self.dataset_root, self.dataset_key, split, self.cfg)


def setup(args):
    cfg = get_cfg()
    add_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    
    if args.task:
        task_yaml = os.path.join("configs", args.task.split('/')[0], args.task.split('/')[1] + ".yaml")
        if os.path.exists(task_yaml):
            cfg.merge_from_file(task_yaml)
            print(f"Merged task config: {task_yaml}")
    
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def get_anchor_centers(h, w, device):
    """Get anchor centers for YOLO-World (80x80 + 40x40 + 20x20 = 8400)"""
    strides = [8, 16, 32]
    centers = []
    for s in strides:
        gh, gw = h // s, w // s
        y = torch.arange(gh, device=device).float() * s + s / 2
        x = torch.arange(gw, device=device).float() * s + s / 2
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        centers.append(torch.stack([xx.flatten(), yy.flatten()], dim=-1))
    return torch.cat(centers, dim=0)


def _logmap0_numpy(x, c):
    """Logarithmic map from Poincaré ball to tangent space at origin (numpy)."""
    x = np.asarray(x, dtype=np.float64)
    sqrt_c = np.sqrt(c)
    x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
    x_norm = np.clip(x_norm, 1e-15, 1.0 / sqrt_c - 1e-5)
    atanh_val = np.arctanh(sqrt_c * x_norm)
    return x / (sqrt_c * x_norm + 1e-15) * atanh_val


def _expmap0_numpy(v, c):
    """Exponential map from tangent space at origin to Poincaré ball (numpy)."""
    v = np.asarray(v, dtype=np.float64)
    sqrt_c = np.sqrt(c)
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    v_norm = np.clip(v_norm, 1e-15, None)
    return np.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm + 1e-15) * v


def collect_embeddings(model, train_loader, num_batches=30, samples_per_class=30):
    """Collect hyperbolic embeddings for GT boxes with horosphere scores."""
    print(f"\n=== Collecting Embeddings ===")
    print(f"  Target: {samples_per_class} samples per class")
    print(f"  Max batches: {num_batches}")
    
    model.eval()
    all_embeddings = []
    all_labels = []
    all_max_scores = []  # Max horosphere score (higher = more ID)
    samples_count = {}
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(train_loader, desc="Collecting embeddings", total=num_batches)):
            if i >= num_batches:
                break
            
            # Check if we have enough samples
            if samples_count and all(v >= samples_per_class for v in samples_count.values()):
                print(f"\n  Early stop: collected enough samples at batch {i}")
                break
            
            try:
                data_batch = model.parent.data_preprocessor(batch)
                
                # Get FPN features
                x = model.parent.backbone.forward_image(data_batch['inputs'])
                
                # Apply neck
                if model.parent.with_neck:
                    if model.parent.mm_neck:
                        txt_feats = model.frozen_embeddings if model.frozen_embeddings is not None else model.embeddings
                        txt_feats = txt_feats.repeat(x[0].shape[0], 1, 1)
                        x = model.parent.neck(x, txt_feats)
                    else:
                        x = model.parent.neck(x)
                
                # Project to hyperbolic space
                hyp_embeddings = model.hyp_projector(x)  # (B, 8400, dim)
                
                # Get anchor grid
                h, w = data_batch['inputs'].shape[-2:]
                anchor_centers = get_anchor_centers(h, w, device=hyp_embeddings.device)
                
                # Process each image
                for b_idx, data_sample in enumerate(data_batch['data_samples']):
                    gt_bboxes = data_sample.gt_instances.bboxes
                    gt_labels = data_sample.gt_instances.labels
                    
                    if len(gt_labels) == 0:
                        continue
                    
                    # For each GT box, get nearest anchor's embedding
                    for box_idx in range(len(gt_bboxes)):
                        cls_id = int(gt_labels[box_idx].item())
                        
                        # Skip unknown/invalid class labels
                        num_known = model.num_classes
                        if cls_id >= num_known:
                            continue
                        
                        if cls_id not in samples_count:
                            samples_count[cls_id] = 0
                        
                        if samples_count[cls_id] >= samples_per_class:
                            continue
                        
                        # Box center
                        box = gt_bboxes[box_idx]
                        cx = (box[0] + box[2]) / 2
                        cy = (box[1] + box[3]) / 2
                        
                        # Find nearest anchor
                        dists_to_anchors = (anchor_centers[:, 0] - cx)**2 + (anchor_centers[:, 1] - cy)**2
                        nearest_idx = dists_to_anchors.argmin().item()
                        
                        # Get embedding
                        emb = hyp_embeddings[b_idx, nearest_idx]
                        all_embeddings.append(emb.cpu())
                        all_labels.append(cls_id)
                        
                        # Compute horosphere scores
                        scores = model.compute_horosphere_scores(emb.unsqueeze(0))  # (1, K)
                        max_score = scores.max().item()
                        all_max_scores.append(max_score)
                        
                        samples_count[cls_id] = samples_count.get(cls_id, 0) + 1
                        
            except Exception as e:
                print(f"  Error in batch {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n  Collected samples per class:")
    for cls_id, count in sorted(samples_count.items()):
        print(f"    Class {cls_id}: {count}")
    
    if len(all_embeddings) == 0:
        return None, None, None
    
    embeddings = torch.stack(all_embeddings)
    labels = torch.tensor(all_labels)
    max_scores = torch.tensor(all_max_scores)
    
    print(f"\n  Max horosphere score (higher = more ID):")
    print(f"    Min: {max_scores.min():.4f}")
    print(f"    Max: {max_scores.max():.4f}")
    print(f"    Mean: {max_scores.mean():.4f}")
    
    return embeddings, labels, max_scores


def project_to_2d_hyperbolic_umap(embeddings, prototypes, curvature=1.0, n_neighbors=15, min_dist=0.1):
    """
    Project high-dim Poincaré embeddings to 2D using PROPER hyperbolic UMAP.
    
    For horospherical approach:
    - Prototypes are on the boundary (||p|| = 1 for c=1)
    - We need special handling since logmap0 diverges for boundary points
    """
    try:
        import umap
    except ImportError:
        raise ImportError("Please install umap-learn: pip install umap-learn")
    
    # Convert to numpy
    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().numpy()
    if torch.is_tensor(prototypes):
        prototypes = prototypes.detach().cpu().numpy()
    
    # For boundary prototypes, we can't use logmap0 directly
    # Instead, scale prototypes slightly inside (0.95 * p)
    proto_norms = np.linalg.norm(prototypes, axis=-1, keepdims=True)
    print(f"  Original prototype norms: {proto_norms.flatten()}")
    
    # Scale prototypes to be slightly inside ball for UMAP
    prototypes_scaled = prototypes * (0.9 / (proto_norms + 1e-8))
    
    # Map to tangent space
    emb_tangent = _logmap0_numpy(embeddings, curvature)
    proto_tangent = _logmap0_numpy(prototypes_scaled, curvature)
    
    print(f"  Embeddings tangent shape: {emb_tangent.shape}")
    print(f"  Prototypes tangent shape: {proto_tangent.shape}")
    
    # Combine for joint embedding
    combined = np.vstack([emb_tangent, proto_tangent])
    
    # Handle NaN/Inf
    if np.any(~np.isfinite(combined)):
        print("  Warning: Non-finite values detected, replacing with zeros")
        combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Apply UMAP with hyperboloid output metric
    print(f"  Running UMAP with hyperboloid output metric on {len(combined)} points...")
    reducer = umap.UMAP(
        output_metric='hyperboloid',
        n_neighbors=min(n_neighbors, len(combined) - 1),
        min_dist=min_dist,
        n_components=2,
        random_state=42
    )
    combined_hyp = reducer.fit_transform(combined)
    
    # Convert hyperboloid coords to Poincaré disk
    x = combined_hyp[:, 0]
    y = combined_hyp[:, 1]
    z = np.sqrt(1 + x**2 + y**2)
    disk_x = x / (1 + z)
    disk_y = y / (1 + z)
    
    n_emb = len(emb_tangent)
    embeddings_2d = np.stack([disk_x[:n_emb], disk_y[:n_emb]], axis=1)
    prototypes_2d = np.stack([disk_x[n_emb:], disk_y[n_emb:]], axis=1)
    
    # Scale prototypes back to boundary in 2D visualization
    proto_2d_norms = np.linalg.norm(prototypes_2d, axis=-1, keepdims=True)
    prototypes_2d = prototypes_2d * (0.99 / (proto_2d_norms + 1e-8))  # Put on boundary
    
    return embeddings_2d, prototypes_2d


def project_to_2d_pca_tangent(embeddings, prototypes, curvature=1.0):
    """Project to 2D using PCA in tangent space."""
    from sklearn.decomposition import PCA
    
    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().numpy()
    if torch.is_tensor(prototypes):
        prototypes = prototypes.detach().cpu().numpy()
    
    # Scale prototypes inside for logmap
    proto_norms = np.linalg.norm(prototypes, axis=-1, keepdims=True)
    prototypes_scaled = prototypes * (0.9 / (proto_norms + 1e-8))
    
    # Map to tangent space
    emb_tangent = _logmap0_numpy(embeddings, curvature)
    proto_tangent = _logmap0_numpy(prototypes_scaled, curvature)
    
    # Handle NaN/Inf
    combined_tangent = np.vstack([emb_tangent, proto_tangent])
    combined_tangent = np.nan_to_num(combined_tangent, nan=0.0, posinf=0.0, neginf=0.0)
    
    # PCA
    pca = PCA(n_components=2)
    combined_2d_tangent = pca.fit_transform(combined_tangent)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Scale and map back
    combined_2d_tangent = combined_2d_tangent * 0.3
    combined_2d_poincare = _expmap0_numpy(combined_2d_tangent, curvature)
    
    n_emb = len(embeddings)
    embeddings_2d = combined_2d_poincare[:n_emb]
    prototypes_2d = combined_2d_poincare[n_emb:]
    
    # Put prototypes on boundary
    proto_2d_norms = np.linalg.norm(prototypes_2d, axis=-1, keepdims=True)
    prototypes_2d = prototypes_2d * (0.99 / (proto_2d_norms + 1e-8))
    
    return embeddings_2d, prototypes_2d


def plot_poincare_disk(embeddings_2d, labels, prototypes_2d, class_names, max_scores, save_path, curvature=1.0):
    """
    Create Poincaré disk visualization with horospherical prototypes.
    
    Key difference from distance-based:
    - Prototypes are ON the boundary (ideal points)
    - Color scale shows horosphere score (higher = more ID)
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    n_protos = prototypes_2d.shape[0]
    n_classes = len(class_names)
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_protos, n_classes) + 1))
    
    labels_np = labels.numpy() if hasattr(labels, 'numpy') else labels
    max_scores_np = max_scores.numpy() if hasattr(max_scores, 'numpy') else max_scores
    
    # Filter invalid labels
    valid_mask = labels_np < n_protos
    if not valid_mask.all():
        print(f"  Warning: Filtering {(~valid_mask).sum()} samples with label >= {n_protos}")
        embeddings_2d = embeddings_2d[valid_mask]
        labels_np = labels_np[valid_mask]
        max_scores_np = max_scores_np[valid_mask]
    
    # --- Left plot: Poincaré disk ---
    ax = axes[0]
    
    # Draw unit circle (boundary)
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=3, label='Ball boundary (ideal points)')
    ax.fill(np.cos(theta), np.sin(theta), alpha=0.05, color='gray')
    
    # Plot embeddings by class
    unique_labels = np.unique(labels_np)
    for i in unique_labels:
        if i >= n_classes:
            class_name = f"Unknown_{i}"
        else:
            class_name = class_names[i]
        mask = labels_np == i
        if mask.sum() > 0:
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                      c=[colors[i]], label=f'{class_name} ({mask.sum()})', 
                      alpha=0.6, s=30)
    
    # Plot prototypes as stars ON the boundary
    for i in range(n_protos):
        if i < n_classes:
            class_name = class_names[i]
        else:
            class_name = f"Proto_{i}"
        ax.scatter(prototypes_2d[i, 0], prototypes_2d[i, 1],
                  c=[colors[i]], s=500, marker='*', edgecolors='black', 
                  linewidths=2, zorder=5)
        # Draw label outside the ball
        direction = prototypes_2d[i] / (np.linalg.norm(prototypes_2d[i]) + 1e-8)
        label_pos = direction * 1.15
        ax.annotate(f'{class_name}', label_pos, 
                   fontsize=9, ha='center', va='center', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    ax.set_title('Poincaré Disk: Horospherical Classifier\n'
                 '(★ = ideal prototypes on boundary, points inside ball)')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    
    # --- Right plot: Horosphere score distribution ---
    ax = axes[1]
    
    # Plot score histogram for each class
    for i in unique_labels:
        if i >= n_classes:
            class_name = f"Unknown_{i}"
        else:
            class_name = class_names[i]
        mask = labels_np == i
        if mask.sum() > 0:
            class_scores = max_scores_np[mask]
            ax.hist(class_scores, bins=20, alpha=0.5, label=f'{class_name}', color=colors[i])
    
    # Draw threshold line (if we had one)
    median_score = np.median(max_scores_np)
    ax.axvline(x=median_score - 1.0, color='red', linestyle='--', 
               label=f'Example OOD threshold (~{median_score-1.0:.1f})')
    
    ax.set_xlabel('Max Horosphere Score (ξ = -B + a)')
    ax.set_ylabel('Count')
    ax.set_title('Horosphere Score Distribution by Class\n'
                 '(Higher score = closer to some prototype → more ID)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved visualization: {save_path}")


if __name__ == "__main__":
    parser0 = default_argument_parser()
    parser0.add_argument("--task", default="IDD/t1")
    parser0.add_argument("--ckpt", default="IDD/t1/hyperbolic/model_0.pth")
    parser0.add_argument("--hyp_c", type=float, default=1.0)  # Changed to c=1.0
    parser0.add_argument("--hyp_dim", type=int, default=256)
    parser0.add_argument("--clip_r", type=float, default=0.95)  # Changed for c=1.0
    parser0.add_argument("--output_dir", default="visualizations")
    parser0.add_argument("--num_batches", type=int, default=50)
    parser0.add_argument("--samples_per_class", type=int, default=50)
    parser0.add_argument("--projection", type=str, default="hyperbolic_umap",
                         choices=["hyperbolic_umap", "pca_tangent"])
    parser0.add_argument("--n_neighbors", type=int, default=15)
    parser0.add_argument("--min_dist", type=float, default=0.1)
    
    args = parser0.parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)

    task_name = args.task.split('/')[0]
    split_name = args.task.split('/')[1]
    
    # Handle IDD_HYP -> IDD for dataset registration (dataset is registered as IDD)
    base_dataset = task_name.replace('_HYP', '')
    dataset_key = base_dataset
    
    # Use base dataset path for data registration
    data_split = f"{base_dataset}/{split_name}"
    data_register = Register('./datasets/', data_split, cfg, dataset_key)
    data_register.register_dataset()

    class_names = list(inital_prompts()[dataset_key])

    # Model config
    config_file = os.path.join("./configs", task_name, split_name + ".py")
    cfgY = Config.fromfile(config_file)
    cfgY.work_dir = "."
    
    unknown_index = cfg.TEST.PREV_INTRODUCED_CLS + cfg.TEST.CUR_INTRODUCED_CLS
    class_names = class_names[:unknown_index]
    classnames = [class_names]

    print(f"\n=== Configuration ===")
    print(f"  Task: {args.task}")
    print(f"  Classes: {class_names}")
    print(f"  Curvature: {args.hyp_c} (ball radius = 1.0)")

    # Initialize YOLO-World
    runner = Runner.from_cfg(cfgY)
    runner.call_hook("before_run")
    runner.load_or_resume()
    runner.model.reparameterize(classnames)
    runner.model.eval()

    # Use train loader for visualization
    train_loader = Runner.build_dataloader(cfgY.trlder)

    # Build hyperbolic model
    model = HypCustomYoloWorld(
        runner.model,
        unknown_index,
        hyp_c=args.hyp_c,
        hyp_dim=args.hyp_dim,
        clip_r=args.clip_r
    )
    
    # Load checkpoint
    print(f"\n=== Loading Checkpoint: {args.ckpt} ===")
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    print(f"  Checkpoint keys: {list(state_dict.keys())[:20]}...")
    
    # Check for horospherical classifier weights
    hyp_keys = [k for k in state_dict.keys() if 'classifier' in k.lower() or 'prototype' in k.lower()]
    print(f"  Classifier-related keys: {hyp_keys}")
    
    if 'hyp_projector.classifier.prototype_direction' in state_dict:
        proto_dir = state_dict['hyp_projector.classifier.prototype_direction']
        proto_norms = proto_dir.norm(dim=-1)
        print(f"  prototype_direction shape: {proto_dir.shape}")
        print(f"  prototype_direction norms: {proto_norms}")
    
    if 'hyp_projector.classifier.prototype_bias' in state_dict:
        proto_bias = state_dict['hyp_projector.classifier.prototype_bias']
        print(f"  prototype_bias shape: {proto_bias.shape}")
        print(f"  prototype_bias values: {proto_bias}")
    
    del checkpoint, state_dict
    
    with torch.no_grad():
        model = load_hyp_ckpt(
            model, args.ckpt,
            cfg.TEST.PREV_INTRODUCED_CLS,
            cfg.TEST.CUR_INTRODUCED_CLS,
            eval=True
        )
        model = model.cuda()
        model.add_generic_text(class_names, generic_prompt='object', alpha=0.4)

    model.eval()
    
    # Model verification
    print(f"\n=== Model Info (After Loading) ===")
    print(f"  num_classes: {model.num_classes}")
    print(f"  frozen_directions: {model.frozen_directions.shape if model.frozen_directions is not None else None}")
    print(f"  frozen_biases: {model.frozen_biases.shape if model.frozen_biases is not None else None}")
    
    # Get prototypes (on boundary)
    prototypes = model.prototypes  # These are ideal prototypes on boundary
    print(f"\n  Prototypes shape: {prototypes.shape}")
    print(f"  Prototype norms (should be ~1.0): {prototypes.norm(dim=-1)}")
    
    # Collect embeddings
    embeddings, labels, max_scores = collect_embeddings(
        model, train_loader, 
        num_batches=args.num_batches,
        samples_per_class=args.samples_per_class
    )
    
    if embeddings is None:
        print("ERROR: No embeddings collected!")
        exit(1)
    
    # Project to 2D
    print(f"\n=== Projecting to 2D using {args.projection} ===")
    
    if args.projection == "hyperbolic_umap":
        embeddings_2d, prototypes_2d = project_to_2d_hyperbolic_umap(
            embeddings, prototypes.detach().cpu(), 
            curvature=args.hyp_c,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist
        )
    elif args.projection == "pca_tangent":
        embeddings_2d, prototypes_2d = project_to_2d_pca_tangent(
            embeddings, prototypes.detach().cpu(),
            curvature=args.hyp_c
        )
    else:
        raise ValueError(f"Unknown projection method: {args.projection}")
    
    # Plot
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_name = Path(args.ckpt).stem
    save_path = os.path.join(args.output_dir, f"horosphere_{args.projection}_{ckpt_name}.png")
    
    plot_poincare_disk(embeddings_2d, labels, prototypes_2d, class_names, max_scores, 
                       save_path, curvature=args.hyp_c)
    
    # Save raw data
    np.savez(os.path.join(args.output_dir, f"data_{ckpt_name}.npz"),
             embeddings=embeddings.numpy(),
             embeddings_2d=embeddings_2d,
             labels=labels.numpy(),
             prototypes=prototypes.detach().cpu().numpy(),
             prototypes_2d=prototypes_2d,
             max_scores=max_scores.numpy(),
             class_names=class_names,
             projection_method=args.projection)
    print(f"Saved data: {args.output_dir}/data_{ckpt_name}.npz")
    
    # Print summary
    print("\n" + "="*60)
    print("HOROSPHERICAL SPACE ANALYSIS SUMMARY")
    print("="*60)
    
    prototypes_np = prototypes.detach().cpu().numpy()
    proto_norms = np.linalg.norm(prototypes_np, axis=1)
    print(f"\nPrototype norms (should be ~1.0 for ideal prototypes on boundary):")
    for i, (name, norm) in enumerate(zip(class_names, proto_norms)):
        print(f"  {name}: {norm:.6f}")
    
    # Get biases
    biases = model.hyp_projector.classifier.prototype_bias.detach().cpu().numpy()
    print(f"\nPrototype biases (learnable offsets):")
    for i, (name, bias) in enumerate(zip(class_names, biases)):
        print(f"  {name}: {bias:.4f}")
    
    scores_np = max_scores.numpy()
    print(f"\nMax horosphere score distribution:")
    print(f"  Min: {scores_np.min():.4f}")
    print(f"  Mean: {scores_np.mean():.4f}")
    print(f"  Max: {scores_np.max():.4f}")
    print(f"  Std: {scores_np.std():.4f}")
    
    print("="*60)
