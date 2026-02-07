"""
Simple Poincaré Ball Visualization - Uses same model loading as test_hyp.py

This script loads the model exactly like test_hyp.py, then:
1. Extracts embeddings for GT boxes from a small subset of images
2. Plots embeddings + prototypes on a 2D Poincaré disk using PROPER hyperbolic UMAP

NOTE: Previous version had a critical bug - it used raw PCA on Poincaré embeddings,
which is mathematically WRONG because hyperbolic space is curved. This version:
1. Maps Poincaré embeddings to tangent space (Euclidean) via logmap0
2. Uses UMAP with output_metric='hyperboloid' for proper hyperbolic embedding
3. Converts hyperboloid (x, y, z) to Poincaré disk: (x/(1+z), y/(1+z))
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
from core.hyperbolic import dist_matrix
# Import hyperbolic UMAP visualization  
from core.hyperbolic.visualization import hyperbolic_umap_visualization, _logmap0_numpy, _expmap0_numpy

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


def collect_embeddings(model, train_loader, num_batches=30, samples_per_class=30):
    """Collect hyperbolic embeddings for GT boxes."""
    print(f"\n=== Collecting Embeddings ===")
    print(f"  Target: {samples_per_class} samples per class")
    print(f"  Max batches: {num_batches}")
    
    model.eval()
    all_embeddings = []
    all_labels = []
    all_distances = []  # Distance to nearest prototype
    samples_count = {}
    
    with torch.no_grad():
        # Use total=num_batches so tqdm shows accurate progress
        for i, batch in enumerate(tqdm(train_loader, desc="Collecting embeddings", total=num_batches)):
            if i >= num_batches:
                break
            
            # Check if we have enough samples
            if samples_count and all(v >= samples_per_class for v in samples_count.values()):
                print(f"\n  Early stop: collected enough samples at batch {i}")
                break
            
            try:
                data_batch = model.parent.data_preprocessor(batch)
                
                # Get FPN features (use forward_image, not direct backbone call)
                x = model.parent.backbone.forward_image(data_batch['inputs'])
                
                # Apply neck (may need text features for mm_neck)
                if model.parent.with_neck:
                    if model.parent.mm_neck:
                        # For multimodal neck, need text features
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
                        num_known = model.prototypes.shape[0]
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
                        
                        # Compute distance to nearest prototype
                        dist_to_protos = dist_matrix(emb.unsqueeze(0), model.prototypes, c=model.hyp_c)
                        min_dist = dist_to_protos.min().item()
                        all_distances.append(min_dist)
                        
                        samples_count[cls_id] = samples_count.get(cls_id, 0) + 1
                        
            except Exception as e:
                print(f"  Error in batch {i}: {e}")
                continue
    
    print(f"\n  Collected samples per class:")
    for cls_id, count in sorted(samples_count.items()):
        print(f"    Class {cls_id}: {count}")
    
    if len(all_embeddings) == 0:
        return None, None, None
    
    embeddings = torch.stack(all_embeddings)
    labels = torch.tensor(all_labels)
    distances = torch.tensor(all_distances)
    
    print(f"\n  Distance to nearest prototype:")
    print(f"    Min: {distances.min():.4f}")
    print(f"    Max: {distances.max():.4f}")
    print(f"    Mean: {distances.mean():.4f}")
    
    return embeddings, labels, distances


def project_to_2d_hyperbolic_umap(embeddings, prototypes, curvature=0.1, n_neighbors=15, min_dist=0.1):
    """
    Project high-dim Poincaré embeddings to 2D using PROPER hyperbolic UMAP.
    
    This is the CORRECT way to visualize hyperbolic embeddings:
    1. Map Poincaré embeddings to tangent space (Euclidean) via logmap0
    2. Run UMAP with output_metric='hyperboloid' 
    3. Convert hyperboloid (x, y) with z=sqrt(1+x²+y²) to Poincaré disk: (x/(1+z), y/(1+z))
    
    DO NOT use raw PCA on Poincaré embeddings - hyperbolic space is curved!
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
    
    # Step 1: Map Poincaré ball embeddings to tangent space (Euclidean) via logmap0
    # This is critical - UMAP needs Euclidean input
    emb_tangent = _logmap0_numpy(embeddings, curvature)
    proto_tangent = _logmap0_numpy(prototypes, curvature)
    
    print(f"  Embeddings shape: {emb_tangent.shape}")
    print(f"  Prototypes shape: {proto_tangent.shape}")
    
    # Combine for joint embedding
    combined = np.vstack([emb_tangent, proto_tangent])
    
    # Step 2: Apply UMAP with hyperboloid output metric
    print(f"  Running UMAP with hyperboloid output metric on {len(combined)} points...")
    reducer = umap.UMAP(
        output_metric='hyperboloid',
        n_neighbors=min(n_neighbors, len(combined) - 1),
        min_dist=min_dist,
        n_components=2,  # 2D for hyperboloid = 3D (x, y, z with z=sqrt(1+x²+y²))
        random_state=42
    )
    combined_hyp = reducer.fit_transform(combined)
    
    # Step 3: Convert hyperboloid coords (x, y) to Poincaré disk
    # Hyperboloid: (x, y) where z = sqrt(1 + x² + y²)
    # Poincaré disk: (x/(1+z), y/(1+z))
    x = combined_hyp[:, 0]
    y = combined_hyp[:, 1]
    z = np.sqrt(1 + x**2 + y**2)
    disk_x = x / (1 + z)
    disk_y = y / (1 + z)
    
    n_emb = len(emb_tangent)
    embeddings_2d = np.stack([disk_x[:n_emb], disk_y[:n_emb]], axis=1)
    prototypes_2d = np.stack([disk_x[n_emb:], disk_y[n_emb:]], axis=1)
    
    # Verify we're inside the Poincaré disk (norm < 1)
    emb_norms = np.linalg.norm(embeddings_2d, axis=1)
    proto_norms = np.linalg.norm(prototypes_2d, axis=1)
    print(f"  Embedding norms: min={emb_norms.min():.4f}, max={emb_norms.max():.4f}")
    print(f"  Prototype norms: min={proto_norms.min():.4f}, max={proto_norms.max():.4f}")
    
    return embeddings_2d, prototypes_2d


def project_to_2d_pca_tangent(embeddings, prototypes, curvature=0.1):
    """
    Alternative: Project to 2D using PCA in TANGENT SPACE (not on Poincaré directly!).
    
    This is a simpler alternative to hyperbolic UMAP that is still mathematically sound:
    1. Map Poincaré embeddings to tangent space via logmap0
    2. Apply PCA in tangent space (which is Euclidean)
    3. Map back to Poincaré ball via expmap0
    """
    from sklearn.decomposition import PCA
    
    # Convert to numpy
    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().numpy()
    if torch.is_tensor(prototypes):
        prototypes = prototypes.detach().cpu().numpy()
    
    # Step 1: Map to tangent space
    emb_tangent = _logmap0_numpy(embeddings, curvature)
    proto_tangent = _logmap0_numpy(prototypes, curvature)
    
    # Step 2: PCA in tangent space
    combined_tangent = np.vstack([emb_tangent, proto_tangent])
    pca = PCA(n_components=2)
    combined_2d_tangent = pca.fit_transform(combined_tangent)
    
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Step 3: Map back to Poincaré ball (scale down to fit in ball)
    combined_2d_tangent = combined_2d_tangent * 0.5  # Scale to avoid edge of ball
    combined_2d_poincare = _expmap0_numpy(combined_2d_tangent, curvature)
    
    n_emb = len(embeddings)
    embeddings_2d = combined_2d_poincare[:n_emb]
    prototypes_2d = combined_2d_poincare[n_emb:]
    
    return embeddings_2d, prototypes_2d


def plot_poincare_disk(embeddings_2d, labels, prototypes_2d, class_names, distances, save_path, curvature=0.1):
    """Create Poincaré disk visualization with proper hyperbolic structure."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Color map - use number of prototypes, not class_names
    n_protos = prototypes_2d.shape[0]
    n_classes = len(class_names)
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_protos, n_classes) + 1))
    
    # Filter out any labels that are >= n_protos (unknown/out-of-range)
    labels_np = labels.numpy() if hasattr(labels, 'numpy') else labels
    valid_mask = labels_np < n_protos
    if not valid_mask.all():
        print(f"  Warning: Filtering {(~valid_mask).sum()} samples with label >= {n_protos}")
        embeddings_2d = embeddings_2d[valid_mask]
        labels_np = labels_np[valid_mask]
        distances = distances[torch.tensor(valid_mask)] if torch.is_tensor(distances) else distances[valid_mask]
    
    # --- Left plot: Poincaré disk ---
    ax = axes[0]
    
    # Draw unit circle (Poincaré disk boundary for c=1, or radius=1/sqrt(c))
    # For UMAP hyperboloid output, the disk has radius ~1
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, label='Disk boundary')
    ax.fill(np.cos(theta), np.sin(theta), alpha=0.05, color='gray')
    
    # Plot embeddings by class (only for classes that exist in data)
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
    
    # Plot prototypes as stars
    for i in range(n_protos):
        if i < n_classes:
            class_name = class_names[i]
        else:
            class_name = f"Proto_{i}"
        ax.scatter(prototypes_2d[i, 0], prototypes_2d[i, 1],
                  c=[colors[i]], s=400, marker='*', edgecolors='black', linewidths=2, zorder=5)
        ax.annotate(f'{class_name}', (prototypes_2d[i, 0], prototypes_2d[i, 1] + 0.05), 
                   fontsize=8, ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    ax.set_title('Poincaré Disk: Hyperbolic UMAP Projection\n(Points closer to boundary = higher norm)')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    
    # --- Right plot: Distance distribution ---
    ax = axes[1]
    
    distances_np = distances.numpy() if hasattr(distances, 'numpy') else distances
    
    # Plot distance histogram for each class present in data
    for i in unique_labels:
        if i >= n_classes:
            class_name = f"Unknown_{i}"
        else:
            class_name = class_names[i]
        mask = labels_np == i
        if mask.sum() > 0:
            class_dists = distances_np[mask]
            ax.hist(class_dists, bins=20, alpha=0.5, label=f'{class_name}', color=colors[i])
    
    ax.axvline(x=2.5, color='red', linestyle='--', label='OOD thresh (2.5)')
    ax.set_xlabel('Hyperbolic Distance to Nearest Prototype')
    ax.set_ylabel('Count')
    ax.set_title('Distance Distribution by Class\n(Higher distance = more OOD-like)')
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
    parser0.add_argument("--hyp_c", type=float, default=0.1)
    parser0.add_argument("--hyp_dim", type=int, default=256)
    parser0.add_argument("--clip_r", type=float, default=2.3)
    parser0.add_argument("--temperature", type=float, default=0.1)
    parser0.add_argument("--output_dir", default="visualizations")
    parser0.add_argument("--num_batches", type=int, default=50)
    parser0.add_argument("--samples_per_class", type=int, default=50)
    parser0.add_argument("--projection", type=str, default="hyperbolic_umap",
                         choices=["hyperbolic_umap", "pca_tangent"],
                         help="Projection method: 'hyperbolic_umap' (UMAP with hyperboloid output) "
                              "or 'pca_tangent' (PCA in tangent space then map back)")
    parser0.add_argument("--n_neighbors", type=int, default=15,
                         help="UMAP n_neighbors parameter")
    parser0.add_argument("--min_dist", type=float, default=0.1,
                         help="UMAP min_dist parameter")
    
    args = parser0.parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)

    task_name = args.task.split('/')[0]
    split_name = args.task.split('/')[1]
    dataset_key = task_name
    
    data_register = Register('./datasets/', args.task, cfg, dataset_key)
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

    # Initialize YOLO-World
    runner = Runner.from_cfg(cfgY)
    runner.call_hook("before_run")
    runner.load_or_resume()
    runner.model.reparameterize(classnames)
    runner.model.eval()

    # Use train loader for visualization (has GT boxes)
    train_loader = Runner.build_dataloader(cfgY.trlder)

    # Build hyperbolic model
    model = HypCustomYoloWorld(
        runner.model,
        unknown_index,
        hyp_c=args.hyp_c,
        hyp_dim=args.hyp_dim,
        clip_r=args.clip_r,
        temperature=args.temperature
    )
    
    # ========== CRITICAL: Verify checkpoint loading ==========
    print(f"\n=== Loading Checkpoint: {args.ckpt} ===")
    
    # First, peek at what's in the checkpoint
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    print(f"  Checkpoint keys: {list(state_dict.keys())[:20]}...")
    
    # Check for hyperbolic projector weights
    hyp_keys = [k for k in state_dict.keys() if 'hyp_projector' in k or 'prototype' in k.lower()]
    print(f"  Hyperbolic-related keys: {hyp_keys}")
    
    if 'hyp_projector.prototype_tangent' in state_dict:
        proto_tangent = state_dict['hyp_projector.prototype_tangent']
        print(f"  prototype_tangent shape: {proto_tangent.shape}")
        print(f"  prototype_tangent norm (mean): {proto_tangent.norm(dim=-1).mean():.4f}")
    else:
        print("  WARNING: No hyp_projector.prototype_tangent found in checkpoint!")
    
    if 'frozen_prototypes' in state_dict:
        print(f"  frozen_prototypes shape: {state_dict['frozen_prototypes'].shape}")
    else:
        print("  No frozen_prototypes (expected for T1 training)")
    
    del checkpoint, state_dict  # Free memory
    
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
    
    # ========== Detailed Model Verification ==========
    print(f"\n=== Model Info (After Loading) ===")
    print(f"  frozen_embeddings: {model.frozen_embeddings.shape if model.frozen_embeddings is not None else None}")
    print(f"  embeddings: {model.embeddings.shape if model.embeddings is not None else None}")
    print(f"  frozen_prototypes: {model.frozen_prototypes.shape if model.frozen_prototypes is not None else None}")
    print(f"  hyp_projector.prototype_tangent: {model.hyp_projector.prototype_tangent.shape}")
    print(f"\n  Combined prototypes shape: {model.prototypes.shape}")
    print(f"  Prototype norms (in Poincaré ball): {model.prototypes.norm(dim=-1)}")
    
    # Verify projector weights are loaded (not random)
    print(f"\n  Projector weights check:")
    print(f"    proj_p3[0] weight mean: {model.hyp_projector.proj_p3[0].weight.mean():.6f}")
    print(f"    proj_p3[0] weight std: {model.hyp_projector.proj_p3[0].weight.std():.6f}")
    
    # Collect embeddings
    embeddings, labels, distances = collect_embeddings(
        model, train_loader, 
        num_batches=args.num_batches,
        samples_per_class=args.samples_per_class
    )
    
    if embeddings is None:
        print("ERROR: No embeddings collected!")
        exit(1)
    
    # Project to 2D using proper hyperbolic method
    print(f"\n=== Projecting to 2D using {args.projection} ===")
    
    if args.projection == "hyperbolic_umap":
        print("  Using UMAP with hyperboloid output metric (RECOMMENDED)")
        print("  This properly preserves hyperbolic geometry!")
        embeddings_2d, prototypes_2d = project_to_2d_hyperbolic_umap(
            embeddings, model.prototypes, 
            curvature=args.hyp_c,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist
        )
    elif args.projection == "pca_tangent":
        print("  Using PCA in tangent space (faster but less accurate)")
        embeddings_2d, prototypes_2d = project_to_2d_pca_tangent(
            embeddings, model.prototypes,
            curvature=args.hyp_c
        )
    else:
        raise ValueError(f"Unknown projection method: {args.projection}")
    
    # Plot
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_name = Path(args.ckpt).stem
    save_path = os.path.join(args.output_dir, f"poincare_{args.projection}_{ckpt_name}.png")
    
    plot_poincare_disk(embeddings_2d, labels, prototypes_2d, class_names, distances, save_path, curvature=args.hyp_c)
    
    # Also save raw data
    np.savez(os.path.join(args.output_dir, f"data_{ckpt_name}.npz"),
             embeddings=embeddings.numpy(),
             embeddings_2d=embeddings_2d,
             labels=labels.numpy(),
             prototypes=model.prototypes.detach().cpu().numpy(),
             prototypes_2d=prototypes_2d,
             distances=distances.numpy(),
             class_names=class_names,
             projection_method=args.projection)
    print(f"Saved data: {args.output_dir}/data_{ckpt_name}.npz")
    
    # Print summary of what we learned about the hyperbolic space
    print("\n" + "="*60)
    print("HYPERBOLIC SPACE ANALYSIS SUMMARY")
    print("="*60)
    prototypes_np = model.prototypes.detach().cpu().numpy()
    proto_norms = np.linalg.norm(prototypes_np, axis=1)
    print(f"\nPrototype norms (should be high, near ball boundary for good separation):")
    for i, (name, norm) in enumerate(zip(class_names, proto_norms)):
        print(f"  {name}: {norm:.4f}")
    
    # Compute inter-prototype distances
    from core.hyperbolic import dist_matrix as hyp_dist_matrix
    proto_dists = hyp_dist_matrix(model.prototypes, model.prototypes, c=args.hyp_c).detach().cpu().numpy()
    print(f"\nInter-prototype hyperbolic distances (higher = better separation):")
    print(f"  Min (excl. diagonal): {proto_dists[~np.eye(len(proto_dists), dtype=bool)].min():.4f}")
    print(f"  Mean (excl. diagonal): {proto_dists[~np.eye(len(proto_dists), dtype=bool)].mean():.4f}")
    print(f"  Max: {proto_dists.max():.4f}")
    
    dist_np = distances.numpy()
    print(f"\nEmbedding distances to nearest prototype:")
    print(f"  Min: {dist_np.min():.4f}")
    print(f"  Mean: {dist_np.mean():.4f}")
    print(f"  Max: {dist_np.max():.4f}")
    print(f"  Samples > OOD threshold (2.5): {(dist_np > 2.5).sum()} / {len(dist_np)}")
    print("="*60)
