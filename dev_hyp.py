"""
Hyperbolic YOLO World Training Script.

Training loop for hyperbolic prototype-based open-world detection.
Uses HypCustomYoloWorld with distance-based OOD detection.
"""

import os
import itertools
import weakref
from typing import Any, Dict, List, Set

import torch
import torch.optim as optim
from fvcore.nn.precise_bn import get_bn_modules


from core import DatasetMapper, add_config
from core.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer
from core.pascal_voc import register_pascal_voc, inital_prompts
from core.pascal_voc_evaluation import PascalVOCDetectionEvaluator
from core.hyp_customyoloworld import HypCustomYoloWorld, load_hyp_ckpt
from core.eval_utils import Trainer

from mmengine.config import Config
from mmengine.runner import Runner
from torchvision.ops import nms, batched_nms
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup
from detectron2.config import get_cfg


from tqdm import tqdm


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
        """Register all splits of datasets."""
        for name, split in self.PREDEFINED_SPLITS_DATASET.items():
            register_pascal_voc(name, self.dataset_root, self.dataset_key, split, self.cfg)


def setup(args):
    """Create configs and perform basic setups."""
    cfg = get_cfg()
    add_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    
    # IMPORTANT: Also merge task-specific YAML to get correct CUR_INTRODUCED_CLS
    # base.yaml doesn't have it, t1.yaml does
    task_name = args.task.split('/')[0]
    split_name = args.task.split('/')[1]
    task_yaml = os.path.join("./configs", task_name, f"{split_name}.yaml")
    if os.path.exists(task_yaml):
        cfg.merge_from_file(task_yaml)
        print(f"Merged task config: {task_yaml}")
    
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def save_model(model, optimizer, epoch, save_dir='checkpoints', file_name="model", actual_epoch=None):
    """Save model checkpoint with optimizer state."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, f"{file_name}_{epoch}.pth")
    epoch_to_save = actual_epoch if actual_epoch is not None else epoch
    checkpoint = {
        'epoch': epoch_to_save,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, file_path)
    print(f"Model saved to {file_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load checkpoint and return epoch to resume from."""
    import re
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        if not isinstance(epoch, int):
            epoch = int(epoch)
        start_epoch = epoch + 1
        print(f"✓ Resuming from epoch {start_epoch} (loaded checkpoint from epoch {epoch})")
    else:
        model.load_state_dict(checkpoint)
        match = re.search(r'model_(\d+)\.pth', checkpoint_path)
        if match:
            start_epoch = int(match.group(1)) + 1
            print(f"✓ Loaded weights from epoch {int(match.group(1))}, resuming from {start_epoch}")
        else:
            start_epoch = 0
            print(f"⚠ Loaded weights only, couldn't determine epoch")
    
    return start_epoch


def visualize_training_embeddings(model, dataloader, epoch, save_dir, num_samples=50, method='pca'):
    """
    Visualize hyperbolic embeddings during training.
    
    Uses spatially-aware sampling: for each GT box, samples the nearest anchor embedding.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    print(f"\n  [Visualizing embeddings (epoch {epoch})...]", flush=True)
    
    model.eval()
    all_embeddings = []
    all_labels = []
    samples_per_class = {}
    
    # Collect embeddings
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Check if we have enough samples
            have_enough = all(
                samples_per_class.get(c, 0) >= num_samples 
                for c in samples_per_class
            ) if samples_per_class else False
            
            if i > num_samples * 3 or (have_enough and len(samples_per_class) > 0):
                break
            
            try:
                data_batch = model.parent.data_preprocessor(batch)
                
                # Get FPN features
                x = model.parent.backbone(data_batch['inputs'])
                x = model.parent.neck(x)
                
                # Project to hyperbolic space
                hyp_embeddings = model.hyperbolic_projector(x)  # (B, 8400, dim)
                
                # Process each image in batch
                for b_idx, data_sample in enumerate(data_batch['data_samples']):
                    gt_bboxes = data_sample.gt_instances.bboxes  # (N, 4)
                    gt_labels = data_sample.gt_instances.labels   # (N,)
                    
                    if len(gt_labels) == 0:
                        continue
                    
                    # Get anchor grid (8400 anchors from FPN)
                    # YOLO-World uses 80x80 + 40x40 + 20x20 = 8400 anchors for 640x640
                    h, w = data_batch['inputs'].shape[-2:]
                    anchor_centers = _get_anchor_centers(h, w, device=gt_bboxes.device)
                    
                    # For each GT box, find nearest anchor
                    for box_idx in range(len(gt_bboxes)):
                        cls_id = int(gt_labels[box_idx].item())
                        
                        if cls_id not in samples_per_class:
                            samples_per_class[cls_id] = 0
                        
                        if samples_per_class[cls_id] >= num_samples:
                            continue
                        
                        # Box center
                        box = gt_bboxes[box_idx]
                        cx = (box[0] + box[2]) / 2
                        cy = (box[1] + box[3]) / 2
                        
                        # Find nearest anchor
                        dists = (anchor_centers[:, 0] - cx)**2 + (anchor_centers[:, 1] - cy)**2
                        nearest_idx = dists.argmin().item()
                        
                        # Get embedding for this anchor
                        emb = hyp_embeddings[b_idx, nearest_idx].cpu().numpy()
                        all_embeddings.append(emb)
                        all_labels.append(cls_id)
                        samples_per_class[cls_id] += 1
                        
            except Exception as e:
                continue
    
    model.train()
    
    if len(all_embeddings) == 0:
        print("  [No embeddings collected]")
        return
    
    # Stack embeddings
    embeddings = np.stack(all_embeddings, axis=0)
    labels = np.array(all_labels)
    
    print(f"  Collected {len(embeddings)} embeddings across {len(np.unique(labels))} classes")
    
    # Dimensionality reduction for visualization
    if embeddings.shape[1] > 2:
        if method == 'pca':
            reducer = PCA(n_components=2)
            vis_embeddings = reducer.fit_transform(embeddings)
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            perp = min(30, len(embeddings) - 1)
            reducer = TSNE(n_components=2, perplexity=perp)
            vis_embeddings = reducer.fit_transform(embeddings)
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=2, n_neighbors=min(15, len(embeddings)-1))
                vis_embeddings = reducer.fit_transform(embeddings)
            except ImportError:
                reducer = PCA(n_components=2)
                vis_embeddings = reducer.fit_transform(embeddings)
        
        # Also reduce prototypes with same method
        prototypes = model.hyperbolic_projector.get_prototypes().detach().cpu().numpy()
        # Fit on combined data for consistent projection
        combined = np.vstack([embeddings, prototypes])
        if method == 'pca':
            combined_vis = PCA(n_components=2).fit_transform(combined)
        else:
            combined_vis = combined[:, :2]  # Just take first 2 dims for others
        vis_embeddings = combined_vis[:len(embeddings)]
        proto_vis = combined_vis[len(embeddings):]
    else:
        vis_embeddings = embeddings
        proto_vis = model.hyperbolic_projector.get_prototypes().detach().cpu().numpy()
    
    # Plot Poincaré disk
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
    
    # Plot embeddings by class
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap('tab20', max(len(unique_labels), 20))
    
    for idx, cls_id in enumerate(unique_labels):
        mask = labels == cls_id
        ax.scatter(
            vis_embeddings[mask, 0], 
            vis_embeddings[mask, 1],
            c=[cmap(idx % 20)],
            label=f'Class {cls_id}',
            alpha=0.6,
            s=30
        )
    
    # Plot prototypes with stars
    for idx in range(len(proto_vis)):
        if idx < len(unique_labels):
            ax.scatter(
                proto_vis[idx, 0], proto_vis[idx, 1], 
                c=[cmap(idx % 20)], 
                marker='*', s=300, edgecolors='black', linewidths=1.5
            )
        else:
            ax.scatter(
                proto_vis[idx, 0], proto_vis[idx, 1], 
                c='gray', 
                marker='*', s=200, edgecolors='black', linewidths=1, alpha=0.5
            )
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax.set_title(f'Hyperbolic Embeddings - Epoch {epoch} ({method.upper()})')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Save
    vis_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    save_path = os.path.join(vis_dir, f'embeddings_epoch_{epoch}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved visualization to {save_path}")


def _get_anchor_centers(h, w, device):
    """Generate anchor center coordinates for YOLO-World's FPN structure."""
    # YOLO-World XL uses strides 8, 16, 32
    anchors = []
    for stride in [8, 16, 32]:
        fh, fw = h // stride, w // stride
        y = torch.arange(fh, device=device) * stride + stride / 2
        x = torch.arange(fw, device=device) * stride + stride / 2
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        anchors.append(torch.stack([xx.flatten(), yy.flatten()], dim=1))
    return torch.cat(anchors, dim=0)  # (8400, 2)


@torch.no_grad()
def calibrate_thresholds(model, train_loader, num_classes, save_dir, percentile=99):
    """
    Calibration pass: collect distance statistics for adaptive OOD thresholds.
    
    Run this AFTER training completes to compute per-class distance thresholds.
    Uses the maximum (or percentile) distance between GT embeddings and their
    assigned prototypes as the class-specific OOD threshold.
    
    Parameters
    ----------
    model : HypCustomYoloWorld
        Trained model
    train_loader : DataLoader
        Training dataloader (has GT boxes)
    num_classes : int
        Number of known classes
    save_dir : str
        Directory to save calibration results
    percentile : int
        Percentile for threshold (99 = use 99th percentile of distances)
    
    Returns
    -------
    dict
        Calibration results including per-class thresholds
    """
    import numpy as np
    from core.hyperbolic import dist_matrix
    
    print(f"\n{'='*60}")
    print(f"CALIBRATION PHASE: Computing OOD Thresholds")
    print(f"{'='*60}")
    
    model.eval()
    
    # Per-class distance lists
    class_distances = [[] for _ in range(num_classes)]
    samples_per_class = [0] * num_classes
    
    print(f"  Collecting distances from training set...")
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Calibration")):
        try:
            data_batch = model.parent.data_preprocessor(batch)
            
            # Get FPN features
            x = model.parent.backbone.forward_image(data_batch['inputs'])
            
            # Apply neck (handle mm_neck)
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
            anchor_centers = _get_anchor_centers(h, w, device=hyp_embeddings.device)
            
            prototypes = model.prototypes  # (K, dim)
            
            # Process each image
            for b_idx, data_sample in enumerate(data_batch['data_samples']):
                gt_bboxes = data_sample.gt_instances.bboxes
                gt_labels = data_sample.gt_instances.labels
                
                if len(gt_labels) == 0:
                    continue
                
                # For each GT box, get nearest anchor's embedding
                for box_idx in range(len(gt_bboxes)):
                    cls_id = int(gt_labels[box_idx].item())
                    
                    if cls_id >= num_classes:
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
                    
                    # Compute distance to assigned prototype
                    dist_to_own_proto = dist_matrix(
                        emb.unsqueeze(0), 
                        prototypes[cls_id:cls_id+1], 
                        c=model.hyp_c
                    ).item()
                    
                    class_distances[cls_id].append(dist_to_own_proto)
                    samples_per_class[cls_id] += 1
                    
        except Exception as e:
            if batch_idx == 0:
                print(f"  Warning in batch {batch_idx}: {e}")
            continue
    
    # Compute thresholds
    print(f"\n  Computing thresholds (percentile={percentile})...")
    
    thresholds = torch.zeros(num_classes)
    stats = {}
    
    for c in range(num_classes):
        if len(class_distances[c]) > 0:
            dists = np.array(class_distances[c])
            thresholds[c] = np.percentile(dists, percentile)
            stats[c] = {
                'count': len(dists),
                'min': float(dists.min()),
                'max': float(dists.max()),
                'mean': float(dists.mean()),
                'std': float(dists.std()),
                f'p{percentile}': float(thresholds[c])
            }
            print(f"    Class {c}: count={len(dists)}, "
                  f"min={dists.min():.4f}, max={dists.max():.4f}, "
                  f"mean={dists.mean():.4f}, threshold={thresholds[c]:.4f}")
        else:
            thresholds[c] = 5.0  # fallback
            stats[c] = {'count': 0, 'threshold': 5.0}
            print(f"    Class {c}: NO SAMPLES - using fallback threshold=5.0")
    
    # Global threshold (max of all class thresholds with margin)
    global_threshold = thresholds.max().item() * 1.1
    print(f"\n  Global OOD threshold (max + 10% margin): {global_threshold:.4f}")
    
    # Save calibration results
    calibration_path = os.path.join(save_dir, 'calibration.pth')
    calibration_results = {
        'class_thresholds': thresholds,
        'global_threshold': global_threshold,
        'percentile': percentile,
        'stats': stats,
        'num_classes': num_classes
    }
    torch.save(calibration_results, calibration_path)
    print(f"  Saved calibration to: {calibration_path}")
    
    # Also update the final checkpoint with thresholds
    final_ckpt_path = os.path.join(save_dir, 'model_final.pth')
    if os.path.exists(final_ckpt_path):
        checkpoint = torch.load(final_ckpt_path, map_location='cpu')
        checkpoint['class_thresholds'] = thresholds
        checkpoint['global_threshold'] = global_threshold
        torch.save(checkpoint, final_ckpt_path)
        print(f"  Updated final checkpoint with thresholds")
    
    print(f"{'='*60}")
    
    return calibration_results


if __name__ == "__main__":
    parser0 = default_argument_parser()
    parser0.add_argument("--task", default="")
    parser0.add_argument("--ckpt", default="model.pth")
    parser0.add_argument("--resume_from", default="", help="Path to checkpoint to resume from")
    parser0.add_argument("--exp_name", default="", help="Experiment name for separate save directory")
    
    # Hyperbolic hyperparameters
    parser0.add_argument("--hyp_c", type=float, default=0.1, help="Poincaré ball curvature")
    parser0.add_argument("--hyp_dim", type=int, default=256, help="Hyperbolic embedding dimension")
    parser0.add_argument("--clip_r", type=float, default=2.3, help="Clip radius for ToPoincare")
    parser0.add_argument("--temperature", type=float, default=0.1, help="Contrastive loss temperature")
    parser0.add_argument("--hyp_loss_weight", type=float, default=1.0, help="Weight for hyperbolic loss")
    
    # Prototype separation hyperparameters (EXPERIMENTAL - disabled by default)
    # These are heuristic additions, not from literature. Use with caution.
    parser0.add_argument("--separation_weight", type=float, default=0.0, help="Weight for inter-prototype separation loss (0=disabled)")
    parser0.add_argument("--boundary_weight", type=float, default=0.0, help="Weight for boundary push loss (0=disabled)")
    parser0.add_argument("--min_proto_dist", type=float, default=2.0, help="Minimum desired distance between prototypes")
    parser0.add_argument("--target_norm", type=float, default=0.9, help="Target norm for prototypes (push toward boundary)")
    
    # Visualization arguments
    parser0.add_argument("--visualize_every", type=int, default=5, help="Visualize embeddings every N epochs (0=disabled)")
    parser0.add_argument("--vis_samples", type=int, default=50, help="Number of samples per class for visualization")
    parser0.add_argument("--vis_method", type=str, default="pca", choices=["pca", "tsne", "umap"], help="Dim reduction method")
    
    # Calibration arguments
    parser0.add_argument("--skip_calibration", action="store_true", help="Skip calibration phase after training")
    parser0.add_argument("--calibration_percentile", type=int, default=99, help="Percentile for OOD threshold")
    
    args = parser0.parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)

    # Extract task components
    task_name = args.task.split('/')[0]
    split_name = args.task.split('/')[1]
    
    # Determine dataset key for registration
    if task_name == "nu-OWODB":
        dataset_key = 'nu-prompt'
        class_names = list(inital_prompts()[dataset_key])
    elif split_name in ['t2', 't3', 't4']:
        dataset_key = f"{task_name}_T{split_name[1].upper()}"
        class_names = list(inital_prompts()[dataset_key])
    else:
        dataset_key = task_name
        class_names = list(inital_prompts()[task_name])
    
    print(f"\n=== Dataset Configuration ===")
    print(f"Task: {args.task}")
    print(f"Dataset key: {dataset_key}")
    print(f"Total classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    
    # Register dataset
    data_register = Register('./datasets/', args.task, cfg, dataset_key)
    data_register.register_dataset()

    # Model config
    config_file = os.path.join("./configs", task_name, split_name + ".py")
    cfgY = Config.fromfile(config_file)
    cfgY.work_dir = "."
    
    # Handle resume mode
    if args.resume_from:
        cfgY.load_from = None
        cfgY.resume = False
        print(f"Resume mode: Will load checkpoint from {args.resume_from}")
    else:
        if cfg.TEST.PREV_INTRODUCED_CLS == 0:
            cfgY.load_from = args.ckpt
    
    unknown_index = cfg.TEST.PREV_INTRODUCED_CLS + cfg.TEST.CUR_INTRODUCED_CLS
    class_names = class_names[:unknown_index]
    classnames = [class_names]

    print(f"\n=== Hyperbolic Parameters ===")
    print(f"  Curvature (c): {args.hyp_c}")
    print(f"  Embedding dim: {args.hyp_dim}")
    print(f"  Clip radius: {args.clip_r}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Loss weight: {args.hyp_loss_weight}")

    print(f"\n=== Initializing YOLO-World model ===")
    runner = Runner.from_cfg(cfgY)
    print(f"✓ Runner created")
    runner.call_hook("before_run")
    print(f"✓ Before-run hooks called")
    runner.load_or_resume()
    print(f"✓ Checkpoint loaded")
    
    print(f"=== Moving model to GPU ===")
    runner.model = runner.model.cuda()
    print(f"✓ Model on GPU: {next(runner.model.parameters()).device}")
    
    print(f"=== Reparameterizing model with {len(class_names)} classes ===")
    runner.model.reparameterize(classnames)
    print(f"✓ Model reparameterized")
    runner.model.train()
    print(f"✓ Model set to training mode")

    print(f"=== Building data loaders ===")
    train_loader = Runner.build_dataloader(cfgY.trlder)
    print(f"✓ Training loader built ({len(train_loader)} batches)")
    test_loader = Runner.build_dataloader(cfgY.test_dataloader)
    print(f"✓ Test loader built")

    evaluator = Trainer.build_evaluator(cfg, "my_val")
    evaluator.reset()

    # ============================================================================
    # HYPERBOLIC TRAINING: HyperbolicProjector + Poincaré prototypes
    # ============================================================================
    # - frozen_embeddings: base class text embeddings (FROZEN)
    # - embeddings: novel class text embeddings (TRAINABLE)
    # - frozen_prototypes: base class prototypes (FROZEN for T2+)
    # - hyp_projector.prototype_tangent: novel class prototypes (TRAINABLE)
    # ============================================================================
    
    trainable = ['embeddings']
    model = HypCustomYoloWorld(
        runner.model,
        unknown_index,
        hyp_c=args.hyp_c,
        hyp_dim=args.hyp_dim,
        clip_r=args.clip_r,
        temperature=args.temperature,
        separation_weight=args.separation_weight,
        boundary_weight=args.boundary_weight,
        min_proto_dist=args.min_proto_dist,
        target_norm=args.target_norm
    )
    
    if args.resume_from:
        # Resume from a hyperbolic checkpoint
        print(f"=== RESUMING TRAINING ===")
        print(f"  Loading from: {args.resume_from}")
        with torch.no_grad():
            model = load_hyp_ckpt(
                model, args.resume_from,
                cfg.TEST.PREV_INTRODUCED_CLS,
                cfg.TEST.CUR_INTRODUCED_CLS
            )
        model = model.cuda()
    elif cfg.TEST.PREV_INTRODUCED_CLS > 0:
        # T2+: Load previous task checkpoint to get frozen prototypes
        print(f"=== T2+ TRAINING (loading previous task model) ===")
        print(f"  Base classes (frozen): {cfg.TEST.PREV_INTRODUCED_CLS}")
        print(f"  Novel classes (trainable): {cfg.TEST.CUR_INTRODUCED_CLS}")
        with torch.no_grad():
            model = load_hyp_ckpt(
                model, args.ckpt,
                cfg.TEST.PREV_INTRODUCED_CLS,
                cfg.TEST.CUR_INTRODUCED_CLS
            )
        model = model.cuda()
    else:
        # T1: Fresh training - Runner already loaded pretrained YOLO-World
        # HyperbolicProjector starts with random weights, prototypes initialized
        print(f"=== T1 FRESH TRAINING ===")
        print(f"  Pretrained YOLO-World loaded by Runner")
        print(f"  HyperbolicProjector: random init")
        print(f"  Prototypes: random init ({unknown_index} classes)")
        model = model.cuda()
    
    # Set trainable parameters
    for name, param in model.named_parameters():
        if name in trainable:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    
    # Enable gradients for novel class prototypes
    model.enable_projector_grad(cfg.TEST.PREV_INTRODUCED_CLS)
    
    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=cfgY.base_lr, 
        weight_decay=cfgY.weight_decay
    )
    print(f"  Optimizer LR: {cfgY.base_lr}, Weight Decay: {cfgY.weight_decay}")

    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume_from:
        start_epoch = load_checkpoint(model, optimizer, args.resume_from)

    model.train()
    
    # Determine save directory
    if args.exp_name:
        save_dir = os.path.join(args.task, args.exp_name)
    else:
        save_dir = os.path.join(args.task, "hyperbolic")
    
    print(f"\nModels will be saved to: {save_dir}")
    
    # Training loop
    for epoch in range(start_epoch, cfgY.max_epochs):
        print(f"\nEpoch: {epoch}", flush=True)
        step = 0
        epoch_loss_cls = 0.0
        epoch_loss_dfl = 0.0
        epoch_loss_bbox = 0.0
        epoch_loss_hyp = 0.0
        last_hyp_breakdown = {}
        
        for i in train_loader:
            optimizer.zero_grad()
            data_batch = model.parent.data_preprocessor(i)
            
            # Use detailed loss breakdown every 100 steps for logging
            if step % 100 == 0:
                head_losses, hyp_loss, hyp_breakdown = model.head_loss_with_breakdown(
                    data_batch['inputs'],
                    data_batch['data_samples']
                )
                last_hyp_breakdown = hyp_breakdown
            else:
                head_losses, hyp_loss = model.head_loss(
                    data_batch['inputs'],
                    data_batch['data_samples']
                )
            
            loss = (
                head_losses['loss_cls'] + 
                head_losses['loss_dfl'] + 
                head_losses['loss_bbox'] + 
                args.hyp_loss_weight * hyp_loss
            )
            loss.backward()
            
            # Accumulate losses for logging
            epoch_loss_cls += head_losses['loss_cls'].item()
            epoch_loss_dfl += head_losses['loss_dfl'].item()
            epoch_loss_bbox += head_losses['loss_bbox'].item()
            epoch_loss_hyp += hyp_loss.item()
            
            if step % 20 == 0:
                print(f'  step {step}: cls={head_losses["loss_cls"].item():.4f} '
                      f'dfl={head_losses["loss_dfl"].item():.4f} '
                      f'bbox={head_losses["loss_bbox"].item():.4f} '
                      f'hyp={hyp_loss.item():.4f}', flush=True)
            
            # Log prototype stats every 100 steps
            if step % 100 == 0 and last_hyp_breakdown:
                # Ball radius = 1/sqrt(c), with c=0.1 → radius ≈ 3.16
                ball_radius = 1.0 / (args.hyp_c ** 0.5)
                norm_pct = last_hyp_breakdown.get("proto_norm_mean", 0) / ball_radius * 100
                print(f'    [Proto] norm={last_hyp_breakdown.get("proto_norm_mean", 0):.3f} '
                      f'({norm_pct:.1f}% to boundary) '
                      f'dist={last_hyp_breakdown.get("proto_dist_mean", 0):.3f} '
                      f'(min={last_hyp_breakdown.get("proto_dist_min", 0):.3f})', flush=True)
            
            optimizer.step()
            step += 1
        
        # Log epoch summary
        n_steps = max(step, 1)
        print(f"✓ Epoch {epoch} completed ({step} steps)", flush=True)
        print(f"  Avg losses: cls={epoch_loss_cls/n_steps:.4f} "
              f"dfl={epoch_loss_dfl/n_steps:.4f} "
              f"bbox={epoch_loss_bbox/n_steps:.4f} "
              f"hyp={epoch_loss_hyp/n_steps:.4f}", flush=True)
        
        # Log prototype stats at end of epoch
        if last_hyp_breakdown:
            ball_radius = 1.0 / (args.hyp_c ** 0.5)
            norm_pct = last_hyp_breakdown.get('proto_norm_mean', 0) / ball_radius * 100
            print(f"  Proto stats: norm_mean={last_hyp_breakdown.get('proto_norm_mean', 0):.4f} "
                  f"({norm_pct:.1f}% to boundary, radius={ball_radius:.2f}) "
                  f"dist_mean={last_hyp_breakdown.get('proto_dist_mean', 0):.4f} "
                  f"dist_min={last_hyp_breakdown.get('proto_dist_min', 0):.4f}", flush=True)
        
        # Save checkpoints
        if epoch % 5 == 0:
            save_model(model, optimizer, epoch, save_dir=save_dir)
        save_model(model, optimizer, 'latest', save_dir=save_dir, actual_epoch=epoch)
        
        # Visualize embeddings periodically (every 5 epochs)
        if args.visualize_every > 0 and (epoch % args.visualize_every == 0 or epoch == cfgY.max_epochs - 1):
            try:
                visualize_training_embeddings(
                    model, train_loader, epoch, save_dir,
                    num_samples=args.vis_samples,
                    method=args.vis_method
                )
            except Exception as e:
                print(f"  [Visualization failed: {e}]")
    
    # Save final model
    save_model(model, optimizer, 'final', save_dir=save_dir)
    print(f"\n=== Training complete ===")
    
    # Run calibration phase to compute OOD thresholds
    # This happens AFTER training is saved, so training is not lost if calibration fails
    if not args.skip_calibration:
        try:
            calibration_results = calibrate_thresholds(
                model, train_loader, 
                num_classes=unknown_index,
                save_dir=save_dir,
                percentile=args.calibration_percentile
            )
            print(f"\n✓ Calibration complete!")
            print(f"  Global OOD threshold: {calibration_results['global_threshold']:.4f}")
        except Exception as e:
            print(f"\n⚠ Calibration failed: {e}")
            print(f"  Training checkpoint is still saved. Run calibration separately if needed.")
    else:
        print(f"\n⚠ Calibration skipped (--skip_calibration flag)")
