"""
Calibration script for horospherical OOD thresholds.

Run this AFTER training to compute the OOD threshold based on max horosphere scores.
For horospherical classifiers: max_score < threshold → unknown

Key difference from distance-based:
- OLD: min_distance > threshold → unknown (higher = more OOD)
- NEW: max_score < threshold → unknown (lower = more OOD)

Usage:
    python calibrate.py --task IDD/t1 --ckpt IDD/t1/hyperbolic/model_final.pth
"""

import os
import torch
import numpy as np
from tqdm import tqdm

from core import add_config
from core.util.model_ema import add_model_ema_configs
from core.pascal_voc import register_pascal_voc, inital_prompts
from core.hyp_customyoloworld import HypCustomYoloWorld, load_hyp_ckpt

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


@torch.no_grad()
def calibrate_thresholds(model, train_loader, num_classes, percentile=1):
    """
    Calibration pass: collect max horosphere score statistics for OOD threshold.
    
    For horospherical approach:
    - Collect max_score = max_k(ξ_k(x)) for each GT embedding
    - OOD threshold = percentile(max_scores) - margin
    - Detection: max_score < threshold → unknown
    
    Parameters
    ----------
    model : HypCustomYoloWorld
        Trained model
    train_loader : DataLoader
        Training dataloader (has GT boxes)
    num_classes : int
        Number of known classes
    percentile : int
        Low percentile for threshold (1 = use 1st percentile as lower bound)
        Since known class samples should have HIGH scores, we use low percentile
    
    Returns
    -------
    dict
        Calibration results including global threshold
    """
    print(f"\n{'='*60}")
    print(f"HOROSPHERICAL CALIBRATION: Computing OOD Threshold")
    print(f"{'='*60}")
    
    model.eval()
    
    # Collect all max scores
    all_max_scores = []
    class_scores = [[] for _ in range(num_classes)]
    samples_per_class = [0] * num_classes
    
    print(f"  Collecting max horosphere scores from training set...")
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Calibration")):
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
                    
                    # Compute horosphere scores and get max
                    scores = model.compute_horosphere_scores(emb.unsqueeze(0))  # (1, K)
                    max_score = scores.max().item()
                    
                    all_max_scores.append(max_score)
                    class_scores[cls_id].append(max_score)
                    samples_per_class[cls_id] += 1
                    
        except Exception as e:
            if batch_idx == 0:
                print(f"  Warning in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
            continue
    
    # Compute threshold
    print(f"\n  Computing threshold (percentile={percentile})...")
    
    all_max_scores = np.array(all_max_scores)
    
    # For known classes, max_score should be HIGH
    # OOD threshold = low percentile of max_scores - margin
    # Unknown samples should have max_score BELOW this threshold
    low_percentile_score = np.percentile(all_max_scores, percentile)
    margin = 0.5  # Safety margin
    global_threshold = low_percentile_score - margin
    
    print(f"\n  Score distribution for known classes:")
    print(f"    Total samples: {len(all_max_scores)}")
    print(f"    Min score: {all_max_scores.min():.4f}")
    print(f"    Max score: {all_max_scores.max():.4f}")
    print(f"    Mean score: {all_max_scores.mean():.4f}")
    print(f"    Std score: {all_max_scores.std():.4f}")
    print(f"    {percentile}th percentile: {low_percentile_score:.4f}")
    print(f"    Global OOD threshold: {global_threshold:.4f}")
    
    # Per-class statistics
    stats = {}
    print(f"\n  Per-class statistics:")
    for c in range(num_classes):
        if len(class_scores[c]) > 0:
            scores = np.array(class_scores[c])
            stats[c] = {
                'count': len(scores),
                'min': float(scores.min()),
                'max': float(scores.max()),
                'mean': float(scores.mean()),
                'std': float(scores.std()),
            }
            print(f"    Class {c}: count={len(scores)}, "
                  f"min={scores.min():.4f}, max={scores.max():.4f}, "
                  f"mean={scores.mean():.4f}")
        else:
            stats[c] = {'count': 0}
            print(f"    Class {c}: NO SAMPLES")
    
    calibration_results = {
        'global_threshold': global_threshold,
        'low_percentile_score': low_percentile_score,
        'percentile': percentile,
        'margin': margin,
        'all_scores_mean': float(all_max_scores.mean()),
        'all_scores_std': float(all_max_scores.std()),
        'stats': stats,
        'num_classes': num_classes
    }
    
    print(f"\n{'='*60}")
    print(f"  RECOMMENDATION: Use --ood_threshold {global_threshold:.4f}")
    print(f"  Detection rule: max_score < {global_threshold:.4f} → unknown")
    print(f"{'='*60}")
    
    return calibration_results


if __name__ == "__main__":
    parser0 = default_argument_parser()
    parser0.add_argument("--task", default="IDD/t1")
    parser0.add_argument("--ckpt", required=True, help="Path to checkpoint to calibrate")
    parser0.add_argument("--hyp_c", type=float, default=1.0)  # Changed to c=1.0
    parser0.add_argument("--hyp_dim", type=int, default=256)
    parser0.add_argument("--clip_r", type=float, default=0.95)  # Changed for c=1.0
    parser0.add_argument("--percentile", type=int, default=1, help="Low percentile for threshold")
    parser0.add_argument("--output_dir", default=None, help="Output directory (default: same as checkpoint)")
    
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
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Classes: {class_names}")
    print(f"  Curvature: {args.hyp_c}")
    print(f"  Percentile: {args.percentile}")

    # Initialize YOLO-World
    runner = Runner.from_cfg(cfgY)
    runner.call_hook("before_run")
    runner.load_or_resume()
    runner.model.reparameterize(classnames)
    runner.model.eval()

    # Use train loader for calibration
    train_loader = Runner.build_dataloader(cfgY.trlder)

    # Build hyperbolic model
    model = HypCustomYoloWorld(
        runner.model,
        unknown_index,
        hyp_c=args.hyp_c,
        hyp_dim=args.hyp_dim,
        clip_r=args.clip_r
    )
    
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
    
    print(f"\n=== Model Info ===")
    print(f"  num_classes: {model.num_classes}")
    print(f"  Prototypes shape: {model.prototypes.shape}")
    print(f"  Prototype norms (should be ~1.0): {model.prototypes.norm(dim=-1)}")
    
    # Run calibration
    calibration_results = calibrate_thresholds(
        model, train_loader, 
        num_classes=unknown_index,
        percentile=args.percentile
    )
    
    # Save results
    output_dir = args.output_dir if args.output_dir else os.path.dirname(args.ckpt)
    os.makedirs(output_dir, exist_ok=True)
    
    calibration_path = os.path.join(output_dir, 'calibration.pth')
    torch.save(calibration_results, calibration_path)
    print(f"\n✓ Saved calibration to: {calibration_path}")
    
    # Update checkpoint with threshold
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    checkpoint['ood_threshold'] = calibration_results['global_threshold']
    
    # Save updated checkpoint
    ckpt_name = os.path.basename(args.ckpt).replace('.pth', '_calibrated.pth')
    calibrated_ckpt_path = os.path.join(output_dir, ckpt_name)
    torch.save(checkpoint, calibrated_ckpt_path)
    print(f"✓ Saved calibrated checkpoint to: {calibrated_ckpt_path}")
    
    print(f"\n=== Calibration Complete ===")
    print(f"  OOD threshold: {calibration_results['global_threshold']:.4f}")
    print(f"  Use: python test_hyp.py --ood_threshold {calibration_results['global_threshold']:.4f}")
