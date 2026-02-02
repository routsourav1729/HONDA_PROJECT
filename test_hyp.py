"""
Hyperbolic YOLO World Test/Evaluation Script.

Inference with distance-based OOD detection using hyperbolic embeddings.
Objects far from all prototypes in Poincaré ball are classified as unknown.
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
import supervision as sv
import os.path as osp
import cv2

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
    
    # Merge task-specific config to get correct class counts
    if args.task:
        task_yaml = os.path.join("configs", args.task.split('/')[0], args.task.split('/')[1] + ".yaml")
        if os.path.exists(task_yaml):
            cfg.merge_from_file(task_yaml)
            print(f"Merged task config: {task_yaml}")
    
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


if __name__ == "__main__":
    parser0 = default_argument_parser()
    parser0.add_argument("--task", default="")
    parser0.add_argument("--ckpt", default="model.pth")
    
    # Hyperbolic hyperparameters (should match training)
    parser0.add_argument("--hyp_c", type=float, default=0.1, help="Poincaré ball curvature")
    parser0.add_argument("--hyp_dim", type=int, default=256, help="Hyperbolic embedding dimension")
    parser0.add_argument("--clip_r", type=float, default=2.3, help="Clip radius for ToPoincare")
    parser0.add_argument("--temperature", type=float, default=0.1, help="Temperature (used in training)")
    
    # OOD detection threshold
    parser0.add_argument("--ood_threshold", type=float, default=None, 
                        help="Distance threshold for OOD detection (higher = stricter)")
    parser0.add_argument("--visualize", action="store_true", help="Visualize Poincaré embeddings")
    parser0.add_argument("--output_dir", default="visualizations", help="Directory for visualizations")
    
    args = parser0.parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)

    task_name = args.task.split('/')[0]
    split_name = args.task.split('/')[1]
    
    # For incremental tasks, use the full class list
    if split_name in ['t2', 't3', 't4']:
        dataset_key = f"{task_name}_T{split_name[1].upper()}"
    else:
        dataset_key = task_name
    
    print(f"\n=== Evaluation Configuration ===")
    print(f"Task: {args.task}")
    print(f"Dataset key: {dataset_key}")
    
    data_register = Register('./datasets/', args.task, cfg, dataset_key)
    data_register.register_dataset()

    class_names = list(inital_prompts()[dataset_key])

    # Model config
    config_file = os.path.join("./configs", task_name, split_name + ".py")
    cfgY = Config.fromfile(config_file)
    cfgY.work_dir = "."
    
    # Use config OOD threshold if not specified via CLI
    # Priority: CLI arg > checkpoint calibration > config > default
    if args.ood_threshold is None:
        # Try to load from checkpoint calibration
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        if 'global_threshold' in checkpoint:
            args.ood_threshold = checkpoint['global_threshold']
            print(f"  Loaded calibrated threshold from checkpoint: {args.ood_threshold:.4f}")
        elif hasattr(cfgY, 'hyp_ood_threshold'):
            args.ood_threshold = cfgY.hyp_ood_threshold
        else:
            args.ood_threshold = 5.0  # Updated default for hyperbolic distances
            print(f"  WARNING: Using default threshold {args.ood_threshold}. Consider running calibration!")
    
    # Also try to load per-class thresholds for more precise OOD detection
    class_thresholds = None
    try:
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        if 'class_thresholds' in checkpoint:
            class_thresholds = checkpoint['class_thresholds']
            print(f"  Loaded per-class thresholds from checkpoint")
    except:
        pass
    
    unknown_index = cfg.TEST.PREV_INTRODUCED_CLS + cfg.TEST.CUR_INTRODUCED_CLS
    class_names = class_names[:unknown_index]
    classnames = [class_names]

    print(f"\n=== Hyperbolic Parameters ===")
    print(f"  Curvature (c): {args.hyp_c}")
    print(f"  Embedding dim: {args.hyp_dim}")
    print(f"  Clip radius: {args.clip_r}")
    print(f"  OOD threshold: {args.ood_threshold}")
    print(f"  Number of classes: {len(class_names)} + 1 (unknown)")

    # Initialize YOLO-World
    runner = Runner.from_cfg(cfgY)
    runner.call_hook("before_run")
    runner.load_or_resume()
    runner.model.reparameterize(classnames)
    runner.model.eval()

    # Build data loaders
    train_loader = Runner.build_dataloader(cfgY.trlder)
    test_loader = Runner.build_dataloader(cfgY.test_dataloader)

    evaluator = Trainer.build_evaluator(cfg, "my_val")
    evaluator.reset()

    # Build hyperbolic model
    model = HypCustomYoloWorld(
        runner.model,
        unknown_index,
        hyp_c=args.hyp_c,
        hyp_dim=args.hyp_dim,
        clip_r=args.clip_r,
        temperature=args.temperature
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
    
    print(f"\n=== Starting Evaluation ===")
    print(f"  Total prototypes: {model.prototypes.shape[0]}")
    
    # Optional: Collect embeddings for visualization
    if args.visualize:
        all_embeddings = []
        all_labels = []
    
    for i in tqdm(test_loader, desc="Evaluating"):
        data_batch = model.parent.data_preprocessor(i)
        
        with torch.no_grad():
            outputs = model.predict(data_batch['inputs'], data_batch['data_samples'])
        
        preds = []
        for j in outputs:
            pred_instances = j.pred_instances
            
            # Distance-based OOD detection
            # hyp_distances is the minimum distance to any prototype
            # Higher distance = more likely to be unknown
            if hasattr(pred_instances, 'hyp_distances'):
                pred_instances.ood_score = pred_instances.hyp_distances
                
                # Mark as unknown if distance exceeds threshold
                for k in range(len(pred_instances.ood_score)):
                    if pred_instances.ood_score[k] > args.ood_threshold:
                        pred_instances.labels[k] = unknown_index
            else:
                # Fallback if hyp_distances not available
                pred_instances.ood_score = torch.zeros_like(pred_instances.scores)
            
            # NMS
            keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=0.5)
            pred_instances = pred_instances[keep_idxs]
            preds.append(pred_instances)
        
        evaluator.process_mm(i['data_samples'], preds, unknown_index, use_ood_score=True)
    
    # Run evaluation
    results = evaluator.evaluate()
    
    print(f"\n=== Evaluation Results ===")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    # Optional: Generate Poincaré disk visualization
    if args.visualize:
        try:
            from core.hyperbolic.visualization import visualize_hyperbolic_embeddings
            print(f"\n=== Generating Poincaré Disk Visualization ===")
            
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(args.output_dir, f"{task_name}_{split_name}_poincare.png")
            
            visualize_hyperbolic_embeddings(
                prototypes=model.prototypes.detach().cpu(),
                class_names=class_names + ['unknown'],
                curvature=args.hyp_c,
                save_path=output_path
            )
            print(f"  Saved to: {output_path}")
        except Exception as e:
            print(f"  Warning: Could not generate visualization: {e}")


def compute_ood_statistics(model, test_loader, args):
    """
    Compute OOD score statistics to help tune threshold.
    
    Returns distribution of hyperbolic distances for known vs unknown objects.
    """
    known_distances = []
    unknown_distances = []
    
    model.eval()
    
    for batch in tqdm(test_loader, desc="Computing OOD statistics"):
        data_batch = model.parent.data_preprocessor(batch)
        
        with torch.no_grad():
            outputs = model.predict(data_batch['inputs'], data_batch['data_samples'])
        
        for output in outputs:
            pred = output.pred_instances
            gt = output.gt_instances if hasattr(output, 'gt_instances') else None
            
            if hasattr(pred, 'hyp_distances'):
                # For now, collect all distances
                # In practice, you'd match predictions to GT to separate known/unknown
                known_distances.extend(pred.hyp_distances.cpu().tolist())
    
    known_distances = torch.tensor(known_distances)
    
    print(f"\n=== OOD Distance Statistics ===")
    print(f"  Min: {known_distances.min():.4f}")
    print(f"  Max: {known_distances.max():.4f}")
    print(f"  Mean: {known_distances.mean():.4f}")
    print(f"  Std: {known_distances.std():.4f}")
    print(f"  Median: {known_distances.median():.4f}")
    print(f"  90th percentile: {torch.quantile(known_distances, 0.9):.4f}")
    print(f"  95th percentile: {torch.quantile(known_distances, 0.95):.4f}")
    
    return known_distances
