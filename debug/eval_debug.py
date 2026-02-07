"""
Debug Evaluation Script - Test hyperbolic distances and MSCAL interaction.

This script:
1. Logs actual hyperbolic distance values 
2. Can run evaluation with/without OOD relabeling to isolate issues
3. Prints detailed statistics to understand what's happening
"""

import os
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
import numpy as np

from core import DatasetMapper, add_config
from core.util.model_ema import add_model_ema_configs
from core.pascal_voc import register_pascal_voc, inital_prompts
from core.pascal_voc_evaluation import PascalVOCDetectionEvaluator
from core.hyp_customyoloworld import HypCustomYoloWorld, load_hyp_ckpt
from core.eval_utils import Trainer
from core.hyperbolic import dist_matrix

from mmengine.config import Config
from mmengine.runner import Runner
from torchvision.ops import nms
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


def compute_distance_stats(model, test_loader, num_batches=20):
    """Compute detailed statistics about hyperbolic distances."""
    print("\n" + "="*60)
    print("COMPUTING DISTANCE STATISTICS")
    print("="*60)
    
    all_min_distances = []
    all_scores = []
    all_labels_before_ood = []
    
    model.eval()
    
    for i, batch in enumerate(tqdm(test_loader, desc="Computing distances")):
        if i >= num_batches:
            break
            
        data_batch = model.parent.data_preprocessor(batch)
        
        with torch.no_grad():
            # Get FPN features and hyperbolic embeddings
            img_feats, txt_feats, hyp_embeddings = model.extract_feat(
                data_batch['inputs'], data_batch['data_samples']
            )
            
            # Get predictions (before OOD relabeling)
            outputs = model.predict(data_batch['inputs'], data_batch['data_samples'])
            
            for b_idx, output in enumerate(outputs):
                pred = output.pred_instances
                if len(pred.scores) == 0:
                    continue
                
                # Collect stats
                if hasattr(pred, 'hyp_distances'):
                    all_min_distances.extend(pred.hyp_distances.cpu().tolist())
                    all_scores.extend(pred.scores.cpu().tolist())
                    all_labels_before_ood.extend(pred.labels.cpu().tolist())
    
    if len(all_min_distances) == 0:
        print("WARNING: No predictions collected!")
        return
    
    distances = np.array(all_min_distances)
    scores = np.array(all_scores)
    labels = np.array(all_labels_before_ood)
    
    print(f"\nTotal predictions analyzed: {len(distances)}")
    print(f"\n--- Distance Statistics ---")
    print(f"  Min distance: {distances.min():.4f}")
    print(f"  Max distance: {distances.max():.4f}")
    print(f"  Mean distance: {distances.mean():.4f}")
    print(f"  Std distance: {distances.std():.4f}")
    print(f"  Percentiles: 10%={np.percentile(distances, 10):.4f}, "
          f"50%={np.percentile(distances, 50):.4f}, "
          f"90%={np.percentile(distances, 90):.4f}")
    
    print(f"\n--- Score Statistics ---")
    print(f"  Min score: {scores.min():.4f}")
    print(f"  Max score: {scores.max():.4f}")
    print(f"  Mean score: {scores.mean():.4f}")
    
    print(f"\n--- Label Distribution (BEFORE OOD relabeling) ---")
    unique, counts = np.unique(labels, return_counts=True)
    for lbl, cnt in zip(unique, counts):
        print(f"  Label {lbl}: {cnt} predictions")
    
    # Suggest thresholds
    print(f"\n--- Suggested OOD Thresholds ---")
    for p in [50, 70, 80, 90, 95]:
        thresh = np.percentile(distances, p)
        print(f"  {p}th percentile: {thresh:.4f}")
    
    return distances, scores, labels


def eval_with_ood_mode(model, test_loader, evaluator, unknown_index, ood_threshold, mode='full'):
    """
    Run evaluation with different OOD modes.
    
    mode:
        'full' - Apply OOD relabeling (default behavior)
        'no_ood' - Don't relabel anything as unknown (test pure YOLO)
        'all_ood' - Label everything as unknown (sanity check)
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION MODE: {mode.upper()}")
    print(f"OOD Threshold: {ood_threshold}")
    print(f"{'='*60}")
    
    evaluator.reset()
    model.eval()
    
    total_preds = 0
    total_ood = 0
    
    for batch in tqdm(test_loader, desc=f"Evaluating ({mode})"):
        data_batch = model.parent.data_preprocessor(batch)
        
        with torch.no_grad():
            outputs = model.predict(data_batch['inputs'], data_batch['data_samples'])
        
        preds = []
        for output in outputs:
            pred = output.pred_instances
            
            if mode == 'full':
                # Normal OOD detection
                if hasattr(pred, 'hyp_distances'):
                    for k in range(len(pred.hyp_distances)):
                        if pred.hyp_distances[k] > ood_threshold:
                            pred.labels[k] = unknown_index
                            total_ood += 1
                    pred.ood_score = pred.hyp_distances
                else:
                    pred.ood_score = torch.zeros_like(pred.scores)
                    
            elif mode == 'no_ood':
                # Don't change any labels - test pure YOLO predictions
                if hasattr(pred, 'hyp_distances'):
                    pred.ood_score = pred.hyp_distances
                else:
                    pred.ood_score = torch.zeros_like(pred.scores)
                    
            elif mode == 'all_ood':
                # Mark everything as unknown
                pred.labels[:] = unknown_index
                pred.ood_score = torch.ones_like(pred.scores)
                total_ood += len(pred.labels)
            
            total_preds += len(pred.labels)
            
            # NMS
            keep_idxs = nms(pred.bboxes, pred.scores, iou_threshold=0.5)
            pred = pred[keep_idxs]
            preds.append(pred)
        
        evaluator.process_mm(batch['data_samples'], preds, unknown_index, use_ood_score=True)
    
    print(f"\nTotal predictions: {total_preds}")
    print(f"Total marked as OOD: {total_ood} ({100*total_ood/max(1,total_preds):.1f}%)")
    
    results = evaluator.evaluate()
    return results


if __name__ == "__main__":
    parser0 = default_argument_parser()
    parser0.add_argument("--task", default="IDD/t1")
    parser0.add_argument("--ckpt", default="IDD/t1/hyperbolic/model_0.pth")
    parser0.add_argument("--hyp_c", type=float, default=0.1)
    parser0.add_argument("--hyp_dim", type=int, default=256)
    parser0.add_argument("--clip_r", type=float, default=2.3)
    parser0.add_argument("--temperature", type=float, default=0.1)
    parser0.add_argument("--ood_threshold", type=float, default=2.5)
    parser0.add_argument("--mode", choices=['stats', 'compare', 'full'], default='stats',
                        help="stats: just compute distances; compare: run all modes; full: normal eval")
    
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
    print(f"  Classes: {len(class_names)} known + 1 unknown")
    print(f"  OOD threshold: {args.ood_threshold}")
    print(f"  Mode: {args.mode}")

    # Initialize YOLO-World
    runner = Runner.from_cfg(cfgY)
    runner.call_hook("before_run")
    runner.load_or_resume()
    runner.model.reparameterize(classnames)
    runner.model.eval()

    test_loader = Runner.build_dataloader(cfgY.test_dataloader)
    evaluator = Trainer.build_evaluator(cfg, "my_val")

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
    
    print(f"\n=== Model Info ===")
    print(f"  Prototypes shape: {model.prototypes.shape}")
    print(f"  Prototype norms: min={model.prototypes.norm(dim=-1).min():.4f}, "
          f"max={model.prototypes.norm(dim=-1).max():.4f}, "
          f"mean={model.prototypes.norm(dim=-1).mean():.4f}")
    
    if args.mode == 'stats':
        # Just compute and print distance statistics
        compute_distance_stats(model, test_loader, num_batches=30)
        
    elif args.mode == 'compare':
        # Compare different OOD modes
        print("\n" + "#"*60)
        print("# COMPARISON: WITH vs WITHOUT OOD RELABELING")
        print("#"*60)
        
        # Mode 1: No OOD relabeling (pure YOLO predictions)
        results_no_ood = eval_with_ood_mode(
            model, test_loader, evaluator, unknown_index, 
            args.ood_threshold, mode='no_ood'
        )
        print("\nResults WITHOUT OOD relabeling:")
        for k, v in results_no_ood.items():
            print(f"  {k}: {v}")
        
        # Mode 2: With OOD relabeling
        results_full = eval_with_ood_mode(
            model, test_loader, evaluator, unknown_index,
            args.ood_threshold, mode='full'
        )
        print("\nResults WITH OOD relabeling (threshold={:.2f}):".format(args.ood_threshold))
        for k, v in results_full.items():
            print(f"  {k}: {v}")
            
    elif args.mode == 'full':
        # Normal full evaluation
        results = eval_with_ood_mode(
            model, test_loader, evaluator, unknown_index,
            args.ood_threshold, mode='full'
        )
        print("\n=== Final Results ===")
        for k, v in results.items():
            print(f"  {k}: {v}")
