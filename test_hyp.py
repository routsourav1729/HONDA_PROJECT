"""
Horospherical YOLO World Evaluation Script.

Inference with Busemann function-based OOD detection.

Supports two OOD strategies:
  1. Global threshold (default):
     ood_score = -max_k(xi_k) > threshold → unknown

  2. Adaptive per-prototype threshold (--adaptive_threshold):
     For each detection, find assigned prototype k = argmax xi_k(x)
     If xi_k(x) < tau_k → unknown
     Where tau_k = train_mean_k - alpha * train_std_k
     Requires JSON from debug/adaptive_threshold_analysis.py
"""

import os
import json
import torch
from tqdm import tqdm

from core import add_config
from core.util.model_ema import add_model_ema_configs
from core.pascal_voc import register_pascal_voc, inital_prompts
from core.hyp_customyoloworld import HypCustomYoloWorld, load_hyp_ckpt
from core.eval_utils import Trainer

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
    
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def load_adaptive_thresholds(json_path, class_names, alpha_override=None):
    """
    Load per-prototype adaptive thresholds from calibration JSON.
    
    Returns a tensor of shape (num_classes,) with per-class thresholds.
    If alpha_override is given, recomputes thresholds from stored mean/std.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    alpha = alpha_override if alpha_override is not None else data['best_alpha']
    cal = data['train_calibration']
    
    thresholds = []
    print(f"\n  Adaptive thresholds (alpha={alpha:.2f}):")
    for cls_name in class_names:
        if cls_name in cal:
            mu = cal[cls_name]['mean']
            std = cal[cls_name]['std']
            tau = mu - alpha * std
            print(f"    {cls_name:<20s}: tau={tau:.4f} (mean={mu:.4f}, std={std:.4f}, n={cal[cls_name]['count']})")
        else:
            # Fallback: very permissive threshold
            tau = -10.0
            print(f"    {cls_name:<20s}: tau={tau:.4f} (NOT IN CALIBRATION, using fallback)")
        thresholds.append(tau)
    
    return torch.tensor(thresholds, dtype=torch.float32), alpha


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--task", default="")
    parser.add_argument("--ckpt", default="model.pth")
    
    # Hyperbolic parameters (must match training)
    parser.add_argument("--hyp_c", type=float, default=1.0, help="Curvature")
    parser.add_argument("--hyp_dim", type=int, default=256, help="Embedding dim")
    parser.add_argument("--clip_r", type=float, default=0.95, help="Clip radius")
    
    # OOD threshold strategy
    # Option 1: Global threshold (legacy)
    parser.add_argument("--ood_threshold", type=float, default=0.0,
                        help="Global OOD threshold: if -max_score > threshold → unknown")
    # Option 2: Adaptive per-prototype threshold (recommended)
    parser.add_argument("--adaptive_threshold", type=str, default="",
                        help="Path to adaptive_stats JSON from calibration script")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Override alpha from JSON (tau_k = mean_k - alpha*std_k)")
    
    args = parser.parse_args()
    print("Args:", args)
    cfg = setup(args)

    task_name, split_name = args.task.split('/')
    base_dataset = task_name.replace('_HYP', '')
    
    if split_name in ['t2', 't3', 't4']:
        dataset_key = f"{base_dataset}_T{split_name[1].upper()}"
    else:
        dataset_key = base_dataset
    
    data_split = f"{base_dataset}/{split_name}"
    data_register = Register('./datasets/', data_split, cfg, dataset_key)
    data_register.register_dataset()

    class_names = list(inital_prompts()[dataset_key])
    unknown_index = cfg.TEST.PREV_INTRODUCED_CLS + cfg.TEST.CUR_INTRODUCED_CLS
    class_names = class_names[:unknown_index]

    # Determine OOD strategy
    use_adaptive = bool(args.adaptive_threshold)
    adaptive_thresholds = None
    adaptive_alpha = None

    print(f"\n=== Evaluation ===")
    print(f"Task: {args.task}, Dataset: {dataset_key}")
    print(f"Classes: {len(class_names)} + 1 (unknown)")
    
    if use_adaptive:
        print(f"OOD Strategy: ADAPTIVE PER-PROTOTYPE")
        print(f"  Calibration: {args.adaptive_threshold}")
        adaptive_thresholds, adaptive_alpha = load_adaptive_thresholds(
            args.adaptive_threshold, class_names, args.alpha
        )
    else:
        print(f"OOD Strategy: GLOBAL (threshold={args.ood_threshold})")

    # Load config and model
    config_file = os.path.join("./configs", task_name, f"{split_name}.py")
    cfgY = Config.fromfile(config_file)
    cfgY.work_dir = "."

    runner = Runner.from_cfg(cfgY)
    runner.call_hook("before_run")
    runner.load_or_resume()
    runner.model.reparameterize([class_names])
    runner.model.eval()

    test_loader = Runner.build_dataloader(cfgY.test_dataloader)
    evaluator = Trainer.build_evaluator(cfg, "my_val")
    evaluator.reset()

    model = HypCustomYoloWorld(
        runner.model, unknown_index,
        hyp_c=args.hyp_c, hyp_dim=args.hyp_dim, clip_r=args.clip_r
    )
    
    model = load_hyp_ckpt(model, args.ckpt, cfg.TEST.PREV_INTRODUCED_CLS, cfg.TEST.CUR_INTRODUCED_CLS, eval=True)
    model = model.cuda()
    model.add_generic_text(class_names, generic_prompt='object', alpha=0.4)
    model.eval()
    
    if use_adaptive:
        adaptive_thresholds = adaptive_thresholds.cuda()
    
    print(f"\nPrototypes: {model.prototypes.shape[0]} (norm={model.prototypes.norm(dim=-1).mean():.3f})")
    
    # Counters for summary
    total_dets = 0
    total_relabeled = 0
    
    # Evaluation loop
    for batch in tqdm(test_loader, desc="Evaluating"):
        data = model.parent.data_preprocessor(batch)
        
        with torch.no_grad():
            outputs = model.predict(data['inputs'], data['data_samples'])
        
        preds = []
        for out in outputs:
            pred = out.pred_instances
            n_dets = len(pred.scores)
            total_dets += n_dets
            
            if n_dets == 0:
                preds.append(pred)
                continue
            
            if use_adaptive:
                # Adaptive per-prototype thresholding
                # pred.horo_max_scores: max horosphere score per detection
                # pred.horo_assigned_proto: which prototype (0..K-1) gave max score
                if hasattr(pred, 'horo_max_scores') and hasattr(pred, 'horo_assigned_proto'):
                    proto_indices = pred.horo_assigned_proto.long()  # (N,)
                    per_det_thresholds = adaptive_thresholds[proto_indices]  # (N,)
                    is_unknown = pred.horo_max_scores < per_det_thresholds
                    pred.labels[is_unknown] = unknown_index
                    total_relabeled += is_unknown.sum().item()
            else:
                # Global threshold (legacy)
                if hasattr(pred, 'ood_scores'):
                    is_unknown = pred.ood_scores > args.ood_threshold
                    pred.labels[is_unknown] = unknown_index
                    total_relabeled += is_unknown.sum().item()
            
            preds.append(pred)
        
        evaluator.process_mm(batch['data_samples'], preds, unknown_index, use_ood_score=True)
    
    results = evaluator.evaluate()
    
    strategy_str = f"adaptive (alpha={adaptive_alpha:.2f})" if use_adaptive else f"global (tau={args.ood_threshold})"
    print(f"\n{'='*60}")
    print(f"RESULTS — {strategy_str}")
    print(f"{'='*60}")
    print(f"  Total detections: {total_dets}")
    print(f"  Relabeled as unknown: {total_relabeled} ({total_relabeled/max(total_dets,1)*100:.1f}%)")
    for key, value in results.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}")
