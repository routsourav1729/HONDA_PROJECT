"""
Horospherical YOLO World Evaluation Script.

Inference with Busemann function-based OOD detection using
adaptive per-prototype thresholds.

Thresholds are loaded directly from the checkpoint (embedded during training).
No external JSON file needed — the checkpoint is self-contained.

For each detection assigned to prototype k with max_score < tau_k → unknown,
where tau_k = mean_k - alpha * std_k  (calibrated from training data).
"""

import os
import torch
from tqdm import tqdm

from core import add_config
from core.util.model_ema import add_model_ema_configs
from core.pascal_voc import register_pascal_voc, inital_prompts
from core.hyp_customyoloworld import HypCustomYoloWorld, load_hyp_ckpt
from core.calibrate_thresholds import compute_thresholds
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


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--task", default="")
    parser.add_argument("--ckpt", default="model.pth")
    
    # Alpha override (optional — defaults to value stored in checkpoint)
    parser.add_argument("--alpha", type=float, default=None,
                        help="Override alpha (tau_k = mean_k - alpha*std_k). "
                             "Default: use value from checkpoint calibration.")
    
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

    # ---- Load checkpoint and extract embedded config + thresholds ----
    print(f"\n=== Loading checkpoint: {args.ckpt} ===")
    ckpt_data = torch.load(args.ckpt, map_location='cpu')

    # Extract hyp_config (clip_r, curvature, etc.) from checkpoint
    hyp_config = ckpt_data.get('hyp_config', {})
    hyp_c = hyp_config.get('curvature', 1.0)
    hyp_dim = hyp_config.get('embed_dim', 256)
    clip_r = hyp_config.get('clip_r', 0.95)

    print(f"  hyp_config from checkpoint: c={hyp_c}, dim={hyp_dim}, clip_r={clip_r}")

    # Extract adaptive_stats
    adaptive_stats = ckpt_data.get('adaptive_stats', None)
    if adaptive_stats is None:
        print("\n  WARNING: No adaptive_stats in checkpoint!")
        print("  This checkpoint was saved before calibration was integrated.")
        print("  Re-train or run calibration separately to embed thresholds.")
        print("  Proceeding WITHOUT adaptive thresholding (no unknowns will be detected).\n")

    # ---- Build model ----
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
        hyp_c=hyp_c, hyp_dim=hyp_dim, clip_r=clip_r
    )
    
    model = load_hyp_ckpt(model, args.ckpt,
                          cfg.TEST.PREV_INTRODUCED_CLS, cfg.TEST.CUR_INTRODUCED_CLS,
                          eval=True)
    model = model.cuda()
    model.add_generic_text(class_names, generic_prompt='object', alpha=0.4)
    model.eval()

    print(f"\n=== Evaluation ===")
    print(f"Task: {args.task}, Dataset: {dataset_key}")
    print(f"Classes: {len(class_names)} + 1 (unknown)")
    print(f"Prototypes: {model.prototypes.shape[0]} (norm={model.prototypes.norm(dim=-1).mean():.3f})")
    print(f"clip_r: {clip_r} (from checkpoint)")

    # Compute adaptive thresholds
    adaptive_thresholds = None
    alpha_used = None
    if adaptive_stats is not None:
        adaptive_thresholds, alpha_used = compute_thresholds(
            adaptive_stats, class_names[:unknown_index], alpha=args.alpha
        )
        adaptive_thresholds = adaptive_thresholds.cuda()
        print(f"\nOOD Strategy: ADAPTIVE PER-PROTOTYPE (alpha={alpha_used:.2f})")
    else:
        print(f"\nOOD Strategy: NONE (no calibration data in checkpoint)")

    # Counters
    total_dets = 0
    total_relabeled = 0
    
    # ---- Evaluation loop ----
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
            
            if adaptive_thresholds is not None:
                if hasattr(pred, 'horo_max_scores') and hasattr(pred, 'horo_assigned_proto'):
                    proto_indices = pred.horo_assigned_proto.long()
                    per_det_thresholds = adaptive_thresholds[proto_indices]
                    is_unknown = pred.horo_max_scores < per_det_thresholds
                    pred.labels[is_unknown] = unknown_index
                    total_relabeled += is_unknown.sum().item()
            
            preds.append(pred)
        
        evaluator.process_mm(batch['data_samples'], preds, unknown_index, use_ood_score=True)
    
    results = evaluator.evaluate()
    
    strategy_str = f"adaptive (alpha={alpha_used:.2f})" if alpha_used else "NONE"
    print(f"\n{'='*60}")
    print(f"RESULTS — {strategy_str}")
    print(f"{'='*60}")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  clip_r: {clip_r} (from checkpoint)")
    print(f"  Total detections: {total_dets}")
    print(f"  Relabeled as unknown: {total_relabeled} ({total_relabeled/max(total_dets,1)*100:.1f}%)")
    for key, value in results.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}")
