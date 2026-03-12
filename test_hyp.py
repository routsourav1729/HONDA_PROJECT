"""
vMF Hyperspherical YOLO World Evaluation Script.

Inference with vMF distribution-based OOD detection using
adaptive per-prototype thresholds.

Thresholds are loaded directly from the checkpoint (embedded during training).
No external JSON file needed -- the checkpoint is self-contained.

For each detection assigned to prototype k with max_vmf_score < tau_k -> unknown,
where tau_k = mean_k - alpha * std_k  (calibrated from training data).

vMF scores = log Z_d(kappa_c) + kappa_c * mu_c^T * r: higher = more ID, lower = more OOD.
"""

import os
import torch
from tqdm import tqdm
from torchvision.ops import nms, batched_nms

from core import add_config
from core.util.model_ema import add_model_ema_configs
from core.pascal_voc import register_pascal_voc, inital_prompts
from core.hyp_customyoloworld import HypCustomYoloWorld, load_hyp_ckpt
from core.calibrate_thresholds import compute_thresholds
from core.eval_utils import Trainer

from mmengine.config import Config


def soft_nms(pred, iou_threshold=0.5, sigma=0.5, score_threshold=0.001):
    """Gaussian Soft-NMS: decay overlapping scores instead of removing.
    
    For each highest-scoring detection, overlapping detections get their
    score decayed by: score *= exp(-iou^2 / sigma)
    This preserves cross-class overlaps (e.g., rider on motorcycle) while
    still reducing true duplicates.
    """
    bboxes = pred.bboxes
    scores = pred.scores.clone()
    N = len(scores)
    
    if N == 0:
        return pred
    
    # Compute all pairwise IoUs once
    x1 = bboxes[:, 0]; y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]; y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    order = scores.argsort(descending=True)
    
    for i in range(N):
        idx = order[i]
        if scores[idx] < score_threshold:
            continue
        
        # Compute IoU of idx with all remaining
        remaining = order[i+1:]
        if len(remaining) == 0:
            break
        
        xx1 = torch.max(x1[idx], x1[remaining])
        yy1 = torch.max(y1[idx], y1[remaining])
        xx2 = torch.min(x2[idx], x2[remaining])
        yy2 = torch.min(y2[idx], y2[remaining])
        
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        union = areas[idx] + areas[remaining] - inter
        iou = inter / union.clamp(min=1e-6)
        
        # Gaussian decay
        decay = torch.exp(-(iou ** 2) / sigma)
        scores[remaining] *= decay
    
    # Filter by score threshold
    keep = scores >= score_threshold
    pred.scores = scores
    pred = pred[keep]
    return pred
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
    parser.add_argument("--nms_mode", type=str, default="hard",
                        choices=["hard", "soft", "class", "none"],
                        help="NMS mode: hard=class-agnostic@0.5 (ovow default), "
                             "soft=Gaussian soft-NMS, class=per-class NMS, none=skip external NMS")
    parser.add_argument("--nms_iou", type=float, default=0.5,
                        help="IoU threshold for external NMS (default: 0.5, matching ovow)")
    
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

    # Extract hyp_config from checkpoint
    hyp_config = ckpt_data.get('hyp_config', {})
    framework = hyp_config.get('framework', 'vmf_spherical')
    hyp_dim = hyp_config.get('embed_dim', 64)
    bi_lipschitz = hyp_config.get('bi_lipschitz', True)
    kappa_init = hyp_config.get('kappa_init', 10.0)
    ema_alpha = hyp_config.get('ema_alpha', 0.95)
    use_projection_head = hyp_config.get('use_projection_head', True)
    vmf_loss_weight = hyp_config.get('vmf_loss_weight', 1.5)

    print(f"  hyp_config: framework={framework}, dim={hyp_dim}")
    print(f"  kappa_init={kappa_init}, ema_alpha={ema_alpha}")
    print(f"  use_projection_head={use_projection_head}, bi_lipschitz={bi_lipschitz}")

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
    # Strip EMA hook — it deep-copies the entire XL model (~20 min!) and is unused
    runner._hooks = [h for h in runner._hooks if not h.__class__.__name__.startswith('EMA')]
    runner.call_hook("before_run")
    runner.load_or_resume()
    runner.model.reparameterize([class_names])
    runner.model.eval()

    test_loader = Runner.build_dataloader(cfgY.test_dataloader)
    evaluator = Trainer.build_evaluator(cfg, "my_val")
    evaluator.reset()

    # For T2+ eval: classifier holds only novel classes; base in frozen_* buffers
    prev_cls = cfg.TEST.PREV_INTRODUCED_CLS
    cur_cls = cfg.TEST.CUR_INTRODUCED_CLS
    classifier_num_classes = cur_cls if prev_cls > 0 else unknown_index

    model = HypCustomYoloWorld(
        runner.model, unknown_index,
        hyp_dim=hyp_dim,
        num_classifier_classes=classifier_num_classes,
        bi_lipschitz=bi_lipschitz,
        kappa_init=kappa_init,
        ema_alpha=ema_alpha,
        use_projection_head=use_projection_head,
        vmf_loss_weight=vmf_loss_weight,
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
    print(f"Framework: {framework}")

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
    total_dets_after_nms = 0
    total_relabeled = 0
    
    # ---- Evaluation loop ----
    # Pipeline matches ovow test.py: predict → relabel OOD → NMS → evaluate
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
            
            # --- Step 1: OOD relabeling (BEFORE NMS, matching ovow test.py) ---
            # vMF scores = log Z + kappa * cos_sim: higher = more ID, lower = more OOD
            # If max_vmf_score < adaptive_threshold[proto_k] → unknown
            if adaptive_thresholds is not None:
                if hasattr(pred, 'geo_max_scores') and hasattr(pred, 'geo_assigned_proto'):
                    proto_indices = pred.geo_assigned_proto.long()
                    per_det_thresholds = adaptive_thresholds[proto_indices]
                    is_unknown = pred.geo_max_scores < per_det_thresholds
                    pred.labels[is_unknown] = unknown_index
                    total_relabeled += is_unknown.sum().item()
            
            # --- Step 2: NMS (AFTER relabeling, matching ovow test.py) ---
            if args.nms_mode == 'hard':
                # Class-agnostic hard NMS (ovow default)
                keep_idxs = nms(pred.bboxes, pred.scores, iou_threshold=args.nms_iou)
                pred = pred[keep_idxs]
            elif args.nms_mode == 'soft':
                # Gaussian Soft-NMS: decay scores instead of removing
                pred = soft_nms(pred, iou_threshold=args.nms_iou, sigma=0.5, score_threshold=0.001)
            elif args.nms_mode == 'class':
                # Per-class NMS: preserves cross-class overlaps (rider+motorcycle both survive)
                keep_idxs = batched_nms(pred.bboxes, pred.scores, pred.labels, iou_threshold=args.nms_iou)
                pred = pred[keep_idxs]
            # else: nms_mode=='none', skip external NMS entirely
            total_dets_after_nms += len(pred.scores)
            
            preds.append(pred)
        
        evaluator.process_mm(batch['data_samples'], preds, unknown_index, use_ood_score=True)
    
    results = evaluator.evaluate()
    
    strategy_str = f"adaptive (alpha={alpha_used:.2f})" if alpha_used else "NONE"
    print(f"\n{'='*60}")
    print(f"RESULTS — {strategy_str}")
    print(f"{'='*60}")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Framework: {framework}")
    print(f"  NMS mode: {args.nms_mode} (iou={args.nms_iou})")
    print(f"  Total detections (pre-NMS): {total_dets}")
    print(f"  Total detections (post-NMS): {total_dets_after_nms}")
    print(f"  NMS reduction: {(1 - total_dets_after_nms/max(total_dets,1))*100:.1f}%")
    print(f"  Relabeled as unknown: {total_relabeled} ({total_relabeled/max(total_dets,1)*100:.1f}% of pre-NMS)")
    for key, value in results.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}")
