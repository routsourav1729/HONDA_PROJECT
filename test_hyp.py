"""
Horospherical YOLO World Evaluation Script.

Inference with Busemann function-based OOD detection.
Objects with low max horosphere score → classified as unknown.
"""

import os
import torch
from tqdm import tqdm
from torchvision.ops import nms

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


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--task", default="")
    parser.add_argument("--ckpt", default="model.pth")
    
    # Hyperbolic parameters (must match training)
    parser.add_argument("--hyp_c", type=float, default=1.0, help="Curvature")
    parser.add_argument("--hyp_dim", type=int, default=256, help="Embedding dim")
    parser.add_argument("--clip_r", type=float, default=0.95, help="Clip radius")
    
    # OOD threshold: ood_score = -max(horosphere_scores)
    # Higher ood_score = farther from all prototypes = more likely OOD
    # If ood_score > threshold → classify as unknown
    parser.add_argument("--ood_threshold", type=float, default=0.0,
                        help="OOD threshold: if ood_score > threshold → unknown")
    
    args = parser.parse_args()
    print("Args:", args)
    cfg = setup(args)

    task_name, split_name = args.task.split('/')
    
    # Handle IDD_HYP -> IDD for dataset registration
    # Config is in IDD_HYP but dataset is registered as IDD
    base_dataset = task_name.replace('_HYP', '')
    
    # Dataset key
    if split_name in ['t2', 't3', 't4']:
        dataset_key = f"{base_dataset}_T{split_name[1].upper()}"
    else:
        dataset_key = base_dataset
    
    print(f"\n=== Evaluation ===")
    print(f"Task: {args.task}, Dataset: {dataset_key}")
    
    # Use base_dataset path for data registration
    data_split = f"{base_dataset}/{split_name}"
    data_register = Register('./datasets/', data_split, cfg, dataset_key)
    data_register.register_dataset()

    class_names = list(inital_prompts()[dataset_key])
    unknown_index = cfg.TEST.PREV_INTRODUCED_CLS + cfg.TEST.CUR_INTRODUCED_CLS
    class_names = class_names[:unknown_index]

    print(f"Classes: {len(class_names)} + 1 (unknown)")
    print(f"Curvature: {args.hyp_c}, OOD threshold: {args.ood_threshold}")

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

    # Build model
    model = HypCustomYoloWorld(
        runner.model, unknown_index,
        hyp_c=args.hyp_c, hyp_dim=args.hyp_dim, clip_r=args.clip_r
    )
    
    model = load_hyp_ckpt(model, args.ckpt, cfg.TEST.PREV_INTRODUCED_CLS, cfg.TEST.CUR_INTRODUCED_CLS, eval=True)
    model = model.cuda()
    model.add_generic_text(class_names, generic_prompt='object', alpha=0.4)
    model.eval()
    
    print(f"Prototypes: {model.prototypes.shape[0]} (on boundary, norm={model.prototypes.norm(dim=-1).mean():.3f})")
    
    # Evaluation loop
    for batch in tqdm(test_loader, desc="Evaluating"):
        data = model.parent.data_preprocessor(batch)
        
        with torch.no_grad():
            outputs = model.predict(data['inputs'], data['data_samples'])
        
        preds = []
        for out in outputs:
            pred = out.pred_instances
            
            # OOD detection via horosphere scores
            # ood_scores = -max_k(horosphere_score) → higher = more OOD
            # Detection: ood_score > threshold → unknown
            if hasattr(pred, 'ood_scores'):
                for k in range(len(pred.ood_scores)):
                    if pred.ood_scores[k] > args.ood_threshold:
                        pred.labels[k] = unknown_index
            
            # NMS
            keep = nms(pred.bboxes, pred.scores, iou_threshold=0.5)
            preds.append(pred[keep])
        
        evaluator.process_mm(batch['data_samples'], preds, unknown_index, use_ood_score=True)
    
    results = evaluator.evaluate()
    
    print(f"\n=== Results ===")
    for key, value in results.items():
        print(f"  {key}: {value}")
