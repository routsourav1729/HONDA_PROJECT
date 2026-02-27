#!/usr/bin/env python
"""
Patch an existing checkpoint with adaptive threshold stats and hyp_config.

Use this for checkpoints saved BEFORE calibration was integrated into training.
Future checkpoints from dev_hyp.py will have this embedded automatically.

Usage:
    python patch_checkpoint.py \
        --ckpt IDD_HYP/t1/horospherical_15epoch/model_final.pth \
        --hyp_c 1.0 --hyp_dim 256 --clip_r 3.0
        
    This will:
    1. Load the model
    2. Run calibration over training data
    3. Re-save the checkpoint with adaptive_stats + hyp_config embedded
"""

import os
import sys
import torch
import argparse

# Add YOLO-World to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'YOLO-World'))

from core import add_config
from core.util.model_ema import add_model_ema_configs
from core.pascal_voc import register_pascal_voc, inital_prompts
from core.hyp_customyoloworld import HypCustomYoloWorld, load_hyp_ckpt
from core.calibrate_thresholds import calibrate

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
        task_yaml = os.path.join("configs", args.task.split('/')[0],
                                 args.task.split('/')[1] + ".yaml")
        if os.path.exists(task_yaml):
            cfg.merge_from_file(task_yaml)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--task", default="IDD_HYP/t1")
    parser.add_argument("--ckpt", required=True, help="Checkpoint to patch")
    parser.add_argument("--hyp_c", type=float, default=1.0)
    parser.add_argument("--hyp_dim", type=int, default=256)
    parser.add_argument("--clip_r", type=float, required=True,
                        help="clip_r used during training (MUST match)")

    args = parser.parse_args()
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

    config_file = os.path.join("./configs", task_name, f"{split_name}.py")
    cfgY = Config.fromfile(config_file)
    cfgY.work_dir = "."

    runner = Runner.from_cfg(cfgY)
    runner.call_hook("before_run")
    runner.load_or_resume()
    runner.model.reparameterize([class_names])
    runner.model.eval()

    # Build train loader for calibration
    train_loader = Runner.build_dataloader(cfgY.trlder)
    print(f"Train loader: {len(train_loader)} batches")

    # Build model
    model = HypCustomYoloWorld(
        runner.model, unknown_index,
        hyp_c=args.hyp_c, hyp_dim=args.hyp_dim, clip_r=args.clip_r
    )
    model = load_hyp_ckpt(model, args.ckpt,
                          cfg.TEST.PREV_INTRODUCED_CLS, cfg.TEST.CUR_INTRODUCED_CLS,
                          eval=True)
    model = model.cuda()
    # Use 9 known classes (before add_generic_text mutation)
    known_class_names = list(class_names)
    model.add_generic_text(class_names, generic_prompt='object', alpha=0.4)
    model.eval()

    # Run calibration
    adaptive_stats = calibrate(
        model, train_loader, known_class_names,
        dataset_root='./datasets',
        hyp_c=args.hyp_c,
    )

    # Load existing checkpoint and patch it
    print(f"\n=== Patching checkpoint: {args.ckpt} ===")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    ckpt['adaptive_stats'] = adaptive_stats
    ckpt['hyp_config'] = {
        'curvature': args.hyp_c,
        'embed_dim': args.hyp_dim,
        'clip_r': args.clip_r,
    }
    torch.save(ckpt, args.ckpt)
    print(f"  Patched with adaptive_stats + hyp_config")
    print(f"  clip_r={args.clip_r}, c={args.hyp_c}, dim={args.hyp_dim}")
    print(f"  Saved to: {args.ckpt}")
    print(f"\nDone! You can now run evaluation with:")
    print(f"  sbatch test_hyp.sbatch")
