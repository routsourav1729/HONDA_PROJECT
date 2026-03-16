"""Compute GPM bases from a trained T1 model. Run once, use for all T2 runs."""

import os
import argparse
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from detectron2.engine import default_argument_parser, default_setup
from detectron2.config import get_cfg

from core import add_config
from core.util.model_ema import add_model_ema_configs
from core.pascal_voc import register_pascal_voc, inital_prompts
from core.hyp_customyoloworld import HypCustomYoloWorld, load_hyp_ckpt
from core.gpm import compute_gpm_bases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="T1 model_final.pth")
    parser.add_argument("--output", default=None, help="Output path (default: same dir as ckpt)")
    parser.add_argument("--threshold", type=float, default=0.97)
    parser.add_argument("--max_batches", type=int, default=20)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.ckpt), "gpm_bases.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load T1 config
    cfg = get_cfg()
    add_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file("configs/IDD_HYP/base.yaml")
    cfg.merge_from_file("configs/IDD_HYP/t1.yaml")
    cfg.freeze()

    # Register dataset
    register_pascal_voc("my_train", "./datasets/", "IDD/t1", cfg)
    class_names = list(inital_prompts().get("IDD", {}))
    unknown_index = cfg.TEST.CUR_INTRODUCED_CLS  # 8

    # Build YOLO-World
    cfgY = Config.fromfile("configs/IDD_HYP/t1.py")
    cfgY.work_dir = "."
    cfgY.load_from = args.ckpt
    runner = Runner.from_cfg(cfgY)
    runner._hooks = [h for h in runner._hooks if not h.__class__.__name__.startswith("EMA")]
    runner.call_hook("before_run")
    runner.load_or_resume()
    runner.model = runner.model.to(device)
    runner.model.reparameterize([class_names[:unknown_index]])
    runner.model.eval()

    # Build hyp model
    hyp_cfg = cfgY.get("hyp_config", {})
    model = HypCustomYoloWorld(
        runner.model, unknown_index,
        hyp_dim=hyp_cfg.get("embed_dim", 64),
        num_classifier_classes=unknown_index,
        bi_lipschitz=hyp_cfg.get("bi_lipschitz", True),
        kappa_init=hyp_cfg.get("kappa_init", 10.0),
        ema_alpha=hyp_cfg.get("ema_alpha", 0.95),
        use_projection_head=hyp_cfg.get("use_projection_head", True),
    )
    model = load_hyp_ckpt(model, args.ckpt, 0, unknown_index)
    model = model.to(device)
    model.eval()

    # Build T1 dataloader
    train_loader = Runner.build_dataloader(cfgY.trlder)
    print(f"T1 dataloader: {len(train_loader)} batches")

    # Compute and save
    gpm_bases = compute_gpm_bases(
        model, train_loader,
        threshold=args.threshold,
        max_batches=args.max_batches,
        device=device,
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(gpm_bases, args.output)
    print(f"\nSaved GPM bases to {args.output}")


if __name__ == "__main__":
    main()
