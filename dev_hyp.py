"""
Horospherical YOLO World Training Script.

Training with Busemann function and ideal prototypes on boundary.
Uses HypCustomYoloWorld with horospherical classification.
"""

import os
import torch
import torch.optim as optim
from tqdm import tqdm

from core import DatasetMapper, add_config
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
    
    task_name, split_name = args.task.split('/')
    task_yaml = os.path.join("./configs", task_name, f"{split_name}.yaml")
    if os.path.exists(task_yaml):
        cfg.merge_from_file(task_yaml)
        print(f"Merged task config: {task_yaml}")
    
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def save_model(model, optimizer, epoch, save_dir, file_name="model", actual_epoch=None):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{file_name}_{epoch}.pth")
    torch.save({
        'epoch': actual_epoch if actual_epoch is not None else epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Model saved to {path}")


def load_checkpoint(model, optimizer, path):
    import re
    print(f"Loading checkpoint from {path}")
    ckpt = torch.load(path, map_location='cuda')
    
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start = int(ckpt['epoch']) + 1
        print(f"✓ Resuming from epoch {start}")
    else:
        model.load_state_dict(ckpt)
        match = re.search(r'model_(\d+)\.pth', path)
        start = int(match.group(1)) + 1 if match else 0
        print(f"✓ Loaded weights, starting from epoch {start}")
    return start


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--task", default="")
    parser.add_argument("--ckpt", default="model.pth")
    parser.add_argument("--resume_from", default="")
    parser.add_argument("--exp_name", default="")
    
    # Hyperbolic/Horospherical parameters
    parser.add_argument("--hyp_c", type=float, default=1.0, help="Curvature (ball radius = 1/√c)")
    parser.add_argument("--hyp_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--clip_r", type=float, default=0.95, help="Clip radius for ToPoincare")
    parser.add_argument("--hyp_loss_weight", type=float, default=1.0, help="Horospherical loss weight")
    parser.add_argument("--dispersion_weight", type=float, default=0.1, help="Prototype dispersion loss weight")
    parser.add_argument("--clip_cache", type=str, default=None, 
                        help="Path to CLIP cache for prototype initialization (from compute_clip_cache.py)")
    
    args = parser.parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)

    task_name, split_name = args.task.split('/')
    
    # Dataset key
    if task_name == "nu-OWODB":
        dataset_key = 'nu-prompt'
    elif split_name in ['t2', 't3', 't4']:
        dataset_key = f"{task_name}_T{split_name[1].upper()}"
    else:
        dataset_key = task_name
    
    class_names = list(inital_prompts().get(dataset_key, inital_prompts()[task_name]))
    
    print(f"\n=== Configuration ===")
    print(f"Task: {args.task}, Dataset: {dataset_key}")
    print(f"Classes: {len(class_names)}")
    print(f"Curvature: {args.hyp_c} (ball radius = {1.0 / args.hyp_c**0.5:.2f})")
    
    data_register = Register('./datasets/', args.task, cfg, dataset_key)
    data_register.register_dataset()

    # Model config
    config_file = os.path.join("./configs", task_name, f"{split_name}.py")
    cfgY = Config.fromfile(config_file)
    cfgY.work_dir = "."
    
    if args.resume_from:
        cfgY.load_from = None
        cfgY.resume = False
    elif cfg.TEST.PREV_INTRODUCED_CLS == 0:
        cfgY.load_from = args.ckpt
    
    unknown_index = cfg.TEST.PREV_INTRODUCED_CLS + cfg.TEST.CUR_INTRODUCED_CLS
    class_names = class_names[:unknown_index]

    # Initialize YOLO-World
    print(f"\n=== Initializing YOLO-World ===")
    runner = Runner.from_cfg(cfgY)
    runner.call_hook("before_run")
    runner.load_or_resume()
    runner.model = runner.model.cuda()
    runner.model.reparameterize([class_names])
    runner.model.train()

    # Build data loaders
    train_loader = Runner.build_dataloader(cfgY.trlder)
    print(f"✓ Training loader: {len(train_loader)} batches")

    # Build Horospherical model
    print(f"\n=== Building Horospherical Model ===")
    model = HypCustomYoloWorld(
        runner.model, unknown_index,
        hyp_c=args.hyp_c, hyp_dim=args.hyp_dim, clip_r=args.clip_r
    )
    
    # Initialize prototypes from CLIP cache (if provided)
    if args.clip_cache and os.path.exists(args.clip_cache):
        print(f"Initializing prototypes from CLIP cache: {args.clip_cache}")
        model.init_prototypes_from_clip(args.clip_cache)
    elif args.clip_cache:
        print(f"WARNING: CLIP cache not found: {args.clip_cache}")
        print("  Run: python compute_clip_cache.py --task {t1|t2} first!")
    else:
        print("INFO: No CLIP cache provided, using learned projection for prototypes")
        # Initialize from current CLIP embeddings via learned projection
        model.init_prototypes_from_clip(clip_cache_path=None)
    
    if args.resume_from:
        print(f"Resuming from: {args.resume_from}")
        model = load_hyp_ckpt(model, args.resume_from, cfg.TEST.PREV_INTRODUCED_CLS, cfg.TEST.CUR_INTRODUCED_CLS)
    elif cfg.TEST.PREV_INTRODUCED_CLS > 0:
        print(f"T2+ training: loading {args.ckpt}")
        model = load_hyp_ckpt(model, args.ckpt, cfg.TEST.PREV_INTRODUCED_CLS, cfg.TEST.CUR_INTRODUCED_CLS)
    else:
        print("T1 training: fresh start")
    
    model = model.cuda()
    
    # Set trainable parameters
    trainable = ['embeddings']
    for name, param in model.named_parameters():
        param.requires_grad = name in trainable
    model.enable_projector_grad(cfg.TEST.PREV_INTRODUCED_CLS)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfgY.base_lr, weight_decay=cfgY.weight_decay
    )
    print(f"Optimizer: AdamW, LR={cfgY.base_lr}")

    start_epoch = 0
    if args.resume_from:
        start_epoch = load_checkpoint(model, optimizer, args.resume_from)

    model.train()
    save_dir = os.path.join(args.task, args.exp_name if args.exp_name else "horospherical")
    print(f"\nSave directory: {save_dir}")
    
    # Training loop
    for epoch in range(start_epoch, cfgY.max_epochs):
        print(f"\n=== Epoch {epoch} ===")
        epoch_loss = {'cls': 0, 'dfl': 0, 'bbox': 0, 'horo': 0, 'disp': 0}
        steps = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            data = model.parent.data_preprocessor(batch)
            
            if steps % 100 == 0:
                head_losses, hyp_loss, breakdown = model.head_loss_with_breakdown(data['inputs'], data['data_samples'])
            else:
                head_losses, hyp_loss = model.head_loss(data['inputs'], data['data_samples'])
                breakdown = None
            
            # Add dispersion loss to prevent prototype direction collapse
            disp_loss = model.hyp_projector.classifier.angular_dispersion_loss()
            
            loss = (head_losses['loss_cls'] + head_losses['loss_dfl'] + head_losses['loss_bbox'] 
                    + args.hyp_loss_weight * hyp_loss 
                    + args.dispersion_weight * disp_loss)
            loss.backward()
            
            epoch_loss['cls'] += head_losses['loss_cls'].item()
            epoch_loss['dfl'] += head_losses['loss_dfl'].item()
            epoch_loss['bbox'] += head_losses['loss_bbox'].item()
            epoch_loss['horo'] += hyp_loss.item()
            epoch_loss['disp'] += disp_loss.item()
            
            if steps % 50 == 0:
                print(f"  step {steps}: cls={head_losses['loss_cls'].item():.4f} "
                      f"bbox={head_losses['loss_bbox'].item():.4f} horo={hyp_loss.item():.4f} disp={disp_loss.item():.4f}")
                if breakdown:
                    print(f"    [Proto] bias_mean={breakdown.get('bias_mean', 0):.3f} num_protos={breakdown.get('num_prototypes', 0)}")
            
            optimizer.step()
            steps += 1
        
        # Epoch summary
        n = max(steps, 1)
        print(f"✓ Epoch {epoch} done | Avg: cls={epoch_loss['cls']/n:.4f} bbox={epoch_loss['bbox']/n:.4f} horo={epoch_loss['horo']/n:.4f} disp={epoch_loss['disp']/n:.4f}")
        
        # Save checkpoints
        if epoch % 5 == 0:
            save_model(model, optimizer, epoch, save_dir)
        save_model(model, optimizer, 'latest', save_dir, actual_epoch=epoch)
    
    save_model(model, optimizer, 'final', save_dir)
    print("\n=== Training Complete ===")
