"""
Horospherical YOLO World Training Script.

Training with Busemann function and ideal prototypes on boundary.
Uses HypCustomYoloWorld with horospherical classification.
"""

import os
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from core import DatasetMapper, add_config
from core.util.model_ema import add_model_ema_configs
from core.pascal_voc import register_pascal_voc, inital_prompts
from core.hyp_customyoloworld import HypCustomYoloWorld, load_hyp_ckpt
from core.calibrate_thresholds import calibrate
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


def save_model(model, optimizer, epoch, save_dir, file_name="model",
               actual_epoch=None, adaptive_stats=None, hyp_config=None):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{file_name}_{epoch}.pth")
    save_dict = {
        'epoch': actual_epoch if actual_epoch is not None else epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if adaptive_stats is not None:
        save_dict['adaptive_stats'] = adaptive_stats
    if hyp_config is not None:
        save_dict['hyp_config'] = hyp_config
    torch.save(save_dict, path)
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
    parser.add_argument("--ckpt", default="")  # Override config's prev_ckpt if provided
    parser.add_argument("--resume_from", default="")
    parser.add_argument("--exp_name", default="")
    
    # Hyperbolic params can be overridden from command line (but config is preferred)
    parser.add_argument("--hyp_c", type=float, default=None, help="Override curvature from config")
    parser.add_argument("--hyp_dim", type=int, default=None, help="Override embed_dim from config")
    parser.add_argument("--clip_r", type=float, default=None, help="Override clip_r from config")
    parser.add_argument("--hyp_loss_weight", type=float, default=None, help="Override hyp_loss_weight from config")
    parser.add_argument("--dispersion_weight", type=float, default=None, help="Override dispersion_weight from config")
    parser.add_argument("--bias_reg_weight", type=float, default=None, help="Override bias_reg_weight from config")
    parser.add_argument("--compactness_weight", type=float, default=None, help="Override compactness_weight from config")
    parser.add_argument("--init_protos", type=str, default=None, help="Override init_protos from config")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB")
    
    args = parser.parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)

    task_name, split_name = args.task.split('/')
    
    # Dataset key - map IDD_HYP to IDD for dataset registration
    base_task = task_name.replace('_HYP', '')  # IDD_HYP -> IDD
    
    if base_task == "nu-OWODB":
        dataset_key = 'nu-prompt'
    elif split_name in ['t2', 't3', 't4']:
        dataset_key = f"{base_task}_T{split_name[1].upper()}"
    else:
        dataset_key = base_task
    
    class_names = list(inital_prompts().get(dataset_key, inital_prompts()[base_task]))
    
    print(f"\n=== Configuration ===")
    print(f"Task: {args.task}, Dataset: {dataset_key}")
    print(f"Classes: {len(class_names)}")
    
    # Register dataset using base_task path (IDD not IDD_HYP)
    data_split = f"{base_task}/{split_name}"  # e.g., IDD/t1
    data_register = Register('./datasets/', data_split, cfg, dataset_key)
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

    # ==========================================================================
    # Load Hyperbolic Config from cfgY.hyp_config (with CLI overrides)
    # ==========================================================================
    hyp_cfg = cfgY.get('hyp_config', {})
    
    # Get values from config, allow CLI override
    hyp_c = args.hyp_c if args.hyp_c is not None else hyp_cfg.get('curvature', 1.0)
    hyp_dim = args.hyp_dim if args.hyp_dim is not None else hyp_cfg.get('embed_dim', 256)
    clip_r = args.clip_r if args.clip_r is not None else hyp_cfg.get('clip_r', 0.95)
    hyp_loss_weight = args.hyp_loss_weight if args.hyp_loss_weight is not None else hyp_cfg.get('hyp_loss_weight', 1.0)
    dispersion_weight = args.dispersion_weight if args.dispersion_weight is not None else hyp_cfg.get('dispersion_weight', 0.0)
    bias_reg_weight = args.bias_reg_weight if args.bias_reg_weight is not None else hyp_cfg.get('bias_reg_weight', 0.0)
    compactness_weight = args.compactness_weight if args.compactness_weight is not None else hyp_cfg.get('compactness_weight', 0.0)
    init_protos_path = args.init_protos if args.init_protos is not None else hyp_cfg.get('init_protos', '')
    prev_ckpt = args.ckpt if args.ckpt else hyp_cfg.get('prev_ckpt', '')
    
    print(f"\n=== Hyperbolic Config ===")
    print(f"  curvature: {hyp_c}")
    print(f"  embed_dim: {hyp_dim}")
    print(f"  clip_r: {clip_r}")
    print(f"  hyp_loss_weight: {hyp_loss_weight}")
    print(f"  dispersion_weight: {dispersion_weight}")
    print(f"  bias_reg_weight: {bias_reg_weight}")
    print(f"  compactness_weight: {compactness_weight}")
    print(f"  init_protos: {init_protos_path}")
    if cfg.TEST.PREV_INTRODUCED_CLS > 0:
        print(f"  prev_ckpt (T2+): {prev_ckpt}")
    
    # Load init_prototypes
    init_prototypes = None
    if init_protos_path:
        if os.path.exists(init_protos_path):
            print(f"Loading prototype initialization from: {init_protos_path}")
            proto_data = torch.load(init_protos_path)
            init_prototypes = proto_data['init_directions']
            print(f"  Loaded {init_prototypes.shape[0]} prototype directions (dim={init_prototypes.shape[1]})")
        else:
            print(f"ERROR: init_protos file not found: {init_protos_path}")
            print("  Run: python init_prototypes.py --classes '...' --output init_protos_t1.pt")
            exit(1)
    else:
        print("⚠ WARNING: No init_protos in config! Using random initialization.")
        print("  For proper training, add init_protos to hyp_config in configs/IDD/t1.py!")

    # Build Horospherical model
    print(f"\n=== Building Horospherical Model ===")
    model = HypCustomYoloWorld(
        runner.model, unknown_index,
        hyp_c=hyp_c, hyp_dim=hyp_dim, clip_r=clip_r,
        init_prototypes=init_prototypes,
        dispersion_weight=dispersion_weight,
        bias_reg_weight=bias_reg_weight,
        compactness_weight=compactness_weight
    )
    
    if args.resume_from:
        print(f"Resuming from: {args.resume_from}")
        model = load_hyp_ckpt(model, args.resume_from, cfg.TEST.PREV_INTRODUCED_CLS, cfg.TEST.CUR_INTRODUCED_CLS)
    elif cfg.TEST.PREV_INTRODUCED_CLS > 0:
        # T2+: load previous task checkpoint
        ckpt_path = prev_ckpt if prev_ckpt else args.ckpt
        print(f"T2+ training: loading {ckpt_path}")
        model = load_hyp_ckpt(model, ckpt_path, cfg.TEST.PREV_INTRODUCED_CLS, cfg.TEST.CUR_INTRODUCED_CLS)
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
    
    # Optimizer + LR schedule
    PROTO_FREEZE_EPOCHS = 0  # Co-train prototypes from epoch 0 (freeze caused cold-start trap)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfgY.base_lr, weight_decay=cfgY.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfgY.max_epochs, eta_min=1e-6)
    print(f"Optimizer: AdamW, LR={cfgY.base_lr}, CosineAnnealing to 1e-6")
    print(f"Prototype freeze: epochs 0-{PROTO_FREEZE_EPOCHS - 1}")

    wb = None
    if args.wandb:
        from wandb_config import WandbLogger
        wb = WandbLogger(name=f"{args.task.replace('/', '_')}_{args.exp_name}", config={'lr': cfgY.base_lr, 'c': hyp_c})

    start_epoch = 0
    if args.resume_from:
        start_epoch = load_checkpoint(model, optimizer, args.resume_from)

    model.train()
    save_dir = os.path.join(args.task, args.exp_name if args.exp_name else "horospherical")
    print(f"\nSave directory: {save_dir}")
    
    gs = 0
    # Training loop
    for epoch in range(start_epoch, cfgY.max_epochs):
        epoch_start = time.time()
        print(f"\n=== Epoch {epoch} ===")

        # Prototype bias freeze/unfreeze (directions are permanently frozen as buffer)
        if epoch < PROTO_FREEZE_EPOCHS:
            model.hyp_projector.classifier.prototype_bias.requires_grad_(False)
            if epoch == start_epoch:
                print(f"  [Bias FROZEN] epochs 0-{PROTO_FREEZE_EPOCHS - 1} (directions always frozen)")
        elif epoch == PROTO_FREEZE_EPOCHS:
            model.hyp_projector.classifier.prototype_bias.requires_grad_(True)
            print(f"  [Bias UNFROZEN] from epoch {epoch} (directions always frozen)")

        epoch_loss = {'cls': 0, 'dfl': 0, 'bbox': 0, 'horo': 0}
        steps = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            data = model.parent.data_preprocessor(batch)
            
            if steps % 100 == 0:
                head_losses, hyp_loss, breakdown = model.head_loss_with_breakdown(data['inputs'], data['data_samples'])
            else:
                head_losses, hyp_loss = model.head_loss(data['inputs'], data['data_samples'])
                breakdown = None
            
            # hyp_loss already includes dispersion loss internally via HorosphericalLoss
            # No need to add it externally - would be double counting!
            
            loss = (head_losses['loss_cls'] + head_losses['loss_dfl'] + head_losses['loss_bbox'] 
                    + hyp_loss_weight * hyp_loss)
            loss.backward()
            
            epoch_loss['cls'] += head_losses['loss_cls'].item()
            epoch_loss['dfl'] += head_losses['loss_dfl'].item()
            epoch_loss['bbox'] += head_losses['loss_bbox'].item()
            epoch_loss['horo'] += hyp_loss.item()
            
            if steps % 50 == 0:
                print(f"  step {steps}: cls={head_losses['loss_cls'].item():.4f} "
                      f"bbox={head_losses['loss_bbox'].item():.4f} horo={hyp_loss.item():.4f}")
                if breakdown:
                    # Log detailed breakdown for debugging
                    bias_mean = breakdown.get('bias_mean', 0)
                    bias_std = breakdown.get('bias_std', 0)
                    bias_max = breakdown.get('bias_max', 0)
                    ce_loss = breakdown.get('horo_ce_loss', 0)
                    disp_loss = breakdown.get('horo_disp_loss', 0)
                    bias_reg = breakdown.get('horo_bias_reg', 0)
                    compact_loss = breakdown.get('horo_compact_loss', 0)
                    proto_norm = breakdown.get('proto_norm_mean', 0)
                    print(f"    [Proto] bias_mean={bias_mean:.3f} bias_max={bias_max:.3f} bias_std={bias_std:.3f} "
                          f"ce={ce_loss:.4f} disp={disp_loss:.4f} bias_reg={bias_reg:.4f} compact={compact_loss:.4f}")
                if wb:
                    log_dict = {
                        'cls': head_losses['loss_cls'].item(), 
                        'bbox': head_losses['loss_bbox'].item(), 
                        'horo': hyp_loss.item()
                    }
                    if breakdown:
                        log_dict.update({
                            'bias_mean': breakdown.get('bias_mean', 0),
                            'bias_std': breakdown.get('bias_std', 0),
                            'horo_ce': breakdown.get('horo_ce_loss', 0),
                            'horo_disp': breakdown.get('horo_disp_loss', 0),
                            'proto_norm': breakdown.get('proto_norm_mean', 0),
                            'pos_score': breakdown.get('pos_score_mean', 0),
                            'max_score': breakdown.get('max_score_mean', 0),
                        })
                    wb.log(log_dict, step=gs)
            
            # Log gradient stats every 500 steps to monitor learning
            if steps % 500 == 0 and steps > 0:
                bias_param = model.hyp_projector.prototype_bias
                dir_param = model.hyp_projector.prototype_direction
                if bias_param.grad is not None:
                    bias_grad_norm = bias_param.grad.norm().item()
                    dir_grad_norm = dir_param.grad.norm().item() if dir_param.grad is not None else 0
                    print(f"    [Grads] bias_grad={bias_grad_norm:.6f} dir_grad={dir_grad_norm:.6f}")
            
            optimizer.step()
            steps += 1
            gs += 1
        
        # LR schedule step
        scheduler.step()

        # Epoch summary
        n = max(steps, 1)
        epoch_time = time.time() - epoch_start
        print(f"✓ Epoch {epoch} done ({epoch_time/60:.1f} min) | Avg: cls={epoch_loss['cls']/n:.4f} bbox={epoch_loss['bbox']/n:.4f} horo={epoch_loss['horo']/n:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoints
        if epoch % 5 == 0:
            save_model(model, optimizer, epoch, save_dir)
        save_model(model, optimizer, 'latest', save_dir, actual_epoch=epoch)
    
    # =========================================================================
    # Post-training: calibrate adaptive thresholds (model already in GPU)
    # =========================================================================
    print("\n=== Calibrating Adaptive Thresholds (post-training) ===")
    adaptive_stats = calibrate(
        model, train_loader, class_names,
        dataset_root='./datasets',
        hyp_c=hyp_c,
    )

    # Store hyp config so test_hyp.py can auto-configure
    hyp_config_dict = {
        'curvature': hyp_c,
        'embed_dim': hyp_dim,
        'clip_r': clip_r,
    }

    # Save final checkpoint WITH embedded thresholds + config
    save_model(model, optimizer, 'final', save_dir,
               adaptive_stats=adaptive_stats, hyp_config=hyp_config_dict)
    print("\n=== Training Complete ===")
    print(f"  Adaptive thresholds embedded in model_final.pth")
    print(f"  hyp_config: c={hyp_c}, dim={hyp_dim}, clip_r={clip_r}")
    if wb:
        wb.finish()
