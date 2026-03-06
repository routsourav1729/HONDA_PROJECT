"""
Geodesic Prototypical YOLO World Training Script - Multi-GPU DDP Version.

Supports both single-GPU and multi-GPU training.
Launch:
  Single GPU:  python dev_hyp_ddp.py --config-file ... --task ... --ckpt ...
  Multi-GPU:   torchrun --nproc_per_node=N dev_hyp_ddp.py --config-file ... --task ... --ckpt ...
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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


# ============================================================================
# DDP Utilities
# ============================================================================

def setup_ddp():
    """Initialize DDP if launched with torchrun. Returns (rank, world_size, is_distributed)."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        if rank == 0:
            print(f"[DDP] Initialized: {world_size} GPUs, backend=nccl")
        return rank, local_rank, world_size, True
    else:
        return 0, 0, 1, False


def cleanup_ddp():
    """Clean up DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """Check if this is the main process (rank 0)."""
    return rank == 0


def print_rank0(msg, rank=0):
    """Print only on rank 0."""
    if rank == 0:
        print(msg)


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
    
    # Unwrap DDP if needed
    model_to_save = model.module if isinstance(model, DDP) else model
    
    save_dict = {
        'epoch': actual_epoch if actual_epoch is not None else epoch,
        'model_state_dict': model_to_save.state_dict(),
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
    
    # Unwrap DDP if needed
    model_to_load = model.module if isinstance(model, DDP) else model
    
    if 'model_state_dict' in ckpt:
        model_to_load.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start = int(ckpt['epoch']) + 1
        print(f"Resuming from epoch {start}")
    else:
        model_to_load.load_state_dict(ckpt)
        match = re.search(r'model_(\d+)\.pth', path)
        start = int(match.group(1)) + 1 if match else 0
        print(f"Loaded weights, starting from epoch {start}")
    return start


def build_ddp_dataloader(cfgY, is_distributed, world_size, rank):
    """
    Build dataloader. mmengine's DefaultSampler automatically handles DDP:
    it calls get_dist_info() internally to shard data across ranks.
    No custom DistributedSampler needed.
    """
    return Runner.build_dataloader(cfgY.trlder)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--task", default="")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--resume_from", default="")
    parser.add_argument("--exp_name", default="")
    
    # Hyperbolic params
    parser.add_argument("--hyp_c", type=float, default=None)
    parser.add_argument("--hyp_dim", type=int, default=None)
    parser.add_argument("--clip_r", type=float, default=None)
    parser.add_argument("--hyp_loss_weight", type=float, default=None)
    parser.add_argument("--beta_reg", type=float, default=None)
    parser.add_argument("--lambda_sep", type=float, default=None)
    parser.add_argument("--init_protos", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    
    args = parser.parse_args()
    
    # =========================================================================
    # DDP Setup
    # =========================================================================
    rank, local_rank, world_size, is_distributed = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    
    print_rank0(f"Command Line Args: {args}", rank)
    print_rank0(f"Distributed: {is_distributed}, World Size: {world_size}", rank)
    
    cfg = setup(args)
    task_name, split_name = args.task.split('/')
    
    base_task = task_name.replace('_HYP', '')
    if base_task == "nu-OWODB":
        dataset_key = 'nu-prompt'
    elif split_name in ['t2', 't3', 't4']:
        dataset_key = f"{base_task}_T{split_name[1].upper()}"
    else:
        dataset_key = base_task
    
    class_names = list(inital_prompts().get(dataset_key, inital_prompts()[base_task]))
    
    print_rank0(f"\n=== Configuration ===", rank)
    print_rank0(f"Task: {args.task}, Dataset: {dataset_key}", rank)
    print_rank0(f"Classes: {len(class_names)}", rank)
    
    data_split = f"{base_task}/{split_name}"
    data_register = Register('./datasets/', data_split, cfg, dataset_key)
    data_register.register_dataset()

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

    # Initialize YOLO-World (on rank 0, then broadcast weights; or all ranks load same ckpt)
    print_rank0(f"\n=== Initializing YOLO-World ===", rank)
    runner = Runner.from_cfg(cfgY)
    # Strip the EMA hook — it deep-copies the entire XL model (~26 min!)
    # and is unused since we have our own training loop, not Runner.train().
    runner._hooks = [h for h in runner._hooks if not h.__class__.__name__.startswith('EMA')]
    print_rank0(f"  Hooks after EMA removal: {[h.__class__.__name__ for h in runner._hooks]}", rank)
    runner.call_hook("before_run")
    runner.load_or_resume()
    print_rank0(f"  [DEBUG] checkpoint loaded, moving model to device {device}...", rank)
    import sys; sys.stdout.flush()
    runner.model = runner.model.to(device)
    print_rank0(f"  [DEBUG] model on device, calling reparameterize...", rank)
    sys.stdout.flush()
    runner.model.reparameterize([class_names])
    print_rank0(f"  [DEBUG] reparameterize done, setting train mode...", rank)
    sys.stdout.flush()
    runner.model.train()
    print_rank0(f"  [DEBUG] train mode set, building dataloaders...", rank)
    sys.stdout.flush()

    # Build data loaders (with DDP sampler if distributed)
    train_loader = build_ddp_dataloader(cfgY, is_distributed, world_size, rank)
    print_rank0(f"Training loader: {len(train_loader)} batches/GPU (effective batch={cfgY.trlder.get('batch_size', 32) * world_size})", rank)

    # Hyperbolic config
    hyp_cfg = cfgY.get('hyp_config', {})
    hyp_c = args.hyp_c if args.hyp_c is not None else hyp_cfg.get('curvature', 1.0)
    hyp_dim = args.hyp_dim if args.hyp_dim is not None else hyp_cfg.get('embed_dim', 64)
    clip_r = args.clip_r if args.clip_r is not None else hyp_cfg.get('clip_r', 2.0)
    hyp_loss_weight = args.hyp_loss_weight if args.hyp_loss_weight is not None else hyp_cfg.get('hyp_loss_weight', 1.0)
    ce_weight = hyp_cfg.get('ce_weight', 1.0)
    class_balance_smoothing = hyp_cfg.get('class_balance_smoothing', 0.5)
    beta_reg = args.beta_reg if args.beta_reg is not None else hyp_cfg.get('beta_reg', 0.1)
    lambda_sep = args.lambda_sep if args.lambda_sep is not None else hyp_cfg.get('lambda_sep', 1.0)
    sep_margin = hyp_cfg.get('sep_margin', 1.0)
    init_protos_path = args.init_protos if args.init_protos is not None else hyp_cfg.get('init_protos', '')
    prev_ckpt = args.ckpt if args.ckpt else hyp_cfg.get('prev_ckpt', '')
    bi_lipschitz = hyp_cfg.get('bi_lipschitz', False)
    prototype_init_norm = hyp_cfg.get('prototype_init_norm', 0.4)
    prototype_lr = hyp_cfg.get('prototype_lr', 1e-3)
    trainable_prototypes = hyp_cfg.get('trainable_prototypes', True)
    
    print_rank0(f"\n=== Geodesic Prototypical Config ===", rank)
    print_rank0(f"  curvature: {hyp_c}, embed_dim: {hyp_dim}, clip_r: {clip_r}", rank)
    print_rank0(f"  hyp_loss_weight: {hyp_loss_weight}, beta_reg: {beta_reg}, lambda_sep: {lambda_sep}", rank)
    print_rank0(f"  sep_margin: {sep_margin}, prototype_init_norm: {prototype_init_norm}", rank)
    print_rank0(f"  bi_lipschitz: {bi_lipschitz}, prototype_lr: {prototype_lr}", rank)
    
    # Load init prototypes
    init_prototypes = None
    if init_protos_path and os.path.exists(init_protos_path):
        proto_data = torch.load(init_protos_path)
        init_prototypes = proto_data['init_directions']
        print_rank0(f"  Loaded {init_prototypes.shape[0]} prototype directions", rank)
    elif init_protos_path:
        print_rank0(f"ERROR: init_protos not found: {init_protos_path}", rank)
        cleanup_ddp()
        exit(1)

    # Build model
    # For T2+, the classifier only holds NOVEL class prototypes.
    # Base class prototypes are loaded into frozen_prototypes buffer by load_hyp_ckpt.
    prev_cls = cfg.TEST.PREV_INTRODUCED_CLS
    cur_cls = cfg.TEST.CUR_INTRODUCED_CLS
    classifier_num_classes = cur_cls if prev_cls > 0 else unknown_index
    
    print_rank0(f"\n=== Building Geodesic Prototypical Model ===", rank)
    print_rank0(f"  Classifier classes: {classifier_num_classes} ({'novel only' if prev_cls > 0 else 'all'})", rank)
    model = HypCustomYoloWorld(
        runner.model, unknown_index,
        hyp_c=hyp_c, hyp_dim=hyp_dim, clip_r=clip_r,
        num_classifier_classes=classifier_num_classes,
        init_prototypes=init_prototypes,
        trainable_prototypes=trainable_prototypes,
        bi_lipschitz=bi_lipschitz,
        prototype_init_norm=prototype_init_norm,
        ce_weight=ce_weight,
        class_balance_smoothing=class_balance_smoothing,
        beta_reg=beta_reg,
        lambda_sep=lambda_sep,
        sep_margin=sep_margin,
    )
    
    if args.resume_from:
        model = load_hyp_ckpt(model, args.resume_from, cfg.TEST.PREV_INTRODUCED_CLS, cfg.TEST.CUR_INTRODUCED_CLS)
    elif cfg.TEST.PREV_INTRODUCED_CLS > 0:
        ckpt_path = prev_ckpt if prev_ckpt else args.ckpt
        model = load_hyp_ckpt(model, ckpt_path, cfg.TEST.PREV_INTRODUCED_CLS, cfg.TEST.CUR_INTRODUCED_CLS)
    
    model = model.to(device)
    
    # Set trainable parameters (BEFORE wrapping in DDP)
    trainable = ['embeddings']
    for name, param in model.named_parameters():
        param.requires_grad = name in trainable
    model.enable_projector_grad(cfg.TEST.PREV_INTRODUCED_CLS)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print_rank0(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)", rank)
    
    # =========================================================================
    # Wrap in DDP
    # =========================================================================
    if is_distributed:
        # find_unused_parameters=False is safe because:
        #   - All frozen params have requires_grad=False (set before wrapping)
        #   - frozen_prototypes is a register_buffer (not a parameter)
        #   - All trainable params (projector convs, BN, prototypes) participate in every forward
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        print_rank0(f"[DDP] Model wrapped with find_unused_parameters=False", rank)
    
    # Access the underlying model for methods not in nn.Module.forward()
    raw_model = model.module if isinstance(model, DDP) else model
    
    # Optimizer + LR schedule
    # Separate param groups: prototypes get their own LR
    proto_params = []
    other_params = []
    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if 'classifier.prototypes' in name:
            proto_params.append(param)
        else:
            other_params.append(param)
    
    param_groups = [
        {'params': other_params, 'lr': cfgY.base_lr, 'weight_decay': cfgY.weight_decay},
    ]
    if proto_params:
        param_groups.append(
            {'params': proto_params, 'lr': prototype_lr, 'weight_decay': 0.0}
        )
        print_rank0(f"  Prototype LR: {prototype_lr} (separate group, no weight decay)", rank)
    
    PROTO_FREEZE_EPOCHS = 0
    optimizer = optim.AdamW(param_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfgY.max_epochs, eta_min=1e-6)
    print_rank0(f"Optimizer: AdamW, base_LR={cfgY.base_lr}, CosineAnnealing", rank)

    wb = None
    if args.wandb and is_main_process(rank):
        try:
            from wandb_config import WandbLogger
            wb = WandbLogger(
                name=f"{args.task.replace('/', '_')}_{args.exp_name}_ddp{world_size}", 
                config={'lr': cfgY.base_lr, 'c': hyp_c, 'world_size': world_size}
            )
        except Exception as e:
            print(f"WARNING: WandB init failed ({e}), continuing without logging.")
            wb = None

    start_epoch = 0
    if args.resume_from:
        start_epoch = load_checkpoint(model, optimizer, args.resume_from)

    model.train()
    save_dir = os.path.join(args.task, args.exp_name if args.exp_name else "horospherical")
    print_rank0(f"Save directory: {save_dir}", rank)
    
    gs = 0
    
    # =========================================================================
    # Training loop
    # =========================================================================
    for epoch in range(start_epoch, cfgY.max_epochs):
        epoch_start = time.time()
        
        # Set epoch for DefaultSampler (ensures different shuffling each epoch in DDP)
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        print_rank0(f"\n=== Epoch {epoch} ===", rank)

        # Prototype freeze/unfreeze
        if epoch < PROTO_FREEZE_EPOCHS:
            raw_model.hyp_projector.classifier.prototypes.requires_grad_(False)
        elif epoch == PROTO_FREEZE_EPOCHS:
            if trainable_prototypes:
                raw_model.hyp_projector.classifier.prototypes.requires_grad_(True)

        epoch_loss = {'cls': 0, 'dfl': 0, 'bbox': 0, 'geo': 0}
        steps = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_main_process(rank))
        for batch in pbar:
            optimizer.zero_grad()
            data = raw_model.parent.data_preprocessor(batch)
            
            # CRITICAL: Call through the DDP wrapper (model.__call__ → forward()).
            # This ensures DDP's reducer properly sets up gradient sync hooks
            # via prepare_for_backward(). Calling raw_model directly bypasses
            # DDP entirely, breaking gradient AllReduce synchronization.
            head_losses, hyp_loss = model(data['inputs'], data['data_samples'])
            
            loss = (head_losses['loss_cls'] + head_losses['loss_dfl'] + head_losses['loss_bbox'] 
                    + hyp_loss_weight * hyp_loss)
            loss.backward()
            
            epoch_loss['cls'] += head_losses['loss_cls'].item()
            epoch_loss['dfl'] += head_losses['loss_dfl'].item()
            epoch_loss['bbox'] += head_losses['loss_bbox'].item()
            epoch_loss['geo'] += hyp_loss.item()
            
            if steps % 50 == 0 and is_main_process(rank):
                print(f"  step {steps}: cls={head_losses['loss_cls'].item():.4f} "
                      f"bbox={head_losses['loss_bbox'].item():.4f} geo={hyp_loss.item():.4f}")
                if steps % 100 == 0:
                    with torch.no_grad():
                        protos = raw_model.hyp_projector.classifier.prototypes
                        proto_norms = protos.norm(dim=-1)
                        print(f"    [Proto] norms: mean={proto_norms.mean():.4f} "
                              f"min={proto_norms.min():.4f} max={proto_norms.max():.4f}")
                if wb:
                    wb.log({
                        'cls': head_losses['loss_cls'].item(), 
                        'bbox': head_losses['loss_bbox'].item(), 
                        'geo': hyp_loss.item()
                    }, step=gs)
            
            optimizer.step()
            
            # Project prototypes back inside ball after optimizer step
            raw_model.hyp_projector.classifier.project_prototypes_to_ball()
            
            steps += 1
            gs += 1
        
        scheduler.step()

        # Epoch summary (rank 0 only)
        if is_main_process(rank):
            n = max(steps, 1)
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch} done ({epoch_time/60:.1f} min) | "
                  f"Avg: cls={epoch_loss['cls']/n:.4f} bbox={epoch_loss['bbox']/n:.4f} geo={epoch_loss['geo']/n:.4f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save checkpoints (rank 0 only)
            hyp_config_save = {
                'curvature': hyp_c, 'embed_dim': hyp_dim, 'clip_r': clip_r,
                'bi_lipschitz': bi_lipschitz,
                'beta_reg': beta_reg, 'lambda_sep': lambda_sep,
                'sep_margin': sep_margin, 'prototype_init_norm': prototype_init_norm,
                'framework': 'geodesic_prototypical',
            }
            if epoch % 5 == 0:
                save_model(model, optimizer, epoch, save_dir, hyp_config=hyp_config_save)
            save_model(model, optimizer, 'latest', save_dir, actual_epoch=epoch, hyp_config=hyp_config_save)
        
        # Sync all processes before next epoch
        if is_distributed:
            dist.barrier()
    
    # =========================================================================
    # Post-training: calibration (rank 0 only, on unwrapped model)
    # =========================================================================
    if is_main_process(rank):
        print("\n=== Calibrating Adaptive Thresholds ===")
        # Use raw_model for calibration (unwrapped from DDP)
        raw_model.eval()
        
        # Build a NON-distributed calibration loader that sees ALL images.
        # We override batch_size=4 (small to avoid OOM on single-GPU inference),
        # shuffle=False, and ensure no DDP sampler is injected.
        import copy
        cal_cfg = copy.deepcopy(cfgY.trlder)
        cal_cfg['batch_size'] = 4
        cal_cfg['sampler'] = dict(type='DefaultSampler', shuffle=False)
        cal_loader = Runner.build_dataloader(cal_cfg)
        print(f"  Calibration loader: {len(cal_loader)} batches "
              f"(batch_size={cal_cfg['batch_size']}, "
              f"images={len(cal_loader.dataset)})")
        
        # For T2+: preserve T1 base-class calibration stats (computed on the
        # full training set) and only calibrate the NOVEL classes from the
        # few-shot T2 data.  Merging avoids catastrophic stat degradation for
        # base classes (T2 few-shot data has <<100 base-class instances).
        if prev_cls > 0:
            # --- Load T1 calibration stats ---
            t1_ckpt_path = prev_ckpt if prev_ckpt else args.ckpt
            print(f"  [T2+] Loading T1 calibration from: {t1_ckpt_path}")
            t1_ckpt = torch.load(t1_ckpt_path, map_location='cpu')
            t1_stats = t1_ckpt.get('adaptive_stats', None)
            if t1_stats is None:
                print("  WARNING: T1 checkpoint has no adaptive_stats — "
                      "falling back to full recalibration.")
            
            # Calibrate ALL classes from T2 data (labels use global indices
            # so we must pass the full class_names list), but we will only
            # use the novel-class stats in the merge below.
            novel_class_names = class_names[prev_cls:]
            print(f"  [T2+] Novel classes to calibrate: {novel_class_names}")
            print(f"  [T2+] Running calibration pass over T2 few-shot data...")
            t2_stats = calibrate(
                raw_model, cal_loader, class_names,  # full list (global indices)
                dataset_root='./datasets',
                hyp_c=hyp_c,
            )
            
            # Merge: base stats from T1, novel stats from T2
            if t1_stats is not None:
                adaptive_stats = {'per_class': {}, 'alpha': t1_stats.get('alpha', 0.75)}
                base_class_names = class_names[:prev_cls]
                for cls_name in base_class_names:
                    if cls_name in t1_stats['per_class']:
                        adaptive_stats['per_class'][cls_name] = t1_stats['per_class'][cls_name]
                        print(f"  [T2+] BASE  {cls_name:<20s}: PRESERVED from T1 "
                              f"(n={t1_stats['per_class'][cls_name]['count']})")
                    else:
                        adaptive_stats['per_class'][cls_name] = {
                            'mean': 0.0, 'std': 1.0, 'count': 0}
                        print(f"  [T2+] BASE  {cls_name:<20s}: MISSING in T1 (fallback)")
                for cls_name in novel_class_names:
                    if cls_name in t2_stats['per_class'] and t2_stats['per_class'][cls_name]['count'] > 0:
                        adaptive_stats['per_class'][cls_name] = t2_stats['per_class'][cls_name]
                        n = t2_stats['per_class'][cls_name]['count']
                        print(f"  [T2+] NOVEL {cls_name:<20s}: calibrated from T2 (n={n})")
                    else:
                        adaptive_stats['per_class'][cls_name] = {
                            'mean': 0.0, 'std': 1.0, 'count': 0}
                        print(f"  [T2+] NOVEL {cls_name:<20s}: NO samples (fallback)")
            else:
                # No T1 stats available — fall through to full calibration
                adaptive_stats = t2_stats
        else:
            # T1: calibrate all classes normally on the full training set
            adaptive_stats = calibrate(
                raw_model, cal_loader, class_names,
                dataset_root='./datasets',
                hyp_c=hyp_c,
            )
        
        hyp_config_dict = {
            'curvature': hyp_c,
            'embed_dim': hyp_dim,
            'clip_r': clip_r,
            'bi_lipschitz': bi_lipschitz,
            'beta_reg': beta_reg,
            'lambda_sep': lambda_sep,
            'sep_margin': sep_margin,
            'prototype_init_norm': prototype_init_norm,
            'framework': 'geodesic_prototypical',
        }
        
        save_model(model, optimizer, 'final', save_dir,
                   adaptive_stats=adaptive_stats, hyp_config=hyp_config_dict)
        print(f"\n=== Training Complete ({world_size} GPU{'s' if world_size > 1 else ''}) ===")
        if wb:
            wb.finish()
    
    cleanup_ddp()
