import os
import itertools
import weakref
from typing import Any, Dict, List, Set

import torch
import torch.optim as optim
from fvcore.nn.precise_bn import get_bn_modules


from core import DatasetMapper, add_config
from core.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer
from core.pascal_voc import register_pascal_voc, inital_prompts
from core.pascal_voc_evaluation import PascalVOCDetectionEvaluator
from core.customyoloworld import CustomYoloWorld, load_ckpt
from core.eval_utils import Trainer

from mmengine.config import Config
from mmengine.runner import Runner
from torchvision.ops import nms, batched_nms
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup
from detectron2.config import get_cfg




from tqdm import tqdm


class Register:
    def __init__(self, dataset_root, split, cfg, dataset_key=None):
        self.dataset_root = dataset_root
        self.super_split = split.split('/')[0]
        self.cfg = cfg
        # Use dataset_key if provided, otherwise use super_split
        self.dataset_key = dataset_key if dataset_key is not None else self.super_split

        self.PREDEFINED_SPLITS_DATASET = {
            "my_train": split,
            "my_val": os.path.join(self.super_split, 'test')
        }

    def register_dataset(self):
        """
        purpose: register all splits of datasets with PREDEFINED_SPLITS_DATASET
        """
        for name, split in self.PREDEFINED_SPLITS_DATASET.items():
            register_pascal_voc(name, self.dataset_root, self.dataset_key, split, self.cfg)



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg




# Function to save the model with optimizer state
def save_model(model, optimizer, epoch, save_dir='checkpoints', file_name="model", actual_epoch=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, f"{file_name}_{epoch}.pth")
    # Use actual_epoch if provided (for model_latest.pth), otherwise use epoch
    epoch_to_save = actual_epoch if actual_epoch is not None else epoch
    checkpoint = {
        'epoch': epoch_to_save,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, file_path)
    print(f"Model saved to {file_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load checkpoint and return the epoch to resume from"""
    import re
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    # Handle both old format (state_dict only) and new format (full checkpoint)
    if 'model_state_dict' in checkpoint:
        # New format with optimizer state
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        # Ensure epoch is integer
        if not isinstance(epoch, int):
            epoch = int(epoch)
        start_epoch = epoch + 1
        print(f"✓ Resuming from epoch {start_epoch} (loaded checkpoint from epoch {epoch})")
    else:
        # Old format - just model weights
        model.load_state_dict(checkpoint)
        # Try to extract epoch from filename
        match = re.search(r'model_(\d+)\.pth', checkpoint_path)
        if match:
            start_epoch = int(match.group(1)) + 1
            print(f"✓ Loaded model weights from epoch {int(match.group(1))}, resuming from epoch {start_epoch}")
            print(f"⚠ Warning: Optimizer state not found in checkpoint - optimizer will restart")
        else:
            start_epoch = 0
            print(f"⚠ Loaded model weights only (old format), couldn't determine epoch from filename")
    
    return start_epoch


if __name__ == "__main__":
    parser0 = default_argument_parser()
    parser0.add_argument("--task", default="")
    parser0.add_argument("--ckpt", default="model.pth")
    parser0.add_argument("--resume_from", default="", help="Path to checkpoint to resume from")
    parser0.add_argument("--exp_name", default="", help="Experiment name for separate save directory")
    args = parser0.parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)

    # For T2/T3/T4, we need to use the full class list (base + novel + unknown)
    # Extract task components
    task_name = args.task.split('/')[0]
    split_name = args.task.split('/')[1]
    
    # Determine dataset key for registration (T2/T3/T4 need full class lists)
    if task_name == "nu-OWODB":
        dataset_key = 'nu-prompt'
        class_names = list(inital_prompts()[dataset_key])
    elif split_name in ['t2', 't3', 't4']:
        # For incremental tasks, use the full class list (e.g., IDD_T2 for T2)
        dataset_key = f"{task_name}_T{split_name[1].upper()}"
        class_names = list(inital_prompts()[dataset_key])
    else:
        # For T1 or other tasks, use base task name
        dataset_key = task_name
        class_names = list(inital_prompts()[task_name])
    
    print(f"\n=== Dataset Configuration ===")
    print(f"Task: {args.task}")
    print(f"Dataset key: {dataset_key}")
    print(f"Total classes in dataset: {len(class_names)}")
    print(f"Classes: {class_names}")
    
    # Register dataset with correct class list
    data_register = Register('./datasets/', args.task, cfg, dataset_key)
    data_register.register_dataset()

    # model's config
    config_file = os.path.join("./configs", task_name, split_name + ".py")
    cfgY = Config.fromfile(config_file)
    cfgY.work_dir = "."
    
    # Check if resuming from checkpoint
    if args.resume_from:
        # When resuming, don't load the pretrained checkpoint
        cfgY.load_from = None
        cfgY.resume = False  # We handle resume manually below
        print(f"Resume mode: Will load checkpoint from {args.resume_from}")
    else:
        # Fresh training - load pretrained checkpoint
        if cfg.TEST.PREV_INTRODUCED_CLS == 0:
            cfgY.load_from = args.ckpt
    
    unknown_index = cfg.TEST.PREV_INTRODUCED_CLS + cfg.TEST.CUR_INTRODUCED_CLS


    class_names = class_names[:unknown_index]
    classnames = [class_names]

    

    print(f"=== Initializing YOLO-World model ===")
    runner = Runner.from_cfg(cfgY)
    print(f"✓ Runner created")
    runner.call_hook("before_run")
    print(f"✓ Before-run hooks called")
    runner.load_or_resume()
    print(f"✓ Checkpoint loaded")
    
    # CRITICAL: Move model to GPU BEFORE reparameterization for fast CLIP encoding
    print(f"=== Moving model to GPU ===")
    runner.model = runner.model.cuda()
    print(f"✓ Model on GPU: {next(runner.model.parameters()).device}")
    
    print(f"=== Reparameterizing model with {len(class_names)} classes ===")
    print(f"  (Computing CLIP text embeddings on GPU - this should be fast)")
    runner.model.reparameterize(classnames)
    print(f"✓ Model reparameterized")
    runner.model.train()
    print(f"✓ Model set to training mode")

    print(f"=== Building data loaders ===")
    train_loader = Runner.build_dataloader(cfgY.trlder)
    print(f"✓ Training loader built ({len(train_loader)} batches)")
    test_loader = Runner.build_dataloader(cfgY.test_dataloader)
    print(f"✓ Test loader built")

    evaluator = Trainer.build_evaluator(cfg,"my_val")
    evaluator.reset()

    # ============================================================================
    # FEW-SHOT FINE-TUNING: Freeze base, train novel
    # ============================================================================
    # - frozen_embeddings: base class embeddings from T1 (FROZEN)
    # - embeddings: novel class embeddings (TRAINABLE)
    # - projectors[0:prev_cls]: base class projectors (FROZEN)
    # - projectors[prev_cls:]: novel class projectors (TRAINABLE)
    # ============================================================================
    trainable = ['embeddings']
    model = CustomYoloWorld(runner.model,unknown_index)
    
    # Determine which checkpoint to load from
    if args.resume_from:
        # Resuming training - skip initial checkpoint load
        print(f"=== RESUMING TRAINING ===")
        model = model.cuda()
    else:
        # Fresh training - load pretrained checkpoint
        # load_ckpt splits embeddings into frozen_embeddings (base) and embeddings (novel)
        print(f"=== STARTING FRESH TRAINING ===")
        print(f"  Base classes (frozen): {cfg.TEST.PREV_INTRODUCED_CLS}")
        print(f"  Novel classes (trainable): {cfg.TEST.CUR_INTRODUCED_CLS}")
        with torch.no_grad():
            model = load_ckpt(model, args.ckpt,cfg.TEST.PREV_INTRODUCED_CLS,cfg.TEST.CUR_INTRODUCED_CLS)
        model = model.cuda()
    
    # Set trainable parameters - ONLY novel class embeddings
    # All other parameters (backbone, neck, head, base embeddings) are FROZEN
    for name, param in model.named_parameters():
        if name in trainable:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    
    # Enable gradients for novel class projectors only
    model.enable_projector_grad(cfg.TEST.PREV_INTRODUCED_CLS)
    
    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Initialize optimizer with only trainable parameters
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=cfgY.base_lr, 
        weight_decay=cfgY.weight_decay
    )
    print(f"  Optimizer LR: {cfgY.base_lr}, Weight Decay: {cfgY.weight_decay}")

    # Load checkpoint if resuming (must be AFTER optimizer initialization)
    start_epoch = 0
    if args.resume_from:
        start_epoch = load_checkpoint(model, optimizer, args.resume_from)

    model.train()
    
    # Determine save directory
    if args.exp_name:
        save_dir = os.path.join(args.task, args.exp_name)
    else:
        save_dir = args.task
    
    print(f"Models will be saved to: {save_dir}")
    
    import sys
    for epoch in range(start_epoch, cfgY.max_epochs):
        print(f"Epoch: {epoch}", flush=True)
        step = 0
        print(f"  Starting data iteration...", flush=True)
        for i in train_loader:
            if step == 0:
                print(f"  First batch received!", flush=True)
            optimizer.zero_grad()
            data_batch = model.parent.data_preprocessor(i)
            loss1,loss2 = model.head_loss(data_batch['inputs'],data_batch['data_samples'])
            loss = loss1['loss_cls'] + loss1['loss_dfl'] + loss1['loss_bbox'] + loss2
            loss.backward()
            if step%20 == 0: #20 changed to one for quick setup, make sure its 20 not 1
                print(f'step {step}: cls={loss1["loss_cls"].item():.4f} dfl={loss1["loss_dfl"].item():.4f} bbox={loss1["loss_bbox"].item():.4f} contr={loss2.item():.4f}', flush=True)
            optimizer.step()
            step += 1
        
        print(f"✓ Epoch {epoch} completed ({step} steps)", flush=True)
        
        if epoch%5 == 0:
            save_model(model, optimizer, epoch, save_dir=save_dir)
        #each epoch save the latest model (with actual epoch number for resuming)
        save_model(model, optimizer, 'latest', save_dir=save_dir, actual_epoch=epoch)
    #save the final model
    save_model(model, optimizer, 'final', save_dir=save_dir)