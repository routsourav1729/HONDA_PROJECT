#!/usr/bin/env python3
"""
Few-Shot Fine-tuning Script for OVOW on IDD

Usage:
    python finetune_fewshot.py --task IDD/t1 --ckpt IDD/t1/model_final.pth --shots 10 --class car --exp_name 10shot_car
"""

import os
import sys
import argparse
import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

# Add OVOW to path
sys.path.insert(0, '/home/agipml/sourav.rout/ALL_FILES/ovow')

from core import add_config
from core.pascal_voc import register_pascal_voc, inital_prompts
from core.customyoloworld import CustomYoloWorld, load_ckpt
from mmengine.config import Config
from mmengine.runner import Runner
from detectron2.engine import default_argument_parser, default_setup
from detectron2.config import get_cfg

class FewShotFineTuner:
    def __init__(self, args):
        self.args = args
        self.setup_config()
        self.setup_model()
    
    def setup_config(self):
        """Setup detectron2 and mmyolo configs"""
        cfg = get_cfg()
        add_config(cfg)
        
        # Load detectron2 config
        config_file = os.path.join("./configs", self.args.task.split('/')[0], 
                                   self.args.task.split('/')[1] + ".yaml")
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(self.args.opts)
        cfg.freeze()
        default_setup(cfg, self.args)
        self.cfg = cfg
        
        # Load mmyolo config
        task_name = self.args.task.split('/')[0]
        split_name = self.args.task.split('/')[1]
        config_file = os.path.join("./configs", task_name, split_name + ".py")
        cfgY = Config.fromfile(config_file)
        cfgY.work_dir = "."
        cfgY.load_from = None  # Don't load pretrained, we'll load finetuning ckpt
        self.cfgY = cfgY
    
    def setup_model(self):
        """Initialize YOLO-World model"""
        print("=== Initializing YOLO-World for Few-Shot Fine-tuning ===")
        
        # Get class names
        task_name = self.args.task.split('/')[0]
        class_names = list(inital_prompts()[task_name])
        unknown_index = self.cfg.TEST.PREV_INTRODUCED_CLS + self.cfg.TEST.CUR_INTRODUCED_CLS
        class_names = class_names[:unknown_index]
        
        # Create runner
        runner = Runner.from_cfg(self.cfgY)
        runner.call_hook("before_run")
        runner.load_or_resume()
        
        # Move to GPU and reparameterize
        runner.model = runner.model.cuda()
        runner.model.reparameterize([class_names])
        runner.model.train()
        
        # Create custom model
        self.model = CustomYoloWorld(runner.model, unknown_index)
        self.runner = runner
        self.class_names = class_names
        self.unknown_index = unknown_index
        
        print(f"✓ Model initialized for few-shot fine-tuning")
        print(f"  Classes: {len(class_names)}")
        print(f"  Target class: {self.args.target_class}")
        print(f"  Shots: {self.args.shots}")
    
    def create_fewshot_dataloader(self):
        """Create dataloader for few-shot split"""
        print(f"=== Creating {self.args.shots}-shot dataloader ===")
        
        # Register few-shot dataset
        fewshot_anno_dir = f"datasets/FewShot_Annotations/{self.args.shots}shot/{self.args.target_class}"
        
        if not os.path.exists(fewshot_anno_dir):
            raise FileNotFoundError(f"Few-shot annotations not found: {fewshot_anno_dir}")
        
        # Count images
        num_images = len([f for f in os.listdir(fewshot_anno_dir) if f.endswith('.xml')])
        print(f"✓ Found {num_images} few-shot images for {self.args.target_class}")
        
        # For now, we'll use a simple approach: update config to use fewshot annos
        # This requires manual dataloader update (TODO: implement proper registration)
        
        return num_images
    
    def finetune(self, num_epochs=10, learning_rate=1e-5):
        """Fine-tune model on few-shot data"""
        print(f"\n=== Few-Shot Fine-tuning ===")
        print(f"  Epochs: {num_epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Target class: {self.args.target_class}")
        
        # For demonstration, we'll create a minimal training loop
        # In practice, you'd use a proper dataloader
        
        # Set trainable parameters (same as main training)
        for name, param in self.model.named_parameters():
            if 'embeddings' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        self.model.enable_projector_grad(self.cfg.TEST.PREV_INTRODUCED_CLS)
        self.model.cuda()
        
        # Initialize optimizer with lower learning rate for fine-tuning
        optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=self.cfgY.weight_decay
        )
        
        # Load trained checkpoint
        checkpoint = torch.load(self.args.ckpt, map_location='cuda')
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        self.model.train()
        print("✓ Model in training mode")
        print("\nNote: Actual fine-tuning requires proper few-shot dataloader setup")
        print("     Current script demonstrates structure only")

def main():
    parser = argparse.ArgumentParser(description='Few-shot fine-tuning for OVOW')
    parser.add_argument('--task', required=True, help='Task (e.g., IDD/t1)')
    parser.add_argument('--ckpt', required=True, help='Checkpoint to fine-tune from')
    parser.add_argument('--shots', type=int, default=10, help='Number of shots (1, 10, 20, 30)')
    parser.add_argument('--target-class', required=True, help='Class to fine-tune on')
    parser.add_argument('--exp-name', default='fewshot', help='Experiment name')
    parser.add_argument('--opts', nargs='+', default=[], help='Other options')
    args = parser.parse_args()
    
    # Validate shots
    if args.shots not in [1, 10, 20, 30]:
        print(f"✗ Invalid shots: {args.shots}. Must be one of [1, 10, 20, 30]")
        sys.exit(1)
    
    # Create fine-tuner
    finetuner = FewShotFineTuner(args)
    
    # Prepare data
    num_images = finetuner.create_fewshot_dataloader()
    
    # Fine-tune
    finetuner.finetune(num_epochs=10, learning_rate=1e-5)

if __name__ == '__main__':
    main()
