#!/usr/bin/env python3
"""
Training script with improved multi-GPU support and error handling
"""
import os
import argparse
import torch
import yaml
from pathlib import Path

def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(description="Train object detection models")
    subparsers = parser.add_subparsers(dest='model', help='Model type')
    
    # YOLOv8 subparser
    yolo_parser = subparsers.add_parser('yolov8', help='Train YOLOv8 model')
    yolo_parser.add_argument('--gpu', type=int, nargs='+', default=[0], 
                           help="GPU ID(s) to use for training. For multiple GPUs, use: --gpu 0 1")
    yolo_parser.add_argument('--output', type=str, default="runs", 
                           help="Output directory for training results")
    yolo_parser.add_argument('--config', type=str, required=True,
                           help="Path to YOLOv8 model config")
    yolo_parser.add_argument('--data', type=str, required=True,
                           help="Path to dataset config")
    yolo_parser.add_argument('--resume', action='store_true',
                           help="Resume training from last checkpoint")
    
    # Faster R-CNN subparser
    rcnn_parser = subparsers.add_parser('faster_rcnn', help='Train Faster R-CNN model')
    rcnn_parser.add_argument('--gpu', type=int, nargs='+', default=[0],
                           help="GPU ID(s) to use for training. For multiple GPUs, use: --gpu 0 1")
    rcnn_parser.add_argument('--output', type=str, default="runs", 
                           help="Output directory for training results")
    rcnn_parser.add_argument('--config', type=str, required=True,
                           help="Path to Faster R-CNN model config")
    rcnn_parser.add_argument('--data', type=str, required=True, 
                           help="Path to dataset config")
    rcnn_parser.add_argument('--pretrained', action='store_true',
                           help="Use pretrained backbone")
    rcnn_parser.add_argument('--resume', action='store_true',
                           help="Resume training from last checkpoint")
    
    return parser

def validate_gpus(gpu_ids):
    """Validate GPU IDs and return available ones"""
    if not gpu_ids:
        return []
        
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("Warning: No GPUs available, using CPU instead")
        return []
    
    # Validate GPU IDs
    valid_gpu_ids = []
    for gpu_id in gpu_ids:
        if gpu_id >= num_gpus:
            print(f"Warning: Requested GPU {gpu_id} but only {num_gpus} GPUs available. Skipping.")
        else:
            valid_gpu_ids.append(gpu_id)
    
    if not valid_gpu_ids:
        print(f"Warning: None of the requested GPUs {gpu_ids} are available. Using CPU instead.")
        return []
        
    return valid_gpu_ids

def main():
    """Main training function with improved error handling and multi-GPU support"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.model:
        parser.print_help()
        return
    
    # Validate and setup GPUs
    valid_gpu_ids = validate_gpus(args.gpu)
    
    if valid_gpu_ids:
        print(f"Using {'GPU' if len(valid_gpu_ids) == 1 else 'GPUs'}: {valid_gpu_ids}")
    else:
        print("Using CPU for training")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Dispatch to appropriate trainer
        if args.model == 'yolov8':
            from trainers import YOLOTrainer
            
            # Load model config to get name for resume handling
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            
            # Initialize trainer
            trainer = YOLOTrainer(
                config_path=args.config,
                data_path=args.data,
                output_dir=args.output,
                gpu_ids=valid_gpu_ids
            )
            
            # Handle resume if requested
            if args.resume:
                run_name = config.get('name', 'yolov8_run')
                weights_dir = os.path.join(args.output, run_name, 'weights')
                last_weights = os.path.join(weights_dir, 'last.pt')
                
                if os.path.exists(last_weights):
                    print(f"Resuming training from {last_weights}")
                    trainer.load_model(last_weights)
            
            # Run training
            results = trainer.train()
            
            if results:
                print("Training completed successfully")
                print(f"Best model saved to {args.output}/{config.get('name', 'yolov8_run')}/weights/best.pt")
            
        elif args.model == 'faster_rcnn':
            from trainers import FasterRCNNTrainer
            
            trainer = FasterRCNNTrainer(
                config_path=args.config,
                data_path=args.data,
                output_dir=args.output,
                gpu_ids=valid_gpu_ids,
                pretrained=args.pretrained
            )
            
            # Handle resume
            if args.resume:
                with open(args.config, 'r') as f:
                    config = yaml.safe_load(f)
                run_name = config.get('name', 'faster_rcnn_run')
                last_weights = os.path.join(args.output, run_name, 'checkpoints', 'last.pth')
                
                if os.path.exists(last_weights):
                    print(f"Resuming training from {last_weights}")
                    trainer.load_model(last_weights)
            
            results = trainer.train()
            print("Training completed successfully")
            
        else:
            parser.print_help()
            print(f"Unsupported model type: {args.model}")
            return
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()