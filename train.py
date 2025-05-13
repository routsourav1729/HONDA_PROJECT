#!/usr/bin/env python3

import os
import argparse

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
    
    return parser

def main():
    """Main training function"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Check available GPUs
    import torch
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0 and args.gpu != []:
        print("Warning: No GPUs available, using CPU instead")
        args.gpu = []
    else:
        # Validate GPU IDs
        for gpu_id in args.gpu:
            if gpu_id >= num_gpus:
                raise ValueError(f"Requested GPU {gpu_id} but only {num_gpus} GPUs available")
        print(f"Training on GPUs: {args.gpu}")
    
    # Dispatch to appropriate trainer
    if args.model == 'yolov8':
        from trainers import YOLOTrainer
        trainer = YOLOTrainer(
            config_path=args.config,
            data_path=args.data,
            output_dir=args.output,
            gpu_ids=args.gpu
        )
        results = trainer.train()
        print("Training completed successfully")
        
    elif args.model == 'faster_rcnn':
        from trainers import FasterRCNNTrainer
        trainer = FasterRCNNTrainer(
            config_path=args.config,
            data_path=args.data,
            output_dir=args.output,
            gpu_ids=args.gpu,
            pretrained=args.pretrained
        )
        results = trainer.train()
        print("Training completed successfully")
        
    else:
        parser.print_help()
        raise ValueError(f"Please specify a valid model type")

if __name__ == "__main__":
    main()