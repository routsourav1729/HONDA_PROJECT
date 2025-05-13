import os
import argparse
import yaml
import torch
from pathlib import Path

def train_yolov8(args):
    """Train YOLOv8 model"""
    from ultralytics import YOLO
    
    # Handle multiple GPUs
    if isinstance(args.gpu, list):
        # For multiple GPUs, set CUDA_VISIBLE_DEVICES to a comma-separated list
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu))
        # Use all selected GPUs
        device = list(range(len(args.gpu)))
    else:
        # Single GPU case
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = 0
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    model = YOLO(config.get('model', 'yolov8n.pt'))
    
    # Train model
    results = model.train(
        data=args.data,
        epochs=config.get('epochs', 50),
        imgsz=config.get('imgsz', 640),
        batch=config.get('batch_size', 16),
        project=args.output,
        name=config.get('name', 'yolov8_idd'),
        device=device  # Now can be either single GPU or list of GPUs
    )
    
    return results

def create_parser():
    """Create modular argument parser"""
    # First create subparsers for different model types
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
    parser = create_parser()
    args = parser.parse_args()
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    
    # Validate GPU IDs
    if isinstance(args.gpu, list):
        for gpu_id in args.gpu:
            if gpu_id >= num_gpus:
                raise ValueError(f"Requested GPU {gpu_id} but only {num_gpus} GPUs available")
        print(f"Training on GPUs: {args.gpu}")
    else:
        if args.gpu >= num_gpus:
            raise ValueError(f"Requested GPU {args.gpu} but only {num_gpus} GPUs available") 
        print(f"Training on GPU: {args.gpu}")
    
    # Dispatch to appropriate training function
    if args.model == 'yolov8':
        train_yolov8(args)
    elif args.model == 'faster_rcnn':
        raise NotImplementedError("Faster R-CNN training not yet implemented")
    else:
        parser.print_help()
        raise ValueError(f"Please specify a model type")

if __name__ == "__main__":
    main()