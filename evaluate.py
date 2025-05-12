#!/usr/bin/env python3
import os
import argparse
import torch
import yaml
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model on test dataset")
    parser.add_argument('--weights', type=str, required=True,
                        help="Path to trained model weights (e.g., runs/yolov8n_idd3/weights/best.pt)")
    parser.add_argument('--data', type=str, default="data/yolo_format/data.yaml",
                        help="Path to dataset yaml file")
    parser.add_argument('--gpu', type=int, default=5, 
                        help="GPU ID to use for evaluation")
    parser.add_argument('--img-size', type=int, default=640,
                        help="Input image size")
    parser.add_argument('--batch-size', type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help="Confidence threshold for detections")
    parser.add_argument('--iou-thres', type=float, default=0.7,
                        help="IoU threshold for NMS")
    parser.add_argument('--output', type=str, default="results",
                        help="Directory to save evaluation results")
    parser.add_argument('--save-visualizations', action='store_true',
                        help="Save detection visualizations")
    parser.add_argument('--max-vis', type=int, default=100,
                        help="Maximum number of visualization images to save")
    return parser.parse_args()

def evaluate_model(model, data_config, img_size=640, batch_size=16, 
                   conf_thres=0.25, iou_thres=0.7, output_dir="results",
                   save_visualizations=False, max_vis=100):
    """Evaluate trained model on test dataset"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run evaluation
    print(f"Running evaluation using data config: {data_config}")
    results = model.val(
        data=data_config,
        imgsz=img_size,
        batch=batch_size,
        conf=conf_thres,
        iou=iou_thres,
        plots=True,
        save_dir=output_dir,
        save_json=True,
        project=output_dir,
        name="test_results22"
    )
    
    # Print metrics
    metrics = results.box
    print("\n--- Performance Metrics ---")
    print(f"mAP@50: {metrics.map50:.3f}")
    print(f"mAP@50-95: {metrics.map:.3f}")
    print(f"Precision: {metrics.mp:.3f}")
    print(f"Recall: {metrics.mr:.3f}")
    
    # Class-wise performance
    print("\n--- Class Performance ---")
    class_names = model.names
    
    print("Class               AP@50    AP@50-95  Precision  Recall   Instances")
    print("-" * 75)
    
    # Access per-class metrics
    ap50_per_class = metrics.ap50
    ap_per_class = metrics.ap
    
    # We need to extract class indices that are present in the results
    for i in range(len(class_names)):
        class_name = class_names[i]
        
        # Find if this class is in the results
        class_indices = np.where(metrics.ap_class_index == i)[0]
        
        if len(class_indices) == 0:
            # No detections for this class
            print(f"{class_name:<18} {'N/A':<8} {'N/A':<9} {'N/A':<10} {'N/A':<8} 0")
            continue
            
        # Get index in the ap array
        class_idx = class_indices[0]
        
        # Get metrics for this class
        ap50 = ap50_per_class[class_idx]
        ap = ap_per_class[class_idx]
        
        # Get precision and recall for this class
        # We'll use the correct index in the metrics arrays
        precision = metrics.p[i] if i < len(metrics.p) else 0
        recall = metrics.r[i] if i < len(metrics.r) else 0
        
        # The actual instance count would come from the dataset
        # Here we'll just indicate "Yes" if ap50 > 0
        has_instances = "Yes" if ap50 > 0 else "No"
        
        print(f"{class_name:<18} {ap50:.3f}    {ap:.3f}     {precision:.3f}     {recall:.3f}    {has_instances}")
    
    # Write full results to CSV
    results_file = os.path.join(output_dir, "test_performance.csv")
    with open(results_file, 'w') as f:
        f.write("Class,AP@50,AP@50-95,Precision,Recall\n")
        
        for i in range(len(class_names)):
            class_name = class_names[i]
            
            # Find if this class is in the results
            class_indices = np.where(metrics.ap_class_index == i)[0]
            
            if len(class_indices) == 0:
                # No detections for this class
                f.write(f"{class_name},0,0,0,0\n")
                continue
                
            # Get index in the ap array
            class_idx = class_indices[0]
            
            # Get metrics for this class
            ap50 = ap50_per_class[class_idx]
            ap = ap_per_class[class_idx]
            
            # Get precision and recall for this class
            precision = metrics.p[i] if i < len(metrics.p) else 0
            recall = metrics.r[i] if i < len(metrics.r) else 0
            
            f.write(f"{class_name},{ap50:.4f},{ap:.4f},{precision:.4f},{recall:.4f}\n")
    
    print(f"\nDetailed results saved to {results_file}")
    return results

def main():
    args = parse_args()
    
    # Set environment variables for GPU selection
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"Using GPU: {args.gpu}")
    
    # Load trained model
    print(f"Loading model from {args.weights}")
    model = YOLO(args.weights)
    
    # Load data configuration
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Run evaluation
    results = evaluate_model(
        model=model,
        data_config=args.data,
        img_size=args.img_size,
        batch_size=args.batch_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        output_dir=args.output,
        save_visualizations=args.save_visualizations,
        max_vis=args.max_vis
    )
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()