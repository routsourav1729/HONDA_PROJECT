#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm

def create_yolo_dataset_files(root_dir, output_dir):
    """Generate proper YOLO-compatible dataset files with full image paths"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a mapping dictionary for the classes
    classes = [
        'rider', 'motorcycle', 'person', 'car', 'truck', 
        'traffic sign', 'vehicle fallback', 'animal', 'autorickshaw', 
        'bus', 'bicycle', 'traffic light', 'caravan', 'train', 'trailer'
    ]
    
    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"Processing {split} split...")
        split_file = Path(root_dir) / f"{split}.txt"
        output_file = Path(output_dir) / f"{split}.txt"
        
        with open(split_file, 'r') as f:
            entries = [line.strip() for line in f.readlines()]
        
        # Look for corresponding image files and write full paths
        with open(output_file, 'w') as out_f:
            for entry in tqdm(entries):
                entry_path = Path(entry)
                img_dir = Path(root_dir) / "JPEGImages" / entry_path.parent
                base_name = entry_path.name
                
                # Search for image with common extensions
                for ext in ['.jpg', '.jpeg', '.png']:
                    img_path = img_dir / f"{base_name}{ext}"
                    if img_path.exists():
                        # Write absolute path to output file
                        out_f.write(f"{os.path.abspath(img_path)}\n")
                        break
                        
        print(f"Created {output_file} for {split} split")
    
    # Create YAML configuration
    yaml_path = Path(output_dir) / "data.yaml"
    data_yaml = {
        'path': os.path.abspath(output_dir),
        'train': os.path.abspath(Path(output_dir) / "train.txt"),
        'val': os.path.abspath(Path(output_dir) / "val.txt"),
        'test': os.path.abspath(Path(output_dir) / "test.txt"),
        'nc': len(classes),
        'names': classes
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Created YOLO-compatible dataset configuration at {yaml_path}")
    return str(yaml_path)

def main():
    parser = argparse.ArgumentParser(description="Create YOLO-compatible dataset files")
    parser.add_argument('--dataset', type=str, default="/raid/biplab/souravr/honda/dataset/IDD_Detection",
                        help="Path to IDD_Detection dataset")
    parser.add_argument('--output', type=str, default="data/yolo_dataset",
                        help="Output directory for YOLO dataset files")
    args = parser.parse_args()
    
    yaml_path = create_yolo_dataset_files(args.dataset, args.output)
    
    print("\nDataset preparation complete!")
    print(f"To train YOLOv8, use: python train.py --gpu 4 yolov8 --config configs/models/yolov8.yaml --data {yaml_path}")

if __name__ == "__main__":
    main()