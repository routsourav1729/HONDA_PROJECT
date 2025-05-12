import os
import yaml
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET
import argparse

def create_data_yaml(root_dir, output_dir):
    """Create YAML configuration file for models"""
    classes = [
        'rider', 'motorcycle', 'person', 'car', 'truck', 
        'traffic sign', 'vehicle fallback', 'animal', 'autorickshaw', 
        'bus', 'bicycle', 'traffic light', 'caravan', 'train', 'trailer'
    ]
    
    data = {
        'path': str(root_dir),
        'train': str(Path(root_dir) / 'train.txt'),
        'val': str(Path(root_dir) / 'val.txt'),
        'test': str(Path(root_dir) / 'test.txt'),
        'nc': len(classes),
        'names': classes
    }
    
    with open(Path(output_dir) / 'idd.yaml', 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"Created idd.yaml in {output_dir}")

def convert_to_yolo_format(root_dir, output_dir, split='train'):
    """Convert XML annotations to YOLO format"""
    classes = [
        'rider', 'motorcycle', 'person', 'car', 'truck', 
        'traffic sign', 'vehicle fallback', 'animal', 'autorickshaw', 
        'bus', 'bicycle', 'traffic light', 'caravan', 'train', 'trailer'
    ]
    class_dict = {name: i for i, name in enumerate(classes)}
    
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    
    split_file = Path(root_dir) / f"{split}.txt"
    with open(split_file, 'r') as f:
        entries = [line.strip() for line in f.readlines()]
    
    for entry in tqdm(entries, desc=f"Converting {split}"):
        entry_path = Path(entry)
        anno_path = Path(root_dir) / "Annotations" / entry_path.parent / f"{entry_path.name}.xml"
        
        if not anno_path.exists():
            continue
            
        try:
            tree = ET.parse(anno_path)
            root = tree.getroot()
            
            # Get image dimensions
            size = root.find('size')
            if size is None:
                continue
                
            width = float(size.find('width').text)
            height = float(size.find('height').text)
            
            output_file = os.path.join(split_dir, f"{entry_path.name}.txt")
            with open(output_file, 'w') as f:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    if class_name not in class_dict:
                        continue
                        
                    class_id = class_dict[class_name]
                    
                    bbox = obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                    
                    # Convert to YOLO format
                    center_x = (xmin + xmax) / (2 * width)
                    center_y = (ymin + ymax) / (2 * height)
                    bbox_width = (xmax - xmin) / width
                    bbox_height = (ymax - ymin) / height
                    
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
        except Exception as e:
            print(f"Error processing {anno_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert IDD annotations to YOLO format")
    parser.add_argument('--dataset', type=str, default="/raid/biplab/souravr/honda/dataset/IDD_Detection",
                        help="Path to IDD_Detection dataset")
    parser.add_argument('--output', type=str, default="data/labels_yolo",
                        help="Output directory for YOLO format labels")
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    
    # Create dataset YAML
    create_data_yaml(args.dataset, "configs/datasets")
    
    # Convert annotations
    for split in ['train', 'val', 'test']:
        convert_to_yolo_format(args.dataset, args.output, split)
    
    print("Conversion complete. Labels are ready for training.")
    
if __name__ == "__main__":
    main()