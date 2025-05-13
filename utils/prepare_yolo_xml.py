import os
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml
from tqdm import tqdm
import shutil

def convert_annotation_to_yolo(xml_path, img_width, img_height, class_dict):
    """Convert XML annotation to YOLO format"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        yolo_lines = []
        
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
            
            # Convert to YOLO format: center_x, center_y, width, height (normalized)
            center_x = (xmin + xmax) / (2 * img_width)
            center_y = (ymin + ymax) / (2 * img_height)
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        
        return yolo_lines
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        return []

def prepare_yolo_dataset(root_dir, output_dir):
    """Create a YOLO-compatible dataset with proper directory structure"""
    # Define class mapping
    classes = [
        'rider', 'motorcycle', 'person', 'car', 'truck', 
        'traffic sign', 'vehicle fallback', 'animal', 'autorickshaw', 
        'bus', 'bicycle', 'traffic light', 'caravan', 'train', 'trailer'
    ]
    class_dict = {name: i for i, name in enumerate(classes)}
    
    # Create output directories
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # For each split, create subdirectories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
        
        split_file = os.path.join(root_dir, f"{split}.txt")
        with open(split_file, 'r') as f:
            entries = [line.strip() for line in f.readlines()]
        
        print(f"Processing {split} split ({len(entries)} entries)...")
        for entry in tqdm(entries):
            # Parse entry path
            entry_path = Path(entry)
            img_dir = Path(root_dir) / "JPEGImages" / entry_path.parent
            anno_dir = Path(root_dir) / "Annotations" / entry_path.parent
            
            base_name = entry_path.name
            anno_path = anno_dir / f"{base_name}.xml"
            
            if not anno_path.exists():
                continue
            
            # Find corresponding image file
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                test_path = img_dir / f"{base_name}{ext}"
                if test_path.exists():
                    img_path = test_path
                    break
            
            if img_path is None:
                continue
            
            # Copy image to output directory
            dest_img_path = os.path.join(images_dir, split, f"{base_name}{img_path.suffix}")
            shutil.copy(img_path, dest_img_path)
            
            # Extract image dimensions from XML
            try:
                tree = ET.parse(anno_path)
                root = tree.getroot()
                size = root.find('size')
                if size is None:
                    continue
                img_width = float(size.find('width').text)
                img_height = float(size.find('height').text)
                
                # Convert XML to YOLO format
                yolo_lines = convert_annotation_to_yolo(anno_path, img_width, img_height, class_dict)
                
                # Write YOLO format labels
                if yolo_lines:
                    label_path = os.path.join(labels_dir, split, f"{base_name}.txt")
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(yolo_lines))
            except Exception as e:
                print(f"Error processing {anno_path}: {e}")
    
    # Create YAML configuration
    yaml_path = os.path.join(output_dir, "data.yaml")
    data_yaml = {
        'path': os.path.abspath(output_dir),
        'train': os.path.join(os.path.abspath(images_dir), 'train'),
        'val': os.path.join(os.path.abspath(images_dir), 'val'),
        'test': os.path.join(os.path.abspath(images_dir), 'test'),
        'nc': len(classes),
        'names': classes
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Created YOLO dataset in {output_dir}")
    return yaml_path

def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset for IDD")
    parser.add_argument('--dataset', type=str, 
                        default="/raid/biplab/souravr/honda/detection_benchmark/data/IDD_Detection",
                        help="Path to IDD_Detection dataset")
    parser.add_argument('--output', type=str, 
                        default="data/yolo_format",
                        help="Output directory for YOLO dataset")
    args = parser.parse_args()
    
    yaml_path = prepare_yolo_dataset(args.dataset, args.output)
    
    print("\nDataset preparation complete!")
    print(f"To train YOLOv8, use: python train.py --gpu 4 yolov8 --config configs/models/yolov8.yaml --data {yaml_path}")

if __name__ == "__main__":
    main()