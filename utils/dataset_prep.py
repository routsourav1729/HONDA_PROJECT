"""
Consolidated dataset preparation utilities that merge functionality
from convert_annotations.py, prepare_yolo_xml.py, and prepare_yolo_dataset.py
"""
import os
import yaml
from pathlib import Path
import xml.etree.ElementTree as ET
import shutil
from tqdm import tqdm
import argparse

# Common IDD classes definition
IDD_CLASSES = [
    'rider', 'motorcycle', 'person', 'car', 'truck', 
    'traffic sign', 'vehicle fallback', 'animal', 'autorickshaw', 
    'bus', 'bicycle', 'traffic light', 'caravan', 'train', 'trailer'
]

def create_data_yaml(root_dir, output_dir, dataset_name="idd"):
    """Create YAML configuration file for models"""
    os.makedirs(output_dir, exist_ok=True)
    
    data = {
        'path': str(root_dir),
        'train': str(Path(root_dir) / 'train.txt'),
        'val': str(Path(root_dir) / 'val.txt'),
        'test': str(Path(root_dir) / 'test.txt'),
        'nc': len(IDD_CLASSES),
        'names': IDD_CLASSES
    }
    
    yaml_path = Path(output_dir) / f"{dataset_name}.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"Created {dataset_name}.yaml in {output_dir}")
    return str(yaml_path)

def convert_to_yolo_format(xml_path, img_width, img_height):
    """Convert XML annotation to YOLO format (center_x, center_y, width, height)"""
    class_dict = {name: i for i, name in enumerate(IDD_CLASSES)}
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        yolo_annotations = []
        
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
            center_x = (xmin + xmax) / (2 * img_width)
            center_y = (ymin + ymax) / (2 * img_height)
            bbox_width = (xmax - xmin) / img_width
            bbox_height = (ymax - ymin) / img_height
            
            yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}")
        
        return yolo_annotations
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        return []

def prepare_yolo_dataset(root_dir, output_dir, dataset_format="labels_only"):
    """
    Prepare YOLO dataset with flexible output formats
    
    Args:
        root_dir: Path to original dataset
        output_dir: Path for prepared dataset
        dataset_format: 
            "labels_only" - Only create label files in YOLO format
            "full_structure" - Create complete dataset with images and labels directories
            "txt_files" - Create text files with paths to images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a mapping dictionary for the classes
    class_dict = {name: i for i, name in enumerate(IDD_CLASSES)}
    
    if dataset_format == "full_structure":
        # Create output directories with proper structure
        images_dir = os.path.join(output_dir, "images")
        labels_dir = os.path.join(output_dir, "labels")
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(images_dir, split), exist_ok=True)
            os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
    
    elif dataset_format == "labels_only":
        # Just create output directories for labels
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"Processing {split} split...")
        
        # For txt_files format, create output files
        if dataset_format == "txt_files":
            output_file = os.path.join(output_dir, f"{split}.txt")
            out_f = open(output_file, 'w')
        
        # Read the split file entries
        split_file = Path(root_dir) / f"{split}.txt"
        with open(split_file, 'r') as f:
            entries = [line.strip() for line in f.readlines()]
        
        for entry in tqdm(entries):
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
            
            # Get image dimensions from XML
            try:
                tree = ET.parse(anno_path)
                root = tree.getroot()
                size = root.find('size')
                if size is None:
                    continue
                img_width = float(size.find('width').text)
                img_height = float(size.find('height').text)
                
                # Convert XML to YOLO format
                yolo_annotations = convert_to_yolo_format(anno_path, img_width, img_height)
                
                if not yolo_annotations:
                    continue
                
                # Handle based on format
                if dataset_format == "txt_files":
                    # Write image path to output file
                    out_f.write(f"{os.path.abspath(img_path)}\n")
                
                elif dataset_format == "full_structure":
                    # Copy image to output directory
                    dest_img_path = os.path.join(images_dir, split, f"{base_name}{img_path.suffix}")
                    shutil.copy(img_path, dest_img_path)
                    
                    # Write YOLO format annotations
                    label_path = os.path.join(labels_dir, split, f"{base_name}.txt")
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(yolo_annotations))
                
                elif dataset_format == "labels_only":
                    # Just write YOLO format annotations
                    label_path = os.path.join(output_dir, split, f"{base_name}.txt")
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(yolo_annotations))
                
            except Exception as e:
                print(f"Error processing {anno_path}: {e}")
        
        # Close output file for txt_files format
        if dataset_format == "txt_files":
            out_f.close()
            print(f"Created {output_file}")
    
    # Create YAML configuration
    yaml_path = os.path.join(output_dir, "data.yaml")
    
    if dataset_format == "full_structure":
        data_yaml = {
            'path': os.path.abspath(output_dir),
            'train': os.path.join(os.path.abspath(images_dir), 'train'),
            'val': os.path.join(os.path.abspath(images_dir), 'val'),
            'test': os.path.join(os.path.abspath(images_dir), 'test'),
            'nc': len(IDD_CLASSES),
            'names': IDD_CLASSES
        }
    elif dataset_format == "txt_files":
        data_yaml = {
            'path': os.path.abspath(output_dir),
            'train': os.path.abspath(os.path.join(output_dir, "train.txt")),
            'val': os.path.abspath(os.path.join(output_dir, "val.txt")),
            'test': os.path.abspath(os.path.join(output_dir, "test.txt")),
            'nc': len(IDD_CLASSES),
            'names': IDD_CLASSES
        }
    else:
        data_yaml = {
            'path': os.path.abspath(output_dir),
            'train': os.path.abspath(os.path.join(output_dir, "train")),
            'val': os.path.abspath(os.path.join(output_dir, "val")),
            'test': os.path.abspath(os.path.join(output_dir, "test")),
            'nc': len(IDD_CLASSES),
            'names': IDD_CLASSES
        }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Created data.yaml in {output_dir}")
    return yaml_path

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Prepare dataset for YOLO training")
    parser.add_argument('--dataset', type=str, required=True,
                        help="Path to original dataset directory")
    parser.add_argument('--output', type=str, required=True,
                        help="Output directory for prepared dataset")
    parser.add_argument('--format', type=str, default="full_structure",
                        choices=["labels_only", "full_structure", "txt_files"],
                        help="Format of prepared dataset")
    parser.add_argument('--yaml-dir', type=str, default=None,
                        help="Directory to save YAML config (default: output dir)")
    
    args = parser.parse_args()
    
    # Prepare dataset
    yaml_path = prepare_yolo_dataset(args.dataset, args.output, args.format)
    
    # Create YAML config in separate location if specified
    if args.yaml_dir:
        create_data_yaml(args.output, args.yaml_dir)
    
    print("\nDataset preparation complete!")
    print(f"To train YOLOv8, use: python train.py yolov8 --config configs/models/yolov8.yaml --data {yaml_path}")

if __name__ == "__main__":
    main()