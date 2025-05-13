#!/usr/bin/env python3
import os
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

# IDD Classes
IDD_CLASSES = [
    'rider', 'motorcycle', 'person', 'car', 'truck', 
    'traffic sign', 'vehicle fallback', 'animal', 'autorickshaw', 
    'bus', 'bicycle', 'traffic light', 'caravan', 'train', 'trailer'
]

def convert_idd_to_coco(dataset_dir, output_dir):
    """Convert IDD dataset XML annotations to COCO format JSONs"""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process train and val splits
    for split in ['train', 'val']:
        print(f"Processing {split} split...")
        
        # Read relative paths from split file
        split_file = dataset_dir / f"{split}.txt"
        with open(split_file, 'r') as f:
            rel_paths = [line.strip() for line in f.readlines()]
        
        # Initialize COCO format data
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Add categories
        for i, name in enumerate(IDD_CLASSES):
            coco_data['categories'].append({
                'id': i + 1,  # COCO categories start from 1
                'name': name,
                'supercategory': 'none'
            })
        
        # Process each entry
        ann_id = 1
        for img_id, rel_path in enumerate(tqdm(rel_paths, desc=f"Converting {split}")):
            # Construct paths to XML and image files
            xml_path = dataset_dir / "Annotations" / f"{rel_path}.xml"
            
            if not xml_path.exists():
                continue
                
            # Parse XML file
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # Get image dimensions
                size = root.find('size')
                if size is None:
                    continue
                width = int(size.find('width').text)
                height = int(size.find('height').text)
                
                # Add image to COCO format
                coco_data['images'].append({
                    'id': img_id + 1,  # COCO image ids start from 1
                    'file_name': f"{rel_path}",  # Store relative path
                    'width': width,
                    'height': height
                })
                
                # Process annotations
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    if class_name not in IDD_CLASSES:
                        continue
                    
                    class_id = IDD_CLASSES.index(class_name)
                    
                    # Extract bounding box
                    bbox = obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                    
                    # Convert to COCO format [x, y, width, height]
                    width = xmax - xmin
                    height = ymax - ymin
                    
                    # Add annotation
                    coco_data['annotations'].append({
                        'id': ann_id,
                        'image_id': img_id + 1,
                        'category_id': class_id + 1,  # COCO categories start from 1
                        'bbox': [xmin, ymin, width, height],
                        'area': width * height,
                        'iscrowd': 0
                    })
                    
                    ann_id += 1
                    
            except Exception as e:
                print(f"Error processing {xml_path}: {e}")
        
        # Save COCO JSON
        output_file = output_dir / f"instances_{split}.json"
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"Created {output_file}")
        print(f"  - Images: {len(coco_data['images'])}")
        print(f"  - Annotations: {len(coco_data['annotations'])}")
        print(f"  - Categories: {len(coco_data['categories'])}")

def main():
    parser = argparse.ArgumentParser(description="Convert IDD XML annotations to COCO format JSON")
    parser.add_argument('--dataset', type=str, default="/raid/biplab/souravr/honda/detection_benchmark/data/IDD_Detection",
                        help="Path to IDD_Detection dataset")
    parser.add_argument('--output', type=str, default="annotations",
                        help="Output directory for COCO format JSONs")
    
    args = parser.parse_args()
    convert_idd_to_coco(args.dataset, args.output)

if __name__ == "__main__":
    main()