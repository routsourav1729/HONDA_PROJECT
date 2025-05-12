import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

class IDDDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=640, transform=None):
        """
        Indian Driving Dataset loader
        
        Args:
            root_dir: Root directory of IDD_Detection
            split: 'train', 'val' or 'test'
            img_size: Target image size
            transform: Optional transforms
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.transform = transform
        
        # Define class mapping
        self.classes = [
            'rider', 'motorcycle', 'person', 'car', 'truck', 
            'traffic sign', 'vehicle fallback', 'animal', 'autorickshaw', 
            'bus', 'bicycle', 'traffic light', 'caravan', 'train', 'trailer'
        ]
        self.class_dict = {name: i for i, name in enumerate(self.classes)}
        
        # Parse split file
        self.img_paths = []
        self.anno_paths = []
        
        split_file = self.root_dir / f"{split}.txt"
        with open(split_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            entry = line.strip()
            img_dir = self.root_dir / "JPEGImages" / Path(entry).parent
            anno_dir = self.root_dir / "Annotations" / Path(entry).parent
            
            base_name = Path(entry).name
            anno_path = anno_dir / f"{base_name}.xml"
            
            if not anno_path.exists():
                continue  # Skip if annotation doesn't exist
            
            # Find image file
            for ext in ['.jpg', '.jpeg', '.png']:
                img_path = img_dir / f"{base_name}{ext}"
                if img_path.exists():
                    self.img_paths.append(img_path)
                    self.anno_paths.append(anno_path)
                    break
        
        print(f"Loaded {len(self.img_paths)} images for {split} split")
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        anno_path = self.anno_paths[idx]
        
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get dimensions
        height, width, _ = img.shape
        
        # Parse XML
        tree = ET.parse(anno_path)
        root = tree.getroot()
        
        boxes = []
        class_ids = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in self.class_dict:
                continue
                
            class_id = self.class_dict[class_name]
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Convert to required format (depends on model)
            # For YOLO: normalized center, width, height
            center_x = (xmin + xmax) / (2 * width)
            center_y = (ymin + ymax) / (2 * height)
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height
            
            boxes.append([center_x, center_y, bbox_width, bbox_height])
            class_ids.append(class_id)
            
        boxes = torch.tensor(boxes, dtype=torch.float32)
        class_ids = torch.tensor(class_ids, dtype=torch.long)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': class_ids,
            'image_id': torch.tensor([idx]),
            'img_path': str(img_path),
            'anno_path': str(anno_path)
        }
        
        if self.transform:
            img, target = self.transform(img, target)
            
        return img, target