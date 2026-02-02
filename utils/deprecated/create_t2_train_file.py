#!/usr/bin/env python3
"""
Create t2.txt training file for few-shot incremental learning.

For T2 stage in open-world detection:
- Uses few-shot samples from NOVEL classes only (7 classes)
- Each image in t2.txt contains ONLY annotations for the target class
- Images are listed with full paths to filtered XMLs

This follows TFA (Two-stage Fine-tuning Approach) strategy:
1. T1 model trained on base classes (11 classes)
2. T2 fine-tunes on few-shot novel classes while preserving base knowledge
"""

import os
from pathlib import Path
from collections import defaultdict

def create_t2_train_file(
    fewshot_annotations_dir='/home/agipml/sourav.rout/ALL_FILES/ovow/datasets/FewShot_Annotations',
    output_dir='/home/agipml/sourav.rout/ALL_FILES/ovow/datasets/ImageSets/Main/IDD',
    shot=10,  # Which shot setting to use
    novel_classes_only=True,  # Only novel classes for T2
    include_base_classes=False  # Set True for joint fine-tuning
):
    """
    Create t2.txt listing few-shot training images.
    
    Args:
        shot: Number of shots (1, 10, 20, 30)
        novel_classes_only: If True, only use novel classes (TFA strategy)
        include_base_classes: If True, also include base class few-shot samples
    """
    
    # IDD Novel classes (T2 new classes)
    NOVEL_CLASSES = [
        "concrete_mixer",
        "crane_truck",
        "excavator",
        "pole",
        "street_cart",
        "tanker_vehicle",
        "tractor"
    ]
    
    # IDD Base classes (T1 classes) - only if joint training
    BASE_CLASSES = [
        "car",
        "motorcycle",
        "rider",
        "person",
        "autorickshaw",
        "bus",
        "truck",
        "bicycle",
        "traffic sign",
        "traffic light",
        "ego vehicle"
    ]
    
    shot_dir = os.path.join(fewshot_annotations_dir, f'{shot}shot')
    
    if not os.path.exists(shot_dir):
        print(f"Error: Shot directory not found: {shot_dir}")
        return None
    
    # Determine which classes to use
    if novel_classes_only:
        target_classes = NOVEL_CLASSES
        strategy = "TFA (novel classes only)"
    elif include_base_classes:
        target_classes = BASE_CLASSES + NOVEL_CLASSES
        strategy = "Joint (base + novel classes)"
    else:
        target_classes = NOVEL_CLASSES
        strategy = "TFA (novel classes only)"
    
    print("=" * 60)
    print(f"Creating T2 training file for {shot}-shot")
    print(f"Strategy: {strategy}")
    print("=" * 60)
    
    # Collect all image IDs
    all_images = []
    stats = defaultdict(int)
    
    for class_name in target_classes:
        class_dir = os.path.join(shot_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"  ⚠ Warning: Class directory not found: {class_name}")
            continue
        
        # List all XML files in this class folder
        xml_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.xml')])
        
        for xml_file in xml_files:
            img_id = Path(xml_file).stem
            # Use relative path from datasets/ root
            # Format: JPEGImages/XXXXXX.jpg (standard VOC format)
            img_path = f"JPEGImages/{img_id}.jpg"
            all_images.append(img_path)
            stats[class_name] += 1
    
    # Remove duplicates (same image might be in multiple class folders)
    unique_images = sorted(set(all_images))
    
    print(f"\nClass statistics:")
    for cls_name in sorted(stats.keys()):
        print(f"  {cls_name:25s}: {stats[cls_name]} images")
    
    print(f"\nTotal images: {len(all_images)}")
    print(f"Unique images: {len(unique_images)}")
    
    # Write t2.txt
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 't2.txt')
    
    with open(output_file, 'w') as f:
        for img_path in unique_images:
            f.write(img_path + '\n')
    
    print(f"\n✓ Created: {output_file}")
    print(f"  Lines: {len(unique_images)}")
    
    return output_file

def create_multiple_t2_files(shots=[1, 10, 20, 30]):
    """Create t2 train files for different shot settings"""
    
    base_dir = '/home/agipml/sourav.rout/ALL_FILES/ovow/datasets/ImageSets/Main/IDD'
    
    for shot in shots:
        # Create TFA-style file (novel only)
        output_file = os.path.join(base_dir, f't2_{shot}shot.txt')
        create_t2_train_file(shot=shot, novel_classes_only=True)
        
        # Rename to shot-specific file
        default_file = os.path.join(base_dir, 't2.txt')
        if os.path.exists(default_file):
            os.rename(default_file, output_file)
            print(f"  Renamed to: {output_file}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--shot', type=int, default=10, help='Shot setting (1, 10, 20, 30)')
    parser.add_argument('--all', action='store_true', help='Create files for all shot settings')
    args = parser.parse_args()
    
    if args.all:
        create_multiple_t2_files()
    else:
        # Create t2.txt for specified shot
        create_t2_train_file(shot=args.shot)
