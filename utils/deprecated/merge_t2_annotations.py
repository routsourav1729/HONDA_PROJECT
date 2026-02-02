#!/usr/bin/env python3
"""
Merge filtered few-shot XMLs into a single annotation folder for T2 training.

This creates a t2_Annotations_Xshot/ folder that can be used with standard
VOC dataset loader. Each image has ONLY annotations for the target class
(following TFA strategy).
"""

import os
import shutil
from pathlib import Path

def merge_fewshot_annotations(
    fewshot_dir='/home/agipml/sourav.rout/ALL_FILES/ovow/datasets/FewShot_Annotations',
    output_base_dir='/home/agipml/sourav.rout/ALL_FILES/ovow/datasets',
    shot=10,
    novel_classes_only=True
):
    """
    Merge per-class filtered XMLs into single annotation folder.
    
    Note: If same image appears in multiple classes, the LAST class written wins.
    This is intentional for TFA - each image-class pair is treated separately.
    For multiple classes in same image, the original approach (copy per-class)
    needs class-specific loading.
    """
    
    # Classes
    NOVEL_CLASSES = [
        "concrete_mixer", "crane_truck", "excavator", "pole",
        "street_cart", "tanker_vehicle", "tractor"
    ]
    
    BASE_CLASSES = [
        "car", "motorcycle", "rider", "person", "autorickshaw",
        "bus", "truck", "bicycle", "traffic sign", "traffic light", "ego vehicle"
    ]
    
    target_classes = NOVEL_CLASSES if novel_classes_only else (BASE_CLASSES + NOVEL_CLASSES)
    
    shot_dir = os.path.join(fewshot_dir, f'{shot}shot')
    output_dir = os.path.join(output_base_dir, f't2_Annotations_{shot}shot')
    
    print(f"Merging {shot}-shot annotations for classes: {len(target_classes)}")
    print(f"Source: {shot_dir}")
    print(f"Output: {output_dir}")
    
    # Clean output
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Copy XMLs
    copied = 0
    for class_name in target_classes:
        class_dir = os.path.join(shot_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"  Warning: {class_name} not found")
            continue
            
        for xml_file in os.listdir(class_dir):
            if xml_file.endswith('.xml'):
                src = os.path.join(class_dir, xml_file)
                dst = os.path.join(output_dir, xml_file)
                # NOTE: If duplicate, later class overwrites earlier
                shutil.copy2(src, dst)
                copied += 1
    
    print(f"Copied {copied} XMLs to {output_dir}")
    return output_dir

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--shot', type=int, default=10)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    
    if args.all:
        for shot in [1, 10, 20, 30]:
            merge_fewshot_annotations(shot=shot)
    else:
        merge_fewshot_annotations(shot=args.shot)
