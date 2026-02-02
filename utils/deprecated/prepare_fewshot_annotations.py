#!/usr/bin/env python3
"""
Few-Shot Fine-tuning Data Preparation for OVOW
Following M-OWODB style - simple flat structure:
- FewShot_Annotations/30shot/ contains merged XMLs (all classes combined per image)
- t2.txt lists all image IDs for training
- If same image used for multiple classes, annotations are MERGED into one XML
"""

import os
import sys
from pathlib import Path
from xml.etree import ElementTree as ET
from collections import defaultdict
import shutil

def convert_honda_path_to_current(path_str):
    """Convert HONDA/ovow path references to current repo"""
    if 'HONDA/ovow' in path_str:
        path_str = path_str.replace('/home/agipml/sourav.rout/ALL_FILES/HONDA/ovow', 
                                   '/home/agipml/sourav.rout/ALL_FILES/ovow')
    return path_str

def extract_image_id(jpg_path):
    """Extract image ID from JPG path (e.g., 000000 from .../000000.jpg)"""
    return Path(jpg_path).stem

def read_fewshot_split(split_txt_path):
    """Read fewshot txt file and return list of image IDs"""
    image_ids = []
    try:
        with open(split_txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    line = convert_honda_path_to_current(line)
                    img_id = extract_image_id(line)
                    image_ids.append(img_id)
    except FileNotFoundError:
        print(f"  ✗ File not found: {split_txt_path}")
        return []
    return image_ids

def filter_xml_for_classes(src_xml_path, target_classes):
    """
    Extract annotations matching target_classes from XML
    Returns list of object elements
    """
    try:
        tree = ET.parse(src_xml_path)
        root = tree.getroot()
        
        matching_objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name in target_classes:
                # Deep copy the object element
                matching_objects.append(obj)
        
        return root, matching_objects
    except Exception as e:
        print(f"    ✗ Error parsing {src_xml_path}: {e}")
        return None, []

def merge_xmls_for_image(original_xml_path, classes_for_image, output_xml_path):
    """
    Create merged XML with annotations for all specified classes.
    If same image used for multiple classes, combines all matching annotations.
    """
    try:
        tree = ET.parse(original_xml_path)
        root = tree.getroot()
        
        # Remove objects not in our target classes
        objects_to_remove = []
        kept_count = 0
        kept_classes = set()
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name in classes_for_image:
                kept_count += 1
                kept_classes.add(name)
            else:
                objects_to_remove.append(obj)
        
        for obj in objects_to_remove:
            root.remove(obj)
        
        # Save merged XML
        os.makedirs(os.path.dirname(output_xml_path), exist_ok=True)
        tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)
        
        return kept_count, kept_classes
    except Exception as e:
        print(f"    ✗ Error processing {original_xml_path}: {e}")
        return 0, set()

def prepare_fewshot_annotations(
    fewshot_dir='/home/agipml/sourav.rout/ALL_FILES/ovow/datasets/fewshot/idd_splits/seed1',
    original_annotations_dir='/home/agipml/sourav.rout/ALL_FILES/ovow/datasets/Annotations',
    output_base_dir='/home/agipml/sourav.rout/ALL_FILES/ovow/datasets/FewShot_Annotations',
    imagesets_dir='/home/agipml/sourav.rout/ALL_FILES/ovow/datasets/ImageSets/Main/IDD',
    shots=[1, 10, 20, 30],
    novel_classes_only=False,  # If True, only include novel classes in filtered XMLs
    novel_classes=['concrete_mixer', 'crane_truck', 'excavator', 'pole', 'street_cart', 'tanker_vehicle', 'tractor']
):
    """
    Prepare few-shot annotations for OVOW fine-tuning (M-OWODB style)
    
    Creates:
    - FewShot_Annotations/{k}shot/ - flat folder with merged XMLs
    - ImageSets/Main/IDD/t2_{k}shot.txt - list of image IDs
    
    If same image appears in multiple class splits, annotations are MERGED.
    """
    
    print("=" * 70)
    print("FEW-SHOT ANNOTATION PREPARATION FOR OVOW (M-OWODB Style)")
    print("=" * 70)
    print("Strategy: Flat folder structure, merge annotations if same image")
    print("          used for multiple classes")
    if novel_classes_only:
        print(f"Mode: NOVEL CLASSES ONLY - {novel_classes}")
    else:
        print("Mode: ALL CLASSES (base + novel)")
    
    # Clean output directory
    if os.path.exists(output_base_dir):
        shutil.rmtree(output_base_dir)
    os.makedirs(imagesets_dir, exist_ok=True)
    
    # Collect all fewshot splits: shot -> class -> list(image_ids)
    fewshot_splits = defaultdict(lambda: defaultdict(list))
    
    print("\n[1] Reading few-shot split files...")
    for filename in sorted(os.listdir(fewshot_dir)):
        if not filename.endswith('_train.txt'):
            continue
        
        # Parse filename: box_Kshot_CLASSNAME_train.txt
        parts = filename.replace('_train.txt', '').split('_')
        try:
            shot_str = parts[1]  # e.g., "10shot"
            shot_num = int(shot_str.replace('shot', ''))
            class_name = '_'.join(parts[2:])  # Handle underscores in class names
        except (IndexError, ValueError):
            print(f"  ✗ Skipping malformed filename: {filename}")
            continue
        
        if shot_num not in shots:
            continue
        
        # Skip base classes if novel_classes_only mode
        if novel_classes_only and class_name not in novel_classes:
            continue
        
        filepath = os.path.join(fewshot_dir, filename)
        image_ids = read_fewshot_split(filepath)
        
        print(f"  ✓ {shot_num}-shot {class_name}: {len(image_ids)} images")
        fewshot_splits[shot_num][class_name] = image_ids
    
    # Build image -> classes mapping for each shot
    print("\n[2] Building image-to-classes mapping...")
    for shot_num in sorted(fewshot_splits.keys()):
        # image_id -> set of classes it's used for
        image_classes = defaultdict(set)
        
        for class_name, image_ids in fewshot_splits[shot_num].items():
            for img_id in image_ids:
                image_classes[img_id].add(class_name)
        
        # Count duplicates
        duplicates = {k: v for k, v in image_classes.items() if len(v) > 1}
        if duplicates:
            print(f"  {shot_num}-shot: {len(duplicates)} images used for multiple classes:")
            for img_id, classes in sorted(duplicates.items())[:5]:
                print(f"    {img_id}: {classes}")
            if len(duplicates) > 5:
                print(f"    ... and {len(duplicates) - 5} more")
        else:
            print(f"  {shot_num}-shot: No duplicate images across classes")
    
    # Create merged XMLs and t2.txt files
    print("\n[3] Creating merged annotations and t2.txt files...")
    
    for shot_num in sorted(fewshot_splits.keys()):
        shot_dir = os.path.join(output_base_dir, f'{shot_num}shot')
        os.makedirs(shot_dir, exist_ok=True)
        
        # Build image -> classes mapping
        image_classes = defaultdict(set)
        for class_name, image_ids in fewshot_splits[shot_num].items():
            for img_id in image_ids:
                image_classes[img_id].add(class_name)
        
        # Create merged XMLs
        total_annotations = 0
        merged_classes_count = defaultdict(int)
        
        for img_id in sorted(image_classes.keys()):
            classes_for_img = image_classes[img_id]
            src_xml = os.path.join(original_annotations_dir, f"{img_id}.xml")
            dst_xml = os.path.join(shot_dir, f"{img_id}.xml")
            
            if os.path.exists(src_xml):
                count, kept_classes = merge_xmls_for_image(src_xml, classes_for_img, dst_xml)
                total_annotations += count
                for c in kept_classes:
                    merged_classes_count[c] += 1
            else:
                print(f"    ✗ Missing XML: {img_id}.xml")
        
        # Create t2.txt
        t2_txt_path = os.path.join(imagesets_dir, f't2_{shot_num}shot.txt')
        with open(t2_txt_path, 'w') as f:
            for img_id in sorted(image_classes.keys()):
                f.write(f"{img_id}\n")
        
        print(f"\n  {shot_num}-shot:")
        print(f"    ✓ XMLs created: {len(image_classes)} (merged)")
        print(f"    ✓ Total annotations: {total_annotations}")
        print(f"    ✓ t2.txt: {t2_txt_path}")
        print(f"    Classes breakdown:")
        for cls, cnt in sorted(merged_classes_count.items()):
            print(f"      - {cls}: {cnt} annotations")
    
    # Create default t2.txt symlink to 10shot (or specified default)
    default_shot = 10
    default_t2 = os.path.join(imagesets_dir, 't2.txt')
    source_t2 = os.path.join(imagesets_dir, f't2_{default_shot}shot.txt')
    if os.path.exists(source_t2):
        shutil.copy(source_t2, default_t2)
        print(f"\n  ✓ Default t2.txt copied from {default_shot}-shot")
    
    print("\n" + "=" * 70)
    print("✓ FEW-SHOT PREPARATION COMPLETE (M-OWODB Style)")
    print(f"  Annotations: {output_base_dir}")
    print(f"  ImageSets: {imagesets_dir}")
    print("=" * 70)

if __name__ == '__main__':
    # Run with novel classes only for T2 incremental learning
    prepare_fewshot_annotations(
        shots=[1, 10, 20, 30],
        novel_classes_only=True,  # Only novel classes for T2
        novel_classes=['concrete_mixer', 'crane_truck', 'excavator', 'pole', 
                       'street_cart', 'tanker_vehicle', 'tractor']
    )
