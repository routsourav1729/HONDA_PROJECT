30	347	725	BASE + NOVEL#!/usr/bin/env python3
"""
Few-Shot Data Preparation for OVOW (End-to-End)
===============================================

Following CED-FOOD/TFA strategy:
- For k-shot fine-tuning, sample k images per class
- Includes BOTH base classes AND novel classes (like CED-FOOD's "all" mode)
- If same image used for multiple classes, MERGE annotations into single XML
- Creates flat folder structure: FewShot_Annotations/{k}shot/
- Creates t2_{k}shot.txt listing all image IDs

Usage:
    python prepare_fewshot.py                    # Default: all shots, base+novel
    python prepare_fewshot.py --shot 10          # Specific shot
    python prepare_fewshot.py --novel-only       # Only novel classes (TFA style)
    python prepare_fewshot.py --list             # List available splits
"""

import os
import sys
import argparse
from pathlib import Path
from xml.etree import ElementTree as ET
from collections import defaultdict
import shutil

# ============================================================================
# IDD Dataset Configuration
# ============================================================================

# IDD T1 Base Classes (11 classes - learned in T1)
IDD_BASE_CLASSES = [
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

# IDD T2 Novel Classes (7 classes - new in T2)
IDD_NOVEL_CLASSES = [
    "concrete_mixer",
    "crane_truck",
    "excavator",
    "pole",
    "street_cart",
    "tanker_vehicle",
    "tractor"
]

# All known classes for T2 (base + novel = 18)
IDD_ALL_CLASSES = IDD_BASE_CLASSES + IDD_NOVEL_CLASSES

# ============================================================================
# Path Configuration
# ============================================================================

DEFAULT_CONFIG = {
    'fewshot_splits_dir': '/home/agipml/sourav.rout/ALL_FILES/ovow/datasets/fewshot/idd_splits/seed1',
    'original_annotations_dir': '/home/agipml/sourav.rout/ALL_FILES/ovow/datasets/Annotations',
    'output_annotations_dir': '/home/agipml/sourav.rout/ALL_FILES/ovow/datasets/FewShot_Annotations',
    'imagesets_dir': '/home/agipml/sourav.rout/ALL_FILES/ovow/datasets/ImageSets/Main/IDD',
    'shots': [1, 10, 20, 30],
}

# ============================================================================
# Utility Functions
# ============================================================================

def convert_path(path_str):
    """Convert old HONDA/ovow paths to current ovow paths"""
    if 'HONDA/ovow' in path_str:
        path_str = path_str.replace(
            '/home/agipml/sourav.rout/ALL_FILES/HONDA/ovow',
            '/home/agipml/sourav.rout/ALL_FILES/ovow'
        )
    return path_str


def extract_image_id(jpg_path):
    """Extract image ID from path (e.g., 000000 from .../000000.jpg)"""
    return Path(jpg_path).stem


def read_split_file(filepath):
    """Read fewshot split txt file, return list of image IDs"""
    image_ids = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    line = convert_path(line)
                    img_id = extract_image_id(line)
                    image_ids.append(img_id)
    except FileNotFoundError:
        return []
    return image_ids


def parse_split_filename(filename):
    """
    Parse fewshot split filename to extract shot and class name.
    Format: box_{K}shot_{CLASSNAME}_train.txt
    Returns: (shot_num, class_name) or (None, None) if invalid
    """
    if not filename.endswith('_train.txt'):
        return None, None
    
    # Remove suffix
    name = filename.replace('_train.txt', '')
    parts = name.split('_')
    
    try:
        # parts[0] = 'box', parts[1] = 'Kshot', parts[2:] = class_name
        shot_str = parts[1]  # e.g., "10shot"
        shot_num = int(shot_str.replace('shot', ''))
        class_name = '_'.join(parts[2:])  # Handle underscores in class names
        return shot_num, class_name
    except (IndexError, ValueError):
        return None, None


def merge_xml_for_classes(src_xml_path, target_classes, dst_xml_path):
    """
    Create XML with ONLY annotations matching target_classes.
    If image has multiple target classes, all their annotations are included.
    
    Args:
        src_xml_path: Path to original XML
        target_classes: Set of class names to keep
        dst_xml_path: Path to write filtered XML
    
    Returns:
        (annotation_count, kept_classes_set)
    """
    try:
        tree = ET.parse(src_xml_path)
        root = tree.getroot()
        
        # Filter objects
        objects_to_remove = []
        kept_count = 0
        kept_classes = set()
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name in target_classes:
                kept_count += 1
                kept_classes.add(name)
            else:
                objects_to_remove.append(obj)
        
        # Remove non-matching objects
        for obj in objects_to_remove:
            root.remove(obj)
        
        # Save filtered XML
        os.makedirs(os.path.dirname(dst_xml_path), exist_ok=True)
        tree.write(dst_xml_path, encoding='utf-8', xml_declaration=True)
        
        return kept_count, kept_classes
    
    except Exception as e:
        print(f"    ✗ Error processing {src_xml_path}: {e}")
        return 0, set()


# ============================================================================
# Main Functions
# ============================================================================

def list_available_splits(config):
    """List all available fewshot splits"""
    splits_dir = config['fewshot_splits_dir']
    
    print("=" * 60)
    print("AVAILABLE FEW-SHOT SPLITS")
    print("=" * 60)
    print(f"Directory: {splits_dir}\n")
    
    # Collect splits by shot
    splits = defaultdict(list)
    
    for filename in sorted(os.listdir(splits_dir)):
        shot_num, class_name = parse_split_filename(filename)
        if shot_num is not None:
            splits[shot_num].append(class_name)
    
    for shot_num in sorted(splits.keys()):
        classes = splits[shot_num]
        base_classes = [c for c in classes if c in IDD_BASE_CLASSES]
        novel_classes = [c for c in classes if c in IDD_NOVEL_CLASSES]
        
        print(f"{shot_num}-shot:")
        print(f"  Base classes ({len(base_classes)}): {base_classes}")
        print(f"  Novel classes ({len(novel_classes)}): {novel_classes}")
        print()


def prepare_fewshot_data(
    config,
    shots=None,
    novel_only=False,
    verbose=True
):
    """
    Main function to prepare few-shot annotations.
    
    Args:
        config: Configuration dict with paths
        shots: List of shot values to process (default: all)
        novel_only: If True, only include novel classes (TFA style)
                   If False, include both base + novel (CED-FOOD style)
        verbose: Print detailed progress
    """
    
    splits_dir = config['fewshot_splits_dir']
    annotations_dir = config['original_annotations_dir']
    output_dir = config['output_annotations_dir']
    imagesets_dir = config['imagesets_dir']
    
    if shots is None:
        shots = config['shots']
    
    # Determine target classes
    if novel_only:
        target_classes_list = IDD_NOVEL_CLASSES
        mode = "NOVEL ONLY (TFA style)"
    else:
        target_classes_list = IDD_ALL_CLASSES
        mode = "BASE + NOVEL (CED-FOOD style)"
    
    target_classes_set = set(target_classes_list)
    
    print("=" * 70)
    print("FEW-SHOT DATA PREPARATION FOR OVOW")
    print("=" * 70)
    print(f"Mode: {mode}")
    print(f"Target classes: {len(target_classes_list)}")
    print(f"Shots: {shots}")
    print(f"Input splits: {splits_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Clean output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(imagesets_dir, exist_ok=True)
    
    # ========================================================================
    # Step 1: Read all split files
    # ========================================================================
    print("[1] Reading split files...")
    
    # Structure: shot -> class -> list of image_ids
    all_splits = defaultdict(lambda: defaultdict(list))
    
    for filename in sorted(os.listdir(splits_dir)):
        shot_num, class_name = parse_split_filename(filename)
        
        if shot_num is None or shot_num not in shots:
            continue
        
        # Skip classes not in target list
        if class_name not in target_classes_set:
            continue
        
        filepath = os.path.join(splits_dir, filename)
        image_ids = read_split_file(filepath)
        
        if verbose:
            print(f"  ✓ {shot_num}-shot {class_name}: {len(image_ids)} images")
        
        all_splits[shot_num][class_name] = image_ids
    
    # ========================================================================
    # Step 2: Build image -> classes mapping (detect duplicates)
    # ========================================================================
    print("\n[2] Building image-to-classes mapping...")
    
    for shot_num in sorted(all_splits.keys()):
        # image_id -> set of classes it appears in
        image_classes = defaultdict(set)
        
        for class_name, image_ids in all_splits[shot_num].items():
            for img_id in image_ids:
                image_classes[img_id].add(class_name)
        
        # Count images appearing in multiple classes
        duplicates = {k: v for k, v in image_classes.items() if len(v) > 1}
        
        if duplicates:
            print(f"  {shot_num}-shot: {len(duplicates)} images used for multiple classes (will merge)")
            if verbose:
                for img_id, classes in list(duplicates.items())[:3]:
                    print(f"    {img_id}: {classes}")
                if len(duplicates) > 3:
                    print(f"    ... and {len(duplicates) - 3} more")
        else:
            print(f"  {shot_num}-shot: {len(image_classes)} unique images, no duplicates")
    
    # ========================================================================
    # Step 3: Create merged XMLs and t2.txt files
    # ========================================================================
    print("\n[3] Creating merged annotations...")
    
    for shot_num in sorted(all_splits.keys()):
        shot_dir = os.path.join(output_dir, f'{shot_num}shot')
        os.makedirs(shot_dir, exist_ok=True)
        
        # Build image -> classes mapping for this shot
        image_classes = defaultdict(set)
        for class_name, image_ids in all_splits[shot_num].items():
            for img_id in image_ids:
                image_classes[img_id].add(class_name)
        
        # Create merged XMLs
        total_annotations = 0
        class_annotation_counts = defaultdict(int)
        
        for img_id in sorted(image_classes.keys()):
            classes_for_img = image_classes[img_id]
            src_xml = os.path.join(annotations_dir, f"{img_id}.xml")
            dst_xml = os.path.join(shot_dir, f"{img_id}.xml")
            
            if os.path.exists(src_xml):
                count, kept_classes = merge_xml_for_classes(
                    src_xml, classes_for_img, dst_xml
                )
                total_annotations += count
                for cls in kept_classes:
                    class_annotation_counts[cls] += 1
            else:
                print(f"    ✗ Missing: {img_id}.xml")
        
        # Create t2_{k}shot.txt
        t2_path = os.path.join(imagesets_dir, f't2_{shot_num}shot.txt')
        with open(t2_path, 'w') as f:
            for img_id in sorted(image_classes.keys()):
                f.write(f"{img_id}\n")
        
        # Print summary
        print(f"\n  {shot_num}-shot:")
        print(f"    ✓ XMLs: {len(image_classes)}")
        print(f"    ✓ Annotations: {total_annotations}")
        print(f"    ✓ t2.txt: {t2_path}")
        
        if verbose:
            print(f"    Per-class annotations:")
            for cls in sorted(class_annotation_counts.keys()):
                marker = "[BASE]" if cls in IDD_BASE_CLASSES else "[NOVEL]"
                print(f"      {marker} {cls}: {class_annotation_counts[cls]}")
    
    # ========================================================================
    # Step 4: Create default t2.txt (symlink to 10-shot)
    # ========================================================================
    default_shot = 10 if 10 in shots else shots[0]
    default_t2_src = os.path.join(imagesets_dir, f't2_{default_shot}shot.txt')
    default_t2_dst = os.path.join(imagesets_dir, 't2.txt')
    
    if os.path.exists(default_t2_src):
        shutil.copy(default_t2_src, default_t2_dst)
        print(f"\n  ✓ Default t2.txt → {default_shot}-shot")
    
    print("\n" + "=" * 70)
    print("✓ FEW-SHOT PREPARATION COMPLETE")
    print(f"  Annotations: {output_dir}")
    print(f"  ImageSets: {imagesets_dir}")
    print("=" * 70)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Prepare few-shot annotations for OVOW T2 fine-tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_fewshot.py                  # All shots, base+novel (CED-FOOD style)
  python prepare_fewshot.py --shot 10        # Only 10-shot
  python prepare_fewshot.py --novel-only     # Only novel classes (TFA style)
  python prepare_fewshot.py --list           # List available splits
        """
    )
    
    parser.add_argument('--shot', type=int, nargs='+', default=None,
                        help='Shot value(s) to process (default: 1,10,20,30)')
    parser.add_argument('--novel-only', action='store_true',
                        help='Only include novel classes (TFA style). '
                             'Default includes both base+novel (CED-FOOD style)')
    parser.add_argument('--list', action='store_true',
                        help='List available splits and exit')
    parser.add_argument('--quiet', action='store_true',
                        help='Less verbose output')
    
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    
    if args.list:
        list_available_splits(config)
        return
    
    shots = args.shot if args.shot else config['shots']
    
    prepare_fewshot_data(
        config=config,
        shots=shots,
        novel_only=args.novel_only,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
