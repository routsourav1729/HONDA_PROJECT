#!/usr/bin/env python3
"""
Fast class distribution analysis for train/test splits
Uses multiprocessing to speed up XML parsing
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import time


def parse_xml_file(xml_path):
    """Parse single XML file and return class counts"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        class_counts = defaultdict(int)
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            class_counts[class_name] += 1
        
        return dict(class_counts)
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return {}


def process_batch(xml_paths):
    """Process batch of XML files (for multiprocessing)"""
    batch_stats = defaultdict(int)
    
    for xml_path in xml_paths:
        class_counts = parse_xml_file(xml_path)
        for cls_name, count in class_counts.items():
            batch_stats[cls_name] += count
    
    return dict(batch_stats)


def merge_stats(stats_list):
    """Merge statistics from multiple processes"""
    merged = defaultdict(int)
    for stats in stats_list:
        for cls_name, count in stats.items():
            merged[cls_name] += count
    return dict(merged)


def analyze_split(split_file, annotations_dir, num_workers=None):
    """Analyze class distribution for a split"""
    # Read image IDs
    with open(split_file, 'r') as f:
        image_ids = [line.strip() for line in f if line.strip()]
    
    # Build XML paths
    xml_paths = [Path(annotations_dir) / f"{img_id}.xml" for img_id in image_ids]
    
    # Filter existing files
    xml_paths = [p for p in xml_paths if p.exists()]
    
    print(f"  Found {len(xml_paths)} annotation files")
    
    if not xml_paths:
        return {}
    
    # Use multiprocessing for speed
    if num_workers is None:
        num_workers = min(cpu_count(), 16)  # Cap at 16 to avoid overload
    
    # Split into batches
    batch_size = max(1, len(xml_paths) // num_workers)
    batches = [xml_paths[i:i+batch_size] for i in range(0, len(xml_paths), batch_size)]
    
    print(f"  Processing with {num_workers} workers, {len(batches)} batches...")
    
    # Process in parallel
    with Pool(num_workers) as pool:
        batch_results = pool.map(process_batch, batches)
    
    # Merge results
    return merge_stats(batch_results)


def print_stats(stats, title):
    """Pretty print statistics"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    if not stats:
        print("No data found")
        return
    
    # Sort by count descending
    sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
    
    total_objects = sum(stats.values())
    
    print(f"\n{'Class':<30} {'Count':>10} {'Percentage':>10}")
    print(f"{'-'*30} {'-'*10} {'-'*10}")
    
    for cls_name, count in sorted_stats:
        percentage = (count / total_objects) * 100
        print(f"{cls_name:<30} {count:>10,} {percentage:>9.2f}%")
    
    print(f"{'-'*30} {'-'*10} {'-'*10}")
    print(f"{'TOTAL':<30} {total_objects:>10,} {100.0:>9.2f}%")
    print(f"\nUnique classes: {len(stats)}")


def main():
    # Configuration
    base_dir = Path("/home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2/datasets")
    annotations_dir = base_dir / "Annotations"
    train_file = base_dir / "ImageSets/Main/IDD/t1.txt"
    test_file = base_dir / "ImageSets/Main/IDD/test.txt"
    
    print("Class Distribution Analysis")
    print(f"Annotations directory: {annotations_dir}")
    print(f"Using {cpu_count()} CPU cores")
    
    # Analyze train split
    print(f"\n[1/2] Analyzing TRAIN split: {train_file}")
    start = time.time()
    train_stats = analyze_split(train_file, annotations_dir)
    train_time = time.time() - start
    print(f"  Completed in {train_time:.2f} seconds")
    
    # Analyze test split
    print(f"\n[2/2] Analyzing TEST split: {test_file}")
    start = time.time()
    test_stats = analyze_split(test_file, annotations_dir)
    test_time = time.time() - start
    print(f"  Completed in {test_time:.2f} seconds")
    
    # Combined stats
    combined_stats = defaultdict(int)
    for cls_name, count in train_stats.items():
        combined_stats[cls_name] += count
    for cls_name, count in test_stats.items():
        combined_stats[cls_name] += count
    
    # Print combined table
    print(f"\n{'='*72}")
    print("FINAL CLASS DISTRIBUTION")
    print(f"{'='*72}")
    
    all_classes = sorted(combined_stats.keys(), key=lambda c: combined_stats[c], reverse=True)
    
    print(f"\n{'Class':<22s} {'Train':>10s} {'Test':>10s} {'Combined':>10s} {'Test%':>7s}")
    print("-" * 72)
    
    for cls in all_classes:
        train_count = train_stats.get(cls, 0)
        test_count = test_stats.get(cls, 0)
        combined = combined_stats[cls]
        test_pct = (test_count / combined * 100) if combined > 0 else 0
        print(f"{cls:<22s} {train_count:>10,} {test_count:>10,} {combined:>10,} {test_pct:>6.1f}%")
    
    total_train = sum(train_stats.values())
    total_test = sum(test_stats.values())
    total_combined = sum(combined_stats.values())
    
    print("-" * 72)
    print(f"{'TOTAL':<22s} {total_train:>10,} {total_test:>10,} {total_combined:>10,}")
    
    # Summary
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print(f"Train: {total_train:,} objects, {len(train_stats)} classes")
    print(f"Test:  {total_test:,} objects, {len(test_stats)} classes")
    print(f"Total time: {train_time + test_time:.2f} seconds")
    
    # Class comparison
    train_only = set(train_stats.keys()) - set(test_stats.keys())
    test_only = set(test_stats.keys()) - set(train_stats.keys())
    
    if train_only:
        print(f"\nClasses only in TRAIN: {', '.join(sorted(train_only))}")
    if test_only:
        print(f"\nClasses only in TEST: {', '.join(sorted(test_only))}")


if __name__ == "__main__":
    main()
