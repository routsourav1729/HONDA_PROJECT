"""
Diagnose where Unknown FPs come from:
  - How many overlap known-class GT boxes (misclassified knowns)
  - How many overlap nothing (background)
"""

import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict
from tqdm import tqdm
import sys
sys.path.insert(0, '/home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2')

from fvcore.common.file_io import PathManager

def parse_rec(filename, known_classes):
    """Parse a PASCAL VOC xml file."""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        cls_name = obj.find("name").text
        
        # Map to "unknown" if not in known classes
        if cls_name not in known_classes:
            obj_struct["name"] = "unknown"
            obj_struct["original_name"] = cls_name
        else:
            obj_struct["name"] = cls_name
            obj_struct["original_name"] = cls_name
        
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)
    return objects


def compute_iou(box1, box2):
    """Compute IoU between two boxes [xmin, ymin, xmax, ymax]."""
    ixmin = max(box1[0], box2[0])
    iymin = max(box1[1], box2[1])
    ixmax = min(box1[2], box2[2])
    iymax = min(box1[3], box2[3])
    iw = max(ixmax - ixmin + 1.0, 0.0)
    ih = max(iymax - iymin + 1.0, 0.0)
    inters = iw * ih
    uni = ((box1[2] - box1[0] + 1.0) * (box1[3] - box1[1] + 1.0) +
           (box2[2] - box2[0] + 1.0) * (box2[3] - box2[1] + 1.0) - inters)
    return inters / uni if uni > 0 else 0.0


def main():
    # IDD Task 1 config
    known_classes = ('car', 'motorcycle', 'rider', 'person', 'autorickshaw', 
                     'traffic sign', 'traffic light', 'pole', 'bicycle')
    
    dataset_root = './datasets/IDD/t1/test'
    annopath = dataset_root + '/Annotations/{:s}.xml'
    imagesetfile = dataset_root + '/ImageSets/Main/test.txt'
    detpath = './output/IDD/t1/Main/{:s}.txt'  # from evaluator
    
    # Load image list
    with open(imagesetfile, 'r') as f:
        imagenames = [x.strip() for x in f.readlines()]
    
    print(f"Loading annotations for {len(imagenames)} images...")
    
    # Parse all GT annotations
    recs = {}
    for imagename in tqdm(imagenames, desc="Parsing XML"):
        recs[imagename] = parse_rec(annopath.format(imagename), known_classes)
    
    # Load unknown predictions
    unknown_detfile = detpath.format('unknown')
    try:
        with open(unknown_detfile, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: {unknown_detfile} not found. Run evaluation first.")
        return
    
    print(f"\nAnalyzing {len(lines)} unknown predictions...")
    
    splitlines = [x.strip().split(" ") for x in lines]
    
    # Categorize each unknown prediction
    overlaps_unknown_gt = 0
    overlaps_known_gt = 0
    overlaps_nothing = 0
    
    for line in tqdm(splitlines, desc="Matching predictions"):
        image_id = line[0]
        bbox = [float(x) for x in line[2:]]
        
        # Get GT for this image
        gt_objects = recs[image_id]
        
        best_iou_unknown = 0.0
        best_iou_known = 0.0
        
        for obj in gt_objects:
            iou = compute_iou(bbox, obj["bbox"])
            if obj["name"] == "unknown":
                best_iou_unknown = max(best_iou_unknown, iou)
            else:  # known class
                best_iou_known = max(best_iou_known, iou)
        
        # Categorize (0.5 IoU threshold)
        if best_iou_unknown > 0.5:
            overlaps_unknown_gt += 1
        elif best_iou_known > 0.5:
            overlaps_known_gt += 1
        else:
            overlaps_nothing += 1
    
    total = len(splitlines)
    
    print("\n" + "="*60)
    print("UNKNOWN PREDICTION BREAKDOWN")
    print("="*60)
    print(f"Total unknown predictions: {total:,}")
    print(f"  Correctly matched unknown GT (TP):    {overlaps_unknown_gt:>10,}  ({overlaps_unknown_gt/total*100:>5.1f}%)")
    print(f"  Overlaps known-class GT (stolen):     {overlaps_known_gt:>10,}  ({overlaps_known_gt/total*100:>5.1f}%)")
    print(f"  Overlaps nothing (background):        {overlaps_nothing:>10,}  ({overlaps_nothing/total*100:>5.1f}%)")
    print("="*60)
    print(f"\nFalse Positives breakdown:")
    fps = overlaps_known_gt + overlaps_nothing
    print(f"  Total FPs: {fps:,}")
    print(f"    - Misclassified knowns: {overlaps_known_gt:,} ({overlaps_known_gt/fps*100:.1f}% of FPs)")
    print(f"    - Pure background:      {overlaps_nothing:,} ({overlaps_nothing/fps*100:.1f}% of FPs)")
    print("="*60)


if __name__ == "__main__":
    main()
