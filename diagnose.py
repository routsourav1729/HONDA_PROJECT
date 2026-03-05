"""
Diagnostic script: analyses WHERE unknown objects go.

For every GT unknown object, finds the best-overlapping prediction and reports:
  - Was it detected at all? (IoU > 0.5 with any prediction box)
  - If detected, what class label did the prediction have?
  - What is the YOLO confidence score for that prediction?
  - What is the horo_max_score and which prototype was it assigned to?
  - Was it relabeled as unknown by the horosphere threshold?
  - After NMS, did the prediction survive?

Also shows: score distributions per predicted class, horo_max_score distributions,
and a confusion matrix of "GT unknown class X → predicted as class Y".

Usage:
  python diagnose.py --config-file configs/IDD_HYP/base.yaml \
      --task IDD_HYP/t2 --ckpt IDD_HYP/t2/fewshotfinetunev3/model_final.pth
"""

import os
import torch
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from collections import defaultdict
from torchvision.ops import nms, box_iou
from fvcore.common.file_io import PathManager

from core import add_config
from core.util.model_ema import add_model_ema_configs
from core.pascal_voc import register_pascal_voc, inital_prompts
from core.hyp_customyoloworld import HypCustomYoloWorld, load_hyp_ckpt
from core.calibrate_thresholds import compute_thresholds
from core.eval_utils import Trainer

from mmengine.config import Config
from mmengine.runner import Runner
from detectron2.engine import default_argument_parser, default_setup
from detectron2.config import get_cfg


class Register:
    def __init__(self, dataset_root, split, cfg, dataset_key=None):
        self.dataset_root = dataset_root
        self.super_split = split.split('/')[0]
        self.cfg = cfg
        self.dataset_key = dataset_key if dataset_key is not None else self.super_split
        self.PREDEFINED_SPLITS_DATASET = {
            "my_train": split,
            "my_val": os.path.join(self.super_split, 'test')
        }

    def register_dataset(self):
        for name, split in self.PREDEFINED_SPLITS_DATASET.items():
            register_pascal_voc(name, self.dataset_root, self.dataset_key, split, self.cfg)


def setup(args):
    cfg = get_cfg()
    add_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    if args.task:
        task_yaml = os.path.join("configs", args.task.split('/')[0], args.task.split('/')[1] + ".yaml")
        if os.path.exists(task_yaml):
            cfg.merge_from_file(task_yaml)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def parse_gt_boxes(anno_path, known_classes):
    """Parse GT annotations, returning known and unknown boxes with original class names."""
    try:
        with PathManager.open(anno_path) as f:
            tree = ET.parse(f)
    except:
        return [], []
    
    known_boxes = []
    unknown_boxes = []
    
    for obj in tree.findall("object"):
        cls_name = obj.find("name").text
        difficult = obj.find("difficult")
        difficult = int(difficult.text) if difficult is not None else 0
        if difficult:
            continue
        
        bbox_elem = obj.find("bndbox")
        bbox = [
            float(bbox_elem.find("xmin").text),
            float(bbox_elem.find("ymin").text),
            float(bbox_elem.find("xmax").text),
            float(bbox_elem.find("ymax").text),
        ]
        
        if cls_name in known_classes:
            known_boxes.append({'class': cls_name, 'bbox': bbox})
        else:
            unknown_boxes.append({'class': cls_name, 'bbox': bbox})
    
    return known_boxes, unknown_boxes


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--task", default="")
    parser.add_argument("--ckpt", default="model.pth")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--max_images", type=int, default=0,
                        help="Max images to process (0=all)")
    args = parser.parse_args()
    cfg = setup(args)

    task_name, split_name = args.task.split('/')
    base_dataset = task_name.replace('_HYP', '')
    
    if split_name in ['t2', 't3', 't4']:
        dataset_key = f"{base_dataset}_T{split_name[1].upper()}"
    else:
        dataset_key = base_dataset
    
    data_split = f"{base_dataset}/{split_name}"
    data_register = Register('./datasets/', data_split, cfg, dataset_key)
    data_register.register_dataset()

    class_names = list(inital_prompts()[dataset_key])
    unknown_index = cfg.TEST.PREV_INTRODUCED_CLS + cfg.TEST.CUR_INTRODUCED_CLS
    known_class_names = class_names[:unknown_index]
    prev_cls = cfg.TEST.PREV_INTRODUCED_CLS
    cur_cls = cfg.TEST.CUR_INTRODUCED_CLS
    
    base_class_names = known_class_names[:prev_cls]
    novel_class_names = known_class_names[prev_cls:prev_cls+cur_cls]
    
    print(f"\nBase classes ({len(base_class_names)}): {base_class_names}")
    print(f"Novel classes ({len(novel_class_names)}): {novel_class_names}")
    print(f"Unknown index: {unknown_index}")

    # Load model
    ckpt_data = torch.load(args.ckpt, map_location='cpu')
    hyp_config = ckpt_data.get('hyp_config', {})
    hyp_c = hyp_config.get('curvature', 1.0)
    hyp_dim = hyp_config.get('embed_dim', 256)
    clip_r = hyp_config.get('clip_r', 0.95)
    adaptive_stats = ckpt_data.get('adaptive_stats', None)

    config_file = os.path.join("./configs", task_name, f"{split_name}.py")
    cfgY = Config.fromfile(config_file)
    cfgY.work_dir = "."
    runner = Runner.from_cfg(cfgY)
    # Strip EMA hook — it deep-copies the entire XL model (~20 min!) and is unused
    runner._hooks = [h for h in runner._hooks if not h.__class__.__name__.startswith('EMA')]
    runner.call_hook("before_run")
    runner.load_or_resume()
    runner.model.reparameterize([known_class_names])
    runner.model.eval()
    test_loader = Runner.build_dataloader(cfgY.test_dataloader)

    classifier_num_classes = cur_cls if prev_cls > 0 else unknown_index
    model = HypCustomYoloWorld(
        runner.model, unknown_index,
        hyp_c=hyp_c, hyp_dim=hyp_dim, clip_r=clip_r,
        num_classifier_classes=classifier_num_classes,
    )
    model = load_hyp_ckpt(model, args.ckpt, prev_cls, cur_cls, eval=True)
    model = model.cuda()
    model.add_generic_text(known_class_names, generic_prompt='object', alpha=0.4)
    model.eval()

    # Compute thresholds
    adaptive_thresholds = None
    if adaptive_stats is not None:
        adaptive_thresholds, alpha_used = compute_thresholds(
            adaptive_stats, known_class_names, alpha=args.alpha
        )
        adaptive_thresholds = adaptive_thresholds.cuda()

    # Get annotation path template
    from detectron2.data import MetadataCatalog
    meta = MetadataCatalog.get("my_val")
    anno_template = os.path.join(meta.dirname, "Annotations", "{}.xml")
    all_class_names = meta.thing_classes  # includes unknown classes

    # =====================================================================
    # DIAGNOSTIC COUNTERS
    # =====================================================================
    
    # For each GT unknown: what happened to it?
    unk_gt_total = defaultdict(int)                    # GT unknown class → count
    unk_detected = defaultdict(int)                    # detected (IoU>0.5 with ANY pred)
    unk_detected_as = defaultdict(lambda: defaultdict(int))  # GT class → pred class → count
    unk_detected_as_base = defaultdict(int)            # GT class → misclassified as base
    unk_detected_as_novel = defaultdict(int)           # GT class → misclassified as novel
    unk_detected_as_unknown = defaultdict(int)         # GT class → correctly labeled unknown
    unk_not_detected = defaultdict(int)                # GT class → no matching prediction
    
    # Score distributions for matched unknowns
    unk_yolo_scores = defaultdict(list)                # GT class → list of YOLO scores
    unk_horo_scores = defaultdict(list)                # GT class → list of horo_max_scores
    unk_assigned_protos = defaultdict(lambda: defaultdict(int))  # GT class → proto name → count
    
    # For known GTs: how well are they detected?
    known_gt_total = defaultdict(int)
    known_detected = defaultdict(int)
    known_misclassified = defaultdict(lambda: defaultdict(int))
    
    # NMS survival tracking
    unk_survived_nms = defaultdict(int)
    unk_killed_by_nms = defaultdict(int)
    
    # Score distribution for ALL predictions
    all_pred_scores_by_label = defaultdict(list)
    
    n_images = 0
    
    print(f"\n{'='*70}")
    print(f"RUNNING DIAGNOSTICS")
    print(f"{'='*70}\n")

    for batch in tqdm(test_loader, desc="Diagnosing"):
        data = model.parent.data_preprocessor(batch)
        
        with torch.no_grad():
            outputs = model.predict(data['inputs'], data['data_samples'])
        
        for out, data_sample in zip(outputs, batch['data_samples']):
            n_images += 1
            if args.max_images > 0 and n_images > args.max_images:
                break
            
            img_id = data_sample.img_id
            
            # Parse GT
            anno_path = anno_template.format(img_id)
            gt_known, gt_unknown = parse_gt_boxes(anno_path, known_class_names)
            
            pred = out.pred_instances
            if len(pred.scores) == 0:
                # No predictions at all
                for g in gt_unknown:
                    unk_gt_total[g['class']] += 1
                    unk_not_detected[g['class']] += 1
                for g in gt_known:
                    known_gt_total[g['class']] += 1
                continue
            
            # Get pre-NMS predictions
            pred_bboxes = pred.bboxes
            pred_scores = pred.scores
            pred_labels = pred.labels.clone()  # labels BEFORE relabeling
            has_horo = hasattr(pred, 'horo_max_scores') and hasattr(pred, 'horo_assigned_proto')
            
            if has_horo:
                horo_max = pred.horo_max_scores
                horo_proto = pred.horo_assigned_proto.long()
            
            # Apply OOD relabeling (same as test_hyp.py)
            labels_after_ood = pred_labels.clone()
            if adaptive_thresholds is not None and has_horo:
                per_det_thresholds = adaptive_thresholds[horo_proto]
                is_unknown = horo_max < per_det_thresholds
                labels_after_ood[is_unknown] = unknown_index
            
            # Apply NMS
            keep_after_nms = nms(pred_bboxes, pred_scores, iou_threshold=0.5)
            nms_survived = torch.zeros(len(pred_scores), dtype=torch.bool)
            nms_survived[keep_after_nms] = True
            
            # ---- Analyze GT unknown objects ----
            for g in gt_unknown:
                gt_cls = g['class']
                unk_gt_total[gt_cls] += 1
                
                gt_box = torch.tensor([g['bbox']], device=pred_bboxes.device)
                ious = box_iou(gt_box, pred_bboxes)[0]
                
                # Find best matching prediction (highest IoU)
                if ious.max() < 0.5:
                    unk_not_detected[gt_cls] += 1
                    continue
                
                best_idx = ious.argmax().item()
                best_iou = ious[best_idx].item()
                unk_detected[gt_cls] += 1
                
                # What class did the model predict?
                yolo_label = pred_labels[best_idx].item()
                ood_label = labels_after_ood[best_idx].item()
                yolo_score = pred_scores[best_idx].item()
                
                # Record YOLO's original classification
                if yolo_label < len(known_class_names):
                    pred_class_name = known_class_names[yolo_label]
                elif yolo_label == unknown_index:
                    pred_class_name = "unknown"
                else:
                    pred_class_name = f"cls_{yolo_label}"
                
                # After OOD relabeling
                if ood_label == unknown_index:
                    final_class = "unknown"
                elif ood_label < len(known_class_names):
                    final_class = known_class_names[ood_label]
                else:
                    final_class = f"cls_{ood_label}"
                
                # Confusion: GT unknown → predicted as what? (AFTER OOD relabeling)
                unk_detected_as[gt_cls][final_class] += 1
                
                if ood_label == unknown_index:
                    unk_detected_as_unknown[gt_cls] += 1
                elif ood_label < prev_cls:
                    unk_detected_as_base[gt_cls] += 1
                else:
                    unk_detected_as_novel[gt_cls] += 1
                
                unk_yolo_scores[gt_cls].append(yolo_score)
                
                if has_horo:
                    unk_horo_scores[gt_cls].append(horo_max[best_idx].item())
                    proto_idx = horo_proto[best_idx].item()
                    if proto_idx < len(known_class_names):
                        unk_assigned_protos[gt_cls][known_class_names[proto_idx]] += 1
                    else:
                        unk_assigned_protos[gt_cls][f"proto_{proto_idx}"] += 1
                
                # NMS survival
                if nms_survived[best_idx]:
                    unk_survived_nms[gt_cls] += 1
                else:
                    unk_killed_by_nms[gt_cls] += 1
            
            # ---- Analyze GT known objects ----
            for g in gt_known:
                gt_cls = g['class']
                known_gt_total[gt_cls] += 1
                
                gt_box = torch.tensor([g['bbox']], device=pred_bboxes.device)
                ious = box_iou(gt_box, pred_bboxes)[0]
                
                if ious.max() < 0.5:
                    continue
                
                best_idx = ious.argmax().item()
                ood_label = labels_after_ood[best_idx].item()
                known_detected[gt_cls] += 1
                
                if ood_label == unknown_index:
                    known_misclassified[gt_cls]['unknown'] += 1
                elif ood_label < len(known_class_names):
                    pred_name = known_class_names[ood_label]
                    if pred_name != gt_cls:
                        known_misclassified[gt_cls][pred_name] += 1
            
            # Score distributions (post-OOD relabeling)
            for i in range(len(pred_scores)):
                lbl = labels_after_ood[i].item()
                if lbl == unknown_index:
                    all_pred_scores_by_label['unknown'].append(pred_scores[i].item())
                elif lbl < len(known_class_names):
                    all_pred_scores_by_label[known_class_names[lbl]].append(pred_scores[i].item())
        
        if args.max_images > 0 and n_images > args.max_images:
            break

    # =====================================================================
    # PRINT RESULTS
    # =====================================================================
    
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC RESULTS ({n_images} images)")
    print(f"{'='*70}")
    
    # ---- 1. Per-unknown-class breakdown ----
    print(f"\n{'='*70}")
    print(f"1. GT UNKNOWN OBJECTS: WHERE DO THEY GO?")
    print(f"{'='*70}")
    print(f"{'GT Class':<22s} {'Total':>6s} {'Det':>6s} {'→Unk':>6s} {'→Base':>6s} {'→Novel':>7s} {'NoDet':>6s} {'NMS↓':>6s}")
    print(f"{'-'*70}")
    
    tot_gt = tot_det = tot_unk = tot_base = tot_novel = tot_nodet = tot_nms = 0
    for cls in sorted(unk_gt_total.keys()):
        gt = unk_gt_total[cls]
        det = unk_detected[cls]
        as_unk = unk_detected_as_unknown[cls]
        as_base = unk_detected_as_base[cls]
        as_novel = unk_detected_as_novel[cls]
        nodet = unk_not_detected[cls]
        nms_killed = unk_killed_by_nms[cls]
        
        tot_gt += gt; tot_det += det; tot_unk += as_unk
        tot_base += as_base; tot_novel += as_novel
        tot_nodet += nodet; tot_nms += nms_killed
        
        print(f"{cls:<22s} {gt:>6d} {det:>6d} {as_unk:>6d} {as_base:>6d} {as_novel:>7d} {nodet:>6d} {nms_killed:>6d}")
    
    print(f"{'-'*70}")
    print(f"{'TOTAL':<22s} {tot_gt:>6d} {tot_det:>6d} {tot_unk:>6d} {tot_base:>6d} {tot_novel:>7d} {tot_nodet:>6d} {tot_nms:>6d}")
    print(f"\nDetection rate: {tot_det}/{tot_gt} = {tot_det/max(tot_gt,1)*100:.1f}%")
    print(f"Correctly relabeled as unknown: {tot_unk}/{tot_det} = {tot_unk/max(tot_det,1)*100:.1f}% of detected")
    print(f"Misclassified as base class: {tot_base}/{tot_det} = {tot_base/max(tot_det,1)*100:.1f}%")
    print(f"Misclassified as novel class: {tot_novel}/{tot_det} = {tot_novel/max(tot_det,1)*100:.1f}%")
    print(f"NMS killed (detected but suppressed): {tot_nms}/{tot_det} = {tot_nms/max(tot_det,1)*100:.1f}%")
    
    # ---- 2. Confusion detail: GT unknown → misclassified as which known class? ----
    print(f"\n{'='*70}")
    print(f"2. CONFUSION: GT UNKNOWN → PREDICTED AS WHICH CLASS?")
    print(f"{'='*70}")
    for gt_cls in sorted(unk_detected_as.keys()):
        print(f"\n  {gt_cls} (GT total={unk_gt_total[gt_cls]}, detected={unk_detected[gt_cls]}):")
        sorted_preds = sorted(unk_detected_as[gt_cls].items(), key=lambda x: -x[1])
        for pred_cls, count in sorted_preds:
            pct = count / max(unk_detected[gt_cls], 1) * 100
            group = ""
            if pred_cls == 'unknown':
                group = " [CORRECT]"
            elif pred_cls in base_class_names:
                group = " [BASE]"
            elif pred_cls in novel_class_names:
                group = " [NOVEL]"
            print(f"    → {pred_cls:<20s}: {count:>4d} ({pct:>5.1f}%){group}")

    # ---- 3. Prototype assignment for unknowns ----
    print(f"\n{'='*70}")
    print(f"3. PROTOTYPE ASSIGNMENT: WHICH PROTOTYPE DO UNKNOWNS GET?")
    print(f"{'='*70}")
    for gt_cls in sorted(unk_assigned_protos.keys()):
        print(f"\n  {gt_cls}:")
        sorted_protos = sorted(unk_assigned_protos[gt_cls].items(), key=lambda x: -x[1])
        for proto, count in sorted_protos[:5]:
            print(f"    → prototype '{proto}': {count}")

    # ---- 4. Score distributions ----
    print(f"\n{'='*70}")
    print(f"4. YOLO SCORE DISTRIBUTIONS FOR GT UNKNOWNS (best matching pred)")
    print(f"{'='*70}")
    for gt_cls in sorted(unk_yolo_scores.keys()):
        scores = np.array(unk_yolo_scores[gt_cls])
        if len(scores) > 0:
            print(f"  {gt_cls:<20s}: mean={scores.mean():.4f}, median={np.median(scores):.4f}, "
                  f"min={scores.min():.4f}, max={scores.max():.4f}, "
                  f">0.1={np.sum(scores>0.1)}/{len(scores)}, >0.3={np.sum(scores>0.3)}/{len(scores)}")
    
    print(f"\n{'='*70}")
    print(f"5. HORO_MAX_SCORE DISTRIBUTIONS FOR GT UNKNOWNS")
    print(f"{'='*70}")
    for gt_cls in sorted(unk_horo_scores.keys()):
        scores = np.array(unk_horo_scores[gt_cls])
        if len(scores) > 0:
            if adaptive_thresholds is not None:
                thresholds_np = adaptive_thresholds.cpu().numpy()
                # What fraction would be relabeled at each threshold?
                below_min = np.sum(scores < thresholds_np.min()) 
                below_max = np.sum(scores < thresholds_np.max())
                below_mean = np.sum(scores < thresholds_np.mean())
            else:
                below_min = below_max = below_mean = 0
            print(f"  {gt_cls:<20s}: mean={scores.mean():.4f}, median={np.median(scores):.4f}, "
                  f"std={scores.std():.4f}, below_thresh_mean={below_mean}/{len(scores)}")

    # ---- 5. Known class detection / misclassification ----
    print(f"\n{'='*70}")
    print(f"6. KNOWN CLASS: FALSE RELABELING AS UNKNOWN")
    print(f"{'='*70}")
    print(f"{'Known Class':<22s} {'GT':>6s} {'Det':>6s} {'→Unk':>6s} {'→Wrong':>7s}")
    print(f"{'-'*50}")
    for cls in known_class_names:
        gt = known_gt_total.get(cls, 0)
        det = known_detected.get(cls, 0)
        to_unk = known_misclassified.get(cls, {}).get('unknown', 0)
        to_wrong = sum(v for k, v in known_misclassified.get(cls, {}).items() if k != 'unknown')
        if gt > 0:
            print(f"{cls:<22s} {gt:>6d} {det:>6d} {to_unk:>6d} {to_wrong:>7d}")
    
    # ---- 6. Global score distributions ----
    print(f"\n{'='*70}")
    print(f"7. PREDICTION SCORE DISTRIBUTIONS BY FINAL LABEL (post-OOD)")
    print(f"   (How confident is the model for each class?)")  
    print(f"{'='*70}")
    for cls in sorted(all_pred_scores_by_label.keys()):
        scores = np.array(all_pred_scores_by_label[cls])
        if len(scores) > 100:  # only show classes with enough predictions
            print(f"  {cls:<22s}: n={len(scores):>8d}, mean={scores.mean():.4f}, "
                  f"median={np.median(scores):.4f}, p90={np.percentile(scores, 90):.4f}, "
                  f"p99={np.percentile(scores, 99):.4f}")

    print(f"\n{'='*70}")
    print(f"DIAGNOSIS COMPLETE")
    print(f"{'='*70}")
