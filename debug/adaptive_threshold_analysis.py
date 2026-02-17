"""
Adaptive Per-Prototype Threshold Analysis for Horospherical OOD Detection.

TWO-PHASE approach:
  Phase 1 (CALIBRATION): Run model over ENTIRE TRAINING set.
    - For each GT box, find nearest anchor and collect its max horosphere score.
    - Compute per-class mean and std of max horosphere scores.
    - These define the "known score profile" for each prototype.

  Phase 2 (EVALUATION): Run model over ENTIRE TEST set.
    - Parse XML annotations DIRECTLY to get ALL boxes (known + unknown).
    - For each box, compute horosphere scores and check against per-prototype
      adaptive threshold: tau_k = mean_k - alpha * std_k
    - Evaluate with multiple alpha values to find best operating point.

IMPORTANT: The mmengine YOLO VOC dataloader only produces GT labels for the
known classes. Unknown class objects are SILENTLY DROPPED from gt_instances.
We parse XML annotations DIRECTLY for each image to get ALL bounding boxes
(including unknowns), bypassing the dataloader's class filtering.

Output:
  - Per-class score statistics from training data
  - Unknown recall / Known recall for multiple alpha values
  - Per-unknown-subclass breakdown 
  - JSON with threshold stats for use in test_hyp.py
  - Visualization plots

Usage:
    python debug/adaptive_threshold_analysis.py \
        --config-file configs/IDD_HYP/base.yaml \
        --task IDD_HYP/t1 \
        --ckpt IDD_HYP/t1/horospherical_v2/model_30.pth \
        --hyp_c 1.0 --hyp_dim 256 --clip_r 0.95 \
        --output_dir visualizations/adaptive
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from core import add_config
from core.util.model_ema import add_model_ema_configs
from core.pascal_voc import register_pascal_voc, inital_prompts
from core.hyp_customyoloworld import HypCustomYoloWorld, load_hyp_ckpt
from core.hyperbolic import busemann

from mmengine.config import Config
from mmengine.runner import Runner
from detectron2.engine import default_argument_parser, default_setup
from detectron2.config import get_cfg


# =============================================================================
# Utilities (shared with analyze_unknowns.py)
# =============================================================================

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
        task_yaml = os.path.join("configs", args.task.split('/')[0],
                                 args.task.split('/')[1] + ".yaml")
        if os.path.exists(task_yaml):
            cfg.merge_from_file(task_yaml)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def get_anchor_centers(h, w, device):
    """Get anchor centers for YOLO-World (80x80 + 40x40 + 20x20 = 8400 for 640x640)."""
    strides = [8, 16, 32]
    centers = []
    for s in strides:
        gh, gw = h // s, w // s
        y = torch.arange(gh, device=device).float() * s + s / 2
        x = torch.arange(gw, device=device).float() * s + s / 2
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        centers.append(torch.stack([xx.flatten(), yy.flatten()], dim=-1))
    return torch.cat(centers, dim=0)


def parse_all_xml_boxes(dataset_root, img_id, known_set):
    """
    Parse ALL bounding boxes from an XML annotation, returning each box
    with its true class name and known/unknown status.
    """
    xml_path = os.path.join(dataset_root, 'Annotations', f'{img_id}.xml')
    if not os.path.exists(xml_path):
        return []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        for obj in root.findall('object'):
            name_elem = obj.find('name')
            if name_elem is None or not name_elem.text:
                continue
            cls_name = name_elem.text.strip()
            bbox_elem = obj.find('bndbox')
            if bbox_elem is None:
                continue
            try:
                x1 = float(bbox_elem.find('xmin').text) - 1.0
                y1 = float(bbox_elem.find('ymin').text) - 1.0
                x2 = float(bbox_elem.find('xmax').text)
                y2 = float(bbox_elem.find('ymax').text)
            except (ValueError, AttributeError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            is_known = cls_name in known_set
            boxes.append({
                'bbox': [x1, y1, x2, y2],
                'class_name': cls_name,
                'is_known': is_known,
            })
        return boxes
    except Exception:
        return []


# =============================================================================
# Phase 1: Calibrate from TRAINING data
# =============================================================================

def calibrate_from_train(model, train_loader, known_class_names, dataset_root,
                         hyp_c):
    """
    Run the model over the ENTIRE training set and collect per-class
    horosphere score statistics from GT-assigned anchors.

    Uses the SAME XML parsing approach as analyze_unknowns.py — parses each
    image's XML annotation directly to find known GT boxes and map them to
    the nearest anchor embedding.

    Returns
    -------
    train_stats : dict
        Per-class statistics: {class_name: {'scores': [...], 'mean': float, 'std': float}}
    """
    total_images = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else '?'
    print(f"\n{'='*60}")
    print(f"PHASE 1: CALIBRATING FROM TRAINING DATA")
    print(f"  Known classes: {known_class_names}")
    print(f"  Total images in train loader: {total_images}")
    print(f"  Dataset root: {dataset_root}")
    print(f"{'='*60}")

    model.eval()
    known_set = set(known_class_names)

    prototypes = model.prototypes.detach()
    biases = model.prototype_biases.detach()
    K = prototypes.shape[0]

    # Collect per-class max horosphere scores
    per_class_scores = defaultdict(list)
    samples_count = defaultdict(int)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(train_loader, desc="Phase 1: Train calibration")):
            if i > 0 and i % 500 == 0:
                total = sum(samples_count.values())
                tqdm.write(f"  [batch {i}] {total} known GT boxes processed")

            try:
                data_batch = model.parent.data_preprocessor(batch)

                # Extract features
                x = model.parent.backbone.forward_image(data_batch['inputs'])
                if model.parent.with_neck:
                    if model.parent.mm_neck:
                        txt = model.frozen_embeddings if model.frozen_embeddings is not None else model.embeddings
                        txt = txt.repeat(x[0].shape[0], 1, 1)
                        x = model.parent.neck(x, txt)
                    else:
                        x = model.parent.neck(x)

                hyp_embeddings = model.hyp_projector(x)  # (B, 8400, dim)

                h, w = data_batch['inputs'].shape[-2:]
                anchor_centers = get_anchor_centers(h, w, device=hyp_embeddings.device)

                for b_idx, data_sample in enumerate(data_batch['data_samples']):
                    meta = data_sample.metainfo
                    img_path = meta.get('img_path', '')
                    img_id = Path(img_path).stem if img_path else None
                    if img_id is None:
                        continue

                    scale_factor = meta.get('scale_factor', (1.0, 1.0))
                    pad_param = meta.get('pad_param', None)

                    # Parse XML — only known boxes matter for calibration
                    xml_boxes = parse_all_xml_boxes(dataset_root, img_id, known_set)
                    if not xml_boxes:
                        continue

                    # Filter to known boxes only for calibration
                    known_boxes = [b for b in xml_boxes if b['is_known']]
                    if not known_boxes:
                        continue

                    sx, sy = scale_factor if isinstance(scale_factor, (list, tuple)) else (scale_factor, scale_factor)

                    # Batch: compute all box centers at once
                    centers_list = []
                    for box_info in known_boxes:
                        ox1, oy1, ox2, oy2 = box_info['bbox']
                        x1, y1 = ox1 * sx, oy1 * sy
                        x2, y2 = ox2 * sx, oy2 * sy
                        if pad_param is not None:
                            pad_top, pad_bottom, pad_left, pad_right = pad_param
                            x1 += pad_left; y1 += pad_top
                            x2 += pad_left; y2 += pad_top
                        cx = max(0, min((x1 + x2) / 2.0, w - 1))
                        cy = max(0, min((y1 + y2) / 2.0, h - 1))
                        centers_list.append([cx, cy])

                    # Vectorized anchor matching for all boxes in this image
                    box_centers = torch.tensor(centers_list, device=hyp_embeddings.device)  # (M, 2)
                    # (M, 1, 2) - (1, A, 2) -> (M, A)
                    dists = ((box_centers.unsqueeze(1) - anchor_centers.unsqueeze(0)) ** 2).sum(-1)
                    nearest_indices = dists.argmin(dim=1)  # (M,)

                    # Batch busemann: gather all M embeddings, compute scores in one call
                    embs = hyp_embeddings[b_idx, nearest_indices]  # (M, dim)
                    B_vals = busemann(prototypes, embs, c=hyp_c)  # (M, K)
                    horo_scores = -B_vals + biases  # (M, K)
                    max_horos = horo_scores.max(dim=-1).values  # (M,)

                    for j, box_info in enumerate(known_boxes):
                        cls_name = box_info['class_name']
                        per_class_scores[cls_name].append(max_horos[j].item())
                        samples_count[cls_name] += 1

            except Exception as e:
                print(f"  Error batch {i}: {e}")
                import traceback; traceback.print_exc()
                continue

    # Compute statistics
    train_stats = {}
    print(f"\n  Training calibration summary:")
    print(f"  {'Class':<25s} {'Count':>7s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
    print(f"  {'-'*65}")

    for cls_name in known_class_names:
        scores = np.array(per_class_scores.get(cls_name, []))
        if len(scores) > 0:
            train_stats[cls_name] = {
                'scores': scores,
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'min': float(scores.min()),
                'max': float(scores.max()),
                'count': len(scores),
            }
            print(f"  {cls_name:<25s} {len(scores):>7d} {scores.mean():>8.4f} "
                  f"{scores.std():>8.4f} {scores.min():>8.4f} {scores.max():>8.4f}")
        else:
            # Fallback — should not happen with full train data
            train_stats[cls_name] = {
                'scores': np.array([0.0]),
                'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 0.0, 'count': 0,
            }
            print(f"  {cls_name:<25s} {'0 (WARN)':>7s}")

    total = sum(samples_count.values())
    print(f"  {'-'*65}")
    print(f"  Total known GT boxes from training: {total}")

    return train_stats


# =============================================================================
# Phase 2: Evaluate on TEST data with adaptive thresholds
# =============================================================================

def evaluate_test_adaptive(model, test_loader, known_class_names, dataset_root,
                           hyp_c, train_stats, alphas):
    """
    Run the model over the ENTIRE test set and evaluate adaptive thresholding
    with multiple alpha values. Parses XML to get both known and unknown boxes.

    For each GT box:
      1. Find nearest anchor, get horosphere scores
      2. Determine assigned prototype (argmax of scores)
      3. For each alpha: check if max_score < tau_k (mean_k - alpha*std_k)
         If yes → detected as unknown

    Parameters
    ----------
    train_stats : dict
        From calibrate_from_train(): per-class {mean, std} from training data.
    alphas : list of float
        Alpha values to evaluate.

    Returns
    -------
    class_data : dict
        Per-class raw data (scores, assigned protos, etc.)
    samples_count : dict
        Per-class sample counts
    alpha_results : list of dict
        Results for each alpha value
    """
    total_images = len(test_loader.dataset) if hasattr(test_loader, 'dataset') else '?'
    print(f"\n{'='*60}")
    print(f"PHASE 2: EVALUATING ON TEST DATA (ADAPTIVE THRESHOLD)")
    print(f"  Known classes: {known_class_names}")
    print(f"  Total images in test loader: {total_images}")
    print(f"  Alphas to evaluate: {alphas}")
    print(f"{'='*60}")

    model.eval()
    known_set = set(known_class_names)

    prototypes = model.prototypes.detach()
    biases = model.prototype_biases.detach()
    K = prototypes.shape[0]

    # Per-class test data collection
    class_data = defaultdict(lambda: {
        'horo_scores_all': [],
        'max_horo_score': [],
        'assigned_proto': [],
        'embedding_norm': [],
        'is_known': None,
    })
    samples_count = defaultdict(int)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Phase 2: Test evaluation")):
            if i > 0 and i % 500 == 0:
                unk_total = sum(v for k, v in samples_count.items() if k not in known_set)
                known_total = sum(v for k, v in samples_count.items() if k in known_set)
                n_classes = len(samples_count)
                tqdm.write(f"  [batch {i}] {n_classes} classes seen | "
                           f"known={known_total} | unknown={unk_total}")

            try:
                data_batch = model.parent.data_preprocessor(batch)

                x = model.parent.backbone.forward_image(data_batch['inputs'])
                if model.parent.with_neck:
                    if model.parent.mm_neck:
                        txt = model.frozen_embeddings if model.frozen_embeddings is not None else model.embeddings
                        txt = txt.repeat(x[0].shape[0], 1, 1)
                        x = model.parent.neck(x, txt)
                    else:
                        x = model.parent.neck(x)

                hyp_embeddings = model.hyp_projector(x)

                h, w = data_batch['inputs'].shape[-2:]
                anchor_centers = get_anchor_centers(h, w, device=hyp_embeddings.device)

                for b_idx, data_sample in enumerate(data_batch['data_samples']):
                    meta = data_sample.metainfo
                    img_path = meta.get('img_path', '')
                    img_id = Path(img_path).stem if img_path else None
                    if img_id is None:
                        continue

                    scale_factor = meta.get('scale_factor', (1.0, 1.0))
                    pad_param = meta.get('pad_param', None)

                    xml_boxes = parse_all_xml_boxes(dataset_root, img_id, known_set)
                    if not xml_boxes:
                        continue

                    sx, sy = scale_factor if isinstance(scale_factor, (list, tuple)) else (scale_factor, scale_factor)

                    # Batch: compute all box centers at once
                    centers_list = []
                    for box_info in xml_boxes:
                        ox1, oy1, ox2, oy2 = box_info['bbox']
                        x1, y1 = ox1 * sx, oy1 * sy
                        x2, y2 = ox2 * sx, oy2 * sy
                        if pad_param is not None:
                            pad_top, pad_bottom, pad_left, pad_right = pad_param
                            x1 += pad_left; y1 += pad_top
                            x2 += pad_left; y2 += pad_top
                        cx = max(0, min((x1 + x2) / 2.0, w - 1))
                        cy = max(0, min((y1 + y2) / 2.0, h - 1))
                        centers_list.append([cx, cy])

                    # Vectorized anchor matching for all boxes
                    box_centers = torch.tensor(centers_list, device=hyp_embeddings.device)  # (M, 2)
                    dists = ((box_centers.unsqueeze(1) - anchor_centers.unsqueeze(0)) ** 2).sum(-1)
                    nearest_indices = dists.argmin(dim=1)  # (M,)

                    # Batch busemann: all M embeddings at once
                    embs = hyp_embeddings[b_idx, nearest_indices]  # (M, dim)
                    B_vals = busemann(prototypes, embs, c=hyp_c)  # (M, K)
                    horo_scores_batch = -B_vals + biases  # (M, K)
                    max_horos, assigned_batch = horo_scores_batch.max(dim=-1)  # (M,)
                    norms = embs.norm(dim=-1)  # (M,)

                    for j, box_info in enumerate(xml_boxes):
                        cls_name = box_info['class_name']
                        is_known = box_info['is_known']

                        class_data[cls_name]['horo_scores_all'].append(horo_scores_batch[j].cpu())
                        class_data[cls_name]['max_horo_score'].append(max_horos[j].item())
                        class_data[cls_name]['assigned_proto'].append(assigned_batch[j].item())
                        class_data[cls_name]['embedding_norm'].append(norms[j].item())
                        class_data[cls_name]['is_known'] = is_known
                        samples_count[cls_name] += 1

            except Exception as e:
                print(f"  Error batch {i}: {e}")
                import traceback; traceback.print_exc()
                continue

    # Print collection summary
    print(f"\n  Test collection summary:")
    print(f"  {'Class':<25s} {'Type':<8s} {'Count':>6s}")
    print(f"  {'-'*45}")
    for cls_key in sorted(samples_count.keys()):
        is_known = class_data[cls_key]['is_known']
        tag = "KNOWN" if is_known else "UNKNOWN"
        print(f"  {cls_key:<25s} {tag:<8s} {samples_count[cls_key]:>6d}")

    total_known = sum(v for k, v in samples_count.items() if class_data[k]['is_known'])
    total_unk = sum(v for k, v in samples_count.items() if not class_data[k]['is_known'])
    print(f"  {'-'*45}")
    print(f"  Total known: {total_known}, Total unknown: {total_unk}")

    # =========================================================================
    # Compute per-class test statistics table
    # =========================================================================
    print(f"\n{'='*100}")
    print(f"TEST SET PER-CLASS STATISTICS")
    print(f"{'='*100}")
    print(f"{'Class':<25s} {'Type':<8s} {'N':>5s} {'HoroMean':>9s} {'HoroStd':>8s} "
          f"{'HoroMin':>8s} {'%Neg':>6s} {'NormMean':>9s}")
    print(f"{'-'*80}")

    for cls_key in sorted(class_data.keys(),
                          key=lambda k: (not class_data[k]['is_known'], k)):
        d = class_data[cls_key]
        if len(d['max_horo_score']) == 0:
            continue
        scores = np.array(d['max_horo_score'])
        norms = np.array(d['embedding_norm'])
        is_known = d['is_known']
        tag = "KNOWN" if is_known else "UNK"
        pct_neg = (scores < 0).mean()
        print(f"{cls_key:<25s} {tag:<8s} {len(scores):>5d} "
              f"{scores.mean():>9.3f} {scores.std():>8.3f} "
              f"{scores.min():>8.3f} {pct_neg:>5.0%} {norms.mean():>9.4f}")

    # =========================================================================
    # Evaluate adaptive thresholds for each alpha
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"ADAPTIVE THRESHOLD EVALUATION")
    print(f"  tau_k = train_mean_k - alpha * train_std_k")
    print(f"{'='*60}")

    # Print calibration stats used
    print(f"\n  Training calibration stats used:")
    for cls_name in known_class_names:
        s = train_stats[cls_name]
        print(f"    {cls_name:<20s}: mean={s['mean']:.4f}, std={s['std']:.4f}, n={s['count']}")

    alpha_results = []

    print(f"\n  {'Alpha':<8s} {'UnkRecall':>10s} {'KnownRecall':>12s} {'UnkPrec':>9s} "
          f"{'F1':>6s} {'UnkCaught':>10s} {'KnownLost':>10s}")
    print(f"  {'-'*70}")

    for alpha in alphas:
        # Compute per-class thresholds from TRAINING stats
        class_thresholds = {}
        for cls_name in known_class_names:
            mu = train_stats[cls_name]['mean']
            sigma = train_stats[cls_name]['std']
            class_thresholds[cls_name] = mu - alpha * sigma

        unk_caught = 0
        known_lost = 0
        total_unk_samples = 0
        total_known_samples = 0

        # Per-unknown-subclass tracking
        per_unk_class = defaultdict(lambda: {'total': 0, 'caught': 0})

        for cls_key, d in class_data.items():
            is_known = d['is_known']
            for score, assigned_idx in zip(d['max_horo_score'], d['assigned_proto']):
                # Get the threshold for the assigned prototype
                assigned_name = known_class_names[assigned_idx] if assigned_idx < len(known_class_names) else None
                if assigned_name is None or assigned_name not in class_thresholds:
                    continue

                tau_k = class_thresholds[assigned_name]
                below_threshold = score < tau_k

                if is_known:
                    total_known_samples += 1
                    if below_threshold:
                        known_lost += 1
                else:
                    total_unk_samples += 1
                    per_unk_class[cls_key]['total'] += 1
                    if below_threshold:
                        unk_caught += 1
                        per_unk_class[cls_key]['caught'] += 1

        unk_recall = unk_caught / max(total_unk_samples, 1)
        known_recall = 1.0 - known_lost / max(total_known_samples, 1)
        unk_precision = unk_caught / max(unk_caught + known_lost, 1)
        f1 = 2 * unk_precision * unk_recall / max(unk_precision + unk_recall, 1e-8)

        print(f"  {alpha:<8.2f} {unk_recall:>10.1%} {known_recall:>12.1%} "
              f"{unk_precision:>9.1%} {f1:>6.3f} "
              f"{unk_caught:>10d} {known_lost:>10d}")

        alpha_results.append({
            'alpha': alpha,
            'unk_recall': float(unk_recall),
            'known_recall': float(known_recall),
            'unk_precision': float(unk_precision),
            'f1': float(f1),
            'unk_caught': unk_caught,
            'known_lost': known_lost,
            'total_unk': total_unk_samples,
            'total_known': total_known_samples,
            'thresholds': {k: float(v) for k, v in class_thresholds.items()},
            'per_unk_class': {k: dict(v) for k, v in per_unk_class.items()},
        })

    # =========================================================================
    # Per-unknown-subclass breakdown for best alpha
    # =========================================================================
    best_result = max(alpha_results, key=lambda x: x['f1'])
    best_alpha = best_result['alpha']

    print(f"\n{'='*60}")
    print(f"BEST ALPHA = {best_alpha:.2f} (F1={best_result['f1']:.3f})")
    print(f"  UnkRecall={best_result['unk_recall']:.1%}, "
          f"KnownRecall={best_result['known_recall']:.1%}")
    print(f"{'='*60}")

    print(f"\n  Per-class thresholds (tau_k = mean_k - {best_alpha}*std_k):")
    for cls_name in known_class_names:
        tau = best_result['thresholds'][cls_name]
        s = train_stats[cls_name]
        print(f"    {cls_name:<20s}: tau={tau:.4f}  (mean={s['mean']:.4f}, std={s['std']:.4f})")

    print(f"\n  Per-unknown-subclass recall (alpha={best_alpha:.2f}):")
    print(f"  {'Unknown Class':<25s} {'Detected':>10s} {'Total':>7s} {'Recall':>8s}")
    print(f"  {'-'*55}")
    for cls_key in sorted(best_result['per_unk_class'].keys()):
        info = best_result['per_unk_class'][cls_key]
        recall = info['caught'] / max(info['total'], 1) * 100
        print(f"  {cls_key:<25s} {info['caught']:>10d} {info['total']:>7d} {recall:>7.1f}%")

    return dict(class_data), dict(samples_count), alpha_results


# =============================================================================
# Plotting
# =============================================================================

def plot_adaptive_analysis(class_data, known_class_names, train_stats,
                           alpha_results, save_dir):
    """Generate visualization plots for adaptive threshold analysis."""
    known_set = set(known_class_names)
    num_known = len(known_class_names)

    known_keys = sorted([k for k in class_data if class_data[k]['is_known']
                         and len(class_data[k]['max_horo_score']) > 0])
    unknown_keys = sorted([k for k in class_data if not class_data[k]['is_known']
                          and len(class_data[k]['max_horo_score']) > 0])

    # Collect known/unknown scores
    known_scores = []
    unknown_scores = []
    for k in known_keys:
        known_scores.extend(class_data[k]['max_horo_score'])
    for k in unknown_keys:
        unknown_scores.extend(class_data[k]['max_horo_score'])
    known_scores = np.array(known_scores)
    unknown_scores = np.array(unknown_scores)

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle('Adaptive Threshold Analysis (Per-Prototype, Calibrated from Training)',
                 fontsize=14, fontweight='bold')

    # ---- Plot 1: Known vs Unknown score distributions ----
    ax = axes[0, 0]
    all_min = min(known_scores.min(), unknown_scores.min()) - 0.2
    all_max = max(known_scores.max(), unknown_scores.max()) + 0.2
    bins = np.linspace(all_min, all_max, 60)
    ax.hist(known_scores, bins=bins, alpha=0.6, color='blue',
            label=f'Known (n={len(known_scores)})', density=True)
    ax.hist(unknown_scores, bins=bins, alpha=0.6, color='red',
            label=f'Unknown (n={len(unknown_scores)})', density=True)

    # Show best alpha threshold range
    best = max(alpha_results, key=lambda x: x['f1'])
    thresholds = list(best['thresholds'].values())
    ax.axvspan(min(thresholds), max(thresholds), alpha=0.15, color='green',
               label=f'Threshold range (α={best["alpha"]:.1f})')
    ax.axvline(np.mean(thresholds), color='green', linestyle='--', linewidth=2,
               label=f'Mean threshold={np.mean(thresholds):.2f}')
    ax.set_xlabel('Max Horosphere Score')
    ax.set_ylabel('Density')
    ax.set_title('Known vs Unknown: Score Distributions')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Plot 2: Alpha sweep — UnkRecall vs KnownRecall ----
    ax = axes[0, 1]
    alphas_plot = [r['alpha'] for r in alpha_results]
    unk_recalls = [r['unk_recall'] * 100 for r in alpha_results]
    known_recalls = [r['known_recall'] * 100 for r in alpha_results]
    f1s = [r['f1'] * 100 for r in alpha_results]

    ax.plot(alphas_plot, unk_recalls, 'r-o', label='Unknown Recall', linewidth=2, markersize=5)
    ax.plot(alphas_plot, known_recalls, 'b-o', label='Known Recall', linewidth=2, markersize=5)
    ax.plot(alphas_plot, f1s, 'g--s', label='F1 Score', linewidth=2, markersize=5)
    ax.axvline(best['alpha'], color='green', linestyle=':', alpha=0.5,
               label=f'Best α={best["alpha"]:.1f}')
    ax.set_xlabel('Alpha (threshold strictness)')
    ax.set_ylabel('Rate (%)')
    ax.set_title('Alpha Sweep: Recall / F1 Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    # ---- Plot 3: Per-class boxplot with adaptive thresholds ----
    ax = axes[0, 2]
    plot_data = []
    plot_labels = []
    plot_colors = []
    threshold_positions = []

    for cls_key in known_keys + unknown_keys:
        d = class_data[cls_key]
        scores = d['max_horo_score']
        is_known = d['is_known']
        label = cls_key if is_known else f'*{cls_key}'
        plot_data.append(scores)
        plot_labels.append(label)
        plot_colors.append('lightblue' if is_known else 'salmon')

    bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True, vert=True)
    for patch, color in zip(bp['boxes'], plot_colors):
        patch.set_facecolor(color)

    # Overlay per-class thresholds for best alpha
    for idx, cls_key in enumerate(known_keys):
        if cls_key in best['thresholds']:
            tau = best['thresholds'][cls_key]
            ax.plot(idx + 1, tau, 'gv', markersize=10, zorder=5)

    ax.set_ylabel('Max Horosphere Score')
    ax.set_title(f'Per-Class Scores + Thresholds (▼α={best["alpha"]:.1f})\n(* = unknown)')
    ax.tick_params(axis='x', rotation=60, labelsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # ---- Plot 4: Per-unknown-subclass recall at best alpha ----
    ax = axes[1, 0]
    unk_classes = sorted(best['per_unk_class'].keys())
    if unk_classes:
        recalls = [best['per_unk_class'][k]['caught'] / max(best['per_unk_class'][k]['total'], 1) * 100
                    for k in unk_classes]
        totals = [best['per_unk_class'][k]['total'] for k in unk_classes]
        bars = ax.barh(range(len(unk_classes)), recalls, color='salmon', edgecolor='darkred')
        ax.set_yticks(range(len(unk_classes)))
        ax.set_yticklabels([f'{k} (n={t})' for k, t in zip(unk_classes, totals)], fontsize=9)
        ax.set_xlabel('Recall (%)')
        ax.set_title(f'Per-Unknown-Subclass Recall (α={best["alpha"]:.1f})')
        ax.set_xlim(0, 105)
        for bar, val in zip(bars, recalls):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=8)
        ax.grid(True, alpha=0.3, axis='x')

    # ---- Plot 5: Train vs Test score distributions per known class ----
    ax = axes[1, 1]
    x_pos = np.arange(len(known_class_names))
    width = 0.35

    train_means = [train_stats[c]['mean'] for c in known_class_names]
    train_stds = [train_stats[c]['std'] for c in known_class_names]
    test_means = []
    test_stds = []
    for c in known_class_names:
        if c in class_data and len(class_data[c]['max_horo_score']) > 0:
            scores = np.array(class_data[c]['max_horo_score'])
            test_means.append(scores.mean())
            test_stds.append(scores.std())
        else:
            test_means.append(0)
            test_stds.append(0)

    ax.bar(x_pos - width/2, train_means, width, yerr=train_stds,
           label='Train', color='steelblue', alpha=0.7, capsize=3)
    ax.bar(x_pos + width/2, test_means, width, yerr=test_stds,
           label='Test', color='darkorange', alpha=0.7, capsize=3)

    # Show thresholds
    for idx, c in enumerate(known_class_names):
        tau = best['thresholds'].get(c, 0)
        ax.plot(idx, tau, 'gv', markersize=8, zorder=5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(known_class_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Max Horosphere Score')
    ax.set_title(f'Train vs Test Score Distribution + Threshold (▼)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # ---- Plot 6: Prototype assignment confusion (unknowns) ----
    ax = axes[1, 2]
    if unknown_keys:
        unk_matrix = np.zeros((len(unknown_keys), num_known))
        for r, cls_key in enumerate(unknown_keys):
            assigned = np.array(class_data[cls_key]['assigned_proto'])
            for p in range(num_known):
                unk_matrix[r, p] = (assigned == p).sum()
        if unk_matrix.sum() > 0:
            unk_matrix_pct = unk_matrix / unk_matrix.sum(axis=1, keepdims=True).clip(1)
        else:
            unk_matrix_pct = unk_matrix
        im = ax.imshow(unk_matrix_pct, aspect='auto', cmap='Reds')
        ax.set_xticks(range(num_known))
        ax.set_xticklabels(known_class_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(unknown_keys)))
        ax.set_yticklabels(unknown_keys, fontsize=8)
        ax.set_xlabel('Assigned Prototype')
        ax.set_ylabel('True Unknown Class')
        ax.set_title('Unknown → Prototype Assignment')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    path = os.path.join(save_dir, 'adaptive_threshold_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")

    # ---- Additional: Per-unknown-subclass score distributions ----
    if len(unknown_keys) > 0:
        n_unk = len(unknown_keys)
        ncols = min(n_unk, 4)
        nrows = (n_unk + ncols - 1) // ncols
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                                   squeeze=False)

        for idx, cls_key in enumerate(unknown_keys):
            ax = axes2[idx // ncols][idx % ncols]
            d = class_data[cls_key]
            scores = np.array(d['max_horo_score'])
            assigned = np.array(d['assigned_proto'])

            for proto_id in range(num_known):
                mask = assigned == proto_id
                if mask.sum() > 0:
                    pname = known_class_names[proto_id]
                    ax.hist(scores[mask], bins=30, alpha=0.5,
                            label=f'{pname} ({mask.sum()})')
                    # Show that prototype's threshold
                    if pname in best['thresholds']:
                        ax.axvline(best['thresholds'][pname], color='green',
                                  linestyle='--', linewidth=1.5, alpha=0.7)

            caught = best['per_unk_class'].get(cls_key, {}).get('caught', 0)
            total = best['per_unk_class'].get(cls_key, {}).get('total', len(scores))
            recall = caught / max(total, 1) * 100
            ax.set_title(f'{cls_key}\n(n={len(scores)}, recall={recall:.0f}% @ α={best["alpha"]:.1f})',
                        fontsize=9)
            ax.set_xlabel('Max Horo Score')
            ax.legend(fontsize=6, loc='upper left')
            ax.grid(True, alpha=0.3)

        for idx in range(len(unknown_keys), nrows * ncols):
            axes2[idx // ncols][idx % ncols].set_visible(False)

        plt.tight_layout()
        path2 = os.path.join(save_dir, 'adaptive_per_subclass.png')
        plt.savefig(path2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path2}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--task", default="IDD_HYP/t1")
    parser.add_argument("--ckpt", default="IDD_HYP/t1/horospherical_v2/model_30.pth")
    parser.add_argument("--hyp_c", type=float, default=1.0)
    parser.add_argument("--hyp_dim", type=int, default=256)
    parser.add_argument("--clip_r", type=float, default=0.95)
    parser.add_argument("--output_dir", default="visualizations/adaptive")

    args = parser.parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)

    task_name = args.task.split('/')[0]
    split_name = args.task.split('/')[1]
    base_dataset = task_name.replace('_HYP', '')
    dataset_key = base_dataset

    data_split = f"{base_dataset}/{split_name}"
    data_register = Register('./datasets/', data_split, cfg, dataset_key)
    data_register.register_dataset()

    all_class_names = list(inital_prompts()[dataset_key])
    unknown_index = cfg.TEST.PREV_INTRODUCED_CLS + cfg.TEST.CUR_INTRODUCED_CLS
    known_class_names = all_class_names[:unknown_index]

    print(f"\n=== Configuration ===")
    print(f"  Task: {args.task}")
    print(f"  Known classes ({unknown_index}): {known_class_names}")
    print(f"  Curvature: {args.hyp_c}")
    print(f"  Checkpoint: {args.ckpt}")

    # Model config
    config_file = os.path.join("./configs", task_name, split_name + ".py")
    cfgY = Config.fromfile(config_file)
    cfgY.work_dir = "."

    # Initialize YOLO-World
    runner = Runner.from_cfg(cfgY)
    runner.call_hook("before_run")
    runner.load_or_resume()
    runner.model.reparameterize([known_class_names])
    runner.model.eval()

    # Build BOTH train and test loaders
    # Train loader for calibration (Phase 1)
    train_loader = Runner.build_dataloader(cfgY.trlder)
    # Test loader for evaluation (Phase 2)
    test_loader = Runner.build_dataloader(cfgY.test_dataloader)

    # Build hyperbolic model
    model = HypCustomYoloWorld(
        runner.model, unknown_index,
        hyp_c=args.hyp_c, hyp_dim=args.hyp_dim, clip_r=args.clip_r
    )

    print(f"\n=== Loading Checkpoint: {args.ckpt} ===")
    with torch.no_grad():
        model = load_hyp_ckpt(model, args.ckpt,
                              cfg.TEST.PREV_INTRODUCED_CLS,
                              cfg.TEST.CUR_INTRODUCED_CLS, eval=True)
        model = model.cuda()
        # IMPORTANT: add_generic_text MUTATES known_class_names in-place (appends 'object')
        # Copy the 9 known class names BEFORE mutation for calibration/evaluation
        known_class_names_9 = list(known_class_names)
        model.add_generic_text(known_class_names, generic_prompt='object', alpha=0.4)
    model.eval()

    prototypes = model.prototypes.detach()  # (9, dim) — only known prototypes
    biases = model.prototype_biases.detach()  # (9,)

    print(f"\n=== Model Info ===")
    print(f"  Prototypes: {prototypes.shape[0]} "
          f"(norms: {[f'{n:.4f}' for n in prototypes.norm(dim=-1).tolist()]})")
    print(f"  Biases: {[f'{b:.4f}' for b in biases.cpu().tolist()]}")
    print(f"  Known classes (9): {known_class_names_9}")
    print(f"  Text classes (10, after add_generic_text): {known_class_names}")

    # =========================================================================
    # Phase 1: Calibrate from training data
    # Use known_class_names_9 (9 known classes, NOT the mutated 10-class list)
    # =========================================================================
    train_stats = calibrate_from_train(
        model, train_loader, known_class_names_9,
        dataset_root='./datasets',
        hyp_c=args.hyp_c,
    )

    # =========================================================================
    # Phase 2: Evaluate on test data with multiple alphas
    # =========================================================================
    alphas = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

    class_data, samples_count, alpha_results = evaluate_test_adaptive(
        model, test_loader, known_class_names_9,
        dataset_root='./datasets',
        hyp_c=args.hyp_c,
        train_stats=train_stats,
        alphas=alphas,
    )

    if not class_data:
        print("ERROR: No test embeddings collected!")
        sys.exit(1)

    # =========================================================================
    # Save results
    # =========================================================================
    os.makedirs(args.output_dir, exist_ok=True)

    # Plots
    plot_adaptive_analysis(class_data, known_class_names_9, train_stats,
                           alpha_results, args.output_dir)

    # Save JSON with all statistics
    ckpt_name = Path(args.ckpt).stem
    json_path = os.path.join(args.output_dir, f"adaptive_stats_{ckpt_name}.json")

    best_result = max(alpha_results, key=lambda x: x['f1'])

    def convert_types(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): convert_types(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert_types(v) for v in obj]
        return obj

    save_data = convert_types({
        'checkpoint': args.ckpt,
        'class_names': known_class_names_9,
        'train_calibration': {
            cls_name: {
                'mean': train_stats[cls_name]['mean'],
                'std': train_stats[cls_name]['std'],
                'count': train_stats[cls_name]['count'],
            }
            for cls_name in known_class_names_9
        },
        'alpha_results': [
            {
                'alpha': r['alpha'],
                'unk_recall': r['unk_recall'],
                'known_recall': r['known_recall'],
                'unk_precision': r['unk_precision'],
                'f1': r['f1'],
                'thresholds': r['thresholds'],
                'per_unk_class': r['per_unk_class'],
            }
            for r in alpha_results
        ],
        'best_alpha': best_result['alpha'],
        'best_f1': best_result['f1'],
        'best_thresholds': best_result['thresholds'],
    })

    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # Save NPZ with raw data for further analysis
    npz_path = os.path.join(args.output_dir, f"adaptive_data_{ckpt_name}.npz")

    known_all_scores = []
    unknown_all_scores = []
    for k in sorted(class_data.keys()):
        d = class_data[k]
        if d['is_known']:
            known_all_scores.extend(d['max_horo_score'])
        else:
            unknown_all_scores.extend(d['max_horo_score'])

    np.savez(npz_path,
             known_scores=np.array(known_all_scores),
             unknown_scores=np.array(unknown_all_scores),
             prototypes=prototypes.cpu().numpy(),
             biases=biases.cpu().numpy(),
             class_names=np.array(known_class_names_9, dtype=object),
             alphas=np.array(alphas),
             unk_recalls=np.array([r['unk_recall'] for r in alpha_results]),
             known_recalls=np.array([r['known_recall'] for r in alpha_results]),
             f1s=np.array([r['f1'] for r in alpha_results]))
    print(f"  Saved: {npz_path}")

    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"  Best alpha: {best_result['alpha']:.2f}")
    print(f"  Best F1: {best_result['f1']:.3f}")
    print(f"  UnkRecall: {best_result['unk_recall']:.1%}")
    print(f"  KnownRecall: {best_result['known_recall']:.1%}")
    print(f"\n  To use in test_hyp.py, set --ood_threshold based on these results.")
    print(f"{'='*60}")
