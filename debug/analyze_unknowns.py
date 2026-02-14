"""
Unknown Embedding Analysis for Horospherical Classifiers.

Projects ALL test embeddings (known + unknown) through the trained model and
computes detailed statistics to understand OOD detection behavior.

IMPORTANT: The mmengine YOLO VOC dataloader only produces GT labels for the 9
known classes. Unknown class objects are SILENTLY DROPPED from gt_instances.
Therefore, we parse XML annotations DIRECTLY for each image to get ALL bounding
boxes (including unknowns), bypassing the dataloader's class filtering.

Analysis includes:
- Per-class horosphere score distributions (known vs unknown)
- Hyperbolic distance to nearest prototype
- Class-wise adaptive threshold analysis
- Score overlap analysis between known/unknown
- True unknown subclass breakdown via XML annotation parsing
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


def compute_poincare_distance(x, y, c=1.0):
    """
    Poincare ball distance.
    x: (N, D), y: (K, D) -> returns (N, K)
    """
    x_sq = (x ** 2).sum(-1, keepdim=True)
    y_sq = (y ** 2).sum(-1, keepdim=True).T
    diff_sq = ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(-1)

    denom = (1.0 - c * x_sq) * (1.0 - c * y_sq)
    denom = denom.clamp(min=1e-8)
    arg = 1.0 + 2.0 * c * diff_sq / denom
    arg = arg.clamp(min=1.0 + 1e-7)

    return (1.0 / (c ** 0.5)) * torch.acosh(arg)


def parse_all_xml_boxes(dataset_root, img_id, known_set):
    """
    Parse ALL bounding boxes from an XML annotation, returning each box
    with its true class name and known/unknown status.

    This bypasses the dataloader's class filtering to get unknowns.
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


def collect_all_embeddings(model, data_loader, known_class_names, dataset_root,
                           hyp_c):
    """
    Collect hyperbolic embeddings for BOTH known and unknown GT boxes.
    Iterates through the ENTIRE dataloader — no subsampling.

    KEY INSIGHT: The mmengine YOLO dataloader only returns gt_instances for
    the 9 known classes. Unknown class objects are SILENTLY DROPPED.

    Therefore, for each image we:
    1. Run the model to get hyperbolic embeddings at every anchor position
    2. Parse the XML annotation DIRECTLY to find ALL bounding boxes
    3. Map original bbox coords -> preprocessed image coords using scale/pad
    4. For each bbox, find nearest anchor and grab its embedding
    5. Compute horosphere scores and Poincare distances

    This gives us true unknown subclass names (bus, truck, animal, etc.)
    rather than a single 'unknown' label.
    """
    total_images = len(data_loader.dataset) if hasattr(data_loader, 'dataset') else '?'
    print(f"\n{'='*60}")
    print(f"COLLECTING EMBEDDINGS — FULL DATASET (no subsampling)")
    print(f"  Known classes: {known_class_names}")
    print(f"  Total images in dataloader: {total_images}")
    print(f"  Dataset root: {dataset_root}")
    print(f"{'='*60}")

    model.eval()
    known_set = set(known_class_names)

    class_data = defaultdict(lambda: {
        'horo_scores_all': [],
        'max_horo_score': [],
        'assigned_proto': [],
        'min_poincare_dist': [],
        'nearest_proto': [],
        'embedding_norm': [],
        'is_known': None,
    })
    samples_count = defaultdict(int)

    prototypes = model.prototypes.detach()
    biases = model.prototype_biases.detach()
    K = prototypes.shape[0]
    proto_inside = prototypes * 0.99

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc="Collecting (full dataset)")):
            # Log progress every 500 batches
            if i > 0 and i % 500 == 0:
                unk_total = sum(v for k, v in samples_count.items() if k not in known_set)
                known_total = sum(v for k, v in samples_count.items() if k in known_set)
                n_classes = len(samples_count)
                tqdm.write(f"  [batch {i}] {n_classes} classes seen | known={known_total} | unknown={unk_total}")
            try:
                data_batch = model.parent.data_preprocessor(batch)

                # Extract features manually (same pattern as visualize_simple.py)
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

                    xml_boxes = parse_all_xml_boxes(dataset_root, img_id, known_set)
                    if not xml_boxes:
                        continue

                    for box_info in xml_boxes:
                        cls_name = box_info['class_name']
                        is_known = box_info['is_known']
                        cls_key = cls_name

                        ox1, oy1, ox2, oy2 = box_info['bbox']

                        # Map original coords to preprocessed image coords
                        sx, sy = scale_factor if isinstance(scale_factor, (list, tuple)) else (scale_factor, scale_factor)
                        x1 = ox1 * sx
                        y1 = oy1 * sy
                        x2 = ox2 * sx
                        y2 = oy2 * sy

                        if pad_param is not None:
                            pad_top, pad_bottom, pad_left, pad_right = pad_param
                            x1 += pad_left
                            y1 += pad_top
                            x2 += pad_left
                            y2 += pad_top

                        cx = max(0, min((x1 + x2) / 2.0, w - 1))
                        cy = max(0, min((y1 + y2) / 2.0, h - 1))

                        dists = (anchor_centers[:, 0] - cx)**2 + (anchor_centers[:, 1] - cy)**2
                        nearest_idx = dists.argmin().item()

                        emb = hyp_embeddings[b_idx, nearest_idx]

                        B_vals = busemann(prototypes, emb.unsqueeze(0), c=hyp_c)
                        horo_scores = (-B_vals + biases).squeeze(0)

                        p_dists = compute_poincare_distance(
                            emb.unsqueeze(0), proto_inside, c=hyp_c
                        ).squeeze(0)

                        max_horo, assigned = horo_scores.max(dim=0)
                        min_dist, nearest = p_dists.min(dim=0)

                        # Skip raw embedding/poincare_dists storage to save memory on full dataset
                        class_data[cls_key]['horo_scores_all'].append(horo_scores.cpu())
                        class_data[cls_key]['max_horo_score'].append(max_horo.item())
                        class_data[cls_key]['assigned_proto'].append(assigned.item())
                        class_data[cls_key]['min_poincare_dist'].append(min_dist.item())
                        class_data[cls_key]['nearest_proto'].append(nearest.item())
                        class_data[cls_key]['embedding_norm'].append(emb.norm().item())
                        class_data[cls_key]['is_known'] = is_known

                        samples_count[cls_key] += 1

            except Exception as e:
                print(f"  Error batch {i}: {e}")
                import traceback; traceback.print_exc()
                continue

    print(f"\n  Collection summary:")
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

    return dict(class_data), dict(samples_count)


def compute_statistics(class_data, samples_count, known_class_names):
    """Compute comprehensive statistics for known vs unknown classes."""
    known_set = set(known_class_names)

    stats = {
        'known': defaultdict(list),
        'unknown': defaultdict(list),
        'per_class': {},
    }

    for cls_key in sorted(class_data.keys()):
        d = class_data[cls_key]
        if len(d['max_horo_score']) == 0:
            continue

        is_known = d['is_known']
        group = 'known' if is_known else 'unknown'

        max_horo = np.array(d['max_horo_score'])
        min_dist = np.array(d['min_poincare_dist'])
        emb_norm = np.array(d['embedding_norm'])
        assigned = np.array(d['assigned_proto'])
        nearest = np.array(d['nearest_proto'])

        cls_stats = {
            'class_key': cls_key,
            'is_known': is_known,
            'n_samples': len(max_horo),
            'horo_score': {
                'mean': float(max_horo.mean()),
                'std': float(max_horo.std()),
                'min': float(max_horo.min()),
                'max': float(max_horo.max()),
                'median': float(np.median(max_horo)),
                'pct_negative': float((max_horo < 0).mean()),
                'pct_below_0.5': float((max_horo < 0.5).mean()),
            },
            'poincare_dist': {
                'mean': float(min_dist.mean()),
                'std': float(min_dist.std()),
                'min': float(min_dist.min()),
                'max': float(min_dist.max()),
                'median': float(np.median(min_dist)),
            },
            'embedding_norm': {
                'mean': float(emb_norm.mean()),
                'std': float(emb_norm.std()),
                'min': float(emb_norm.min()),
                'max': float(emb_norm.max()),
            },
            'assigned_proto_dist': {int(k): int(v) for k, v in zip(*np.unique(assigned, return_counts=True))},
            'nearest_proto_dist': {int(k): int(v) for k, v in zip(*np.unique(nearest, return_counts=True))},
        }

        if len(d['horo_scores_all']) > 0:
            all_horo = torch.stack(d['horo_scores_all']).numpy()
            cls_stats['per_proto_horo_mean'] = all_horo.mean(axis=0).tolist()
            cls_stats['per_proto_horo_std'] = all_horo.std(axis=0).tolist()

        stats['per_class'][cls_key] = cls_stats
        stats[group]['max_horo_scores'].extend(max_horo.tolist())
        stats[group]['min_poincare_dists'].extend(min_dist.tolist())
        stats[group]['embedding_norms'].extend(emb_norm.tolist())

    return stats


def find_optimal_thresholds(stats, class_data, known_class_names):
    """Analyze different thresholding strategies."""

    known_scores = np.array(stats['known']['max_horo_scores'])
    unknown_scores = np.array(stats['unknown']['max_horo_scores'])

    if len(unknown_scores) == 0 or len(known_scores) == 0:
        print("  WARNING: No known or unknown samples collected!")
        return {}

    results = {}

    # ==== Strategy 1: Global horosphere threshold (tau) ====
    print(f"\n{'='*60}")
    print("STRATEGY 1: Global Horosphere Threshold")
    print(f"  Score = max_k(xi_k(x)),  OOD if score < tau")
    print(f"{'='*60}")

    all_scores = np.concatenate([known_scores, unknown_scores])
    all_labels = np.concatenate([np.ones(len(known_scores)), np.zeros(len(unknown_scores))])

    best_f1 = 0
    best_tau = 0
    tau_results = []

    for tau in np.linspace(all_scores.min() - 0.1, all_scores.max() + 0.1, 200):
        preds = (all_scores >= tau).astype(float)

        tp = ((preds == 0) & (all_labels == 0)).sum()
        fp = ((preds == 0) & (all_labels == 1)).sum()
        fn = ((preds == 1) & (all_labels == 0)).sum()
        tn = ((preds == 1) & (all_labels == 1)).sum()

        unk_recall = tp / max(tp + fn, 1)
        unk_precision = tp / max(tp + fp, 1)
        known_recall = tn / max(tn + fp, 1)
        f1 = 2 * unk_precision * unk_recall / max(unk_precision + unk_recall, 1e-8)

        tau_results.append({
            'tau': float(tau),
            'unk_recall': float(unk_recall),
            'unk_precision': float(unk_precision),
            'known_recall': float(known_recall),
            'f1': float(f1),
        })

        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau

    for target in [0.5, 0.7, 0.9]:
        for r in sorted(tau_results, key=lambda x: abs(x['unk_recall'] - target)):
            if abs(r['unk_recall'] - target) < 0.05:
                print(f"  tau={r['tau']:.3f}: UnkRecall={r['unk_recall']:.1%}, "
                      f"KnownRecall={r['known_recall']:.1%}, F1={r['f1']:.3f}")
                break

    print(f"\n  Best F1={best_f1:.3f} at tau={best_tau:.3f}")
    print(f"  Known scores:  mean={known_scores.mean():.3f}, std={known_scores.std():.3f}, "
          f"min={known_scores.min():.3f}, median={np.median(known_scores):.3f}, max={known_scores.max():.3f}")
    print(f"  Unknown scores: mean={unknown_scores.mean():.3f}, std={unknown_scores.std():.3f}, "
          f"min={unknown_scores.min():.3f}, median={np.median(unknown_scores):.3f}, max={unknown_scores.max():.3f}")

    overlap = (known_scores.min() < unknown_scores.max()) and (unknown_scores.min() < known_scores.max())
    if overlap:
        overlap_lo = max(known_scores.min(), unknown_scores.min())
        overlap_hi = min(known_scores.max(), unknown_scores.max())
        known_in_overlap = ((known_scores >= overlap_lo) & (known_scores <= overlap_hi)).mean()
        unknown_in_overlap = ((unknown_scores >= overlap_lo) & (unknown_scores <= overlap_hi)).mean()
        print(f"  Score overlap: YES [{overlap_lo:.3f}, {overlap_hi:.3f}]")
        print(f"    Known in overlap: {known_in_overlap:.1%}")
        print(f"    Unknown in overlap: {unknown_in_overlap:.1%}")
    else:
        print(f"  Score overlap: NO (perfect separation possible!)")

    results['global_horo'] = {
        'best_tau': float(best_tau), 'best_f1': float(best_f1),
        'tau_curve': tau_results,
    }

    # ==== Strategy 2: Poincare distance threshold ====
    known_dists = np.array(stats['known']['min_poincare_dists'])
    unknown_dists = np.array(stats['unknown']['min_poincare_dists'])

    print(f"\n{'='*60}")
    print("STRATEGY 2: Poincare Distance Threshold")
    print(f"  dist = min_k d(x, p_k*0.99),  OOD if dist > tau_d")
    print(f"{'='*60}")

    best_f1_d = 0
    best_tau_d = 0
    all_dists = np.concatenate([known_dists, unknown_dists])

    for tau_d in np.linspace(all_dists.min(), all_dists.max(), 200):
        preds = (all_dists <= tau_d).astype(float)

        tp = ((preds == 0) & (all_labels == 0)).sum()
        fp = ((preds == 0) & (all_labels == 1)).sum()
        fn = ((preds == 1) & (all_labels == 0)).sum()
        tn = ((preds == 1) & (all_labels == 1)).sum()

        unk_recall = tp / max(tp + fn, 1)
        unk_precision = tp / max(tp + fp, 1)
        known_recall = tn / max(tn + fp, 1)
        f1 = 2 * unk_precision * unk_recall / max(unk_precision + unk_recall, 1e-8)

        if f1 > best_f1_d:
            best_f1_d = f1
            best_tau_d = tau_d

    print(f"  Best F1={best_f1_d:.3f} at tau_d={best_tau_d:.3f}")
    print(f"  Known dists:  mean={known_dists.mean():.3f}, std={known_dists.std():.3f}")
    print(f"  Unknown dists: mean={unknown_dists.mean():.3f}, std={unknown_dists.std():.3f}")

    results['global_dist'] = {
        'best_tau': float(best_tau_d), 'best_f1': float(best_f1_d),
    }

    # ==== Strategy 3: Per-class adaptive threshold ====
    print(f"\n{'='*60}")
    print("STRATEGY 3: Per-Prototype Adaptive Threshold")
    print(f"  tau_k = mean_k - alpha*std_k for known class k")
    print(f"{'='*60}")

    for alpha in [1.0, 1.5, 2.0, 2.5, 3.0]:
        class_thresholds = {}
        for cls_key, cls_stats in stats['per_class'].items():
            if cls_stats['is_known']:
                mu = cls_stats['horo_score']['mean']
                sigma = cls_stats['horo_score']['std']
                class_thresholds[cls_key] = mu - alpha * sigma

        unk_caught = 0
        known_lost = 0
        total_unk = len(unknown_scores)
        total_known = len(known_scores)

        for cls_key, d in class_data.items():
            is_known = d['is_known']
            for score, assigned in zip(d['max_horo_score'], d['assigned_proto']):
                assigned_name = known_class_names[assigned] if assigned < len(known_class_names) else None
                if assigned_name and assigned_name in class_thresholds:
                    below = score < class_thresholds[assigned_name]
                    if not is_known and below:
                        unk_caught += 1
                    elif is_known and below:
                        known_lost += 1

        unk_recall = unk_caught / max(total_unk, 1)
        known_recall = 1 - known_lost / max(total_known, 1)

        print(f"  alpha={alpha:.1f}: UnkRecall={unk_recall:.1%}, KnownRecall={known_recall:.1%}")
        if alpha == 2.0:
            results['classwise'] = class_thresholds

    return results


def plot_analysis(stats, class_data, known_class_names, threshold_results, save_dir):
    """Generate comprehensive analysis plots."""
    known_set = set(known_class_names)

    known_scores = np.array(stats['known']['max_horo_scores'])
    unknown_scores = np.array(stats['unknown']['max_horo_scores'])
    known_dists = np.array(stats['known']['min_poincare_dists'])
    unknown_dists = np.array(stats['unknown']['min_poincare_dists'])
    known_norms = np.array(stats['known']['embedding_norms'])
    unknown_norms = np.array(stats['unknown']['embedding_norms'])

    num_known = len(known_class_names)

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))

    # ---- Plot 1: Known vs Unknown horosphere score distributions ----
    ax = axes[0, 0]
    all_min = min(known_scores.min(), unknown_scores.min()) - 0.2
    all_max = max(known_scores.max(), unknown_scores.max()) + 0.2
    bins = np.linspace(all_min, all_max, 60)
    ax.hist(known_scores, bins=bins, alpha=0.6, color='blue', label=f'Known (n={len(known_scores)})', density=True)
    ax.hist(unknown_scores, bins=bins, alpha=0.6, color='red', label=f'Unknown (n={len(unknown_scores)})', density=True)
    if 'global_horo' in threshold_results:
        tau = threshold_results['global_horo']['best_tau']
        ax.axvline(tau, color='green', linestyle='--', linewidth=2, label=f'Best tau={tau:.2f}')
    ax.axvline(0, color='black', linestyle=':', linewidth=1.5, label='tau=0 (geometric)')
    ax.set_xlabel('Max Horosphere Score (xi = -B + a)')
    ax.set_ylabel('Density')
    ax.set_title('Known vs Unknown: Horosphere Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Plot 2: Known vs Unknown Poincare distance ----
    ax = axes[0, 1]
    bins_d = np.linspace(min(known_dists.min(), unknown_dists.min()),
                          max(known_dists.max(), unknown_dists.max()), 60)
    ax.hist(known_dists, bins=bins_d, alpha=0.6, color='blue', label='Known', density=True)
    ax.hist(unknown_dists, bins=bins_d, alpha=0.6, color='red', label='Unknown', density=True)
    if 'global_dist' in threshold_results:
        tau_d = threshold_results['global_dist']['best_tau']
        ax.axvline(tau_d, color='green', linestyle='--', linewidth=2, label=f'Best tau_d={tau_d:.2f}')
    ax.set_xlabel('Min Poincare Distance to Nearest Proto')
    ax.set_ylabel('Density')
    ax.set_title('Known vs Unknown: Poincare Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Plot 3: Embedding norms ----
    ax = axes[0, 2]
    bins_n = np.linspace(min(known_norms.min(), unknown_norms.min()),
                          max(known_norms.max(), unknown_norms.max()), 60)
    ax.hist(known_norms, bins=bins_n, alpha=0.6, color='blue', label='Known', density=True)
    ax.hist(unknown_norms, bins=bins_n, alpha=0.6, color='red', label='Unknown', density=True)
    ax.set_xlabel('||x|| (Poincare ball norm)')
    ax.set_ylabel('Density')
    ax.set_title('Known vs Unknown: Embedding Norms\n(Closer to 1.0 = near boundary)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Plot 4: Per-class horosphere score boxplot ----
    ax = axes[1, 0]
    plot_data = []
    plot_labels = []
    plot_colors = []

    known_keys = sorted([k for k in class_data if class_data[k]['is_known'] and len(class_data[k]['max_horo_score']) > 0])
    unknown_keys = sorted([k for k in class_data if not class_data[k]['is_known'] and len(class_data[k]['max_horo_score']) > 0])

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
    ax.axhline(0, color='black', linestyle=':', linewidth=1.5, label='tau=0')
    if 'global_horo' in threshold_results:
        ax.axhline(threshold_results['global_horo']['best_tau'], color='green',
                    linestyle='--', linewidth=1, label=f"best tau={threshold_results['global_horo']['best_tau']:.2f}")
    ax.set_ylabel('Max Horosphere Score')
    ax.set_title('Per-Class Score Distribution (* = unknown)')
    ax.tick_params(axis='x', rotation=60, labelsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # ---- Plot 5: Score vs Distance scatter ----
    ax = axes[1, 1]
    ax.scatter(known_scores, known_dists, alpha=0.15, c='blue', s=10, label='Known')
    ax.scatter(unknown_scores, unknown_dists, alpha=0.15, c='red', s=10, label='Unknown')
    ax.set_xlabel('Max Horosphere Score (xi)')
    ax.set_ylabel('Min Poincare Distance')
    ax.set_title('Horosphere Score vs Poincare Distance\n(Top-left = ideal unknown)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Plot 6: ROC-like curve ----
    ax = axes[1, 2]
    if 'global_horo' in threshold_results:
        curve = threshold_results['global_horo']['tau_curve']
        taus = [r['tau'] for r in curve]
        ax.plot(taus, [r['unk_recall'] for r in curve], 'r-', label='Unknown Recall', linewidth=2)
        ax.plot(taus, [r['known_recall'] for r in curve], 'b-', label='Known Recall', linewidth=2)
        ax.plot(taus, [r['f1'] for r in curve], 'g--', label='F1 Score', linewidth=2)
        ax.axvline(threshold_results['global_horo']['best_tau'], color='green', linestyle=':', alpha=0.5)
        ax.axvline(0, color='black', linestyle=':', alpha=0.5, label='tau=0')
    ax.set_xlabel('Threshold tau')
    ax.set_ylabel('Rate')
    ax.set_title('OOD Detection Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'unknown_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")

    # ---- Per-unknown-subclass breakdown ----
    if len(unknown_keys) > 0:
        n_unk = len(unknown_keys)
        ncols = min(n_unk, 4)
        nrows = (n_unk + ncols - 1) // ncols
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

        for idx, cls_key in enumerate(sorted(unknown_keys)):
            ax = axes2[idx // ncols][idx % ncols]
            d = class_data[cls_key]
            scores = np.array(d['max_horo_score'])
            assigned = np.array(d['assigned_proto'])

            for proto_id in range(num_known):
                mask = assigned == proto_id
                if mask.sum() > 0:
                    pname = known_class_names[proto_id]
                    ax.hist(scores[mask], bins=30, alpha=0.5, label=f'{pname} ({mask.sum()})')

            ax.axvline(0, color='black', linestyle=':', linewidth=1.5, label='tau=0')
            pct_neg = (scores < 0).mean() * 100
            ax.set_title(f'{cls_key}\n(n={len(scores)}, {pct_neg:.0f}% below tau=0)', fontsize=9)
            ax.set_xlabel('Max Horo Score')
            ax.legend(fontsize=6, loc='upper left')
            ax.grid(True, alpha=0.3)

        for idx in range(len(unknown_keys), nrows * ncols):
            axes2[idx // ncols][idx % ncols].set_visible(False)

        plt.tight_layout()
        path2 = os.path.join(save_dir, 'unknown_per_subclass.png')
        plt.savefig(path2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path2}")

    # ---- Prototype assignment confusion matrix ----
    fig4, axes4 = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes4[0]
    known_matrix = np.zeros((len(known_keys), num_known))
    for r, cls_key in enumerate(known_keys):
        assigned = np.array(class_data[cls_key]['assigned_proto'])
        for p in range(num_known):
            known_matrix[r, p] = (assigned == p).sum()
    if known_matrix.sum() > 0:
        known_matrix_pct = known_matrix / known_matrix.sum(axis=1, keepdims=True).clip(1)
    else:
        known_matrix_pct = known_matrix
    im = ax.imshow(known_matrix_pct, aspect='auto', cmap='Blues')
    ax.set_xticks(range(num_known))
    ax.set_xticklabels(known_class_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(known_keys)))
    ax.set_yticklabels(known_keys, fontsize=8)
    ax.set_xlabel('Assigned Prototype')
    ax.set_ylabel('True Class')
    ax.set_title('Known -> Prototype Assignment')
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes4[1]
    unk_matrix = np.zeros((len(unknown_keys), num_known))
    for r, cls_key in enumerate(sorted(unknown_keys)):
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
    ax.set_yticklabels(sorted(unknown_keys), fontsize=8)
    ax.set_xlabel('Assigned Prototype (misclassified as)')
    ax.set_ylabel('True Unknown Class')
    ax.set_title('Unknown -> Prototype Assignment\n(Which known class absorbs each unknown?)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    path4 = os.path.join(save_dir, 'prototype_assignment.png')
    plt.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path4}")

    # ---- Embedding norms per class ----
    fig3, ax3 = plt.subplots(1, 1, figsize=(14, 6))
    for cls_key in known_keys + unknown_keys:
        d = class_data[cls_key]
        if len(d['embedding_norm']) == 0:
            continue
        norms = np.array(d['embedding_norm'])
        is_known = d['is_known']
        style = '-' if is_known else '--'
        label = cls_key if is_known else f'*{cls_key}'
        ax3.hist(norms, bins=30, alpha=0.3, label=label,
                 linestyle=style, histtype='step', linewidth=2, density=True)
    ax3.set_xlabel('||x|| (Poincare ball norm)')
    ax3.set_ylabel('Density')
    ax3.set_title('Per-Class Embedding Norms (solid=known, dashed=unknown)')
    ax3.legend(fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    path3 = os.path.join(save_dir, 'embedding_norms.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path3}")


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--task", default="IDD_HYP/t1")
    parser.add_argument("--ckpt", default="IDD_HYP/t1/horospherical/model_5.pth")
    parser.add_argument("--hyp_c", type=float, default=1.0)
    parser.add_argument("--hyp_dim", type=int, default=256)
    parser.add_argument("--clip_r", type=float, default=0.95)
    parser.add_argument("--output_dir", default="visualizations")

    args = parser.parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)

    task_name = args.task.split('/')[0]
    split_name = args.task.split('/')[1]

    # Handle IDD_HYP -> IDD for dataset registration
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

    # Model config - use task_name (IDD_HYP) for config paths
    config_file = os.path.join("./configs", task_name, split_name + ".py")
    cfgY = Config.fromfile(config_file)
    cfgY.work_dir = "."

    # Initialize YOLO-World
    runner = Runner.from_cfg(cfgY)
    runner.call_hook("before_run")
    runner.load_or_resume()
    runner.model.reparameterize([known_class_names])
    runner.model.eval()

    # Build TEST loader (images from test split)
    test_loader = Runner.build_dataloader(cfgY.test_dataloader)

    # Build hyperbolic model
    model = HypCustomYoloWorld(
        runner.model, unknown_index,
        hyp_c=args.hyp_c, hyp_dim=args.hyp_dim, clip_r=args.clip_r
    )

    print(f"\n=== Loading Checkpoint: {args.ckpt} ===")
    with torch.no_grad():
        model = load_hyp_ckpt(model, args.ckpt,
                              cfg.TEST.PREV_INTRODUCED_CLS, cfg.TEST.CUR_INTRODUCED_CLS, eval=True)
        model = model.cuda()
        model.add_generic_text(known_class_names, generic_prompt='object', alpha=0.4)
    model.eval()

    prototypes = model.prototypes.detach()
    biases = model.prototype_biases.detach()

    print(f"\n=== Model Info ===")
    print(f"  Prototypes: {prototypes.shape[0]} (norms: {[f'{n:.4f}' for n in prototypes.norm(dim=-1).tolist()]})")
    print(f"  Biases: {[f'{b:.4f}' for b in biases.cpu().tolist()]}")
    print(f"  Classes: {known_class_names}")

    # Collect embeddings — parses XML directly to get ALL boxes (full dataset)
    class_data, samples_count = collect_all_embeddings(
        model, test_loader, known_class_names,
        dataset_root='./datasets',
        hyp_c=args.hyp_c,
    )

    if not class_data:
        print("ERROR: No embeddings collected!")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("COMPUTING STATISTICS")
    print(f"{'='*60}")

    stats = compute_statistics(class_data, samples_count, known_class_names)

    # Print table
    print(f"\n{'='*100}")
    print(f"{'Class':<25s} {'Type':<8s} {'N':>5s} {'HoroMean':>9s} {'HoroStd':>8s} "
          f"{'HoroMin':>8s} {'%Neg':>6s} {'DistMean':>9s} {'NormMean':>9s}")
    print(f"{'-'*100}")

    for cls_key in sorted(stats['per_class'].keys(), key=lambda k: (not stats['per_class'][k]['is_known'], k)):
        s = stats['per_class'][cls_key]
        tag = "KNOWN" if s['is_known'] else "UNK"
        neg_pct = s['horo_score']['pct_negative']
        print(f"{cls_key:<25s} {tag:<8s} {s['n_samples']:>5d} "
              f"{s['horo_score']['mean']:>9.3f} {s['horo_score']['std']:>8.3f} "
              f"{s['horo_score']['min']:>8.3f} {neg_pct:>5.0%} "
              f"{s['poincare_dist']['mean']:>9.3f} {s['embedding_norm']['mean']:>9.4f}")

    threshold_results = find_optimal_thresholds(stats, class_data, known_class_names)

    os.makedirs(args.output_dir, exist_ok=True)
    plot_analysis(stats, class_data, known_class_names, threshold_results, args.output_dir)

    # Save data
    ckpt_name = Path(args.ckpt).stem
    save_path = os.path.join(args.output_dir, f"unknown_analysis_{ckpt_name}.npz")

    save_dict = {
        'known_horo_scores': np.array(stats['known']['max_horo_scores']),
        'unknown_horo_scores': np.array(stats['unknown']['max_horo_scores']),
        'known_poincare_dists': np.array(stats['known']['min_poincare_dists']),
        'unknown_poincare_dists': np.array(stats['unknown']['min_poincare_dists']),
        'known_norms': np.array(stats['known']['embedding_norms']),
        'unknown_norms': np.array(stats['unknown']['embedding_norms']),
        'prototypes': prototypes.cpu().numpy(),
        'biases': biases.cpu().numpy(),
        'class_names': np.array(known_class_names, dtype=object),
        'num_known': unknown_index,
    }
    np.savez(save_path, **save_dict)
    print(f"\n  Saved: {save_path}")

    json_path = os.path.join(args.output_dir, f"unknown_stats_{ckpt_name}.json")

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

    json_stats = convert_types({
        'per_class': stats['per_class'],
        'global_horo_best_tau': threshold_results.get('global_horo', {}).get('best_tau', 0),
        'global_horo_best_f1': threshold_results.get('global_horo', {}).get('best_f1', 0),
        'global_dist_best_tau': threshold_results.get('global_dist', {}).get('best_tau', 0),
        'global_dist_best_f1': threshold_results.get('global_dist', {}).get('best_f1', 0),
    })
    with open(json_path, 'w') as f:
        json.dump(json_stats, f, indent=2)
    print(f"  Saved: {json_path}")

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
