"""
Hyperbolic Model Diagnostic Report
===================================
Generates comprehensive statistics about the trained model:

1. Per-class prototype analysis
   - Distance distributions (own proto vs other protos)
   - Misassignment rates (% of GT embeddings closer to wrong prototype)
   - Embedding norm distributions per class

2. Unknown/novel class proximity
   - How close unknown GT embeddings are to each known prototype
   - Confusion matrix: which prototype each unknown class gets assigned to

3. Detection confidence analysis
   - YOLO objectness/confidence scores for known vs unknown GT boxes
   - Score distributions per class — are unknowns even detected?

4. Hyperbolic space statistics
   - Embedding norms (how far from origin)
   - Prototype separation (geodesic distances between prototypes)
   - Score range analysis (can we separate known from unknown?)
   - Horosphere bias analysis

Outputs a text report + matplotlib figures.

Usage:
    python debug/diagnose_model.py \\
        --config-file configs/IDD_HYP/base.yaml \\
        --task IDD_HYP/t1 \\
        --ckpt IDD_HYP/t1/newmethod/model_final.pth \\
        --output_dir diagnostics
"""

import os
import sys
import copy
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from core import DatasetMapper, add_config
from core.util.model_ema import add_model_ema_configs
from core.pascal_voc import register_pascal_voc, inital_prompts
from core.hyp_customyoloworld import HypCustomYoloWorld, load_hyp_ckpt
from core.hyperbolic.pmath import busemann

from mmengine.config import Config
from mmengine.runner import Runner
from detectron2.engine import default_argument_parser, default_setup
from detectron2.config import get_cfg


# ============================================================================
# Shared setup from visualize_simple.py
# ============================================================================

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
    task_name, split_name = args.task.split('/')
    task_yaml = os.path.join("./configs", task_name, f"{split_name}.yaml")
    if os.path.exists(task_yaml):
        cfg.merge_from_file(task_yaml)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def parse_all_xml_boxes(dataset_root, img_id, known_set):
    """Parse XML annotation file and return all GT boxes."""
    import xml.etree.ElementTree as ET
    xml_path = os.path.join(dataset_root, 'Annotations', f'{img_id}.xml')
    if not os.path.exists(xml_path):
        return []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text.strip()
        bbox = obj.find('bndbox')
        x1, y1 = float(bbox.find('xmin').text), float(bbox.find('ymin').text)
        x2, y2 = float(bbox.find('xmax').text), float(bbox.find('ymax').text)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append({
            'class_name': name,
            'bbox': [x1, y1, x2, y2],
            'is_known': name in known_set,
        })
    return boxes


def get_anchor_centers(h, w, device='cpu'):
    """Get 8400 anchor centers for 640x640 input (P3/P4/P5 strides)."""
    strides = [8, 16, 32]
    centers = []
    for s in strides:
        gh, gw = h // s, w // s
        yy, xx = torch.meshgrid(torch.arange(gh, device=device), torch.arange(gw, device=device), indexing='ij')
        c = torch.stack([(xx.float() + 0.5) * s, (yy.float() + 0.5) * s], dim=-1).reshape(-1, 2)
        centers.append(c)
    return torch.cat(centers, dim=0)


# ============================================================================
# Data collection — richer than visualize_simple
# ============================================================================

def collect_diagnostic_data(model, data_loader, known_class_names, dataset_root, hyp_c):
    """
    Collect comprehensive per-GT-box statistics:
    - Hyperbolic embedding
    - Full horosphere score vector (all K prototypes)
    - Max horo score + assigned prototype
    - YOLO detection confidence (matched by IoU to GT)
    - Embedding norm
    """
    print(f"\n{'='*60}")
    print(f"COLLECTING DIAGNOSTIC DATA")
    print(f"{'='*60}")

    model.eval()
    known_set = set(known_class_names)
    prototypes = model.prototypes.detach()   # (K, D)
    biases = model.prototype_biases.detach() # (K,)
    K = prototypes.shape[0]

    records = []  # list of dicts per GT box

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc="Collecting")):
            try:
                data_batch = model.parent.data_preprocessor(batch)

                # Extract features once (backbone + neck + projector)
                img_feats, txt_feats, hyp_embeddings = model.extract_feat(
                    data_batch['inputs'], data_batch['data_samples']
                )

                # Run predict_by_feat (uses already-extracted features, no redundant backbone pass)
                results_list = model.predict_by_feat(
                    img_feats, txt_feats, data_batch['data_samples'], hyp_embeddings, rescale=False
                )
                results = model.parent.add_pred_to_datasample(data_batch['data_samples'], results_list)

                h, w = data_batch['inputs'].shape[-2:]
                anchor_centers = get_anchor_centers(h, w, device=hyp_embeddings.device)

                for b_idx, data_sample in enumerate(results):
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

                    # Get detections for this image (for objectness matching)
                    pred_instances = data_sample.pred_instances
                    pred_bboxes = pred_instances.bboxes if hasattr(pred_instances, 'bboxes') and len(pred_instances.bboxes) > 0 else None
                    pred_scores = pred_instances.scores if pred_bboxes is not None else None

                    for box_info in xml_boxes:
                        cls_name = box_info['class_name']
                        is_known = box_info['is_known']

                        ox1, oy1, ox2, oy2 = box_info['bbox']
                        sx, sy = scale_factor if isinstance(scale_factor, (list, tuple)) else (scale_factor, scale_factor)
                        bx1, by1 = ox1 * sx, oy1 * sy
                        bx2, by2 = ox2 * sx, oy2 * sy

                        if pad_param is not None:
                            pad_top, _, pad_left, _ = pad_param
                            bx1 += pad_left; by1 += pad_top
                            bx2 += pad_left; by2 += pad_top

                        cx = max(0, min((bx1 + bx2) / 2.0, w - 1))
                        cy = max(0, min((by1 + by2) / 2.0, h - 1))

                        dists = (anchor_centers[:, 0] - cx)**2 + (anchor_centers[:, 1] - cy)**2
                        nearest_idx = dists.argmin().item()

                        emb = hyp_embeddings[b_idx, nearest_idx]  # (D,)
                        emb_norm = emb.norm().item()

                        # Full horosphere scores for ALL prototypes
                        B_vals = busemann(prototypes, emb.unsqueeze(0), c=hyp_c)  # (1, K)
                        horo_scores = (-B_vals + biases).squeeze(0)  # (K,)
                        max_horo, assigned = horo_scores.max(dim=0)

                        # Match GT box to best detection (IoU-based)
                        best_det_score = 0.0
                        best_det_horo = -999.0
                        detected = False
                        gt_box = torch.tensor([bx1, by1, bx2, by2], device=hyp_embeddings.device)
                        if pred_bboxes is not None and len(pred_bboxes) > 0:
                            ious = _box_iou(gt_box.unsqueeze(0), pred_bboxes)  # (1, N_det)
                            max_iou, det_idx = ious.max(dim=1)
                            if max_iou.item() > 0.3:
                                detected = True
                                best_det_score = pred_scores[det_idx.item()].item()
                                if hasattr(pred_instances, 'horo_max_scores'):
                                    best_det_horo = pred_instances.horo_max_scores[det_idx.item()].item()

                        record = {
                            'class_name': cls_name,
                            'is_known': is_known,
                            'emb_norm': emb_norm,
                            'horo_scores': horo_scores.cpu().numpy(),  # (K,)
                            'max_horo': max_horo.item(),
                            'assigned_proto': assigned.item(),
                            'detected': detected,
                            'det_score': best_det_score,
                            'det_horo': best_det_horo,
                        }
                        records.append(record)

            except Exception as e:
                tqdm.write(f"  Error batch {i}: {e}")
                import traceback; traceback.print_exc()
                continue

    print(f"\n  Total records: {len(records)}")
    return records


def _box_iou(box1, box2):
    """IoU between box1 (M,4) and box2 (N,4) -> (M,N)"""
    x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    y2 = torch.min(box1[:, None, 3], box2[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    a1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    a2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    return inter / (a1[:, None] + a2[None, :] - inter + 1e-6)


# ============================================================================
# Analysis functions
# ============================================================================

def analyze_prototypes(prototypes, biases, known_class_names, hyp_c):
    """Section 4: Hyperbolic space — prototype geometry."""
    K, D = prototypes.shape
    R = 1.0 / (hyp_c ** 0.5)

    lines = []
    lines.append("=" * 70)
    lines.append("SECTION 4: HYPERBOLIC SPACE — PROTOTYPE GEOMETRY")
    lines.append("=" * 70)

    # Prototype norms (should all be R = 1/√c for ideal points)
    norms = prototypes.norm(dim=-1)
    lines.append(f"\nPrototype norms (should be R={R:.4f} for ideal boundary points):")
    for i, name in enumerate(known_class_names):
        lines.append(f"  {name:<20s}: ||p|| = {norms[i].item():.6f}")

    # Biases
    lines.append(f"\nPrototype biases (positive = tighter horosphere, negative = looser):")
    for i, name in enumerate(known_class_names):
        lines.append(f"  {name:<20s}: bias = {biases[i].item():+.4f}")

    # Cosine similarity between prototype directions
    dirs = F.normalize(prototypes, dim=-1)  # (K, D)
    cosine_sim = dirs @ dirs.T  # (K, K)
    lines.append(f"\nCosine similarity between prototype directions:")
    header = "                " + "  ".join(f"{n[:6]:>8s}" for n in known_class_names)
    lines.append(header)
    for i, name in enumerate(known_class_names):
        row = f"  {name:<14s}"
        for j in range(K):
            val = cosine_sim[i, j].item()
            row += f"  {val:8.3f}"
        lines.append(row)

    # Min separation
    mask = ~torch.eye(K, dtype=torch.bool)
    min_sep = cosine_sim[mask].max().item()
    max_sep = cosine_sim[mask].min().item()
    mean_sep = cosine_sim[mask].mean().item()
    lines.append(f"\n  Max cosine (closest pair): {min_sep:.4f}")
    lines.append(f"  Min cosine (farthest pair): {max_sep:.4f}")
    lines.append(f"  Mean cosine (off-diagonal): {mean_sep:.4f}")

    # Geodesic distances between prototypes (approximate for boundary points)
    # Since prototypes are on the boundary, geodesic distance is infinite.
    # Instead report angular separation in degrees.
    angles = torch.acos(cosine_sim.clamp(-1+1e-6, 1-1e-6)) * 180 / np.pi
    lines.append(f"\nAngular separation (degrees) between prototypes:")
    header = "                " + "  ".join(f"{n[:6]:>8s}" for n in known_class_names)
    lines.append(header)
    for i, name in enumerate(known_class_names):
        row = f"  {name:<14s}"
        for j in range(K):
            val = angles[i, j].item()
            row += f"  {val:8.1f}"
        lines.append(row)

    lines.append(f"\n  Min angular sep: {angles[mask].min().item():.1f}°")
    lines.append(f"  Max angular sep: {angles[mask].max().item():.1f}°")
    lines.append(f"  Mean angular sep: {angles[mask].float().mean().item():.1f}°")

    # Effective dimension usage
    _, S, _ = torch.linalg.svd(dirs)
    energy = (S ** 2).cumsum(0) / (S ** 2).sum()
    lines.append(f"\nSVD of prototype directions (dimension usage):")
    lines.append(f"  Singular values: {', '.join(f'{s:.3f}' for s in S[:min(K, 10)].tolist())}")
    for t in [0.90, 0.95, 0.99]:
        d = (energy < t).sum().item() + 1
        lines.append(f"  Dims for {t*100:.0f}% energy: {d}/{D}")

    return "\n".join(lines)


def analyze_per_class(records, known_class_names, hyp_c):
    """Section 1: Per-class prototype modeling quality."""
    K = len(known_class_names)
    cls_to_idx = {n: i for i, n in enumerate(known_class_names)}

    lines = []
    lines.append("=" * 70)
    lines.append("SECTION 1: PER-CLASS PROTOTYPE MODELING")
    lines.append("=" * 70)

    # Group records by class
    class_records = defaultdict(list)
    for r in records:
        class_records[r['class_name']].append(r)

    lines.append(f"\n{'Class':<20s} {'Count':>6s} {'MeanNorm':>8s} {'MeanHoro':>9s} {'StdHoro':>8s} {'Correct%':>9s} {'TopConfuse':>12s}")
    lines.append("-" * 80)

    for cls_name in known_class_names:
        recs = class_records.get(cls_name, [])
        if not recs:
            lines.append(f"  {cls_name:<20s} {'N/A':>6s}")
            continue

        n = len(recs)
        norms = [r['emb_norm'] for r in recs]
        own_idx = cls_to_idx[cls_name]
        own_scores = [r['horo_scores'][own_idx] for r in recs]
        max_horos = [r['max_horo'] for r in recs]
        assigned = [r['assigned_proto'] for r in recs]

        # Accuracy: % assigned to correct prototype
        correct = sum(1 for a in assigned if a == own_idx)
        correct_pct = 100.0 * correct / n

        # Most confused prototype (excluding own)
        confusion_counts = defaultdict(int)
        for a in assigned:
            if a != own_idx:
                confusion_counts[a] += 1
        if confusion_counts:
            top_confuse_idx = max(confusion_counts, key=confusion_counts.get)
            top_confuse_name = known_class_names[top_confuse_idx] if top_confuse_idx < K else f"idx{top_confuse_idx}"
            top_confuse_pct = 100.0 * confusion_counts[top_confuse_idx] / n
            confuse_str = f"{top_confuse_name[:8]}({top_confuse_pct:.1f}%)"
        else:
            confuse_str = "none"

        lines.append(f"  {cls_name:<20s} {n:>6d} {np.mean(norms):>8.4f} {np.mean(own_scores):>9.4f} "
                     f"{np.std(own_scores):>8.4f} {correct_pct:>8.1f}% {confuse_str:>12s}")

    # Detailed per-class breakdown
    lines.append(f"\n\n--- DETAILED: Score distribution per known class ---")
    for cls_name in known_class_names:
        recs = class_records.get(cls_name, [])
        if not recs:
            continue
        own_idx = cls_to_idx[cls_name]
        own_scores = np.array([r['horo_scores'][own_idx] for r in recs])
        max_horos = np.array([r['max_horo'] for r in recs])
        norms = np.array([r['emb_norm'] for r in recs])

        lines.append(f"\n  [{cls_name}] (n={len(recs)})")
        lines.append(f"    Own-proto score:  mean={own_scores.mean():.4f} std={own_scores.std():.4f} "
                     f"min={own_scores.min():.4f} max={own_scores.max():.4f}")
        lines.append(f"    Max-horo score:   mean={max_horos.mean():.4f} std={max_horos.std():.4f}")
        lines.append(f"    Embedding norm:   mean={norms.mean():.4f} std={norms.std():.4f} "
                     f"min={norms.min():.4f} max={norms.max():.4f}")

        # Score for OTHER prototypes (confusion profile)
        for j, other_name in enumerate(known_class_names):
            if j == own_idx:
                continue
            other_scores = np.array([r['horo_scores'][j] for r in recs])
            pct_higher = 100.0 * (other_scores > own_scores).mean()
            if pct_higher > 1.0:  # Only show if >1% confused
                lines.append(f"    vs {other_name:<15s}: mean={other_scores.mean():.4f} "
                             f"({pct_higher:.1f}% score higher than own)")

    return "\n".join(lines)


def analyze_unknowns(records, known_class_names, hyp_c):
    """Section 2: Unknown/novel class proximity to known prototypes."""
    K = len(known_class_names)

    lines = []
    lines.append("=" * 70)
    lines.append("SECTION 2: UNKNOWN CLASS PROXIMITY TO KNOWN PROTOTYPES")
    lines.append("=" * 70)

    unknown_records = [r for r in records if not r['is_known']]
    known_records = [r for r in records if r['is_known']]

    if not unknown_records:
        lines.append("\n  No unknown class GT boxes found in test set.")
        return "\n".join(lines)

    # Group unknowns by class
    unk_classes = defaultdict(list)
    for r in unknown_records:
        unk_classes[r['class_name']].append(r)

    lines.append(f"\n  Known classes: {known_class_names}")
    lines.append(f"  Total known GT boxes: {len(known_records)}")
    lines.append(f"  Total unknown GT boxes: {len(unknown_records)}")
    lines.append(f"  Unknown classes found: {sorted(unk_classes.keys())}")

    # Reference: known class score statistics
    known_max_horos = np.array([r['max_horo'] for r in known_records])
    lines.append(f"\n  REFERENCE — Known class max-horo scores:")
    lines.append(f"    mean={known_max_horos.mean():.4f} std={known_max_horos.std():.4f} "
                 f"p5={np.percentile(known_max_horos, 5):.4f} p50={np.percentile(known_max_horos, 50):.4f}")

    # Unknown class score summary
    lines.append(f"\n  {'Unknown Class':<20s} {'Count':>6s} {'MaxHoro':>9s} {'StdHoro':>8s} "
                 f"{'TopProto':>10s} {'TopPct':>7s} {'Overlap%':>8s}")
    lines.append("  " + "-" * 75)

    for cls_name in sorted(unk_classes.keys()):
        recs = unk_classes[cls_name]
        n = len(recs)
        max_horos = np.array([r['max_horo'] for r in recs])
        assigned = [r['assigned_proto'] for r in recs]

        # Most common assigned prototype
        proto_counts = defaultdict(int)
        for a in assigned:
            proto_counts[a] += 1
        top_proto = max(proto_counts, key=proto_counts.get)
        top_name = known_class_names[top_proto] if top_proto < K else f"idx{top_proto}"
        top_pct = 100.0 * proto_counts[top_proto] / n

        # Overlap: % of unknowns with max_horo > known_class p5
        known_p5 = np.percentile(known_max_horos, 5) if len(known_max_horos) > 0 else 0
        overlap_pct = 100.0 * (max_horos > known_p5).mean()

        lines.append(f"  {cls_name:<20s} {n:>6d} {max_horos.mean():>9.4f} {max_horos.std():>8.4f} "
                     f"{top_name[:10]:>10s} {top_pct:>6.1f}% {overlap_pct:>7.1f}%")

    # Per-unknown-class: full prototype affinity profile
    lines.append(f"\n\n--- DETAILED: Unknown class → prototype affinity ---")
    for cls_name in sorted(unk_classes.keys()):
        recs = unk_classes[cls_name]
        if len(recs) < 5:
            continue
        lines.append(f"\n  [{cls_name}] (n={len(recs)})")
        # Mean score for each known prototype
        all_scores = np.array([r['horo_scores'] for r in recs])  # (n, K)
        for j, pname in enumerate(known_class_names):
            proto_scores = all_scores[:, j]
            lines.append(f"    → {pname:<15s}: mean={proto_scores.mean():.4f} "
                         f"std={proto_scores.std():.4f} max={proto_scores.max():.4f}")

    return "\n".join(lines)


def analyze_detection(records, known_class_names):
    """Section 3: Detection confidence — are unknowns even detected?"""
    lines = []
    lines.append("=" * 70)
    lines.append("SECTION 3: DETECTION CONFIDENCE (YOLO OBJECTNESS)")
    lines.append("=" * 70)

    # Group by known/unknown
    known_recs = [r for r in records if r['is_known']]
    unknown_recs = [r for r in records if not r['is_known']]

    def _det_stats(recs, label):
        if not recs:
            return f"\n  {label}: no records"
        det_flags = [r['detected'] for r in recs]
        scores = [r['det_score'] for r in recs if r['detected']]
        det_rate = 100.0 * sum(det_flags) / len(det_flags)
        s = f"\n  {label} (n={len(recs)}):"
        s += f"\n    Detection rate (IoU>0.3): {det_rate:.1f}%"
        if scores:
            scores = np.array(scores)
            s += f"\n    Confidence: mean={scores.mean():.4f} std={scores.std():.4f} "
            s += f"p10={np.percentile(scores, 10):.4f} p50={np.percentile(scores, 50):.4f} p90={np.percentile(scores, 90):.4f}"
        return s

    lines.append(_det_stats(known_recs, "ALL KNOWN"))
    lines.append(_det_stats(unknown_recs, "ALL UNKNOWN"))

    # Per known class
    lines.append(f"\n\n--- Per known class detection ---")
    class_recs = defaultdict(list)
    for r in records:
        class_recs[r['class_name']].append(r)

    lines.append(f"\n  {'Class':<20s} {'Count':>6s} {'DetRate':>8s} {'MeanConf':>9s} {'MeanHoro':>9s}")
    lines.append("  " + "-" * 55)
    for cls_name in known_class_names:
        recs = class_recs.get(cls_name, [])
        if not recs:
            continue
        det_rate = 100.0 * sum(1 for r in recs if r['detected']) / len(recs)
        det_scores = [r['det_score'] for r in recs if r['detected']]
        det_horos = [r['det_horo'] for r in recs if r['detected'] and r['det_horo'] > -500]
        mean_conf = np.mean(det_scores) if det_scores else 0
        mean_horo = np.mean(det_horos) if det_horos else 0
        lines.append(f"  {cls_name:<20s} {len(recs):>6d} {det_rate:>7.1f}% {mean_conf:>9.4f} {mean_horo:>9.4f}")

    # Per unknown class
    lines.append(f"\n--- Per unknown class detection ---")
    lines.append(f"\n  {'Class':<20s} {'Count':>6s} {'DetRate':>8s} {'MeanConf':>9s} {'MeanHoro':>9s}")
    lines.append("  " + "-" * 55)
    for cls_name in sorted(set(r['class_name'] for r in unknown_recs)):
        recs = class_recs.get(cls_name, [])
        if not recs:
            continue
        det_rate = 100.0 * sum(1 for r in recs if r['detected']) / len(recs)
        det_scores = [r['det_score'] for r in recs if r['detected']]
        det_horos = [r['det_horo'] for r in recs if r['detected'] and r['det_horo'] > -500]
        mean_conf = np.mean(det_scores) if det_scores else 0
        mean_horo = np.mean(det_horos) if det_horos else 0
        lines.append(f"  {cls_name:<20s} {len(recs):>6d} {det_rate:>7.1f}% {mean_conf:>9.4f} {mean_horo:>9.4f}")

    return "\n".join(lines)


def analyze_score_separation(records, known_class_names):
    """Section 5: Score separation — can we distinguish known from unknown?"""
    lines = []
    lines.append("=" * 70)
    lines.append("SECTION 5: SCORE SEPARATION (KNOWN vs UNKNOWN)")
    lines.append("=" * 70)

    known_scores = np.array([r['max_horo'] for r in records if r['is_known']])
    unknown_scores = np.array([r['max_horo'] for r in records if not r['is_known']])

    if len(known_scores) == 0 or len(unknown_scores) == 0:
        lines.append("\n  Insufficient data for separation analysis.")
        return "\n".join(lines)

    lines.append(f"\n  Max horosphere score distributions:")
    lines.append(f"    Known   (n={len(known_scores):>6d}): "
                 f"mean={known_scores.mean():.4f} std={known_scores.std():.4f} "
                 f"[p1={np.percentile(known_scores,1):.4f}, p5={np.percentile(known_scores,5):.4f}, "
                 f"p50={np.percentile(known_scores,50):.4f}, p95={np.percentile(known_scores,95):.4f}]")
    lines.append(f"    Unknown (n={len(unknown_scores):>6d}): "
                 f"mean={unknown_scores.mean():.4f} std={unknown_scores.std():.4f} "
                 f"[p1={np.percentile(unknown_scores,1):.4f}, p5={np.percentile(unknown_scores,5):.4f}, "
                 f"p50={np.percentile(unknown_scores,50):.4f}, p95={np.percentile(unknown_scores,95):.4f}]")

    # Separation metrics
    gap = known_scores.mean() - unknown_scores.mean()
    pooled_std = np.sqrt((known_scores.std()**2 + unknown_scores.std()**2) / 2)
    d_prime = gap / (pooled_std + 1e-8)
    lines.append(f"\n  Separation:")
    lines.append(f"    Mean gap (known - unknown): {gap:.4f}")
    lines.append(f"    d' (signal-to-noise): {d_prime:.4f}")
    lines.append(f"    (d'>1.0 = decent separation, d'>2.0 = good, d'>3.0 = excellent)")

    # AUROC estimate (simple Mann-Whitney U)
    from scipy import stats as scipy_stats
    try:
        u_stat, p_val = scipy_stats.mannwhitneyu(known_scores, unknown_scores, alternative='greater')
        auroc = u_stat / (len(known_scores) * len(unknown_scores))
        lines.append(f"    AUROC (known > unknown): {auroc:.4f}")
    except Exception:
        lines.append(f"    AUROC: could not compute (scipy missing?)")

    # Optimal threshold (Youden's J)
    thresholds = np.linspace(
        min(known_scores.min(), unknown_scores.min()),
        max(known_scores.max(), unknown_scores.max()),
        1000
    )
    best_j, best_t = -1, 0
    for t in thresholds:
        tpr = (known_scores >= t).mean()
        fpr = (unknown_scores >= t).mean()
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_t = t
    tpr_at_best = (known_scores >= best_t).mean()
    fpr_at_best = (unknown_scores >= best_t).mean()
    lines.append(f"\n  Optimal threshold (Youden's J):")
    lines.append(f"    τ = {best_t:.4f}")
    lines.append(f"    Known recall (TPR): {tpr_at_best*100:.1f}%")
    lines.append(f"    Unknown FPR: {fpr_at_best*100:.1f}%")

    # Embedding norm analysis
    known_norms = np.array([r['emb_norm'] for r in records if r['is_known']])
    unknown_norms = np.array([r['emb_norm'] for r in records if not r['is_known']])
    lines.append(f"\n  Embedding norms:")
    lines.append(f"    Known:   mean={known_norms.mean():.4f} std={known_norms.std():.4f} "
                 f"[{known_norms.min():.4f}, {known_norms.max():.4f}]")
    lines.append(f"    Unknown: mean={unknown_norms.mean():.4f} std={unknown_norms.std():.4f} "
                 f"[{unknown_norms.min():.4f}, {unknown_norms.max():.4f}]")
    lines.append(f"    (clip_r=0.95 for c=1.0 → max possible norm = 0.95)")

    return "\n".join(lines)


# ============================================================================
# Plotting
# ============================================================================

def plot_diagnostics(records, known_class_names, prototypes, biases, save_dir, hyp_c):
    """Generate diagnostic figures."""
    K = len(known_class_names)
    cls_to_idx = {n: i for i, n in enumerate(known_class_names)}
    os.makedirs(save_dir, exist_ok=True)

    # --- Figure 1: Score distributions (known vs unknown) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    known_scores = [r['max_horo'] for r in records if r['is_known']]
    unknown_scores = [r['max_horo'] for r in records if not r['is_known']]

    axes[0].hist(known_scores, bins=100, alpha=0.6, color='blue', label=f'Known (n={len(known_scores)})', density=True)
    axes[0].hist(unknown_scores, bins=100, alpha=0.6, color='red', label=f'Unknown (n={len(unknown_scores)})', density=True)
    axes[0].set_xlabel('Max Horosphere Score')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Max Horo Score: Known vs Unknown')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Norm distributions
    known_norms = [r['emb_norm'] for r in records if r['is_known']]
    unknown_norms = [r['emb_norm'] for r in records if not r['is_known']]
    axes[1].hist(known_norms, bins=100, alpha=0.6, color='blue', label=f'Known', density=True)
    axes[1].hist(unknown_norms, bins=100, alpha=0.6, color='red', label=f'Unknown', density=True)
    axes[1].set_xlabel('Embedding Norm (||x||)')
    axes[1].set_title('Embedding Norms: Known vs Unknown')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'diag_1_score_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: diag_1_score_distributions.png")

    # --- Figure 2: Per-class own-proto score boxplot ---
    fig, ax = plt.subplots(figsize=(14, 6))
    class_data = []
    class_labels = []
    for cls_name in known_class_names:
        recs = [r for r in records if r['class_name'] == cls_name]
        if recs:
            own_idx = cls_to_idx[cls_name]
            class_data.append([r['horo_scores'][own_idx] for r in recs])
            class_labels.append(f"{cls_name}\n(n={len(recs)})")

    bp = ax.boxplot(class_data, labels=class_labels, patch_artist=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_data)))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel('Own-Prototype Horosphere Score')
    ax.set_title('Per-Class: Score for Own Prototype (higher=better modeled)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'diag_2_per_class_own_score.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: diag_2_per_class_own_score.png")

    # --- Figure 3: Confusion heatmap (known class → assigned prototype) ---
    confusion = np.zeros((K, K))
    for r in records:
        if r['is_known'] and r['class_name'] in cls_to_idx:
            gt_idx = cls_to_idx[r['class_name']]
            pred_idx = r['assigned_proto']
            if pred_idx < K:
                confusion[gt_idx, pred_idx] += 1

    # Normalize rows
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    confusion_pct = confusion / row_sums * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion_pct, cmap='Blues')
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(known_class_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(known_class_names, fontsize=9)
    ax.set_xlabel('Assigned Prototype')
    ax.set_ylabel('GT Class')
    ax.set_title('Prototype Assignment Confusion (% of GT class)')
    for i in range(K):
        for j in range(K):
            val = confusion_pct[i, j]
            color = 'white' if val > 50 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center', color=color, fontsize=8)
    plt.colorbar(im, ax=ax, label='%')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'diag_3_confusion_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: diag_3_confusion_heatmap.png")

    # --- Figure 4: Unknown class → prototype affinity heatmap ---
    unk_classes = sorted(set(r['class_name'] for r in records if not r['is_known']))
    if unk_classes:
        affinity = np.zeros((len(unk_classes), K))
        counts = []
        for ui, ucls in enumerate(unk_classes):
            recs = [r for r in records if r['class_name'] == ucls]
            counts.append(len(recs))
            if recs:
                scores = np.array([r['horo_scores'] for r in recs])
                affinity[ui] = scores.mean(axis=0)

        fig, ax = plt.subplots(figsize=(12, max(4, len(unk_classes) * 0.5)))
        im = ax.imshow(affinity, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(K))
        ax.set_xticklabels(known_class_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(unk_classes)))
        ax.set_yticklabels([f"{c} (n={n})" for c, n in zip(unk_classes, counts)], fontsize=8)
        ax.set_xlabel('Known Prototype')
        ax.set_ylabel('Unknown Class')
        ax.set_title('Unknown Class → Known Prototype Affinity (mean horo score)')
        plt.colorbar(im, ax=ax, label='Mean Horosphere Score')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'diag_4_unknown_affinity.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: diag_4_unknown_affinity.png")

    # --- Figure 5: Detection rate bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Known classes
    known_names = []
    known_det_rates = []
    for cls_name in known_class_names:
        recs = [r for r in records if r['class_name'] == cls_name]
        if recs:
            known_names.append(cls_name)
            known_det_rates.append(100.0 * sum(1 for r in recs if r['detected']) / len(recs))

    bars = axes[0].barh(known_names, known_det_rates, color='steelblue')
    axes[0].set_xlabel('Detection Rate (%)')
    axes[0].set_title('Known Class Detection Rate (IoU>0.3)')
    axes[0].set_xlim(0, 100)
    for bar, val in zip(bars, known_det_rates):
        axes[0].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{val:.0f}%', va='center', fontsize=8)

    # Unknown classes
    if unk_classes:
        unk_det_rates = []
        unk_labels = []
        for cls_name in unk_classes:
            recs = [r for r in records if r['class_name'] == cls_name]
            if recs:
                unk_labels.append(cls_name)
                unk_det_rates.append(100.0 * sum(1 for r in recs if r['detected']) / len(recs))

        bars = axes[1].barh(unk_labels, unk_det_rates, color='coral')
        axes[1].set_xlabel('Detection Rate (%)')
        axes[1].set_title('Unknown Class Detection Rate (IoU>0.3)')
        axes[1].set_xlim(0, 100)
        for bar, val in zip(bars, unk_det_rates):
            axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{val:.0f}%', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'diag_5_detection_rates.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: diag_5_detection_rates.png")

    # --- Figure 6: Per-class score separation violin ---
    fig, ax = plt.subplots(figsize=(14, 6))
    # Collect max_horo scores per class (both known and unknown)
    all_classes = known_class_names + unk_classes
    all_data = []
    all_labels = []
    all_colors = []
    for cls_name in all_classes:
        recs = [r for r in records if r['class_name'] == cls_name]
        if len(recs) >= 5:
            all_data.append([r['max_horo'] for r in recs])
            is_k = cls_name in set(known_class_names)
            all_labels.append(f"{'★' if is_k else '?'} {cls_name}\n(n={len(recs)})")
            all_colors.append('steelblue' if is_k else 'coral')

    if all_data:
        parts = ax.violinplot(all_data, positions=range(len(all_data)), showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(all_colors[i])
            pc.set_alpha(0.6)
        ax.set_xticks(range(len(all_data)))
        ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Max Horosphere Score')
        ax.set_title('Score Distribution per Class (★=known, ?=unknown)')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'diag_6_score_violins.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: diag_6_score_violins.png")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--task", default="IDD_HYP/t1")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output_dir", default="diagnostics")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    args = parser.parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)

    task_name, split_name = args.task.split('/')
    base_dataset = task_name.replace('_HYP', '')
    dataset_key = base_dataset

    data_split = f"{base_dataset}/{split_name}"
    data_register = Register('./datasets/', data_split, cfg, dataset_key)
    data_register.register_dataset()

    all_class_names = list(inital_prompts()[dataset_key])
    unknown_index = cfg.TEST.PREV_INTRODUCED_CLS + cfg.TEST.CUR_INTRODUCED_CLS
    known_class_names = all_class_names[:unknown_index]

    print(f"\n=== Configuration ===")
    print(f"  Task: {args.task}, Split: {args.split}")
    print(f"  Known classes ({unknown_index}): {known_class_names}")
    print(f"  Checkpoint: {args.ckpt}")

    # Model config
    config_file = os.path.join("./configs", task_name, split_name + ".py")
    cfgY = Config.fromfile(config_file)
    cfgY.work_dir = "."

    # Auto-detect hyp config from checkpoint
    ckpt_data = torch.load(args.ckpt, map_location='cpu')
    hyp_config = ckpt_data.get('hyp_config', {})
    hyp_c = hyp_config.get('curvature', 1.0)
    hyp_dim = hyp_config.get('embed_dim', 64)
    clip_r = hyp_config.get('clip_r', 0.95)
    print(f"  hyp_config: c={hyp_c}, dim={hyp_dim}, clip_r={clip_r}")
    del ckpt_data

    # Initialize YOLO-World
    runner = Runner.from_cfg(cfgY)
    runner._hooks = [h for h in runner._hooks if not h.__class__.__name__.startswith('EMA')]
    runner.call_hook("before_run")
    runner.load_or_resume()
    runner.model.reparameterize([known_class_names])
    runner.model.eval()

    # Build data loader
    if args.split == 'train':
        import copy as _copy
        dl_cfg = _copy.deepcopy(cfgY.train_dataloader)
        dl_cfg['sampler'] = dict(type='DefaultSampler', shuffle=False)
        dl_cfg['batch_size'] = 16
        data_loader = Runner.build_dataloader(dl_cfg)
    else:
        data_loader = Runner.build_dataloader(cfgY.test_dataloader)

    print(f"  Data loader: {len(data_loader.dataset)} images ({args.split})")

    # Build hyperbolic model
    model = HypCustomYoloWorld(
        runner.model, unknown_index,
        hyp_c=hyp_c, hyp_dim=hyp_dim, clip_r=clip_r
    )

    print(f"\n=== Loading Checkpoint ===")
    with torch.no_grad():
        model = load_hyp_ckpt(model, args.ckpt,
                              cfg.TEST.PREV_INTRODUCED_CLS, cfg.TEST.CUR_INTRODUCED_CLS, eval=True)
        model = model.cuda()
        known_class_names_orig = list(known_class_names)
        model.add_generic_text(known_class_names, generic_prompt='object', alpha=0.4)
    model.eval()

    prototypes = model.prototypes.detach()
    biases = model.prototype_biases.detach()

    print(f"  Prototypes: {prototypes.shape}")
    print(f"  Biases: {[f'{b:.4f}' for b in biases.cpu().tolist()]}")

    # =====================================================================
    # Collect data
    # =====================================================================
    records = collect_diagnostic_data(
        model, data_loader, known_class_names_orig,
        dataset_root='./datasets', hyp_c=hyp_c,
    )

    if not records:
        print("ERROR: No records collected!")
        sys.exit(1)

    # =====================================================================
    # Generate report
    # =====================================================================
    os.makedirs(args.output_dir, exist_ok=True)

    report_sections = []
    report_sections.append(f"HYPERBOLIC MODEL DIAGNOSTIC REPORT")
    report_sections.append(f"Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report_sections.append(f"Checkpoint: {args.ckpt}")
    report_sections.append(f"Split: {args.split}")
    report_sections.append(f"Total GT boxes: {len(records)}")
    report_sections.append(f"Known classes: {known_class_names_orig}")
    report_sections.append(f"hyp_config: c={hyp_c}, dim={hyp_dim}, clip_r={clip_r}")
    report_sections.append("")

    print("\n\n" + "=" * 70)
    print("GENERATING REPORT...")
    print("=" * 70)

    # Section 1: Per-class prototype analysis
    s1 = analyze_per_class(records, known_class_names_orig, hyp_c)
    report_sections.append(s1)
    print(s1)

    # Section 2: Unknown proximity
    s2 = analyze_unknowns(records, known_class_names_orig, hyp_c)
    report_sections.append(s2)
    print(s2)

    # Section 3: Detection confidence
    s3 = analyze_detection(records, known_class_names_orig)
    report_sections.append(s3)
    print(s3)

    # Section 4: Hyperbolic geometry
    s4 = analyze_prototypes(prototypes.cpu(), biases.cpu(), known_class_names_orig, hyp_c)
    report_sections.append(s4)
    print(s4)

    # Section 5: Score separation
    s5 = analyze_score_separation(records, known_class_names_orig)
    report_sections.append(s5)
    print(s5)

    # Save report
    report_path = os.path.join(args.output_dir, 'diagnostic_report.txt')
    with open(report_path, 'w') as f:
        f.write("\n\n".join(report_sections))
    print(f"\n  Report saved: {report_path}")

    # Generate plots
    print(f"\n  Generating plots...")
    plot_diagnostics(records, known_class_names_orig, prototypes.cpu(), biases.cpu(),
                     args.output_dir, hyp_c)

    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC COMPLETE")
    print(f"  Report: {report_path}")
    print(f"  Plots:  {args.output_dir}/diag_*.png")
    print(f"{'='*60}")
