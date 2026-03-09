vcxz"""
Poincaré Ball Visualization for Geodesic Prototypical Classifiers — FULL TEST SET

Visualises ALL test-set hyperbolic embeddings (known + unknown) via UMAP
projection to the Poincaré disk, with geodesic Voronoi boundaries and
adaptive-threshold analysis.

Key diagnostic panels
---------------------
 1. UMAP Poincaré disk — all classes colour-coded
 2. Geodesic Voronoi decision boundaries (replaces old horospheres)
 3. Score distributions + adaptive thresholds overlay
 4. Per-class geodesic detail (3×3 grid)
 5. Known-only UMAP
 6. Unknown-only UMAP
 7. Embedding norms (zoomed + per-class box)
 8. Distance heatmap: mean d²(cls, proto_k)
 9. Norm pipeline (FPN → projector → Poincaré)
10. Per-class projector output norms
11. Prototype analysis: inter-prototype distances + norm bar chart
12. Statistical summary: AUROC, Cohen's d, per-class calibration table

Design changes vs v1 (Horospherical → Geodesic):
- busemann() replaced by model.compute_geodesic_scores()
- prototype_biases removed (no bias in geodesic classifier)
- Horosphere circles replaced by geodesic Voronoi boundaries
- Interior prototypes visualised AT true norm (not pushed to boundary)
- Adaptive thresholds from calibration stats overlaid on score plots
- Added AUROC / Cohen-d / separation metrics
"""

import os
import sys
import xml.etree.ElementTree as ET
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize as mplNorm
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from core import add_config
from core.util.model_ema import add_model_ema_configs
from core.pascal_voc import register_pascal_voc, inital_prompts
from core.hyp_customyoloworld import HypCustomYoloWorld, load_hyp_ckpt
from core.hyperbolic import pmath
from core.calibrate_thresholds import compute_thresholds

from mmengine.config import Config
from mmengine.runner import Runner
from detectron2.engine import default_argument_parser, default_setup
from detectron2.config import get_cfg


# ============================================================================
# Setup helpers
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


# ============================================================================
# Poincaré / tangent-space math (numpy)
# ============================================================================

def _logmap0_numpy(x, c):
    """Logarithmic map from Poincaré ball to tangent space at origin."""
    x = np.asarray(x, dtype=np.float64)
    sqrt_c = np.sqrt(c)
    x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
    x_norm = np.clip(x_norm, 1e-15, 1.0 / sqrt_c - 1e-5)
    atanh_val = np.arctanh(sqrt_c * x_norm)
    return x / (sqrt_c * x_norm + 1e-15) * atanh_val


def _expmap0_numpy(v, c):
    """Exponential map from tangent space at origin to Poincaré ball."""
    v = np.asarray(v, dtype=np.float64)
    sqrt_c = np.sqrt(c)
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    v_norm = np.clip(v_norm, 1e-15, None)
    return np.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm + 1e-15) * v


def _poincare_dist_2d(x, y, c=1.0):
    """Poincaré distance between two 2D points (numpy)."""
    x, y = np.asarray(x), np.asarray(y)
    diff = x - y
    nx2 = (x ** 2).sum(-1)
    ny2 = (y ** 2).sum(-1)
    num = (diff ** 2).sum(-1)
    denom = (1 - c * nx2) * (1 - c * ny2)
    denom = np.clip(denom, 1e-15, None)
    arg = 1 + 2 * c * num / denom
    return (1.0 / np.sqrt(c)) * np.arccosh(np.clip(arg, 1.0, None))


# ============================================================================
# XML parsing
# ============================================================================

def parse_all_xml_boxes(dataset_root, img_id, known_set):
    """Parse ALL bounding boxes from XML annotation, including unknowns."""
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
            boxes.append({
                'bbox': [x1, y1, x2, y2],
                'class_name': cls_name,
                'is_known': cls_name in known_set,
            })
        return boxes
    except Exception:
        return []


# ============================================================================
# Collect embeddings — FULL test set, XML-based (known + unknown)
# ============================================================================

def collect_embeddings(model, data_loader, known_class_names, dataset_root, hyp_c):
    """
    Iterate the ENTIRE test dataloader and collect:
      - Poincaré embeddings
      - Geodesic scores (-d²) to each prototype
      - Assigned prototype per sample
      - FPN / pre-clip / Poincaré norms for pipeline analysis
    """
    total_images = len(data_loader.dataset) if hasattr(data_loader, 'dataset') else '?'
    print(f"\n{'='*60}")
    print(f"COLLECTING EMBEDDINGS — FULL TEST SET (geodesic framework)")
    print(f"  Known classes: {known_class_names}")
    print(f"  Total images: {total_images}")
    print(f"{'='*60}")

    model.eval()
    known_set = set(known_class_names)

    all_embeddings = []
    all_class_names = []
    all_is_known = []
    all_geo_scores = []       # Full score vector per sample (K,)
    all_max_score = []        # max_k(-d²) per sample
    all_assigned_proto = []   # argmax_k per sample
    all_fpn_norms = []
    all_pre_clip_norms = []
    samples_count = defaultdict(int)

    with torch.no_grad():
        # Enable norm caching
        model.hyp_projector.store_norms = True

        for i, batch in enumerate(tqdm(data_loader, desc="Collecting embeddings")):
            if i > 0 and i % 500 == 0:
                n_known = sum(v for k, v in samples_count.items() if k in known_set)
                n_unk = sum(v for k, v in samples_count.items() if k not in known_set)
                tqdm.write(f"  [batch {i}] {len(samples_count)} classes | known={n_known} | unknown={n_unk}")
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
                # hyp_projector returns (poincare_embs, pre_expmap_norms)
                if isinstance(hyp_embeddings, tuple):
                    hyp_embeddings, _ = hyp_embeddings

                # Geodesic scores via model API
                geo_scores = model.compute_geodesic_scores(hyp_embeddings)  # (B, 8400, K)

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

                        emb = hyp_embeddings[b_idx, nearest_idx]       # (D,)
                        scores = geo_scores[b_idx, nearest_idx]         # (K,)

                        # Pipeline norms
                        fpn_norm_val = model.hyp_projector._cached_fpn_norms[b_idx, nearest_idx].item()
                        pre_clip_norm_val = model.hyp_projector._cached_pre_clip_norms[b_idx, nearest_idx].item()

                        max_score, assigned = scores.max(dim=0)

                        all_embeddings.append(emb.cpu())
                        all_class_names.append(cls_name)
                        all_is_known.append(is_known)
                        all_geo_scores.append(scores.cpu())
                        all_max_score.append(max_score.item())
                        all_assigned_proto.append(assigned.item())
                        all_fpn_norms.append(fpn_norm_val)
                        all_pre_clip_norms.append(pre_clip_norm_val)
                        samples_count[cls_name] += 1

            except Exception as e:
                print(f"  Error batch {i}: {e}")
                import traceback; traceback.print_exc()
                continue

    print(f"\n  Collection summary:")
    print(f"  {'Class':<25s} {'Type':<8s} {'Count':>6s}")
    print(f"  {'-'*45}")
    for cls_key in sorted(samples_count.keys()):
        tag = "KNOWN" if cls_key in known_set else "UNKNOWN"
        print(f"  {cls_key:<25s} {tag:<8s} {samples_count[cls_key]:>6d}")

    n_known = sum(v for k, v in samples_count.items() if k in known_set)
    n_unk = sum(v for k, v in samples_count.items() if k not in known_set)
    print(f"  {'-'*45}")
    print(f"  Total known: {n_known}, Total unknown: {n_unk}")

    if len(all_embeddings) == 0:
        return None

    embeddings = torch.stack(all_embeddings)
    geo_scores = torch.stack(all_geo_scores)
    return {
        'embeddings': embeddings,
        'class_names': np.array(all_class_names),
        'is_known': np.array(all_is_known),
        'geo_scores': geo_scores,          # (N, K) — full score vectors
        'max_score': np.array(all_max_score),
        'assigned_proto': np.array(all_assigned_proto),
        'fpn_norms': np.array(all_fpn_norms),
        'pre_clip_norms': np.array(all_pre_clip_norms),
    }


# ============================================================================
# UMAP projection
# ============================================================================

def project_umap(embeddings, prototypes, curvature=1.0, n_neighbors=15, min_dist=0.1):
    """
    Project high-dim Poincaré embeddings + interior prototypes to 2D Poincaré disk.
    Prototypes are kept at THEIR ACTUAL relative norm (not pushed to boundary).
    """
    import umap

    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().numpy()
    if torch.is_tensor(prototypes):
        prototypes = prototypes.detach().cpu().numpy()

    emb_tangent = _logmap0_numpy(embeddings, curvature)
    proto_tangent = _logmap0_numpy(prototypes, curvature)

    combined = np.vstack([emb_tangent, proto_tangent])
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Running UMAP (hyperboloid metric) on {len(combined)} points ...")
    reducer = umap.UMAP(
        output_metric='hyperboloid',
        n_neighbors=min(n_neighbors, len(combined) - 1),
        min_dist=min_dist,
        n_components=2,
        random_state=42,
    )
    combined_hyp = reducer.fit_transform(combined)

    # Hyperboloid → Poincaré disk
    x, y = combined_hyp[:, 0], combined_hyp[:, 1]
    z = np.sqrt(1 + x**2 + y**2)
    disk_x = x / (1 + z)
    disk_y = y / (1 + z)

    n_emb = len(embeddings)
    embeddings_2d = np.stack([disk_x[:n_emb], disk_y[:n_emb]], axis=1)
    prototypes_2d = np.stack([disk_x[n_emb:], disk_y[n_emb:]], axis=1)

    # NOTE: Unlike v1, we do NOT push prototypes to the boundary.
    # Interior prototypes stay at their UMAP-projected position.

    return embeddings_2d, prototypes_2d


# ============================================================================
# Drawing helpers
# ============================================================================

def _draw_disk(ax):
    """Draw the unit Poincaré disk boundary."""
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
    ax.fill(np.cos(theta), np.sin(theta), alpha=0.03, color='gray')


def draw_geodesic_voronoi(ax, prototypes_2d, known_class_names, color_map,
                          curvature=1.0, resolution=200, alpha=0.12):
    """
    Draw approximate geodesic Voronoi regions on the 2D Poincaré disk.

    For each point on a dense grid inside the disk, assign it to the nearest
    prototype (by Poincaré distance) and colour the background accordingly.
    This replaces horosphere circles for the geodesic framework.
    """
    lin = np.linspace(-0.99, 0.99, resolution)
    xx, yy = np.meshgrid(lin, lin)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)       # (res*res, 2)
    inside = (pts ** 2).sum(-1) < 0.99  # inside disk

    K = len(prototypes_2d)
    dists = np.full((len(pts), K), np.inf)
    for k in range(K):
        dists[inside, k] = _poincare_dist_2d(pts[inside], prototypes_2d[k], c=curvature)

    assignment = np.argmin(dists, axis=-1)
    assignment[~inside] = -1

    # Create coloured image
    img = np.ones((resolution, resolution, 4))  # RGBA white
    for k in range(K):
        cls = known_class_names[k] if k < len(known_class_names) else f'proto_{k}'
        c = color_map.get(cls, [0.5, 0.5, 0.5, 1.0])
        mask_k = (assignment.reshape(resolution, resolution) == k)
        for ch in range(3):
            img[:, :, ch][mask_k] = c[ch] if hasattr(c, '__len__') else c
        img[:, :, 3][mask_k] = alpha

    ax.imshow(img, extent=(-0.99, 0.99, -0.99, 0.99), origin='lower', aspect='equal', zorder=0)


def make_versioned_dir(base_dir):
    """Create a versioned subdirectory: v1, v2, ... inside base_dir."""
    os.makedirs(base_dir, exist_ok=True)
    version = 1
    while True:
        candidate = os.path.join(base_dir, f'v{version}')
        if not os.path.exists(candidate):
            os.makedirs(candidate)
            return candidate
        version += 1


# ============================================================================
# Metric helpers
# ============================================================================

def compute_auroc(known_scores, unknown_scores):
    """AUROC for OOD detection (known = positive, higher score = more ID)."""
    try:
        from sklearn.metrics import roc_auc_score
        labels = np.concatenate([np.ones(len(known_scores)), np.zeros(len(unknown_scores))])
        scores = np.concatenate([known_scores, unknown_scores])
        return roc_auc_score(labels, scores)
    except Exception:
        return float('nan')


def compute_cohens_d(a, b):
    """Effect size: Cohen's d between two distributions."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float('nan')
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std < 1e-12:
        return float('nan')
    return (a.mean() - b.mean()) / pooled_std


def compute_fpr_at_tpr(known_scores, unknown_scores, tpr_level=0.95):
    """FPR@TPR95: fraction of unknowns misclassified as known at 95% known recall."""
    try:
        from sklearn.metrics import roc_curve
        labels = np.concatenate([np.ones(len(known_scores)), np.zeros(len(unknown_scores))])
        scores = np.concatenate([known_scores, unknown_scores])
        fpr, tpr, _ = roc_curve(labels, scores)
        idx = np.searchsorted(tpr, tpr_level)
        return fpr[min(idx, len(fpr) - 1)]
    except Exception:
        return float('nan')


# ============================================================================
# Plotting — all diagnostic panels
# ============================================================================

def plot_all_diagnostics(data, prototypes_2d, embeddings_2d, prototypes_hd,
                         known_class_names, save_dir, curvature=1.0,
                         adaptive_stats=None):
    """
    Generate all diagnostic visualisations.

    Parameters
    ----------
    data : dict from collect_embeddings
    prototypes_2d : (K, 2) array — 2D Poincaré positions
    embeddings_2d : (N, 2) array — 2D Poincaré positions
    prototypes_hd : tensor (K, D) — high-dimensional prototypes
    known_class_names : list[str]
    save_dir : str
    curvature : float
    adaptive_stats : dict or None — calibration data from checkpoint
    """
    class_names_arr = data['class_names']
    is_known_arr = data['is_known']
    max_score_arr = data['max_score']
    assigned_proto_arr = data['assigned_proto']
    geo_scores = data['geo_scores']  # (N, K) tensor
    embeddings_hd = data['embeddings']
    fpn_norms = data['fpn_norms']
    pre_clip_norms = data['pre_clip_norms']

    known_set = set(known_class_names)
    known_mask = is_known_arr.astype(bool)
    unk_mask = ~known_mask
    n_protos = len(known_class_names)

    unique_known = sorted([c for c in np.unique(class_names_arr) if c in known_set])
    unique_unknown = sorted([c for c in np.unique(class_names_arr) if c not in known_set])

    cmap_known = plt.cm.tab20(np.linspace(0, 1, max(len(unique_known), 1)))
    cmap_unknown = plt.cm.Set1(np.linspace(0, 1, max(len(unique_unknown), 1)))
    color_map = {}
    for i, c in enumerate(unique_known):
        color_map[c] = cmap_known[i]
    for i, c in enumerate(unique_unknown):
        color_map[c] = cmap_unknown[i]

    # Compute adaptive thresholds if stats available
    thresholds = None
    alpha_used = None
    if adaptive_stats is not None and adaptive_stats.get('per_class'):
        thresholds, alpha_used = compute_thresholds(adaptive_stats, known_class_names)
        thresholds = thresholds.numpy()

    # ==================================================================
    # Figure 1: UMAP Poincaré disk — all classes (no boundaries)
    # ==================================================================
    fig1, ax1 = plt.subplots(1, 1, figsize=(14, 14))
    _draw_disk(ax1)

    for cls in unique_known:
        mask = class_names_arr == cls
        ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=[color_map[cls]], s=12, alpha=0.5, label=f'{cls} ({mask.sum()})')
    for cls in unique_unknown:
        mask = class_names_arr == cls
        ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=[color_map[cls]], s=12, alpha=0.5, marker='x',
                    label=f'*{cls} ({mask.sum()})')

    for i in range(n_protos):
        c = color_map.get(known_class_names[i], 'gray')
        ax1.scatter(prototypes_2d[i, 0], prototypes_2d[i, 1],
                    c=[c], s=400, marker='*', edgecolors='black', linewidths=1.5, zorder=5)
        direction = prototypes_2d[i] / (np.linalg.norm(prototypes_2d[i]) + 1e-8)
        label_pos = direction * 1.12
        ax1.annotate(known_class_names[i], label_pos, fontsize=8, ha='center',
                     va='center', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    ax1.set_xlim(-1.25, 1.25); ax1.set_ylim(-1.25, 1.25)
    ax1.set_aspect('equal')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7, markerscale=2)
    ax1.set_title('Poincare Disk — Geodesic Prototypical (star = interior prototypes)\n'
                  '(dot = known,  x = unknown)', fontsize=13)
    ax1.grid(True, alpha=0.2)
    plt.tight_layout()
    fig1.savefig(os.path.join(save_dir, '01_umap_full_testset.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"  Saved: 01_umap_full_testset.png")

    # ==================================================================
    # Figure 2: UMAP + Geodesic Voronoi boundaries
    # ==================================================================
    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 14))
    _draw_disk(ax2)

    print("  Drawing geodesic Voronoi boundaries ...")
    draw_geodesic_voronoi(ax2, prototypes_2d, known_class_names, color_map,
                          curvature=curvature, resolution=250, alpha=0.10)

    for cls in unique_known:
        mask = class_names_arr == cls
        ax2.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=[color_map[cls]], s=10, alpha=0.4)
    for cls in unique_unknown:
        mask = class_names_arr == cls
        ax2.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=[color_map[cls]], s=10, alpha=0.4, marker='x')

    for i in range(n_protos):
        c = color_map.get(known_class_names[i], 'gray')
        ax2.scatter(prototypes_2d[i, 0], prototypes_2d[i, 1],
                    c=[c], s=400, marker='*', edgecolors='black', linewidths=1.5, zorder=5)
        direction = prototypes_2d[i] / (np.linalg.norm(prototypes_2d[i]) + 1e-8)
        label_pos = direction * 1.12
        ax2.annotate(known_class_names[i], label_pos, fontsize=8, ha='center',
                     va='center', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    ax2.set_xlim(-1.25, 1.25); ax2.set_ylim(-1.25, 1.25)
    ax2.set_aspect('equal')
    ax2.set_title('Poincare Disk + Geodesic Voronoi Decision Boundaries\n'
                  'Coloured regions = nearest-prototype assignment', fontsize=13)
    ax2.grid(True, alpha=0.2)
    plt.tight_layout()
    fig2.savefig(os.path.join(save_dir, '02_umap_voronoi.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: 02_umap_voronoi.png")

    # ==================================================================
    # Figure 3: Score distributions + adaptive threshold overlay
    # ==================================================================
    known_scores = max_score_arr[known_mask]
    unknown_scores = max_score_arr[unk_mask] if unk_mask.any() else np.array([])

    fig3, axes3 = plt.subplots(1, 3, figsize=(24, 7))

    # Left: overall known vs unknown
    ax = axes3[0]
    all_scores = np.concatenate([known_scores, unknown_scores]) if len(unknown_scores) > 0 else known_scores
    lo, hi = all_scores.min() - 0.5, all_scores.max() + 0.5
    bins = np.linspace(lo, hi, 80)
    ax.hist(known_scores, bins=bins, alpha=0.6, color='blue',
            label=f'Known (n={len(known_scores)})', density=True)
    if len(unknown_scores) > 0:
        ax.hist(unknown_scores, bins=bins, alpha=0.6, color='red',
                label=f'Unknown (n={len(unknown_scores)})', density=True)
    if thresholds is not None:
        global_tau = thresholds.mean()
        ax.axvline(global_tau, color='green', linestyle='--', linewidth=2,
                   label=f'Mean threshold tau={global_tau:.2f}')
    ax.set_xlabel('Max Geodesic Score  max_k(-d^2)')
    ax.set_ylabel('Density')
    ax.set_title('Known vs Unknown: Max Geodesic Score\n(higher = more in-distribution)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Middle: per-known-class with individual thresholds
    ax = axes3[1]
    for ci, cls in enumerate(known_class_names[:9]):  # max 9
        mask = class_names_arr == cls
        if mask.sum() > 0:
            ax.hist(max_score_arr[mask], bins=40, alpha=0.4,
                    label=f'{cls} ({mask.sum()})', density=True)
            if thresholds is not None and ci < len(thresholds):
                ax.axvline(thresholds[ci], linestyle=':', linewidth=1.5,
                           color=color_map.get(cls, 'gray')[:3])
    ax.set_xlabel('Max Geodesic Score')
    ax.set_ylabel('Density')
    ax.set_title(f'Per-Known-Class Scores\n(dotted lines = adaptive tau, alpha={alpha_used or "N/A"})')
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    # Right: per-unknown-class
    ax = axes3[2]
    for cls in unique_unknown:
        mask = class_names_arr == cls
        if mask.sum() > 0:
            ax.hist(max_score_arr[mask], bins=30, alpha=0.5,
                    label=f'{cls} ({mask.sum()})', density=True)
    if thresholds is not None:
        ax.axvline(thresholds.mean(), color='green', linestyle='--',
                   linewidth=2, label=f'Mean tau={thresholds.mean():.2f}')
    ax.set_xlabel('Max Geodesic Score')
    ax.set_ylabel('Density')
    ax.set_title('Per-Unknown-Class Score Distributions')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig3.savefig(os.path.join(save_dir, '03_score_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"  Saved: 03_score_distributions.png")

    # ==================================================================
    # Figure 4: Per-class geodesic detail (3x3 grid)
    # ==================================================================
    n_grid = min(n_protos, 9)
    ncols = 3
    nrows = (n_grid + ncols - 1) // ncols
    fig4, axes4 = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    if nrows == 1:
        axes4 = axes4[np.newaxis, :]

    for idx in range(n_grid):
        r, c_idx = idx // ncols, idx % ncols
        ax = axes4[r][c_idx]
        cls = known_class_names[idx]
        _draw_disk(ax)

        mask_k = (class_names_arr == cls)
        if mask_k.sum() > 0:
            ax.scatter(embeddings_2d[mask_k, 0], embeddings_2d[mask_k, 1],
                       c='blue', s=8, alpha=0.4, label=f'{cls} ({mask_k.sum()})')
        if unk_mask.sum() > 0:
            ax.scatter(embeddings_2d[unk_mask, 0], embeddings_2d[unk_mask, 1],
                       c='red', s=6, alpha=0.15, marker='x', label=f'Unknowns ({unk_mask.sum()})')

        ax.scatter(prototypes_2d[idx, 0], prototypes_2d[idx, 1],
                   c='gold', s=300, marker='*', edgecolors='black', linewidths=1.5, zorder=5)

        # Draw distance circles around prototype (iso-distance contours)
        for radius_frac in [0.3, 0.6, 0.9]:
            theta = np.linspace(0, 2 * np.pi, 100)
            circ_x = prototypes_2d[idx, 0] + radius_frac * np.cos(theta)
            circ_y = prototypes_2d[idx, 1] + radius_frac * np.sin(theta)
            inside = (circ_x**2 + circ_y**2) < 0.99
            circ_x[~inside] = np.nan
            circ_y[~inside] = np.nan
            ax.plot(circ_x, circ_y, 'k:', linewidth=0.5, alpha=0.3)

        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        tau_str = f', tau={thresholds[idx]:.1f}' if thresholds is not None and idx < len(thresholds) else ''
        ax.set_title(f'{cls}{tau_str}', fontsize=10)
        ax.legend(fontsize=6, loc='lower right')

    for idx in range(n_grid, nrows * ncols):
        r, c_idx = idx // ncols, idx % ncols
        axes4[r][c_idx].set_visible(False)

    plt.suptitle('Per-Class Geodesic Detail: Known (blue) vs Unknown (red)', fontsize=14, y=1.01)
    plt.tight_layout()
    fig4.savefig(os.path.join(save_dir, '04_per_class_detail.png'), dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print(f"  Saved: 04_per_class_detail.png")

    # ==================================================================
    # Figure 5: Known-only UMAP
    # ==================================================================
    fig5, ax5 = plt.subplots(1, 1, figsize=(14, 14))
    _draw_disk(ax5)
    draw_geodesic_voronoi(ax5, prototypes_2d, known_class_names, color_map,
                          curvature=curvature, resolution=150, alpha=0.06)

    for cls in unique_known:
        mask = class_names_arr == cls
        ax5.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=[color_map[cls]], s=14, alpha=0.5, label=f'{cls} ({mask.sum()})')
    for i in range(n_protos):
        c = color_map.get(known_class_names[i], 'gray')
        ax5.scatter(prototypes_2d[i, 0], prototypes_2d[i, 1],
                    c=[c], s=400, marker='*', edgecolors='black', linewidths=1.5, zorder=5)
        direction = prototypes_2d[i] / (np.linalg.norm(prototypes_2d[i]) + 1e-8)
        ax5.annotate(known_class_names[i], direction * 1.12, fontsize=8, ha='center',
                     va='center', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    ax5.set_xlim(-1.25, 1.25); ax5.set_ylim(-1.25, 1.25)
    ax5.set_aspect('equal')
    ax5.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7, markerscale=2)
    ax5.set_title('KNOWN CLASSES ONLY — Geodesic Prototypical UMAP\n'
                  '(Are clusters tight around their interior prototype?)', fontsize=13)
    ax5.grid(True, alpha=0.2)
    plt.tight_layout()
    fig5.savefig(os.path.join(save_dir, '05_umap_known_only.png'), dpi=150, bbox_inches='tight')
    plt.close(fig5)
    print(f"  Saved: 05_umap_known_only.png")

    # ==================================================================
    # Figure 6: Unknown-only UMAP
    # ==================================================================
    fig6, ax6 = plt.subplots(1, 1, figsize=(14, 14))
    _draw_disk(ax6)
    draw_geodesic_voronoi(ax6, prototypes_2d, known_class_names, color_map,
                          curvature=curvature, resolution=150, alpha=0.04)

    for cls in unique_unknown:
        mask = class_names_arr == cls
        ax6.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=[color_map[cls]], s=20, alpha=0.6, marker='x',
                    label=f'{cls} ({mask.sum()})')
    for i in range(n_protos):
        c = color_map.get(known_class_names[i], 'gray')
        ax6.scatter(prototypes_2d[i, 0], prototypes_2d[i, 1],
                    c=[c], s=400, marker='*', edgecolors='black', linewidths=1.5, zorder=5)
        direction = prototypes_2d[i] / (np.linalg.norm(prototypes_2d[i]) + 1e-8)
        ax6.annotate(known_class_names[i], direction * 1.12, fontsize=8, ha='center',
                     va='center', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    ax6.set_xlim(-1.25, 1.25); ax6.set_ylim(-1.25, 1.25)
    ax6.set_aspect('equal')
    ax6.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7, markerscale=2)
    ax6.set_title('UNKNOWN CLASSES ONLY — Geodesic Prototypical UMAP\n'
                  '(Ideal: unknowns far from prototypes, scattered or near center)', fontsize=13)
    ax6.grid(True, alpha=0.2)
    plt.tight_layout()
    fig6.savefig(os.path.join(save_dir, '06_umap_unknown_only.png'), dpi=150, bbox_inches='tight')
    plt.close(fig6)
    print(f"  Saved: 06_umap_unknown_only.png")

    # ==================================================================
    # Figure 7: Embedding norms (zoomed + per-class box)
    # ==================================================================
    embhd = embeddings_hd.numpy() if torch.is_tensor(embeddings_hd) else embeddings_hd
    all_poincare_norms = np.linalg.norm(embhd, axis=-1)
    known_norms_hd = all_poincare_norms[known_mask]
    unknown_norms_hd = all_poincare_norms[unk_mask] if unk_mask.any() else np.array([])

    fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(18, 6))

    all_n = np.concatenate([known_norms_hd, unknown_norms_hd]) if len(unknown_norms_hd) > 0 else known_norms_hd
    lo, hi = all_n.min() - 0.01, all_n.max() + 0.01
    bins = np.linspace(lo, hi, 80)
    ax7a.hist(known_norms_hd, bins=bins, alpha=0.5, color='blue',
              label=f'Known (n={len(known_norms_hd)})', density=True)
    if len(unknown_norms_hd) > 0:
        ax7a.hist(unknown_norms_hd, bins=bins, alpha=0.5, color='red',
                  label=f'Unknown (n={len(unknown_norms_hd)})', density=True)
    ax7a.set_xlabel('||x||_Poincare')
    ax7a.set_ylabel('Density')
    ax7a.set_title('Poincare Norm Distribution\n(higher = closer to boundary = more "confident")')
    ax7a.legend(fontsize=8); ax7a.grid(True, alpha=0.3)

    # Per-class boxplot
    class_norm_data = []
    class_norm_labels = []
    class_norm_colors = []
    for cls_name in known_class_names:
        m = class_names_arr == cls_name
        if m.sum() > 0:
            class_norm_data.append(np.linalg.norm(embhd[m], axis=-1))
            class_norm_labels.append(f'{cls_name}\n(n={m.sum()})')
            class_norm_colors.append('steelblue')
    for cls_name in unique_unknown[:6]:
        m = class_names_arr == cls_name
        if m.sum() > 0:
            class_norm_data.append(np.linalg.norm(embhd[m], axis=-1))
            class_norm_labels.append(f'*{cls_name}\n(n={m.sum()})')
            class_norm_colors.append('coral')
    if class_norm_data:
        bp = ax7b.boxplot(class_norm_data, labels=class_norm_labels, patch_artist=True, vert=True)
        for patch, col in zip(bp['boxes'], class_norm_colors):
            patch.set_facecolor(col); patch.set_alpha(0.5)
        ax7b.tick_params(axis='x', rotation=45, labelsize=7)
        ax7b.set_ylabel('||x||_Poincare')
        ax7b.set_title('Per-Class Embedding Norms (blue=known, red=unknown)')
        ax7b.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig7.savefig(os.path.join(save_dir, '07_embedding_norms.png'), dpi=150, bbox_inches='tight')
    plt.close(fig7)
    print(f"  Saved: 07_embedding_norms.png")

    # ==================================================================
    # Figure 8: Distance heatmap — mean geodesic score per class per proto
    # ==================================================================
    geo_np = geo_scores.numpy() if torch.is_tensor(geo_scores) else geo_scores  # (N, K)

    all_classes_for_heatmap = list(unique_known) + list(unique_unknown)
    valid_labels = []
    mat_rows = []
    for cls_name in all_classes_for_heatmap:
        mask = class_names_arr == cls_name
        if mask.sum() < 3:
            continue
        is_k = cls_name in known_set
        valid_labels.append(f"{'[K]' if is_k else '[U]'} {cls_name} (n={mask.sum()})")
        # Mean geodesic score (-d²) to each prototype — less negative = closer
        mat_rows.append(geo_np[mask].mean(axis=0))  # (K,)

    if len(mat_rows) > 0:
        mat = np.stack(mat_rows)
        fig8, ax8 = plt.subplots(figsize=(12, max(5, len(valid_labels) * 0.4)))
        im = ax8.imshow(mat, cmap='RdYlGn', aspect='auto')  # green=high score(close), red=low(far)
        ax8.set_xticks(range(n_protos))
        ax8.set_xticklabels(known_class_names, rotation=45, ha='right', fontsize=9)
        ax8.set_yticks(range(len(valid_labels)))
        ax8.set_yticklabels(valid_labels, fontsize=7)
        ax8.set_xlabel('Known Prototype')
        ax8.set_ylabel('Class')
        ax8.set_title('Mean Geodesic Score -d^2 to Each Prototype\n'
                      '(green = high score / close,  red = low score / far)')
        for i in range(len(valid_labels)):
            for j in range(n_protos):
                ax8.text(j, i, f'{mat[i,j]:.1f}', ha='center', va='center', fontsize=6)
        plt.colorbar(im, ax=ax8, label='Mean -d^2', shrink=0.8)
        plt.tight_layout()
        fig8.savefig(os.path.join(save_dir, '08_geodesic_score_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close(fig8)
        print(f"  Saved: 08_geodesic_score_heatmap.png")

    # ==================================================================
    # Figure 9: Norm pipeline (FPN → Projector → Poincaré)
    # ==================================================================
    poincare_norms = all_poincare_norms
    stages = ['FPN Output', 'Post-Projector\n(pre-clip)', 'Poincare\n(post-expmap0)']
    known_data = [fpn_norms[known_mask], pre_clip_norms[known_mask], poincare_norms[known_mask]]
    unk_data = [fpn_norms[unk_mask], pre_clip_norms[unk_mask], poincare_norms[unk_mask]]

    fig9, axes9 = plt.subplots(1, 3, figsize=(20, 6))
    fig9.suptitle('Norm Pipeline: Where Does Information Collapse?', fontsize=16, fontweight='bold')

    for idx, (ax, stage, kd, ud) in enumerate(zip(axes9, stages, known_data, unk_data)):
        bp_data = [kd]
        bp_labels = ['Known']
        if len(ud) > 0:
            bp_data.append(ud)
            bp_labels.append('Unknown')
        bp = ax.boxplot(bp_data, labels=bp_labels, widths=0.5,
                        patch_artist=True, showfliers=False)
        bp['boxes'][0].set_facecolor('#4CAF50'); bp['boxes'][0].set_alpha(0.7)
        if len(ud) > 0 and len(bp['boxes']) > 1:
            bp['boxes'][1].set_facecolor('#F44336'); bp['boxes'][1].set_alpha(0.7)

        ax.set_title(stage, fontsize=13, fontweight='bold')
        ax.set_ylabel('||x||' if idx == 0 else '')

        k_mean = kd.mean() if len(kd) > 0 else 0
        u_mean = ud.mean() if len(ud) > 0 else 0
        ax.scatter([1], [k_mean], marker='D', color='darkgreen', s=60, zorder=5, label=f'mu={k_mean:.3f}')
        if len(ud) > 0:
            ax.scatter([2], [u_mean], marker='D', color='darkred', s=60, zorder=5, label=f'mu={u_mean:.3f}')
        ax.legend(fontsize=9)

        if k_mean > 0 and u_mean > 0:
            ratio = u_mean / k_mean
            ax.text(0.5, 0.02, f'unk/known ratio: {ratio:.3f}',
                    transform=ax.transAxes, ha='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    fig9.savefig(os.path.join(save_dir, '09_norm_pipeline.png'), dpi=150, bbox_inches='tight')
    plt.close(fig9)
    print(f"  Saved: 09_norm_pipeline.png")

    # ==================================================================
    # Figure 10: Per-class projector output norms
    # ==================================================================
    fig10, (ax10a, ax10b) = plt.subplots(2, 1, figsize=(16, 10))
    fig10.suptitle('Per-Class Projector Output Norms (Pre-Clip Euclidean)', fontsize=14, fontweight='bold')

    known_classes_sorted = sorted(set(class_names_arr[known_mask]))
    known_vals = [pre_clip_norms[class_names_arr == c] for c in known_classes_sorted]
    if known_vals:
        bp_k = ax10a.boxplot(known_vals, labels=known_classes_sorted, patch_artist=True, showfliers=False)
        for box in bp_k['boxes']:
            box.set_facecolor('#4CAF50'); box.set_alpha(0.6)
        ax10a.set_title('Known Classes'); ax10a.set_ylabel('Pre-clip ||v||'); ax10a.tick_params(axis='x', rotation=30)

    unk_classes_sorted = sorted(set(class_names_arr[unk_mask]))
    unk_vals = [pre_clip_norms[class_names_arr == c] for c in unk_classes_sorted]
    if unk_vals:
        bp_u = ax10b.boxplot(unk_vals, labels=unk_classes_sorted, patch_artist=True, showfliers=False)
        for box in bp_u['boxes']:
            box.set_facecolor('#F44336'); box.set_alpha(0.6)
        ax10b.set_title('Unknown Classes'); ax10b.set_ylabel('Pre-clip ||v||'); ax10b.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    fig10.savefig(os.path.join(save_dir, '10_per_class_projector_norms.png'), dpi=150, bbox_inches='tight')
    plt.close(fig10)
    print(f"  Saved: 10_per_class_projector_norms.png")

    # ==================================================================
    # Figure 11: Prototype analysis — inter-distances + norm bar chart
    # ==================================================================
    proto_np = prototypes_hd.cpu().numpy() if torch.is_tensor(prototypes_hd) else prototypes_hd
    proto_norms = np.linalg.norm(proto_np, axis=-1)
    K = len(proto_np)

    # Pairwise geodesic distances between prototypes (using torch pmath)
    proto_t = torch.tensor(proto_np, dtype=torch.float32)
    pair_dists = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i != j:
                pair_dists[i, j] = pmath.dist(proto_t[i:i+1], proto_t[j:j+1], c=curvature).item()

    fig11, (ax11a, ax11b) = plt.subplots(1, 2, figsize=(18, 7))
    fig11.suptitle('Prototype Analysis', fontsize=16, fontweight='bold')

    # Left: pairwise distance heatmap
    im = ax11a.imshow(pair_dists, cmap='YlOrRd', aspect='equal')
    ax11a.set_xticks(range(K)); ax11a.set_xticklabels(known_class_names, rotation=45, ha='right', fontsize=8)
    ax11a.set_yticks(range(K)); ax11a.set_yticklabels(known_class_names, fontsize=8)
    for i in range(K):
        for j in range(K):
            ax11a.text(j, i, f'{pair_dists[i,j]:.2f}', ha='center', va='center', fontsize=6)
    plt.colorbar(im, ax=ax11a, label='Geodesic dist d(z_i, z_j)', shrink=0.8)
    ax11a.set_title('Inter-Prototype Geodesic Distances\n(Are prototypes well-separated?)')

    # Right: prototype norm bar chart
    bars = ax11b.bar(range(K), proto_norms, color='steelblue', alpha=0.8)
    ax11b.set_xticks(range(K)); ax11b.set_xticklabels(known_class_names, rotation=45, ha='right', fontsize=8)
    ax11b.set_ylabel('||z_k||_Poincare')
    ax11b.set_title('Prototype Norms (should be <= max_proto_norm)')
    ax11b.axhline(0.5, color='red', linestyle='--', linewidth=2, label='max_proto_norm=0.5')
    ax11b.axhline(0.4, color='orange', linestyle=':', linewidth=1.5, label='init_norm=0.4')
    ax11b.legend()
    for i, (bar, n) in enumerate(zip(bars, proto_norms)):
        ax11b.text(bar.get_x() + bar.get_width()/2, n + 0.01, f'{n:.3f}',
                   ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig11.savefig(os.path.join(save_dir, '11_prototype_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close(fig11)
    print(f"  Saved: 11_prototype_analysis.png")

    # ==================================================================
    # Figure 12: Statistical summary dashboard
    # ==================================================================
    auroc = compute_auroc(known_scores, unknown_scores) if len(unknown_scores) > 0 else float('nan')
    cohens_d = compute_cohens_d(known_scores, unknown_scores) if len(unknown_scores) > 0 else float('nan')
    fpr95 = compute_fpr_at_tpr(known_scores, unknown_scores) if len(unknown_scores) > 0 else float('nan')

    fig12, axes12 = plt.subplots(2, 2, figsize=(18, 14))
    fig12.suptitle('Statistical Diagnostic Summary', fontsize=16, fontweight='bold')

    # Top-left: OOD metrics summary text
    ax = axes12[0, 0]
    ax.axis('off')
    summary_lines = [
        f"AUROC (known vs unknown):  {auroc:.4f}",
        f"Cohen's d:                 {cohens_d:.4f}",
        f"FPR@95% TPR:               {fpr95:.4f}",
        "",
        f"Known samples:    {len(known_scores)}",
        f"Unknown samples:  {len(unknown_scores)}",
        "",
        f"Known score:   mu={known_scores.mean():.2f}  sigma={known_scores.std():.2f}",
    ]
    if len(unknown_scores) > 0:
        summary_lines.append(f"Unknown score: mu={unknown_scores.mean():.2f}  sigma={unknown_scores.std():.2f}")
        gap = known_scores.mean() - unknown_scores.mean()
        summary_lines.append(f"Score gap (mu_k - mu_u): {gap:.2f}")
    summary_lines += [
        "",
        f"Prototype norms:  min={proto_norms.min():.4f}  max={proto_norms.max():.4f}  mean={proto_norms.mean():.4f}",
        f"Emb norms (known):  mu={known_norms_hd.mean():.4f}  sigma={known_norms_hd.std():.4f}",
    ]
    if len(unknown_norms_hd) > 0:
        summary_lines.append(f"Emb norms (unk):    mu={unknown_norms_hd.mean():.4f}  sigma={unknown_norms_hd.std():.4f}")

    summary_lines += [
        "",
        f"Inter-proto dist:  min={pair_dists[pair_dists > 0].min():.4f}" if (pair_dists > 0).any() else "",
        f"                   max={pair_dists.max():.4f}  mean={pair_dists[pair_dists > 0].mean():.4f}" if (pair_dists > 0).any() else "",
    ]

    if thresholds is not None:
        summary_lines += [
            "",
            f"Thresholds (alpha={alpha_used:.2f}):",
        ]
        for ci, cls in enumerate(known_class_names):
            if ci < len(thresholds):
                summary_lines.append(f"  {cls}: tau={thresholds[ci]:.2f}")

    ax.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # Top-right: per-class calibration quality (mean +/- std per class)
    ax = axes12[0, 1]
    if adaptive_stats is not None and adaptive_stats.get('per_class'):
        cal = adaptive_stats['per_class']
        cls_labels_cal = []
        means = []
        stds = []
        for cls_name in known_class_names:
            if cls_name in cal and cal[cls_name]['count'] > 0:
                cls_labels_cal.append(cls_name)
                means.append(cal[cls_name]['mean'])
                stds.append(cal[cls_name]['std'])
        if cls_labels_cal:
            y_pos = np.arange(len(cls_labels_cal))
            ax.barh(y_pos, means, xerr=stds, height=0.6, color='steelblue', alpha=0.7,
                    capsize=3, label='mu +/- sigma (training calibration)')
            if thresholds is not None:
                for yi, ci in enumerate(range(len(cls_labels_cal))):
                    if ci < len(thresholds):
                        ax.plot(thresholds[ci], yi, 'rv', markersize=8, zorder=5)
                ax.plot([], [], 'rv', markersize=8, label='Threshold tau')
            ax.set_yticks(y_pos); ax.set_yticklabels(cls_labels_cal, fontsize=8)
            ax.set_xlabel('Geodesic Score (-d^2)')
            ax.set_title('Per-Class Calibration (training set)\nbar=mu, whisker=sigma, triangle=tau')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='x')
    else:
        ax.text(0.5, 0.5, 'No adaptive_stats\nin checkpoint', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='gray')
        ax.set_title('Per-Class Calibration')

    # Bottom-left: prototype utilization (how many test samples assigned to each proto)
    ax = axes12[1, 0]
    proto_counts = np.zeros(n_protos)
    for pi in range(n_protos):
        proto_counts[pi] = (assigned_proto_arr == pi).sum()
    bars = ax.bar(range(n_protos), proto_counts, color='steelblue', alpha=0.8)
    for i, bar in enumerate(bars):
        c = color_map.get(known_class_names[i], 'steelblue')
        bar.set_facecolor(c)
    ax.set_xticks(range(n_protos))
    ax.set_xticklabels(known_class_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('# Test Samples Assigned')
    ax.set_title('Prototype Utilization (test set)\n(Unbalanced = some prototypes attract too many/few)')
    for i, (bar, cnt) in enumerate(zip(bars, proto_counts)):
        ax.text(bar.get_x() + bar.get_width()/2, cnt + 0.5, f'{int(cnt)}',
                ha='center', va='bottom', fontsize=8)

    # Bottom-right: known vs unknown prototype assignment confusion
    ax = axes12[1, 1]
    known_correct = 0
    known_total = 0
    for ci, cls in enumerate(known_class_names):
        mask = (class_names_arr == cls) & known_mask
        if mask.sum() > 0:
            known_total += mask.sum()
            known_correct += (assigned_proto_arr[mask] == ci).sum()

    unk_proto_hist = np.zeros(n_protos)
    if unk_mask.any():
        for pi in range(n_protos):
            unk_proto_hist[pi] = ((assigned_proto_arr == pi) & unk_mask).sum()

    accuracy = known_correct / max(known_total, 1)
    bars = ax.bar(range(n_protos), unk_proto_hist, color='#F44336', alpha=0.6, label='Unknowns')
    ax.set_xticks(range(n_protos))
    ax.set_xticklabels(known_class_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('# Unknown Samples Assigned')
    ax.set_title(f'Unknown -> Prototype Attraction\n'
                 f'(Known assignment accuracy: {accuracy:.1%} [{known_correct}/{known_total}])')
    ax.legend()
    for i, (bar, cnt) in enumerate(zip(bars, unk_proto_hist)):
        if cnt > 0:
            ax.text(bar.get_x() + bar.get_width()/2, cnt + 0.5, f'{int(cnt)}',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig12.savefig(os.path.join(save_dir, '12_statistical_summary.png'), dpi=150, bbox_inches='tight')
    plt.close(fig12)
    print(f"  Saved: 12_statistical_summary.png")

    # ==================================================================
    # Print norm pipeline summary to stdout
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"NORM PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Stage':<25s} {'Known (mu+/-sigma)':<20s} {'Unknown (mu+/-sigma)':<20s} {'Ratio unk/kn':<15s}")
    print(f"{'-'*70}")
    for stage, kd, ud in zip(['FPN output', 'Post-projector', 'Poincare'], known_data, unk_data):
        km, ks = (kd.mean(), kd.std()) if len(kd) > 0 else (0, 0)
        um, us = (ud.mean(), ud.std()) if len(ud) > 0 else (0, 0)
        ratio = um / km if km > 0 else float('inf')
        print(f"{stage:<25s} {km:.4f}+/-{ks:.4f}      {um:.4f}+/-{us:.4f}      {ratio:.4f}")
    pct_clipped_known = (pre_clip_norms[known_mask] >= 0.999).mean() * 100
    pct_clipped_unk = (pre_clip_norms[unk_mask] >= 0.999).mean() * 100 if unk_mask.any() else 0
    print(f"% clipped (norm>=clip_r):  Known={pct_clipped_known:.1f}%  Unknown={pct_clipped_unk:.1f}%")
    print(f"{'='*70}")

    # Print OOD detection metrics
    print(f"\n{'='*70}")
    print(f"OOD DETECTION METRICS (Geodesic Prototypical)")
    print(f"{'='*70}")
    print(f"  AUROC:              {auroc:.4f}")
    print(f"  Cohen's d:          {cohens_d:.4f}")
    print(f"  FPR@95%TPR:         {fpr95:.4f}")
    print(f"  Known score (mu+/-sigma):  {known_scores.mean():.2f} +/- {known_scores.std():.2f}")
    if len(unknown_scores) > 0:
        print(f"  Unk   score (mu+/-sigma):  {unknown_scores.mean():.2f} +/- {unknown_scores.std():.2f}")
    print(f"  Proto norms:        min={proto_norms.min():.4f} max={proto_norms.max():.4f}")
    print(f"{'='*70}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--task", default="IDD_HYP/t1")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--hyp_c", type=float, default=1.0)
    parser.add_argument("--hyp_dim", type=int, default=64)
    parser.add_argument("--clip_r", type=float, default=2.0)
    parser.add_argument("--output_dir", default="visualizations")
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)

    args = parser.parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)

    # Parse task
    task_parts = args.task.rstrip('/').split('/')
    task_name = task_parts[-2]
    split_name = task_parts[-1]
    base_dataset = task_name.replace('_HYP', '')
    dataset_key = base_dataset

    task_yaml = os.path.join("./configs", task_name, split_name + ".yaml")
    if os.path.exists(task_yaml) and 'base' in args.config_file:
        print(f"  Auto-merging task config: {task_yaml}")
        cfg.merge_from_file(task_yaml)
        cfg.freeze()

    data_split = f"{base_dataset}/{split_name}"
    data_register = Register('./datasets/', data_split, cfg, dataset_key)
    data_register.register_dataset()

    all_class_names = list(inital_prompts()[dataset_key])
    unknown_index = cfg.TEST.PREV_INTRODUCED_CLS + cfg.TEST.CUR_INTRODUCED_CLS
    known_class_names = all_class_names[:unknown_index]

    print(f"\n=== Configuration ===")
    print(f"  Task: {args.task}")
    print(f"  Known classes ({unknown_index}): {known_class_names}")
    print(f"  Checkpoint: {args.ckpt}")

    config_file = os.path.join("./configs", task_name, split_name + ".py")
    cfgY = Config.fromfile(config_file)
    cfgY.work_dir = "."

    runner = Runner.from_cfg(cfgY)
    runner._hooks = [h for h in runner._hooks if not h.__class__.__name__.startswith('EMA')]
    runner.call_hook("before_run")
    runner.load_or_resume()
    runner.model = runner.model.cuda()
    runner.model.reparameterize([known_class_names])
    runner.model.eval()

    test_loader = Runner.build_dataloader(cfgY.test_dataloader)

    # --- Auto-detect hyp config from checkpoint ---
    ckpt_data = torch.load(args.ckpt, map_location='cpu')
    hyp_config = ckpt_data.get('hyp_config', {})
    hyp_c = hyp_config.get('curvature', args.hyp_c)
    hyp_dim = hyp_config.get('embed_dim', args.hyp_dim)
    clip_r = hyp_config.get('clip_r', args.clip_r)
    bi_lipschitz = hyp_config.get('bi_lipschitz', False)
    prototype_init_norm = hyp_config.get('prototype_init_norm', 0.4)
    max_proto_norm = hyp_config.get('max_proto_norm', 0.5)
    adaptive_stats = ckpt_data.get('adaptive_stats', None)

    if hyp_dim != args.hyp_dim:
        print(f"  NOTE: Using hyp_dim={hyp_dim} from checkpoint (CLI was {args.hyp_dim})")
    if hyp_c != args.hyp_c:
        print(f"  NOTE: Using hyp_c={hyp_c} from checkpoint (CLI was {args.hyp_c})")
    if bi_lipschitz:
        print(f"  NOTE: Using BiLipschitz projectors")
    if adaptive_stats:
        print(f"  NOTE: Loaded adaptive_stats from checkpoint ({len(adaptive_stats.get('per_class', {}))} classes)")
    else:
        print(f"  WARNING: No adaptive_stats in checkpoint")

    del ckpt_data

    # For T2+ eval
    prev_cls = cfg.TEST.PREV_INTRODUCED_CLS
    cur_cls = cfg.TEST.CUR_INTRODUCED_CLS
    classifier_num_classes = cur_cls if prev_cls > 0 else unknown_index

    # Build geodesic prototypical model
    model = HypCustomYoloWorld(
        runner.model, unknown_index,
        hyp_c=hyp_c, hyp_dim=hyp_dim, clip_r=clip_r,
        num_classifier_classes=classifier_num_classes,
        bi_lipschitz=bi_lipschitz,
        prototype_init_norm=prototype_init_norm,
        max_proto_norm=max_proto_norm,
    )

    print(f"\n=== Loading Checkpoint: {args.ckpt} ===")
    with torch.no_grad():
        model = load_hyp_ckpt(model, args.ckpt, prev_cls, cur_cls, eval=True)
        model = model.cuda()
        known_class_names_orig = list(known_class_names)
        model.add_generic_text(known_class_names, generic_prompt='object', alpha=0.4)
    model.eval()

    prototypes = model.prototypes.detach()

    print(f"\n=== Model Info ===")
    proto_norms = prototypes.norm(dim=-1)
    print(f"  Prototypes: {prototypes.shape[0]} classes")
    print(f"  Proto norms: {[f'{n:.4f}' for n in proto_norms.tolist()]}")
    print(f"  Max proto norm: {proto_norms.max():.4f} (ceiling: {max_proto_norm})")
    print(f"  Known classes: {known_class_names}")

    # ---- Collect embeddings from FULL test set ----
    data = collect_embeddings(model, test_loader, known_class_names,
                              dataset_root='./datasets', hyp_c=hyp_c)

    if data is None:
        print("ERROR: No embeddings collected!")
        sys.exit(1)

    embeddings = data['embeddings']
    poincare_norms = embeddings.norm(dim=-1).numpy()
    print(f"\n  Total embeddings: {len(embeddings)}")
    print(f"  Embedding dim: {embeddings.shape[1]}")
    print(f"  Poincare norms: min={poincare_norms.min():.4f}, max={poincare_norms.max():.4f}")
    print(f"  Pre-clip norms: min={data['pre_clip_norms'].min():.4f}, max={data['pre_clip_norms'].max():.4f}")
    print(f"  FPN norms: min={data['fpn_norms'].min():.4f}, max={data['fpn_norms'].max():.4f}")
    print(f"  Geodesic scores: min={data['max_score'].min():.4f}, max={data['max_score'].max():.4f}")

    # ---- UMAP projection ----
    print(f"\n=== UMAP Projection ===")
    embeddings_2d, prototypes_2d = project_umap(
        embeddings, prototypes.cpu(),
        curvature=hyp_c,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )
    print(f"  2D norms: min={np.linalg.norm(embeddings_2d, axis=-1).min():.4f}, "
          f"max={np.linalg.norm(embeddings_2d, axis=-1).max():.4f}")

    # ---- Create versioned output directory ----
    save_dir = make_versioned_dir(args.output_dir)
    print(f"\n  Output directory: {save_dir}")

    # ---- Generate all diagnostic plots ----
    plot_all_diagnostics(
        data, prototypes_2d, embeddings_2d, prototypes.cpu(),
        known_class_names_orig, save_dir, curvature=hyp_c,
        adaptive_stats=adaptive_stats,
    )

    # ---- Save raw data ----
    ckpt_name = Path(args.ckpt).stem
    np.savez(os.path.join(save_dir, f"data_{ckpt_name}.npz"),
             embeddings_2d=embeddings_2d,
             prototypes_2d=prototypes_2d,
             class_names=data['class_names'],
             is_known=data['is_known'],
             max_score=data['max_score'],
             geo_scores=data['geo_scores'].numpy(),
             assigned_proto=data['assigned_proto'],
             known_class_names=np.array(known_class_names_orig),
             prototypes_hd=prototypes.cpu().numpy(),
             fpn_norms=data['fpn_norms'],
             pre_clip_norms=data['pre_clip_norms'])
    print(f"  Saved: {save_dir}/data_{ckpt_name}.npz")

    # ---- Write interpretation guide ----
    guide = """
================================================================================
VISUALIZATION GUIDE — Geodesic Prototypical Framework
================================================================================

01_umap_full_testset.png — Overview UMAP (all classes)
   LOOK FOR: Known clusters tight around their interior prototype star.
   Unknown (x markers) scattered, ideally not overlapping with knowns.

02_umap_voronoi.png — UMAP + Geodesic Voronoi decision boundaries
   LOOK FOR: Each coloured region = nearest prototype by geodesic distance.
   Ideal: known class dots fully inside their own colour region.
   Problem: if unknowns fall deep inside a known region.

03_score_distributions.png — Geodesic scores + adaptive thresholds
   KEY FIGURE FOR LOSS TUNING.
   Left: overall known vs unknown separation -> want large gap.
   Middle: per-class + threshold (dotted lines).
   Right: per-unknown-class.
   WHAT TO TUNE:
   - Overlap = weak separation -> increase beta_reg (push unknowns to lower norm)
   - All scores bunched -> decrease score_scale or increase clip_r
   - Some classes have wide std -> those prototypes are poorly learned

04_per_class_detail.png — Per-class zoomed view
   LOOK FOR: blue dots (own class) clustered near prototype star.
   Red x (unknowns) should be scattered/distant.

05_umap_known_only.png — Known classes alone + Voronoi
   LOOK FOR: Tight, well-separated clusters within their Voronoi cells.
   Diffuse clusters = CE weight too low or prototype poorly initialised.

06_umap_unknown_only.png — Unknown classes alone + Voronoi
   IDEAL: Unknowns far from all prototypes, near center or spread outside regions.
   PROBLEM: If unknowns cluster near/around prototypes = they'll be classified as known.

07_embedding_norms.png — Poincare ball norms
   IDEAL: Known at high norms (confident), unknown at LOWER norms.
   PROBLEM: Both at same norm = loss doesn't create norm separation.
   -> TUNE: increase beta_reg or add explicit norm-based regularisation.

08_geodesic_score_heatmap.png — Score affinity matrix
   KEY FIGURE: Each cell = mean score of class->prototype.
   Diagonal of known block should be darkest green (highest score = closest).
   Unknown rows should be uniformly red (far from all protos).
   If an unknown has high score to some proto -> that proto is absorbing it.

09_norm_pipeline.png — FPN -> Projector -> Poincare norms
   Shows where known/unknown norm separation exists or collapses.
   If separation exists at FPN but not at Poincare -> projector is collapsing signals.

10_per_class_projector_norms.png — Pre-clip norms per class
   Checks whether all classes are being clipped equally.
   If every class is at clip_r ceiling -> all info is lost -> reduce clip_r.

11_prototype_analysis.png — Inter-prototype distances + norms
   Left: pairwise distances. Should ALL be large (well separated).
   Small distances = prototypes too close -> increase lambda_sep or sep_margin.
   Right: prototype norms. Should all be <= max_proto_norm.
   If at ceiling -> optimizer is still pushing them toward boundary.

12_statistical_summary.png — AUROC, Cohen's d, per-class calibration
   TOP-LEFT: Key OOD metrics.
     AUROC > 0.90 = good separation. < 0.70 = poor.
     Cohen's d > 1.0 = large effect. < 0.5 = weak.
     FPR@95: fraction of unknowns falsely classified as known at 95% recall.
   TOP-RIGHT: Per-class calibration bars (from training set).
   BOTTOM-LEFT: Prototype utilization -- imbalance means loss is biased.
   BOTTOM-RIGHT: Where unknowns are attracted to -- guides loss weight tuning.

LOSS TUNING DECISION TREE:
  Low AUROC + overlapping score distributions:
    -> increase beta_reg (norm regularisation)
    -> increase lambda_sep (push prototypes apart)
  Low AUROC + unknowns at high norm:
    -> beta_reg too low, L_reg not effective
    -> consider adding explicit unknown norm penalty
  Good AUROC but bad per-class thresholds:
    -> change calibration alpha (currently alpha=0.75)
  Prototypes at max_proto_norm ceiling:
    -> working as designed, but consider lowering max_proto_norm
  Wide inter-class score variance:
    -> class_balance_smoothing may need adjustment
================================================================================
"""
    with open(os.path.join(save_dir, 'INTERPRETATION_GUIDE.txt'), 'w') as f:
        f.write(guide)
    print(f"  Saved: INTERPRETATION_GUIDE.txt")

    print(f"\n{'='*60}")
    print(f"VISUALIZATION COMPLETE — {save_dir}")
    print(f"{'='*60}")
