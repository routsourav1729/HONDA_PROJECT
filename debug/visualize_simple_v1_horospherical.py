"""
Poincaré Ball Visualization for Horospherical Classifiers — FULL TEST SET

Visualizes ALL test-set hyperbolic embeddings (known + unknown) via UMAP
projection to the Poincaré disk, with horosphere decision boundaries drawn.

Key design choices:
- Uses XML parsing (not dataloader GT) to get BOTH known and unknown boxes
- Iterates the ENTIRE test dataloader — no subsampling
- Projects high-dim Poincaré embeddings → 2D Poincaré disk via UMAP
- Draws per-class horospheres as circles on the 2D disk
"""

import os
import sys
import xml.etree.ElementTree as ET
import torch
import numpy as np
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
from core.hyperbolic import pmath

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


# ============================================================================
# XML parsing — gets ALL boxes (known + unknown)
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
    Iterate the ENTIRE test dataloader and collect hyperbolic embeddings for
    ALL GT boxes (known + unknown) via XML annotation parsing.
    """
    total_images = len(data_loader.dataset) if hasattr(data_loader, 'dataset') else '?'
    print(f"\n{'='*60}")
    print(f"COLLECTING EMBEDDINGS — FULL TEST SET (XML parsing)")
    print(f"  Known classes: {known_class_names}")
    print(f"  Total images: {total_images}")
    print(f"{'='*60}")

    model.eval()
    known_set = set(known_class_names)

    all_embeddings = []
    all_class_names = []
    all_is_known = []
    all_max_horo = []
    all_assigned_proto = []
    all_fpn_norms = []        # FPN feature norms (pre-projector)
    all_pre_clip_norms = []   # Projector output norms (pre-ToPoincare clip)
    samples_count = defaultdict(int)

    prototypes = model.prototypes.detach()

    with torch.no_grad():
        # Enable norm caching for pipeline analysis
        model.hyp_projector.store_norms = True
        
        for i, batch in enumerate(tqdm(data_loader, desc="Collecting (full test set)")):
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

                hyp_embeddings, _ = model.hyp_projector(x)  # (B, 8400, dim), discard pre_expmap_norms

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

                        emb = hyp_embeddings[b_idx, nearest_idx]

                        # Capture pipeline norms at same spatial location
                        fpn_norm_val = model.hyp_projector._cached_fpn_norms[b_idx, nearest_idx].item()
                        pre_clip_norm_val = model.hyp_projector._cached_pre_clip_norms[b_idx, nearest_idx].item()

                        # Geodesic scoring: s_k = -d^2(x, z_k)
                        emb_exp = emb.unsqueeze(0).expand(prototypes.shape[0], -1)
                        dists = pmath.dist(emb_exp, prototypes, c=hyp_c)
                        geo_scores = -dists.pow(2)
                        max_horo, assigned = geo_scores.max(dim=0)

                        all_embeddings.append(emb.cpu())
                        all_class_names.append(cls_name)
                        all_is_known.append(is_known)
                        all_max_horo.append(max_horo.item())
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
        return None, None, None, None, None, None, None

    embeddings = torch.stack(all_embeddings)
    return (embeddings,
            np.array(all_class_names),
            np.array(all_is_known),
            np.array(all_max_horo),
            np.array(all_assigned_proto),
            np.array(all_fpn_norms),
            np.array(all_pre_clip_norms))



# ============================================================================
# UMAP projection — high-dim Poincaré → 2D Poincaré disk
# ============================================================================

def project_umap(embeddings, prototypes, curvature=1.0, n_neighbors=15, min_dist=0.1):
    """
    Project high-dim Poincaré embeddings + prototypes to 2D Poincaré disk
    via UMAP with hyperboloid output metric.
    """
    import umap

    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().numpy()
    if torch.is_tensor(prototypes):
        prototypes = prototypes.detach().cpu().numpy()

    # Scale boundary prototypes slightly inside for logmap0
    proto_norms = np.linalg.norm(prototypes, axis=-1, keepdims=True)
    prototypes_scaled = prototypes * (0.9 / (proto_norms + 1e-8))

    emb_tangent = _logmap0_numpy(embeddings, curvature)
    proto_tangent = _logmap0_numpy(prototypes_scaled, curvature)

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

    # Push prototypes back to boundary in 2D
    p2d_norms = np.linalg.norm(prototypes_2d, axis=-1, keepdims=True)
    prototypes_2d = prototypes_2d * (0.99 / (p2d_norms + 1e-8))

    return embeddings_2d, prototypes_2d


# ============================================================================
# Horosphere drawing on 2D Poincaré disk
# ============================================================================

def draw_geodesic_ball_2d(ax, prototype_2d, radius_2d=0.15, color='gray',
                          label=None, linestyle='-', alpha=0.25):
    """
    Draw a geodesic ball around a prototype on the 2D UMAP Poincaré disk.

    Since UMAP is a non-linear projection, we approximate the geodesic ball
    as a Euclidean circle of fixed radius centred on the prototype's 2D position.
    """
    theta = np.linspace(0, 2 * np.pi, 200)
    circle_x = prototype_2d[0] + radius_2d * np.cos(theta)
    circle_y = prototype_2d[1] + radius_2d * np.sin(theta)

    # Clip to inside the unit disk
    inside = (circle_x**2 + circle_y**2) <= 1.01
    circle_x[~inside] = np.nan
    circle_y[~inside] = np.nan

    ax.plot(circle_x, circle_y, color=color, linestyle=linestyle,
            linewidth=2.0, alpha=0.8, label=label)
    ax.fill(circle_x, circle_y, color=color, alpha=alpha)


def _draw_disk(ax):
    """Draw the unit Poincaré disk boundary."""
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
    ax.fill(np.cos(theta), np.sin(theta), alpha=0.03, color='gray')


# ============================================================================
# Plotting
# ============================================================================

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


def plot_full_visualization(embeddings_2d, class_names_arr, is_known_arr,
                            max_horo_arr, prototypes_2d, known_class_names,
                            save_dir, curvature=1.0,
                            embeddings_hd=None, prototypes_hd=None, hyp_c=1.0,
                            fpn_norms=None, pre_clip_norms=None):
    """
    Create 9 figures:
      1. UMAP Poincaré disk — all classes colour-coded (combined)
      2. Same disk + horosphere circles
      3. Score distributions (known vs unknown)
      4. Per-class horosphere detail (3x3 grid)
      5. Known-only UMAP (see cluster tightness)
      6. Unknown-only UMAP (see where unknowns project)
      7. Embedding norms (fixed: zoomed, per-class)
      8. Per-class score heatmap (unknown → known prototype affinity)
      9. Norm pipeline: FPN → projector → Poincaré (known vs unknown)
    """
    known_set = set(known_class_names)
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

    # Geodesic model: no biases. Prototype norms used for annotation.
    proto_norms_np = np.linalg.norm(prototypes_2d, axis=-1)

    # ---- Figure 1: Colour-coded UMAP (no horospheres) ----
    fig1, ax1 = plt.subplots(1, 1, figsize=(14, 14))
    _draw_disk(ax1)

    for cls in unique_known:
        mask = class_names_arr == cls
        n = mask.sum()
        ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=[color_map[cls]], s=12, alpha=0.5, label=f'{cls} ({n})')

    for cls in unique_unknown:
        mask = class_names_arr == cls
        n = mask.sum()
        ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=[color_map[cls]], s=12, alpha=0.5, marker='x',
                    label=f'*{cls} ({n})')

    for i in range(n_protos):
        ax1.scatter(prototypes_2d[i, 0], prototypes_2d[i, 1],
                    c=[color_map.get(known_class_names[i], 'gray')],
                    s=400, marker='*', edgecolors='black', linewidths=1.5, zorder=5)
        direction = prototypes_2d[i] / (np.linalg.norm(prototypes_2d[i]) + 1e-8)
        label_pos = direction * 1.12
        ax1.annotate(known_class_names[i], label_pos, fontsize=8, ha='center',
                     va='center', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    ax1.set_xlim(-1.25, 1.25); ax1.set_ylim(-1.25, 1.25)
    ax1.set_aspect('equal')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7, markerscale=2)
    ax1.set_title('Poincare Disk -- Full Test Set UMAP (star = ideal prototypes)\n'
                  '(dot = known,  x = unknown)', fontsize=13)
    ax1.grid(True, alpha=0.2)
    plt.tight_layout()
    path1 = os.path.join(save_dir, 'umap_full_testset.png')
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"  Saved: {path1}")

    # ---- Figure 2: UMAP + Horospheres ----
    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 14))
    _draw_disk(ax2)

    for i in range(n_protos):
        c = color_map.get(known_class_names[i], 'gray')
        draw_geodesic_ball_2d(
            ax2, prototypes_2d[i], radius_2d=0.15, color=c,
            label=f'Proto {known_class_names[i]} (||z||={proto_norms_np[i]:.3f})',
            alpha=0.10,
        )

    for cls in unique_known:
        mask = class_names_arr == cls
        ax2.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=[color_map[cls]], s=10, alpha=0.4)
    for cls in unique_unknown:
        mask = class_names_arr == cls
        ax2.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=[color_map[cls]], s=10, alpha=0.4, marker='x')

    for i in range(n_protos):
        ax2.scatter(prototypes_2d[i, 0], prototypes_2d[i, 1],
                    c=[color_map.get(known_class_names[i], 'gray')],
                    s=400, marker='*', edgecolors='black', linewidths=1.5, zorder=5)
        direction = prototypes_2d[i] / (np.linalg.norm(prototypes_2d[i]) + 1e-8)
        label_pos = direction * 1.12
        ax2.annotate(known_class_names[i], label_pos, fontsize=8, ha='center',
                     va='center', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    ax2.set_xlim(-1.25, 1.25); ax2.set_ylim(-1.25, 1.25)
    ax2.set_aspect('equal')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7)
    ax2.set_title('Poincare Disk + Geodesic Prototype Regions\n'
                  'Points NEAR a prototype -> classified as that class', fontsize=13)
    ax2.grid(True, alpha=0.2)
    plt.tight_layout()
    path2 = os.path.join(save_dir, 'umap_with_horospheres.png')
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: {path2}")

    # ---- Figure 3: Score distributions ----
    fig3, axes3 = plt.subplots(1, 2, figsize=(20, 8))

    ax = axes3[0]
    known_scores = max_horo_arr[is_known_arr]
    unknown_scores = max_horo_arr[~is_known_arr]
    lo = min(known_scores.min(), unknown_scores.min()) - 0.2 if len(unknown_scores) > 0 else known_scores.min() - 0.2
    hi = max(known_scores.max(), unknown_scores.max()) + 0.2 if len(unknown_scores) > 0 else known_scores.max() + 0.2
    bins = np.linspace(lo, hi, 60)
    ax.hist(known_scores, bins=bins, alpha=0.6, color='blue',
            label=f'Known (n={len(known_scores)})', density=True)
    if len(unknown_scores) > 0:
        ax.hist(unknown_scores, bins=bins, alpha=0.6, color='red',
                label=f'Unknown (n={len(unknown_scores)})', density=True)
    ax.axvline(0, color='black', linestyle=':', linewidth=1.5, label='tau=0')
    ax.set_xlabel('Max Geodesic Score (-d²)')
    ax.set_ylabel('Density')
    ax.set_title('Known vs Unknown: Max Geodesic Score')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes3[1]
    for cls in unique_unknown:
        mask = class_names_arr == cls
        if mask.sum() > 0:
            ax.hist(max_horo_arr[mask], bins=30, alpha=0.5,
                    label=f'{cls} ({mask.sum()})', density=True)
    ax.axvline(0, color='black', linestyle=':', linewidth=1.5, label='tau=0')
    ax.set_xlabel('Max Geodesic Score (-d²)')
    ax.set_ylabel('Density')
    ax.set_title('Per Unknown Subclass: Score Distributions')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path3 = os.path.join(save_dir, 'score_distributions.png')
    fig3.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"  Saved: {path3}")

    # ---- Figure 4: Per-known-class horosphere detail (3x3) ----
    n_grid = min(n_protos, 9)
    ncols = 3
    nrows = (n_grid + ncols - 1) // ncols
    fig4, axes4 = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    if nrows == 1:
        axes4 = axes4[np.newaxis, :]

    for idx in range(n_grid):
        r, c = idx // ncols, idx % ncols
        ax = axes4[r][c]
        cls = known_class_names[idx]
        _draw_disk(ax)

        draw_geodesic_ball_2d(ax, prototypes_2d[idx], radius_2d=0.15,
                              color=color_map.get(cls, 'blue'), alpha=0.08)

        mask_k = (class_names_arr == cls)
        if mask_k.sum() > 0:
            ax.scatter(embeddings_2d[mask_k, 0], embeddings_2d[mask_k, 1],
                       c='blue', s=8, alpha=0.4, label=f'{cls} ({mask_k.sum()})')

        mask_u = ~is_known_arr
        if mask_u.sum() > 0:
            ax.scatter(embeddings_2d[mask_u, 0], embeddings_2d[mask_u, 1],
                       c='red', s=6, alpha=0.15, marker='x', label=f'Unknowns ({mask_u.sum()})')

        ax.scatter(prototypes_2d[idx, 0], prototypes_2d[idx, 1],
                   c='gold', s=300, marker='*', edgecolors='black', linewidths=1.5, zorder=5)

        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_title(f'{cls} (||z||={proto_norms_np[idx]:.3f})', fontsize=10)
        ax.legend(fontsize=6, loc='lower right')

    for idx in range(n_grid, nrows * ncols):
        r, c = idx // ncols, idx % ncols
        axes4[r][c].set_visible(False)

    plt.suptitle('Per-Class Geodesic Ball: Known (blue) vs Unknown (red)', fontsize=14, y=1.01)
    plt.tight_layout()
    path4 = os.path.join(save_dir, 'per_class_horosphere.png')
    fig4.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print(f"  Saved: {path4}")

    # ---- Figure 5: Known-only UMAP ----
    fig5, ax5 = plt.subplots(1, 1, figsize=(14, 14))
    _draw_disk(ax5)

    known_mask = is_known_arr
    for cls in unique_known:
        mask = class_names_arr == cls
        n = mask.sum()
        ax5.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=[color_map[cls]], s=14, alpha=0.5, label=f'{cls} ({n})')

    for i in range(n_protos):
        c = color_map.get(known_class_names[i], 'gray')
        draw_geodesic_ball_2d(ax5, prototypes_2d[i], radius_2d=0.15, color=c, alpha=0.06)
        ax5.scatter(prototypes_2d[i, 0], prototypes_2d[i, 1],
                    c=[c], s=400, marker='*', edgecolors='black', linewidths=1.5, zorder=5)
        direction = prototypes_2d[i] / (np.linalg.norm(prototypes_2d[i]) + 1e-8)
        label_pos = direction * 1.12
        ax5.annotate(known_class_names[i], label_pos, fontsize=8, ha='center',
                     va='center', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    ax5.set_xlim(-1.25, 1.25); ax5.set_ylim(-1.25, 1.25)
    ax5.set_aspect('equal')
    ax5.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7, markerscale=2)
    ax5.set_title('KNOWN CLASSES ONLY — Poincare Disk UMAP\n'
                  '(Check: are clusters tight around their prototype?)', fontsize=13)
    ax5.grid(True, alpha=0.2)
    plt.tight_layout()
    path5 = os.path.join(save_dir, 'umap_known_only.png')
    fig5.savefig(path5, dpi=150, bbox_inches='tight')
    plt.close(fig5)
    print(f"  Saved: {path5}")

    # ---- Figure 6: Unknown-only UMAP ----
    fig6, ax6 = plt.subplots(1, 1, figsize=(14, 14))
    _draw_disk(ax6)

    for cls in unique_unknown:
        mask = class_names_arr == cls
        n = mask.sum()
        ax6.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=[color_map[cls]], s=20, alpha=0.6, marker='x',
                    label=f'{cls} ({n})')

    # Show prototypes and horospheres as reference
    for i in range(n_protos):
        c = color_map.get(known_class_names[i], 'gray')
        draw_geodesic_ball_2d(ax6, prototypes_2d[i], radius_2d=0.15, color=c, alpha=0.04)
        ax6.scatter(prototypes_2d[i, 0], prototypes_2d[i, 1],
                    c=[c], s=400, marker='*', edgecolors='black', linewidths=1.5, zorder=5)
        direction = prototypes_2d[i] / (np.linalg.norm(prototypes_2d[i]) + 1e-8)
        label_pos = direction * 1.12
        ax6.annotate(known_class_names[i], label_pos, fontsize=8, ha='center',
                     va='center', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    ax6.set_xlim(-1.25, 1.25); ax6.set_ylim(-1.25, 1.25)
    ax6.set_aspect('equal')
    ax6.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7, markerscale=2)
    ax6.set_title('UNKNOWN CLASSES ONLY — Poincare Disk UMAP\n'
                  '(Ideal: unknowns near center/origin, FAR from prototypes)', fontsize=13)
    ax6.grid(True, alpha=0.2)
    plt.tight_layout()
    path6 = os.path.join(save_dir, 'umap_unknown_only.png')
    fig6.savefig(path6, dpi=150, bbox_inches='tight')
    plt.close(fig6)
    print(f"  Saved: {path6}")

    # ---- Figure 7: Embedding norms (zoomed + box plot) ----
    fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(18, 6))

    known_norms_2d = np.linalg.norm(embeddings_2d[known_mask], axis=-1)
    unknown_norms_2d = np.linalg.norm(embeddings_2d[~known_mask], axis=-1) if (~known_mask).any() else np.array([])

    # Use high-dim norms if available, otherwise 2D norms
    if embeddings_hd is not None:
        embhd = embeddings_hd.numpy() if torch.is_tensor(embeddings_hd) else embeddings_hd
        all_norms = np.linalg.norm(embhd, axis=-1)
        known_norms_hd = all_norms[known_mask]
        unknown_norms_hd = all_norms[~known_mask] if (~known_mask).any() else np.array([])
    else:
        known_norms_hd = known_norms_2d
        unknown_norms_hd = unknown_norms_2d

    # Left: Zoomed histogram with proper range
    all_n = np.concatenate([known_norms_hd, unknown_norms_hd]) if len(unknown_norms_hd) > 0 else known_norms_hd
    lo, hi = all_n.min() - 0.01, all_n.max() + 0.01
    bins = np.linspace(lo, hi, 80)
    ax7a.hist(known_norms_hd, bins=bins, alpha=0.5, color='blue',
              label=f'Known (n={len(known_norms_hd)})', density=True)
    if len(unknown_norms_hd) > 0:
        ax7a.hist(unknown_norms_hd, bins=bins, alpha=0.5, color='red',
                  label=f'Unknown (n={len(unknown_norms_hd)})', density=True)
    ax7a.axvline(0.95, color='green', linestyle='--', label='clip_r=0.95')
    ax7a.set_xlabel('||x|| (Poincare ball norm)')
    ax7a.set_ylabel('Density')
    ax7a.set_title('Embedding Norm Distribution (zoomed to data range)')
    ax7a.legend(fontsize=8)
    ax7a.grid(True, alpha=0.3)

    # Right: Per-class boxplot of norms
    class_norm_data = []
    class_norm_labels = []
    class_norm_colors = []
    for cls_name in known_class_names:
        m = class_names_arr == cls_name
        if m.sum() > 0:
            norms_cls = np.linalg.norm(embhd[m], axis=-1) if embeddings_hd is not None else np.linalg.norm(embeddings_2d[m], axis=-1)
            class_norm_data.append(norms_cls)
            class_norm_labels.append(f'{cls_name}\n(n={m.sum()})')
            class_norm_colors.append('steelblue')
    for cls_name in unique_unknown[:6]:  # top 6 unknown classes
        m = class_names_arr == cls_name
        if m.sum() > 0:
            norms_cls = np.linalg.norm(embhd[m], axis=-1) if embeddings_hd is not None else np.linalg.norm(embeddings_2d[m], axis=-1)
            class_norm_data.append(norms_cls)
            class_norm_labels.append(f'*{cls_name}\n(n={m.sum()})')
            class_norm_colors.append('coral')
    if class_norm_data:
        bp = ax7b.boxplot(class_norm_data, labels=class_norm_labels, patch_artist=True, vert=True)
        for patch, col in zip(bp['boxes'], class_norm_colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.5)
        ax7b.tick_params(axis='x', rotation=45, labelsize=7)
        ax7b.set_ylabel('||x|| (Poincare ball norm)')
        ax7b.set_title('Per-Class Embedding Norms\n(blue=known, red=unknown)')
        ax7b.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path7 = os.path.join(save_dir, 'embedding_norms.png')
    fig7.savefig(path7, dpi=150, bbox_inches='tight')
    plt.close(fig7)
    print(f"  Saved: {path7}")

    # ---- Figure 8: Score heatmap — unknown mean horo score per known prototype ----
    if len(unique_unknown) > 0:
        # Also add known classes for reference comparison
        all_classes_for_heatmap = list(unique_known) + list(unique_unknown)
        heatmap_data = []
        heatmap_labels = []
        heatmap_counts = []
        for cls_name in all_classes_for_heatmap:
            mask = class_names_arr == cls_name
            if mask.sum() < 3:
                continue
            heatmap_labels.append(cls_name)
            heatmap_counts.append(int(mask.sum()))
            # For each sample of this class, we only have max_horo + assigned_proto
            # We can show: what fraction gets assigned to each prototype
            assign_fracs = np.zeros(n_protos)
            for cls_idx in range(n_protos):
                # This requires assigned_proto_arr — check if accessible
                pass
            heatmap_data.append(max_horo_arr[mask].mean())  # placeholder

        # Better approach: show the norm-distance from disk center as proxy
        # Actually, the most useful is a radial distance from each prototype in 2D
        proto_dist_matrix = np.zeros((len(all_classes_for_heatmap), n_protos))
        valid_rows = []
        valid_labels = []
        for ri, cls_name in enumerate(all_classes_for_heatmap):
            mask = class_names_arr == cls_name
            if mask.sum() < 3:
                continue
            valid_rows.append(ri)
            is_k = cls_name in known_set
            valid_labels.append(f"{'[K]' if is_k else '[U]'} {cls_name} (n={mask.sum()})")
            emb_cls = embeddings_2d[mask]  # (n_cls, 2)
            for pi in range(n_protos):
                # Mean euclidean distance in 2D disk from this class to prototype pi
                dists = np.linalg.norm(emb_cls - prototypes_2d[pi], axis=-1)
                proto_dist_matrix[ri, pi] = dists.mean()

        if len(valid_rows) > 0:
            mat = proto_dist_matrix[valid_rows]
            fig8, ax8 = plt.subplots(figsize=(12, max(5, len(valid_labels) * 0.4)))
            im = ax8.imshow(mat, cmap='RdYlGn_r', aspect='auto')  # green=close, red=far
            ax8.set_xticks(range(n_protos))
            ax8.set_xticklabels(known_class_names, rotation=45, ha='right', fontsize=9)
            ax8.set_yticks(range(len(valid_labels)))
            ax8.set_yticklabels(valid_labels, fontsize=7)
            ax8.set_xlabel('Known Prototype')
            ax8.set_ylabel('Class ([K]=known, [U]=unknown)')
            ax8.set_title('Mean 2D Distance to Each Prototype\n'
                          '(green=close, red=far — known classes should be green for own proto)')
            for i in range(len(valid_labels)):
                for j in range(n_protos):
                    ax8.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center', fontsize=6)
            plt.colorbar(im, ax=ax8, label='Mean Euclidean dist in 2D disk', shrink=0.8)
            plt.tight_layout()
            path8 = os.path.join(save_dir, 'prototype_distance_heatmap.png')
            fig8.savefig(path8, dpi=150, bbox_inches='tight')
            plt.close(fig8)
            print(f"  Saved: {path8}")

    # ---- Figure 9: Norm Pipeline Analysis (FPN → Projector → Poincaré) ----
    if fpn_norms is not None and pre_clip_norms is not None:
        poincare_norms = np.array([np.linalg.norm(e) for e in
            (embeddings_hd.numpy() if torch.is_tensor(embeddings_hd) else embeddings_hd)])
        
        known_mask = is_known_arr.astype(bool)
        unk_mask = ~known_mask
        
        stages = ['FPN Output', 'Post-Projector\n(pre-clip)', 'Poincaré\n(post-expmap0)']
        known_data = [fpn_norms[known_mask], pre_clip_norms[known_mask], poincare_norms[known_mask]]
        unk_data =   [fpn_norms[unk_mask],   pre_clip_norms[unk_mask],   poincare_norms[unk_mask]]
        
        fig9, axes9 = plt.subplots(1, 3, figsize=(20, 6))
        fig9.suptitle('Norm Pipeline: Where Does Information Collapse?', fontsize=16, fontweight='bold')
        
        for idx, (ax, stage, kd, ud) in enumerate(zip(axes9, stages, known_data, unk_data)):
            # Box plots side by side
            data = [kd, ud]
            bp = ax.boxplot(data, labels=['Known', 'Unknown'], widths=0.5,
                           patch_artist=True, showfliers=False)
            bp['boxes'][0].set_facecolor('#4CAF50')
            bp['boxes'][0].set_alpha(0.7)
            bp['boxes'][1].set_facecolor('#F44336')
            bp['boxes'][1].set_alpha(0.7)
            
            ax.set_title(stage, fontsize=13, fontweight='bold')
            ax.set_ylabel('||x||' if idx == 0 else '')
            
            # Add mean markers
            k_mean = kd.mean() if len(kd) > 0 else 0
            u_mean = ud.mean() if len(ud) > 0 else 0
            ax.scatter([1], [k_mean], marker='D', color='darkgreen', s=60, zorder=5, label=f'μ={k_mean:.3f}')
            ax.scatter([2], [u_mean], marker='D', color='darkred', s=60, zorder=5, label=f'μ={u_mean:.3f}')
            ax.legend(fontsize=9)
            
            # Annotate separation
            if k_mean > 0 and u_mean > 0:
                ratio = u_mean / k_mean
                ax.text(0.5, 0.02, f'unk/known ratio: {ratio:.3f}',
                       transform=ax.transAxes, ha='center', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        path9 = os.path.join(save_dir, 'norm_pipeline.png')
        fig9.savefig(path9, dpi=150, bbox_inches='tight')
        plt.close(fig9)
        print(f"  Saved: {path9}")
        
        # ---- Figure 10: Per-class norm breakdown (pre-clip) ----
        fig10, (ax10a, ax10b) = plt.subplots(2, 1, figsize=(16, 10))
        fig10.suptitle('Per-Class Projector Output Norms (Pre-Clip Euclidean)', fontsize=14, fontweight='bold')
        
        # Top: known classes
        known_classes = sorted(set(class_names_arr[known_mask]))
        known_vals = [pre_clip_norms[class_names_arr == c] for c in known_classes]
        if known_vals:
            bp_k = ax10a.boxplot(known_vals, labels=known_classes, patch_artist=True, showfliers=False)
            for box in bp_k['boxes']:
                box.set_facecolor('#4CAF50')
                box.set_alpha(0.6)
            ax10a.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label=f'clip_r = 1.0')
            ax10a.set_title('Known Classes', fontsize=12)
            ax10a.set_ylabel('Pre-clip Euclidean norm')
            ax10a.tick_params(axis='x', rotation=30)
            ax10a.legend()
        
        # Bottom: unknown classes
        unk_classes = sorted(set(class_names_arr[unk_mask]))
        unk_vals = [pre_clip_norms[class_names_arr == c] for c in unk_classes]
        if unk_vals:
            bp_u = ax10b.boxplot(unk_vals, labels=unk_classes, patch_artist=True, showfliers=False)
            for box in bp_u['boxes']:
                box.set_facecolor('#F44336')
                box.set_alpha(0.6)
            ax10b.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label=f'clip_r = 1.0')
            ax10b.set_title('Unknown Classes (*)', fontsize=12)
            ax10b.set_ylabel('Pre-clip Euclidean norm')
            ax10b.tick_params(axis='x', rotation=30)
            ax10b.legend()
        
        plt.tight_layout()
        path10 = os.path.join(save_dir, 'per_class_projector_norms.png')
        fig10.savefig(path10, dpi=150, bbox_inches='tight')
        plt.close(fig10)
        print(f"  Saved: {path10}")
        
        # ---- Print summary table to stdout ----
        print(f"\n{'='*70}")
        print(f"NORM PIPELINE SUMMARY")
        print(f"{'='*70}")
        print(f"{'Stage':<25s} {'Known (μ±σ)':<20s} {'Unknown (μ±σ)':<20s} {'Ratio unk/kn':<15s}")
        print(f"{'-'*70}")
        for stage, kd, ud in zip(['FPN output', 'Post-projector', 'Poincaré'], known_data, unk_data):
            km, ks = (kd.mean(), kd.std()) if len(kd) > 0 else (0, 0)
            um, us = (ud.mean(), ud.std()) if len(ud) > 0 else (0, 0)
            ratio = um / km if km > 0 else float('inf')
            print(f"{stage:<25s} {km:.4f}±{ks:.4f}      {um:.4f}±{us:.4f}      {ratio:.4f}")
        print(f"{'-'*70}")
        clip_r_val = getattr(embeddings_hd, '_clip_r', 1.0) if embeddings_hd is not None else 1.0
        pct_clipped_known = (pre_clip_norms[known_mask] >= 0.999).mean() * 100
        pct_clipped_unk = (pre_clip_norms[unk_mask] >= 0.999).mean() * 100 if unk_mask.any() else 0
        print(f"% clipped (norm≥clip_r):  Known={pct_clipped_known:.1f}%  Unknown={pct_clipped_unk:.1f}%")
        print(f"{'='*70}")
    else:
        print("  [SKIP] Norm pipeline analysis — no FPN/pre-clip data available")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--task", default="IDD_HYP/t1")
    parser.add_argument("--ckpt", default="IDD_HYP/t1/horospherical/model_5.pth")
    parser.add_argument("--hyp_c", type=float, default=1.0)
    parser.add_argument("--hyp_dim", type=int, default=64)
    parser.add_argument("--clip_r", type=float, default=0.95)
    parser.add_argument("--output_dir", default="visualizations")
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--num_batches", type=int, default=100,
                        help="Ignored (full test set used), kept for CLI compat")
    parser.add_argument("--samples_per_class", type=int, default=50,
                        help="Ignored (full test set used), kept for CLI compat")
    parser.add_argument("--projection", type=str, default="hyperbolic_umap",
                        help="Ignored (always hyperbolic UMAP), kept for CLI compat")

    args = parser.parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)

    # Normalise --task: accept both absolute paths and relative like "IDD_HYP/t1"
    task_parts = args.task.rstrip('/').split('/')
    task_name = task_parts[-2]   # e.g. "IDD_HYP"
    split_name = task_parts[-1]  # e.g. "t1"
    base_dataset = task_name.replace('_HYP', '')
    dataset_key = base_dataset

    # Auto-merge task-specific yaml if base.yaml was used (so class counts are correct)
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
    print(f"  Curvature: {args.hyp_c}")
    print(f"  Checkpoint: {args.ckpt}")

    # Model config
    config_file = os.path.join("./configs", task_name, split_name + ".py")
    cfgY = Config.fromfile(config_file)
    cfgY.work_dir = "."

    # Initialize YOLO-World
    runner = Runner.from_cfg(cfgY)
    # Strip EMA hook — it deep-copies the entire XL model (~20 min!) and is unused
    runner._hooks = [h for h in runner._hooks if not h.__class__.__name__.startswith('EMA')]
    runner.call_hook("before_run")
    runner.load_or_resume()
    runner.model = runner.model.cuda()
    runner.model.reparameterize([known_class_names])
    runner.model.eval()

    # Build TEST dataloader (not train!)
    test_loader = Runner.build_dataloader(cfgY.test_dataloader)

    # --- Auto-detect hyp config from checkpoint ---
    ckpt_data = torch.load(args.ckpt, map_location='cpu')
    hyp_config = ckpt_data.get('hyp_config', {})
    hyp_c = hyp_config.get('curvature', args.hyp_c)
    hyp_dim = hyp_config.get('embed_dim', args.hyp_dim)
    clip_r = hyp_config.get('clip_r', args.clip_r)
    bi_lipschitz = hyp_config.get('bi_lipschitz', False)
    if hyp_dim != args.hyp_dim:
        print(f"  NOTE: Using hyp_dim={hyp_dim} from checkpoint (CLI default was {args.hyp_dim})")
    if hyp_c != args.hyp_c:
        print(f"  NOTE: Using hyp_c={hyp_c} from checkpoint (CLI default was {args.hyp_c})")
    if bi_lipschitz:
        print(f"  NOTE: Using BiLipschitz projectors")
    del ckpt_data  # free memory

    # Build hyperbolic model
    model = HypCustomYoloWorld(
        runner.model, unknown_index,
        hyp_c=hyp_c, hyp_dim=hyp_dim, clip_r=clip_r,
        bi_lipschitz=bi_lipschitz,
    )

    print(f"\n=== Loading Checkpoint: {args.ckpt} ===")
    with torch.no_grad():
        model = load_hyp_ckpt(model, args.ckpt,
                              cfg.TEST.PREV_INTRODUCED_CLS, cfg.TEST.CUR_INTRODUCED_CLS, eval=True)
        model = model.cuda()
        # IMPORTANT: add_generic_text MUTATES known_class_names in-place (appends 'object')
        # Copy the original known class names BEFORE mutation for visualization
        known_class_names_orig = list(known_class_names)
        model.add_generic_text(known_class_names, generic_prompt='object', alpha=0.4)
    model.eval()

    prototypes = model.prototypes.detach()

    print(f"\n=== Model Info ===")
    print(f"  Prototypes: {prototypes.shape[0]} (norms: {[f'{n:.4f}' for n in prototypes.norm(dim=-1).tolist()]})")
    print(f"  Classes: {known_class_names}")

    # ---- Collect embeddings from FULL test set ----
    result = collect_embeddings(
        model, test_loader, known_class_names,
        dataset_root='./datasets',
        hyp_c=args.hyp_c,
    )

    if result[0] is None:
        print("ERROR: No embeddings collected!")
        sys.exit(1)

    embeddings, class_names_arr, is_known_arr, max_horo_arr, assigned_proto_arr, fpn_norms_arr, pre_clip_norms_arr = result

    poincare_norms = embeddings.norm(dim=-1).numpy()
    print(f"\n  Total embeddings: {len(embeddings)}")
    print(f"  Embedding dim: {embeddings.shape[1]}")
    print(f"  Poincaré norms: min={poincare_norms.min():.4f}, max={poincare_norms.max():.4f}")
    print(f"  Pre-clip Euclidean norms: min={pre_clip_norms_arr.min():.4f}, max={pre_clip_norms_arr.max():.4f}")
    print(f"  FPN norms: min={fpn_norms_arr.min():.4f}, max={fpn_norms_arr.max():.4f}")

    # ---- UMAP projection ----
    print(f"\n=== UMAP Projection (hyperboloid output metric) ===")
    embeddings_2d, prototypes_2d = project_umap(
        embeddings, prototypes.cpu(),
        curvature=args.hyp_c,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )

    print(f"  Embeddings 2D: {embeddings_2d.shape}")
    print(f"  Prototypes 2D: {prototypes_2d.shape}")
    print(f"  2D norms: min={np.linalg.norm(embeddings_2d, axis=-1).min():.4f}, "
          f"max={np.linalg.norm(embeddings_2d, axis=-1).max():.4f}")

    # ---- Create versioned output directory ----
    save_dir = make_versioned_dir(args.output_dir)
    print(f"\n  Output directory: {save_dir}")

    plot_full_visualization(
        embeddings_2d, class_names_arr, is_known_arr, max_horo_arr,
        prototypes_2d, known_class_names_orig,
        save_dir, curvature=args.hyp_c,
        embeddings_hd=embeddings, prototypes_hd=prototypes.cpu(), hyp_c=args.hyp_c,
        fpn_norms=fpn_norms_arr, pre_clip_norms=pre_clip_norms_arr,
    )

    # ---- Save raw data ----
    ckpt_name = Path(args.ckpt).stem
    np.savez(os.path.join(save_dir, f"umap_data_{ckpt_name}.npz"),
             embeddings_2d=embeddings_2d,
             prototypes_2d=prototypes_2d,
             class_names=class_names_arr,
             is_known=is_known_arr,
             max_horo=max_horo_arr,
             assigned_proto=assigned_proto_arr,
             known_class_names=np.array(known_class_names_orig),
             prototypes_hd=prototypes.cpu().numpy(),
             fpn_norms=fpn_norms_arr,
             pre_clip_norms=pre_clip_norms_arr)
    print(f"  Saved: {save_dir}/umap_data_{ckpt_name}.npz")

    # ---- Write interpretation guide ----
    guide = """
================================================================================
VISUALIZATION GUIDE — What to look for
================================================================================

1. umap_full_testset.png (Combined UMAP — all classes)
   PURPOSE: Overview of entire embedding space.
   IDEAL:   Known classes (dots) form tight clusters near their prototype star.
            Unknown classes (x) should be near the disk CENTER (low norm),
            far from all prototypes. This gives clean score separation.
   PROBLEM: If unknowns scatter at the BOUNDARY near prototypes, the model
            cannot distinguish them from knowns via geodesic scores.

2. umap_with_horospheres.png (Combined + geodesic prototype regions)
   PURPOSE: See which regions are claimed by each class.
   IDEAL:   Each prototype region cleanly encloses its class cluster.
            Unknowns fall FAR from all prototypes.
   PROBLEM: If unknowns fall NEAR prototypes, they'll be misclassified
            as that known class (false positives).

3. score_distributions.png (Known vs Unknown max geodesic score histograms)
   PURPOSE: The key OOD metric — can we threshold to separate known/unknown?
   IDEAL:   Known distribution shifted RIGHT (high scores = small d²), unknown shifted
            LEFT (low scores = large d²), with minimal overlap.
   PROBLEM: Heavy overlap means no clean threshold exists.

4. per_class_horosphere.png (Individual class views)
   PURPOSE: Per-class quality check — which classes are well-modeled?
   IDEAL:   Blue dots (own class) tightly clustered near their prototype.
            Red x's (unknowns) far away.

5. umap_known_only.png (Known classes ONLY)
   PURPOSE: See cluster quality without unknown clutter.
   IDEAL:   Tight, well-separated clusters around their prototype.
   PROBLEM: Diffuse or overlapping clusters = poor discrimination.

6. umap_unknown_only.png (Unknown classes ONLY)
   PURPOSE: See where unknowns project in the space.
   IDEAL:   Unknowns concentrate near the disk center (low Poincare norm).
            This means low geodesic scores → easy to threshold as OOD.
   PROBLEM: Unknowns at the boundary near prototypes = indistinguishable
            from knowns.

7. embedding_norms.png (Poincare ball norms)
   PURPOSE: In hyperbolic space, norm = confidence. Near boundary = certain.
   IDEAL:   Known classes at high norms (near clip_r=0.95, committed).
            Unknown classes at LOWER norms (uncertain, toward origin).
   PROBLEM: If both known and unknown have identical high norms, the model
            is equally "confident" about unknowns — bad for OOD detection.

8. prototype_distance_heatmap.png (Class → Prototype affinity)
   PURPOSE: Which prototype does each class (known & unknown) end up near?
   IDEAL:   Known classes show LOW distance (green) to their OWN prototype,
            HIGH distance (red) to all others.
            Unknown classes show HIGH distance to ALL prototypes.
   PROBLEM: If a unknown class (e.g., bus/truck) shows low distance to a
            known prototype (e.g., car), that unknown is being absorbed.

================================================================================
"""
    guide_path = os.path.join(save_dir, 'INTERPRETATION_GUIDE.txt')
    with open(guide_path, 'w') as f:
        f.write(guide)
    print(f"  Saved: {guide_path}")

    print(f"\n{'='*60}")
    print(f"VISUALIZATION COMPLETE — {save_dir}")
    print(f"{'='*60}")