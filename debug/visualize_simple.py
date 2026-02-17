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
from core.hyperbolic import busemann

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
    samples_count = defaultdict(int)

    prototypes = model.prototypes.detach()
    biases = model.prototype_biases.detach()

    with torch.no_grad():
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

                        B_vals = busemann(prototypes, emb.unsqueeze(0), c=hyp_c)
                        horo_scores = (-B_vals + biases).squeeze(0)
                        max_horo, assigned = horo_scores.max(dim=0)

                        all_embeddings.append(emb.cpu())
                        all_class_names.append(cls_name)
                        all_is_known.append(is_known)
                        all_max_horo.append(max_horo.item())
                        all_assigned_proto.append(assigned.item())
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
        return None, None, None, None, None

    embeddings = torch.stack(all_embeddings)
    return (embeddings,
            np.array(all_class_names),
            np.array(all_is_known),
            np.array(all_max_horo),
            np.array(all_assigned_proto))



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

def draw_horosphere_2d(ax, prototype_2d, bias, curvature=1.0, color='gray',
                       label=None, linestyle='-', alpha=0.25):
    """
    Draw a horosphere on the 2D Poincaré disk.

    In the Poincaré disk, a horosphere at ideal point p with Busemann level
    B_p(x) = bias  is a Euclidean circle internally tangent to the boundary
    at p.

    For the unit disk (c=1):
        Euclidean radius   r = 1 / (1 + exp(bias))
        Centre             = p * (1 - r)

    Positive bias -> smaller horosphere (tighter acceptance).
    Negative bias -> larger horosphere (looser acceptance).
    """
    R = 1.0 / np.sqrt(curvature)
    p = prototype_2d / (np.linalg.norm(prototype_2d) + 1e-15)  # unit direction

    r = R / (1.0 + np.exp(bias))
    centre = p * (R - r)

    theta = np.linspace(0, 2 * np.pi, 200)
    circle_x = centre[0] + r * np.cos(theta)
    circle_y = centre[1] + r * np.sin(theta)

    # Clip to inside the disk
    inside = (circle_x**2 + circle_y**2) <= R**2 * 1.01
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

def plot_full_visualization(embeddings_2d, class_names_arr, is_known_arr,
                            max_horo_arr, prototypes_2d, known_class_names,
                            biases, save_dir, curvature=1.0):
    """
    Create 4 figures:
      1. UMAP Poincaré disk — all classes colour-coded
      2. Same disk + horosphere circles
      3. Score distributions (known vs unknown)
      4. Per-class horosphere detail (3x3 grid)
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

    biases_np = biases.detach().cpu().numpy() if torch.is_tensor(biases) else np.asarray(biases)

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
        draw_horosphere_2d(
            ax2, prototypes_2d[i], biases_np[i],
            curvature=curvature, color=c,
            label=f'Horo {known_class_names[i]} (a={biases_np[i]:.3f})',
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
    ax2.set_title('Poincare Disk + Horosphere Boundaries (xi=0 level set)\n'
                  'Points INSIDE a horosphere -> classified as that class', fontsize=13)
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
    ax.set_xlabel('Max Horosphere Score (xi = -B + a)')
    ax.set_ylabel('Density')
    ax.set_title('Known vs Unknown: Max Horosphere Score')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes3[1]
    for cls in unique_unknown:
        mask = class_names_arr == cls
        if mask.sum() > 0:
            ax.hist(max_horo_arr[mask], bins=30, alpha=0.5,
                    label=f'{cls} ({mask.sum()})', density=True)
    ax.axvline(0, color='black', linestyle=':', linewidth=1.5, label='tau=0')
    ax.set_xlabel('Max Horosphere Score')
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

        draw_horosphere_2d(ax, prototypes_2d[idx], biases_np[idx],
                           curvature=curvature,
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
        ax.set_title(f'{cls} (bias={biases_np[idx]:.3f})', fontsize=10)
        ax.legend(fontsize=6, loc='lower right')

    for idx in range(n_grid, nrows * ncols):
        r, c = idx // ncols, idx % ncols
        axes4[r][c].set_visible(False)

    plt.suptitle('Per-Class Horosphere: Known (blue) vs Unknown (red)', fontsize=14, y=1.01)
    plt.tight_layout()
    path4 = os.path.join(save_dir, 'per_class_horosphere.png')
    fig4.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print(f"  Saved: {path4}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--task", default="IDD_HYP/t1")
    parser.add_argument("--ckpt", default="IDD_HYP/t1/horospherical/model_5.pth")
    parser.add_argument("--hyp_c", type=float, default=1.0)
    parser.add_argument("--hyp_dim", type=int, default=256)
    parser.add_argument("--clip_r", type=float, default=0.95)
    parser.add_argument("--output_dir", default="visualizations")
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)

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

    # Build TEST dataloader (not train!)
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

    # ---- Collect embeddings from FULL test set ----
    result = collect_embeddings(
        model, test_loader, known_class_names,
        dataset_root='./datasets',
        hyp_c=args.hyp_c,
    )

    if result[0] is None:
        print("ERROR: No embeddings collected!")
        sys.exit(1)

    embeddings, class_names_arr, is_known_arr, max_horo_arr, assigned_proto_arr = result

    print(f"\n  Total embeddings: {len(embeddings)}")
    print(f"  Embedding dim: {embeddings.shape[1]}")
    print(f"  Embedding norms: min={embeddings.norm(dim=-1).min():.4f}, max={embeddings.norm(dim=-1).max():.4f}")

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

    # ---- Plot ----
    os.makedirs(args.output_dir, exist_ok=True)

    plot_full_visualization(
        embeddings_2d, class_names_arr, is_known_arr, max_horo_arr,
        prototypes_2d, known_class_names, biases,
        args.output_dir, curvature=args.hyp_c,
    )

    # ---- Save raw data ----
    ckpt_name = Path(args.ckpt).stem
    np.savez(os.path.join(args.output_dir, f"umap_data_{ckpt_name}.npz"),
             embeddings_2d=embeddings_2d,
             prototypes_2d=prototypes_2d,
             class_names=class_names_arr,
             is_known=is_known_arr,
             max_horo=max_horo_arr,
             assigned_proto=assigned_proto_arr,
             known_class_names=np.array(known_class_names),
             biases=biases.cpu().numpy(),
             prototypes_hd=prototypes.cpu().numpy())
    print(f"  Saved: {args.output_dir}/umap_data_{ckpt_name}.npz")

    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*60}")