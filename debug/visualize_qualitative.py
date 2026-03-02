"""
Qualitative Visualization: GT vs Predictions for selected test images.
Generates side-by-side plots: (left) Ground Truth, (right) Model Predictions.
Supports T1 and T2 checkpoints with hyperbolic OOD detection.

Usage (from hypyolov2/):
  python debug/visualize_qualitative.py \
      --task IDD_HYP/t2 \
      --ckpt IDD_HYP/t2/fewshotfinetunev3/model_final.pth \
      --images 116487 111921 106652 098971 103429 110117 111444 098454 \
      --output_dir visualizations/qualitative
"""

import os, sys, argparse
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from pathlib import Path
from PIL import Image

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
# Class groupings for IDD
# ============================================================================
BASE_CLASSES = {"car", "motorcycle", "rider", "person", "autorickshaw",
                "bicycle", "traffic sign", "traffic light",
                "traffic_sign", "traffic_light"}
NOVEL_CLASSES = {"bus", "truck", "tanker_vehicle", "crane_truck",
                 "street_cart", "excavator"}
UNKNOWN_CLASSES = {"animal", "concrete_mixer", "pole", "pull_cart",
                   "road_roller", "tractor"}

# Colors: base=blue, novel=green, unknown=red, pred_known=blue, pred_unk=red
GROUP_COLORS = {
    'BASE': '#2196F3',     # blue
    'NOVEL': '#4CAF50',    # green
    'UNKNOWN': '#F44336',  # red
    'PREDICTED': '#FF9800', # orange (for predicted unknown)
}

def get_class_color_gt(cls_name):
    """Color for ground truth boxes."""
    if cls_name in BASE_CLASSES:
        return GROUP_COLORS['BASE']
    elif cls_name in NOVEL_CLASSES:
        return GROUP_COLORS['NOVEL']
    elif cls_name in UNKNOWN_CLASSES:
        return GROUP_COLORS['UNKNOWN']
    return '#9E9E9E'  # gray for unrecognized

def parse_xml(anno_dir, img_id):
    """Parse VOC XML annotation."""
    xml_path = os.path.join(anno_dir, f'{img_id}.xml')
    boxes = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall('object'):
        name = obj.find('name').text.strip()
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        boxes.append({'name': name, 'bbox': [x1, y1, x2, y2]})
    return boxes


def get_anchor_centers(h, w, device):
    strides = [8, 16, 32]
    centers = []
    for s in strides:
        gh, gw = h // s, w // s
        y = torch.arange(gh, device=device).float() * s + s / 2
        x = torch.arange(gw, device=device).float() * s + s / 2
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        centers.append(torch.stack([xx.flatten(), yy.flatten()], dim=-1))
    return torch.cat(centers, dim=0)


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


def run_inference_single(model, img_tensor, known_class_names, hyp_c,
                         adaptive_stats=None, conf_thresh=0.25, nms_iou=0.45):
    """
    Run full inference on a single preprocessed image tensor.
    Returns list of {'bbox': [x1,y1,x2,y2], 'class': str, 'score': float, 'is_unknown': bool}
    """
    model.eval()
    prototypes = model.prototypes.detach()
    biases = model.prototype_biases.detach()
    K = prototypes.shape[0]

    with torch.no_grad():
        # Forward through backbone + neck
        x = model.parent.backbone.forward_image(img_tensor)
        if model.parent.with_neck:
            if model.parent.mm_neck:
                txt = model.frozen_embeddings if model.frozen_embeddings is not None else model.embeddings
                txt = txt.repeat(x[0].shape[0], 1, 1)
                x = model.parent.neck(x, txt)
            else:
                x = model.parent.neck(x)

        # YOLO-World bbox head predictions
        head = model.parent.bbox_head
        cls_scores, bbox_preds, objectnesses = head(x, txt)

        # Get decoded bboxes from head
        from mmdet.structures.bbox import distance2bbox
        B, _, H, W = cls_scores[0].shape
        device = cls_scores[0].device

        all_boxes_decoded = []
        all_objectness = []
        all_anchors = []

        strides = [8, 16, 32]
        for lvl, (cls_s, bbox_p, obj_s) in enumerate(zip(cls_scores, bbox_preds, objectnesses)):
            s = strides[lvl]
            B, C, fH, fW = cls_s.shape
            # Anchor centers for this level
            yy, xx = torch.meshgrid(
                torch.arange(fH, device=device).float() * s + s/2,
                torch.arange(fW, device=device).float() * s + s/2,
                indexing='ij'
            )
            anchors = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # (fH*fW, 2)

            # Decode bboxes: bbox_p is (B, 4, fH, fW) -> (B, fH*fW, 4)
            bbox_flat = bbox_p[0].permute(1, 2, 0).reshape(-1, 4) * s  # scale by stride
            decoded = distance2bbox(anchors, bbox_flat)  # (fH*fW, 4) in x1y1x2y2

            obj_flat = obj_s[0].permute(1, 2, 0).reshape(-1, 1).sigmoid()

            all_boxes_decoded.append(decoded)
            all_objectness.append(obj_flat)
            all_anchors.append(anchors)

        all_boxes = torch.cat(all_boxes_decoded, dim=0)  # (8400, 4)
        all_obj = torch.cat(all_objectness, dim=0)        # (8400, 1)

        # Hyperbolic embeddings
        hyp_embeddings = model.hyp_projector(x)  # (1, 8400, dim)
        embs = hyp_embeddings[0]  # (8400, dim)

        # Busemann scores
        B_vals = busemann(prototypes, embs, c=hyp_c)    # (8400, K)
        horo_scores = -B_vals + biases                    # (8400, K)
        max_horo, assigned_cls = horo_scores.max(dim=-1)  # (8400,)

        # Combined score = objectness * max_horo
        combined = all_obj.squeeze(-1) * max_horo.sigmoid()

        # Filter by confidence
        keep = combined > conf_thresh
        boxes = all_boxes[keep]
        scores = combined[keep]
        cls_ids = assigned_cls[keep]
        horo_vals = max_horo[keep]

        # OOD detection via adaptive thresholds
        is_unknown = torch.zeros(len(boxes), dtype=torch.bool, device=device)
        if adaptive_stats:
            per_class = adaptive_stats.get('per_class', {})
            alpha = adaptive_stats.get('alpha', 0.75)
            for i in range(len(boxes)):
                cls_idx = cls_ids[i].item()
                if cls_idx < len(known_class_names):
                    cls_name = known_class_names[cls_idx]
                    if cls_name in per_class:
                        tau = per_class[cls_name].get('threshold', 0)
                        if horo_vals[i].item() < tau:
                            is_unknown[i] = True

        # NMS per class
        from torchvision.ops import batched_nms
        # Treat unknown as a separate class for NMS
        nms_cls = cls_ids.clone()
        nms_cls[is_unknown] = K  # unknown class ID
        keep_nms = batched_nms(boxes, scores, nms_cls, nms_iou)

        # Limit to top 200
        keep_nms = keep_nms[:200]

        results = []
        for idx in keep_nms:
            i = idx.item()
            cid = cls_ids[i].item()
            unk = is_unknown[i].item()
            if unk:
                label = 'unknown'
            elif cid < len(known_class_names):
                label = known_class_names[cid]
            else:
                label = f'cls_{cid}'

            results.append({
                'bbox': boxes[i].cpu().tolist(),
                'class': label,
                'score': scores[i].item(),
                'horo_score': horo_vals[i].item(),
                'is_unknown': unk,
            })

    return results


def draw_boxes(ax, img, boxes, title, mode='gt'):
    """Draw bounding boxes on image.
    mode='gt': color by BASE/NOVEL/UNKNOWN group
    mode='pred': color by predicted class, unknown in red
    """
    ax.imshow(img)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')

    for box in boxes:
        if mode == 'gt':
            name = box['name']
            color = get_class_color_gt(name)
            lw = 2
            label = name
            if name in UNKNOWN_CLASSES:
                label = f"*{name}"  # mark unknowns
        else:
            name = box['class']
            score = box['score']
            is_unk = box.get('is_unknown', False)
            if is_unk:
                color = GROUP_COLORS['UNKNOWN']
                label = f"UNK ({score:.2f})"
            elif name in NOVEL_CLASSES:
                color = GROUP_COLORS['NOVEL']
                label = f"{name} ({score:.2f})"
            else:
                color = GROUP_COLORS['BASE']
                label = f"{name} ({score:.2f})"
            lw = 2

        x1, y1, x2, y2 = box['bbox'] if mode == 'gt' else box['bbox']
        w, h = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), w, h, linewidth=lw, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 3, label, fontsize=6, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor=color, alpha=0.8))


def make_legend():
    """Create a legend for the color coding."""
    patches = [
        mpatches.Patch(color=GROUP_COLORS['BASE'], label='Base (T1 known)'),
        mpatches.Patch(color=GROUP_COLORS['NOVEL'], label='Novel (T2 known)'),
        mpatches.Patch(color=GROUP_COLORS['UNKNOWN'], label='Unknown (OOD)'),
    ]
    return patches


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--task", default="IDD_HYP/t2")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--images", nargs='+', required=True,
                        help="Image IDs (without extension)")
    parser.add_argument("--output_dir", default="visualizations/qualitative")
    parser.add_argument("--conf_thresh", type=float, default=0.15)
    parser.add_argument("--nms_iou", type=float, default=0.45)
    parser.add_argument("--anno_dir", default="datasets/Annotations")
    parser.add_argument("--img_dir", default="datasets/JPEGImages")
    args = parser.parse_args()

    task_name = args.task.split('/')[0]
    split_name = args.task.split('/')[1]
    base_dataset = task_name.replace('_HYP', '')

    cfg = setup(args)
    prev_cls = cfg.TEST.PREV_INTRODUCED_CLS
    cur_cls = cfg.TEST.CUR_INTRODUCED_CLS
    unknown_index = prev_cls + cur_cls

    # Register dataset
    if split_name in ['t2', 't3', 't4']:
        dataset_key = f"{base_dataset}_T{split_name[1].upper()}"
    else:
        dataset_key = base_dataset
    data_split = f"{base_dataset}/{split_name}"
    data_register = Register('./datasets/', data_split, cfg, dataset_key)
    data_register.register_dataset()

    all_class_names = list(inital_prompts()[dataset_key])
    known_class_names = all_class_names[:unknown_index]
    print(f"Known classes ({unknown_index}): {known_class_names}")

    # Read hyp_config from checkpoint
    ckpt_data = torch.load(args.ckpt, map_location='cpu')
    hyp_config = ckpt_data.get('hyp_config', {})
    hyp_c = hyp_config.get('curvature', 1.0)
    hyp_dim = hyp_config.get('embed_dim', 256)
    clip_r = hyp_config.get('clip_r', 1.0)
    adaptive_stats = ckpt_data.get('adaptive_stats', None)
    print(f"hyp_c={hyp_c}, hyp_dim={hyp_dim}, clip_r={clip_r}")
    if adaptive_stats:
        print(f"Adaptive stats: {len(adaptive_stats.get('per_class', {}))} classes")
    del ckpt_data

    # Build model
    config_file = os.path.join("./configs", task_name, split_name + ".py")
    cfgY = Config.fromfile(config_file)
    cfgY.work_dir = "."
    runner = Runner.from_cfg(cfgY)
    runner.call_hook("before_run")
    runner.load_or_resume()
    runner.model.reparameterize([known_class_names])
    runner.model.eval()

    classifier_num = cur_cls if prev_cls > 0 else unknown_index
    model = HypCustomYoloWorld(
        runner.model, unknown_index,
        hyp_c=hyp_c, hyp_dim=hyp_dim, clip_r=clip_r,
        num_classifier_classes=classifier_num,
    )
    with torch.no_grad():
        model = load_hyp_ckpt(model, args.ckpt, prev_cls, cur_cls, eval=True)
        model = model.cuda()
        model.add_generic_text(known_class_names, generic_prompt='object', alpha=0.4)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    # Process each image
    for img_id in args.images:
        img_path = os.path.join(args.img_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            print(f"  WARNING: {img_path} not found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {img_id}")
        print(f"{'='*60}")

        # Load image
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)

        # Parse GT annotations
        gt_boxes = parse_xml(args.anno_dir, img_id)
        n_base = sum(1 for b in gt_boxes if b['name'] in BASE_CLASSES)
        n_novel = sum(1 for b in gt_boxes if b['name'] in NOVEL_CLASSES)
        n_unk = sum(1 for b in gt_boxes if b['name'] in UNKNOWN_CLASSES)
        print(f"  GT: {len(gt_boxes)} objects (base={n_base}, novel={n_novel}, unknown={n_unk})")

        # Preprocess for model: resize to 640x640 with letterbox
        orig_h, orig_w = img_np.shape[:2]
        scale = min(640/orig_h, 640/orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)

        from torchvision.transforms.functional import resize, to_tensor
        img_resized = resize(to_tensor(img_np), [new_h, new_w])

        # Pad to 640x640
        pad_h = 640 - new_h
        pad_w = 640 - new_w
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        img_padded = F.pad(img_resized, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top),
                           value=114/255.0)
        img_tensor = img_padded.unsqueeze(0).cuda()

        # Run inference
        preds = run_inference_single(
            model, img_tensor, known_class_names, hyp_c,
            adaptive_stats=adaptive_stats,
            conf_thresh=args.conf_thresh,
            nms_iou=args.nms_iou
        )

        # Map predictions back to original image coords
        for p in preds:
            x1, y1, x2, y2 = p['bbox']
            p['bbox'] = [
                (x1 - pad_left) / scale,
                (y1 - pad_top) / scale,
                (x2 - pad_left) / scale,
                (y2 - pad_top) / scale,
            ]

        n_pred_known = sum(1 for p in preds if not p['is_unknown'])
        n_pred_unk = sum(1 for p in preds if p['is_unknown'])
        print(f"  Pred: {len(preds)} detections (known={n_pred_known}, unknown={n_pred_unk})")

        # ===== Draw side-by-side =====
        fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(24, 10))

        # Left: Ground Truth
        draw_boxes(ax_gt, img_np, gt_boxes,
                   f'Ground Truth: {n_base}B + {n_novel}N + {n_unk}U = {len(gt_boxes)} objects',
                   mode='gt')

        # Right: Predictions
        draw_boxes(ax_pred, img_np, preds,
                   f'Predictions: {n_pred_known} known + {n_pred_unk} unknown '
                   f'(conf>{args.conf_thresh})',
                   mode='pred')

        # Add legend
        legend_patches = make_legend()
        fig.legend(handles=legend_patches, loc='lower center', ncol=3,
                   fontsize=11, framealpha=0.9)

        plt.suptitle(f'Image {img_id} | Task: {args.task}', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.04, 1, 0.96])

        save_path = os.path.join(args.output_dir, f'{img_id}_gt_vs_pred.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {args.output_dir}")
    print(f"{'='*60}")
