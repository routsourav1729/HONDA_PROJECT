"""
Qualitative Visualization: GT vs Predictions for selected test images.

(left)  Ground Truth — all GT boxes color-coded by group
(right) Matched Predictions — only preds that match a GT box (IoU>0.5),
        plus GT boxes with NO match highlighted in yellow (MISSED).

Directly builds a mini-dataloader with only the requested images
(no scanning of 20K test set).

Usage (from hypyolov2/):
  python debug/visualize_qualitative.py \
      --task IDD_HYP/t2 \
      --ckpt IDD_HYP/t2/fewshotfinetunev3/model_final.pth \
      --images 116487 111921 106652 098971 103429 110117 \
      --output_dir visualizations/qualitative \
      --config-file configs/IDD_HYP/t2.yaml
"""

import os, sys, copy, tempfile
import xml.etree.ElementTree as ET
import numpy as np
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from PIL import Image

from core import add_config
from core.util.model_ema import add_model_ema_configs
from core.pascal_voc import register_pascal_voc, inital_prompts
from core.hyp_customyoloworld import HypCustomYoloWorld, load_hyp_ckpt
from core.calibrate_thresholds import compute_thresholds

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

GROUP_COLORS = {
    'BASE': '#2196F3',     # blue
    'NOVEL': '#4CAF50',    # green
    'UNKNOWN': '#F44336',  # red
    'MISSED': '#FFD600',   # bright yellow — GT objects the model missed
}


def get_class_group(cls_name):
    """Return (group, color) for a class name."""
    if cls_name in BASE_CLASSES:
        return 'BASE', GROUP_COLORS['BASE']
    elif cls_name in NOVEL_CLASSES:
        return 'NOVEL', GROUP_COLORS['NOVEL']
    elif cls_name in UNKNOWN_CLASSES:
        return 'UNKNOWN', GROUP_COLORS['UNKNOWN']
    return 'UNKNOWN', '#9E9E9E'


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


def iou(boxA, boxB):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def match_preds_to_gt(gt_boxes, preds, known_class_names, unknown_index,
                      iou_thresh=0.5):
    """
    Match predictions to GT boxes.

    Returns
    -------
    matched_preds : list[dict]
        Predictions with IoU>thresh to some GT box (successful detections).
    missed_gt : list[dict]
        GT boxes that had no matching prediction.
    """
    gt_matched = [False] * len(gt_boxes)
    matched_preds = []

    # Sort preds by score descending (greedy matching)
    sorted_preds = sorted(preds, key=lambda p: p['score'], reverse=True)

    for pred in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1
        pred_name = pred.get('pred_name', '')
        is_unk = pred.get('is_unknown', False)

        for gi, gt in enumerate(gt_boxes):
            if gt_matched[gi]:
                continue
            cur_iou = iou(pred['bbox'], gt['bbox'])
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_gt_idx = gi

        if best_iou >= iou_thresh and best_gt_idx >= 0:
            gt_name = gt_boxes[best_gt_idx]['name']
            # Accept match if:
            #   - predicted class matches GT class, OR
            #   - pred is "unknown" and GT is truly unknown, OR
            #   - pred is "unknown" and GT is novel/base (model flagged OOD — still a spatial match)
            # We show it as a spatial match so the user sees the model found the object.
            gt_matched[best_gt_idx] = True
            pred['matched_gt_name'] = gt_name
            matched_preds.append(pred)

    missed_gt = [gt_boxes[i] for i in range(len(gt_boxes)) if not gt_matched[i]]
    return matched_preds, missed_gt


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


def draw_gt_panel(ax, img, gt_boxes, title):
    """Left panel: all GT boxes color-coded by group."""
    ax.imshow(img)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.axis('off')
    for box in gt_boxes:
        name = box['name']
        _, color = get_class_group(name)
        x1, y1, x2, y2 = box['bbox']
        w, h = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), w, h, linewidth=2.0, edgecolor=color,
                          facecolor='none', linestyle='-')
        ax.add_patch(rect)
        ax.text(x1, max(y1 - 4, 2), name, fontsize=5.5, color='white',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor=color, alpha=0.85,
                          edgecolor='none'))


def draw_pred_panel(ax, img, matched_preds, missed_gt, title,
                    class_names, unknown_index):
    """Right panel: matched preds + missed GT in yellow dashed."""
    ax.imshow(img)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.axis('off')

    # Draw matched predictions (solid boxes)
    for pred in matched_preds:
        lbl_idx = pred['label']
        score = pred['score']
        is_unk = pred.get('is_unknown', False)
        if is_unk:
            color = GROUP_COLORS['UNKNOWN']
            label = f"unknown ({score:.2f})"
        else:
            name = class_names[lbl_idx] if (class_names and lbl_idx < len(class_names)) else f'cls{lbl_idx}'
            _, color = get_class_group(name)
            label = f"{name} ({score:.2f})"

        x1, y1, x2, y2 = pred['bbox']
        w, h = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), w, h, linewidth=2.0, edgecolor=color,
                          facecolor='none', linestyle='-')
        ax.add_patch(rect)
        ax.text(x1, max(y1 - 4, 2), label, fontsize=5.5, color='white',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor=color, alpha=0.85,
                          edgecolor='none'))

    # Draw missed GT boxes (dashed yellow)
    for gt in missed_gt:
        name = gt['name']
        x1, y1, x2, y2 = gt['bbox']
        w, h = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), w, h, linewidth=2.5, edgecolor=GROUP_COLORS['MISSED'],
                          facecolor=GROUP_COLORS['MISSED'], alpha=0.12,
                          linestyle='--')
        ax.add_patch(rect)
        ax.text(x1, max(y1 - 4, 2), f"MISS: {name}", fontsize=5.5,
                color='black', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor=GROUP_COLORS['MISSED'],
                          alpha=0.9, edgecolor='none'))


def make_legend():
    return [
        mpatches.Patch(color=GROUP_COLORS['BASE'], label='Base (T1 known)'),
        mpatches.Patch(color=GROUP_COLORS['NOVEL'], label='Novel (T2 known)'),
        mpatches.Patch(color=GROUP_COLORS['UNKNOWN'], label='Unknown (OOD)'),
        mpatches.Patch(facecolor=GROUP_COLORS['MISSED'], edgecolor=GROUP_COLORS['MISSED'],
                       linestyle='--', label='MISSED (no matching pred)'),
    ]


def extract_predictions(pred, class_names, unknown_index):
    """Convert InstanceData pred to list of dicts."""
    results = []
    n = len(pred.scores)
    for i in range(n):
        lbl = int(pred.labels[i].item())
        is_unk = (lbl == unknown_index)
        pred_name = 'unknown' if is_unk else (class_names[lbl] if lbl < len(class_names) else f'cls{lbl}')
        results.append({
            'bbox': pred.bboxes[i].cpu().tolist(),
            'label': lbl,
            'score': pred.scores[i].item(),
            'is_unknown': is_unk,
            'pred_name': pred_name,
        })
    return results


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--task", default="IDD_HYP/t2")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--images", nargs='+', required=True,
                        help="Image IDs (without extension)")
    parser.add_argument("--output_dir", default="visualizations/qualitative")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Override alpha for adaptive thresholds")
    parser.add_argument("--iou_thresh", type=float, default=0.5,
                        help="IoU threshold for matching preds to GT")
    parser.add_argument("--score_thr", type=float, default=0.15,
                        help="Score threshold for visualization (default 0.15, eval uses 0.001)")
    parser.add_argument("--nms_iou", type=float, default=0.45,
                        help="NMS IoU threshold for visualization (default 0.45, eval uses 0.7)")
    parser.add_argument("--max_per_img", type=int, default=100,
                        help="Max detections per image (default 100, eval uses 300)")
    parser.add_argument("--anno_dir", default="datasets/Annotations")
    parser.add_argument("--img_dir", default="datasets/JPEGImages")
    args = parser.parse_args()

    print(f"\nTarget images: {args.images}")

    task_name = args.task.split('/')[0]
    split_name = args.task.split('/')[1]
    base_dataset = task_name.replace('_HYP', '')

    cfg = setup(args)
    prev_cls = cfg.TEST.PREV_INTRODUCED_CLS
    cur_cls = cfg.TEST.CUR_INTRODUCED_CLS
    unknown_index = prev_cls + cur_cls

    # ---- Register dataset ----
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

    # ---- Load checkpoint ----
    print(f"\n=== Loading checkpoint: {args.ckpt} ===")
    ckpt_data = torch.load(args.ckpt, map_location='cpu')
    hyp_config = ckpt_data.get('hyp_config', {})
    hyp_c = hyp_config.get('curvature', 1.0)
    hyp_dim = hyp_config.get('embed_dim', 256)
    clip_r = hyp_config.get('clip_r', 1.0)
    adaptive_stats = ckpt_data.get('adaptive_stats', None)
    print(f"  hyp_c={hyp_c}, hyp_dim={hyp_dim}, clip_r={clip_r}")
    del ckpt_data

    # ---- Build YOLO-World runner ----
    config_file = os.path.join("./configs", task_name, split_name + ".py")
    cfgY = Config.fromfile(config_file)
    cfgY.work_dir = "."

    # ---- Override test_cfg for visualization (tighter than eval defaults) ----
    # Eval uses score_thr=0.001, nms_iou=0.7, max_per_img=300, multi_label=True
    # which produces ~200 dets/img (mostly junk). For vis we want clean boxes.
    cfgY.model.test_cfg = dict(
        multi_label=False,               # one class per anchor (argmax)
        nms_pre=1000,                    # 1k proposals into NMS (vs 30k)
        score_thr=args.score_thr,        # 0.15 default (vs 0.001)
        nms=dict(type='nms', iou_threshold=args.nms_iou),  # 0.45 (vs 0.7)
        max_per_img=args.max_per_img,    # 100 (vs 300)
    )
    print(f"\nVisualization test_cfg: score_thr={args.score_thr}, "
          f"nms_iou={args.nms_iou}, max_per_img={args.max_per_img}, multi_label=False")

    runner = Runner.from_cfg(cfgY)
    runner.call_hook("before_run")
    runner.load_or_resume()
    runner.model.reparameterize([known_class_names])
    runner.model.eval()

    # ---- Build mini-dataloader with ONLY target images ----
    # Write a temp file listing just the requested image IDs
    tmp_ann = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                          dir='./datasets/ImageSets/Main/IDD')
    for img_id in args.images:
        tmp_ann.write(f"{img_id}\n")
    tmp_ann.flush()
    tmp_ann_relpath = os.path.relpath(tmp_ann.name, './datasets')
    print(f"\nMini image list ({len(args.images)} images): {tmp_ann.name}")

    mini_dl_cfg = copy.deepcopy(cfgY.test_dataloader)
    mini_dl_cfg['dataset']['ann_file'] = tmp_ann_relpath
    mini_dl_cfg['batch_size'] = len(args.images)  # all in one batch
    mini_loader = Runner.build_dataloader(mini_dl_cfg)
    print(f"Mini dataloader: {len(mini_loader)} batches")

    # ---- Build HypCustomYoloWorld ----
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

    # ---- Compute adaptive thresholds ----
    adaptive_thresholds = None
    if adaptive_stats is not None:
        adaptive_thresholds, alpha_used = compute_thresholds(
            adaptive_stats, known_class_names, alpha=args.alpha
        )
        adaptive_thresholds = adaptive_thresholds.cuda()
        print(f"\nOOD Strategy: ADAPTIVE PER-PROTOTYPE (alpha={alpha_used:.2f})")
    else:
        print(f"\nOOD Strategy: NONE (no calibration data)")

    os.makedirs(args.output_dir, exist_ok=True)
    found = {}  # img_id -> preds_list

    # ---- Run inference on ONLY the target images ----
    print(f"\nRunning inference on {len(args.images)} target images...")
    for batch in mini_loader:
        data = model.parent.data_preprocessor(batch)
        with torch.no_grad():
            outputs = model.predict(data['inputs'], data['data_samples'])

        for out in outputs:
            img_id = out.img_id
            pred = out.pred_instances
            n_dets = len(pred.scores)

            if n_dets > 0 and adaptive_thresholds is not None:
                if hasattr(pred, 'horo_max_scores') and hasattr(pred, 'horo_assigned_proto'):
                    proto_indices = pred.horo_assigned_proto.long()
                    per_det_thresholds = adaptive_thresholds[proto_indices]
                    is_unknown = pred.horo_max_scores < per_det_thresholds
                    pred.labels[is_unknown] = unknown_index

            preds_list = extract_predictions(pred, known_class_names, unknown_index)
            found[img_id] = preds_list
            print(f"  {img_id}: {n_dets} raw detections "
                  f"({sum(1 for p in preds_list if not p['is_unknown'])} known, "
                  f"{sum(1 for p in preds_list if p['is_unknown'])} unknown)")

    # Clean up temp file
    tmp_ann.close()
    os.unlink(tmp_ann.name)

    missing = set(args.images) - set(found.keys())
    if missing:
        print(f"\n  WARNING: Missing images: {sorted(missing)}")

    # ---- Generate visualizations ----
    print(f"\n{'='*60}")
    print(f"Generating visualizations (IoU threshold={args.iou_thresh})...")
    print(f"{'='*60}")

    for img_id in args.images:
        if img_id not in found:
            continue

        img_path = os.path.join(args.img_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            print(f"  WARNING: {img_path} not found")
            continue

        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)

        # Parse GT
        gt_boxes = parse_xml(args.anno_dir, img_id)
        n_base = sum(1 for b in gt_boxes if b['name'] in BASE_CLASSES)
        n_novel = sum(1 for b in gt_boxes if b['name'] in NOVEL_CLASSES)
        n_unk = sum(1 for b in gt_boxes if b['name'] in UNKNOWN_CLASSES)

        # Match predictions to GT
        preds_list = found[img_id]
        matched_preds, missed_gt = match_preds_to_gt(
            gt_boxes, preds_list, known_class_names, unknown_index,
            iou_thresh=args.iou_thresh
        )

        n_matched = len(matched_preds)
        n_missed = len(missed_gt)
        missed_names = [g['name'] for g in missed_gt]
        print(f"  {img_id}: {n_matched}/{len(gt_boxes)} GT matched, "
              f"{n_missed} missed {missed_names}")

        # ===== Side-by-side figure =====
        fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(26, 10))

        draw_gt_panel(ax_gt, img_np, gt_boxes,
                      f'Ground Truth: {n_base}B + {n_novel}N + {n_unk}U '
                      f'= {len(gt_boxes)} objects')

        draw_pred_panel(ax_pred, img_np, matched_preds, missed_gt,
                        f'HypYOLO: {n_matched}/{len(gt_boxes)} matched, '
                        f'{n_missed} missed',
                        class_names=known_class_names,
                        unknown_index=unknown_index)

        fig.legend(handles=make_legend(), loc='lower center', ncol=4,
                   fontsize=11, framealpha=0.9, fancybox=True)
        plt.suptitle(f'HypYOLO — Image {img_id} | Task: {args.task}',
                     fontsize=15, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        save_path = os.path.join(args.output_dir, f'{img_id}_gt_vs_pred.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {save_path}")

    # ===== Summary grid =====
    valid_ids = [iid for iid in args.images if iid in found]
    if len(valid_ids) > 1:
        n_imgs = len(valid_ids)
        fig, axes = plt.subplots(n_imgs, 2, figsize=(26, 6 * n_imgs))
        if n_imgs == 1:
            axes = axes.reshape(1, -1)

        for row, img_id in enumerate(valid_ids):
            img = Image.open(os.path.join(args.img_dir, f"{img_id}.jpg")).convert('RGB')
            img_np = np.array(img)
            gt_boxes = parse_xml(args.anno_dir, img_id)
            preds_list = found[img_id]
            matched_preds, missed_gt = match_preds_to_gt(
                gt_boxes, preds_list, known_class_names, unknown_index,
                iou_thresh=args.iou_thresh)

            n_base = sum(1 for b in gt_boxes if b['name'] in BASE_CLASSES)
            n_novel = sum(1 for b in gt_boxes if b['name'] in NOVEL_CLASSES)
            n_unk = sum(1 for b in gt_boxes if b['name'] in UNKNOWN_CLASSES)

            draw_gt_panel(axes[row, 0], img_np, gt_boxes,
                          f'[{img_id}] GT: {n_base}B+{n_novel}N+{n_unk}U')

            draw_pred_panel(axes[row, 1], img_np, matched_preds, missed_gt,
                            f'[{img_id}] {len(matched_preds)}/{len(gt_boxes)} matched, '
                            f'{len(missed_gt)} missed',
                            class_names=known_class_names,
                            unknown_index=unknown_index)

        fig.legend(handles=make_legend(), loc='lower center', ncol=4,
                   fontsize=13, framealpha=0.9, fancybox=True)
        plt.suptitle(f'HypYOLO Qualitative Results | {args.task}',
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        grid_path = os.path.join(args.output_dir, 'summary_grid.png')
        plt.savefig(grid_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved summary grid: {grid_path}")

    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {args.output_dir}")
    print(f"{'='*60}")
