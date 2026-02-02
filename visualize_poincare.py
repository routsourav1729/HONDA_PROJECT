"""
Fast Poincaré Ball Visualization using 30-shot subset.

This script:
1. Loads only the HyperbolicProjector from checkpoint (not full YOLO-World)
2. Uses vis_subset.txt (182 images from 30-shot files)
3. Uses frozen YOLO-World backbone to extract FPN features
4. Projects to Poincaré ball and plots 2D projections
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm
from PIL import Image
import xml.etree.ElementTree as ET

from mmengine.config import Config
from mmdet.apis import init_detector
from mmdet.registry import MODELS
import mmcv

# Import our hyperbolic components
from core.hyp_customyoloworld import HyperbolicProjector
from core.hyperbolic import ToPoincare


# IDD T1 classes
T1_CLASSES = ['car', 'motorcycle', 'rider', 'person', 'autorickshaw', 
              'bus', 'truck', 'bicycle', 'traffic sign', 'traffic light', 'ego vehicle']
T1_CLASS_TO_IDX = {name: idx for idx, name in enumerate(T1_CLASSES)}


def load_vis_subset(subset_file):
    """Load image IDs from vis_subset.txt"""
    with open(subset_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_annotations(xml_path, classes):
    """Load GT boxes and labels from VOC XML annotation"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except:
        return [], []
    
    boxes = []
    labels = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name in classes:
            label = T1_CLASS_TO_IDX.get(name, -1)
            if label >= 0:
                bbox = obj.find('bndbox')
                xmin = int(float(bbox.find('xmin').text))
                ymin = int(float(bbox.find('ymin').text))
                xmax = int(float(bbox.find('xmax').text))
                ymax = int(float(bbox.find('ymax').text))
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)
    
    return boxes, labels


def load_yolo_backbone(config_path, weights_path, device):
    """Load YOLO-World and freeze it - only use for feature extraction"""
    cfg = Config.fromfile(config_path)
    model = init_detector(cfg, weights_path, device=device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def extract_fpn_features(model, img_tensor):
    """Extract FPN features from YOLO-World backbone+neck"""
    with torch.no_grad():
        # YOLO-World extracts features through backbone and neck
        feats = model.backbone(img_tensor)
        neck_feats = model.neck(feats)
    return neck_feats  # List of 3 FPN scales


def preprocess_image(img_path, img_size=640):
    """Preprocess image for YOLO-World"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return img_tensor


def project_to_2d(embeddings, method='pca'):
    """Project high-dim hyperbolic embeddings to 2D for visualization"""
    if method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        return pca.fit_transform(embeddings.cpu().numpy())
    elif method == 'first2':
        # Just take first 2 dimensions
        return embeddings[:, :2].cpu().numpy()
    else:
        raise ValueError(f"Unknown projection method: {method}")


def plot_poincare_disk(embeddings_2d, labels, prototypes_2d, class_names, save_path):
    """Plot embeddings on a Poincaré disk visualization"""
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw unit circle (boundary of Poincaré disk)
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
    
    # Color map for classes
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
    
    # Plot embeddings
    for i, class_name in enumerate(class_names):
        mask = labels == i
        if mask.sum() > 0:
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                      c=[colors[i]], label=f'{class_name} ({mask.sum()})', 
                      alpha=0.5, s=20)
    
    # Plot prototypes as larger markers
    for i, class_name in enumerate(class_names):
        ax.scatter(prototypes_2d[i, 0], prototypes_2d[i, 1],
                  c=[colors[i]], s=200, marker='*', edgecolors='black', linewidths=2)
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    ax.set_title('Poincaré Disk: Anchor Embeddings and Class Prototypes')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to hyperbolic checkpoint')
    parser.add_argument('--config', type=str, 
                       default='configs/idd_owod/yolo_world_v2_xl_vlpan_bn_2e-4_80e_8gpus_finetune_idd.py',
                       help='YOLO-World config file')
    parser.add_argument('--yolo-weights', type=str,
                       default='yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth',
                       help='YOLO-World pretrained weights')
    parser.add_argument('--subset-file', type=str,
                       default='datasets/ImageSets/Main/IDD/vis_subset.txt',
                       help='File with image IDs for visualization')
    parser.add_argument('--img-dir', type=str,
                       default='datasets/JPEGImages',
                       help='Directory with images')
    parser.add_argument('--ann-dir', type=str,
                       default='datasets/Annotations',
                       help='Directory with VOC annotations')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                       help='Output directory for plots')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max-images', type=int, default=50,
                       help='Max images to process for speed')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    
    # Extract hyperbolic projector state
    projector_state = {}
    prototype_weights = None
    for k, v in state_dict.items():
        if k.startswith('hyp_projector.'):
            new_key = k.replace('hyp_projector.', '')
            projector_state[new_key] = v
        if k == 'hyp_projector.prototype_tangent':
            # This is the learnable prototype parameter
            prototype_weights = v
        elif k == 'prototypes':
            prototype_weights = v
    
    if not projector_state:
        print("ERROR: No hyp_projector weights found in checkpoint!")
        return
    
    # Create projector and load weights
    print("Creating HyperbolicProjector...")
    projector = HyperbolicProjector(
        fpn_dims=[384, 768, 768],  # YOLO-World XL FPN dims
        hidden_dim=512,
        output_dim=256,
        curvature=0.1
    ).to(device)
    projector.load_state_dict(projector_state)
    projector.eval()
    
    # Load prototypes
    if prototype_weights is not None:
        prototypes = prototype_weights.to(device)
        print(f"Loaded prototypes: {prototypes.shape}")
    else:
        print("WARNING: No prototype weights found!")
        return
    
    # Load YOLO-World backbone
    print(f"Loading YOLO-World backbone from: {args.yolo_weights}")
    yolo_model = load_yolo_backbone(args.config, args.yolo_weights, device)
    
    # Load image subset
    image_ids = load_vis_subset(args.subset_file)
    if args.max_images > 0:
        image_ids = image_ids[:args.max_images]
    print(f"Processing {len(image_ids)} images...")
    
    # Collect embeddings
    all_embeddings = []
    all_labels = []
    
    for img_id in tqdm(image_ids, desc="Extracting features"):
        img_path = Path(args.img_dir) / f"{img_id}.jpg"
        ann_path = Path(args.ann_dir) / f"{img_id}.xml"
        
        if not img_path.exists():
            continue
        
        # Load and preprocess image
        img_tensor = preprocess_image(img_path).to(device)
        
        # Extract FPN features
        fpn_feats = extract_fpn_features(yolo_model, img_tensor)
        
        # Project to hyperbolic space
        with torch.no_grad():
            hyp_embeddings = projector(fpn_feats)  # [1, N_anchors, 256]
        
        # Load GT annotations
        boxes, labels = load_annotations(ann_path, T1_CLASSES)
        
        if len(boxes) == 0:
            continue
        
        # Simple approach: take embeddings from grid cells near GT boxes
        # For 640x640 image with stride 8/16/32, we have 80x80, 40x40, 20x20 grids
        # Total anchors = 8400
        H, W = 640, 640
        strides = [8, 16, 32]
        grid_sizes = [H // s for s in strides]  # [80, 40, 20]
        
        embeddings_flat = hyp_embeddings[0]  # [8400, 256]
        
        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
            
            # Map to grid cell in middle scale (stride 16, grid 40x40)
            grid_x = int(cx / W * 40)
            grid_y = int(cy / H * 40)
            grid_x = min(max(grid_x, 0), 39)
            grid_y = min(max(grid_y, 0), 39)
            
            # Index in flattened tensor (after 80x80=6400 from first scale)
            anchor_idx = 80*80 + grid_y * 40 + grid_x
            
            all_embeddings.append(embeddings_flat[anchor_idx])
            all_labels.append(label)
    
    if len(all_embeddings) == 0:
        print("No embeddings collected! Check data paths.")
        return
    
    # Stack embeddings
    embeddings = torch.stack(all_embeddings)  # [N, 256]
    labels = torch.tensor(all_labels)
    print(f"Collected {len(embeddings)} embeddings from {len(T1_CLASSES)} classes")
    
    # Count per class
    for i, cls_name in enumerate(T1_CLASSES):
        count = (labels == i).sum().item()
        print(f"  {cls_name}: {count}")
    
    # Project to 2D for visualization
    print("Projecting to 2D...")
    embeddings_2d = project_to_2d(embeddings, method='pca')
    prototypes_2d = project_to_2d(prototypes, method='pca')
    
    # Normalize to fit in unit disk for visualization
    max_norm = max(np.abs(embeddings_2d).max(), np.abs(prototypes_2d).max())
    embeddings_2d = embeddings_2d / (max_norm + 0.1)
    prototypes_2d = prototypes_2d / (max_norm + 0.1)
    
    # Plot
    checkpoint_name = Path(args.checkpoint).stem
    save_path = output_dir / f"poincare_disk_{checkpoint_name}.png"
    plot_poincare_disk(embeddings_2d, labels.numpy(), prototypes_2d, T1_CLASSES, save_path)
    
    # Also save raw data for further analysis
    np.savez(output_dir / f"embeddings_{checkpoint_name}.npz",
             embeddings=embeddings.cpu().numpy(),
             labels=labels.numpy(),
             prototypes=prototypes.cpu().numpy(),
             class_names=T1_CLASSES)
    print(f"Saved embeddings to: {output_dir}/embeddings_{checkpoint_name}.npz")


if __name__ == '__main__':
    main()
