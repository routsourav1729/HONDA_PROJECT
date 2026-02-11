#!/usr/bin/env python
"""
Compute CLIP text embeddings for class prompts and save to cache.

This script should be run on compute node BEFORE training to:
1. Get CLIP text embeddings for all class prompts
2. Project them to hyperbolic embedding dimension
3. Save to cache file for loading during training

Usage:
    python compute_clip_cache.py --task t1 --output clip_cache/idd_t1.pt
    python compute_clip_cache.py --task t2 --output clip_cache/idd_t2.pt
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# IDD class definitions
IDD_T1_CLASSES = [
    "car", "motorcycle", "rider", "person", "autorickshaw",
    "traffic sign", "traffic light", "pole", "bicycle"
]

IDD_T2_NOVEL_CLASSES = [
    "bus", "truck", "tanker_vehicle", "tractor", "street_cart"
]

IDD_UNKNOWN_CLASSES = [
    "animal", "pull_cart", "road_roller", "concrete_mixer", "crane_truck", "excavator"
]


def get_class_prompts(class_names, prompt_template="a photo of a {}"):
    """Generate text prompts for each class."""
    return [prompt_template.format(name) for name in class_names]


def compute_clip_embeddings(class_names, model_name='openai/clip-vit-large-patch14-336'):
    """
    Compute CLIP text embeddings for class names.
    
    Parameters
    ----------
    class_names : list of str
        Class names to encode
    model_name : str
        HuggingFace CLIP model name
    
    Returns
    -------
    tensor (num_classes, embed_dim)
        Normalized CLIP text embeddings
    """
    from transformers import CLIPModel, CLIPProcessor
    
    print(f"Loading CLIP model: {model_name}")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # Generate prompts
    prompts = get_class_prompts(class_names)
    print(f"Encoding {len(prompts)} class prompts...")
    
    # Encode
    with torch.no_grad():
        inputs = processor(text=prompts, return_tensors="pt", padding=True)
        text_features = model.get_text_features(**inputs)
        text_features = F.normalize(text_features, dim=-1)
    
    print(f"CLIP embeddings shape: {text_features.shape}")
    return text_features


def project_to_hyp_dim(clip_embeddings, hyp_dim=256, method='pca'):
    """
    Project CLIP embeddings to hyperbolic embedding dimension.
    
    Parameters
    ----------
    clip_embeddings : tensor (num_classes, clip_dim)
        CLIP text embeddings (typically 768-dim for ViT-L)
    hyp_dim : int
        Target hyperbolic dimension (default: 256)
    method : str
        Projection method: 'pca' or 'random'
    
    Returns
    -------
    tensor (num_classes, hyp_dim)
        Projected embeddings
    projection_matrix : tensor (clip_dim, hyp_dim)
        Projection matrix for later use
    """
    clip_dim = clip_embeddings.shape[-1]
    
    if clip_dim == hyp_dim:
        print(f"Dimensions match ({clip_dim}), no projection needed")
        return clip_embeddings, torch.eye(clip_dim)
    
    if method == 'pca':
        # PCA projection to preserve most variance
        print(f"PCA projection: {clip_dim} -> {hyp_dim}")
        
        # Center the data
        mean = clip_embeddings.mean(dim=0, keepdim=True)
        centered = clip_embeddings - mean
        
        # SVD for PCA
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        
        # Take top hyp_dim components
        projection_matrix = Vh[:hyp_dim, :].T  # (clip_dim, hyp_dim)
        
        # Project
        projected = centered @ projection_matrix
        projected = F.normalize(projected, dim=-1)  # Re-normalize
        
    else:
        # Random orthogonal projection
        print(f"Random orthogonal projection: {clip_dim} -> {hyp_dim}")
        projection_matrix = torch.randn(clip_dim, hyp_dim)
        projection_matrix = torch.linalg.qr(projection_matrix)[0]  # Orthogonalize
        
        projected = clip_embeddings @ projection_matrix
        projected = F.normalize(projected, dim=-1)
    
    return projected, projection_matrix


def main():
    parser = argparse.ArgumentParser(description='Compute CLIP embeddings for IDD classes')
    parser.add_argument('--task', type=str, choices=['t1', 't2'], default='t1',
                        help='Task (t1 or t2)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output cache file path')
    parser.add_argument('--hyp_dim', type=int, default=256,
                        help='Hyperbolic embedding dimension')
    parser.add_argument('--model', type=str, default='openai/clip-vit-large-patch14-336',
                        help='CLIP model name')
    parser.add_argument('--projection', type=str, choices=['pca', 'random'], default='pca',
                        help='Projection method')
    args = parser.parse_args()
    
    # Get classes for task
    if args.task == 't1':
        known_classes = IDD_T1_CLASSES
    else:  # t2
        known_classes = IDD_T1_CLASSES + IDD_T2_NOVEL_CLASSES
    
    all_classes = known_classes + ['unknown']
    
    print(f"\n=== Computing CLIP cache for IDD {args.task.upper()} ===")
    print(f"Known classes ({len(known_classes)}): {known_classes}")
    print(f"Total classes (with unknown): {len(all_classes)}")
    
    # Compute CLIP embeddings
    clip_embeddings = compute_clip_embeddings(all_classes, model_name=args.model)
    
    # Project to hyperbolic dimension
    projected_embeddings, projection_matrix = project_to_hyp_dim(
        clip_embeddings, 
        hyp_dim=args.hyp_dim,
        method=args.projection
    )
    
    # Compute semantic hierarchy (cosine similarities)
    print("\nComputing semantic similarity matrix...")
    similarity_matrix = clip_embeddings @ clip_embeddings.T
    
    # Save to cache
    output_path = args.output or f'clip_cache/idd_{args.task}.pt'
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    cache = {
        'class_names': all_classes,
        'known_classes': known_classes,
        'clip_embeddings': clip_embeddings,           # (num_classes, clip_dim)
        'projected_embeddings': projected_embeddings, # (num_classes, hyp_dim)
        'projection_matrix': projection_matrix,       # (clip_dim, hyp_dim)
        'similarity_matrix': similarity_matrix,       # (num_classes, num_classes)
        'hyp_dim': args.hyp_dim,
        'clip_dim': clip_embeddings.shape[-1],
        'model_name': args.model,
        'task': args.task
    }
    
    torch.save(cache, output_path)
    print(f"\n✓ Saved CLIP cache to: {output_path}")
    
    # Print semantic hierarchy
    print("\n=== Semantic Hierarchy (Top-3 similar classes for each) ===")
    for i, name in enumerate(all_classes):
        sims = similarity_matrix[i].clone()
        sims[i] = -1  # Exclude self
        top3_idx = sims.argsort(descending=True)[:3]
        top3 = [(all_classes[j], sims[j].item()) for j in top3_idx]
        print(f"  {name}: {top3}")
    
    print("\n✓ Done! Use this cache file when initializing training.")


if __name__ == '__main__':
    main()
