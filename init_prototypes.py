#!/usr/bin/env python
"""
Initialize prototype directions for horospherical classification.

This script should be run ONCE before training to create semantically-structured
prototype directions on the unit sphere.

Steps:
1. Compute CLIP text similarities between class prompts
2. Generate uniformly distributed directions on S^{D-1} via repulsion
3. Use Gromov-Wasserstein to assign CLIP classes to uniform directions
4. Save the assigned directions for use in training

Usage:
    # For T1 (9 classes):
    python init_prototypes.py \
        --classes "car,motorcycle,rider,person,autorickshaw,traffic sign,traffic light,pole,bicycle" \
        --out_dim 256 \
        --output init_protos_t1.pt
    
    # For T2 (14 classes):
    python init_prototypes.py \
        --classes "car,motorcycle,rider,person,autorickshaw,traffic sign,traffic light,pole,bicycle,bus,truck,tanker_vehicle,tractor,street_cart" \
        --out_dim 256 \
        --output init_protos_t2.pt
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np


def compute_clip_similarities(class_names, prompt_template="a photo of a {} on an Indian road"):
    """
    Compute pairwise cosine similarities between CLIP text embeddings.
    
    Returns:
        text_feats: (K, clip_dim) normalized CLIP embeddings
        cos_sim: (K, K) cosine similarity matrix
    """
    try:
        import clip
    except ImportError:
        raise ImportError("Please install clip: pip install git+https://github.com/openai/CLIP.git")
    
    print(f"Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    
    # Generate prompts
    prompts = [prompt_template.format(name) for name in class_names]
    print(f"Encoding {len(prompts)} class prompts:")
    for i, p in enumerate(prompts):
        print(f"  {i}: {p}")
    
    # Encode text
    with torch.no_grad():
        text_tokens = clip.tokenize(prompts).to(device)
        text_feats = model.encode_text(text_tokens)
        text_feats = F.normalize(text_feats, dim=-1)
    
    # Compute cosine similarities
    cos_sim = (text_feats @ text_feats.T).cpu()
    
    print(f"\nCLIP embeddings shape: {text_feats.shape}")
    print(f"Cosine similarity matrix:\n{cos_sim.numpy().round(3)}")
    
    return text_feats.cpu(), cos_sim


def generate_uniform_directions(K, D, n_iters=3000, lr=0.01):
    """
    Generate K uniformly distributed unit vectors on S^{D-1} via repulsion.
    
    Uses gradient descent to minimize the potential energy of points on a sphere,
    which naturally spreads them apart uniformly.
    
    Args:
        K: Number of directions
        D: Dimension
        n_iters: Optimization iterations
        lr: Learning rate
    
    Returns:
        (K, D) tensor of unit vectors
    """
    print(f"\nGenerating {K} uniform directions on S^{D-1}...")
    
    # Initialize random directions
    dirs = torch.randn(K, D, requires_grad=True)
    optimizer = torch.optim.Adam([dirs], lr=lr)
    
    for i in range(n_iters):
        # Normalize to unit sphere
        p = F.normalize(dirs, dim=-1)
        
        # Compute pairwise squared distances
        dsq = torch.cdist(p, p).pow(2)
        
        # Mask diagonal (self-distances)
        mask = ~torch.eye(K, dtype=torch.bool)
        
        # Repulsion loss: minimize -log(mean(exp(-dsq))) = maximize distances
        # This is equivalent to maximizing the minimum distance (approximately)
        loss = torch.log(torch.exp(-dsq[mask]).mean())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 500 == 0:
            with torch.no_grad():
                p_norm = F.normalize(dirs, dim=-1)
                min_dist = torch.cdist(p_norm, p_norm)[mask].min().item()
                mean_dist = torch.cdist(p_norm, p_norm)[mask].mean().item()
            print(f"  Iter {i+1}/{n_iters}: loss={loss.item():.4f}, min_dist={min_dist:.4f}, mean_dist={mean_dist:.4f}")
    
    # Final normalized directions
    uniform = F.normalize(dirs.detach(), dim=-1)
    
    # Verify uniformity
    final_dists = torch.cdist(uniform, uniform)
    final_dists_offdiag = final_dists[~torch.eye(K, dtype=torch.bool)]
    print(f"Final distribution stats:")
    print(f"  Min pairwise distance: {final_dists_offdiag.min():.4f}")
    print(f"  Mean pairwise distance: {final_dists_offdiag.mean():.4f}")
    print(f"  Max pairwise distance: {final_dists_offdiag.max():.4f}")
    
    return uniform


def gromov_wasserstein_assignment(clip_sim, uniform_dirs):
    """
    Use Gromov-Wasserstein to assign CLIP classes to uniform directions.
    
    This finds a mapping that preserves the relative similarity structure:
    - Classes that are similar in CLIP space should be assigned to 
      directions that are close on the sphere
    - Classes that are dissimilar should be assigned to distant directions
    
    Args:
        clip_sim: (K, K) CLIP cosine similarity matrix
        uniform_dirs: (K, D) uniform directions on sphere
    
    Returns:
        (K,) assignment indices: assigned_dirs = uniform_dirs[assignment]
    """
    try:
        import ot
    except ImportError:
        raise ImportError("Please install POT: pip install POT")
    
    print(f"\nComputing Gromov-Wasserstein assignment...")
    
    K = clip_sim.shape[0]
    
    # Convert to numpy
    clip_sim_np = clip_sim.numpy() if torch.is_tensor(clip_sim) else clip_sim
    uniform_np = uniform_dirs.numpy() if torch.is_tensor(uniform_dirs) else uniform_dirs
    
    # Dissimilarity matrices
    # For CLIP: dissimilarity = 1 - cosine_similarity
    M_T = 1.0 - clip_sim_np
    
    # For uniform dirs: dissimilarity = 1 - cosine_similarity
    uniform_sim = uniform_np @ uniform_np.T
    M_S = 1.0 - uniform_sim
    
    # Uniform marginals
    a = np.ones(K) / K
    b = np.ones(K) / K
    
    # Compute GW optimal transport
    P = ot.gromov.gromov_wasserstein(M_T, M_S, a, b, loss_fun='square_loss', verbose=True)
    
    # Get hard assignment (argmax of transport plan)
    assignment = P.argmax(axis=1)
    
    print(f"Assignment: {assignment}")
    
    # Verify assignment is a permutation (each uniform dir used once)
    if len(set(assignment)) != K:
        print(f"WARNING: Assignment is not a permutation! Some directions reused.")
        # Fall back to Hungarian algorithm for proper matching
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-P)
        assignment = col_ind
        print(f"Using Hungarian assignment: {assignment}")
    
    return assignment


def main():
    parser = argparse.ArgumentParser(description='Initialize prototype directions')
    parser.add_argument('--classes', type=str, required=True,
                        help='Comma-separated class names')
    parser.add_argument('--out_dim', type=int, default=256,
                        help='Output dimension for prototype directions')
    parser.add_argument('--output', type=str, default='init_protos.pt',
                        help='Output file path')
    parser.add_argument('--prompt_template', type=str, 
                        default="a photo of a {} on an Indian road",
                        help='CLIP prompt template')
    parser.add_argument('--n_iters', type=int, default=3000,
                        help='Iterations for uniform direction generation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    # Parse class names
    class_names = [c.strip() for c in args.classes.split(',')]
    K = len(class_names)
    D = args.out_dim
    
    print("=" * 60)
    print("PROTOTYPE DIRECTION INITIALIZATION")
    print("=" * 60)
    print(f"Classes ({K}): {class_names}")
    print(f"Output dimension: {D}")
    print(f"Output file: {args.output}")
    print("=" * 60)
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Step 1: CLIP similarities
    text_feats, clip_sim = compute_clip_similarities(
        class_names, 
        prompt_template=args.prompt_template
    )
    
    # Step 2: Generate uniform directions
    uniform_dirs = generate_uniform_directions(K, D, n_iters=args.n_iters)
    
    # Step 3: Gromov-Wasserstein assignment
    assignment = gromov_wasserstein_assignment(clip_sim, uniform_dirs)
    
    # Step 4: Get final assigned directions
    init_directions = uniform_dirs[assignment]
    
    # Verify final directions preserve similarity structure
    print(f"\n=== Verification ===")
    init_sim = (init_directions @ init_directions.T).numpy()
    clip_sim_np = clip_sim.numpy()
    
    # Check correlation between CLIP similarities and direction similarities
    mask = ~np.eye(K, dtype=bool)
    correlation = np.corrcoef(clip_sim_np[mask].flatten(), init_sim[mask].flatten())[0, 1]
    print(f"Correlation between CLIP sim and direction sim: {correlation:.4f}")
    
    # Save
    save_dict = {
        'init_directions': init_directions,  # (K, D) - USE THIS FOR TRAINING
        'class_names': class_names,
        'clip_text_feats': text_feats,
        'clip_sim': clip_sim,
        'uniform_dirs': uniform_dirs,
        'assignment': torch.tensor(assignment),
        'out_dim': D,
        'prompt_template': args.prompt_template,
    }
    
    torch.save(save_dict, args.output)
    print(f"\n✓ Saved to: {args.output}")
    print(f"  init_directions shape: {init_directions.shape}")
    
    # Print similarity structure
    print(f"\n=== Final Direction Similarities ===")
    print("(Should roughly match CLIP similarity structure)")
    for i, name_i in enumerate(class_names):
        sims = init_sim[i].copy()
        sims[i] = -999  # exclude self
        top3_idx = np.argsort(sims)[-3:][::-1]
        top3 = [(class_names[j], f"{sims[j]:.3f}") for j in top3_idx]
        print(f"  {name_i}: most similar = {top3}")


if __name__ == '__main__':
    main()
