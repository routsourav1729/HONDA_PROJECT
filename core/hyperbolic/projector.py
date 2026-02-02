"""
Hyperbolic Projector for multi-scale FPN features.
Projects visual features from YOLO backbone to Poincare ball.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import ToPoincare
from . import pmath


class HyperbolicProjector(nn.Module):
    """
    Projects multi-scale FPN features to hyperbolic (Poincare ball) space.
    
    Architecture:
    - 3 separate projectors for 3 FPN scales (P3, P4, P5)
    - Each projector: Conv → BN → ReLU → Conv → flatten
    - Concatenate all scales → ToPoincare
    - Learnable class prototypes in Poincare ball
    
    Parameters
    ----------
    in_dims : list of int
        Input dimensions for each scale [P3_dim, P4_dim, P5_dim]
        Default: [384, 768, 768] for YOLO-World XL
    out_dim : int
        Output embedding dimension (default: 256)
    curvature : float
        Poincare ball curvature (default: 0.1)
    num_classes : int
        Number of class prototypes
    clip_r : float
        Clip radius for ToPoincare (default: 2.3)
    riemannian : bool
        Use Riemannian gradient (default: True)
    """
    
    def __init__(
        self,
        in_dims=[384, 768, 768],
        out_dim=256,
        curvature=0.1,
        num_classes=20,
        clip_r=2.3,
        riemannian=True
    ):
        super(HyperbolicProjector, self).__init__()
        
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.c = curvature
        self.num_classes = num_classes
        self.clip_r = clip_r
        
        # Per-scale projectors
        # Architecture: Conv(in→in) → SyncBN → ReLU → Conv(in→out_dim)
        self.proj_p3 = nn.Sequential(
            nn.Conv2d(in_dims[0], in_dims[0], kernel_size=1, bias=False),
            nn.SyncBatchNorm(in_dims[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dims[0], out_dim, kernel_size=1, bias=False)
        )
        
        self.proj_p4 = nn.Sequential(
            nn.Conv2d(in_dims[1], in_dims[1], kernel_size=1, bias=False),
            nn.SyncBatchNorm(in_dims[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dims[1], out_dim, kernel_size=1, bias=False)
        )
        
        self.proj_p5 = nn.Sequential(
            nn.Conv2d(in_dims[2], in_dims[2], kernel_size=1, bias=False),
            nn.SyncBatchNorm(in_dims[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dims[2], out_dim, kernel_size=1, bias=False)
        )
        
        # ToPoincare layer (projects Euclidean → Poincare ball)
        self.to_poincare = ToPoincare(
            c=curvature,
            train_c=False,
            train_x=False,
            ball_dim=out_dim,
            riemannian=riemannian,
            clip_r=clip_r
        )
        
        # Learnable prototypes in TANGENT space (Euclidean)
        # Will be projected to Poincare ball via expmap0
        self.prototype_tangent = nn.Parameter(
            torch.randn(num_classes, out_dim) * 0.01
        )
        
        # Initialize projectors
        self._init_weights()
    
    def _init_weights(self):
        """Initialize convolutional weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    @property
    def prototypes(self):
        """
        Get prototypes in Poincare ball.
        Projects from tangent space to ball using expmap0.
        """
        protos = pmath.expmap0(self.prototype_tangent, c=self.c)
        protos = pmath.project(protos, c=self.c)  # Safety projection
        return protos
    
    def forward(self, img_feats):
        """
        Project FPN features to Poincare ball.
        
        Parameters
        ----------
        img_feats : tuple of tensor
            FPN features (P3, P4, P5) with shapes:
            - P3: (B, C0, H0, W0) e.g., (B, 384, 80, 80)
            - P4: (B, C1, H1, W1) e.g., (B, 768, 40, 40)
            - P5: (B, C2, H2, W2) e.g., (B, 768, 20, 20)
        
        Returns
        -------
        tensor
            Hyperbolic embeddings of shape (B, N_anchors, out_dim)
            where N_anchors = H0*W0 + H1*W1 + H2*W2 (e.g., 8400)
        """
        p3, p4, p5 = img_feats[0], img_feats[1], img_feats[2]
        B = p3.shape[0]
        
        # Project each scale
        z3 = self.proj_p3(p3)  # (B, out_dim, H0, W0)
        z4 = self.proj_p4(p4)  # (B, out_dim, H1, W1)
        z5 = self.proj_p5(p5)  # (B, out_dim, H2, W2)
        
        # Flatten spatial dimensions: (B, out_dim, H, W) → (B, H*W, out_dim)
        z3 = z3.permute(0, 2, 3, 1).reshape(B, -1, self.out_dim)
        z4 = z4.permute(0, 2, 3, 1).reshape(B, -1, self.out_dim)
        z5 = z5.permute(0, 2, 3, 1).reshape(B, -1, self.out_dim)
        
        # Concatenate all anchors: (B, N_total, out_dim)
        z = torch.cat([z3, z4, z5], dim=1)
        
        # Project to Poincare ball
        z_hyp = self.to_poincare(z)
        
        return z_hyp
    
    def compute_distances(self, embeddings, prototype_indices=None):
        """
        Compute distances from embeddings to prototypes.
        
        Parameters
        ----------
        embeddings : tensor
            Hyperbolic embeddings of shape (N, out_dim) or (B, N, out_dim)
        prototype_indices : tensor, optional
            If provided, only compute distance to these prototypes
        
        Returns
        -------
        tensor
            Distance matrix of shape (N, K) or (B, N, K)
        """
        protos = self.prototypes  # (K, out_dim)
        
        if prototype_indices is not None:
            protos = protos[prototype_indices]
        
        # Handle batched input
        if embeddings.dim() == 3:
            B, N, D = embeddings.shape
            embeddings_flat = embeddings.reshape(B * N, D)
            distances = pmath.dist_matrix(embeddings_flat, protos, c=self.c)
            distances = distances.reshape(B, N, -1)
        else:
            distances = pmath.dist_matrix(embeddings, protos, c=self.c)
        
        return distances
    
    def get_ood_scores(self, embeddings):
        """
        Compute OOD scores (minimum distance to any prototype).
        
        Parameters
        ----------
        embeddings : tensor
            Hyperbolic embeddings
        
        Returns
        -------
        tensor
            OOD scores (higher = more likely OOD)
        """
        distances = self.compute_distances(embeddings)  # (..., K)
        min_distances = distances.min(dim=-1).values
        return min_distances
    
    def extra_repr(self):
        return (f"in_dims={self.in_dims}, out_dim={self.out_dim}, "
                f"c={self.c}, num_classes={self.num_classes}, clip_r={self.clip_r}")


class HyperbolicContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss in hyperbolic space with optional prototype separation.
    
    Components:
    1. Contrastive loss: Push embeddings toward their prototype (STANDARD - always on)
    2. Separation loss: Push prototypes apart (EXPERIMENTAL - off by default)
    3. Boundary loss: Push prototypes toward boundary (EXPERIMENTAL - off by default)
    
    NOTE: The separation and boundary losses are heuristic additions, not from literature.
    Use with caution. Set weights to 0 to disable (default).
    
    Parameters
    ----------
    temperature : float
        Temperature for softmax (default: 0.1)
    curvature : float
        Poincare ball curvature (default: 0.1)
    separation_weight : float
        Weight for inter-prototype separation loss (default: 0.0 = disabled)
    boundary_weight : float
        Weight for boundary push loss (default: 0.0 = disabled)
    min_proto_dist : float
        Minimum desired distance between prototypes (default: 2.0)
    target_norm : float
        Target norm for prototypes (push toward boundary) (default: 0.9)
    """
    
    def __init__(self, temperature=0.1, curvature=0.1, 
                 separation_weight=0.0, boundary_weight=0.0,
                 min_proto_dist=2.0, target_norm=0.9):
        super(HyperbolicContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.c = curvature
        self.separation_weight = separation_weight
        self.boundary_weight = boundary_weight
        self.min_proto_dist = min_proto_dist
        self.target_norm = target_norm
    
    def prototype_separation_loss(self, prototypes):
        """
        Loss to push prototypes apart from each other.
        
        Uses hinge loss: penalize if inter-prototype distance < min_proto_dist
        """
        K = prototypes.shape[0]
        if K < 2:
            return prototypes.new_tensor(0.0)
        
        # Compute pairwise distances between all prototypes
        proto_dists = pmath.dist_matrix(prototypes, prototypes, c=self.c)  # (K, K)
        
        # Get upper triangular (unique pairs), exclude diagonal
        mask = torch.triu(torch.ones(K, K, device=prototypes.device), diagonal=1).bool()
        pairwise_dists = proto_dists[mask]  # (K*(K-1)/2,)
        
        # Hinge loss: penalize if distance < min_proto_dist
        # loss = max(0, min_dist - actual_dist)
        separation_loss = F.relu(self.min_proto_dist - pairwise_dists).mean()
        
        return separation_loss
    
    def boundary_push_loss(self, prototypes):
        """
        Loss to push prototypes toward the boundary of the Poincaré ball.
        
        Penalizes prototypes with low norm (near origin).
        The target_norm is interpreted as a FRACTION of the ball radius.
        For c=0.1, ball radius = 1/sqrt(c) ≈ 3.16
        So target_norm=0.9 means push to 90% of boundary = 2.85
        """
        # Compute norms
        norms = prototypes.norm(dim=-1)  # (K,)
        
        # Max norm in Poincaré ball with curvature c is 1/sqrt(c)
        ball_radius = 1.0 / (self.c ** 0.5)  # e.g., 3.16 for c=0.1
        
        # target_norm is a fraction (0-1), convert to absolute target
        absolute_target = self.target_norm * ball_radius * 0.95  # 0.95 for numerical safety
        
        # Penalize if norm < absolute_target
        boundary_loss = F.relu(absolute_target - norms).mean()
        
        return boundary_loss
    
    def forward(self, embeddings, labels, prototypes):
        """
        Compute hyperbolic contrastive loss with prototype separation.
        
        Parameters
        ----------
        embeddings : tensor
            Hyperbolic embeddings of shape (B, N, D) or (N, D)
        labels : tensor
            Class labels of shape (B, N) or (N), -1 for ignore
        prototypes : tensor
            Class prototypes of shape (K, D)
        
        Returns
        -------
        tensor
            Scalar loss value (contrastive + separation + boundary)
        """
        # Flatten if batched
        if embeddings.dim() == 3:
            B, N, D = embeddings.shape
            embeddings = embeddings.reshape(B * N, D)
            labels = labels.reshape(B * N)
        
        # Filter valid samples (ignore background with label=-1)
        valid_mask = labels >= 0
        
        # === Component 1: Contrastive Loss ===
        if valid_mask.sum() == 0:
            contrastive_loss = embeddings.new_tensor(0.0)
        else:
            valid_embeddings = embeddings[valid_mask]  # (N_valid, D)
            valid_labels = labels[valid_mask]          # (N_valid,)
            
            # Safety check: ensure labels are in valid range
            num_protos = prototypes.shape[0]
            if valid_labels.max() >= num_protos:
                raise ValueError(f"Label {valid_labels.max()} >= num_prototypes {num_protos}. "
                               f"Check num_classes in HyperbolicProjector init.")
            
            # Compute distances to all prototypes
            distances = pmath.dist_matrix(valid_embeddings, prototypes, c=self.c)  # (N_valid, K)
            
            # Convert to logits: negative distance / temperature
            logits = -distances / self.temperature
            
            # Cross-entropy loss
            contrastive_loss = F.cross_entropy(logits, valid_labels)
        
        # === Component 2: Prototype Separation Loss ===
        separation_loss = self.prototype_separation_loss(prototypes)
        
        # === Component 3: Boundary Push Loss ===
        boundary_loss = self.boundary_push_loss(prototypes)
        
        # Total loss
        total_loss = (contrastive_loss + 
                      self.separation_weight * separation_loss +
                      self.boundary_weight * boundary_loss)
        
        return total_loss
    
    def forward_with_breakdown(self, embeddings, labels, prototypes):
        """
        Same as forward() but returns loss breakdown for logging.
        
        Returns
        -------
        total_loss : tensor
        loss_dict : dict with individual loss components
        """
        if embeddings.dim() == 3:
            B, N, D = embeddings.shape
            embeddings = embeddings.reshape(B * N, D)
            labels = labels.reshape(B * N)
        
        valid_mask = labels >= 0
        
        if valid_mask.sum() == 0:
            contrastive_loss = embeddings.new_tensor(0.0)
        else:
            valid_embeddings = embeddings[valid_mask]
            valid_labels = labels[valid_mask]
            num_protos = prototypes.shape[0]
            if valid_labels.max() >= num_protos:
                raise ValueError(f"Label {valid_labels.max()} >= num_prototypes {num_protos}")
            distances = pmath.dist_matrix(valid_embeddings, prototypes, c=self.c)
            logits = -distances / self.temperature
            contrastive_loss = F.cross_entropy(logits, valid_labels)
        
        separation_loss = self.prototype_separation_loss(prototypes)
        boundary_loss = self.boundary_push_loss(prototypes)
        
        total_loss = (contrastive_loss + 
                      self.separation_weight * separation_loss +
                      self.boundary_weight * boundary_loss)
        
        # Also compute prototype stats for logging
        proto_norms = prototypes.norm(dim=-1)
        proto_dists = pmath.dist_matrix(prototypes, prototypes, c=self.c)
        K = prototypes.shape[0]
        mask = torch.triu(torch.ones(K, K, device=prototypes.device), diagonal=1).bool()
        
        loss_dict = {
            'contrastive': contrastive_loss.item(),
            'separation': separation_loss.item(),
            'boundary': boundary_loss.item(),
            'proto_norm_mean': proto_norms.mean().item(),
            'proto_norm_min': proto_norms.min().item(),
            'proto_dist_mean': proto_dists[mask].mean().item() if K > 1 else 0.0,
            'proto_dist_min': proto_dists[mask].min().item() if K > 1 else 0.0,
        }
        
        return total_loss, loss_dict


def test_hyperbolic_projector():
    """Quick test to verify shapes and gradients."""
    print("Testing HyperbolicProjector...")
    
    # Create projector
    projector = HyperbolicProjector(
        in_dims=[384, 768, 768],
        out_dim=256,
        curvature=0.1,
        num_classes=10,
        clip_r=2.3
    )
    
    # Create dummy FPN features
    B = 2
    p3 = torch.randn(B, 384, 80, 80)
    p4 = torch.randn(B, 768, 40, 40)
    p5 = torch.randn(B, 768, 20, 20)
    img_feats = (p3, p4, p5)
    
    # Forward pass
    z_hyp = projector(img_feats)
    
    print(f"  Input shapes: P3={p3.shape}, P4={p4.shape}, P5={p5.shape}")
    print(f"  Output shape: {z_hyp.shape}")
    print(f"  Expected: (B, 8400, 256) = ({B}, {80*80 + 40*40 + 20*20}, 256)")
    
    # Check norms (should be < 1/sqrt(c) ≈ 3.16 for c=0.1)
    norms = z_hyp.norm(dim=-1)
    max_norm = 1.0 / (0.1 ** 0.5)
    print(f"  Embedding norms: min={norms.min():.4f}, max={norms.max():.4f}, bound={max_norm:.4f}")
    
    # Check prototypes
    protos = projector.prototypes
    print(f"  Prototype shape: {protos.shape}")
    proto_norms = protos.norm(dim=-1)
    print(f"  Prototype norms: min={proto_norms.min():.4f}, max={proto_norms.max():.4f}")
    
    # Test distance computation
    distances = projector.compute_distances(z_hyp)
    print(f"  Distance matrix shape: {distances.shape}")
    
    # Test gradient flow
    loss = z_hyp.sum()
    loss.backward()
    print(f"  Gradient flow: OK (prototype_tangent.grad norm: {projector.prototype_tangent.grad.norm():.4f})")
    
    print("HyperbolicProjector test PASSED!\n")
    return projector


def test_contrastive_loss():
    """Test hyperbolic contrastive loss."""
    print("Testing HyperbolicContrastiveLoss...")
    
    loss_fn = HyperbolicContrastiveLoss(temperature=0.1, curvature=0.1)
    
    # Create dummy data
    B, N, D, K = 2, 100, 256, 10
    embeddings = torch.randn(B, N, D) * 0.1  # Small to be in ball
    embeddings = pmath.expmap0(embeddings, c=0.1)  # Project to ball
    labels = torch.randint(-1, K, (B, N))  # -1 to K-1
    prototypes = pmath.expmap0(torch.randn(K, D) * 0.01, c=0.1)
    
    # Forward
    loss = loss_fn(embeddings, labels, prototypes)
    print(f"  Loss value: {loss.item():.4f}")
    
    # Backward
    loss.backward()
    print(f"  Gradient flow: OK")
    
    print("HyperbolicContrastiveLoss test PASSED!\n")


if __name__ == "__main__":
    test_hyperbolic_projector()
    test_contrastive_loss()
