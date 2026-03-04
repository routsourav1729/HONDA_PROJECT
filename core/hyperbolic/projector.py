"""
Hyperbolic Projector with Horospherical Classifier — V2.

Projects visual features from YOLO backbone to Poincaré ball,
then classifies using Busemann function with ideal prototypes on boundary.

V2 changes:
- embed_dim 256 → 64 (14 classes only need d >= K-1 = 13)
- Prototypes trainable on unit sphere via Riemannian gradient (manual projection)
- HorosphericalLossV2: class-balanced CE + margin + geodesic pull + dispersion + bias reg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import ToPoincare
from . import pmath


class HorosphericalClassifier(nn.Module):
    """
    Classification via Busemann function with ideal prototypes on boundary.
    
    Score: ξ_k(x) = -B_{p_k}(x) + a_k
    Prediction: argmax_k softmax(ξ_k(x))
    
    For OOD detection: max_k ξ_k(x) < threshold → unknown
    
    V2: Prototypes are now TRAINABLE nn.Parameters that stay on the unit sphere
    via manual Riemannian projection after each optimizer step.
    
    Parameters
    ----------
    num_classes : int
        Number of known classes
    embed_dim : int
        Embedding dimension
    curvature : float
        Poincaré ball curvature (default: 1.0)
    init_directions : tensor, optional
        Pre-computed prototype directions from init_prototypes.py (K, embed_dim)
    trainable_prototypes : bool
        Whether prototype directions are trainable (default: True for V2)
    """
    
    def __init__(self, num_classes, embed_dim, curvature=1.0, init_directions=None,
                 trainable_prototypes=True):
        super().__init__()
        self.c = curvature
        self.R = 1.0 / (curvature ** 0.5)  # Ball radius
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.trainable_prototypes = trainable_prototypes
        
        if init_directions is not None:
            assert init_directions.shape == (num_classes, embed_dim), \
                f"init_directions shape mismatch: {init_directions.shape} vs ({num_classes}, {embed_dim})"
            directions = F.normalize(init_directions.clone(), dim=-1)
        else:
            # Placeholder — will be overwritten by load_hyp_ckpt()
            directions = F.normalize(torch.randn(num_classes, embed_dim), dim=-1)
        
        if trainable_prototypes:
            # TRAINABLE on unit sphere. After each optimizer.step(), call
            # project_prototypes_to_sphere() to re-normalize.
            self.prototype_direction = nn.Parameter(directions)
            print(f"  ✓ Prototype directions TRAINABLE on S^{embed_dim-1} ({num_classes} classes)")
        else:
            # FROZEN (v1 behavior)
            self.register_buffer('prototype_direction', directions)
            print(f"  ✓ Prototype directions FROZEN from pre-computed tensor (not trainable)")
        
        # Learnable bias per class - controls horosphere position
        self.prototype_bias = nn.Parameter(torch.zeros(num_classes))
    
    @torch.no_grad()
    def project_prototypes_to_sphere(self):
        """Project prototype directions back to unit sphere after optimizer step.
        
        This implements the Riemannian constraint: prototypes must lie on S^{d-1}.
        Call this after every optimizer.step() during training.
        """
        if self.trainable_prototypes and isinstance(self.prototype_direction, nn.Parameter):
            self.prototype_direction.data = F.normalize(self.prototype_direction.data, dim=-1)
    
    @property
    def prototypes(self):
        """Ideal prototypes on the Poincaré ball boundary (‖p‖ = R = 1/√c)."""
        direction = F.normalize(self.prototype_direction, dim=-1)
        return direction * self.R
    
    def busemann_scores(self, x):
        """
        Compute horospherical logits: ξ_k(x) = -B_{p_k}(x) + a_k
        
        Higher score = closer to prototype = more likely that class.
        
        Parameters
        ----------
        x : tensor (N, D) or (B, N, D)
            Points in Poincaré ball
        
        Returns
        -------
        tensor (N, K) or (B, N, K)
            Horospherical scores for each class
        """
        p = self.prototypes  # (K, D) on boundary
        
        # Handle batched input
        if x.dim() == 3:
            B_vals = pmath.busemann_batch(p, x, c=self.c)  # (B, N, K)
        else:
            B_vals = pmath.busemann(p, x, c=self.c)  # (N, K)
        
        # Score = -Busemann + bias (higher = closer to prototype)
        scores = -B_vals + self.prototype_bias
        return scores
    
    def forward(self, x):
        """Returns horospherical logits for classification."""
        return self.busemann_scores(x)
    
    def get_ood_scores(self, x):
        """
        OOD score: negative of max horosphere score.
        Higher value = more OOD (far from all prototypes).
        """
        scores = self.busemann_scores(x)
        max_scores = scores.max(dim=-1).values
        return -max_scores
    
    def uniform_dispersion_loss(self):
        """
        Uniform loss (Wang & Isola 2020 / Berg et al. IJCV 2025 Eq. 17).

        L_unif = log( mean_{i≠j} exp(-||p_i - p_j||^2) )

        Continuous gradient on ALL pairs; strongest for closest prototypes.
        """
        directions = F.normalize(self.prototype_direction, dim=-1)  # (K, D)
        K = directions.shape[0]
        if K <= 1:
            return directions.new_tensor(0.0)
        dist_sq = torch.cdist(directions, directions).pow(2)  # (K, K)
        mask = ~torch.eye(K, dtype=torch.bool, device=directions.device)
        return torch.log(torch.exp(-dist_sq[mask]).mean())


class HorosphericalLoss(nn.Module):
    """Backward-compatible alias for HorosphericalLossV2."""
    pass  # Kept for import compatibility — V2 is the actual class now


# =========================================================================
# V2 Loss Helper Functions
# =========================================================================

def compute_class_weights(labels, num_classes, smoothing=0.5):
    """
    Compute sqrt-inverse-frequency weights for class balancing.
    
    Args:
        labels: (N,) tensor of class indices for foreground anchors
        num_classes: int, total number of known classes
        smoothing: float, power for inverse frequency (0.5 = sqrt)
    
    Returns:
        weights: (num_classes,) tensor of per-class weights, normalized so mean=1
    """
    counts = torch.bincount(labels, minlength=num_classes).float().clamp(min=1.0)
    max_count = counts.max()
    weights = (max_count / counts) ** smoothing  # sqrt inverse frequency
    weights = weights / weights.mean()  # normalize so mean weight = 1
    return weights


def busemann_margin_loss(scores, labels, margin=1.0):
    """
    Busemann margin loss for horospherical classification.
    
    For each foreground anchor, enforce: ξ_{y_i}(x_i) - ξ_k(x_i) >= margin ∀ k ≠ y_i
    
    Args:
        scores: (N, K) Busemann-based horosphere scores
        labels: (N,) ground-truth class indices
        margin: float, required score gap
    
    Returns:
        loss: scalar, mean hinge loss over all foreground anchors
    """
    N, K = scores.shape
    if N == 0:
        return scores.sum() * 0.0
    
    # Get correct class score for each anchor: (N,)
    correct_scores = scores.gather(1, labels.unsqueeze(1)).squeeze(1)
    
    # margin - (correct_score - other_score) for each class
    margins = margin - correct_scores.unsqueeze(1) + scores  # (N, K)
    
    # Zero out the correct class (no self-comparison)
    mask = torch.ones_like(margins, dtype=torch.bool)
    mask.scatter_(1, labels.unsqueeze(1), False)
    
    # Hinge: max(0, margin - gap)
    violations = torch.clamp(margins[mask].view(N, K - 1), min=0.0)
    
    return violations.mean()


def geodesic_prototype_pull(embeddings, labels, prototypes, c=1.0, target_norm_fraction=0.85):
    """
    Pull embeddings toward target points near their class prototype.
    
    Instead of pulling to the boundary (which causes collapse), pull toward
    a point at target_norm_fraction * R in the prototype's direction.
    
    Args:
        embeddings: (N, D) points in Poincaré ball
        labels: (N,) class indices
        prototypes: (K, D) ideal prototypes on boundary (norm = R = 1/√c)
        c: float, curvature
        target_norm_fraction: float in (0, 1), how close to boundary the target is
    
    Returns:
        loss: scalar, mean geodesic distance to target points
    """
    if embeddings.shape[0] == 0:
        return embeddings.sum() * 0.0
    
    R = 1.0 / (c ** 0.5)  # ball radius
    
    # Target points: scale prototypes inward from boundary
    # prototypes have norm R, targets have norm target_norm_fraction * R
    target_points = prototypes[labels] * target_norm_fraction  # (N, D)
    
    # Geodesic distance in Poincaré ball
    # d_H(x,y) = acosh(1 + 2||x-y||² / ((1 - c||x||²)(1 - c||y||²)))
    diff_sq = (embeddings - target_points).pow(2).sum(dim=-1)  # (N,)
    emb_sq = (embeddings.pow(2).sum(dim=-1)).clamp(max=R**2 - 1e-5)
    tgt_sq = (target_points.pow(2).sum(dim=-1)).clamp(max=R**2 - 1e-5)
    
    denom = (1.0 - c * emb_sq) * (1.0 - c * tgt_sq)
    denom = denom.clamp(min=1e-8)
    
    argument = 1.0 + 2.0 * c * diff_sq / denom
    argument = argument.clamp(min=1.0 + 1e-7)  # numerical safety for acosh
    
    dist = torch.acosh(argument)
    
    return dist.mean()


class HorosphericalLossV2(nn.Module):
    """
    Horospherical loss V2 with 5 components:
    
    A) Class-balanced cross-entropy (sqrt-inverse-frequency weighting)
    B) Busemann margin loss (enforce gap between correct and other scores)
    C) Geodesic prototype pull (pull embeddings toward target in Poincaré ball)
    D) Direction dispersion loss (push prototypes apart on sphere) — kept from V1
    E) Bias L2 regularization (keep horospheres tight) — kept from V1
    
    Replaces V1's compactness loss with geodesic pull + margin loss.
    """
    
    def __init__(self, 
                 curvature=1.0,
                 # CE
                 ce_weight=1.0,
                 class_balance_smoothing=0.5,
                 # Margin
                 margin=1.0,
                 margin_weight=0.5,
                 # Geodesic pull
                 pull_weight=0.1,
                 target_norm_fraction=0.85,
                 # Dispersion (from V1)
                 dispersion_weight=0.1,
                 # Bias reg (from V1)
                 bias_reg_weight=0.1):
        super().__init__()
        self.c = curvature
        self.ce_weight = ce_weight
        self.class_balance_smoothing = class_balance_smoothing
        self.margin = margin
        self.margin_weight = margin_weight
        self.pull_weight = pull_weight
        self.target_norm_fraction = target_norm_fraction
        self.dispersion_weight = dispersion_weight
        self.bias_reg_weight = bias_reg_weight
    
    def direction_dispersion_loss(self, prototype_direction, frozen_directions=None):
        """
        Uniform loss (Wang & Isola 2020): continuous repulsion on ALL pairs.
        Includes cross-repulsion with frozen base directions at T2+.
        """
        dirs = F.normalize(prototype_direction, dim=-1)
        if frozen_directions is not None and frozen_directions.numel() > 0:
            all_dirs = torch.cat([dirs, F.normalize(frozen_directions, dim=-1)], dim=0)
        else:
            all_dirs = dirs
        K = all_dirs.shape[0]
        if K <= 1:
            return dirs.new_tensor(0.0)
        dist_sq = torch.cdist(all_dirs, all_dirs).pow(2)
        mask = ~torch.eye(K, dtype=torch.bool, device=dirs.device)
        return torch.log(torch.exp(-dist_sq[mask]).mean())
    
    def forward(self, embeddings, scores, labels, prototypes, biases,
                prototype_direction=None, frozen_directions=None):
        """
        Compute V2 loss.
        
        Args:
            embeddings: (M, D) Poincaré ball embeddings for ALL B×8400 anchors
            scores: (M, K) horosphere scores ξ_k(x) = -B_pk(x) + a_k
            labels: (M,) ground truth, -1 for background/ignore
            prototypes: (K, D) ideal prototypes on boundary
            biases: (K,) learnable bias terms
            prototype_direction: (K_train, D) trainable directions for dispersion
            frozen_directions: (K_frozen, D) frozen directions for cross-dispersion
        
        Returns:
            loss: scalar total loss
            loss_dict: dict of individual loss values for logging
        """
        # Flatten if batched
        if scores.dim() == 3:
            B, N, K = scores.shape
            scores = scores.reshape(B * N, K)
            labels = labels.reshape(B * N)
            embeddings = embeddings.reshape(B * N, embeddings.shape[-1])
        
        valid = labels >= 0
        num_classes = prototypes.shape[0]
        
        if valid.sum() == 0:
            zero = scores.sum() * 0.0
            return zero, {
                'ce_loss': 0.0, 'margin_loss': 0.0, 'pull_loss': 0.0,
                'disp_loss': 0.0, 'bias_reg': 0.0, 'total': 0.0,
            }
        
        fg_embeddings = embeddings[valid]   # (N, D)
        fg_scores = scores[valid]           # (N, K)
        fg_labels = labels[valid]           # (N,)
        
        # --- A: Class-balanced CE ---
        class_weights = compute_class_weights(fg_labels, num_classes, self.class_balance_smoothing)
        ce_loss = F.cross_entropy(fg_scores, fg_labels, weight=class_weights.to(fg_scores.device))
        
        # --- B: Busemann margin loss ---
        margin_loss = busemann_margin_loss(fg_scores, fg_labels, self.margin)
        
        # --- C: Geodesic prototype pull ---
        pull_loss = geodesic_prototype_pull(
            fg_embeddings, fg_labels, prototypes,
            c=self.c, target_norm_fraction=self.target_norm_fraction
        )
        
        # --- D: Dispersion ---
        disp_loss = scores.new_tensor(0.0)
        if prototype_direction is not None and self.dispersion_weight > 0:
            disp_loss = self.direction_dispersion_loss(prototype_direction, frozen_directions)
        
        # --- E: Bias regularization ---
        bias_reg = (biases ** 2).mean()
        
        # --- Total ---
        total = (self.ce_weight * ce_loss
                + self.margin_weight * margin_loss
                + self.pull_weight * pull_loss
                + self.dispersion_weight * disp_loss
                + self.bias_reg_weight * bias_reg)
        
        loss_dict = {
            'ce_loss': ce_loss.item(),
            'margin_loss': margin_loss.item(),
            'pull_loss': pull_loss.item(),
            'disp_loss': disp_loss.item(),
            'bias_reg': bias_reg.item(),
            'total': total.item(),
            'margin_violations': (busemann_margin_loss(fg_scores, fg_labels, self.margin) > 0).float().mean().item() if fg_scores.shape[0] > 0 else 0.0,
        }
        
        # Additional diagnostics
        with torch.no_grad():
            pos_scores = fg_scores.gather(1, fg_labels.unsqueeze(1)).squeeze(1)
            max_scores = fg_scores.max(dim=-1).values
            loss_dict['pos_score_mean'] = pos_scores.mean().item()
            loss_dict['pos_score_std'] = pos_scores.std().item() if len(pos_scores) > 1 else 0.0
            loss_dict['max_score_mean'] = max_scores.mean().item()
            loss_dict['score_margin'] = (max_scores - pos_scores).mean().item()
            loss_dict['emb_norm_mean'] = fg_embeddings.norm(dim=-1).mean().item()
            loss_dict['emb_norm_max'] = fg_embeddings.norm(dim=-1).max().item()
            loss_dict['emb_norm_min'] = fg_embeddings.norm(dim=-1).min().item()
        
        return total, loss_dict


class HyperbolicProjector(nn.Module):
    """
    Projects multi-scale FPN features to Poincaré ball with horospherical classification.
    
    Architecture:
    - 3 Conv projectors for FPN scales (P3, P4, P5)
    - ToPoincare layer for Euclidean → Poincaré mapping
    - HorosphericalClassifier with ideal prototypes on boundary
    
    Parameters
    ----------
    in_dims : list of int
        Input dims for each FPN scale [P3, P4, P5]. Default: [384, 768, 768]
    out_dim : int
        Output embedding dimension (default: 256)
    curvature : float
        Poincaré ball curvature (default: 1.0)
    num_classes : int
        Number of class prototypes
    clip_r : float
        Clip radius for ToPoincare. Must be < 1/√c (default: 0.95 for c=1)
    """
    
    def __init__(
        self,
        in_dims=[384, 768, 768],
        out_dim=64,
        curvature=1.0,
        num_classes=20,
        clip_r=0.95,
        riemannian=True,
        init_prototypes=None,
        trainable_prototypes=True
    ):
        super().__init__()
        
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.c = curvature
        self.num_classes = num_classes
        self.clip_r = clip_r
        
        # Per-scale two-conv projectors:
        #   Conv1: in_dim → in_dim (1×1) + BN + ReLU  (cross-channel mixing at full dim)
        #   Conv2: in_dim → out_dim (1×1)              (dimension reduction, linear)
        # This preserves full representational capacity before bottlenecking.
        # ~1.83M params. BatchNorm2d (not SyncBN) avoids NCCL stalls.
        self.proj_p3 = nn.Sequential(
            nn.Conv2d(in_dims[0], in_dims[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(in_dims[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dims[0], out_dim, kernel_size=1, bias=False),
        )
        self.proj_p4 = nn.Sequential(
            nn.Conv2d(in_dims[1], in_dims[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(in_dims[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dims[1], out_dim, kernel_size=1, bias=False),
        )
        self.proj_p5 = nn.Sequential(
            nn.Conv2d(in_dims[2], in_dims[2], kernel_size=1, bias=False),
            nn.BatchNorm2d(in_dims[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dims[2], out_dim, kernel_size=1, bias=False),
        )
        
        # ToPoincare layer
        self.to_poincare = ToPoincare(
            c=curvature,
            train_c=False,
            train_x=False,
            ball_dim=out_dim,
            riemannian=riemannian,
            clip_r=clip_r
        )
        
        # Horospherical classifier
        # init_prototypes should come from init_prototypes.py!
        self.classifier = HorosphericalClassifier(
            num_classes=num_classes,
            embed_dim=out_dim,
            curvature=curvature,
            init_directions=init_prototypes,
            trainable_prototypes=trainable_prototypes
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize Conv weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Standard Kaiming init for all convs (no near-zero override)
        # Near-zero init caused cold-start trap: embeddings near origin
        # are equidistant from all boundary prototypes → no gradient signal
    
    @property
    def prototypes(self):
        """Ideal prototypes on boundary (for compatibility)."""
        return self.classifier.prototypes
    
    @property
    def prototype_direction(self):
        """Access to prototype directions (for checkpoint saving)."""
        return self.classifier.prototype_direction
    
    @property
    def prototype_bias(self):
        """Access to prototype biases (for checkpoint saving)."""
        return self.classifier.prototype_bias
    
    def forward(self, img_feats):
        """
        Project FPN features to Poincaré ball.
        
        Parameters
        ----------
        img_feats : tuple of tensor
            FPN features (P3, P4, P5)
        
        Returns
        -------
        tensor (B, N_anchors, out_dim)
            Hyperbolic embeddings in Poincaré ball
        """
        p3, p4, p5 = img_feats[0], img_feats[1], img_feats[2]
        B = p3.shape[0]
        
        # Project each scale
        z3 = self.proj_p3(p3).permute(0, 2, 3, 1).reshape(B, -1, self.out_dim)
        z4 = self.proj_p4(p4).permute(0, 2, 3, 1).reshape(B, -1, self.out_dim)
        z5 = self.proj_p5(p5).permute(0, 2, 3, 1).reshape(B, -1, self.out_dim)
        
        # Concatenate and project to Poincaré ball
        z = torch.cat([z3, z4, z5], dim=1)
        return self.to_poincare(z)
    
    def compute_scores(self, embeddings):
        """Compute horospherical classification scores."""
        return self.classifier(embeddings)
    
    def get_ood_scores(self, embeddings):
        """OOD scores via negative max horosphere score."""
        return self.classifier.get_ood_scores(embeddings)
    
    def extra_repr(self):
        return (f"in_dims={self.in_dims}, out_dim={self.out_dim}, "
                f"c={self.c}, num_classes={self.num_classes}")


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_horospherical():
    """Quick test to verify shapes and gradients."""
    print("Testing HorosphericalClassifier + HyperbolicProjector...")
    
    # Create projector with c=1.0
    projector = HyperbolicProjector(
        in_dims=[384, 768, 768],
        out_dim=256,
        curvature=1.0,
        num_classes=10,
        clip_r=0.95
    )
    
    # Dummy FPN features
    B = 2
    p3 = torch.randn(B, 384, 80, 80)
    p4 = torch.randn(B, 768, 40, 40)
    p5 = torch.randn(B, 768, 20, 20)
    
    # Forward
    z_hyp = projector((p3, p4, p5))
    print(f"  Output shape: {z_hyp.shape} (expected: {(B, 8400, 256)})")
    
    # Check embeddings are inside ball (‖x‖ < 1 for c=1)
    norms = z_hyp.norm(dim=-1)
    print(f"  Embedding norms: min={norms.min():.4f}, max={norms.max():.4f}, bound=1.0")
    
    # Check prototypes are on boundary (‖p‖ = 1 for c=1)
    proto_norms = projector.prototypes.norm(dim=-1)
    print(f"  Prototype norms: {proto_norms.mean():.4f} (should be ~1.0)")
    
    # Compute scores
    scores = projector.compute_scores(z_hyp)
    print(f"  Scores shape: {scores.shape} (expected: {(B, 8400, 10)})")
    
    # Test loss
    labels = torch.randint(-1, 10, (B, 8400))
    loss_fn = HorosphericalLoss(curvature=1.0)
    loss = loss_fn(scores, labels)
    print(f"  Loss: {loss.item():.4f}")
    
    # Test gradient flow
    loss.backward()
    print(f"  Gradient OK: direction={projector.prototype_direction.grad.norm():.4f}, "
          f"bias={projector.prototype_bias.grad.norm():.4f}")
    
    print("✓ All tests passed!\n")


if __name__ == "__main__":
    test_horospherical()
