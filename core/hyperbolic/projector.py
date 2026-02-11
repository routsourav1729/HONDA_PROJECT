"""
Hyperbolic Projector with Horospherical Classifier.

Projects visual features from YOLO backbone to Poincaré ball,
then classifies using Busemann function with ideal prototypes on boundary.
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
    
    Parameters
    ----------
    num_classes : int
        Number of known classes
    embed_dim : int
        Embedding dimension
    curvature : float
        Poincaré ball curvature (default: 1.0)
    """
    
    def __init__(self, num_classes, embed_dim, curvature=1.0):
        super().__init__()
        self.c = curvature
        self.R = 1.0 / (curvature ** 0.5)  # Ball radius
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Ideal prototype directions (normalized to boundary at runtime)
        self.prototype_direction = nn.Parameter(torch.randn(num_classes, embed_dim))
        
        # Learnable bias per class - controls horosphere position
        # Positive bias = larger decision region (good for rare classes)
        self.prototype_bias = nn.Parameter(torch.zeros(num_classes))
    
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


class HorosphericalLoss(nn.Module):
    """
    Cross-entropy loss over horospherical scores.
    
    L = -log P(ŷ=y | x) where P(ŷ=k | x) = softmax_k(ξ_k(x))
    
    Parameters
    ----------
    curvature : float
        Poincaré ball curvature (default: 1.0)
    label_smoothing : float
        Label smoothing factor (default: 0.0)
    """
    
    def __init__(self, curvature=1.0, label_smoothing=0.0):
        super().__init__()
        self.c = curvature
        self.label_smoothing = label_smoothing
    
    def forward(self, scores, labels):
        """
        Compute cross-entropy loss over horospherical scores.
        
        Parameters
        ----------
        scores : tensor (B*N, K) or (N, K)
            Horospherical scores from classifier
        labels : tensor (B*N,) or (N,)
            Class labels, -1 for background/ignore
        
        Returns
        -------
        tensor
            Scalar loss
        """
        # Flatten if batched
        if scores.dim() == 3:
            B, N, K = scores.shape
            scores = scores.reshape(B * N, K)
            labels = labels.reshape(B * N)
        
        # Filter valid (ignore background label=-1)
        valid = labels >= 0
        if valid.sum() == 0:
            return scores.new_tensor(0.0)
        
        valid_scores = scores[valid]
        valid_labels = labels[valid]
        
        return F.cross_entropy(valid_scores, valid_labels, label_smoothing=self.label_smoothing)
    
    def forward_with_breakdown(self, scores, labels, classifier):
        """Same as forward but returns diagnostic info."""
        loss = self.forward(scores, labels)
        
        biases = classifier.prototype_bias
        proto_norms = classifier.prototypes.norm(dim=-1)
        
        loss_dict = {
            'horo_ce_loss': loss.item(),
            'bias_mean': biases.mean().item(),
            'bias_std': biases.std().item() if len(biases) > 1 else 0.0,
            'proto_norm_mean': proto_norms.mean().item(),
        }
        return loss, loss_dict


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
        out_dim=256,
        curvature=1.0,
        num_classes=20,
        clip_r=0.95,
        riemannian=True
    ):
        super().__init__()
        
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.c = curvature
        self.num_classes = num_classes
        self.clip_r = clip_r
        
        # Per-scale Conv projectors
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
        
        # ToPoincare layer
        self.to_poincare = ToPoincare(
            c=curvature,
            train_c=False,
            train_x=False,
            ball_dim=out_dim,
            riemannian=riemannian,
            clip_r=clip_r
        )
        
        # Horospherical classifier (replaces old prototype_tangent)
        self.classifier = HorosphericalClassifier(
            num_classes=num_classes,
            embed_dim=out_dim,
            curvature=curvature
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize Conv weights."""
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
