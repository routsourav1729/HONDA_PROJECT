"""
Hyperbolic Projector with Geodesic Prototypical Classifier.

Projects visual features from YOLO backbone to Poincare ball,
then classifies using geodesic distance to interior prototypes.

Replaces the Horospherical (Busemann) framework:
- Prototypes are interior points (||z|| < R), NOT on boundary
- Classification: softmax over -d^2_B(x, z_k)
- OOD detection: min_k d^2_B(x, z_k) -- far from all protos = unknown
- L_reg on pre-expmap norms prevents tanh saturation
- L_sep enforces minimum geodesic separation between prototypes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import ToPoincare
from . import pmath


# =========================================================================
# Geodesic Prototype Classifier
# =========================================================================

class GeodesicPrototypeClassifier(nn.Module):
    """
    Classification via geodesic distance to interior prototypes in Poincare ball.

    Score: s_k(x) = -d^2_B(x, z_k)
    Prediction: argmax_k s_k(x)
    OOD: min_k d^2_B(x, z_k) > threshold -> unknown

    Parameters
    ----------
    num_classes : int
        Number of known classes
    embed_dim : int
        Embedding dimension
    curvature : float
        Poincare ball curvature (default: 1.0)
    init_directions : tensor, optional
        Pre-computed prototype directions from init_prototypes.py (K, embed_dim).
        Will be scaled to prototype_init_norm inside the ball.
    prototype_init_norm : float
        Initial Poincare norm for prototypes (default: 0.4)
    trainable_prototypes : bool
        Whether prototypes are trainable (default: True)
    """

    def __init__(self, num_classes, embed_dim, curvature=1.0, init_directions=None,
                 trainable_prototypes=True, prototype_init_norm=0.4,
                 max_proto_norm=0.5):
        super().__init__()
        self.c = curvature
        self.R = 1.0 / (curvature ** 0.5)  # Ball radius
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.trainable_prototypes = trainable_prototypes
        # Hard ceiling on prototype norm — prevents optimizer from pushing
        # prototypes toward boundary (which would inflate geodesic distances
        # and kill OOD discrimination, same problem as boundary prototypes).
        self.max_proto_norm = min(max_proto_norm, self.R * 0.95)

        # Initialize prototypes INSIDE the ball (not on boundary)
        init_norm = min(prototype_init_norm, self.R * 0.95)
        if init_directions is not None:
            assert init_directions.shape == (num_classes, embed_dim), \
                f"init_directions shape mismatch: {init_directions.shape} vs ({num_classes}, {embed_dim})"
            directions = F.normalize(init_directions.clone(), dim=-1)
            protos = directions * init_norm
        else:
            directions = F.normalize(torch.randn(num_classes, embed_dim), dim=-1)
            protos = directions * init_norm

        if trainable_prototypes:
            self.prototypes = nn.Parameter(protos)
            print(f"  + Geodesic prototypes TRAINABLE inside ball "
                  f"(||z||={init_norm:.3f}, {num_classes} classes)")
        else:
            self.register_buffer('prototypes', protos)
            print(f"  + Geodesic prototypes FROZEN (||z||={init_norm:.3f}, {num_classes} classes)")

    @torch.no_grad()
    def project_prototypes_to_ball(self):
        """Clamp prototype norms to max_proto_norm after optimizer step.

        Without this, the optimizer pushes prototypes to ||z||~0.999 (the
        pmath.project limit) to maximize geodesic-distance discrimination.
        That recreates the boundary-prototype problem we're trying to avoid.
        """
        if self.trainable_prototypes and isinstance(self.prototypes, nn.Parameter):
            norms = self.prototypes.data.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            # Clamp to max_proto_norm (e.g., 0.5), not the ball boundary (0.999)
            scale = torch.clamp(self.max_proto_norm / norms, max=1.0)
            self.prototypes.data = self.prototypes.data * scale

    def _pairwise_dist(self, x, protos):
        """Geodesic distance from each point to each prototype.

        x: (N, D), protos: (K, D) -> output: (N, K)
        """
        N, D = x.shape
        K = protos.shape[0]
        x_exp = x.unsqueeze(1).expand(N, K, D)      # (N, K, D)
        p_exp = protos.unsqueeze(0).expand(N, K, D)  # (N, K, D)
        dists = pmath.dist(x_exp, p_exp, c=self.c)   # (N, K)
        return dists

    def geodesic_distances(self, x):
        """
        Compute geodesic distances from embeddings to all prototypes.

        Parameters
        ----------
        x : tensor (N, D) or (B, N, D)
            Points in Poincare ball

        Returns
        -------
        tensor (N, K) or (B, N, K)
            Geodesic distances d_B(x, z_k) for each class
        """
        protos = pmath.project(self.prototypes, c=self.c)  # (K, D), safe

        if x.dim() == 3:
            B, N, D = x.shape
            K = protos.shape[0]
            x_flat = x.reshape(B * N, D)
            dists = self._pairwise_dist(x_flat, protos)
            return dists.reshape(B, N, K)
        else:
            return self._pairwise_dist(x, protos)

    def geodesic_scores(self, x):
        """
        Compute classification logits: s_k(x) = -d^2_B(x, z_k)

        Higher score = closer to prototype = more likely that class.

        Parameters
        ----------
        x : tensor (N, D) or (B, N, D)
            Points in Poincare ball

        Returns
        -------
        tensor (N, K) or (B, N, K)
            Classification scores for each class
        """
        dists = self.geodesic_distances(x)
        return -dists.pow(2)

    def forward(self, x):
        """Returns geodesic classification logits."""
        return self.geodesic_scores(x)

    def get_ood_scores(self, x):
        """
        OOD score: minimum geodesic distance squared to any prototype.
        Higher value = more OOD (far from all prototypes).
        """
        dists_sq = self.geodesic_distances(x).pow(2)
        min_dist_sq = dists_sq.min(dim=-1).values
        return min_dist_sq


# =========================================================================
# Geodesic Prototype Loss
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
    weights = (max_count / counts) ** smoothing
    weights = weights / weights.mean()
    return weights


class GeodesicPrototypeLoss(nn.Module):
    """
    Geodesic prototype loss with 3 components:

    A) Class-balanced cross-entropy over -d^2_B scores
    B) L_reg: norm regularization on pre-expmap Euclidean features
    C) L_sep: prototype separation loss (minimum geodesic distance)

    Total: L = ce_weight * L_cls + beta_reg * L_reg + lambda_sep * L_sep
    """

    def __init__(self,
                 curvature=1.0,
                 ce_weight=1.0,
                 class_balance_smoothing=0.5,
                 beta_reg=0.1,
                 lambda_sep=1.0,
                 sep_margin=1.0,
                 score_scale=0.1):
        super().__init__()
        self.c = curvature
        self.ce_weight = ce_weight
        self.class_balance_smoothing = class_balance_smoothing
        self.beta_reg = beta_reg
        self.lambda_sep = lambda_sep
        self.sep_margin = sep_margin
        self.score_scale = score_scale  # divides -d^2 scores before CE to avoid cold-start saturation

    def prototype_separation_loss(self, prototypes):
        """
        L_sep = mean_{i!=j} max(0, m - d_B(z_i, z_j))

        Forces prototypes to be at least sep_margin apart in geodesic distance.
        """
        K = prototypes.shape[0]
        if K <= 1:
            return prototypes.new_tensor(0.0)

        protos = pmath.project(prototypes, c=self.c)

        # Pairwise geodesic distances
        p1 = protos.unsqueeze(1).expand(K, K, -1)
        p2 = protos.unsqueeze(0).expand(K, K, -1)
        pair_dists = pmath.dist(p1, p2, c=self.c)  # (K, K)

        mask = ~torch.eye(K, dtype=torch.bool, device=prototypes.device)
        violations = torch.clamp(self.sep_margin - pair_dists[mask], min=0.0)

        return violations.mean()

    def forward(self, embeddings, scores, labels, prototypes,
                pre_expmap_norms=None):
        """
        Compute geodesic prototype loss.

        Args:
            embeddings: (M, D) or (B, N, D) Poincare ball embeddings
            scores: (M, K) or (B, N, K) geodesic scores = -d^2_B(x, z_k)
            labels: (M,) or (B, N) ground truth, -1 for background/ignore
            prototypes: (K, D) interior prototypes in Poincare ball
            pre_expmap_norms: (M,) or (B, N) Euclidean norms BEFORE expmap0

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

        if pre_expmap_norms is not None and pre_expmap_norms.dim() == 2:
            pre_expmap_norms = pre_expmap_norms.reshape(-1)

        valid = labels >= 0
        num_classes = prototypes.shape[0]

        if valid.sum() == 0:
            zero = scores.sum() * 0.0
            return zero, {
                'ce_loss': 0.0, 'reg_loss': 0.0, 'sep_loss': 0.0,
                'total': 0.0,
            }

        fg_scores = scores[valid]
        fg_labels = labels[valid]
        fg_embeddings = embeddings[valid]

        # --- A: Class-balanced CE over geodesic scores ---
        class_weights = compute_class_weights(fg_labels, num_classes,
                                              self.class_balance_smoothing)
        # score_scale prevents cold-start saturation:
        # early embeddings near boundary -> large d^2 -> scores in [-25, -4]
        # dividing by score_scale brings them into a healthy softmax range
        ce_loss = F.cross_entropy(fg_scores * self.score_scale, fg_labels,
                                  weight=class_weights.to(fg_scores.device))

        # --- B: L_reg (pre-expmap norm regularization) ---
        # Regularize ALL anchors (FG + BG) to keep pre-expmap norms small
        if pre_expmap_norms is not None:
            reg_loss = pre_expmap_norms.pow(2).mean()
        else:
            reg_loss = scores.new_tensor(0.0)

        # --- C: L_sep (prototype separation) ---
        sep_loss = self.prototype_separation_loss(prototypes)

        # --- Total ---
        total = (self.ce_weight * ce_loss
                 + self.beta_reg * reg_loss
                 + self.lambda_sep * sep_loss)

        loss_dict = {
            'ce_loss': ce_loss.item(),
            'reg_loss': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
            'sep_loss': sep_loss.item(),
            'total': total.item(),
        }

        # Diagnostics
        with torch.no_grad():
            fg_dists_sq = -fg_scores  # scores = -d^2
            pos_dists_sq = fg_dists_sq.gather(1, fg_labels.unsqueeze(1)).squeeze(1)
            min_dists_sq = fg_dists_sq.min(dim=-1).values

            loss_dict['pos_dist_sq_mean'] = pos_dists_sq.mean().item()
            loss_dict['min_dist_sq_mean'] = min_dists_sq.mean().item()
            loss_dict['emb_poincare_norm_mean'] = fg_embeddings.norm(dim=-1).mean().item()
            loss_dict['emb_poincare_norm_max'] = fg_embeddings.norm(dim=-1).max().item()
            loss_dict['proto_norm_mean'] = prototypes.norm(dim=-1).mean().item()

            if pre_expmap_norms is not None:
                loss_dict['pre_expmap_norm_mean'] = pre_expmap_norms.mean().item()
                loss_dict['pre_expmap_norm_max'] = pre_expmap_norms.max().item()

            preds = fg_scores.argmax(dim=-1)
            loss_dict['cls_acc'] = (preds == fg_labels).float().mean().item()

        return total, loss_dict


# =========================================================================
# BiLipschitz Projector (unchanged)
# =========================================================================

class BiLipschitzProjector(nn.Module):
    """
    Distance-aware projector (SNGP-style, Liu et al. NeurIPS 2020).

    Spectral norm = upper Lipschitz bound (prevents catapulting OOD inputs)
    Residual skip = lower Lipschitz bound (preserves input distances)
    """
    def __init__(self, in_dim, out_dim, num_groups=16):
        super().__init__()
        from torch.nn.utils import spectral_norm
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(in_dim, in_dim, 1, bias=False)),
            nn.GroupNorm(min(num_groups, in_dim), in_dim),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(in_dim, out_dim, 1, bias=False)),
        )
        self.skip = spectral_norm(nn.Conv2d(in_dim, out_dim, 1, bias=False))

    def forward(self, x):
        return self.main(x) + self.skip(x)


# =========================================================================
# Hyperbolic Projector (updated for Geodesic framework)
# =========================================================================

class HyperbolicProjector(nn.Module):
    """
    Projects multi-scale FPN features to Poincare ball with geodesic classification.

    Architecture:
    - 3 Conv projectors for FPN scales (P3, P4, P5)
    - ToPoincare layer for Euclidean -> Poincare mapping
    - GeodesicPrototypeClassifier with interior prototypes
    """

    def __init__(
        self,
        in_dims=[384, 768, 768],
        out_dim=64,
        curvature=1.0,
        num_classes=20,
        clip_r=2.0,
        riemannian=True,
        init_prototypes=None,
        trainable_prototypes=True,
        bi_lipschitz=False,
        prototype_init_norm=0.4,
        max_proto_norm=0.5,
    ):
        super().__init__()

        self.in_dims = in_dims
        self.out_dim = out_dim
        self.c = curvature
        self.num_classes = num_classes
        self.clip_r = clip_r
        self.bi_lipschitz = bi_lipschitz

        if bi_lipschitz:
            print(f"  + Using BiLipschitz projectors (spectral norm + residual)")
            self.proj_p3 = BiLipschitzProjector(in_dims[0], out_dim)
            self.proj_p4 = BiLipschitzProjector(in_dims[1], out_dim)
            self.proj_p5 = BiLipschitzProjector(in_dims[2], out_dim)
        else:
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

        # ToPoincare -- clip_r is SAFETY net, L_reg keeps norms small
        self.to_poincare = ToPoincare(
            c=curvature,
            train_c=False,
            train_x=False,
            ball_dim=out_dim,
            riemannian=riemannian,
            clip_r=clip_r,
            tau_init=None  # No learnable tau -- L_reg handles norms
        )

        # Geodesic prototype classifier (interior prototypes)
        self.classifier = GeodesicPrototypeClassifier(
            num_classes=num_classes,
            embed_dim=out_dim,
            curvature=curvature,
            init_directions=init_prototypes,
            trainable_prototypes=trainable_prototypes,
            prototype_init_norm=prototype_init_norm,
            max_proto_norm=max_proto_norm,
        )

        if not bi_lipschitz:
            self._init_weights()

    def _init_weights(self):
        """Initialize Conv weights (only for non-BiLipschitz projectors)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, 'weight_orig'):
                    continue
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @property
    def prototypes(self):
        """Interior prototypes in Poincare ball."""
        return self.classifier.prototypes

    def forward(self, img_feats):
        """
        Project FPN features to Poincare ball.

        Returns
        -------
        poincare_embeddings : tensor (B, N_anchors, out_dim)
        pre_expmap_norms : tensor (B, N_anchors)
            Euclidean norms BEFORE expmap0 (for L_reg)
        """
        p3, p4, p5 = img_feats[0], img_feats[1], img_feats[2]
        B = p3.shape[0]

        z3 = self.proj_p3(p3).permute(0, 2, 3, 1).reshape(B, -1, self.out_dim)
        z4 = self.proj_p4(p4).permute(0, 2, 3, 1).reshape(B, -1, self.out_dim)
        z5 = self.proj_p5(p5).permute(0, 2, 3, 1).reshape(B, -1, self.out_dim)

        z = torch.cat([z3, z4, z5], dim=1)  # (B, 8400, out_dim)

        # Cache pre-expmap norms for L_reg
        pre_expmap_norms = z.norm(dim=-1)  # (B, 8400)

        # Diagnostic norms
        if getattr(self, 'store_norms', False):
            fpn3 = p3.permute(0, 2, 3, 1).reshape(B, -1, p3.shape[1])
            fpn4 = p4.permute(0, 2, 3, 1).reshape(B, -1, p4.shape[1])
            fpn5 = p5.permute(0, 2, 3, 1).reshape(B, -1, p5.shape[1])
            fpn_all = torch.cat([
                fpn3.norm(dim=-1),
                fpn4.norm(dim=-1),
                fpn5.norm(dim=-1),
            ], dim=1)
            self._cached_fpn_norms = fpn_all
            self._cached_pre_clip_norms = pre_expmap_norms

        poincare_embeddings = self.to_poincare(z)

        return poincare_embeddings, pre_expmap_norms

    def compute_scores(self, embeddings):
        """Compute geodesic classification scores."""
        return self.classifier(embeddings)

    def get_ood_scores(self, embeddings):
        """OOD scores: min geodesic distance squared to any prototype."""
        return self.classifier.get_ood_scores(embeddings)

    def extra_repr(self):
        return (f"in_dims={self.in_dims}, out_dim={self.out_dim}, "
                f"c={self.c}, num_classes={self.num_classes}")


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================
HorosphericalClassifier = GeodesicPrototypeClassifier
HorosphericalLoss = GeodesicPrototypeLoss
HorosphericalLossV2 = GeodesicPrototypeLoss


# =============================================================================
# TEST
# =============================================================================
def test_geodesic():
    """Quick test to verify shapes and gradients."""
    print("Testing GeodesicPrototypeClassifier + HyperbolicProjector...")

    projector = HyperbolicProjector(
        in_dims=[384, 768, 768],
        out_dim=64,
        curvature=1.0,
        num_classes=8,
        clip_r=2.0,
        bi_lipschitz=True,
        prototype_init_norm=0.4,
    )

    B = 2
    p3 = torch.randn(B, 384, 80, 80)
    p4 = torch.randn(B, 768, 40, 40)
    p5 = torch.randn(B, 768, 20, 20)

    z_hyp, pre_norms = projector((p3, p4, p5))
    print(f"  Poincare output: {z_hyp.shape}")
    print(f"  Pre-expmap norms: {pre_norms.shape}")

    scores = projector.compute_scores(z_hyp)
    print(f"  Scores shape: {scores.shape}")

    labels = torch.randint(-1, 8, (B, 8400))
    loss_fn = GeodesicPrototypeLoss(curvature=1.0)
    loss, loss_dict = loss_fn(z_hyp, scores, labels, projector.prototypes, pre_norms)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Breakdown: {loss_dict}")

    loss.backward()
    print("+ All tests passed!")


if __name__ == "__main__":
    test_geodesic()
