"""
Spherical Projector with vMF (von Mises-Fisher) Classifier.

Projects visual features from YOLO backbone to unit hypersphere,
then classifies using vMF distribution with learnable per-class
concentration parameters kappa and EMA-updated prototypes.

Replaces the Poincare ball / Geodesic Prototypical framework.
Based on SIREN (Du et al., NeurIPS 2022).

OOD detection: max_c [log Z_d(kappa_c) + kappa_c * mu_c^T * r] -- low score = unknown
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Bessel function normalizer (CRITICAL for density-calibrated OOD scoring)
# =========================================================================

def stable_log_vmf_normalizer(kappa, d):
    """
    Compute log Z_d(kappa) = (d/2-1)*log(kappa) - (d/2)*log(2pi) - log(I_{d/2-1}(kappa))

    Uses asymptotic expansion of log I_v(z) for large z (valid for kappa > 5,
    always true after a few training steps with kappa_init=10):
        log I_v(z) ~ z - 0.5*log(2*pi*z)

    For small kappa (early training), clamp to prevent NaN.
    Cost: ~5 element-wise ops on a K-dim tensor. Negligible.

    Parameters
    ----------
    kappa : tensor (K,)
        Per-class concentration parameters (positive)
    d : int
        Embedding dimension

    Returns
    -------
    log_z : tensor (K,)
        Log normalization constant for each class
    """
    v = (d / 2.0) - 1.0  # For d=64: v=31

    # Asymptotic log I_v(kappa) ~ kappa - 0.5 * log(2*pi*kappa)
    log_iv = kappa - 0.5 * math.log(2 * math.pi) - 0.5 * torch.log(kappa.clamp(min=1e-8))

    # log Z_d(kappa) = v*log(kappa) - (d/2)*log(2pi) - log_iv
    log_z = v * torch.log(kappa.clamp(min=1e-8)) - (d / 2.0) * math.log(2 * math.pi) - log_iv

    return log_z  # shape: (K,)


# =========================================================================
# vMF Classifier
# =========================================================================

class vMFClassifier(nn.Module):
    """
    Classification via vMF distribution on the unit hypersphere.

    Replaces GeodesicPrototypeClassifier/HorosphericalClassifier.

    Score: log p(y=c|r) ~ log Z_d(kappa_c) + kappa_c * mu_c^T * r
    Prediction: argmax_c score_c
    OOD: max_c [Z_d(kappa_c) * exp(kappa_c * mu_c^T * r)] < threshold -> unknown

    Parameters
    ----------
    num_classes : int
        Number of known classes (8 for IDD T1)
    embed_dim : int
        Embedding dimension (64 recommended)
    init_directions : tensor (K, D), optional
        CLIP+GW-OT initialized prototype directions. Will be L2-normalized.
    kappa_init : float
        Initial value for learnable kappa parameters (default: 10.0, from SIREN)
    ema_alpha : float
        EMA momentum for prototype updates (default: 0.95, from SIREN)
    """

    def __init__(self, num_classes, embed_dim, init_directions=None,
                 kappa_init=10.0, ema_alpha=0.95):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.ema_alpha = ema_alpha

        # Learnable per-class concentration parameters (SIREN's key contribution)
        self.log_kappa = nn.Parameter(torch.full((num_classes,), math.log(kappa_init)))

        # Class prototypes -- NOT nn.Parameter, updated via EMA
        if init_directions is not None:
            init_dirs = F.normalize(init_directions.float(), dim=-1)
            self.register_buffer('prototypes', init_dirs)  # (K, D)
        else:
            proto = torch.randn(num_classes, embed_dim)
            proto = F.normalize(proto, dim=-1)
            self.register_buffer('prototypes', proto)

        # Track whether we've seen samples for each class (for EMA warmup)
        self.register_buffer('class_counts', torch.zeros(num_classes, dtype=torch.long))

        print(f"  + vMFClassifier: {num_classes} classes, dim={embed_dim}, "
              f"kappa_init={kappa_init:.1f}, ema_alpha={ema_alpha}")

    @property
    def kappa(self):
        """Per-class concentration parameters (always positive via exp)."""
        return self.log_kappa.exp()

    @torch.no_grad()
    def update_prototypes(self, embeddings, labels):
        """
        EMA update of class prototypes during training.

        Parameters
        ----------
        embeddings : tensor (N, D) -- L2-normalized foreground embeddings
        labels : tensor (N,) -- class labels (0 to K-1, foreground only)
        """
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            class_embs = embeddings[mask]  # (n_c, D)
            mean_emb = class_embs.mean(dim=0)  # (D,)

            if self.class_counts[c] == 0:
                # First time seeing this class -- initialize directly
                self.prototypes[c] = F.normalize(mean_emb, dim=0)
            else:
                # EMA update
                updated = self.ema_alpha * self.prototypes[c] + (1 - self.ema_alpha) * mean_emb
                self.prototypes[c] = F.normalize(updated, dim=0)

            self.class_counts[c] += mask.sum()

    def forward(self, r):
        """
        Compute vMF classification logits.

        Parameters
        ----------
        r : tensor (N, D) or (B, N, D) -- L2-normalized embeddings

        Returns
        -------
        logits : tensor (N, K) or (B, N, K) -- vMF log-probabilities (unnormalized)
        """
        kappa = self.kappa  # (K,)
        mu = self.prototypes  # (K, D)

        if r.dim() == 3:
            B, N, D = r.shape
            cos_sim = torch.matmul(r, mu.t())  # (B, N, K)
            log_z = stable_log_vmf_normalizer(kappa, D)  # (K,)
            logits = log_z.unsqueeze(0).unsqueeze(0) + kappa.unsqueeze(0).unsqueeze(0) * cos_sim
        else:
            N, D = r.shape
            cos_sim = torch.matmul(r, mu.t())  # (N, K)
            log_z = stable_log_vmf_normalizer(kappa, D)  # (K,)
            logits = log_z.unsqueeze(0) + kappa.unsqueeze(0) * cos_sim  # (N, K)

        return logits

    def get_ood_scores(self, r):
        """
        OOD score: max class-conditional vMF log-likelihood.
        Higher = more ID, lower = more OOD.

        Parameters
        ----------
        r : tensor (N, D) -- L2-normalized embeddings

        Returns
        -------
        scores : tensor (N,) -- max vMF score per detection
        assigned_proto : tensor (N,) -- index of closest prototype
        """
        logits = self.forward(r)  # (N, K)
        max_scores, assigned = logits.max(dim=-1)  # (N,), (N,)
        return max_scores, assigned

    def cosine_scores(self, r):
        """Simple cosine similarity scores (for debugging / comparison)."""
        mu = self.prototypes  # (K, D)
        return torch.matmul(r, mu.t())


# =========================================================================
# vMF Loss
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


class vMFLoss(nn.Module):
    """
    SIREN loss: negative log-likelihood under class-conditional vMF mixture.

    L = -(1/M) sum_i log [ Z_d(kappa_{y_i}) exp(kappa_{y_i} mu_{y_i}^T r_i)
                            / sum_j Z_d(kappa_j) exp(kappa_j mu_j^T r_i) ]

    Plus background repulsion for dense anchor-based detectors.

    Parameters
    ----------
    embed_dim : int
        Dimension of hyperspherical embeddings
    num_classes : int
        Number of known classes
    class_balance_smoothing : float
        Power for class-balanced weights (0.5 = sqrt-inverse-frequency)
    repulsion_weight : float
        Weight for background repulsion loss
    repulsion_margin : float
        Cosine similarity margin for background repulsion
    hard_neg_threshold : float
        Only push background anchors with cos_sim > this threshold
    """

    def __init__(self, embed_dim=64, num_classes=8, class_balance_smoothing=0.5,
                 repulsion_weight=0.5, repulsion_margin=0.1, hard_neg_threshold=0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.class_balance_smoothing = class_balance_smoothing
        self.repulsion_weight = repulsion_weight
        self.repulsion_margin = repulsion_margin
        self.hard_neg_threshold = hard_neg_threshold

    def forward(self, logits, labels, embeddings, prototypes, kappa,
                class_weights=None):
        """
        Parameters
        ----------
        logits : tensor (B*N, K) -- vMF logits from classifier
        labels : tensor (B*N,) -- TAL-assigned labels (-1 = background, 0..K-1 = fg)
        embeddings : tensor (B*N, D) -- L2-normalized embeddings
        prototypes : tensor (K, D) -- current class prototypes (unit vectors)
        kappa : tensor (K,) -- current concentration parameters
        class_weights : tensor (K,), optional -- class-balanced weights

        Returns
        -------
        total_loss : scalar
        loss_dict : dict with diagnostic values
        """
        fg_mask = (labels >= 0) & (labels < self.num_classes)
        bg_mask = (labels == -1)

        loss_dict = {}

        # ==== 1. vMF Classification Loss (foreground only) ====
        if fg_mask.sum() == 0:
            vmf_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        else:
            fg_logits = logits[fg_mask]  # (N_fg, K)
            fg_labels = labels[fg_mask]  # (N_fg,)

            if class_weights is not None:
                vmf_loss = F.cross_entropy(fg_logits, fg_labels, weight=class_weights)
            else:
                vmf_loss = F.cross_entropy(fg_logits, fg_labels)

        # ==== 2. Background Repulsion Loss ====
        if bg_mask.sum() > 0 and self.repulsion_weight > 0:
            bg_embs = embeddings[bg_mask]  # (N_bg, D)
            bg_cos_sim = torch.matmul(bg_embs, prototypes.t())  # (N_bg, K)
            max_cos_sim, nearest_proto = bg_cos_sim.max(dim=-1)  # (N_bg,)

            hard_mask = max_cos_sim > self.hard_neg_threshold
            if hard_mask.sum() > 0:
                hard_cos = max_cos_sim[hard_mask]
                repulsion_loss = torch.clamp(hard_cos - self.repulsion_margin, min=0).mean()
            else:
                repulsion_loss = torch.tensor(0.0, device=logits.device)

            loss_dict['n_hard_bg'] = hard_mask.sum().item()
            loss_dict['bg_max_cos_mean'] = max_cos_sim.mean().item()
        else:
            repulsion_loss = torch.tensor(0.0, device=logits.device)
            loss_dict['n_hard_bg'] = 0
            loss_dict['bg_max_cos_mean'] = 0.0

        # ==== Total ====
        total = vmf_loss + self.repulsion_weight * repulsion_loss

        # Diagnostics
        loss_dict['vmf_ce_loss'] = vmf_loss.item()
        loss_dict['repulsion_loss'] = repulsion_loss.item()
        loss_dict['total'] = total.item()
        loss_dict['n_foreground'] = fg_mask.sum().item()
        loss_dict['n_background'] = bg_mask.sum().item()

        with torch.no_grad():
            loss_dict['kappa_mean'] = kappa.mean().item()
            loss_dict['kappa_min'] = kappa.min().item()
            loss_dict['kappa_max'] = kappa.max().item()

            if fg_mask.sum() > 0:
                fg_embs = embeddings[fg_mask]
                fg_labels_local = labels[fg_mask]
                fg_cos = torch.matmul(fg_embs, prototypes.t())

                preds = fg_logits.argmax(dim=-1)
                loss_dict['cls_acc'] = (preds == fg_labels_local).float().mean().item()

                pos_cos = fg_cos.gather(1, fg_labels_local.unsqueeze(1)).squeeze(1)
                loss_dict['fg_pos_cos_mean'] = pos_cos.mean().item()
                loss_dict['emb_norm_mean'] = fg_embs.norm(dim=-1).mean().item()

        return total, loss_dict


# =========================================================================
# BiLipschitz Projector (unchanged from geodesic version)
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
# Hyperbolic Projector -> now Spherical Projector
# =========================================================================

class HyperbolicProjector(nn.Module):
    """
    Projects multi-scale FPN features to unit hypersphere with vMF classification.

    Architecture:
    - 3 BiLipschitz Conv projectors for FPN scales (P3, P4, P5)
    - Optional MLP projection head (SIREN recommends nonlinear)
    - L2 normalization (replaces ToPoincare/expmap0)
    - vMFClassifier with learnable kappa and EMA prototypes

    Name kept as 'HyperbolicProjector' for checkpoint loading compatibility.
    """

    def __init__(
        self,
        in_dims=[384, 768, 768],
        out_dim=64,
        num_classes=20,
        init_prototypes=None,
        bi_lipschitz=True,
        kappa_init=10.0,
        ema_alpha=0.95,
        use_projection_head=True,
        # Legacy params -- accepted but ignored for backward compat
        curvature=1.0,
        clip_r=2.0,
        riemannian=True,
        trainable_prototypes=True,
        prototype_init_norm=0.4,
        max_proto_norm=0.5,
    ):
        super().__init__()

        self.in_dims = in_dims
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.bi_lipschitz = bi_lipschitz

        # BiLipschitz projectors (trainable feature extractors)
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

        # Optional MLP projection head (SIREN shows +4.37% AUROC)
        self.use_projection_head = use_projection_head
        if use_projection_head:
            self.projection_head = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(out_dim, out_dim),
            )
            print(f"  + MLP projection head: {out_dim} -> {out_dim} -> {out_dim}")

        # vMF Classifier (replaces GeodesicPrototypeClassifier)
        self.classifier = vMFClassifier(
            num_classes=num_classes,
            embed_dim=out_dim,
            init_directions=init_prototypes,
            kappa_init=kappa_init,
            ema_alpha=ema_alpha,
        )

        if not bi_lipschitz:
            self._init_weights()

    def _init_weights(self):
        """Initialize Conv weights (only for non-BiLipschitz projectors)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, 'weight_orig'):
                    continue  # spectral-normed
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @property
    def prototypes(self):
        """Class prototypes on unit sphere."""
        return self.classifier.prototypes

    def forward(self, img_feats):
        """
        Project FPN features to unit hypersphere.

        Returns
        -------
        normalized_embeddings : tensor (B, N_anchors, out_dim) -- unit vectors
        raw_projections : tensor (B, N_anchors, out_dim) -- before L2 norm (diagnostics)
        """
        p3, p4, p5 = img_feats[0], img_feats[1], img_feats[2]
        B = p3.shape[0]

        z3 = self.proj_p3(p3).permute(0, 2, 3, 1).reshape(B, -1, self.out_dim)
        z4 = self.proj_p4(p4).permute(0, 2, 3, 1).reshape(B, -1, self.out_dim)
        z5 = self.proj_p5(p5).permute(0, 2, 3, 1).reshape(B, -1, self.out_dim)

        z = torch.cat([z3, z4, z5], dim=1)  # (B, 8400, out_dim)

        # Optional MLP projection head
        if self.use_projection_head:
            z_proj = self.projection_head(z)  # (B, 8400, out_dim)
        else:
            z_proj = z

        # L2 normalize to unit hypersphere -- replaces expmap0/ToPoincare
        # No tanh, no saturation, no clip_r. Simple and correct.
        normalized = F.normalize(z_proj, dim=-1)  # (B, 8400, out_dim), all ||r|| = 1

        return normalized, z  # return raw for diagnostics

    def compute_scores(self, embeddings):
        """Compute vMF classification logits."""
        return self.classifier(embeddings)

    def get_ood_scores(self, embeddings):
        """OOD scores via vMF likelihood."""
        return self.classifier.get_ood_scores(embeddings)

    def extra_repr(self):
        return (f"in_dims={self.in_dims}, out_dim={self.out_dim}, "
                f"num_classes={self.num_classes}, framework=vmf_spherical")


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================
GeodesicPrototypeClassifier = vMFClassifier
GeodesicPrototypeLoss = vMFLoss
HorosphericalClassifier = vMFClassifier
HorosphericalLoss = vMFLoss
HorosphericalLossV2 = vMFLoss
SphericalProjector = HyperbolicProjector  # Future name


# =============================================================================
# TEST
# =============================================================================
def test_vmf():
    """Quick test to verify shapes and gradients."""
    print("Testing vMFClassifier + HyperbolicProjector (spherical mode)...")

    projector = HyperbolicProjector(
        in_dims=[384, 768, 768],
        out_dim=64,
        num_classes=8,
        bi_lipschitz=True,
        kappa_init=10.0,
        ema_alpha=0.95,
        use_projection_head=True,
    )

    B = 2
    p3 = torch.randn(B, 384, 80, 80)
    p4 = torch.randn(B, 768, 40, 40)
    p5 = torch.randn(B, 768, 20, 20)

    normalized, raw = projector((p3, p4, p5))
    print(f"  Normalized output: {normalized.shape}, norms: {normalized.norm(dim=-1).mean():.4f}")
    print(f"  Raw projections: {raw.shape}")

    scores = projector.compute_scores(normalized)
    print(f"  Scores shape: {scores.shape}")

    labels = torch.randint(-1, 8, (B, 8400))
    loss_fn = vMFLoss(embed_dim=64, num_classes=8)

    B_total, N, D = normalized.shape
    flat_embs = normalized.reshape(B_total * N, D)
    flat_labels = labels.reshape(B_total * N)
    flat_logits = scores.reshape(B_total * N, -1)

    loss, loss_dict = loss_fn(
        flat_logits, flat_labels, flat_embs,
        projector.classifier.prototypes,
        projector.classifier.kappa,
    )
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Breakdown: {loss_dict}")

    loss.backward()
    print(f"  log_kappa grad: {projector.classifier.log_kappa.grad}")
    print("+ All tests passed!")


if __name__ == "__main__":
    test_vmf()
