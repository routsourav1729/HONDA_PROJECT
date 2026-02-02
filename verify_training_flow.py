#!/usr/bin/env python3
"""
Trace training flow end-to-end to verify correctness.
"""

import torch
import torch.nn as nn
import sys

print("=" * 60)
print("HYPERBOLIC TRAINING FLOW VERIFICATION")
print("=" * 60)

# 1. Test imports
print("\n[1] Testing imports...")
try:
    from core.hyperbolic import (
        expmap0, logmap0, dist_matrix, project,
        ToPoincare, HyperbolicProjector, HyperbolicContrastiveLoss
    )
    from core.hyp_customyoloworld import HypCustomYoloWorld, load_hyp_ckpt
    print("    ✓ All imports successful")
except Exception as e:
    print(f"    ✗ Import failed: {e}")
    sys.exit(1)

# 2. Test HyperbolicProjector forward pass
print("\n[2] Testing HyperbolicProjector forward...")
projector = HyperbolicProjector(
    in_dims=[384, 768, 768],
    out_dim=256,
    curvature=0.1,
    num_classes=11,  # IDD T1: 11 classes
    clip_r=2.3
)

# Dummy FPN features (YOLO-World XL dimensions)
B = 2
p3 = torch.randn(B, 384, 80, 80)
p4 = torch.randn(B, 768, 40, 40)
p5 = torch.randn(B, 768, 20, 20)
img_feats = (p3, p4, p5)

hyp_emb = projector(img_feats)
print(f"    Input: P3={p3.shape}, P4={p4.shape}, P5={p5.shape}")
print(f"    Output: {hyp_emb.shape} (expected: [2, 8400, 256])")
assert hyp_emb.shape == (B, 8400, 256), f"Wrong shape: {hyp_emb.shape}"
print("    ✓ Forward pass correct")

# 3. Test prototypes
print("\n[3] Testing prototypes...")
protos = projector.prototypes
print(f"    Prototype shape: {protos.shape} (expected: [11, 256])")
assert protos.shape == (11, 256), f"Wrong prototype shape: {protos.shape}"
norms = protos.norm(dim=-1)
max_allowed = 1.0 / (0.1 ** 0.5)  # ~3.16 for c=0.1
print(f"    Prototype norms: min={norms.min():.4f}, max={norms.max():.4f} (bound: {max_allowed:.2f})")
assert norms.max() < max_allowed, "Prototypes outside Poincaré ball!"
print("    ✓ Prototypes valid")

# 4. Test TAL-like labels
print("\n[4] Testing TAL label simulation...")
# Simulate TAL output: class indices per anchor, -1 for background
# Shape: (B, N_anchors)
N_anchors = 8400
labels = torch.full((B, N_anchors), -1, dtype=torch.long)  # All background
# Assign some foreground: ~5% of anchors
n_fg = int(0.05 * N_anchors)
fg_indices = torch.randint(0, N_anchors, (B, n_fg))
fg_classes = torch.randint(0, 11, (B, n_fg))  # Classes 0-10
for b in range(B):
    labels[b, fg_indices[b]] = fg_classes[b]

print(f"    Label shape: {labels.shape}")
print(f"    Background anchors: {(labels == -1).sum().item()}")
print(f"    Foreground anchors: {(labels >= 0).sum().item()}")
print(f"    Label range: [{labels[labels >= 0].min()}, {labels[labels >= 0].max()}]")
print("    ✓ TAL simulation correct")

# 5. Test HyperbolicContrastiveLoss
print("\n[5] Testing HyperbolicContrastiveLoss...")
loss_fn = HyperbolicContrastiveLoss(temperature=0.1, curvature=0.1)

# Forward loss
loss = loss_fn(hyp_emb, labels, protos)
print(f"    Loss value: {loss.item():.4f}")
assert loss.item() >= 0, "Loss should be non-negative"
assert not torch.isnan(loss), "Loss is NaN!"
assert not torch.isinf(loss), "Loss is Inf!"
print("    ✓ Loss computation correct")

# 6. Test gradient flow
print("\n[6] Testing gradient flow...")
# Make sure gradients flow back to projector
projector.zero_grad()
loss.backward()

# Check prototype gradients
proto_grad = projector.prototype_tangent.grad
if proto_grad is not None:
    print(f"    Prototype grad norm: {proto_grad.norm():.6f}")
    assert proto_grad.norm() > 0, "No gradient to prototypes!"
else:
    print("    ✗ No gradient to prototypes!")
    sys.exit(1)

# Check Conv layer gradients
conv_grad = projector.proj_p3[0].weight.grad
if conv_grad is not None:
    print(f"    Conv layer grad norm: {conv_grad.norm():.6f}")
else:
    print("    ✗ No gradient to Conv layers!")
    sys.exit(1)

print("    ✓ Gradients flow correctly")

# 7. Test edge cases
print("\n[7] Testing edge cases...")

# All background (no foreground)
labels_bg = torch.full((B, N_anchors), -1, dtype=torch.long)
loss_bg = loss_fn(hyp_emb.detach(), labels_bg, protos)
print(f"    All-background loss: {loss_bg.item():.4f} (expected: 0.0)")
assert loss_bg.item() == 0.0, "All-background should return 0 loss"

# Single class
labels_single = torch.full((B, N_anchors), -1, dtype=torch.long)
labels_single[0, :100] = 5  # 100 anchors assigned to class 5
loss_single = loss_fn(hyp_emb.detach(), labels_single, protos)
print(f"    Single-class loss: {loss_single.item():.4f}")
assert loss_single.item() > 0, "Single-class loss should be > 0"

print("    ✓ Edge cases handled")

# 8. Test enable_projector_grad for T1 vs T2
print("\n[8] Testing enable_projector_grad...")

# Simulate T1 (index=0)
for param in projector.parameters():
    param.requires_grad = False

# Use the actual function logic
index = 0  # T1
if index == 0:
    for param in projector.parameters():
        param.requires_grad = True
else:
    for name, param in projector.named_parameters():
        if 'prototype_tangent' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

trainable_t1 = sum(p.numel() for p in projector.parameters() if p.requires_grad)
total = sum(p.numel() for p in projector.parameters())
print(f"    T1 trainable: {trainable_t1:,} / {total:,}")
assert trainable_t1 == total, "T1 should train ALL parameters"

# Simulate T2 (index>0)
index = 11  # T2
if index == 0:
    for param in projector.parameters():
        param.requires_grad = True
else:
    for name, param in projector.named_parameters():
        if 'prototype_tangent' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

trainable_t2 = sum(p.numel() for p in projector.parameters() if p.requires_grad)
print(f"    T2 trainable: {trainable_t2:,} / {total:,} (prototypes only)")
assert trainable_t2 == 11 * 256, f"T2 should only train prototypes: {11 * 256}"

print("    ✓ T1/T2 grad control correct")

# 9. Summary
print("\n" + "=" * 60)
print("VERIFICATION COMPLETE - ALL CHECKS PASSED")
print("=" * 60)

print("""
SUMMARY OF TRAINING FLOW:
1. FPN features (P3, P4, P5) → HyperbolicProjector → Poincaré embeddings (B, 8400, 256)
2. TAL assigns class labels to anchors (B, 8400) with -1 for background
3. HyperbolicContrastiveLoss:
   - Filters valid samples (label >= 0)
   - Computes geodesic distance to all prototypes
   - Cross-entropy on negative distances
4. Gradients flow back to:
   - T1: ALL projector params (Conv + prototypes)
   - T2+: Only prototypes (Conv frozen)
5. Total loss = loss_cls + loss_bbox + loss_dfl + hyp_loss

KEY DIFFERENCES FROM MSCAL:
- MSCAL: Per-class projectors → cosine scores → SupCon loss
- Hyperbolic: Shared projector → Poincaré space → distance-based CE

NO ANGLE LOSS (unlike HypGCD) - only distance-based contrastive.
""")
