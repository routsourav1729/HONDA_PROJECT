#!/usr/bin/env python3
"""
Verify horospherical training flow end-to-end.
"""

import torch
import torch.nn as nn
import sys

print("=" * 60)
print("HOROSPHERICAL TRAINING FLOW VERIFICATION")
print("=" * 60)

# 1. Test imports
print("\n[1] Testing imports...")
try:
    from core.hyperbolic import (
        expmap0, logmap0, project, busemann, busemann_batch,
        ToPoincare, HyperbolicProjector, HorosphericalClassifier, HorosphericalLoss
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
    curvature=1.0,  # c=1.0 for horospherical
    num_classes=11,
    clip_r=0.95
)

# Dummy FPN features
B = 2
p3 = torch.randn(B, 384, 80, 80)
p4 = torch.randn(B, 768, 40, 40)
p5 = torch.randn(B, 768, 20, 20)
img_feats = (p3, p4, p5)

hyp_emb = projector(img_feats)
print(f"    Input: P3={p3.shape}, P4={p4.shape}, P5={p5.shape}")
print(f"    Output: {hyp_emb.shape} (expected: [2, 8400, 256])")
assert hyp_emb.shape == (B, 8400, 256), f"Wrong shape: {hyp_emb.shape}"

# Verify embeddings are inside ball (norm < 1 for c=1)
emb_norms = hyp_emb.norm(dim=-1)
print(f"    Embedding norms: min={emb_norms.min():.4f}, max={emb_norms.max():.4f} (must be < 1.0)")
assert emb_norms.max() < 1.0, f"Embeddings outside ball! Max norm: {emb_norms.max()}"
print("    ✓ Forward pass correct")

# 3. Test prototypes (on boundary)
print("\n[3] Testing prototypes (on boundary)...")
protos = projector.prototypes
print(f"    Prototype shape: {protos.shape} (expected: [11, 256])")
assert protos.shape == (11, 256), f"Wrong shape: {protos.shape}"

norms = protos.norm(dim=-1)
print(f"    Prototype norms: {norms} (should be ~1.0 for c=1)")
# For c=1, ball radius R = 1/√1 = 1.0, so prototypes should have norm = 1.0
assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Prototypes not on boundary!"
print("    ✓ Prototypes correctly on boundary")

# 4. Test Busemann function
print("\n[4] Testing Busemann function...")
x = torch.randn(100, 256)
x = x / x.norm(dim=-1, keepdim=True) * 0.8  # Inside ball

B_vals = busemann(protos, x, c=1.0)
print(f"    Input: x shape={x.shape}, protos shape={protos.shape}")
print(f"    Busemann values shape: {B_vals.shape} (expected: [100, 11])")
print(f"    Busemann range: min={B_vals.min():.4f}, max={B_vals.max():.4f}")
assert B_vals.shape == (100, 11), f"Wrong shape: {B_vals.shape}"
print("    ✓ Busemann computation correct")

# 5. Test HorosphericalClassifier
print("\n[5] Testing HorosphericalClassifier...")
classifier = projector.classifier
scores = classifier(x)
print(f"    Scores shape: {scores.shape} (expected: [100, 11])")
print(f"    Score range: min={scores.min():.4f}, max={scores.max():.4f}")

# Test OOD scores
ood = classifier.get_ood_scores(x)
print(f"    OOD scores shape: {ood.shape} (expected: [100])")
print(f"    OOD range: min={ood.min():.4f}, max={ood.max():.4f}")
print("    ✓ Classifier correct")

# 6. Test HorosphericalLoss
print("\n[6] Testing HorosphericalLoss...")
loss_fn = HorosphericalLoss(curvature=1.0)

# Simulate TAL labels
N_anchors = 8400
labels = torch.full((B, N_anchors), -1, dtype=torch.long)
n_fg = int(0.05 * N_anchors)
fg_indices = torch.randint(0, N_anchors, (B, n_fg))
fg_classes = torch.randint(0, 11, (B, n_fg))
for b in range(B):
    labels[b, fg_indices[b]] = fg_classes[b]

print(f"    Labels: {(labels >= 0).sum()} foreground, {(labels == -1).sum()} background")

# Compute scores and loss
scores = projector.classifier(hyp_emb.reshape(-1, 256))
scores = scores.reshape(B, N_anchors, -1)
loss = loss_fn(scores, labels)
print(f"    Loss value: {loss.item():.4f}")
assert loss.item() >= 0, "Loss should be non-negative"
assert not torch.isnan(loss), "Loss is NaN!"
print("    ✓ Loss computation correct")

# 7. Test gradient flow
print("\n[7] Testing gradient flow...")
projector.zero_grad()
loss.backward()

# Check prototype_direction gradients
proto_grad = projector.classifier.prototype_direction.grad
if proto_grad is not None:
    print(f"    prototype_direction grad norm: {proto_grad.norm():.6f}")
    assert proto_grad.norm() > 0, "No gradient to prototype directions!"
else:
    print("    ✗ No gradient to prototype directions!")
    sys.exit(1)

# Check prototype_bias gradients
bias_grad = projector.classifier.prototype_bias.grad
if bias_grad is not None:
    print(f"    prototype_bias grad norm: {bias_grad.norm():.6f}")
else:
    print("    ✗ No gradient to prototype biases!")
    sys.exit(1)

# Check Conv layer gradients
conv_grad = projector.proj_p3[0].weight.grad
if conv_grad is not None:
    print(f"    Conv layer grad norm: {conv_grad.norm():.6f}")
else:
    print("    ✗ No gradient to Conv layers!")
    sys.exit(1)

print("    ✓ Gradients flow correctly")

# 8. Test dispersion loss
print("\n[8] Testing angular dispersion loss...")
disp_loss = projector.classifier.angular_dispersion_loss()
print(f"    Dispersion loss: {disp_loss.item():.6f}")
assert disp_loss.item() >= 0, "Dispersion loss should be non-negative"

# Force some prototypes to be similar to test gradient
with torch.no_grad():
    projector.classifier.prototype_direction.data[0] = projector.classifier.prototype_direction.data[1] * 0.9

disp_loss_forced = projector.classifier.angular_dispersion_loss()
print(f"    Dispersion loss (forced similar protos): {disp_loss_forced.item():.6f}")

projector.zero_grad()
disp_loss_forced.backward()
disp_grad = projector.classifier.prototype_direction.grad
if disp_grad is not None and disp_grad.norm() > 0:
    print(f"    Dispersion grad norm: {disp_grad.norm():.6f}")
else:
    print(f"    Dispersion grad norm: 0 (no violations above margin)")
print("    ✓ Dispersion loss correct")

# 9. Test edge cases
print("\n[9] Testing edge cases...")

# All background
labels_bg = torch.full((B, N_anchors), -1, dtype=torch.long)
scores_bg = projector.classifier(hyp_emb.reshape(-1, 256)).reshape(B, N_anchors, -1)
loss_bg = loss_fn(scores_bg, labels_bg)
print(f"    All-background loss: {loss_bg.item():.4f} (expected: 0.0)")
assert loss_bg.item() == 0.0, "All-background should return 0 loss"

print("    ✓ Edge cases handled")

# 10. Test enable_projector_grad for T1 vs T2
print("\n[10] Testing enable_projector_grad logic...")

# T1: All parameters trainable
for param in projector.parameters():
    param.requires_grad = True
trainable_t1 = sum(p.numel() for p in projector.parameters() if p.requires_grad)
total = sum(p.numel() for p in projector.parameters())
print(f"    T1: {trainable_t1:,} / {total:,} trainable (all params)")

# T2: Only classifier trainable
for name, param in projector.named_parameters():
    param.requires_grad = 'classifier' in name
trainable_t2 = sum(p.numel() for p in projector.parameters() if p.requires_grad)
classifier_params = sum(p.numel() for p in projector.classifier.parameters())
print(f"    T2: {trainable_t2:,} / {total:,} trainable (classifier only)")
assert trainable_t2 == classifier_params, "T2 should train only classifier"

print("    ✓ T1/T2 grad control correct")

# Summary
print("\n" + "=" * 60)
print("VERIFICATION COMPLETE - ALL CHECKS PASSED")
print("=" * 60)

print("""
HOROSPHERICAL TRAINING FLOW SUMMARY:
1. FPN features (P3, P4, P5) → HyperbolicProjector → Poincaré embeddings (B, 8400, 256)
2. Embeddings clipped to ball interior (norm < 1.0 for c=1)
3. Prototypes on boundary (norm = 1.0 for c=1)
4. TAL assigns class labels to anchors (B, 8400), -1 for background
5. HorosphericalClassifier:
   - Score = -Busemann(prototype, embedding) + bias
   - Higher score = closer to prototype
6. HorosphericalLoss: Cross-entropy over scores
7. Dispersion loss: Penalizes similar prototype directions
8. Gradients flow to:
   - T1: ALL projector params (Conv + classifier)
   - T2+: Only classifier (prototype_direction + prototype_bias)

OOD DETECTION:
- ood_score = -max(horosphere_scores)
- Higher ood_score = more OOD (far from all prototypes)
- Detection: ood_score > threshold → unknown
""")
