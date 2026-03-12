# HypYOLO v2 — Loss Function Reference

**File:** `core/hyperbolic/projector.py` → `GeodesicPrototypeLoss`  
(aliases: `HorosphericalLoss`, `HorosphericalLossV2` — all point to the same class now)

---

## Big Picture

Visual features from FPN (P3/P4/P5) are projected into a **Poincaré ball** of curvature `c`.  
Each known class has an **interior prototype** `z_k` (not on boundary — key change from v1).  
Classification = nearest prototype in geodesic distance.  
OOD = far from ALL prototypes → unknown class.

```
FPN feats (B,384/768,H,W)
   → Conv projector (1×1 BN-ReLU)
   → pre-expmap norms cached        ← used for L_reg
   → ToPoincare (expmap₀ + clip_r clip)
   → Poincaré embeddings x ∈ B^D_c
   → GeodesicPrototypeClassifier → s_k(x) = -d²_B(x, z_k)
```

---

## Geometry

**Poincaré ball radius:** $R = 1/\sqrt{c}$

**Geodesic (Riemannian) distance:**
$$d_B(x, y) = \frac{2}{\sqrt{c}} \tanh^{-1}\!\left(\sqrt{c}\,\|{-x \oplus_c y}\|\right)$$

where $\oplus_c$ is Möbius addition. Near the boundary distances blow up → that's what gives geometric separation power.

**Classification logit:**
$$s_k(x) = -d_B^2(x,\, z_k)$$

softmax over $s_k \cdot \alpha$ gives class probabilities ($\alpha$ = `score_scale`).

**OOD score (inference):**
$$\text{OOD}(x) = \min_k\, d_B^2(x,\, z_k)$$
High → unknown. Threshold calibrated post-hoc.

---

## Loss: 3 Components

$$\boxed{\mathcal{L} = \underbrace{\lambda_\text{ce}\, \mathcal{L}_\text{cls}}_{\text{A}} + \underbrace{\beta_\text{reg}\, \mathcal{L}_\text{reg}}_{\text{B}} + \underbrace{\lambda_\text{sep}\, \mathcal{L}_\text{sep}}_{\text{C}}}$$

### A — Class-Balanced CE (`L_cls`)

$$\mathcal{L}_\text{cls} = -\frac{1}{|FG|} \sum_{i \in FG} w_{y_i} \log \frac{e^{\alpha \cdot s_{y_i}(x_i)}}{\sum_k e^{\alpha \cdot s_k(x_i)}}$$

- **`score_scale` = α = 0.1** — divides raw $-d^2$ scores before softmax. Early in training embeddings cluster near the boundary so $d^2$ is huge (e.g., 20–100); without scaling all softmax mass collapses to one class → loss gives no gradient. α rescales to a healthy range. **Raise if training is too slow; lower if loss is NaN early.**
- **Per-class weight** $w_k \propto (\max\_count/count_k)^{0.5}$ — square-root inverse frequency. **`class_balance_smoothing` = 0.5** controls the power; 0 = no balance, 1 = full inverse frequency. IDD has big class imbalance (car >> excavator), so keep ≈ 0.5.
- **`ce_weight` = 1.0** — scalar multiplier on this whole term. Usually leave at 1.

### B — Pre-Expmap Norm Regularization (`L_reg`)

$$\mathcal{L}_\text{reg} = \frac{1}{N} \sum_{i} \|\tilde{x}_i\|^2$$

$\tilde{x}_i$ = Euclidean feature **before** `expmap0`. `ToPoincare` uses $\tanh$ internally, so if norms blow up → all embeddings saturate near boundary → every pairwise geodesic distance becomes huge and equal → classifier can't distinguish anything.

- **`beta_reg` = 0.1** — main knob. **Increase if `pre_expmap_norm_mean` > 3–4 in logs; decrease if embeddings collapse to origin (norm → 0).**
- Applied to **all** 8400 anchors (FG + BG), not just matched ones.

### C — Prototype Separation Loss (`L_sep`)

$$\mathcal{L}_\text{sep} = \frac{1}{\binom{K}{2}} \sum_{i \neq j} \max\!\left(0,\; m - d_B(z_i, z_j)\right)$$

Hinge loss: penalizes any pair of prototypes closer than margin $m$ in geodesic distance.

- **`sep_margin` = m = 1.0** — minimum geodesic gap. Since geodesic distances in the ball (with `max_proto_norm=0.5, c=1`) range roughly 0–2, a margin of 1.0 is meaningful. **Increase if classes are being confused; decrease if training diverges.**
- **`lambda_sep` = 1.0** — weight. Can be high because this term is bounded (hinge saturates at 0 once prototypes spread out enough). **Start at 1.0, lower to 0.1 if it dominates early training.**

---

## Prototype Constraints

| Param | Default | Meaning |
|---|---|---|
| `prototype_init_norm` | 0.4 | Initial $\|z_k\|$ inside ball (ball radius = 1 for c=1) |
| `max_proto_norm` | 0.5 | Hard clamp applied after every optimizer step via `project_prototypes_to_ball()` |
| `trainable_prototypes` | True | T1: train freely. T2: freeze T1 protos, only train new |

Clamping at 0.5 prevents optimizer from pushing $z_k$ to 0.999 (boundary). Boundary prototypes recreate the v1 horospherical problem: all geodesic distances diverge and OOD detection breaks.

---

## Key Hyperparameters at a Glance

| Param | Default | What to tune |
|---|---|---|
| `curvature` (`hyp_c`) | 1.0 | Lower (0.1) = flatter space, more forgiving. Higher = more hierarchical separation. |
| `hyp_dim` | 64 | Poincaré embedding dimension. 64 is good; 256 is overkill for 8 classes. |
| `clip_r` | 2.0 | Safety clip on pre-expmap norms (last resort). L_reg should keep norms < this. |
| `score_scale` (α) | 0.1 | Logit temperature before CE. Lower = softer, higher = peakier. |
| `beta_reg` | 0.1 | Watch `pre_expmap_norm_mean` in logs. |
| `lambda_sep` | 1.0 | Watch `sep_loss` → should decay and plateau near 0. |
| `sep_margin` | 1.0 | Min geodesic gap between prototypes. |
| `hyp_loss_weight` | 1.0 | Scales entire hyperbolic loss vs YOLO detection loss. |
| `class_balance_smoothing` | 0.5 | 0=no balance, 1=full inv-freq. |

---

## What to Watch in Logs

```
ce_loss          → should decrease steadily
reg_loss         → should stay small (< 1); spike = norms blowing up
sep_loss         → should decrease and hit ~0 once prototypes spread
cls_acc          → accuracy on matched FG anchors (not mAP)
pos_dist_sq_mean → mean d²(x, correct proto); should decrease
min_dist_sq_mean → mean min d² over all classes; if this > pos_dist_sq → good OOD
emb_poincare_norm_mean → embedding norms inside ball; healthy = 0.3–0.7
proto_norm_mean  → prototype norms; clamped to max_proto_norm (0.5)
pre_expmap_norm_mean → raw Euclidean norms; keep < 2–3
```

---

## IDD Setups

| Split | Classes in classifier | Unknown (OOD) |
|---|---|---|
| T1 | 8 (car, motorcycle, rider, person, autorickshaw, bicycle, traffic sign, traffic light) | animal, pull_cart, road_roller, pole, tractor, concrete_mixer |
| T2 | 14 (T1 + bus, truck, tanker_vehicle, crane_truck, street_cart, excavator) | same 6 |
| T3 | not yet defined in `core/pascal_voc.py` | — |

T1 protos frozen in T2; only T2-novel protos trained. `load_hyp_ckpt()` handles this.
