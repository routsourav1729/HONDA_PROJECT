"""
Adaptive Per-Prototype Threshold Calibration.

Runs the model over the training set and collects per-class vMF scores
from GT-assigned anchors.  These per-class (mean, std) values
define adaptive OOD thresholds:

    tau_k = mean_k - alpha * std_k

vMF scores = log Z_d(kappa_c) + kappa_c * mu_c^T * r: higher = more ID.
If max_c score < tau_k -> unknown.

This module is called automatically at the end of training (dev_hyp_ddp.py).
The resulting stats are embedded in the final checkpoint so that test_hyp.py
can load them without any external file.
"""

import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

# pmath no longer needed (vMF uses L2-normalized embeddings, no Poincare ops)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_anchor_centers(h, w, device):
    """YOLO-World anchor centers (80x80 + 40x40 + 20x20 = 8400 for 640x640)."""
    strides = [8, 16, 32]
    centers = []
    for s in strides:
        gh, gw = h // s, w // s
        y = torch.arange(gh, device=device).float() * s + s / 2
        x = torch.arange(gw, device=device).float() * s + s / 2
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        centers.append(torch.stack([xx.flatten(), yy.flatten()], dim=-1))
    return torch.cat(centers, dim=0)


# ---------------------------------------------------------------------------
# Main calibration function
# ---------------------------------------------------------------------------

@torch.no_grad()
def calibrate(model, train_loader, known_class_names, dataset_root=None, hyp_c=1.0,
              max_batches=0):
    """
    Calibrate adaptive thresholds from the training set.

    Uses GT boxes from the dataloader directly (already in 640x640 padded
    image space) — no XML parsing, no extra disk I/O.

    Parameters
    ----------
    model : HypCustomYoloWorld
        Trained model (will be set to eval mode temporarily).
    train_loader : DataLoader
        Training dataloader (mmengine-style).
    known_class_names : list[str]
        Ordered list of known class names (length = K).
    dataset_root : str, optional
        Unused (kept for API compatibility).
    hyp_c : float
        Unused (kept for API compatibility with old callers).
    max_batches : int
        Max batches to process (0 = all). Default 0 = full dataset.
        At batch_size=32 with 3001 batches, full pass ~5-8 min (no XML I/O).

    Returns
    -------
    adaptive_stats : dict
        {
            'per_class': {class_name: {'mean': float, 'std': float, 'count': int}},
            'alpha': 0.75
        }
    """
    was_training = model.training
    model.eval()

    prototypes = model.prototypes.detach()   # (K, dim) unit vectors — scoring handled internally
    # Scoring is done via model.compute_vmf_scores() which accesses prototypes internally

    per_class_scores = defaultdict(list)
    samples_count = defaultdict(int)

    total_batches = len(train_loader) if hasattr(train_loader, '__len__') else '?'
    use_batches = (min(max_batches, total_batches)
                   if max_batches > 0 and isinstance(total_batches, int)
                   else total_batches)
    print(f"\n{'='*60}")
    print(f"CALIBRATING ADAPTIVE THRESHOLDS FROM TRAINING DATA")
    print(f"  Known classes ({len(known_class_names)}): {known_class_names}")
    print(f"  Batches: {use_batches}/{total_batches}")
    print(f"  Note: using GT instances from dataloader (no XML I/O)")
    print(f"{'='*60}")

    desc = "Calibrating thresholds"
    for i, batch in enumerate(tqdm(train_loader, desc=desc,
                                   total=use_batches if isinstance(use_batches, int) else None)):
        if max_batches > 0 and i >= max_batches:
            break
        if i > 0 and i % 500 == 0:
            total = sum(samples_count.values())
            tqdm.write(f"  [batch {i}/{use_batches}] {total} known GT boxes processed")

        try:
            data_batch = model.parent.data_preprocessor(batch)

            # Forward through backbone + neck only (no head, no loss, no grads)
            x = model.parent.backbone.forward_image(data_batch['inputs'])
            if model.parent.with_neck:
                if model.parent.mm_neck:
                    txt = (model.frozen_embeddings
                           if model.frozen_embeddings is not None
                           else model.embeddings)
                    txt = txt.repeat(x[0].shape[0], 1, 1)
                    x = model.parent.neck(x, txt)
                else:
                    x = model.parent.neck(x)

            hyp_embeddings, _ = model.hyp_projector(x)  # (B, 8400, dim), ignore raw_proj

            h, w = data_batch['inputs'].shape[-2:]
            anchor_centers = get_anchor_centers(h, w, device=hyp_embeddings.device)

            for b_idx, data_sample in enumerate(data_batch['data_samples']):
                gt = data_sample.gt_instances

                if len(gt) == 0:
                    continue

                # GT bboxes are already in 640x640 padded-image space (xyxy)
                # The dataloader pipeline (YOLOv5KeepRatioResize + LetterResize)
                # transforms them along with the image — no manual scaling needed.
                bboxes = gt.bboxes  # (M, 4) xyxy, may be HorizontalBoxes wrapper
                labels = gt.labels  # (M,) int64, indices into known_class_names

                # Handle mmdet HorizontalBoxes wrapper
                if hasattr(bboxes, 'tensor'):
                    bboxes = bboxes.tensor

                if bboxes.shape[0] == 0:
                    continue

                bboxes = bboxes.to(hyp_embeddings.device)
                labels = labels.to(hyp_embeddings.device)

                # Compute box centers in padded-image space
                cx = ((bboxes[:, 0] + bboxes[:, 2]) / 2.0).clamp(0, w - 1)
                cy = ((bboxes[:, 1] + bboxes[:, 3]) / 2.0).clamp(0, h - 1)
                box_centers = torch.stack([cx, cy], dim=1)  # (M, 2)

                # Vectorised nearest-anchor matching
                dists = ((box_centers.unsqueeze(1)
                          - anchor_centers.unsqueeze(0)) ** 2).sum(-1)  # (M, A)
                nearest_indices = dists.argmin(dim=1)  # (M,)

                # Batch vMF scores
                embs = hyp_embeddings[b_idx, nearest_indices]       # (M, dim)
                vmf_scores = model.compute_vmf_scores(embs)          # (M, K)
                max_geo = vmf_scores.max(dim=-1).values              # (M,)

                # Accumulate per-class
                for j in range(len(labels)):
                    label_idx = labels[j].item()
                    if 0 <= label_idx < len(known_class_names):
                        cls_name = known_class_names[label_idx]
                        per_class_scores[cls_name].append(max_geo[j].item())
                        samples_count[cls_name] += 1

        except Exception as e:
            print(f"  Error batch {i}: {e}")
            import traceback; traceback.print_exc()
            continue

    # Compute per-class statistics
    adaptive_stats = {'per_class': {}, 'alpha': 0.75}

    print(f"\n  {'Class':<25s} {'Count':>7s} {'Mean':>8s} {'Std':>8s} "
          f"{'Min':>8s} {'Max':>8s}")
    print(f"  {'-'*65}")

    for cls_name in known_class_names:
        scores = np.array(per_class_scores.get(cls_name, []))
        if len(scores) > 0:
            adaptive_stats['per_class'][cls_name] = {
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'count': len(scores),
                'min': float(scores.min()),
                'max': float(scores.max()),
            }
            print(f"  {cls_name:<25s} {len(scores):>7d} {scores.mean():>8.4f} "
                  f"{scores.std():>8.4f} {scores.min():>8.4f} {scores.max():>8.4f}")
        else:
            adaptive_stats['per_class'][cls_name] = {
                'mean': 0.0, 'std': 1.0, 'count': 0, 'min': 0.0, 'max': 0.0,
            }
            print(f"  {cls_name:<25s} {'0 (WARN)':>7s}")

    total = sum(samples_count.values())
    print(f"  {'-'*65}")
    print(f"  Total known GT boxes calibrated: {total}")

    if was_training:
        model.train()

    return adaptive_stats


def compute_thresholds(adaptive_stats, class_names, alpha=None):
    """
    Compute per-class threshold tensor from calibration stats.

    tau_k = mean_k - alpha * std_k

    Every class uses its own calibration data, regardless of sample count.
    In few-shot (e.g., 10-shot), even 10 samples give a reasonable mean/std.

    Parameters
    ----------
    adaptive_stats : dict
        Output of calibrate() or loaded from checkpoint.
    class_names : list[str]
        Ordered class names.
    alpha : float, optional
        Override alpha (defaults to stored value).

    Returns
    -------
    thresholds : torch.Tensor  (K,)
    alpha_used : float
    """
    if alpha is None:
        alpha = adaptive_stats.get('alpha', 0.75)

    cal = adaptive_stats['per_class']

    thresholds = []
    print(f"\n  Adaptive thresholds (alpha={alpha:.2f}):")
    for cls_name in class_names:
        if cls_name in cal and cal[cls_name]['count'] > 0:
            mu = cal[cls_name]['mean']
            std = cal[cls_name]['std']
            n = cal[cls_name]['count']
            tau = mu - alpha * std
            print(f"    {cls_name:<20s}: tau={tau:.4f}  (mean={mu:.4f}, std={std:.4f}, n={n})")
        else:
            tau = -10.0
            print(f"    {cls_name:<20s}: tau={tau:.4f}  (MISSING — permissive fallback)")
        thresholds.append(tau)

    return torch.tensor(thresholds, dtype=torch.float32), alpha
