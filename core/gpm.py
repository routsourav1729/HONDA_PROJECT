"""
Gradient Projection Memory (GPM) for continual few-shot learning.

Protects base-class knowledge in projector conv weights during T2 fine-tuning
by projecting out gradient components that lie in the base-class activation subspace.

Reference: Saha et al., "Gradient Projection Memory for Continual Learning", ICLR 2021.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


def _get_projector_conv_layers(model):
    """
    Get all Conv2d layers from BiLipschitz projectors (proj_p3, proj_p4, proj_p5).

    BiLipschitzProjector structure:
        main: Sequential(spectral_norm(Conv2d), GroupNorm, ReLU, spectral_norm(Conv2d))
        skip: spectral_norm(Conv2d)

    Returns dict mapping layer_name -> nn.Conv2d module.
    """
    projector = model.hyp_projector
    conv_layers = {}

    for scale_name in ['proj_p3', 'proj_p4', 'proj_p5']:
        proj = getattr(projector, scale_name)

        if hasattr(proj, 'main'):
            # BiLipschitz: main has conv at [0] and [3], skip is a single conv
            for i, layer in enumerate(proj.main):
                if isinstance(layer, nn.Conv2d):
                    conv_layers[f'{scale_name}.main.{i}'] = layer
            conv_layers[f'{scale_name}.skip'] = proj.skip
        else:
            # Sequential: conv at [0] and [3]
            for i, layer in enumerate(proj):
                if isinstance(layer, nn.Conv2d):
                    conv_layers[f'{scale_name}.{i}'] = layer

    return conv_layers


def compute_gpm_bases(model, dataloader, threshold=0.97, max_batches=20, device=None):
    """
    Compute GPM basis vectors from T1 training data activations.

    For each projector conv layer, collects input activations, runs SVD,
    and keeps top-k singular vectors that capture >= threshold of variance.

    Args:
        model: HypCustomYoloWorld (unwrapped from DDP)
        dataloader: T1 training dataloader
        threshold: variance threshold for selecting basis vectors (0.97 = keep 97%)
        max_batches: max batches to process (20 batches * 32 batch_size = ~640 images)
        device: torch device

    Returns:
        gpm_bases: dict {layer_name: tensor (in_ch, k)} of orthonormal basis vectors
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    conv_layers = _get_projector_conv_layers(model)
    print(f"[GPM] Computing bases for {len(conv_layers)} conv layers, threshold={threshold}")

    # Register hooks to capture input activations
    activations = {name: [] for name in conv_layers}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            # input[0] shape: (B, C_in, H, W) for Conv2d
            inp = input[0].detach()
            # Reshape to (C_in, B*H*W) — input channel is the feature axis
            B, C, H, W = inp.shape
            activations[name].append(inp.permute(1, 0, 2, 3).reshape(C, -1))  # (C_in, B*H*W)
        return hook_fn

    for name, layer in conv_layers.items():
        hooks.append(layer.register_forward_hook(make_hook(name)))

    # Forward pass to collect activations
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="[GPM] Collecting activations",
                                       total=min(max_batches, len(dataloader)))):
            if i >= max_batches:
                break
            data = model.parent.data_preprocessor(batch)
            # Run through backbone + neck + projector
            img_feats = model.parent.backbone.forward_image(data['inputs'])
            if model.parent.with_neck:
                txt_feats = model.embeddings if model.embeddings is not None else model.frozen_embeddings
                txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)
                img_feats = model.parent.neck(img_feats, txt_feats) if model.parent.mm_neck else model.parent.neck(img_feats)
            # Trigger projector forward (hooks will capture activations)
            model.hyp_projector(img_feats)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute SVD bases
    gpm_bases = {}
    for name in conv_layers:
        act = torch.cat(activations[name], dim=1)  # (C_in, total_spatial)
        # Center the activations
        act = act - act.mean(dim=1, keepdim=True)

        U, S, _ = torch.linalg.svd(act, full_matrices=False)
        # U: (C_in, min(C_in, N)), S: (min(C_in, N),)

        # Select top-k vectors capturing >= threshold of variance
        var = S ** 2
        cumvar = var.cumsum(0) / var.sum()
        k = (cumvar < threshold).sum().item() + 1
        k = min(k, U.shape[1])

        basis = U[:, :k].contiguous()  # (C_in, k)
        gpm_bases[name] = basis.to(device)

        explained = cumvar[k - 1].item() * 100
        print(f"  [GPM] {name}: in_ch={act.shape[0]}, k={k}/{U.shape[1]}, "
              f"variance={explained:.1f}%")

    model.train()
    return gpm_bases


def precompute_projection_matrices(gpm_bases):
    """
    Precompute P = M @ M^T projection matrices for each layer.

    Args:
        gpm_bases: dict {name: tensor (C_in, k)}

    Returns:
        proj_matrices: dict {name: tensor (C_in, C_in)}
    """
    proj_matrices = {}
    for name, M in gpm_bases.items():
        proj_matrices[name] = (M @ M.T).contiguous()  # (C_in, C_in)
    return proj_matrices


def project_gradients(model, proj_matrices):
    """
    Project out base-class subspace components from projector conv gradients.

    Called after loss.backward(), before optimizer.step().

    For each conv weight (out_ch, in_ch, 1, 1) or (out_ch, in_ch):
        grad = grad - grad @ P
    This removes the component of the gradient that would modify
    the base-class activation subspace.

    Args:
        model: HypCustomYoloWorld (unwrapped from DDP)
        proj_matrices: dict {name: tensor (C_in, C_in)} from precompute_projection_matrices
    """
    conv_layers = _get_projector_conv_layers(model)

    for name, layer in conv_layers.items():
        if name not in proj_matrices:
            continue

        # Spectral-normed layers store weight as weight_orig
        param_name = 'weight_orig' if hasattr(layer, 'weight_orig') else 'weight'
        param = getattr(layer, param_name)

        if param.grad is None:
            continue

        P = proj_matrices[name]  # (C_in, C_in)
        grad = param.grad.data

        if grad.dim() == 4:
            # Conv2d weight: (out_ch, in_ch, kH, kW) — for 1x1: kH=kW=1
            out_ch, in_ch, kH, kW = grad.shape
            grad_2d = grad.reshape(out_ch, in_ch * kH * kW)
            # For 1x1 conv, in_ch*1*1 = in_ch, so P shape matches
            grad_2d = grad_2d - grad_2d @ P
            param.grad.data = grad_2d.reshape(out_ch, in_ch, kH, kW)
        elif grad.dim() == 2:
            param.grad.data = grad - grad @ P
