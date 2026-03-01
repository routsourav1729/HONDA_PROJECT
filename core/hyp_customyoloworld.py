"""
Hyperbolic Custom YOLO World with Horospherical Classification.

Uses Busemann function with ideal prototypes on boundary for OOD detection.
Prototypes live on the Poincaré ball boundary (‖p‖ = 1 for c=1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmdet.structures import SampleList
from mmdet.utils import InstanceList
import copy
from typing import List, Optional, Tuple, Union, Sequence
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from mmdet.models.utils import unpack_gt_instances, filter_scores_and_topk
from mmyolo.models.utils import gt_instances_preprocess
from mmengine.dist import get_dist_info

from .hyperbolic import HyperbolicProjector, busemann
from .hyperbolic.projector import HorosphericalLoss


def _rebuild_projector_from_ckpt(projector, state_dict):
    """Rebuild projector conv layers to match checkpoint architecture.
    
    Detects single-conv vs two-conv from checkpoint weight shapes and
    rebuilds the Sequential modules accordingly. This allows loading
    old single-conv checkpoints with the current two-conv code (and vice versa).
    """
    import torch.nn as nn
    
    for scale in ['p3', 'p4', 'p5']:
        prefix = f'hyp_projector.proj_{scale}'
        # Gather all conv weight keys for this scale
        conv_keys = sorted([k for k in state_dict if k.startswith(prefix) and 'weight' in k 
                           and 'running' not in k and k.endswith('.weight')])
        
        # Check if it's single-conv (3 layers: conv, bn, relu → indices 0,1,2)
        # or two-conv (4 layers: conv, bn, relu, conv → indices 0,1,2,3)
        has_second_conv = f'{prefix}.3.weight' in state_dict
        
        first_conv_w = state_dict[f'{prefix}.0.weight']
        out_ch, in_ch = first_conv_w.shape[:2]
        
        if has_second_conv:
            # Two-conv: conv(in→in) + BN + ReLU + conv(in→out)
            second_conv_w = state_dict[f'{prefix}.3.weight']
            final_out = second_conv_w.shape[0]
            new_proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, final_out, kernel_size=1, bias=False),
            )
            print(f"    {scale}: two-conv ({in_ch}→{out_ch}→{final_out})")
        else:
            # Single-conv: conv(in→out) + BN + ReLU
            new_proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            print(f"    {scale}: single-conv ({in_ch}→{out_ch})")
        
        setattr(projector, f'proj_{scale}', new_proj)


def load_hyp_ckpt(model, checkpoint_path, prev_classes, current_classes, eval=False):
    """
    Load checkpoint for horospherical model.
    
    Handles:
    - Text embeddings (frozen_embeddings, embeddings)
    - Prototype directions + biases (frozen_directions, frozen_biases, trainable)
    
    Note: frozen_directions and frozen_biases are registered as buffers (no grad).
    """
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Auto-detect projector architecture from checkpoint weights.
    # Single-conv checkpoint: proj_p3.0.weight is (out_dim, in_dim, 1, 1)
    # Two-conv checkpoint:    proj_p3.0.weight is (in_dim, in_dim, 1, 1)
    # If mismatch with current model, rebuild projector to match checkpoint.
    ckpt_p3_key = 'hyp_projector.proj_p3.0.weight'
    if ckpt_p3_key in state_dict:
        ckpt_shape = state_dict[ckpt_p3_key].shape  # (out_ch, in_ch, 1, 1)
        model_shape = model.hyp_projector.proj_p3[0].weight.shape
        if ckpt_shape != model_shape:
            print(f"  ⚠ Projector architecture mismatch: ckpt={list(ckpt_shape)} vs model={list(model_shape)}")
            print(f"    Rebuilding projector to match checkpoint...")
            _rebuild_projector_from_ckpt(model.hyp_projector, state_dict)
    
    # Keys to handle separately
    exclude_keys = ['embeddings', 'frozen_embeddings', 
                    'frozen_directions', 'frozen_biases',
                    'prototype_direction', 'prototype_bias']
    partial_state_dict = {k: v for k, v in state_dict.items() 
                          if not any(ex in k for ex in exclude_keys)}
    model.load_state_dict(partial_state_dict, strict=False)
    
    if eval:
        # Evaluation: load all embeddings and prototypes
        if state_dict.get('frozen_embeddings') is not None:
            model.frozen_embeddings = nn.Parameter(state_dict['frozen_embeddings'], requires_grad=False)
        if state_dict.get('embeddings') is not None:
            model.embeddings = nn.Parameter(state_dict['embeddings'], requires_grad=False)
        
        # Load frozen prototypes as buffers - use register_buffer() method!
        if state_dict.get('frozen_directions') is not None:
            model.register_buffer('frozen_directions', state_dict['frozen_directions'].cuda())
            model.register_buffer('frozen_biases', state_dict['frozen_biases'].cuda())
        
        # Load prototypes: directions as frozen buffer, biases as trainable parameter
        if state_dict.get('hyp_projector.classifier.prototype_direction') is not None:
            # Directions are frozen (buffer, not parameter)
            model.hyp_projector.classifier.prototype_direction = \
                state_dict['hyp_projector.classifier.prototype_direction'].to(model.hyp_projector.classifier.prototype_direction.device)
            model.hyp_projector.classifier.prototype_bias = nn.Parameter(
                state_dict['hyp_projector.classifier.prototype_bias'])
            print(f"  ✓ Prototypes loaded from checkpoint ({model.hyp_projector.classifier.num_classes} classes)")
            print(f"    Directions: FROZEN (buffer), Biases: trainable")
            print(f"    Biases: {[f'{b:.4f}' for b in model.hyp_projector.classifier.prototype_bias.tolist()]}")
        else:
            print(f"  ⚠ WARNING: No prototype_direction found in checkpoint — using random init!")
        return model
    
    # Training mode
    if prev_classes == 0:
        return model  # T1: train from scratch
    
    # T2+: Handle text embeddings
    part_a = state_dict.get('frozen_embeddings')
    part_b = state_dict.get('embeddings')
    if part_a is not None and part_b is not None:
        freeze_emb = torch.cat([part_a, part_b], dim=1)
    else:
        freeze_emb = part_a if part_a is not None else part_b
    
    if freeze_emb is not None:
        model.frozen_embeddings = nn.Parameter(freeze_emb)
        model.embeddings = nn.Parameter(model.text_feats[:, freeze_emb.shape[1]:, :])
    
    # T2+: Handle prototypes (directions + biases)
    dir_a = state_dict.get('frozen_directions')
    dir_b = state_dict.get('hyp_projector.classifier.prototype_direction')
    bias_a = state_dict.get('frozen_biases')
    bias_b = state_dict.get('hyp_projector.classifier.prototype_bias')
    
    if dir_a is not None and dir_b is not None:
        freeze_dirs = torch.cat([dir_a, dir_b], dim=0)
        freeze_biases = torch.cat([bias_a, bias_b], dim=0)
    else:
        freeze_dirs = dir_a if dir_a is not None else dir_b
        freeze_biases = bias_a if bias_a is not None else bias_b
    
    if freeze_dirs is not None:
        # Frozen prototypes are buffers — use register_buffer() method!
        model.register_buffer('frozen_directions', freeze_dirs.cuda())
        model.register_buffer('frozen_biases', freeze_biases.cuda())
        
        # Initialize new trainable prototypes for novel classes
        # NOTE: For T2+, should also use init_prototypes.py for novel class prototypes!
        print(f"  ⚠ T2+: Using random init for {current_classes} novel prototypes")
        print(f"     Consider using init_prototypes.py for better init!")
        model.hyp_projector.classifier.prototype_direction = nn.Parameter(
            torch.randn(current_classes, model.hyp_projector.out_dim))
        model.hyp_projector.classifier.prototype_bias = nn.Parameter(
            torch.zeros(current_classes))
    
    return model


class HypCustomYoloWorld(nn.Module):
    """
    Hyperbolic YOLO World with Horospherical Classification.
    
    Architecture:
    - Wraps frozen YOLO-World backbone/neck/head
    - HyperbolicProjector: FPN → Poincaré ball
    - HorosphericalClassifier: Busemann scores for classification
    - OOD detection: max horosphere score < threshold → unknown
    """
    
    def __init__(
        self,
        yolo_world_model,
        unknown_index,
        hyp_c=1.0,
        hyp_dim=256,
        clip_r=0.95,
        init_prototypes=None,  # Pre-computed from init_prototypes.py
        dispersion_weight=0.1,
        bias_reg_weight=0.1,
        compactness_weight=0.0,
    ):
        super().__init__()
        
        self.parent = yolo_world_model
        self.bbox_head = yolo_world_model.bbox_head
        self.unknown_index = unknown_index
        self.hyp_c = hyp_c
        self.hyp_dim = hyp_dim
        
        # num_classes for visualization script compatibility
        self.num_classes = unknown_index
        
        # TAL assignment labels
        self.tmp_labels = None
        
        # Text embeddings
        self.frozen_embeddings = None
        self.embeddings = None
        
        # Frozen prototypes from previous tasks (T2+)
        # Use register_buffer so they don't require gradients
        self.register_buffer('frozen_directions', None)
        self.register_buffer('frozen_biases', None)
        
        self._init_text_embedding()
        self._init_hyperbolic_projector(clip_r, init_prototypes)
        
        # Loss function with all regularization terms
        self.hyp_loss_fn = HorosphericalLoss(
            curvature=hyp_c, 
            dispersion_weight=dispersion_weight,
            bias_reg_weight=bias_reg_weight,
            compactness_weight=compactness_weight
        )
    
    def _init_text_embedding(self):
        with torch.no_grad():
            self.texts = copy.deepcopy(self.parent.texts)
            self.text_feats = self.parent.text_feats.clone()
            self.embeddings = nn.Parameter(self.text_feats)
    
    def _init_hyperbolic_projector(self, clip_r, init_prototypes=None):
        """
        Initialize hyperbolic projector.
        
        Parameters
        ----------
        clip_r : float
            Clip radius for ToPoincare
        init_prototypes : tensor, optional
            Pre-computed prototype directions from init_prototypes.py (K, hyp_dim)
        """
        self.hyp_projector = HyperbolicProjector(
            in_dims=[384, 768, 768],
            out_dim=self.hyp_dim,
            curvature=self.hyp_c,
            num_classes=self.unknown_index,
            clip_r=clip_r,
            riemannian=True,
            init_prototypes=init_prototypes
        )
    
    @property
    def prototypes(self):
        """All prototypes (frozen + trainable) on boundary."""
        trainable = self.hyp_projector.prototypes
        if self.frozen_directions is not None:
            R = 1.0 / (self.hyp_c ** 0.5)
            frozen = F.normalize(self.frozen_directions, dim=-1) * R
            return torch.cat([frozen, trainable], dim=0)
        return trainable
    
    @property
    def prototype_biases(self):
        """All biases (frozen + trainable)."""
        trainable = self.hyp_projector.prototype_bias
        if self.frozen_biases is not None:
            return torch.cat([self.frozen_biases, trainable], dim=0)
        return trainable
    
    def add_generic_text(self, class_names, generic_prompt='object', alpha=0.05):
        """Add generic 'object' embedding for unknown class."""
        if len(class_names) <= self.unknown_index:
            class_names.append(generic_prompt)
        self.parent.reparameterize([class_names])
        
        with torch.no_grad():
            self.texts = copy.deepcopy(self.parent.texts)
            self.text_feats = self.parent.text_feats.clone()
            
            # Merge embeddings
            if self.frozen_embeddings is not None and self.embeddings is not None:
                freeze = torch.cat([self.frozen_embeddings, self.embeddings], dim=1)
            else:
                freeze = self.frozen_embeddings if self.frozen_embeddings is not None else self.embeddings
            
            generic = self.text_feats[:, self.unknown_index, :]
            if alpha != 0:
                mean_emb = F.normalize(freeze, p=2, dim=2).mean(dim=1)
                generic = generic - alpha * mean_emb
            
            self.frozen_embeddings = nn.Parameter(torch.cat([freeze, generic.unsqueeze(1)], dim=1))
            self.embeddings = None
    
    def extract_feat(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        """Extract FPN features and project to Poincaré ball."""
        # Text features
        if self.frozen_embeddings is not None and self.embeddings is not None:
            txt_feats = torch.cat([self.frozen_embeddings, self.embeddings], dim=1)
        else:
            txt_feats = self.embeddings if self.embeddings is not None else self.frozen_embeddings
        
        # Image features
        img_feats = self.parent.backbone.forward_image(batch_inputs)
        txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)
        
        if self.parent.with_neck:
            img_feats = self.parent.neck(img_feats, txt_feats) if self.parent.mm_neck else self.parent.neck(img_feats)
        
        # Project to Poincaré ball
        hyp_embeddings = self.hyp_projector(img_feats)
        return img_feats, txt_feats, hyp_embeddings
    
    def compute_horosphere_scores(self, hyp_embeddings):
        """Compute horospherical scores using all prototypes."""
        # Get all prototypes and biases
        protos = self.prototypes
        biases = self.prototype_biases
        
        # Compute Busemann function
        if hyp_embeddings.dim() == 3:
            B, N, D = hyp_embeddings.shape
            x_flat = hyp_embeddings.reshape(B * N, D)
            B_vals = busemann(protos, x_flat, c=self.hyp_c)
            scores = (-B_vals + biases).reshape(B, N, -1)
        else:
            B_vals = busemann(protos, hyp_embeddings, c=self.hyp_c)
            scores = -B_vals + biases
        return scores
    
    def horospherical_loss(self, hyp_embeddings):
        """Compute horospherical CE loss with dispersion + bias regularization."""
        if self.tmp_labels is None:
            return hyp_embeddings.new_tensor(0.0)
        scores = self.compute_horosphere_scores(hyp_embeddings)
        return self.hyp_loss_fn(
            scores, self.tmp_labels,
            prototype_direction=self.hyp_projector.prototype_direction,
            frozen_directions=self.frozen_directions,
            prototype_bias=self.hyp_projector.prototype_bias,
            frozen_biases=self.frozen_biases
        )
    
    def horospherical_loss_with_breakdown(self, hyp_embeddings):
        """Loss with diagnostics."""
        if self.tmp_labels is None:
            return hyp_embeddings.new_tensor(0.0), {}
        scores = self.compute_horosphere_scores(hyp_embeddings)
        return self.hyp_loss_fn.forward_with_breakdown(
            scores, self.tmp_labels, 
            all_prototypes=self.prototypes, 
            all_biases=self.prototype_biases,
            prototype_direction=self.hyp_projector.prototype_direction,
            frozen_directions=self.frozen_directions,
            prototype_bias=self.hyp_projector.prototype_bias,
            frozen_biases=self.frozen_biases
        )
    
    def forward(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        """Forward pass — dispatches to head_loss.
        
        MUST be called through the DDP wrapper during training so that
        DDP's reducer properly sets up gradient synchronization hooks.
        """
        return self.head_loss(batch_inputs, batch_data_samples)

    def head_loss(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        """Compute YOLO + horospherical losses."""
        self.bbox_head.num_classes = self.text_feats[0].shape[0]
        self.bbox_head.assigner.num_classes = self.text_feats[0].shape[0]
        
        img_feats, txt_feats, hyp_embeddings = self.extract_feat(batch_inputs, batch_data_samples)
        self.tmp_labels = None
        head_losses = self.bbox_head_loss(img_feats, txt_feats, batch_data_samples)
        hyp_loss = self.horospherical_loss(hyp_embeddings)
        return head_losses, hyp_loss
    
    def head_loss_with_breakdown(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        """Losses with detailed breakdown."""
        self.bbox_head.num_classes = self.text_feats[0].shape[0]
        self.bbox_head.assigner.num_classes = self.text_feats[0].shape[0]
        
        img_feats, txt_feats, hyp_embeddings = self.extract_feat(batch_inputs, batch_data_samples)
        self.tmp_labels = None
        head_losses = self.bbox_head_loss(img_feats, txt_feats, batch_data_samples)
        hyp_loss, hyp_breakdown = self.horospherical_loss_with_breakdown(hyp_embeddings)
        return head_losses, hyp_loss, hyp_breakdown
    
    def bbox_head_loss(self, img_feats, txt_feats, batch_data_samples):
        """YOLO detection loss (extracts TAL assignments for hyp loss)."""
        batch_gt_instances, _, batch_img_metas = unpack_gt_instances(batch_data_samples)
        outs = self.bbox_head(img_feats, txt_feats)
        return self.compute_loss(outs[0], outs[1], outs[2], batch_gt_instances, batch_img_metas)
    
    def compute_loss(self, cls_scores, bbox_preds, bbox_dist_preds, batch_gt_instances, batch_img_metas):
        """YOLO loss computation with TAL assignment extraction."""
        num_imgs = len(batch_img_metas)
        
        # Update feature map sizes
        current_sizes = [s.shape[2:] for s in cls_scores]
        if current_sizes != self.bbox_head.featmap_sizes_train:
            self.bbox_head.featmap_sizes_train = current_sizes
            priors = self.bbox_head.prior_generator.grid_priors(
                current_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device, with_stride=True)
            self.bbox_head.num_level_priors = [len(p) for p in priors]
            self.bbox_head.flatten_priors_train = torch.cat(priors, dim=0)
            self.bbox_head.stride_tensor = self.bbox_head.flatten_priors_train[..., [2]]
        
        # Prepare GT
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels, gt_bboxes = gt_info[:, :, :1], gt_info[:, :, 1:]
        pad_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()
        
        # Flatten predictions
        flat_cls = torch.cat([s.permute(0,2,3,1).reshape(num_imgs, -1, self.bbox_head.num_classes) for s in cls_scores], 1)
        flat_bbox = torch.cat([b.permute(0,2,3,1).reshape(num_imgs, -1, 4) for b in bbox_preds], 1)
        flat_dist = torch.cat([d.reshape(num_imgs, -1, self.bbox_head.head_module.reg_max * 4) for d in bbox_dist_preds], 1)
        
        flat_bbox = self.bbox_head.bbox_coder.decode(
            self.bbox_head.flatten_priors_train[..., :2], flat_bbox, self.bbox_head.stride_tensor[..., 0])
        
        # TAL assignment
        assigned = self.bbox_head.assigner(
            flat_bbox.detach().type(gt_bboxes.dtype), flat_cls.detach().sigmoid(),
            self.bbox_head.flatten_priors_train, gt_labels, gt_bboxes, pad_flag)
        
        # Extract labels for hyperbolic loss
        max_vals, max_idx = assigned['assigned_scores'].max(dim=2)
        max_idx[max_vals <= 0] = -1
        self.tmp_labels = max_idx
        
        # Compute losses
        scores_sum = assigned['assigned_scores'].sum().clamp(min=1)
        loss_cls = self.bbox_head.loss_cls(flat_cls, assigned['assigned_scores']).sum() / scores_sum
        
        flat_bbox /= self.bbox_head.stride_tensor
        assigned['assigned_bboxes'] /= self.bbox_head.stride_tensor
        
        fg_mask = assigned['fg_mask_pre_prior']
        num_pos = fg_mask.sum()
        
        if num_pos > 0:
            mask4 = fg_mask.unsqueeze(-1).repeat(1, 1, 4)
            pred_pos = flat_bbox[mask4].reshape(-1, 4)
            target_pos = assigned['assigned_bboxes'][mask4].reshape(-1, 4)
            weight = assigned['assigned_scores'].sum(-1)[fg_mask].unsqueeze(-1)
            
            loss_bbox = self.bbox_head.loss_bbox(pred_pos, target_pos, weight=weight) / scores_sum
            
            dist_pos = flat_dist[fg_mask]
            ltrb = self.bbox_head.bbox_coder.encode(
                self.bbox_head.flatten_priors_train[..., :2] / self.bbox_head.stride_tensor,
                assigned['assigned_bboxes'], max_dis=self.bbox_head.head_module.reg_max - 1, eps=0.01)
            ltrb_pos = ltrb[mask4].reshape(-1, 4)
            loss_dfl = self.bbox_head.loss_dfl(
                dist_pos.reshape(-1, self.bbox_head.head_module.reg_max),
                ltrb_pos.reshape(-1), weight=weight.expand(-1, 4).reshape(-1), avg_factor=scores_sum)
        else:
            loss_bbox = flat_bbox.sum() * 0
            loss_dfl = flat_bbox.sum() * 0
        
        world_size = get_dist_info()[1] if self.bbox_head.world_size == -1 else self.bbox_head.world_size
        scale = num_imgs * world_size
        return dict(loss_cls=loss_cls * scale, loss_bbox=loss_bbox * scale, loss_dfl=loss_dfl * scale)
    
    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale=True):
        """Inference with horospherical OOD detection."""
        img_feats, txt_feats, hyp_embeddings = self.extract_feat(batch_inputs, batch_data_samples)
        self.parent.bbox_head.num_classes = txt_feats[0].shape[0]
        
        results = self.predict_by_feat(img_feats, txt_feats, batch_data_samples, hyp_embeddings, rescale)
        return self.parent.add_pred_to_datasample(batch_data_samples, results)
    
    def predict_by_feat(self, img_feats, txt_feats, batch_data_samples, hyp_embeddings, rescale=False):
        """Prediction with horosphere scoring."""
        batch_img_metas = [d.metainfo for d in batch_data_samples]
        outs = self.bbox_head(img_feats, txt_feats)
        
        cfg = copy.deepcopy(self.bbox_head.test_cfg)
        cfg.multi_label = cfg.multi_label and self.bbox_head.num_classes > 1
        num_imgs = len(batch_img_metas)
        
        # Setup priors
        sizes = [s.shape[2:] for s in outs[0]]
        if sizes != self.bbox_head.featmap_sizes:
            self.bbox_head.mlvl_priors = self.bbox_head.prior_generator.grid_priors(
                sizes, dtype=outs[0][0].dtype, device=outs[0][0].device)
            self.bbox_head.featmap_sizes = sizes
        
        priors = torch.cat(self.bbox_head.mlvl_priors)
        strides = torch.cat([priors.new_full((s.numel() * self.bbox_head.num_base_priors,), st)
                            for s, st in zip(sizes, self.bbox_head.featmap_strides)])
        
        # Flatten predictions
        flat_cls = torch.cat([s.permute(0,2,3,1).reshape(num_imgs, -1, self.bbox_head.num_classes) for s in outs[0]], 1).sigmoid()
        flat_bbox = torch.cat([b.permute(0,2,3,1).reshape(num_imgs, -1, 4) for b in outs[1]], 1)
        flat_bbox = self.bbox_head.bbox_coder.decode(priors[None], flat_bbox, strides)
        
        # Note: during inference, head returns only (cls, bbox), no objectness
        flat_obj = None
        if len(outs) > 2 and outs[2] is not None:
            flat_obj = torch.cat([o.permute(0,2,3,1).reshape(num_imgs, -1) for o in outs[2]], 1).sigmoid()
        
        results = []
        for idx, (bboxes, scores, img_meta) in enumerate(zip(flat_bbox, flat_cls, batch_img_metas)):
            obj = flat_obj[idx] if flat_obj is not None else None
            hyp_emb = hyp_embeddings[idx]
            
            # Score threshold filter
            if obj is not None and cfg.get('score_thr', -1) > 0 and not cfg.get('yolox_style', False):
                mask = obj > cfg.score_thr
                bboxes, scores, obj, hyp_emb = bboxes[mask], scores[mask], obj[mask], hyp_emb[mask]
            
            if obj is not None:
                scores = scores * obj[:, None]
            
            if scores.shape[0] == 0:
                empty = InstanceData(bboxes=bboxes, scores=scores[:, 0], labels=scores[:, 0].int(), ood_scores=scores[:, 0])
                results.append(empty)
                continue
            
            # Top-k selection
            if not cfg.multi_label:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep, res = filter_scores_and_topk(scores, cfg.score_thr, cfg.get('nms_pre', 100000), dict(labels=labels[:, 0]))
                labels = res['labels']
            else:
                scores, labels, keep, _ = filter_scores_and_topk(scores, cfg.score_thr, cfg.get('nms_pre', 100000))
            
            # Compute OOD scores (negative max horosphere score)
            kept_emb = hyp_emb[keep]
            horo_scores = self.compute_horosphere_scores(kept_emb)  # (N, K)
            max_horo, assigned_proto = horo_scores.max(dim=-1)  # max score & which prototype
            ood_scores = -max_horo  # Higher = more OOD
            
            result = InstanceData(
                scores=scores, labels=labels, bboxes=bboxes[keep],
                ood_scores=ood_scores,
                horo_max_scores=max_horo,       # max horosphere score per detection
                horo_assigned_proto=assigned_proto,  # which prototype gave max score
            )
            
            # Rescale
            if rescale:
                pad = img_meta.get('pad_param')
                if pad is not None:
                    result.bboxes -= result.bboxes.new_tensor([pad[2], pad[0], pad[2], pad[0]])
                result.bboxes /= result.bboxes.new_tensor(img_meta['scale_factor']).repeat(1, 2)
            
            if cfg.get('yolox_style', False):
                cfg.max_per_img = len(result)
            
            # NMS + post-process (InstanceData preserves ood_scores through indexing)
            result = self.bbox_head._bbox_post_process(result, cfg, rescale=False, with_nms=True, img_meta=img_meta)
            
            result.bboxes[:, 0::2].clamp_(0, img_meta['ori_shape'][1])
            result.bboxes[:, 1::2].clamp_(0, img_meta['ori_shape'][0])
            results.append(result)
        
        return results
    
    def enable_projector_grad(self, index):
        """Enable gradients for training."""
        if index == 0:
            # T1: Train all projector params
            for p in self.hyp_projector.parameters():
                p.requires_grad = True
            print("  [T1] Training projector convs + classifier biases (directions frozen as buffer)")
        else:
            # T2+: Freeze Conv, train only classifier biases
            for name, p in self.hyp_projector.named_parameters():
                p.requires_grad = 'classifier' in name
            print("  [T2+] Training only classifier biases (directions frozen as buffer)")
