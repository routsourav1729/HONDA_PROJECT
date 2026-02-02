"""
Hyperbolic Custom YOLO World for Open-World Object Detection.
Replaces MSCAL with hyperbolic prototype-based OOD detection.

Key differences from original CustomYoloWorld:
- Uses HyperbolicProjector instead of per-class ProjectionHead
- Learnable prototypes in Poincaré ball instead of anchor layers  
- Distance-based OOD detection instead of cosine similarity
- Hyperbolic contrastive loss instead of MSCAL loss
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
from mmdet.models.utils import (multi_apply, unpack_gt_instances,
                                filter_scores_and_topk)
from mmyolo.models.utils import gt_instances_preprocess
from mmengine.dist import get_dist_info

# Hyperbolic imports
from .hyperbolic import HyperbolicProjector, dist_matrix, expmap0, project, poincare_mean
from .hyperbolic.projector import HyperbolicContrastiveLoss


def load_hyp_ckpt(model, checkpoint_path, prev_classes, current_classes, eval=False):
    """
    Load checkpoint for hyperbolic model.
    
    Handles:
    - Text embeddings (frozen_embeddings, embeddings) - same as OVOW
    - Hyperbolic prototypes (frozen_prototypes, prototype_tangent)
    - HyperbolicProjector weights
    
    Parameters
    ----------
    model : HypCustomYoloWorld
        Model to load weights into
    checkpoint_path : str
        Path to checkpoint
    prev_classes : int
        Number of previously introduced classes (to freeze)
    current_classes : int
        Number of current classes being trained
    eval : bool
        If True, load for evaluation (load all prototypes)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    # Handle both checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Load everything except embeddings and prototypes (handle separately)
    exclude_keys = ['embeddings', 'frozen_embeddings', 'prototype_tangent', 'frozen_prototypes']
    partial_state_dict = {k: v for k, v in state_dict.items() 
                          if not any(ex in k for ex in exclude_keys)}
    model.load_state_dict(partial_state_dict, strict=False)
    
    if eval:
        # Evaluation mode: load all prototypes and embeddings as-is
        if state_dict.get('frozen_embeddings') is not None:
            model.frozen_embeddings = nn.Parameter(state_dict['frozen_embeddings'])
        else:
            model.frozen_embeddings = None
            
        if state_dict.get('embeddings') is not None:
            model.embeddings = nn.Parameter(state_dict['embeddings'])
        else:
            model.embeddings = None
            
        # Load prototypes
        if state_dict.get('frozen_prototypes') is not None:
            model.frozen_prototypes = nn.Parameter(state_dict['frozen_prototypes'])
        else:
            model.frozen_prototypes = None
            
        if state_dict.get('hyp_projector.prototype_tangent') is not None:
            model.hyp_projector.prototype_tangent = nn.Parameter(
                state_dict['hyp_projector.prototype_tangent']
            )
        return model
    
    # Training mode
    if prev_classes == 0:
        # T1: No previous classes, train from scratch
        return model
    
    # T2+: Merge previous prototypes into frozen, init new trainable ones
    # Handle text embeddings (same as OVOW)
    part_a = state_dict.get('frozen_embeddings')
    part_b = state_dict.get('embeddings')
    if part_a is not None and part_b is not None:
        freeze_emb = torch.cat([part_a, part_b], dim=1)
    elif part_a is None:
        freeze_emb = part_b
    else:
        freeze_emb = part_a
    
    if freeze_emb is not None:
        length = freeze_emb.shape[1]
        model.frozen_embeddings = nn.Parameter(freeze_emb)
        model.embeddings = nn.Parameter(model.text_feats[:, length:, :])
    
    # Handle prototypes
    proto_a = state_dict.get('frozen_prototypes')
    proto_b = state_dict.get('hyp_projector.prototype_tangent')
    
    if proto_a is not None and proto_b is not None:
        freeze_proto = torch.cat([proto_a, proto_b], dim=0)
    elif proto_a is None:
        freeze_proto = proto_b
    else:
        freeze_proto = proto_a
    
    if freeze_proto is not None:
        model.frozen_prototypes = nn.Parameter(freeze_proto)
        # Initialize new prototypes for current classes
        model.hyp_projector.prototype_tangent = nn.Parameter(
            torch.randn(current_classes, model.hyp_projector.out_dim) * 0.01
        )
    
    return model


class HypCustomYoloWorld(nn.Module):
    """
    Hyperbolic YOLO World for Open-World Object Detection.
    
    Architecture:
    - Wraps YOLO-World backbone/neck/head (frozen)
    - Adds HyperbolicProjector for visual features → Poincaré ball
    - Learnable prototypes in Poincaré ball for each class
    - Distance-based OOD detection at inference
    
    Parameters
    ----------
    yolo_world_model : nn.Module
        Pre-trained YOLO-World model
    unknown_index : int
        Index for unknown class (= num_known_classes)
    hyp_c : float
        Poincaré ball curvature (default: 0.1)
    hyp_dim : int
        Hyperbolic embedding dimension (default: 256)
    clip_r : float
        Clip radius for ToPoincare (default: 2.3)
    temperature : float
        Temperature for contrastive loss (default: 0.1)
    separation_weight : float
        Weight for inter-prototype separation loss (default: 0.5)
    boundary_weight : float
        Weight for boundary push loss (default: 0.1)
    min_proto_dist : float
        Minimum desired distance between prototypes (default: 2.0)
    target_norm : float
        Target norm for prototypes (default: 0.9)
    """
    
    def __init__(
        self,
        yolo_world_model,
        unknown_index,
        hyp_c=0.1,
        hyp_dim=256,
        clip_r=2.3,
        temperature=0.1,
        separation_weight=0.5,
        boundary_weight=0.1,
        min_proto_dist=2.0,
        target_norm=0.9
    ):
        super(HypCustomYoloWorld, self).__init__()
        
        # Store parent model components
        self.parent = yolo_world_model
        self.bbox_head = yolo_world_model.bbox_head
        self.unknown_index = unknown_index
        
        # Hyperbolic parameters
        self.hyp_c = hyp_c
        self.hyp_dim = hyp_dim
        self.clip_r = clip_r
        self.temperature = temperature
        
        # For TAL assignment labels
        self.tmp_labels = None
        
        # Text embeddings (like OVOW)
        self.frozen_embeddings = None
        self.embeddings = None
        
        # Frozen prototypes from previous tasks (T2+)
        self.frozen_prototypes = None
        
        # Initialize components
        self._initialize_text_embedding()
        self._initialize_hyperbolic_projector()
        
        # Loss function with prototype separation
        self.hyp_loss_fn = HyperbolicContrastiveLoss(
            temperature=temperature,
            curvature=hyp_c,
            separation_weight=separation_weight,
            boundary_weight=boundary_weight,
            min_proto_dist=min_proto_dist,
            target_norm=target_norm
        )
    
    def _initialize_text_embedding(self):
        """Initialize text embeddings from parent model."""
        with torch.no_grad():
            self.texts = copy.deepcopy(self.parent.texts)
            self.text_feats = self.parent.text_feats.clone()
            self.embeddings = nn.Parameter(self.text_feats)
    
    def _initialize_hyperbolic_projector(self):
        """Initialize hyperbolic projector with prototypes."""
        self.hyp_projector = HyperbolicProjector(
            in_dims=[384, 768, 768],  # YOLO-World XL FPN dims
            out_dim=self.hyp_dim,
            curvature=self.hyp_c,
            num_classes=self.unknown_index,
            clip_r=self.clip_r,
            riemannian=True
        )
    
    def update_unknown_index(self, unknown_index):
        """Update the unknown class index."""
        self.unknown_index = unknown_index
    
    @property
    def prototypes(self):
        """
        Get all prototypes (frozen + trainable) in Poincaré ball.
        
        Returns
        -------
        tensor
            Prototypes of shape (K, hyp_dim) in Poincaré ball
        """
        # Get trainable prototypes from projector
        trainable_protos = self.hyp_projector.prototypes  # Already in ball
        
        if self.frozen_prototypes is not None:
            # Project frozen prototypes to ball (stored in tangent space)
            frozen_protos = expmap0(self.frozen_prototypes, c=self.hyp_c)
            frozen_protos = project(frozen_protos, c=self.hyp_c)
            return torch.cat([frozen_protos, trainable_protos], dim=0)
        
        return trainable_protos
    
    def add_generic_text(self, class_names, generic_prompt='object', alpha=0.05):
        """
        Add generic 'object' embedding for OOD detection.
        Same as OVOW but using distance-based approach.
        """
        if len(class_names) <= self.unknown_index:
            class_names.append(generic_prompt)
        classnames = [class_names]
        self.parent.reparameterize(classnames)
        
        with torch.no_grad():
            self.texts = copy.deepcopy(self.parent.texts)
            self.text_feats = self.parent.text_feats.clone()
            
            part_a = self.frozen_embeddings
            part_b = self.embeddings
            if part_a is not None and part_b is not None:
                freeze = torch.cat([part_a, part_b], dim=1)
            elif part_a is None:
                freeze = part_b
            else:
                freeze = part_a
            
            generic_embedding = self.text_feats[:, self.unknown_index, :]
            
            if alpha != 0:
                normalized_embedding = F.normalize(freeze, p=2, dim=2)
                normalized_embedding = normalized_embedding.mean(dim=1)
                generic_embedding = generic_embedding - alpha * normalized_embedding
            
            freeze = torch.cat([freeze, generic_embedding.unsqueeze(0)], dim=1)
            self.frozen_embeddings = nn.Parameter(freeze)
            self.embeddings = None
    
    def extract_feat(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        """
        Extract features and compute hyperbolic embeddings.
        
        Returns
        -------
        img_feats : tuple
            FPN features from backbone/neck
        txt_feats : tensor
            Text embeddings
        hyp_embeddings : tensor
            Hyperbolic embeddings of shape (B, N_anchors, hyp_dim)
        """
        # Get text features
        if self.frozen_embeddings is not None and self.embeddings is not None:
            txt_feats = torch.cat([self.frozen_embeddings, self.embeddings], dim=1)
        elif self.embeddings is not None:
            txt_feats = self.embeddings
        else:
            txt_feats = self.frozen_embeddings
        
        # Extract image features
        img_feats = self.parent.backbone.forward_image(batch_inputs)
        txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)
        
        # Apply neck
        if self.parent.with_neck:
            if self.parent.mm_neck:
                img_feats = self.parent.neck(img_feats, txt_feats)
            else:
                img_feats = self.parent.neck(img_feats)
        
        # Compute hyperbolic embeddings
        hyp_embeddings = self.hyp_projector(img_feats)  # (B, N_anchors, hyp_dim)
        
        return img_feats, txt_feats, hyp_embeddings
    
    def hyperbolic_contrastive_loss(self, hyp_embeddings: Tensor) -> Tensor:
        """
        Compute hyperbolic contrastive loss.
        
        Uses TAL assignments (self.tmp_labels) to supervise which prototype
        each anchor should be close to.
        
        Parameters
        ----------
        hyp_embeddings : tensor
            Hyperbolic embeddings of shape (B, N_anchors, hyp_dim)
        
        Returns
        -------
        tensor
            Scalar loss value
        """
        if self.tmp_labels is None:
            return hyp_embeddings.new_tensor(0.0)
        
        return self.hyp_loss_fn(hyp_embeddings, self.tmp_labels, self.prototypes)
    
    def hyperbolic_loss_with_breakdown(self, hyp_embeddings: Tensor):
        """
        Compute hyperbolic loss with detailed breakdown for logging.
        
        Returns
        -------
        total_loss : tensor
        loss_dict : dict with individual components
        """
        if self.tmp_labels is None:
            return hyp_embeddings.new_tensor(0.0), {}
        
        return self.hyp_loss_fn.forward_with_breakdown(
            hyp_embeddings, self.tmp_labels, self.prototypes
        )
    
    def head_loss(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        """
        Calculate losses from a batch of inputs.
        
        Returns
        -------
        head_losses : dict
            YOLO detection losses (cls, bbox, dfl)
        hyp_loss : tensor
            Hyperbolic contrastive loss
        """
        self.bbox_head.num_classes = self.text_feats[0].shape[0]
        self.bbox_head.assigner.num_classes = self.text_feats[0].shape[0]
        
        img_feats, txt_feats, hyp_embeddings = self.extract_feat(batch_inputs, batch_data_samples)
        
        self.tmp_labels = None
        head_losses = self.bbox_head_loss(img_feats, txt_feats, batch_data_samples)
        hyp_loss = self.hyperbolic_contrastive_loss(hyp_embeddings)
        
        return head_losses, hyp_loss
    
    def head_loss_with_breakdown(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        """
        Calculate losses with hyperbolic breakdown for detailed logging.
        
        Returns
        -------
        head_losses : dict
        hyp_loss : tensor
        hyp_breakdown : dict with individual loss components
        """
        self.bbox_head.num_classes = self.text_feats[0].shape[0]
        self.bbox_head.assigner.num_classes = self.text_feats[0].shape[0]
        
        img_feats, txt_feats, hyp_embeddings = self.extract_feat(batch_inputs, batch_data_samples)
        
        self.tmp_labels = None
        head_losses = self.bbox_head_loss(img_feats, txt_feats, batch_data_samples)
        hyp_loss, hyp_breakdown = self.hyperbolic_loss_with_breakdown(hyp_embeddings)
        
        return head_losses, hyp_loss, hyp_breakdown
    
    def bbox_head_loss(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
                       batch_data_samples: Union[list, dict]) -> dict:
        """Perform forward propagation and loss calculation."""
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = unpack_gt_instances(
            batch_data_samples
        )
        outs = self.bbox_head(img_feats, txt_feats)
        losses = self.dev_loss_by_feat(outs[0], outs[1], outs[2], 
                                       batch_gt_instances, batch_img_metas)
        return losses
    
    def dev_loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict]) -> dict:
        """Compute YOLO losses and extract TAL assignments for hyperbolic loss."""
        num_imgs = len(batch_img_metas)

        current_featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        if current_featmap_sizes != self.bbox_head.featmap_sizes_train:
            self.bbox_head.featmap_sizes_train = current_featmap_sizes
            mlvl_priors_with_stride = self.bbox_head.prior_generator.grid_priors(
                self.bbox_head.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True
            )
            self.bbox_head.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.bbox_head.flatten_priors_train = torch.cat(mlvl_priors_with_stride, dim=0)
            self.bbox_head.stride_tensor = self.bbox_head.flatten_priors_train[..., [2]]

        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.bbox_head.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_pred_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.bbox_head.head_module.reg_max * 4)
            for bbox_pred_org in bbox_dist_preds
        ]

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_head.bbox_coder.decode(
            self.bbox_head.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.bbox_head.stride_tensor[..., 0]
        )

        assigned_result = self.bbox_head.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(), self.bbox_head.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag
        )

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']

        # Extract TAL assignments for hyperbolic loss
        max_values, max_indices = assigned_scores.max(dim=2)
        max_indices[max_values <= 0] = -1  # Background
        self.tmp_labels = max_indices

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        loss_cls = self.bbox_head.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum

        assigned_bboxes /= self.bbox_head.stride_tensor
        flatten_pred_bboxes /= self.bbox_head.stride_tensor

        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = self.bbox_head.loss_bbox(
                pred_bboxes_pos, assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum

            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_head.bbox_coder.encode(
                self.bbox_head.flatten_priors_train[..., :2] / self.bbox_head.stride_tensor,
                assigned_bboxes,
                max_dis=self.bbox_head.head_module.reg_max - 1,
                eps=0.01
            )
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.bbox_head.loss_dfl(
                pred_dist_pos.reshape(-1, self.bbox_head.head_module.reg_max),
                assigned_ltrb_pos.reshape(-1),
                weight=bbox_weight.expand(-1, 4).reshape(-1),
                avg_factor=assigned_scores_sum
            )
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
        
        if self.bbox_head.world_size == -1:
            _, world_size = get_dist_info()
        else:
            world_size = self.bbox_head.world_size
        
        return dict(
            loss_cls=loss_cls * num_imgs * world_size,
            loss_bbox=loss_bbox * num_imgs * world_size,
            loss_dfl=loss_dfl * num_imgs * world_size
        )
    
    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """
        Predict results with hyperbolic OOD detection.
        
        Returns predictions with hyp_distances field for OOD scoring.
        """
        img_feats, txt_feats, hyp_embeddings = self.extract_feat(batch_inputs, batch_data_samples)
        
        self.parent.bbox_head.num_classes = txt_feats[0].shape[0]
        results_list = self.bbox_head_pred(
            img_feats, txt_feats, batch_data_samples, 
            hyp_embeddings, rescale=rescale
        )
        
        batch_data_samples = self.parent.add_pred_to_datasample(
            batch_data_samples, results_list
        )
        return batch_data_samples
    
    def bbox_head_pred(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
                       batch_data_samples: SampleList, hyp_embeddings: Tensor,
                       rescale: bool = False) -> InstanceList:
        """Perform prediction with hyperbolic distance computation."""
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]
        outs = self.bbox_head(img_feats, txt_feats)
        predictions = self.bbox_head_predict_by_feat(
            *outs, 
            hyp_embeddings=hyp_embeddings,
            batch_img_metas=batch_img_metas,
            rescale=rescale
        )
        return predictions
    
    def bbox_head_predict_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            objectnesses: Optional[List[Tensor]] = None,
            batch_img_metas: Optional[List[dict]] = None,
            cfg: Optional[ConfigDict] = None,
            hyp_embeddings: Optional[Tensor] = None,
            rescale: bool = True,
            with_nms: bool = True) -> List[InstanceData]:
        """Predict with hyperbolic distance computation for OOD detection."""
        assert len(cls_scores) == len(bbox_preds)
        
        with_objectnesses = objectnesses is not None
        if with_objectnesses:
            assert len(cls_scores) == len(objectnesses)

        cfg = self.bbox_head.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.bbox_head.num_classes > 1
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        if featmap_sizes != self.bbox_head.featmap_sizes:
            self.bbox_head.mlvl_priors = self.bbox_head.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device
            )
            self.bbox_head.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.bbox_head.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * self.bbox_head.num_base_priors,), stride)
            for featmap_size, stride in zip(featmap_sizes, self.bbox_head.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.bbox_head.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_head.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride
        )

        if with_objectnesses:
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        else:
            flatten_objectness = [None for _ in range(num_imgs)]

        results_list = []
        for batch_idx, (bboxes, scores, objectness, img_meta) in enumerate(
                zip(flatten_decoded_bboxes, flatten_cls_scores, 
                    flatten_objectness, batch_img_metas)):
            
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            pad_param = img_meta.get('pad_param', None)

            score_thr = cfg.get('score_thr', -1)
            if objectness is not None and score_thr > 0 and not cfg.get('yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]
                # Also filter hyperbolic embeddings
                if hyp_embeddings is not None:
                    batch_hyp_emb = hyp_embeddings[batch_idx][conf_inds]
                else:
                    batch_hyp_emb = None
            else:
                batch_hyp_emb = hyp_embeddings[batch_idx] if hyp_embeddings is not None else None

            if objectness is not None:
                scores *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                empty_results.hyp_distances = scores[:, 0]
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get('nms_pre', 100000)

            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores, score_thr, nms_pre,
                    results=dict(labels=labels[:, 0])
                )
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre
                )
            
            # Get hyperbolic distances for kept predictions
            if batch_hyp_emb is not None:
                kept_hyp_emb = batch_hyp_emb[keep_idxs]  # (N_kept, hyp_dim)
                # Compute distance to all prototypes
                distances = dist_matrix(kept_hyp_emb, self.prototypes, c=self.hyp_c)
                min_distances = distances.min(dim=-1).values  # (N_kept,)
            else:
                min_distances = scores.new_zeros(scores.shape[0])

            results = InstanceData(
                scores=scores,
                labels=labels,
                bboxes=bboxes[keep_idxs],
                hyp_distances=min_distances
            )

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor([
                        pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                    ])
                results.bboxes /= results.bboxes.new_tensor(scale_factor).repeat((1, 2))

            if cfg.get('yolox_style', False):
                cfg.max_per_img = len(results)

            results = self.bbox_head._bbox_post_process(
                results=results, cfg=cfg, rescale=False, 
                with_nms=with_nms, img_meta=img_meta
            )
            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        
        return results_list
    
    def enable_projector_grad(self, index):
        """
        Enable gradients for trainable components.
        
        For T1 (index=0): Train ALL projector parameters (Conv + prototypes)
        For T2+ (index>0): Freeze Conv layers, only train prototypes
        
        Parameters
        ----------
        index : int
            Number of previously introduced classes (0 for T1)
        """
        if index == 0:
            # T1: Train everything in projector
            for param in self.hyp_projector.parameters():
                param.requires_grad = True
            print(f"  [T1] Training ALL projector parameters")
        else:
            # T2+: Freeze Conv layers, only train prototypes
            for name, param in self.hyp_projector.named_parameters():
                if 'prototype_tangent' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print(f"  [T2+] Freezing projector Conv, training only prototypes")
