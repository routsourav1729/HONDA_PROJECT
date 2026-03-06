"""
Hyperbolic Custom YOLO World with Geodesic Prototypical Classification.

Uses geodesic distance to interior prototypes in Poincare ball for OOD detection.
Prototypes live INSIDE the ball (||z_k|| < R), not on the boundary.
Classification: softmax over -d^2_B(x, z_k)
OOD: min_k d^2_B(x, z_k) > threshold -> unknown
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

from .hyperbolic import HyperbolicProjector
from .hyperbolic.projector import GeodesicPrototypeLoss


def load_hyp_ckpt(model, checkpoint_path, prev_classes, current_classes, eval=False):
    """
    Load checkpoint for geodesic prototype model.

    Handles:
    - Text embeddings (frozen_embeddings, embeddings)
    - Interior prototypes (frozen_prototypes, classifier.prototypes)

    Note: In geodesic framework, there are no bias terms or prototype_direction.
    Prototypes are nn.Parameter points inside the ball.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Keys to handle separately
    exclude_keys = ['embeddings', 'frozen_embeddings',
                    'frozen_prototypes',
                    # Legacy keys (backward compat with old checkpoints)
                    'frozen_directions', 'frozen_biases',
                    'prototype_direction', 'prototype_bias',
                    # New keys
                    'classifier.prototypes']
    partial_state_dict = {k: v for k, v in state_dict.items()
                          if not any(ex in k for ex in exclude_keys)}
    model.load_state_dict(partial_state_dict, strict=False)

    if eval:
        # Evaluation: load all embeddings and prototypes
        if state_dict.get('frozen_embeddings') is not None:
            model.frozen_embeddings = nn.Parameter(state_dict['frozen_embeddings'], requires_grad=False)
        if state_dict.get('embeddings') is not None:
            model.embeddings = nn.Parameter(state_dict['embeddings'], requires_grad=False)

        # Load frozen prototypes (from previous tasks)
        if state_dict.get('frozen_prototypes') is not None:
            model.register_buffer('frozen_prototypes', state_dict['frozen_prototypes'].cuda())
        elif state_dict.get('frozen_directions') is not None:
            # Legacy: convert boundary directions to interior prototypes
            frozen_dirs = state_dict['frozen_directions']
            frozen_protos = F.normalize(frozen_dirs, dim=-1) * 0.4
            model.register_buffer('frozen_prototypes', frozen_protos.cuda())
            print(f"  ! Legacy: converted frozen_directions to interior prototypes (norm=0.4)")

        # Load classifier prototypes
        proto_key = 'hyp_projector.classifier.prototypes'
        if state_dict.get(proto_key) is not None:
            saved_protos = state_dict[proto_key]
            device = model.hyp_projector.classifier.prototypes.device
            if isinstance(model.hyp_projector.classifier.prototypes, nn.Parameter):
                model.hyp_projector.classifier.prototypes.data = saved_protos.to(device)
            else:
                model.hyp_projector.classifier.prototypes = saved_protos.to(device)
            is_param = isinstance(model.hyp_projector.classifier.prototypes, nn.Parameter)
            proto_norms = saved_protos.norm(dim=-1)
            print(f"  + Prototypes loaded ({model.hyp_projector.classifier.num_classes} classes, param={is_param})")
            print(f"    Norms: mean={proto_norms.mean():.4f}, min={proto_norms.min():.4f}, max={proto_norms.max():.4f}")
        elif state_dict.get('hyp_projector.classifier.prototype_direction') is not None:
            # Legacy: convert boundary directions to interior prototypes
            dirs = state_dict['hyp_projector.classifier.prototype_direction']
            protos = F.normalize(dirs, dim=-1) * 0.4
            device = model.hyp_projector.classifier.prototypes.device
            if isinstance(model.hyp_projector.classifier.prototypes, nn.Parameter):
                model.hyp_projector.classifier.prototypes.data = protos.to(device)
            else:
                model.hyp_projector.classifier.prototypes = protos.to(device)
            print(f"  ! Legacy: converted prototype_direction to interior prototypes (norm=0.4)")
        else:
            print(f"  ! WARNING: No prototypes found in checkpoint -- using random init!")
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

    # T2+: Handle prototypes
    proto_key = 'hyp_projector.classifier.prototypes'
    protos_a = state_dict.get('frozen_prototypes')
    protos_b = state_dict.get(proto_key)

    # Legacy fallback
    if protos_a is None and state_dict.get('frozen_directions') is not None:
        protos_a = F.normalize(state_dict['frozen_directions'], dim=-1) * 0.4
    if protos_b is None and state_dict.get('hyp_projector.classifier.prototype_direction') is not None:
        protos_b = F.normalize(state_dict['hyp_projector.classifier.prototype_direction'], dim=-1) * 0.4

    if protos_a is not None and protos_b is not None:
        freeze_protos = torch.cat([protos_a, protos_b], dim=0)
    else:
        freeze_protos = protos_a if protos_a is not None else protos_b

    if freeze_protos is not None:
        model.register_buffer('frozen_prototypes', freeze_protos.cuda())

        # Check if classifier needs resizing for novel classes
        existing = model.hyp_projector.classifier.prototypes
        is_param = isinstance(existing, nn.Parameter)
        if existing.shape[0] == current_classes:
            print(f"  + T2+: Novel prototypes from init_protos ({current_classes} classes, param={is_param})")
        else:
            print(f"  ! T2+: Resizing classifier for {current_classes} novel classes (random init)")
            new_protos = F.normalize(torch.randn(current_classes, model.hyp_projector.out_dim), dim=-1) * 0.4
            if model.trainable_prototypes:
                model.hyp_projector.classifier.prototypes = nn.Parameter(new_protos)
            else:
                model.hyp_projector.classifier.register_buffer('prototypes', new_protos)

    return model


class HypCustomYoloWorld(nn.Module):
    """
    Hyperbolic YOLO World with Geodesic Prototypical Classification.

    Architecture:
    - Wraps frozen YOLO-World backbone/neck/head
    - HyperbolicProjector: FPN -> Poincare ball (+ pre-expmap norms for L_reg)
    - GeodesicPrototypeClassifier: -d^2_B scores for classification
    - OOD detection: min_k d^2_B > adaptive threshold -> unknown
    """

    def __init__(
        self,
        yolo_world_model,
        unknown_index,
        hyp_c=1.0,
        hyp_dim=64,
        clip_r=2.0,
        num_classifier_classes=None,
        init_prototypes=None,
        trainable_prototypes=True,
        bi_lipschitz=False,
        prototype_init_norm=0.4,
        # Geodesic loss weights
        ce_weight=1.0,
        class_balance_smoothing=0.5,
        beta_reg=0.1,
        lambda_sep=1.0,
        sep_margin=1.0,
    ):
        super().__init__()

        self.parent = yolo_world_model
        self.bbox_head = yolo_world_model.bbox_head
        self.unknown_index = unknown_index
        self.hyp_c = hyp_c
        self.hyp_dim = hyp_dim
        self.trainable_prototypes = trainable_prototypes
        self.bi_lipschitz = bi_lipschitz

        self.num_classes = unknown_index

        self._classifier_num_classes = num_classifier_classes if num_classifier_classes is not None else unknown_index

        # TAL assignment labels
        self.tmp_labels = None

        # Text embeddings
        self.frozen_embeddings = None
        self.embeddings = None

        # Frozen prototypes from previous tasks (T2+)
        self.register_buffer('frozen_prototypes', None)

        self._init_text_embedding()
        self._init_hyperbolic_projector(clip_r, init_prototypes, prototype_init_norm)
        print(f"  Classifier: {self._classifier_num_classes} classes, {unknown_index} total")

        # Geodesic prototype loss
        self.hyp_loss_fn = GeodesicPrototypeLoss(
            curvature=hyp_c,
            ce_weight=ce_weight,
            class_balance_smoothing=class_balance_smoothing,
            beta_reg=beta_reg,
            lambda_sep=lambda_sep,
            sep_margin=sep_margin,
        )

    def _init_text_embedding(self):
        with torch.no_grad():
            self.texts = copy.deepcopy(self.parent.texts)
            self.text_feats = self.parent.text_feats.clone()
            self.embeddings = nn.Parameter(self.text_feats)

    def _init_hyperbolic_projector(self, clip_r, init_prototypes=None, prototype_init_norm=0.4):
        """Initialize hyperbolic projector with geodesic classifier."""
        self.hyp_projector = HyperbolicProjector(
            in_dims=[384, 768, 768],
            out_dim=self.hyp_dim,
            curvature=self.hyp_c,
            num_classes=self._classifier_num_classes,
            clip_r=clip_r,
            riemannian=True,
            init_prototypes=init_prototypes,
            trainable_prototypes=self.trainable_prototypes,
            bi_lipschitz=getattr(self, 'bi_lipschitz', False),
            prototype_init_norm=prototype_init_norm,
        )

    @property
    def prototypes(self):
        """All prototypes (frozen + trainable) inside the ball."""
        trainable = self.hyp_projector.prototypes
        if self.frozen_prototypes is not None:
            return torch.cat([self.frozen_prototypes, trainable], dim=0)
        return trainable

    def add_generic_text(self, class_names, generic_prompt='object', alpha=0.05):
        """Add generic 'object' embedding for unknown class."""
        if len(class_names) <= self.unknown_index:
            class_names.append(generic_prompt)
        self.parent.reparameterize([class_names])

        with torch.no_grad():
            self.texts = copy.deepcopy(self.parent.texts)
            self.text_feats = self.parent.text_feats.clone()

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
        """Extract FPN features and project to Poincare ball."""
        if self.frozen_embeddings is not None and self.embeddings is not None:
            txt_feats = torch.cat([self.frozen_embeddings, self.embeddings], dim=1)
        else:
            txt_feats = self.embeddings if self.embeddings is not None else self.frozen_embeddings

        img_feats = self.parent.backbone.forward_image(batch_inputs)
        txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)

        if self.parent.with_neck:
            img_feats = self.parent.neck(img_feats, txt_feats) if self.parent.mm_neck else self.parent.neck(img_feats)

        # Projector returns (poincare_embeddings, pre_expmap_norms)
        hyp_embeddings, pre_expmap_norms = self.hyp_projector(img_feats)
        return img_feats, txt_feats, hyp_embeddings, pre_expmap_norms

    def compute_geodesic_scores(self, hyp_embeddings):
        """Compute geodesic scores using all prototypes (frozen + trainable)."""
        protos = self.prototypes  # (K_total, D)
        K = protos.shape[0]

        if hyp_embeddings.dim() == 3:
            B, N, D = hyp_embeddings.shape
            x_flat = hyp_embeddings.reshape(B * N, D)
            # Pairwise distances
            x_exp = x_flat.unsqueeze(1).expand(B * N, K, D)
            p_exp = protos.unsqueeze(0).expand(B * N, K, D)
            from .hyperbolic import pmath
            dists = pmath.dist(x_exp, p_exp, c=self.hyp_c)  # (B*N, K)
            scores = -dists.pow(2)
            return scores.reshape(B, N, K)
        else:
            N, D = hyp_embeddings.shape
            x_exp = hyp_embeddings.unsqueeze(1).expand(N, K, D)
            p_exp = protos.unsqueeze(0).expand(N, K, D)
            from .hyperbolic import pmath
            dists = pmath.dist(x_exp, p_exp, c=self.hyp_c)
            return -dists.pow(2)

    def geodesic_loss(self, hyp_embeddings, pre_expmap_norms):
        """Compute geodesic prototype loss (CE + L_reg + L_sep)."""
        if self.tmp_labels is None:
            return hyp_embeddings.new_tensor(0.0)
        scores = self.compute_geodesic_scores(hyp_embeddings)
        loss, _ = self.hyp_loss_fn(
            embeddings=hyp_embeddings,
            scores=scores,
            labels=self.tmp_labels,
            prototypes=self.prototypes,
            pre_expmap_norms=pre_expmap_norms,
        )
        return loss

    def geodesic_loss_with_breakdown(self, hyp_embeddings, pre_expmap_norms):
        """Geodesic loss with diagnostics."""
        if self.tmp_labels is None:
            return hyp_embeddings.new_tensor(0.0), {}
        scores = self.compute_geodesic_scores(hyp_embeddings)
        loss, loss_dict = self.hyp_loss_fn(
            embeddings=hyp_embeddings,
            scores=scores,
            labels=self.tmp_labels,
            prototypes=self.prototypes,
            pre_expmap_norms=pre_expmap_norms,
        )
        return loss, loss_dict

    def forward(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        """Forward pass -- dispatches to head_loss."""
        return self.head_loss(batch_inputs, batch_data_samples)

    def head_loss(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        """Compute YOLO + geodesic losses."""
        self.bbox_head.num_classes = self.text_feats[0].shape[0]
        self.bbox_head.assigner.num_classes = self.text_feats[0].shape[0]

        img_feats, txt_feats, hyp_embeddings, pre_expmap_norms = self.extract_feat(batch_inputs, batch_data_samples)
        self.tmp_labels = None
        head_losses = self.bbox_head_loss(img_feats, txt_feats, batch_data_samples)
        hyp_loss = self.geodesic_loss(hyp_embeddings, pre_expmap_norms)
        return head_losses, hyp_loss

    def head_loss_with_breakdown(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        """Losses with detailed breakdown."""
        self.bbox_head.num_classes = self.text_feats[0].shape[0]
        self.bbox_head.assigner.num_classes = self.text_feats[0].shape[0]

        img_feats, txt_feats, hyp_embeddings, pre_expmap_norms = self.extract_feat(batch_inputs, batch_data_samples)
        self.tmp_labels = None
        head_losses = self.bbox_head_loss(img_feats, txt_feats, batch_data_samples)
        hyp_loss, hyp_breakdown = self.geodesic_loss_with_breakdown(hyp_embeddings, pre_expmap_norms)
        return head_losses, hyp_loss, hyp_breakdown

    def bbox_head_loss(self, img_feats, txt_feats, batch_data_samples):
        """YOLO detection loss (extracts TAL assignments for hyp loss)."""
        batch_gt_instances, _, batch_img_metas = unpack_gt_instances(batch_data_samples)
        outs = self.bbox_head(img_feats, txt_feats)
        return self.compute_loss(outs[0], outs[1], outs[2], batch_gt_instances, batch_img_metas)

    def compute_loss(self, cls_scores, bbox_preds, bbox_dist_preds, batch_gt_instances, batch_img_metas):
        """YOLO loss computation with TAL assignment extraction."""
        num_imgs = len(batch_img_metas)

        current_sizes = [s.shape[2:] for s in cls_scores]
        if current_sizes != self.bbox_head.featmap_sizes_train:
            self.bbox_head.featmap_sizes_train = current_sizes
            priors = self.bbox_head.prior_generator.grid_priors(
                current_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device, with_stride=True)
            self.bbox_head.num_level_priors = [len(p) for p in priors]
            self.bbox_head.flatten_priors_train = torch.cat(priors, dim=0)
            self.bbox_head.stride_tensor = self.bbox_head.flatten_priors_train[..., [2]]

        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels, gt_bboxes = gt_info[:, :, :1], gt_info[:, :, 1:]
        pad_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        flat_cls = torch.cat([s.permute(0,2,3,1).reshape(num_imgs, -1, self.bbox_head.num_classes) for s in cls_scores], 1)
        flat_bbox = torch.cat([b.permute(0,2,3,1).reshape(num_imgs, -1, 4) for b in bbox_preds], 1)
        flat_dist = torch.cat([d.reshape(num_imgs, -1, self.bbox_head.head_module.reg_max * 4) for d in bbox_dist_preds], 1)

        flat_bbox = self.bbox_head.bbox_coder.decode(
            self.bbox_head.flatten_priors_train[..., :2], flat_bbox, self.bbox_head.stride_tensor[..., 0])

        assigned = self.bbox_head.assigner(
            flat_bbox.detach().type(gt_bboxes.dtype), flat_cls.detach().sigmoid(),
            self.bbox_head.flatten_priors_train, gt_labels, gt_bboxes, pad_flag)

        # Extract labels for geodesic loss
        max_vals, max_idx = assigned['assigned_scores'].max(dim=2)
        max_idx[max_vals <= 0] = -1
        self.tmp_labels = max_idx

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
        """Inference with geodesic OOD detection."""
        img_feats, txt_feats, hyp_embeddings, _ = self.extract_feat(batch_inputs, batch_data_samples)
        self.parent.bbox_head.num_classes = txt_feats[0].shape[0]

        results = self.predict_by_feat(img_feats, txt_feats, batch_data_samples, hyp_embeddings, rescale)
        return self.parent.add_pred_to_datasample(batch_data_samples, results)

    def predict_by_feat(self, img_feats, txt_feats, batch_data_samples, hyp_embeddings, rescale=False):
        """Prediction with geodesic scoring."""
        batch_img_metas = [d.metainfo for d in batch_data_samples]
        outs = self.bbox_head(img_feats, txt_feats)

        cfg = copy.deepcopy(self.bbox_head.test_cfg)
        cfg.multi_label = cfg.multi_label and self.bbox_head.num_classes > 1
        num_imgs = len(batch_img_metas)

        sizes = [s.shape[2:] for s in outs[0]]
        if sizes != self.bbox_head.featmap_sizes:
            self.bbox_head.mlvl_priors = self.bbox_head.prior_generator.grid_priors(
                sizes, dtype=outs[0][0].dtype, device=outs[0][0].device)
            self.bbox_head.featmap_sizes = sizes

        priors = torch.cat(self.bbox_head.mlvl_priors)
        strides = torch.cat([priors.new_full((s.numel() * self.bbox_head.num_base_priors,), st)
                            for s, st in zip(sizes, self.bbox_head.featmap_strides)])

        flat_cls = torch.cat([s.permute(0,2,3,1).reshape(num_imgs, -1, self.bbox_head.num_classes) for s in outs[0]], 1).sigmoid()
        flat_bbox = torch.cat([b.permute(0,2,3,1).reshape(num_imgs, -1, 4) for b in outs[1]], 1)
        flat_bbox = self.bbox_head.bbox_coder.decode(priors[None], flat_bbox, strides)

        flat_obj = None
        if len(outs) > 2 and outs[2] is not None:
            flat_obj = torch.cat([o.permute(0,2,3,1).reshape(num_imgs, -1) for o in outs[2]], 1).sigmoid()

        results = []
        for idx, (bboxes, scores, img_meta) in enumerate(zip(flat_bbox, flat_cls, batch_img_metas)):
            obj = flat_obj[idx] if flat_obj is not None else None
            hyp_emb = hyp_embeddings[idx]

            if obj is not None and cfg.get('score_thr', -1) > 0 and not cfg.get('yolox_style', False):
                mask = obj > cfg.score_thr
                bboxes, scores, obj, hyp_emb = bboxes[mask], scores[mask], obj[mask], hyp_emb[mask]

            if obj is not None:
                scores = scores * obj[:, None]

            if scores.shape[0] == 0:
                empty = InstanceData(bboxes=bboxes, scores=scores[:, 0], labels=scores[:, 0].int(), ood_scores=scores[:, 0])
                results.append(empty)
                continue

            if not cfg.multi_label:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep, res = filter_scores_and_topk(scores, cfg.score_thr, cfg.get('nms_pre', 100000), dict(labels=labels[:, 0]))
                labels = res['labels']
            else:
                scores, labels, keep, _ = filter_scores_and_topk(scores, cfg.score_thr, cfg.get('nms_pre', 100000))

            # Compute geodesic OOD scores
            kept_emb = hyp_emb[keep]
            geo_scores = self.compute_geodesic_scores(kept_emb)  # (N, K)
            # max geodesic score (= -min_d^2) and which prototype
            max_geo, assigned_proto = geo_scores.max(dim=-1)
            # OOD score = min_k d^2_B = -max_k(-d^2_B) = -max_geo
            ood_scores = -max_geo  # Higher = more OOD

            result = InstanceData(
                scores=scores, labels=labels, bboxes=bboxes[keep],
                ood_scores=ood_scores,
                geo_max_scores=max_geo,            # max geodesic score per detection
                geo_assigned_proto=assigned_proto,  # which prototype gave max score
                # Legacy aliases for test_hyp.py compatibility
                horo_max_scores=max_geo,
                horo_assigned_proto=assigned_proto,
            )

            if rescale:
                pad = img_meta.get('pad_param')
                if pad is not None:
                    result.bboxes -= result.bboxes.new_tensor([pad[2], pad[0], pad[2], pad[0]])
                result.bboxes /= result.bboxes.new_tensor(img_meta['scale_factor']).repeat(1, 2)

            if cfg.get('yolox_style', False):
                cfg.max_per_img = len(result)

            result = self.bbox_head._bbox_post_process(result, cfg, rescale=False, with_nms=True, img_meta=img_meta)

            result.bboxes[:, 0::2].clamp_(0, img_meta['ori_shape'][1])
            result.bboxes[:, 1::2].clamp_(0, img_meta['ori_shape'][0])
            results.append(result)

        return results

    def enable_projector_grad(self, index):
        """Enable gradients for training."""
        if index == 0:
            # T1: Train all projector params (including trainable prototypes)
            for p in self.hyp_projector.parameters():
                p.requires_grad = True
            n_trainable = sum(p.numel() for p in self.hyp_projector.parameters() if p.requires_grad)
            proto_trainable = self.hyp_projector.classifier.prototypes.requires_grad
            print(f"  [T1] Training projector convs + classifier (trainable_protos={proto_trainable})")
            print(f"  [T1] Total trainable params: {n_trainable:,}")
        else:
            # T2+: Freeze Conv, train only classifier prototypes
            for name, p in self.hyp_projector.named_parameters():
                p.requires_grad = 'classifier' in name
            print(f"  [T2+] Training only classifier params (protos trainable={self.trainable_prototypes})")
