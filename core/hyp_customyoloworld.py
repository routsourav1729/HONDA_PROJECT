"""
Hyperspherical Custom YOLO World with vMF Classification.

Uses von Mises-Fisher distribution on unit hypersphere for OOD detection.
Prototypes are unit vectors updated via EMA; kappa is learnable per-class.
OOD: max_c [log Z_d(kappa_c) + kappa_c * mu_c^T * r] < threshold -> unknown.
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
from .hyperbolic.projector import vMFLoss, compute_class_weights, stable_log_vmf_normalizer


def load_hyp_ckpt(model, checkpoint_path, prev_classes, current_classes, eval=False):
    """
    Load checkpoint for vMF spherical model.

    Handles:
    - Text embeddings (frozen_embeddings, embeddings)
    - vMF prototypes (buffers, not parameters)
    - Learnable log_kappa
    - Legacy geodesic checkpoint loading (auto-converts)

    Note: In vMF framework, prototypes are buffers (EMA-updated), not parameters.
    log_kappa is the only learnable classifier parameter.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Detect framework from checkpoint
    framework = checkpoint.get('hyp_config', {}).get('framework', 'geodesic_prototypical')
    is_legacy_geodesic = framework != 'vmf_spherical'
    if is_legacy_geodesic:
        print(f"  ! Loading LEGACY geodesic checkpoint -- will convert to vMF")

    # Keys to handle separately
    exclude_keys = ['embeddings', 'frozen_embeddings',
                    'frozen_prototypes',
                    # Legacy keys
                    'frozen_directions', 'frozen_biases',
                    'prototype_direction', 'prototype_bias',
                    # vMF keys
                    'classifier.prototypes', 'classifier.log_kappa',
                    'classifier.class_counts']

    # Filter out legacy Poincare-specific keys that don't exist in new model
    legacy_skip = ['to_poincare', 'log_tau']
    partial_state_dict = {k: v for k, v in state_dict.items()
                          if not any(ex in k for ex in exclude_keys)
                          and not any(ls in k for ls in legacy_skip)}

    # Try loading -- strict=False handles missing projection_head keys if
    # loading from a geodesic checkpoint that didn't have it
    missing, unexpected = model.load_state_dict(partial_state_dict, strict=False)
    if missing:
        print(f"  Missing keys (expected for new modules): {[k for k in missing[:5]]}...")
    if unexpected:
        print(f"  Unexpected keys (legacy, ignored): {[k for k in unexpected[:5]]}...")

    if eval:
        # Evaluation: load all embeddings and prototypes
        if state_dict.get('frozen_embeddings') is not None:
            model.frozen_embeddings = nn.Parameter(state_dict['frozen_embeddings'], requires_grad=False)
        if state_dict.get('embeddings') is not None:
            model.embeddings = nn.Parameter(state_dict['embeddings'], requires_grad=False)

        # Load frozen prototypes (from previous tasks)
        if state_dict.get('frozen_prototypes') is not None:
            frozen_p = state_dict['frozen_prototypes']
            if is_legacy_geodesic:
                frozen_p = F.normalize(frozen_p.float(), dim=-1)
                print(f"  ! Legacy: L2-normalized frozen prototypes to unit sphere")
            model.register_buffer('frozen_prototypes', frozen_p.cuda())
        elif state_dict.get('frozen_directions') is not None:
            frozen_dirs = F.normalize(state_dict['frozen_directions'].float(), dim=-1)
            model.register_buffer('frozen_prototypes', frozen_dirs.cuda())
            print(f"  ! Legacy: converted frozen_directions to unit sphere prototypes")

        # Load classifier prototypes + log_kappa
        proto_key = 'hyp_projector.classifier.prototypes'
        kappa_key = 'hyp_projector.classifier.log_kappa'

        if state_dict.get(proto_key) is not None:
            saved_protos = state_dict[proto_key]
            if is_legacy_geodesic:
                saved_protos = F.normalize(saved_protos.float(), dim=-1)
                print(f"  ! Legacy: L2-normalized classifier prototypes to unit sphere")
            device = model.hyp_projector.classifier.prototypes.device
            model.hyp_projector.classifier.prototypes.copy_(saved_protos.to(device))
            proto_norms = saved_protos.norm(dim=-1)
            print(f"  + Prototypes loaded ({model.hyp_projector.classifier.num_classes} classes)")
            print(f"    Norms: mean={proto_norms.mean():.4f}")
        elif state_dict.get('hyp_projector.classifier.prototype_direction') is not None:
            dirs = F.normalize(state_dict['hyp_projector.classifier.prototype_direction'].float(), dim=-1)
            device = model.hyp_projector.classifier.prototypes.device
            model.hyp_projector.classifier.prototypes.copy_(dirs.to(device))
            print(f"  ! Legacy: converted prototype_direction to unit sphere prototypes")
        else:
            print(f"  ! WARNING: No prototypes found in checkpoint -- using random init!")

        if state_dict.get(kappa_key) is not None:
            model.hyp_projector.classifier.log_kappa.data = state_dict[kappa_key].to(
                model.hyp_projector.classifier.log_kappa.device)
            print(f"  + log_kappa loaded: kappa = {model.hyp_projector.classifier.kappa.detach()}")
        elif is_legacy_geodesic:
            print(f"  ! Legacy checkpoint: using default kappa_init (no learned kappa available)")

        if state_dict.get('hyp_projector.classifier.class_counts') is not None:
            model.hyp_projector.classifier.class_counts.copy_(
                state_dict['hyp_projector.classifier.class_counts'])

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
        protos_a = F.normalize(state_dict['frozen_directions'].float(), dim=-1)
    if protos_b is None and state_dict.get('hyp_projector.classifier.prototype_direction') is not None:
        protos_b = F.normalize(state_dict['hyp_projector.classifier.prototype_direction'].float(), dim=-1)

    # L2-normalize if legacy
    if protos_a is not None and is_legacy_geodesic:
        protos_a = F.normalize(protos_a.float(), dim=-1)
    if protos_b is not None and is_legacy_geodesic:
        protos_b = F.normalize(protos_b.float(), dim=-1)

    if protos_a is not None and protos_b is not None:
        freeze_protos = torch.cat([protos_a, protos_b], dim=0)
    else:
        freeze_protos = protos_a if protos_a is not None else protos_b

    if freeze_protos is not None:
        model.register_buffer('frozen_prototypes', freeze_protos.cuda())
        print(f"  + T2+: Frozen prototypes ({freeze_protos.shape[0]} classes)")

    return model


class HypCustomYoloWorld(nn.Module):
    """
    Hyperspherical YOLO World with vMF Classification.

    Architecture:
    - Wraps frozen YOLO-World backbone/neck/head
    - HyperbolicProjector: FPN -> unit hypersphere (L2 normalize)
    - vMFClassifier: log Z_d(kappa_c) + kappa_c * mu_c^T * r scores
    - OOD detection: max_c score < adaptive threshold -> unknown
    """

    def __init__(
        self,
        yolo_world_model,
        unknown_index,
        hyp_dim=64,
        num_classifier_classes=None,
        init_prototypes=None,
        bi_lipschitz=True,
        kappa_init=10.0,
        ema_alpha=0.95,
        use_projection_head=True,
        # vMF loss weights
        vmf_loss_weight=1.5,
        class_balance_smoothing=0.5,
        repulsion_weight=0.5,
        repulsion_margin=0.1,
        hard_neg_threshold=0.5,
        # Legacy params -- accepted but ignored for backward compat
        hyp_c=1.0,
        clip_r=2.0,
        trainable_prototypes=True,
        prototype_init_norm=0.4,
        max_proto_norm=0.5,
        ce_weight=1.0,
        beta_reg=0.1,
        lambda_sep=1.0,
        sep_margin=1.0,
    ):
        super().__init__()

        self.parent = yolo_world_model
        self.bbox_head = yolo_world_model.bbox_head
        self.unknown_index = unknown_index
        self.hyp_dim = hyp_dim
        self.bi_lipschitz = bi_lipschitz
        self.vmf_loss_weight = vmf_loss_weight

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
        self._init_projector(init_prototypes, kappa_init, ema_alpha, use_projection_head)
        print(f"  Classifier: {self._classifier_num_classes} classes, {unknown_index} total")
        print(f"  Framework: vmf_spherical (kappa_init={kappa_init}, ema_alpha={ema_alpha})")

        # vMF Loss
        self.vmf_loss_fn = vMFLoss(
            embed_dim=hyp_dim,
            num_classes=self._classifier_num_classes,
            class_balance_smoothing=class_balance_smoothing,
            repulsion_weight=repulsion_weight,
            repulsion_margin=repulsion_margin,
            hard_neg_threshold=hard_neg_threshold,
        )

    def _init_text_embedding(self):
        with torch.no_grad():
            self.texts = copy.deepcopy(self.parent.texts)
            self.text_feats = self.parent.text_feats.clone()
            self.embeddings = nn.Parameter(self.text_feats)

    def _init_projector(self, init_prototypes, kappa_init, ema_alpha, use_projection_head):
        """Initialize spherical projector with vMF classifier."""
        self.hyp_projector = HyperbolicProjector(
            in_dims=[384, 768, 768],
            out_dim=self.hyp_dim,
            num_classes=self._classifier_num_classes,
            init_prototypes=init_prototypes,
            bi_lipschitz=self.bi_lipschitz,
            kappa_init=kappa_init,
            ema_alpha=ema_alpha,
            use_projection_head=use_projection_head,
        )

    @property
    def prototypes(self):
        """All prototypes (frozen + current) on unit sphere."""
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
        """Extract FPN features and project to unit hypersphere."""
        if self.frozen_embeddings is not None and self.embeddings is not None:
            txt_feats = torch.cat([self.frozen_embeddings, self.embeddings], dim=1)
        else:
            txt_feats = self.embeddings if self.embeddings is not None else self.frozen_embeddings

        img_feats = self.parent.backbone.forward_image(batch_inputs)
        txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)

        if self.parent.with_neck:
            img_feats = self.parent.neck(img_feats, txt_feats) if self.parent.mm_neck else self.parent.neck(img_feats)

        # Project to unit hypersphere (returns normalized + raw)
        hyp_embeddings, raw_proj = self.hyp_projector(img_feats)
        return img_feats, txt_feats, hyp_embeddings, raw_proj

    def compute_vmf_scores(self, hyp_embeddings):
        """Compute vMF scores using all prototypes (frozen + current)."""
        all_protos = self.prototypes  # (K_total, D) unit vectors
        K = all_protos.shape[0]

        # Build full kappa vector (frozen protos get mean kappa)
        kappa = self.hyp_projector.classifier.kappa
        if self.frozen_prototypes is not None:
            frozen_kappa = kappa.mean().expand(self.frozen_prototypes.shape[0])
            full_kappa = torch.cat([frozen_kappa, kappa])
        else:
            full_kappa = kappa

        if hyp_embeddings.dim() == 3:
            B, N, D = hyp_embeddings.shape
            log_z = stable_log_vmf_normalizer(full_kappa, D)
            cos_sim = torch.matmul(hyp_embeddings, all_protos.t())  # (B, N, K)
            scores = log_z.unsqueeze(0).unsqueeze(0) + full_kappa.unsqueeze(0).unsqueeze(0) * cos_sim
            return scores
        else:
            N, D = hyp_embeddings.shape
            log_z = stable_log_vmf_normalizer(full_kappa, D)
            cos_sim = torch.matmul(hyp_embeddings, all_protos.t())  # (N, K)
            scores = log_z.unsqueeze(0) + full_kappa.unsqueeze(0) * cos_sim
            return scores

    # Backward compat alias
    def compute_geodesic_scores(self, hyp_embeddings):
        """Alias for compute_vmf_scores (backward compat)."""
        return self.compute_vmf_scores(hyp_embeddings)

    def vmf_loss(self, hyp_embeddings, raw_proj=None):
        """Compute vMF prototype loss (CE + repulsion)."""
        if self.tmp_labels is None:
            return hyp_embeddings.new_tensor(0.0)

        B, N, D = hyp_embeddings.shape
        flat_embs = hyp_embeddings.reshape(B * N, D)
        flat_labels = self.tmp_labels.reshape(B * N)

        # EMA update prototypes (foreground only, training only)
        fg_mask = (flat_labels >= 0) & (flat_labels < self.hyp_projector.classifier.num_classes)
        if fg_mask.sum() > 0 and self.training:
            self.hyp_projector.classifier.update_prototypes(
                flat_embs[fg_mask].detach(),
                flat_labels[fg_mask]
            )

        # Compute vMF logits (classifier prototypes only, not frozen)
        logits = self.hyp_projector.compute_scores(flat_embs)  # (B*N, K)

        # Class weights
        if fg_mask.sum() > 0:
            class_weights = compute_class_weights(
                flat_labels[fg_mask],
                self.hyp_projector.classifier.num_classes,
                smoothing=0.5,
            ).to(logits.device)
        else:
            class_weights = None

        loss, _ = self.vmf_loss_fn(
            logits, flat_labels, flat_embs,
            self.hyp_projector.classifier.prototypes,
            self.hyp_projector.classifier.kappa,
            class_weights=class_weights,
        )
        return loss

    def vmf_loss_with_breakdown(self, hyp_embeddings, raw_proj=None):
        """vMF loss with diagnostics."""
        if self.tmp_labels is None:
            return hyp_embeddings.new_tensor(0.0), {}

        B, N, D = hyp_embeddings.shape
        flat_embs = hyp_embeddings.reshape(B * N, D)
        flat_labels = self.tmp_labels.reshape(B * N)

        # EMA update
        fg_mask = (flat_labels >= 0) & (flat_labels < self.hyp_projector.classifier.num_classes)
        if fg_mask.sum() > 0 and self.training:
            self.hyp_projector.classifier.update_prototypes(
                flat_embs[fg_mask].detach(),
                flat_labels[fg_mask]
            )

        logits = self.hyp_projector.compute_scores(flat_embs)

        if fg_mask.sum() > 0:
            class_weights = compute_class_weights(
                flat_labels[fg_mask],
                self.hyp_projector.classifier.num_classes,
                smoothing=0.5,
            ).to(logits.device)
        else:
            class_weights = None

        loss, loss_dict = self.vmf_loss_fn(
            logits, flat_labels, flat_embs,
            self.hyp_projector.classifier.prototypes,
            self.hyp_projector.classifier.kappa,
            class_weights=class_weights,
        )
        return loss, loss_dict

    # Backward compat aliases for training scripts
    def geodesic_loss(self, hyp_embeddings, pre_expmap_norms=None):
        return self.vmf_loss(hyp_embeddings, pre_expmap_norms)

    def geodesic_loss_with_breakdown(self, hyp_embeddings, pre_expmap_norms=None):
        return self.vmf_loss_with_breakdown(hyp_embeddings, pre_expmap_norms)

    def forward(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        """Forward pass -- dispatches to head_loss."""
        return self.head_loss(batch_inputs, batch_data_samples)

    def head_loss(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        """Compute YOLO + vMF losses."""
        self.bbox_head.num_classes = self.text_feats[0].shape[0]
        self.bbox_head.assigner.num_classes = self.text_feats[0].shape[0]

        img_feats, txt_feats, hyp_embeddings, raw_proj = self.extract_feat(batch_inputs, batch_data_samples)
        self.tmp_labels = None
        head_losses = self.bbox_head_loss(img_feats, txt_feats, batch_data_samples)
        hyp_loss = self.vmf_loss(hyp_embeddings, raw_proj)
        return head_losses, hyp_loss

    def head_loss_with_breakdown(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        """Losses with detailed breakdown."""
        self.bbox_head.num_classes = self.text_feats[0].shape[0]
        self.bbox_head.assigner.num_classes = self.text_feats[0].shape[0]

        img_feats, txt_feats, hyp_embeddings, raw_proj = self.extract_feat(batch_inputs, batch_data_samples)
        self.tmp_labels = None
        head_losses = self.bbox_head_loss(img_feats, txt_feats, batch_data_samples)
        hyp_loss, hyp_breakdown = self.vmf_loss_with_breakdown(hyp_embeddings, raw_proj)
        return head_losses, hyp_loss, hyp_breakdown

    def bbox_head_loss(self, img_feats, txt_feats, batch_data_samples):
        """YOLO detection loss (extracts TAL assignments for vMF loss)."""
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

        # Extract labels for vMF loss
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
        """Inference with vMF OOD detection."""
        img_feats, txt_feats, hyp_embeddings, _ = self.extract_feat(batch_inputs, batch_data_samples)
        self.parent.bbox_head.num_classes = txt_feats[0].shape[0]

        results = self.predict_by_feat(img_feats, txt_feats, batch_data_samples, hyp_embeddings, rescale)
        return self.parent.add_pred_to_datasample(batch_data_samples, results)

    def predict_by_feat(self, img_feats, txt_feats, batch_data_samples, hyp_embeddings, rescale=False):
        """Prediction with vMF scoring."""
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

            # Compute vMF OOD scores
            kept_emb = hyp_emb[keep]
            vmf_scores_all = self.compute_vmf_scores(kept_emb)  # (N, K)
            max_vmf, assigned_proto = vmf_scores_all.max(dim=-1)
            # OOD score: -(max vMF log-likelihood). Higher = more OOD.
            ood_scores = -max_vmf

            result = InstanceData(
                scores=scores, labels=labels, bboxes=bboxes[keep],
                ood_scores=ood_scores,
                vmf_max_scores=max_vmf,
                vmf_assigned_proto=assigned_proto,
                # Legacy aliases for test_hyp.py backward compat
                geo_max_scores=max_vmf,
                geo_assigned_proto=assigned_proto,
                horo_max_scores=max_vmf,
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

    def enable_projector_grad(self, index, use_gpm=False):
        """Enable gradients for training.

        Args:
            index: PREV_INTRODUCED_CLS (0 for T1, >0 for T2+)
            use_gpm: If True, unfreeze all projector params (GPM protects convs externally)
        """
        if index == 0:
            # T1: Train projector convs + MLP head + log_kappa
            # Prototypes are EMA-updated (no grad needed, they are buffers)
            for p in self.hyp_projector.parameters():
                p.requires_grad = True
            n_trainable = sum(p.numel() for p in self.hyp_projector.parameters() if p.requires_grad)
            kappa_trainable = self.hyp_projector.classifier.log_kappa.requires_grad
            print(f"  [T1] Training projector convs + MLP head + log_kappa (kappa trainable={kappa_trainable})")
            print(f"  [T1] Prototypes updated via EMA (not gradient)")
            print(f"  [T1] Total trainable params: {n_trainable:,}")
        elif use_gpm:
            # T2+ with GPM: unfreeze all projector params
            # GPM gradient projection protects conv weights externally
            for p in self.hyp_projector.parameters():
                p.requires_grad = True
            n_trainable = sum(p.numel() for p in self.hyp_projector.parameters() if p.requires_grad)
            print(f"  [T2+GPM] All projector params trainable (GPM protects base subspace)")
            print(f"  [T2+GPM] Total trainable params: {n_trainable:,}")
        else:
            # T2+: Freeze Conv + MLP, train only log_kappa (novel)
            # Novel prototypes updated via EMA
            for name, p in self.hyp_projector.named_parameters():
                p.requires_grad = 'classifier' in name
            print(f"  [T2+] Training only classifier log_kappa")
            print(f"  [T2+] Novel prototypes updated via EMA")
