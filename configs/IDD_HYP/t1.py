_base_ = ('../../third_party/mmyolo/configs/yolov8/'
          'yolov8_x_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'],
                      allow_failed_imports=False)

# hyper-parameters
num_classes = 1203
num_training_classes = 8  # IDD base classes (not COCO's 80)
max_epochs = 40  # 40 epochs for full convergence
close_mosaic_epochs = 2
save_epoch_intervals = 2
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4
weight_decay = 0.0125
train_batch_size_per_gpu = 32
affine_scale = 0.5
max_aspect_ratio = 120
text_model_name = 'openai/clip-vit-base-patch32'
ood_threshold = 4.0  # Adjusted for IDD

# =============================================================================
# vMF Hyperspherical Configuration (SIREN-style)
# =============================================================================
hyp_config = dict(
    # Framework identifier
    framework='vmf_spherical',
    
    # Embedding dimension (matches projector output)
    embed_dim=64,
    
    # Overall vMF loss multiplier
    vmf_loss_weight=1.5,         # λ_vmf * (CE + repulsion)
    
    # vMF classifier parameters
    kappa_init=10.0,             # Initial concentration (learnable per-class log_kappa)
    ema_alpha=0.95,              # EMA decay for prototype update
    
    # vMF loss components
    class_balance_smoothing=0.5, # sqrt-inverse-frequency weights
    
    # Background repulsion for dense anchors (our addition)
    repulsion_weight=0.5,        # Weight for hinge loss on background anchors
    repulsion_margin=0.1,        # Margin for cos_sim hinge
    hard_neg_threshold=0.5,      # cos_sim > this → hard negative
    
    # BiLipschitz projector (SNGP-style spectral-normed residual)
    bi_lipschitz=True,
    
    # MLP projection head (SIREN showed +4.37% AUROC)
    use_projection_head=True,
    
    # Prototype initialization (REQUIRED!)
    # Run: bash scripts/init_protos.sh  (outputs to datasets/prototype/init_protos_t1.pt)
    init_protos='datasets/prototype/init_protos_t1.pt',
    
    # NOTE: OOD threshold is computed via adaptive calibration at end of training
    # (see calibrate_thresholds.py). No static threshold needed here.
)

# scaling model from X to XL
deepen_factor = 1.0
widen_factor = 1.5

backbone = _base_.model.backbone
backbone.update(
    deepen_factor=deepen_factor,
    widen_factor=widen_factor
)

# model settings
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOv5DetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model=backbone,
        frozen_stages=4,  # frozen the image backbone
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=text_model_name,
            frozen_modules=['all'])),
    neck=dict(type='YOLOWorldPAFPN',
              freeze_all=True,
              deepen_factor=deepen_factor,
              widen_factor=widen_factor,
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
    bbox_head=dict(type='YOLOWorldHead',
                   head_module=dict(type='YOLOWorldHeadModule',
                                    freeze_all=True,
                                    widen_factor=widen_factor,
                                    use_bn_head=True,
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

# dataset settings
text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]

train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(scale=(
        640,
        640,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=False,
        pad_val=dict(img=114),
        scale=(
            640,
            640,
        ),
        type='LetterResize'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'pad_param',
        ),
        type='mmdet.PackDetInputs'),
]

test_pipeline = [
    *_base_.test_pipeline[:-1],
    #dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param'))
]

# IDD T1 classes - 8 base classes
classes = (
    "car",
    "motorcycle",
    "rider",
    "person",
    "autorickshaw",
    "bicycle",
    "traffic sign",
    "traffic light",
)

voc_dataset_eval = dict(
    _delete_=True,
    type='YOLOv5VOCDataset',
    data_root='./datasets',
    test_mode=True,
    ann_file='ImageSets/Main/IDD/test.txt',  # Changed to IDD
    data_prefix=dict(sub_data_root=''),
    batch_shapes_cfg=None,
    pipeline=test_pipeline)

trlder = dict(
    batch_size=32,
    dataset=dict(
        metainfo=dict(classes=classes),
        ann_file='ImageSets/Main/IDD/t1.txt',  # Changed to IDD
        batch_shapes_cfg=None,
        data_prefix=dict(
            sub_data_root=''
        ),
        data_root='./datasets',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=True,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='YOLOv5VOCDataset'),
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
        sampler=dict(shuffle=True, type='DefaultSampler')
        )

val_dataloader = dict(batch_size=16, dataset=voc_dataset_eval, num_workers=4, persistent_workers=False, pin_memory=True)
test_dataloader = val_dataloader

val_evaluator = dict(type='mmdet.LVISMetric',
                     ann_file='data/coco/lvis/lvis_v1_minival_inserted_image_name.json',
                     metric='bbox')
test_evaluator = val_evaluator

# training settings
default_hooks = dict(param_scheduler=dict(max_epochs=max_epochs),
                     checkpoint=dict(interval=save_epoch_intervals,
                                     rule='greater'))
custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=train_pipeline)
]
train_cfg = dict(max_epochs=max_epochs,
                 val_interval=10,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])
optim_wrapper = dict(optimizer=dict(
    _delete_=True,
    type='AdamW',
    lr=base_lr,
    weight_decay=weight_decay,
    batch_size_per_gpu=train_batch_size_per_gpu),
                     paramwise_cfg=dict(bias_decay_mult=0.0,
                                        norm_decay_mult=0.0,
                                        custom_keys={
                                            'backbone.text_model':
                                            dict(lr_mult=0.01),
                                            'logit_scale':
                                            dict(weight_decay=0.0)
                                        }),
                     constructor='YOLOWv5OptimizerConstructor')