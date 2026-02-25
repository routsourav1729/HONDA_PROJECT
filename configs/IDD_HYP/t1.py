_base_ = ('../../third_party/mmyolo/configs/yolov8/'
          'yolov8_x_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'],
                      allow_failed_imports=False)

# hyper-parameters
num_classes = 1203
num_training_classes = 9  # IDD base classes (not COCO's 80)
max_epochs = 50  # Maximum training epochs (matched to nu-OWODB)
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
# Horospherical/Hyperbolic Configuration
# =============================================================================
hyp_config = dict(
    # Poincaré ball parameters
    curvature=1.0,           # c=1.0 means ball radius R=1/√c=1.0
    embed_dim=256,           # Hyperbolic embedding dimension
    # clip_r controls max Euclidean norm before expmap0.
    # tanh(clip_r) = max Poincaré norm. 3.0 → 0.995, giving full radial discrimination.
    clip_r=3.0,
    
    # Loss weights
    hyp_loss_weight=1.0,     # Weight for horospherical CE loss
    dispersion_weight=0.1,   # Push prototype directions apart (was 0.0)
    bias_reg_weight=0.1,     # L2 penalty on biases: prevent horosphere inflation
    compactness_weight=0.05, # Pull known embeddings into their prototype horosphere
    
    # Prototype initialization (REQUIRED!)
    # Run: python init_prototypes.py --classes "car,motorcycle,..." --output init_protos_t1.pt
    init_protos='init_protos_t1.pt',
    
    # OOD detection threshold (for inference)
    ood_threshold=0.0,       # Higher score = more OOD; adjust after calibration
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

# IDD T1 classes - 9 base classes
classes = (
    "car",
    "motorcycle",
    "rider",
    "person",
    "autorickshaw",
    "traffic sign",
    "traffic light",
    "pole",
    "bicycle"
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
        num_workers=2,
        persistent_workers=False,
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