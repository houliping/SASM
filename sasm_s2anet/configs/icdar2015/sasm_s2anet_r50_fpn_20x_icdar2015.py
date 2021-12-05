# model settings
model = dict(
    type='S2ANetDetector',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
    bbox_head=dict(
        type='SAMS2ANetHead',
        num_classes=3,
        in_channels=256,
        feat_channels=256,
        stacked_convs=2,
        with_orconv=True,
        anchor_ratios=[1.0],
        anchor_strides=[8, 16, 32, 64, 128],
        anchor_scales=[4],
        target_means=[.0, .0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
        loss_fam_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_fam_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_odm_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_odm_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    fam_cfg=dict(
        assigner=dict(
            type='SASAssigner',
            topk=9,
            iou_calculator=dict(type='BboxOverlaps2D_rotated')),
        bbox_coder=dict(type='DeltaXYWHABBoxCoder',
                        target_means=(0., 0., 0., 0., 0.),
                        target_stds=(1., 1., 1., 1., 1.),
                        clip_border=True),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    odm_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.6,
            neg_iou_thr=0.5,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='BboxOverlaps2D_rotated')),
        bbox_coder=dict(type='DeltaXYWHABBoxCoder',
                        target_means=(0., 0., 0., 0., 0.),
                        target_stds=(1., 1., 1., 1., 1.),
                        clip_border=True),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    nms_pre=2000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms_rotated', iou_thr=0.3),
    max_per_img=2000)

# dataset settings
dataset_type = 'ICDAR2015'
data_root = 'data/ICDAR2015/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RotateResize',
         img_scale=[(1333, 800)],
         keep_ratio=True,
         multiscale_mode='range',
         clamp_rbbox=False),
    # dict(type='RotatedResize', img_scale=(800, 1000), keep_ratio=True),
    dict(type='RandomRotate', rate=0.5, angles=[30, 60, 90, 120, 150], auto_bound=False),
    dict(type='RotatedRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 768),
        flip=False,
        transforms=[
            dict(type='RotatedResize',  keep_ratio=True),
            dict(type='RotatedRandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'Train/new_s2a/train_s2anet.pkl',
        img_prefix=data_root + 'Train/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'Train/new_s2a/train_s2anet.pkl',
        img_prefix=data_root + 'Train/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'Test/new_s2a/test_s2anet.pkl',
        img_prefix=data_root + 'Test/images/',
        pipeline=test_pipeline))
evaluation = dict(
    gt_dir='data/ICDAR2015/Test/labelTxt/', # change it to valset for offline validation
    imagesetfile='data/ICDAR2015/Test/test.txt')
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[180, 200, 220])
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# runtime settings
total_epochs = 240
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

'''
testing para: IoU:0.3; AREA:0.3; confidence:0.60
P: 81.4; R: 78.8; hmean: 80.1
'''