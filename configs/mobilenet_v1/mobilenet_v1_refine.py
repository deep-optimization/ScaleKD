_base_ = [
    '../_base_/models/mobilenet_v1_1x.py',
    '../_base_/datasets/imagenet_bs64_mbv1_224.py',
    '../_base_/schedules/imagenet_bs2048_coslr_Lamb.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='MobileNetV1'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, loss_weight=1.0),
        topk=(1, 5),
    ))

# optimizer settings
optim_wrapper = dict(
    optimizer=dict(
        type='Lamb', lr=2e-3, weight_decay=0.004))

# Save settings
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=30))

