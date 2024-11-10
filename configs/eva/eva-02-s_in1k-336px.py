_base_ = [
    '../_base_/datasets/imagenet_bs16_eva02_336.py',
    '../_base_/schedules/imagenet_bs1024_adamw_eva02.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ViTEVA02',
        arch='s',
        img_size=336,
        patch_size=14,
        drop_path_rate=0.1,
        out_indices=[-7,-5,-3,-1],
        sub_ln=True,
        final_norm=False,
        output_cls_token=False,
        avg_token=True),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=384,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=.02),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
    ],)

# EMA Hook
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]

# Checkpoint Hook
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=30))