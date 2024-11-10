_base_ = [
    '../_base_/datasets/imagenet_bs32_eva02_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_eva02.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ViTEVA02',
        arch='b',
        img_size=224,
        patch_size=14,
        drop_path_rate=0.1,
        sub_ln=True,
        final_norm=False,
        output_cls_token=False,
        avg_token=True),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    ),
    pretrained = 'ckpt/eva02-base-p14_in21k-pre_in21k-medft_3rdparty_in1k-448px_20230505-5cd4d87f.pth',
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=.02),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
    ],)

custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]