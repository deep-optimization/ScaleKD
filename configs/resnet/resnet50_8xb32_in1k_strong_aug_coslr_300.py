_base_ = [
    '../_base_/models/resnet50d_strong_aug.py', '../_base_/datasets/imagenet_bs256_resnet_224.py',
    '../_base_/schedules/imagenet_bs2048_coslr_Lamb.py', '../_base_/default_runtime.py'
]

# _base_ = [
#     '../_base_/models/resnet50.py', '../_base_/datasets/imagenet_bs64_swin_224.py',
#     '../_base_/schedules/imagenet_bs1024_adamw_resnet.py', '../_base_/default_runtime.py'
# ]

model = dict(
    backbone=dict(
        out_indices=(2,3))
    )

# Checkpoint Hook
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=30))

optim_wrapper = dict(clip_grad=dict(max_norm=5.0))
