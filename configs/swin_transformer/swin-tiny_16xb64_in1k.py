_base_ = [
    '../_base_/models/swin_transformer/tiny_224.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        out_indices=(0,1,2,3))
    )

train_dataloader = dict(batch_size=16)

# EMA hook
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]

# Checkpoint hook
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=30))

# schedule settings
optim_wrapper = dict(clip_grad=dict(max_norm=5.0))
