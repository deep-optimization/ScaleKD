_base_ = [
    '../_base_/models/mlp_mixer_small_patch16.py',
    '../_base_/datasets/imagenet_bs128_mlp_224.py',
    '../_base_/schedules/imagenet_bs1024_AdamW_mlp.py',
    '../_base_/default_runtime.py',
]

train_dataloader = dict(batch_size=24)

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL', evaluate_on_ema=False)]

# save setting
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=30,))


# grad clip
optim_wrapper = dict(clip_grad=dict(max_norm=1.0))

# auto lr
# 24 (bs) * 8 (gpus) * 8 (nodes) = 1536 
auto_scale_lr = dict(base_batch_size=1536)

