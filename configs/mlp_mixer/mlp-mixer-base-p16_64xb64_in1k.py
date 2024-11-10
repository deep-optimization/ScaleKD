_base_ = [
    '../_base_/models/mlp_mixer_base_patch16.py',
    '../_base_/datasets/imagenet_bs128_mlp_224.py',
    '../_base_/schedules/imagenet_bs1024_AdamW_mlp.py',
    '../_base_/default_runtime.py',
]

# configs/_base_/models/mlp_mixer_base_patch14.py

optim_wrapper = dict(clip_grad=dict(max_norm=1.0))
