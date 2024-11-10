_base_ = [
    '../_base_/models/resnet18.py', '../_base_/datasets/imagenet_bs32_swin_224.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        out_indices=(2,3))
    )
# imagenet_bs64_swin_224