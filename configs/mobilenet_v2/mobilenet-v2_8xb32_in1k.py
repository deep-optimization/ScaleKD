_base_ = [
    '../_base_/models/mobilenet_v2_1x.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256_cos_lr_SGD_150e.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        out_indices=(2,3,4,7))
    )