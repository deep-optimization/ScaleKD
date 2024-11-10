_base_ = [
    '../_base_/models/resnet18_cifar.py', '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/cifar10_bs128.py', '../_base_/default_runtime.py'
]

# _base_ = [
#     '../_base_/models/resnet18.py', '../_base_/datasets/imagenet_bs32.py',
#     '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
# ]

# model = dict(
#     backbone=dict(
#         out_indices=(2,3))
#     )
