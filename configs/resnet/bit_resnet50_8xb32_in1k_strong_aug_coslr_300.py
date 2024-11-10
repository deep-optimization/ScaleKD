_base_ = [
    '../_base_/models/bit_resnet50.py', '../_base_/datasets/imagenet_bs256_resnet_224.py',
    '../_base_/schedules/imagenet_bs2048_coslr_Lamb.py', '../_base_/default_runtime.py'
]
