_base_ = [
    '../../resnet/resnet18_8xb32_in1k.py'
]
# model settings
find_unused_parameters = False

# distillation settings
use_logit = False
is_vit = False

# config settings
vitkd = False
fd = True
kd = False
# method details
model = dict(
    _delete_ = True,
    type='MultiLayerClassificationDistiller',
    use_logit = use_logit,
    is_vit = is_vit,
    teacher_pretrained = 'https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_small_224_b16x64_300e_imagenet_20210615_110219-7f9d988b.pth',
    teacher_cfg = 'configs/swin_transformer/swin-small_16xb64_in1k.py',
    student_cfg ='configs/resnet/resnet18_8xb32_in1k.py',
    distill_cfg = [dict(methods=[dict(type='FDLoss',
                                       name='loss_fd_s1',
                                       use_this=fd,
                                       alpha_fd=0.001,
                                       student_dims=64,
                                       teacher_dims=96,
                                       )
                                  ]
                         ),
                    dict(methods=[dict(type='FDLoss',
                                       name='loss_fd_s2',
                                       use_this=fd,
                                       alpha_fd=0.004,
                                       student_dims=128,
                                       teacher_dims=192,
                                       )
                                  ]
                         ),
                    dict(methods=[dict(type='FDLoss',
                                       name='loss_fd_s3',
                                       use_this=fd,
                                       alpha_fd=0.03,
                                       student_dims=256,
                                       teacher_dims=384,
                                       )
                                  ]
                         ),
                    dict(methods=[dict(type='FDLoss',
                                       name='loss_fd_s4',
                                       use_this=fd,
                                       alpha_fd=0.04,
                                       student_dims=512,
                                       teacher_dims=768,
                                       )
                                  ]
                         ),
                    dict(methods=[dict(type='KDLoss',
                                       name='loss_kd',
                                       use_this = kd,
                                       temp=1.0,
                                       alpha=0.5,
                                       )
                                ]
                        ),                

                   ]
    )