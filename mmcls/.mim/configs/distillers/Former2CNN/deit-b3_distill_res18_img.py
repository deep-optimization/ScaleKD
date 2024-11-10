_base_ = [
    '../../resnet/resnet18_8xb32_in1k.py'
]
# model settings
find_unused_parameters = False

# distillation settings
use_logit = True
is_vit = True

# config settings
vitkd = False
fd = False
kd = True
# method details
model = dict(
    _delete_ = True,
    type='ClassificationDistiller',
    use_logit = use_logit,
    is_vit = is_vit,
    teacher_pretrained = 'https://download.openmmlab.com/mmclassification/v0/deit3/deit3-base-p16_in21k-pre_3rdparty_in1k_20221009-87983ca1.pth',
    teacher_cfg = 'configs/deit3/deit3-base-p16_64xb64_in1k.py',
    student_cfg ='configs/resnet/resnet18_8xb32_in1k.py',
    distill_cfg = [ dict(methods=[dict(type='ViTKDLoss',
                                       name='loss_vitkd',
                                       use_this = vitkd,
                                       student_dims = 384,
                                       teacher_dims = 768,
                                       alpha_vitkd=0.00003,
                                       beta_vitkd=0.000003,
                                       lambda_vitkd=0.5,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='FDLoss',
                                       name='loss_fd',
                                       use_this = fd,
                                       alpha_fd=0.00002,
                                       student_dims = 512,
                                       teacher_dims = 768,
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