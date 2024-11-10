_base_ = [
    '../../mlp_mixer/mlp-mixer-base-p14_64xb64_in1k.py'
]

# configs/_base_/models/mlp_mixer_base_patch14.py
# model settings
find_unused_parameters = True

# distillation settings
use_logit = True
is_vit = True

# config settings

scalekd = True
kd = True
# method details
model = dict(
    _delete_ = True,
    type='ClassificationDistiller',
    use_logit = use_logit,
    is_vit = is_vit,
    sync_bn=False,
    teacher_pretrained = 'ckpt/eva-l-p14_mim-in21k-pre_3rdparty_in1k-196px_20221213-b730c7e7.pth',
    teacher_cfg = 'configs/eva/eva-l-p14_8xb16_in1k-224px.py',
    student_cfg ='configs/mlp_mixer/mlp-mixer-base-p14_64xb64_in1k.py',
    distill_cfg = [dict(methods=[dict(type='FreqMaskingDistillLoss',
                                       name='loss_fmd_s3',
                                       use_this=scalekd,
                                       alpha=[0.016, 0.008],
                                       student_dims=768,
                                       teacher_dims=1024,
                                       query_hw=(16,16),
                                       pos_hw=(16,16),
                                       pos_dims=1024,
                                       self_query=True,
                                       softmax_scale=[1.0, 1.0],
                                       )
                                  ]
                         ),
                    dict(methods=[dict(type='FreqMaskingDistillLoss',
                                       name='loss_fmd_s4',
                                       use_this=scalekd,
                                       alpha=[0.016, 0.008],
                                       student_dims=768,
                                       teacher_dims=1024,
                                       query_hw=(16,16),
                                       pos_hw=(16,16),
                                       pos_dims=1024,
                                       self_query=False,
                                       softmax_scale=[1.0, 1.0],
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
