_base_ = [
    '../../eva/eva-02-s_in1k-336px.py'
]

# configs/eva/eva-02-s_in1k-336px.py
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
    teacher_pretrained = 'ckpt/eva-l-p14_mim-in21k-pre_3rdparty_in1k-336px_20221213-f25b7634.pth',
    teacher_cfg = 'configs/eva/eva-l-p14_8xb16_in1k-336px.py',
    student_cfg ='configs/eva/eva-02-s_in1k-336px.py',
    distill_cfg = [dict(methods=[dict(type='FreqMaskingDistillLoss',
                                       name='loss_fmd_s3',
                                       use_this=scalekd,
                                       alpha=[0.008, 0.004],
                                       student_dims=384,
                                       teacher_dims=1024,
                                       query_hw=(24,24),
                                       pos_hw=(24,24),
                                       pos_dims=1024,
                                       self_query=True,
                                       softmax_scale=[1.0, 1.0],
                                       )
                                  ]
                         ),
                    dict(methods=[dict(type='FreqMaskingDistillLoss',
                                       name='loss_fmd_s4',
                                       use_this=scalekd,
                                       alpha=[0.008, 0.004],
                                       student_dims=384,
                                       teacher_dims=1024,
                                       query_hw=(24,24),
                                       pos_hw=(24,24),
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

