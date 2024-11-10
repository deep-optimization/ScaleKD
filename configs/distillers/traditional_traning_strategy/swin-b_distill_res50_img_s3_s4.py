_base_ = [
    '../../resnet/resnet50_8xb32_in1k.py'
]
# model settings
find_unused_parameters = True

# distillation settings
use_logit = True
is_vit = False

# config settings
scalekd = True
kd = True
# method details
model = dict(
    _delete_ = True,
    type='ClassificationDistiller',
    use_logit = use_logit,
    is_vit = is_vit,
    sync_bn=True,
    teacher_pretrained = 'ckpt/swin_base_patch4_window7_224_22kto1k-f967f799.pth',
    teacher_cfg = 'configs/swin_transformer/swin-base_16xb64_in1k.py',
    student_cfg ='configs/resnet/resnet50_8xb32_in1k.py',
    distill_cfg = [dict(methods=[dict(type='FreqMaskingDistillLossv2',
                                       name='loss_fmd_s3',
                                       use_this=scalekd,
                                       alpha=[0.08, 0.06],
                                       student_dims=1024,
                                       teacher_dims=512,
                                       query_hw=(14,14),
                                       pos_hw=(14,14),
                                       window_shapes=(1,1),
                                       pos_dims=512,
                                       self_query=True,
                                       softmax_scale=5.,
                                       num_heads=8,
                                       dis_freq='high'
                                       )
                                  ]
                         ),
                    dict(methods=[dict(type='FreqMaskingDistillLossv2',
                                       name='loss_fmd_s4',
                                       use_this=scalekd,
                                       alpha=[0.08, 0.06],
                                       student_dims=2048,
                                       teacher_dims=1024,
                                       query_hw=(7,7),
                                       pos_hw=(7,7),
                                       pos_dims=1024,
                                       self_query=False,
                                       softmax_scale=5.,
                                       num_heads=16,
                                       dis_freq='high'
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
