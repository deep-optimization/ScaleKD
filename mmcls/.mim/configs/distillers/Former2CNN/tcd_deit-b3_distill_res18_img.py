_base_ = [
    '../../resnet/resnet18_8xb32_in1k.py'
]
# model settings
find_unused_parameters = False

# distillation settings
use_logit = True
is_vit = True

# config settings
tcd = True
# method details
model = dict(
    _delete_ = True,
    type='ClassificationDistiller',
    use_logit = use_logit,
    is_vit = is_vit,
    teacher_pretrained = 'https://download.openmmlab.com/mmclassification/v0/deit3/deit3-base-p16_in21k-pre_3rdparty_in1k_20221009-87983ca1.pth',
    teacher_cfg = 'configs/deit3/deit3-base-p16_64xb64_in1k.py',
    student_cfg ='configs/resnet/resnet18_8xb32_in1k.py',
    distill_cfg = [ dict(methods=[dict(type='TCDLoss',
                                       name='loss_tcd',
                                       use_this = tcd,
                                       student_dims = 512,
                                       teacher_dims = 768,
                                       alpha_tcd = 0.5,
                                       beta_tcd = 0.0000002,
                                       lambda_tcd = 0.5,
                                       )
                                ]
                        ),
                   ]
    )