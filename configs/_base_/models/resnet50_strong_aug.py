# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        drop_path_rate=0.05,
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, use_sigmoid=True)),
    # train_cfg=dict(augments=[
    #     dict(type='Mixup', alpha=0.2),
    #     dict(type='CutMix', alpha=1.0)
    # ]),
)
