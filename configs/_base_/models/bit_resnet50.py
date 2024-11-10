# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='BiTResNet',
        depth=50,
        drop_path_rate=0.05,
        width_factor=1),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
