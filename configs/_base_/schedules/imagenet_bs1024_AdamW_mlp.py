# optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0007, weight_decay=0.067),
    # optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.1),
    # specific to vit pretrain
    paramwise_cfg=dict(custom_keys={
        '.cls_token': dict(decay_mult=0.0),
        '.pos_embed': dict(decay_mult=0.0)
    }),
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1.0e-06,
        by_epoch=True,
        begin=0,
        end=20,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=280,
        by_epoch=True,
        begin=20,
        end=300,
    )
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=301)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=1536)
# auto_scale_lr = dict(base_batch_size=512)
