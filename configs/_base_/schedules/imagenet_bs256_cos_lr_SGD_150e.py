# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.00004)
    )

# learning policy
param_scheduler = dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True)
# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=150, val_interval=151)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)
