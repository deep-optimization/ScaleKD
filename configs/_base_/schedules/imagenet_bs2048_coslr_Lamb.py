# optimizer
optim_wrapper = dict(
    # optimizer=dict(
    #     type='Lamb', lr=1e-3, weight_decay=0.02))
    optimizer=dict(
        type='Lamb', lr=5e-3, weight_decay=0.02))

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.25,
        by_epoch=True,
        begin=0,
        # about 2500 iterations for ImageNet-1k
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=295,
        by_epoch=True,
        begin=5,
        end=300,
    )
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=301)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=2048)



# # optimizer
# optim_wrapper = dict(
#     optimizer=dict(
#         type='Lamb', lr=5e-3, weight_decay=0.02))

# # learning policy
# param_scheduler = [
#     # warm up learning rate scheduler
#     dict(
#         type='LinearLR',
#         start_factor=0.25,
#         by_epoch=True,
#         begin=0,
#         # about 2500 iterations for ImageNet-1k
#         end=5,
#         # update by iter
#         convert_to_iter_based=True),
#     # main learning rate scheduler
#     dict(
#         type='CosineAnnealingLR',
#         T_max=295,
#         by_epoch=True,
#         begin=5,
#         end=300,
#     )
# ]

# # train, val, test setting
# train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=301)
# val_cfg = dict()
# test_cfg = dict()

# # NOTE: `auto_scale_lr` is for automatically scaling LR,
# # based on the actual training batch size.
# auto_scale_lr = dict(base_batch_size=2048)
