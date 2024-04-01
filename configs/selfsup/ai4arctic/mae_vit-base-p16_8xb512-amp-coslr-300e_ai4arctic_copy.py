# >>>>>>>>>>>>>>>>>>>>> Start of Changed >>>>>>>>>>>>>>>>>>>>>>>>>
_base_ = [
    '../_base_/models/mae_vit-base-p16.py',
    # '../_base_/datasets/imagenet_mae.py',
    '../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# custom dataset
dataset_type = 'mmcls.CustomDataset'
data_root='/home/m32patel/projects/def-dclausi/AI4arctic/dataset/train',
train_pipeline = [
    dict(type='LoadImageFromNetCDFFile', channels = ['nersc_sar_primary','nersc_sar_secondary', 'sar_incidenceangle']),
    dict(
        type='RandomResizedCrop',
        size=224,
        scale=(0.2, 1.0),
        backend='cv2',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]

# # dataset 8 x 512
# train_dataloader = dict(
#     batch_size=512,
#     num_workers=1,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     collate_fn=dict(type='default_collate'),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         # ann_file='meta/train.txt', # removed if you don't have the annotation file
#         data_prefix=dict(img_path='./'),
#         pipeline=train_pipeline))

train_dataloader = dict(
    # _delete_=True,
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='mmpretrain.CustomDataset',
        with_label = False,
        # data_root='../../dataset/train',
        data_root='/home/m32patel/projects/def-dclausi/AI4arctic/dataset/train',
        pipeline=train_pipeline,
        extensions=['.nc']))
# <<<<<<<<<<<<<<<<<<<<<< End of Changed <<<<<<<<<<<<<<<<<<<<<<<<<<<


# optimizer wrapper
optimizer = dict(
    type='AdamW', lr=1.5e-4 * 4096 / 256, betas=(0.9, 0.95), weight_decay=0.05)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=360,
        by_epoch=True,
        begin=40,
        end=400,
        convert_to_iter_based=True)
]

# runtime settings
# pre-train for 400 epochs
train_cfg = dict(max_epochs=400)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))


custom_imports = dict(
    imports=['mmselfsup.transforms.loading'],
    allow_failed_imports=False)

# randomness
randomness = dict(seed=0, diff_rank_seed=True)
resume = True


# python tools/analysis_tools/visualize_reconstruction.py "configs/selfsup/ai4arctic/mae_vit-base-p16_8xb512-amp-coslr-300e_ai4arctic_copy.py" --checkpoint "work_dirs/selfsup/mae_vit-base-p16_8xb512-amp-coslr-300e_ai4arctic_copy/epoch_1.pth" --img-path "/home/m32patel/projects/def-dclausi/AI4arctic/dataset/train/20211220T205630_dmi_prep.nc" --out-file "work_dirs/selfsup/20211220T205630_dmi_prep"