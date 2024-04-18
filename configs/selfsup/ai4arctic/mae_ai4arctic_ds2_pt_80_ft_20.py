# >>>>>>>>>>>>>>>>>>>>> Start of Changed >>>>>>>>>>>>>>>>>>>>>>>>>
_base_ = [
    '../_base_/models/mae_vit-base-p16.py',
    # '../_base_/datasets/imagenet_mae.py',
    '../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

crop_size = 512
scale = (5000, 5000)
downsample_factor = 2
# custom dataset
dataset_type = 'mmpretrain.CustomDataset'
# data_root = '/home/m32patel/projects/def-dclausi/AI4arctic/dataset/ai4arctic_raw_train_v3/'
data_root = '/home/m32patel/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3'
pretrain_ann_file = '/home/m32patel/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3/pretrain_80.txt'
# pretrain_ann_file = '/home/m32patel/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3/test1file.txt'
train_pipeline = [
    # dict(type='LoadImageFromNetCDFFile', channels=[
    #     'nersc_sar_primary', 'nersc_sar_secondary'], mean=[-14.508254953309349, -24.701211250236728],
    #     std=[5.659745919326586, 4.746759336539111], to_float32=False, nan=255),
    dict(type='PreLoadImageFromNetCDFFile', data_root=data_root, ann_file = pretrain_ann_file, channels=[
        'nersc_sar_primary', 'nersc_sar_secondary'], mean=[-14.508254953309349, -24.701211250236728],
        std=[5.659745919326586, 4.746759336539111], to_float32=True, nan=255, downsample_factor=downsample_factor),
    dict(
        type='RandomResizedCrop',
        size=crop_size,
        scale=(0.2, 1.0),
        backend='cv2',
        interpolation='bicubic'),
    # dict(type='CenterCrop', crop_size=512),
    dict(type='RandomFlip', prob=0.5),
    dict(type='NantoNum', nan=255),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]

vis_pipeline = [
    # dict(type='LoadImageFromNetCDFFile', channels = ['nersc_sar_primary','nersc_sar_secondary', 'sar_incidenceangle']),
    dict(type='LoadImageFromNetCDFFile', channels=[
        'nersc_sar_primary', 'nersc_sar_secondary'], mean=[-14.508254953309349, -24.701211250236728],
        std=[5.659745919326586, 4.746759336539111], to_float32=True, nan=255, downsample_factor=downsample_factor),
    #    dict(
    #         type='mmpretrain.RandomCrop',
    #         crop_size=512,
    #         pad_val = 255),
    dict(type='Resize', scale=(512,512)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='NantoNum', nan=255),
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
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='mmpretrain.CustomDataset',
        with_label=False,
        ann_file = pretrain_ann_file,
        # data_root='../../dataset/train',
        data_root=data_root,
        pipeline=train_pipeline,
        extensions=['.nc']))
# <<<<<<<<<<<<<<<<<<<<<< End of Changed <<<<<<<<<<<<<<<<<<<<<<<<<<<

# model settings
model = dict(
    type='MAE',
    data_preprocessor=dict(
        # mean=[-14.508254953309349, -24.701211250236728],
        # std=[5.659745919326586, 4.746759336539111],
        mean=[0, 0],
        std=[1, 1],
        bgr_to_rgb=False),
    backbone=dict(type='MAEViT', arch='b', patch_size=16,
                  mask_ratio=0.75, in_chans=2, img_size=512),
    neck=dict(
        type='MAEPretrainDecoder',
        num_patches=1024,
        patch_size=16,
        in_chans=2,
        embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
    ),
    head=dict(
        type='MAEPretrainHead',
        norm_pix=False,
        patch_size=16,
        loss=dict(type='MAEReconstructionLossWithIgnoreIndex')),
    init_cfg=[
        dict(type='Xavier', distribution='uniform', layer='Linear'),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ])


# optimizer wrapper
optimizer = dict(
    type='AdamW', lr=1.5e-4 * 1, betas=(0.9, 0.95), weight_decay=0.05)
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
        by_epoch=False,
        begin=0,
        end=1000,
        convert_to_iter_based=False),
    dict(
        type='CosineAnnealingLR',
        T_max=39000,
        by_epoch=False,
        begin=1000,
        end=40000,
        convert_to_iter_based=False)
]

# runtime settings
# pre-train for 400 epochs
# train_cfg = dict(max_epochs=200)
# runtime settings
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=40000)

vis_backends = [dict(type='WandbVisBackend',
                     init_kwargs=dict(
                         entity='mmwhale',
                         project='MAE-pretrain',
                         name='{{fileBasenameNoExtension}}',),
                     #  name='filename',),
                     define_metric_cfg=None,
                     commit=True,
                     log_code_name=None,
                     watch_kwargs=None),
                dict(type='LocalVisBackend')]

visualizer = dict(
    vis_backends=vis_backends)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1000, max_keep_ckpts=3, by_epoch=False))

custom_imports = dict(
    imports=['mmselfsup.transforms.loading',
             'mmselfsup.models.losses.mae_loss',
             'mmselfsup.datasets.transforms.processing'],
    allow_failed_imports=False)

# randomness
randomness = dict(seed=0, diff_rank_seed=True)
resume = False