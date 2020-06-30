# work dir
root_workdir = '/DATA/home/tianhewang/work_spaces/seg/x_ray/'

# seed
seed = 0

# 1. logging
logger = dict(
    handlers=(
        dict(type='StreamHandler', level='INFO'),
        dict(type='FileHandler', level='INFO'),
    ),
)

# 2. data
net_size = 609
test_cfg = dict(
    scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    bias=[0.5, 0.25, 0.0, -0.25, -0.5, -0.75],
    flip=True,
)
img_norm_cfg = dict(mean=(123.675, 116.280, 103.530),
                    std=(58.395, 57.120, 57.375))
ignore_label = 255

dataset_type = 'XrayDataset'
dataset_root = '/DATA/home/tianhewang/work_spaces/seg/DATASETS/overfitting'
data = dict(
    train=dict(
        dataset=dict(
            type=dataset_type,
            root=dataset_root,
            imglist_name='trainaug.txt',
            target=3,
        ),
        transforms=[
            dict(type='RandomScale', min_scale=0.5, max_scale=2.0,
                 scale_step=0.25, mode='bilinear'),
            dict(type='RandomCrop', height=net_size, width=net_size,
                 image_value=img_norm_cfg['mean'], mask_value=ignore_label),
            dict(type='PadIfNeeded', height=net_size, width=net_size,
                 image_value=img_norm_cfg['mean'], mask_value=ignore_label),
            dict(type='HorizontalFlip', p=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ToTensor'),
        ],
        loader=dict(
            type='DataLoader',
            batch_size=16,
            num_workers=2,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        ),
    ),
    val=dict(
        dataset=dict(
            type=dataset_type,
            root=dataset_root,
            imglist_name='trainaug.txt',
            target=3,
        ),
        transforms=[
            dict(type='SizeScale', target_size=net_size),
            dict(type='PadIfNeeded', height=net_size, width=net_size,
                 image_value=img_norm_cfg['mean'], mask_value=ignore_label),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ToTensor'),
        ],
        loader=dict(
            type='DataLoader',
            batch_size=16,
            num_workers=4,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        ),
    ),
)

# 3. model
nclasses = 2
model = dict(
    # model/encoder
    encoder=dict(
        backbone=dict(
            type='ResNet',
            arch='resnet50',
            replace_stride_with_dilation=[False, False, True],
            multi_grid=[1, 2, 4],
        ),
        enhance=dict(
            type='ASPP',
            from_layer='c5',
            to_layer='enhance',
            in_channels=2048,
            out_channels=256,
            atrous_rates=[6, 12, 18],
            dropout=0.1,
        ),
    ),
    # model/decoder
    decoder=dict(
        type='GFPN',
        # model/decoder/blocks
        neck=[
            # model/decoder/blocks/block1
            dict(
                type='JunctionBlock',
                fusion_method='concat',
                top_down=dict(
                    from_layer='enhance',
                    upsample=dict(
                        type='Upsample',
                        scale_factor=4,
                        scale_bias=-3,
                        mode='bilinear',
                        align_corners=True,
                    ),
                ),
                lateral=dict(
                    from_layer='c2',
                    type='ConvModule',
                    in_channels=256,
                    out_channels=48,
                    kernel_size=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='Relu', inplace=True),
                ),
                post=None,
                to_layer='p5',
            ),  # 4
        ],
    ),
    # model/head
    head=dict(
        type='Head',
        in_channels=304,
        inter_channels=256,
        out_channels=nclasses,
        num_convs=2,
        upsample=dict(
            type='Upsample',
            size=(net_size, net_size),
            mode='bilinear',
            align_corners=True,
        ),
    )
)

## 3.1 resume
resume = None

# 4. criterion
criterion = dict(type='CrossEntropyLoss', ignore_index=ignore_label)

# 5. optim
optimizer = dict(type='SGD', lr=0.028, momentum=0.9, weight_decay=0.0001)

# 6. lr scheduler
max_epochs = 500
lr_scheduler = dict(type='PolyLR', max_epochs=max_epochs)

# 7. runner
runner = dict(
    type='Runner',
    max_epochs=max_epochs,
    trainval_ratio=5,
    snapshot_interval=5,
)

# 8. device
gpu_id = '6, 7'
