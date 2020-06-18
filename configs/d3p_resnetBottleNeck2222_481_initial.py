# seed
seed = 0

# 1. logging
logger = dict(
    handlers=(
        dict(type='StreamHandler', level='INFO'),
        # dict(type='FileHandler', level='INFO'),
    ),
)

# 2. data
net_size = 481

img_norm_cfg = dict(mean=(123.675, 116.280, 103.530),
                    std=(58.395, 57.120, 57.375))
ignore_label = 255

data = dict(
    infer=dict(
        dataset=dict(
            type='KFCDataset', imglist=[], in_order='BGR', infer=True,
        ),
        transforms=[
            dict(type='SizeScale', target_size=net_size),
            dict(type='PadIfNeeded', height=net_size, width=net_size,
                 image_value=img_norm_cfg['mean'], mask_value=ignore_label),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ToTensor'),
        ],
        loader_setting=dict(
            batch_size=16,
            num_workers=2,
            shuffle=False,
            drop_last=False,
            pin_memory=False
        ),
    )
)

# 3. model
nclasses = 2
model = dict(
    # model/encoder
    encoder=dict(
        backbone=dict(
            type='ResNet',
            arch='resnet_bottleneck2222',
            pretrain=False,
            replace_stride_with_dilation=[False, False, True],
            multi_grid=[1, 4],
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

# 4. runner
runner = dict(
    type='Runner'
)

# 5. device
gpu_id = '0'
