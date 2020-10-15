import logging

import torch.nn as nn

from .registry import HEADS
from ..utils import ConvModules, build_module, build_torch_nn
from ..weight_init import init_weights

logger = logging.getLogger()


@HEADS.register_module
class Head(nn.Module):
    """Head

    Args:
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels=None,
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='Relu', inplace=True),
                 num_convs=0,
                 upsample=None,
                 dropouts=None,
                 global_pool_cfg=None):
        super().__init__()

        layers = []
        if num_convs > 0:
            layers.append(ConvModules(in_channels,
                                      inter_channels,
                                      3,
                                      padding=1,
                                      conv_cfg=conv_cfg,
                                      norm_cfg=norm_cfg,
                                      act_cfg=act_cfg,
                                      num_convs=num_convs,
                                      dropouts=dropouts))

        if global_pool_cfg:
            logger.info('Head siwtched to classification mode')
            global_pool_layer = build_torch_nn(global_pool_cfg)
            layers.append(global_pool_layer)

        layers.append(nn.Conv2d(in_channels, out_channels, 1))

        if global_pool_cfg is None and upsample:
            upsample_layer = build_module(upsample)
            layers.append(upsample_layer)

        self.block = nn.Sequential(*layers)
        logger.info('Head init weights')
        init_weights(self.modules())

    def forward(self, x):
        feat = self.block(x)
        return feat
