import logging
from functools import partial

import torch.nn as nn

from .registry import ENHANCE_MODULES
from ...utils.act import build_act_layer
from ...utils.norm import build_norm_layer
from ...weight_init import init_weights

logger = logging.getLogger()


@ENHANCE_MODULES.register_module
class ReduceChannel(nn.Module):
    def __init__(self, in_channels, out_channels, from_layer,
                 to_layer, dropout=None, norm_cfg=None, act_cfg=None):
        super(ReduceChannel, self).__init__()
        self.from_layer = from_layer
        self.to_layer = to_layer

        if norm_cfg is None:
            norm_cfg = dict(type='BN')
        norm_layer = partial(build_norm_layer, norm_cfg, layer_only=True)

        if act_cfg is None:
            act_cfg = dict(type='Relu', inplace=True)
        act_layer = partial(build_act_layer, act_cfg, layer_only=True)

        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels), act_layer(out_channels))

        self.with_dropout = dropout is not None
        if self.with_dropout:
            self.dropout = nn.Dropout(dropout)

        logger.info('ReduceChannel init weights')

        init_weights(self.modules())

    def forward(self, feats):
        feats_ = feats.copy()
        x = feats_[self.from_layer]
        res = self.project(x)

        if self.with_dropout:
            res = self.dropout(res)
        feats_[self.to_layer] = res

        return feats_
