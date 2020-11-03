import logging

import torch.nn as nn

from .registry import HEADS
from ..utils import ConvModules, build_module, build_torch_nn
from ..weight_init import init_weights

logger = logging.getLogger()


@HEADS.register_module
class XHead(nn.Module):
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
                 with_seg=True,
                 with_cls=True,
                 global_pool_cfg=None,
                 late_global_pool=False):
        super().__init__()

        self.conv_modules = None
        self.conv1x1 = None
        self.upsample = None
        self.global_pool = None
        self.with_seg = with_seg
        self.with_cls = with_cls
        self.late_global_pool = late_global_pool
        #
        # self.seg_branch = None
        # self.cls_branch = None

        if num_convs > 0:
            self.conv_modules = ConvModules(in_channels,
                                            inter_channels,
                                            3,
                                            padding=1,
                                            conv_cfg=conv_cfg,
                                            norm_cfg=norm_cfg,
                                            act_cfg=act_cfg,
                                            num_convs=num_convs,
                                            dropouts=dropouts)

        if num_convs > 0:
            self.conv1x1 = nn.Conv2d(inter_channels, out_channels, 1)
        else:
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1)

        if with_cls:
            self.global_pool = build_torch_nn(global_pool_cfg)
            # self.cls_branch = self._make_cls_branch()

        if with_seg:
            if upsample:
                self.upsample = build_module(upsample)
            # self.seg_branch = self._make_seg_branch()

        logger.info('Head init weights')
        init_weights(self.modules())

    def forward(self, x):
        res = []

        feat = self.conv_modules(x)
        feat_1x1 = None

        if self.with_seg:
            feat_1x1 = self.conv1x1(feat)
            if self.upsample is not None:
                res.append(self.upsample(feat_1x1))
            else:
                res.append(feat_1x1)

        if self.with_cls:
            if self.late_global_pool:
                if feat_1x1 is None:
                    feat_1x1 = self.conv1x1(feat)
                cls_feat = self.global_pool(feat_1x1)
            else:
                cls_feat = self.global_pool(feat)
                cls_feat = self.conv1x1(cls_feat)
            res.append(cls_feat)

        return res

    # def _make_seg_branch(self):
    #     layers = []
    #     if self.conv_modules is not None:
    #         layers.append(self.conv_modules)
    #     layers.append(self.conv1x1)
    #     if self.upsample is not None:
    #         layers.append(self.upsample)
    #     return nn.Sequential(*layers)
    #
    # def _make_cls_branch(self):
    #     layers = []
    #     if self.conv_modules is not None:
    #         layers.append(self.conv_modules)
    #     if not self.late_global_pool:
    #         layers.append(self.global_pool)
    #     layers.append(self.conv1x1)
    #     if self.late_global_pool:
    #         layers.append(self.global_pool)
    #     return nn.Sequential(*layers)
    #
    # def forward(self, x):
    #     feat = self.conv_modules(x)
    #     res = []
    #
    #     if self.with_seg:
    #         seg_feat = self.conv1x1(feat)
    #         if self.upsample is not None:
    #             seg_feat = self.upsample(seg_feat)
    #         res.append(seg_feat)
    #
    #     if self.with_cls:
    #         if self.late_global_pool:
    #             cls_feat = self.conv1x1(feat)
    #             cls_feat = self.global_pool(cls_feat)
    #         else:
    #             cls_feat = self.global_pool(feat)
    #             cls_feat = self.conv1x1(cls_feat)
    #         res.append(cls_feat)
    #
    #     return res
