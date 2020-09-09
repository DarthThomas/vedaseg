import torch.nn as nn

from ..models.decoders import build_brick
from ..models.decoders import build_decoder
from ..models.encoders import build_encoder
from ..models.heads import build_head


def build_model(cfg, default_args=None):
    encoder = build_encoder(cfg.get('encoder'))
    # model = nn.Sequential(encoder)

    # return model

    if cfg.get('decoder'):
        middle = build_decoder(cfg.get('decoder'))
        assert 'collect' not in cfg
    else:
        assert 'collect' in cfg
        middle = build_brick(cfg.get('collect'))

    head = build_head(cfg['head'])

    model = nn.Sequential(encoder, middle, head)
    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    # print(pytorch_total_params)

    return model
