import torch.nn as nn

from ..models.encoders import build_encoder
from ..models.decoders import build_decoder
from ..models.decoders import build_brick
from ..models.heads import build_head


def build_model(cfg, default_args=None):
    encoder = build_encoder(cfg.get('encoder'))

    if cfg.get('decoder'):
        middle = build_decoder(cfg.get('decoder'))
        assert 'collect' not in cfg
    else:
        assert 'collect' in cfg
        middle = build_brick(cfg.get('collect'))

    head = build_head(cfg['head'])

    model = nn.Sequential(encoder, middle, head)

    return model
