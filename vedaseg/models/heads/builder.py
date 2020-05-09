from .registry import HEADS
from ...utils import build_from_cfg


def build_head(cfg, default_args=None):
    head = build_from_cfg(cfg, HEADS, default_args)
    return head
