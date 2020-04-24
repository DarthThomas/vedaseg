from .registry import RUNNERS
from ..utils import build_from_cfg


def build_runner(cfg, default_args=None):
    runner = build_from_cfg(cfg, RUNNERS, default_args)
    return runner
