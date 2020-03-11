from ....utils import build_from_cfg

from .registry import ENHANCE_MODULES


def build_enhance_module(cfg, default_args=None):
    #import pdb
    #pdb.set_trace()
    enhance_module = build_from_cfg(cfg, ENHANCE_MODULES, default_args)
    return enhance_module
