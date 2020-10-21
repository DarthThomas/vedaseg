from ..utils import build_from_cfg
from .registry import METRICS
from .metrics import Compose

from volkscv.metrics import classification


def build_metrics(cfg):
    mtcs = []
    for icfg in cfg:
        if icfg['type'] in classification.__all__:
            mtc = build_from_cfg(icfg, classification, 'module')
        else:
            mtc = build_from_cfg(icfg, METRICS)
        mtcs.append(mtc)
    metrics = Compose(mtcs)

    return metrics
