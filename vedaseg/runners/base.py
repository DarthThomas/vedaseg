import logging

import numpy as np
import torch

from vedaseg.utils.checkpoint import load_checkpoint
from .registry import RUNNERS

np.set_printoptions(precision=4)

logger = logging.getLogger()


@RUNNERS.register_module
class BaseRunner:
    """ Base class for runners

    """

    def __init__(self,
                 model,
                 workdir,
                 gpu=True,
                 **kwargs):
        self.model = model
        self.workdir = workdir
        self.gpu = gpu
        self.epoch = None
        self.start_epoch = None
        self.iter = None
        self.lr = None

    def __call__(self):
        pass

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        logger.info('Resume from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict,
                               logger)

    def resume(self,
               checkpoint,
               resume_optimizer=False,
               resume_lr=True,
               resume_epoch=True,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id)
                # noqa
            )
        else:
            checkpoint = self.load_checkpoint(checkpoint,
                                              map_location=map_location)
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optim.load_state_dict(checkpoint['optimizer'])
        if resume_epoch:
            self.epoch = checkpoint['meta']['epoch']
            self.start_epoch = self.epoch
            self.iter = checkpoint['meta']['iter']
        if resume_lr:
            self.lr = checkpoint['meta']['lr']
