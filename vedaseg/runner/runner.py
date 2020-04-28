import logging

import numpy as np
import torch

from .registry import RUNNERS
from ..utils.checkpoint import load_checkpoint

np.set_printoptions(precision=4)
logger = logging.getLogger()


@RUNNERS.register_module
class Runner:
    """ Light weighted Runner

    """

    def __init__(self,
                 model,
                 gpu=True,
                 infer_tf=None,
                 head_size=None):
        self.model = model
        self.gpu = gpu
        self.infer_tf = infer_tf

    def __call__(self, image=None):
        if isinstance(image, list):
            res = []
            for img in image:
                res.append(self.infer_img(img))
        else:
            res = self.infer_img(image)
        return res

    def infer_batch(self, images):
        pass

    def infer_img(self, image):
        image, details = self.infer_tf(image=image.astype(np.float32),
                                       details=[])

        self.model.eval()
        with torch.no_grad():
            if self.gpu:
                image = image.cuda()
            prob = self.model(image.unsqueeze(0))

        mask = self.infer_tf(mask=prob[0], details=details, inverse=True)

        return mask

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        logger.info('Resume from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict,
                               logger)

    def resume(self,
               checkpoint,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            _ = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            _ = self.load_checkpoint(checkpoint, map_location=map_location)
