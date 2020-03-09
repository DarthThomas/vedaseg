import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from vedaseg.utils.checkpoint import load_checkpoint
from .registry import RUNNERS

np.set_printoptions(precision=4)
logger = logging.getLogger()


@RUNNERS.register_module
class Runner:
    """ Light weighted Runner

    """

    def __init__(self,
                 loader,
                 model,
                 gpu=True,
                 infer_tf=None,
                 infer_size=None):
        self.loader = loader
        self.model = model
        self.gpu = gpu
        self.infer_tf = infer_tf
        self.infer_size = infer_size  # TODO: read infer size from  model so that we don't need this kwarg

    def __call__(self, image=None):
        if isinstance(image, list):
            res = []
            for img in image:
                res.append(self.infer_img(img))
        else:
            res = self.infer_img(image)
        return res

    def infer_img(self, image):
        h, w, c = image.shape
        le = max(h, w)
        factor = self.infer_size / le
        factor = factor // 0.0001 * 0.0001  # make sure that new image won't be larger than self.infer_size
        new_h = int(h * factor)
        new_w = int(w * factor)
        # resize original image so that the long edge = self.infer_size
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        image, _ = self.infer_tf(image.astype(np.float32), np.zeros(image.shape[:2], dtype=np.float32))
        self.model.eval()
        with torch.no_grad():
            if self.gpu:
                image = image.cuda()
            prob = self.model(image.unsqueeze(0))

        # resize prediction to the same size of original image
        prob = F.interpolate(prob, size=(le, le), mode='bilinear', align_corners=True)
        _, pred_label = torch.max(prob, dim=1)

        return pred_label[0, :h, :w].cpu().numpy()

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
