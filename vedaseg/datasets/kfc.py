import logging

import cv2
import numpy as np

from .base import BaseDataset
from .registry import DATASETS

logger = logging.getLogger()


@DATASETS.register_module
class KFCDataset(BaseDataset):
    """Dataset for KFC segmentation project
    """

    def __init__(self, imglist, transform, in_order='BGR', infer=True):
        super().__init__()

        self.imglist = imglist
        logger.debug('Total of images is %d' % len(self.imglist))

        self.in_order = in_order
        self.transform = transform
        self.infer = infer

    def __getitem__(self, idx):
        img = self.imglist[idx]
        img = img.astype(np.float32)
        if self.in_order.lower() == 'bgr':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        kwargs = {'image': img, 'details': []}
        if not self.infer:
            kwargs.pop('details')
        return self.process(**kwargs)

    def __len__(self):
        return len(self.imglist)
