import logging
import os

import cv2
import numpy as np
from PIL import Image

from .base import BaseDataset
from .registry import DATASETS

logger = logging.getLogger()


@DATASETS.register_module
class XrayDataset(BaseDataset):
    """
    """

    def __init__(self, imglist_name, root, transform, target=None):
        super().__init__()

        imglist_fp = '%s/ImageSets/Segmentation/%s' % (root, imglist_name)
        self.imglist = self.read_imglist(imglist_fp)

        logger.debug('Total of images is %d' % len(self.imglist))
        self.root = root
        self.transform = transform
        self.target_class = target

    def __getitem__(self, idx):
        imgname = self.imglist[idx]
        img_fp = os.path.join(self.root, 'JPEGImages', imgname) + '.jpg'
        mask_fp = os.path.join(self.root,
                               'EncodeSegmentationClass',
                               imgname) + f'_{self.target_class}.png'
        img = cv2.imread(img_fp).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = np.array(Image.open(mask_fp), dtype=np.float32)
        image, mask = self.process(img, mask)
        mask = mask.long()
        return image, mask

    def __len__(self):
        return len(self.imglist)

    def read_masks(self, imgname):
        if self.target_class is None:
            masks = []
            for i in range(5):
                mask_fp = os.path.join(self.root,
                                       'EncodeSegmentationClass',
                                       imgname) + f'_{i + 1}.png'
                mask = np.array(Image.open(mask_fp), dtype=np.float32)

        elif isinstance(self.target_class, int):
            mask_fp = os.path.join(self.root,
                                   'EncodeSegmentationClass',
                                   imgname) + f'_{self.target_class}.png'
            mask = np.array(Image.open(mask_fp), dtype=np.float32)
            return mask

    @staticmethod
    def read_imglist(imglist_fp):
        ll = []
        with open(imglist_fp, 'r') as fd:
            for line in fd:
                ll.append(line.strip())
        return ll
