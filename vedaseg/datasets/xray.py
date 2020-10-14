import logging

import cv2
import numpy as np

from vedaseg.datasets.coco import CocoDataset
from .registry import DATASETS

logger = logging.getLogger()


@DATASETS.register_module
class XrayDataset(CocoDataset):
    def __init__(self, ann_file, img_prefix='', transform=None, root='',
                 multi_label=True, as_classification=False, abs_ann_path=True):
        super().__init__(root=root,
                         ann_file=ann_file,
                         img_prefix=img_prefix,
                         transform=transform,
                         multi_label=multi_label,
                         abs_ann_path=abs_ann_path)
        self.as_classification = as_classification
        self.anno_path = self.ann_file

    def __getitem__(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(img_info)

        img = cv2.imread(img_info['filename']).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        masks = self.generate_mask(img.shape, ann_info)
        image, masks = self.process(img, masks)
        if self.as_classification:
            raise NotImplementedError('Not finished yet')
        else:
            return image, masks.long()

    def __len__(self):
        return len(self.data_infos)
