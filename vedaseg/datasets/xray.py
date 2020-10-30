import logging

import cv2
import torch
import numpy as np
from pycocotools.coco import COCO

from vedaseg.datasets.coco import CocoDataset
from .registry import DATASETS

logger = logging.getLogger()


@DATASETS.register_module
class XrayDataset(CocoDataset):
    def __init__(self, ann_file, img_prefix='', transform=None, root='',
                 multi_label=True, as_classification=False,
                 abs_ann_path=True, seg_list=None):
        super().__init__(root=root,
                         ann_file=ann_file,
                         img_prefix=img_prefix,
                         transform=transform,
                         multi_label=multi_label,
                         abs_ann_path=abs_ann_path)
        self.as_classification = as_classification
        self.seg_list = None
        if seg_list is not None:
            seg_coco = COCO(seg_list)
            self.seg_list = set(seg_coco.imgs.keys())
        self.anno_path = self.ann_file

    def __getitem__(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(img_info)

        img = cv2.imread(img_info['filename']).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        masks = self.generate_mask(img.shape, ann_info)
        image, masks = self.process(img, masks)

        if self.seg_list is not None:
            clas_gt = torch.zeros((masks.shape[0], 1, 1))
            for clas_ind in range(masks.shape[0]):
                current = masks[clas_ind, ...].reshape(-1)
                if 1 in current:
                    clas_gt[clas_ind, 0, 0] = 1
            if img_info['id'] in self.seg_list:
                return image, masks.long(), clas_gt.long()
            else:
                dummy_mask = 255 * torch.ones_like(masks)
                return image, dummy_mask.long(), clas_gt.long()

        if self.as_classification:
            clas_gt = torch.zeros((masks.shape[0], 1, 1))
            for clas_ind in range(masks.shape[0]):
                current = masks[clas_ind, ...].reshape(-1)
                if 1 in current:
                    clas_gt[clas_ind, 0, 0] = 1
            return image, clas_gt.long()
        else:
            return image, masks.long()

    def __len__(self):
        return len(self.data_infos)
