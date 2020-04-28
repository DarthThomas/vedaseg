import logging

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

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
                 infer_dataset=None,
                 infer_tf=None,
                 batch_size=32,
                 num_workers=4,
                 shuffle=False,
                 drop_last=False,
                 pin_memory=True,
                 ):
        self.model = model
        self.gpu = gpu
        self.infer_tf = infer_tf
        self.infer_dataset = infer_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pin_memory = pin_memory

    def __call__(self, image=None):
        if isinstance(image, list):
            self.infer_dataset.__init__(imglist=image,
                                        transform=self.infer_tf)
            loader = DataLoader(dataset=self.infer_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=self.shuffle,
                                drop_last=self.drop_last,
                                pin_memory=self.pin_memory,
                                collate_fn=self.my_collate)
            res = []
            for images, details in loader:
                res.extend(self.infer_batch(images, details))

        else:
            image, details = self.infer_tf(image=image.astype(np.float32),
                                           details=[])
            res = self.infer_img(image, details)
        return res

    def infer_batch(self, images, details):
        res = []
        self.model.eval()
        with torch.no_grad():
            if self.gpu:
                images = images.cuda()
            prob = self.model(images)
        _, pred_label = torch.max(prob, dim=1)

        for pred, detail in zip(pred_label, details):
            res.append(self.infer_tf(mask=pred.float(),
                                     details=detail,
                                     inverse=True))

        return res

    def infer_img(self, image, details):
        self.model.eval()
        with torch.no_grad():
            if self.gpu:
                image = image.cuda()
            prob = self.model(image.unsqueeze(0))
        _, pred_label = torch.max(prob, dim=1)
        mask = self.infer_tf(mask=pred_label[0].float(),
                             details=details,
                             inverse=True)

        return mask[0]

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        logger.info('Resume from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict,
                               logger)

    def resume(self,
               checkpoint,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))  # noqa
        else:
            self.load_checkpoint(checkpoint, map_location=map_location)

    @staticmethod
    def my_collate(batch):
        images = [item[0] for item in batch]
        numel = sum([x.numel() for x in images])
        storage = images[0].storage()._new_shared(numel)  # noqa
        out = images[0].new(storage)
        details = [item[1] for item in batch]
        return torch.stack(images, 0, out=out), details
