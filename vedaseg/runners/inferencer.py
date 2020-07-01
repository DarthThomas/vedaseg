import logging

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from .base import BaseRunner
from .registry import RUNNERS

# from volksdep.converters import torch2trt
# from volksdep.converters import save

logger = logging.getLogger()


@RUNNERS.register_module
class Inferencer(BaseRunner):
    """ Runner for inferencing.
    """

    def __init__(self,
                 model,
                 workdir,
                 gpu=True,
                 infer_dataset=None,
                 infer_tf=None,
                 loader_setting=None,
                 **kwargs):
        super().__init__(model=model,
                         workdir=workdir,
                         gpu=gpu)
        self.gpu = gpu
        self.infer_tf = infer_tf
        self.infer_dataset = infer_dataset
        self.loader_setting = loader_setting
        self.model.eval()

    def __call__(self, image=None, thres=0.625):
        if isinstance(image, list):
            logger.debug(f'Entered with batch finer mode with '
                         f'a list of {len(image)} images.')
            self.infer_dataset.__init__(imglist=image,
                                        transform=self.infer_tf)
            logger.debug('Updated dataset for inference')
            loader = DataLoader(dataset=self.infer_dataset,
                                collate_fn=self.my_collate,
                                **self.loader_setting)
            logger.debug(f'Generated temp dataloader with setting:\n'
                         f'{self.loader_setting}')
            res = []
            for images, details in loader:
                res.extend(self.infer_batch(images, details, thres=thres))
            logger.debug(f'Inference done with {len(res)} masks.')

        else:
            logger.debug('Entered with single image finer mode.')
            image, details = self.infer_tf(image=image.astype(np.float32),
                                           details=[])
            res = self.infer_img(image, details, thres=thres)
            logger.debug('Inference done with single image.')
        return res

    def infer_batch(self, images, details, thres=None):
        res = []
        with torch.no_grad():
            if self.gpu:
                images = images.cuda()
            prob = self.model(images)

        if thres is None:
            _, pred_label = torch.max(prob, dim=1)
        else:
            prob = prob.softmax(dim=1)[:, 1, :, :]
            pred_label = torch.zeros_like(prob).long()
            pred_label[prob >= thres] = 1

        for pred, detail in zip(pred_label, details):
            res.append(self.infer_tf(mask=pred.float(),
                                     details=detail,
                                     inverse=True))
        return res

    def infer_img(self, image, details, thres=None):
        with torch.no_grad():
            if self.gpu:
                image = image.cuda()
            prob = self.model(image.unsqueeze(0))
        if thres is None:
            _, pred_label = torch.max(prob, dim=1)
        else:
            prob = prob.softmax(dim=1)[:, 1, :, :]
            pred_label = torch.zeros_like(prob).long()
            pred_label[prob >= thres] = 1

        mask = self.infer_tf(mask=pred_label[0].float(),
                             details=details,
                             inverse=True)
        return mask

    @staticmethod
    def my_collate(batch):
        images = [item[0] for item in batch]
        numel = sum([x.numel() for x in images])
        storage = images[0].storage()._new_shared(numel)  # noqa
        out = images[0].new(storage)
        details = [item[1] for item in batch]
        return torch.stack(images, 0, out=out), details

    # TODO: better saving method(by using config)
    # def save_tensorrt_model(self):
    #     dummy_input = torch.ones(1, 3, 481, 481).cuda()
    #
    #     # build trt model with fp32 mode
    #     trt_model = torch2trt(self.model, dummy_input, max_batch_size=4)
    #
    #     save(trt_model, 'trt_test.engine')
