import logging
import time

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
                 loader_setting=None):
        self.model = model
        self.gpu = gpu
        self.infer_tf = infer_tf
        self.infer_dataset = infer_dataset
        self.loader_setting = loader_setting

    def __call__(self, image=None):
        if isinstance(image, list):
            logger.debug(f"Entered with batch finer mode with "
                         f"a list of {len(image)} images.")
            self.infer_dataset.__init__(imglist=image,
                                        transform=self.infer_tf)
            logger.debug(f"Updated dataset for inference")
            loader = DataLoader(dataset=self.infer_dataset,
                                collate_fn=self.my_collate,
                                **self.loader_setting)
            logger.debug(f"Generated temp dataloader with setting:\n"
                         f"{self.loader_setting}")
            res = []
            for images, details in loader:
                res.extend(self.infer_batch(images, details))
            logger.debug(f"Inference done with {len(res)} masks.")

        else:
            logger.debug(f"Entered with single image finer mode.")
            image, details = self.infer_tf(image=image.astype(np.float32),
                                           details=[])
            res = self.infer_img(image, details)
            logger.debug(f"Inference done with single image.")
        return res

    def infer_batch(self, images, details):
        print(f"\n\n{' ' * 4}batch infer start")
        res = []
        a_ = time.time()
        torch.cuda.synchronize()
        with torch.no_grad():
            if self.gpu:
                a = time.time()
                images = images.cuda()
                torch.cuda.synchronize()
                print(f"{' ' * 8}image to cuda cost: {time.time() - a}")
            a = time.time()
            prob = self.model(images)
            torch.cuda.synchronize()
            infer = time.time() - a
            print(f"{' ' * 8}infer cost: {infer}")
        return prob
        a = time.time()
        _, pred_label = torch.max(prob, dim=1)
        torch.cuda.synchronize()
        print(f"{' ' * 8}take max cost: {time.time() - a}")
        a = time.time()
        for pred, detail in zip(pred_label, details):
            res.append(self.infer_tf(mask=pred.float(),
                                     details=detail,
                                     inverse=True))
        torch.cuda.synchronize()
        b = time.time()
        print(f"{' ' * 8}inverse transfrom cost: {b - a}")
        c = time.time()
        print(f"{' ' * 8}total cost: {c - a_}")
        print(f"{' ' * 4}batch infer finished\n the infer stage cost: "
              f"{infer / (c - a_) * 100 :.2f} % of the total time\n"
              f"\n")
        return res

    def infer_img(self, image, details):
        print(f"\n{' ' * 4}single infer start")
        a_ = time.time()
        torch.cuda.synchronize()
        with torch.no_grad():
            if self.gpu:
                a = time.time()
                image = image.cuda()
                torch.cuda.synchronize()
                print(f"{' ' * 8}image to cuda cost: {time.time() - a}")
            a = time.time()
            prob = self.model(image.unsqueeze(0))
            torch.cuda.synchronize()
            infer = time.time() - a
            print(f"{' ' * 8}infer cost: {infer}")
        a = time.time()
        _, pred_label = torch.max(prob, dim=1)
        torch.cuda.synchronize()
        print(f"{' ' * 8}take max cost: {time.time() - a}")
        a = time.time()
        mask = self.infer_tf(mask=pred_label[0].float(),
                             details=details,
                             inverse=True)
        torch.cuda.synchronize()
        b = time.time()
        print(f"{' ' * 8}inverse transfrom cost: {b - a}")
        c = time.time()
        print(f"{' ' * 8}total cost: {c - a_}")
        print(f"{' ' * 4}single infer finished\n the infer stage cost: "
              f"{infer / (c - a_) * 100 :.2f} % of the total time\n"
              f"\n")
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
                map_location=lambda storage, loc: storage.cuda(
                    device_id))  # noqa
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
