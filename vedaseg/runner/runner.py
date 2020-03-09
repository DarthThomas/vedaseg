import torch
import logging
import os.path as osp
import torch.nn.functional as F
import numpy as np
from collections.abc import Iterable

import cv2
import matplotlib.pyplot as plt

from vedaseg.utils.checkpoint import load_checkpoint, save_checkpoint

from .registry import RUNNERS

np.set_printoptions(precision=4)

logger = logging.getLogger()


@RUNNERS.register_module
class Runner(object):
    """ Runner

    """

    def __init__(self,
                 loader,
                 model,
                 criterion,
                 metric,
                 optim,
                 lr_scheduler,
                 max_epochs,
                 workdir,
                 start_epoch=0,
                 trainval_ratio=1,
                 snapshot_interval=1,
                 gpu=True,
                 test_cfg=None,
                 test_mode=False,
                 infer_mode=False,
                 infer_tf=None,
                 infer_size=None):
        self.loader = loader
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.start_epoch = start_epoch
        self.max_epochs = max_epochs
        self.workdir = workdir
        self.trainval_ratio = trainval_ratio
        self.snapshot_interval = snapshot_interval
        self.gpu = gpu
        self.test_cfg = test_cfg
        self.test_mode = test_mode
        self.infer_mode = infer_mode
        self.infer_tf = infer_tf
        self.infer_size = infer_size  # TODO: read infer size from  model so that we don't need this kwarg

    def __call__(self, image=None):
        if self.infer_mode:
            if isinstance(image, list):
                res = []
                for img in image:
                    res.append(self.infer_img(img))
            else:
                res = self.infer_img(image)
            return res
        elif self.test_mode:
            self.test_epoch()
        else:
            assert self.trainval_ratio > 0
            for epoch in range(self.start_epoch, self.max_epochs):
                self.train_epoch()
                self.save_checkpoint(self.workdir)
                if self.trainval_ratio > 0 \
                        and (epoch + 1) % self.trainval_ratio == 0 \
                        and self.loader.get('val'):
                    self.validate_epoch()

    def infer_img(self, image):
        image = image
        print(image.shape)
        h, w, c = image.shape
        le = max(h, w)
        factor = self.infer_size / le
        factor = factor // 0.0001 * 0.0001  # make sure that new image won't be larger than self.infer_size
        new_h = int(h * factor)
        new_w = int(w * factor)
        # resize image to 'longer edge=infer_size'
        plt.imshow(image)
        plt.show()

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        plt.imshow(image)
        plt.show()
        image, _ = self.infer_tf(image.astype(np.float32), np.zeros(image.shape[:2], dtype=np.float32))
        self.model.eval()
        with torch.no_grad():
            if self.gpu:
                image = image.cuda()
            prob = self.model(image.unsqueeze(0))
        # prob = self.test_time_aug(image.unsqueeze(0))
        prob = F.interpolate(prob, size=(le, le), mode='bilinear', align_corners=True)
        print(prob.size())
        plt.imshow(prob[0, 1, :, :].cpu().numpy())
        plt.show()
        _, pred_label = torch.max(prob, dim=1)
        print(pred_label.size())

        pred_label = pred_label[0, :h, :w].cpu().numpy()
        # pred_label = np.squeeze(pred_label, axis=0)
        print(pred_label.min())
        print(pred_label.max())
        res = pred_label[:h, :w]
        print(res.shape)
        print(res.max())
        print(res.min())
        return res

    def train_epoch(self):
        logger.info('Epoch %d, Start training' % self.epoch)
        iter_based = hasattr(self.lr_scheduler, '_iter_based')
        self.metric.reset()
        for img, label in self.loader['train']:
            self.train_batch(img, label)
            if iter_based:
                self.lr_scheduler.step()
        if not iter_based:
            self.lr_scheduler.step()

    def validate_epoch(self):
        logger.info('Epoch %d, Start validating' % self.epoch)
        self.metric.reset()
        for img, label in self.loader['val']:
            self.validate_batch(img, label)

    def test_epoch(self):
        logger.info('Start testing')
        logger.info('test info: %s' % self.test_cfg)
        self.metric.reset()
        for img, label in self.loader['val']:
            self.test_batch(img, label)

    def train_batch(self, img, label):
        self.model.train()

        self.optim.zero_grad()

        if self.gpu:
            img = img.cuda()
            label = label.cuda()
        pred = self.model(img)
        loss = self.criterion(pred, label)

        loss.backward()
        self.optim.step()

        with torch.no_grad():

            '''
            import matplotlib.pyplot as plt
            pred = (prob[0]).permute(1, 2, 0).float().cpu().numpy()[:, :, 0]
            im = img[0].permute(1, 2, 0).clamp(min=0, max=1).cpu().numpy()
            label_ = label[0].permute(1, 2, 0).clamp(min=0, max=1).cpu().numpy()[:, :, 0]
            import random
            random_num = random.randint(0, 1000)
            pred_name = 'output/%d_pred.jpg' % random_num
            plt.imsave(pred_name, pred, cmap='Greys')
            im_name = 'output/%d.jpg' % random_num
            plt.imsave(im_name, im, cmap='Greys')
            label_name = 'output/%d_gt.jpg' % random_num
            plt.imsave(label_name, label_, cmap='Greys')
            '''
            _, pred_label = torch.max(pred, dim=1)
            self.metric.add(pred_label.cpu().numpy(), label.cpu().numpy())
            miou, ious = self.metric.miou()
        if self.iter != 0 and self.iter % 10 == 0:
            logger.info(
                'Train, Epoch %d, Iter %d, LR %s, Loss %.4f, mIoU %.4f, IoUs %s' %
                (self.epoch, self.iter, self.lr, loss.item(),
                 miou, ious))

    def validate_batch(self, img, label):
        self.model.eval()
        with torch.no_grad():
            if self.gpu:
                img = img.cuda()
                label = label.cuda()

            pred = self.model(img)

            prob = pred.softmax(dim=1)
            _, pred_label = torch.max(prob, dim=1)
            self.metric.add(pred_label.cpu().numpy(), label.cpu().numpy())
            miou, ious = self.metric.miou()
            logger.info('Validate, mIoU %.4f, IoUs %s' % (miou, ious))

    def test_batch(self, img, label):
        prob = self.test_time_aug(img)
        _, pred_label = torch.max(prob, dim=1)
        self.metric.add(pred_label.cpu().numpy(), label.cpu().numpy())
        miou, ious = self.metric.miou()
        logger.info('Test, mIoU %.4f, IoUs %s' % (miou, ious))

    def test_time_aug(self, img):

        scales, flip, biases = [1.0], False, [0.0]
        if self.test_cfg:
            scales = self.test_cfg.get('scales', [1.0])
            flip = self.test_cfg.get('flip', False)
            biases = self.test_cfg.get('bias', [0.0])
        assert len(scales) == len(biases)

        self.model.eval()
        with torch.no_grad():
            if self.gpu:
                img = img.cuda()

            n, c, h, w = img.size()

            probs = []
            for scale, bias in zip(scales, biases):
                new_h, new_w = int(h * scale + bias), int(w * scale + bias)
                new_img = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=True)
                prob = self.model(new_img).softmax(dim=1)
                probs.append(prob)

                if flip:
                    flip_img = new_img.flip(3)
                    flip_prob = self.model(flip_img).softmax(dim=1)
                    prob = flip_prob.flip(3)
                    probs.append(prob)
            prob = torch.stack(probs, dim=0).mean(dim=0)
        return prob

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None):
        if self.epoch % self.snapshot_interval == 0 or self.epoch == self.max_epochs:
            if meta is None:
                meta = dict(epoch=self.epoch, iter=self.iter, lr=self.lr)
            else:
                meta.update(epoch=self.epoch, iter=self.iter, lr=self.lr)

            filename = filename_tmpl.format(self.epoch)
            filepath = osp.join(out_dir, filename)
            linkpath = osp.join(out_dir, 'latest.pth')
            optimizer = self.optim if save_optimizer else None
            logger.info('Save checkpoint %s', filename)
            save_checkpoint(self.model,
                            filepath,
                            optimizer=optimizer,
                            meta=meta)

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        logger.info('Resume from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict,
                               logger)

    @property
    def epoch(self):
        """int: Current epoch."""
        return self.lr_scheduler.last_epoch

    @epoch.setter
    def epoch(self, val):
        """int: Current epoch."""
        self.lr_scheduler.last_epoch = val

    @property
    def lr(self):
        lr = [x['lr'] for x in self.optim.param_groups]
        return np.array(lr)

    @lr.setter
    def lr(self, val):
        for idx, param in enumerate(self.optim.param_groups):
            if isinstance(val, Iterable):
                param['lr'] = val[idx]
            else:
                param['lr'] = val

    @property
    def iter(self):
        """int: Current iteration."""
        return self.lr_scheduler.last_iter

    @iter.setter
    def iter(self, val):
        """int: Current epoch."""
        self.lr_scheduler.last_iter = val

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
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(checkpoint, map_location=map_location)
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optim.load_state_dict(checkpoint['optimizer'])
        if resume_epoch:
            self.epoch = checkpoint['meta']['epoch']
            self.start_epoch = self.epoch
            self.iter = checkpoint['meta']['iter']
        if resume_lr:
            self.lr = checkpoint['meta']['lr']
