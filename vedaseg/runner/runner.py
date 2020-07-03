import logging
import os.path as osp
import sys
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

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
                 max_epochs=50,
                 workdir=None,
                 start_epoch=0,
                 trainval_ratio=1,
                 snapshot_interval=1,
                 gpu=True,
                 test_cfg=None,
                 test_mode=False):
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

    def __call__(self, search=None, ap_ana=False,
                 conf_thresholds=None, iou_thresholds=None, base_on_conf=False):
        if ap_ana:
            self.ana_pred = np.zeros(shape=(len(conf_thresholds),
                                            len(iou_thresholds),
                                            len(self.loader['val'])))
            p, r = self.judge_eopch(conf_thresholds=conf_thresholds,
                                    iou_thresholds=iou_thresholds)
            with np.printoptions(precision=4, suppress=True):
                print('precision:')
                print(p)
                print('recall:')
                print(r)
        # if ap_ana is not None:
        #     total_samples = len(self.loader['val'])
        #     total_thres = len(ap_ana)
        #     self.ana_gt = np.zeros(total_samples)
        #     self.ana_pred = np.zeros(shape=(total_thres, total_samples))
        #     for conf in np.arange(0.1, 1.0, 0.1):  # np.arange(0.1, 1.0, 0.1):
        #         self.ana_epoch(ap_ana, conf=conf)
        elif search is not None:
            for thres in tqdm(search):
                self.search_epoch(thres)
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

    def ana_epoch(self, thres, conf):
        for sample_id, (img, label) in enumerate(tqdm(self.loader['val'],
                                                      desc=f'Confidence='
                                                           f'{conf:.1f}',
                                                      dynamic_ncols=True)):
            if 1 in label:
                self.ana_gt[sample_id] = 1

            pred, gt = self.ana_batch(img, label, conf=conf)

            _, ious = self.metric.miou()
            iou = ious[-1]
            for thres_id, threshold in enumerate(thres):
                if (1 in pred) and (1 not in gt):
                    self.ana_pred[thres_id, sample_id] = 1
                elif iou > threshold:
                    self.ana_pred[thres_id, sample_id] = 1
                else:
                    self.ana_pred[thres_id, sample_id] = 0

        total_p, total_r = np.zeros_like(thres), np.zeros_like(thres)

        for thres_id, threshold in enumerate(thres):
            p, r = self.ana_ap(self.ana_gt, self.ana_pred[thres_id, :])
            tqdm.write(f"Threshold@{threshold:.2f}: "
                       f"{' ' * 4}"
                       f"Precision: {p:.4f},  Recall:{r:.4f}")
            total_p[thres_id] = p
            total_r[thres_id] = r
        tqdm.write(f'Average P:{np.mean(total_p):.3f}, '
                   f'Average R:{np.mean(total_r):.3f}')

    @staticmethod
    def ana_ap(gt, pred):
        tp, tn, fp, fn = 0, 0, 0, 0
        assert len(gt) == len(pred)
        for gt_, pred_ in zip(gt, pred):
            if gt_ == pred_ == 1:
                tp += 1
            elif gt_ == pred_ == 0:
                tn += 1
            elif gt_ == 1:
                fn += 1
            else:
                fp += 1
        precision = tp / (tp + fp + sys.float_info.min)
        recall = tp / (tp + fn + sys.float_info.min)
        return precision, recall

    def judge_eopch(self, conf_thresholds=None, iou_thresholds=None):
        c_l, i_l, s_l = len(conf_thresholds), \
                        len(iou_thresholds), \
                        len(self.loader['val'])
        precision = np.zeros(shape=(c_l, i_l))
        recall = np.zeros_like(precision)
        total_res = np.zeros(shape=(c_l, i_l, s_l))
        gt = np.zeros(len(self.loader['val']))
        for sample_id, (img, label) in enumerate(
                tqdm(self.loader['val'],
                     desc=f'Inference with different thresholds',
                     dynamic_ncols=True)
        ):
            res = self.judge_batch(img,
                                   label,
                                   conf_thresholds=conf_thresholds,
                                   iou_thresholds=iou_thresholds)
            total_res[:, :, sample_id] = res
            if 1 in label:
                gt[sample_id] = 1

        for conf_idx, conf_thres in enumerate(conf_thresholds):
            for iou_idx, iou_thres in enumerate(iou_thresholds):
                p, r = self.ana_ap(gt, total_res[conf_idx, iou_idx, :])
                precision[conf_idx, iou_idx] = p
                recall[conf_idx, iou_idx] = r
        return precision, recall

    def search_epoch(self, thres):
        for img, label in tqdm(self.loader['val'],
                               desc=f'Thres={thres}',
                               dynamic_ncols=True):
            self.metric.reset()
            self.search_batch(img, label, thres)
            miou, ious = self.metric.miou()
        logger.info('Validate, mIoU %.4f, fgIoU %.6f' % (miou, ious[-1]))

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
        self.model.eval()
        with torch.no_grad():
            if self.gpu:
                img = img.cuda()
                label = label.cuda()

            if self.test_cfg:
                scales = self.test_cfg.get('scales', [1.0])
                flip = self.test_cfg.get('flip', False)
                biases = self.test_cfg.get('bias', [0.0])
            else:
                scales = [1.0]
                flip = False
                biases = [0.0]

            assert len(scales) == len(biases)

            n, c, h, w = img.size()
            probs = []
            for scale, bias in zip(scales, biases):
                new_h, new_w = int(h * scale + bias), int(w * scale + bias)
                new_img = F.interpolate(img, size=(new_h, new_w),
                                        mode='bilinear', align_corners=True)
                prob = self.model(new_img).softmax(dim=1)
                probs.append(prob)

                if flip:
                    flip_img = new_img.flip(3)
                    flip_prob = self.model(flip_img).softmax(dim=1)
                    prob = flip_prob.flip(3)
                    probs.append(prob)
            prob = torch.stack(probs, dim=0).mean(dim=0)

            _, pred_label = torch.max(prob, dim=1)
            self.metric.add(pred_label.cpu().numpy(), label.cpu().numpy())
            miou, ious = self.metric.miou()
            logger.info('Test, mIoU %.4f, IoUs %s' % (miou, ious))

    def search_batch(self, img, label, thres=None):
        self.model.eval()
        with torch.no_grad():
            if self.gpu:
                img = img.cuda()
                label = label.cuda()

            pred = self.model(img)

            prob = pred.softmax(dim=1)
            if thres is None:
                _, pred_label = torch.max(prob, dim=1)
            else:
                prob = prob[:, 1, :, :]
                pred_label = torch.zeros_like(prob).long()
                pred_label[prob >= thres] = 1
            self.metric.add(pred_label.cpu().numpy(), label.cpu().numpy())
            # miou, ious = self.metric.miou()
            # logger.info('Validate, mIoU %.4f, fgIoU %.6f' % (miou, ious[-1]))

    def ana_batch(self, img, label, conf=None):
        self.metric.reset()
        self.model.eval()
        with torch.no_grad():
            if self.gpu:
                img = img.cuda()
                label = label.cuda()

            pred = self.model(img)
            prob = pred.softmax(dim=1)
            if conf is None:
                _, pred_label = torch.max(prob, dim=1)
            else:
                prob = prob[:, 1, :, :]
                pred_label = torch.zeros_like(prob).long()
                pred_label[prob > conf] = 1
            # if 1 not in label:
            #     p = pred_label.detach().clone()
            #     assert 1 == 0, f'{torch.sum(p), (p.size(1) * p.size(2))}'
            self.metric.add(pred_label.cpu().numpy(), label.cpu().numpy())
            return pred_label, label

    def judge_batch(self,
                    img, label,
                    conf_thresholds=None, iou_thresholds=None):

        res = np.zeros((len(conf_thresholds), len(iou_thresholds)))
        self.model.eval()
        with torch.no_grad():
            if self.gpu:
                img = img.cuda()
                label = label.cuda()

            pred = self.model(img)
            prob = pred.softmax(dim=1)
        for idx_c, conf_th in enumerate(conf_thresholds):
            for idx_i, iou_th in enumerate(iou_thresholds):
                res[idx_c, idx_i] = self.judge_conf_map(prob, label,
                                                        conf_thres=conf_th,
                                                        iou_thres=iou_th)
        return res

    def judge_conf_map(self, conf_map, label, conf_thres=None, iou_thres=0.5):
        self.metric.reset()

        if conf_thres is None:
            _, pred_label = torch.max(conf_map, dim=1)
        else:
            prob = conf_map[:, 1, :, :]
            pred_label = torch.zeros_like(prob).long()
            pred_label[prob > conf_thres] = 1

        if (1 not in label) and (1 in pred_label):
            return 1

        self.metric.add(pred_label.cpu().numpy(), label.cpu().numpy())
        _, ious = self.metric.miou()

        fg_iou = ious[-1]

        if fg_iou > iou_thres:
            return 1
        return 0

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
            checkpoint = self.load_checkpoint(checkpoint,
                                              map_location=map_location)
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optim.load_state_dict(checkpoint['optimizer'])
        if resume_epoch:
            self.epoch = checkpoint['meta']['epoch']
            self.start_epoch = self.epoch
            self.iter = checkpoint['meta']['iter']
        if resume_lr:
            self.lr = checkpoint['meta']['lr']
