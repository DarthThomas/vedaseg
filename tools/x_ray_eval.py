import argparse
import os
import sys

import cv2
import numpy as np
from pycocotools.coco import COCO
from sklearn import metrics
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from vedaseg.runners import InferenceRunner
from vedaseg.utils import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Test a segmentation model')
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('checkpoint', type=str, help='checkpoint file path')
    parser.add_argument('--distribute', default=False, action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--with_train', action='store_true')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    args = parser.parse_args()
    return args


def get_label(mask):
    mask_ = mask.astype(int)
    res = [0, 0]
    for l in range(2):
        mask_l = mask_[..., l].reshape(-1)
        if 1 in mask_l:
            res[l] = 1
    return res


def get_label_score(mask):
    res = [0, 0]
    for l in range(2):
        mask_l = mask[..., l].reshape(-1)
        max_val = max(mask_l)
        res[l] = max_val
    return res


def inverse_resize(pred, image_shape, interpo=cv2.INTER_NEAREST):
    pred = cv2.resize(pred, (image_shape[1], image_shape[0]),
                      interpolation=interpo)
    return pred


def cal_resized_shape(pred, image_shape):
    h, w = image_shape
    size_h, size_w = pred.shape[0], pred.shape[1]
    scale_factor = min(size_h / h, size_w / w)
    resized_h, resized_w = int(h * scale_factor), int(w * scale_factor)
    return resized_h, resized_w


def inverse_pad(pred, image_shape):
    h, w = image_shape
    return pred[:h, :w]


def eval_model(anno_file, runner, eval_seg=False):
    runner.getmap = True
    ana_coco = COCO(anno_file)
    score_list = []
    label_list = []
    for idx, (img_id, img_info) in enumerate(
            tqdm(ana_coco.imgs.items(),
                 dynamic_ncols=True,
                 unit='images',
                 unit_scale=True)):

        image = cv2.imread(img_info['file_name'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, c = image.shape
        dummy_mask = np.zeros((h, w))
        output = runner(image, [dummy_mask])

        if eval_seg:
            output = output.transpose((1, 2, 0))
            resized_shape = cal_resized_shape(output,
                                              image.shape[:2])
            output = inverse_pad(output,
                                 resized_shape)
            output = inverse_resize(output,
                                    image.shape[:2],
                                    interpo=cv2.INTER_LINEAR)

        ann_ids = ana_coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = ana_coco.loadAnns(ann_ids)

        gt_label = [0, 0]
        for ann in anns:
            gt_label[ann['category_id'] - 1] = 1

        label_list.append(np.array(gt_label))
        if eval_seg:
            score_list.append(np.array(get_label_score(output)))
        else:
            score_list.append(np.array(output))

    label_list = np.vstack(label_list)
    score_list = np.vstack(score_list)
    result_list = np.where(score_list >= 0.5, 1, 0)
    acc = metrics.accuracy_score(label_list, result_list)
    AP = metrics.average_precision_score(label_list, score_list, average=None)
    mAP = metrics.average_precision_score(label_list, score_list)
    recall = metrics.recall_score(label_list, result_list, average=None)
    precision = metrics.precision_score(label_list, result_list, average=None)
    roc_auc_score = metrics.roc_auc_score(label_list, result_list, average=None)
    print(anno_file)
    print(f"acc:{acc}\nAP:{AP}\nmAP:{mAP}\n"
          f"recall:{recall, sum(recall) / 2}\n"
          f"precision:{precision, sum(precision) / 2}\n"
          f"roc_auc_score:{roc_auc_score}")


def main():
    args = parse_args()

    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)
    _, fullname = os.path.split(cfg_path)

    inference_cfg = cfg['inference']
    common_cfg = cfg['common']

    anno_files = [cfg['train']['data']['val']['dataset']['ann_file'],
                  cfg['test']['data']['dataset']['ann_file']]
    if args.with_train:
        anno_files.append(cfg['train']['data']['train']['dataset']['ann_file'])

    runner = InferenceRunner(inference_cfg, common_cfg)
    runner.load_checkpoint(args.checkpoint)
    runner.getmap = True
    eval_seg = True if 'seg' in fullname else False

    for anno_file in anno_files:
        eval_model(anno_file,
                   runner,
                   eval_seg=eval_seg)


if __name__ == '__main__':
    main()
