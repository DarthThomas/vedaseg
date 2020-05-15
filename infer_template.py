import argparse
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange

from vedaseg.assembler import assemble


def parse_args():
    parser = argparse.ArgumentParser(
        description='Use trained semantic segmenter')
    base_dir = '/media/yuhaoye/DATA7/temp_for_upload/vedaseg/'
    parser.add_argument('--config', help='train config file path',
                        default=base_dir + 'configs/d3p_481.py')
    parser.add_argument('--checkpoint', help='train config file path',
                        default=base_dir + 'vedaseg/model/epoch_50.pth')
    parser.add_argument('--img_dir', help='infer image path',
                        default='/media/yuhaoye/DATA7/datasets/kfc_data_temp/'
                                'one_batch_35/')
    args = parser.parse_args()
    return args


def get_contours(image, mask, color):
    contour_img = image.copy()
    res = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = res[1] if len(res) == 3 else res[0]
    cv2.drawContours(contour_img, contours, -1, color, 3)
    return contour_img


def get_image(img_dir, order='BGR'):
    sample = cv2.imread(img_dir)
    assert sample is not None, f"sample from {img_dir} is fucking empty"
    if order.lower() == 'RGB':
        return cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    return sample


def get_plot(img, pd, vis_mask=False, vis_contour=True, output_dir=None):
    plt.figure(figsize=(8, 6))
    plt.suptitle(f'Prediction:')
    plt.tight_layout()

    plt.subplot(121)
    plt.title('input image')
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(122)
    plt.title('image with prediction')
    if vis_mask:
        img = image_overlay(img, pd, [0, 255, 0])
    if vis_contour:
        img = get_contours(img, pd, [0, 255, 0])
    plt.imshow(img)
    plt.axis('off')
    if output_dir is None:
        plt.show()
        return plt
    else:
        plt.savefig(output_dir, transparent=True)

        plt.cla()
        plt.clf()
        plt.close('all')
        return None


def image_overlay(image, mask, color):
    heat_map = np.zeros_like(image)
    heat_map[mask > 0] = color
    fin = cv2.addWeighted(heat_map, -0.5, image, 0.8, 0)
    return fin


def main():
    args = parse_args()
    cfg_fp = args.config
    checkpoint = args.checkpoint
    img_dir = args.img_dir

    runner = assemble(cfg_fp, checkpoint)

    images = []
    for image in os.listdir(img_dir):
        images.append(get_image(img_dir + image))

    b, s = [], []

    for _ in trange(50,
                    dynamic_ncols=True,
                    desc=f'testing single/batch inference with '
                         f'{len(images)} images',
                    unit='round',
                    unit_scale=True):
        torch.cuda.synchronize()
        single_start = time.time()
        for image in images:
            prediction = runner(image=image)
        torch.cuda.synchronize()
        single_end = time.time()
        s.append(single_end - single_start)
        # print(f"single infer cost :{single_end - single_start:.5f}")

        torch.cuda.synchronize()
        batch_start = time.time()
        prediction = runner(image=images)
        torch.cuda.synchronize()
        batch_end = time.time()
        b.append(batch_end - batch_start)
        # print(f"batch infer cost :{:.5f}")

    print(f"single infer cost :{sum(s) / len(s):.5f}")
    print(f"batch infer cost :{sum(b) / len(b):.5f}")

    # for image, pred in zip(images, prediction):
    #     get_plot(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    #              pred,
    #              vis_mask=True,
    #              vis_contour=True)


if __name__ == '__main__':
    main()
