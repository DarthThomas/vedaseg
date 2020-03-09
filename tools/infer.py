import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../vedaseg'))

from vedaseg.assemble import assemble


def parse_args():
    parser = argparse.ArgumentParser(
        description='Use trained semantic segmenter')
    parser.add_argument('config', help='train config file path', default='configs/horc_1_50_d3p_481.py')
    parser.add_argument('checkpoint', help='train config file path', default='checkpoints/epoch_50.pth')
    parser.add_argument('img_dir', help='infer image path',
                        default='/media/yuhaoye/DATA7/git/Seg_Vis/mask_ignore/JPEGImages/20191206_110208_2_3.jpg')
    args = parser.parse_args()
    return args


def get_contours(image, mask):
    mask_img = mask * 255
    _, thres = cv2.threshold(mask_img.astype(np.uint8), 127, 255, 0)
    contour_img = image.copy()
    _, contours, _ = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)
    return contour_img


def get_image(img_dir):
    sample = cv2.imread(img_dir)
    return cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)


def get_plot(img, pd, vis_mask=True, vis_contour=True):
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
        img = get_contours(img, pd)
    plt.imshow(img)
    plt.axis('off')

    plt.show()
    return plt


def image_overlay(image, mask, color):
    # pred = mask * 255
    heat_map = np.zeros_like(image)
    heat_map[mask] = color
    # heat_map = cv2.applyColorMap(pred.astype(np.uint8), cv2.COLORMAP_JET)
    img_pd = image.copy()
    fin = cv2.addWeighted(heat_map, -0.5, img_pd, 0.8, 0)
    return fin


def main():
    args = parse_args()
    cfg_fp = args.config
    checkpoint = args.checkpoint
    img_dir = args.img_dir

    image = get_image(img_dir)
    runner = assemble(cfg_fp, checkpoint, infer_mode=True)

    prediction = runner(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    get_plot(image, prediction, vis_mask=True, vis_contour=True)


if __name__ == '__main__':
    main()
