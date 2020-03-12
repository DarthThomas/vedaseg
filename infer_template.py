import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from vedaseg.assemble import assemble


def parse_args():
    parser = argparse.ArgumentParser(
        description='Use trained semantic segmenter')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='train config file path')
    parser.add_argument('img_dir', help='infer image path')
    args = parser.parse_args()
    return args


def get_contours(image, mask, color):
    contour_img = image.copy()
    _, contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_img, contours, -1, color, 3)
    return contour_img


def get_image(img_dir):
    sample = cv2.imread(img_dir)
    return cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)


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

    image = get_image(img_dir)
    runner = assemble(cfg_fp, checkpoint)

    prediction = runner(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    get_plot(image, prediction, vis_mask=True, vis_contour=True)


if __name__ == '__main__':
    main()
