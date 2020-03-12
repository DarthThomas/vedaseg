import argparse
import os

import cv2
from tqdm import tqdm

from infer_template import get_image, get_plot
from vedaseg.assemble import assemble


def parse_args():
    parser = argparse.ArgumentParser(
        description='Use trained semantic segmenter')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='train config file path')
    parser.add_argument('input_dir', help='folder for input images')
    parser.add_argument('output_dir', help='folder for output renders')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_fp = args.config
    checkpoint = args.checkpoint
    input_dir = args.input_dir
    output_dir = args.output_sir

    runner = assemble(cfg_fp, checkpoint)

    for image_dir in tqdm(os.listdir(input_dir)):
        image = get_image(image_dir)
        target_dir = os.path.join(output_dir, image_dir)
        if '.jpg' in target_dir:
            target_dir = target_dir.replace('.jpg', '.png')
        prediction = runner(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        get_plot(image, prediction, vis_contour=True, output_dir=target_dir)


if __name__ == '__main__':
    main()
