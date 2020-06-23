import argparse
import os
import sys

sys.path.insert(0,
                os.path.join(os.path.abspath(os.path.dirname(__file__)),
                             '../../vedaseg'))

from vedaseg.assembler import assemble
from vedaseg.utils import get_image, get_plot


def parse_args():
    parser = argparse.ArgumentParser(description='Inference with VedaSeg')
    parser.add_argument('--config', help='config file path',
                        default='/media/yuhaoye/DATA7/temp_for_upload/vedaseg'
                                '/configs/d3p_481.py')
    parser.add_argument('--checkpoint', help='model checkpoint file path',
                        default='/media/yuhaoye/DATA7/temp_for_upload/vedaseg'
                                '/vedaseg/model/epoch_50.pth')
    parser.add_argument('--img_dir', help='infer image path',
                        default='/media/yuhaoye/DATA7/datasets/kfc_data_temp/'
                                'one_batch_35/')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_fp = args.config
    checkpoint = args.checkpoint
    img_dir = args.img_dir

    segmenter = assemble(cfg_fp, checkpoint)

    images = []
    for image in os.listdir(img_dir):
        images.append(get_image(img_dir + image, order='BGR'))

    predictions = segmenter(image=images)

    # segmenter.save_tensorrt_model()

    for image, prediction in zip(images, predictions):
        get_plot(image, prediction,
                 vis_mask=True, vis_contour=True,
                 inverse_color_channel=True, n_class=2, color_name='rainbow')


if __name__ == '__main__':
    main()
