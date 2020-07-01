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
                                '/configs/deeplabv3plus_WCE.py')
    parser.add_argument('--checkpoint', help='model checkpoint file path',
                        default='/media/yuhaoye/DATA7/temp_for_upload/vedaseg/'
                                'vedaseg/model/x_ray/WCE_INIT/epoch_150.pth')
    parser.add_argument('--img_dir', help='infer image path',
                        default='/media/yuhaoye/DATA7/datasets/x-ray/'
                                'jinnan2_round2_train_20190401/'
                                'jinnan2_round2_train_20190401/'
                                'restricted_voc/JPEGImages/752.jpg')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_fp = args.config
    checkpoint = args.checkpoint
    img_dir = args.img_dir

    segmenter = assemble(cfg_fp, checkpoint)

    image = get_image(img_dir, order='RGB')
    prediction = segmenter(image=image, thres=None)

    get_plot(image, prediction, vis_mask=True, vis_contour=True,
             inverse_color_channel=False, n_class=2, color_name='autumn')


if __name__ == '__main__':
    main()
