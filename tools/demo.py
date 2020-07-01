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
                        default='/home/tianhe/Demo/vedaseg'
                                '/configs/deeplabv3plus_WCE.py')
    parser.add_argument('--checkpoint', help='model checkpoint file path',
                        default='/home/tianhe/Demo/models/epoch_150.pth')
    parser.add_argument('--img_dir', help='infer image path',
                        default='/home/tianhe/Demo/samples/752.jpg')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_fp = args.config
    checkpoint = args.checkpoint
    img_dir = args.img_dir

    knife_inspector = assemble(cfg_fp, checkpoint)

    image = get_image(img_dir, order='RGB')
    prediction = knife_inspector(image=image, thres=None)

    get_plot(image, prediction, vis_mask=True, vis_contour=True,
             inverse_color_channel=False, n_class=2, color_name='autumn')


if __name__ == '__main__':
    main()
