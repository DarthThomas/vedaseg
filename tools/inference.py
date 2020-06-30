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
                                'vedaseg/model/x_ray/WCE/epoch_50.pth')
    parser.add_argument('--img_dir', help='infer image path',
                        default='/media/yuhaoye/DATA7/datasets/x-ray/'
                                'jinnan2_round2_train_20190401/'
                                'jinnan2_round2_train_20190401/'
                                'restricted_voc/JPEGImages/')
    args = parser.parse_args()
    return args


def read_imglist(imglist_fp):
    ll = []
    with open(imglist_fp, 'r') as fd:
        for line in fd:
            ll.append(f'{line.strip()}.jpg')
    return ll


def main():
    args = parse_args()
    cfg_fp = args.config
    checkpoint = args.checkpoint
    img_dir = args.img_dir

    val_set_dir = ('/media/yuhaoye/DATA7/datasets/x-ray'
                   '/jinnan2_round2_train_20190401/'
                   'jinnan2_round2_train_20190401'
                   '/restricted_voc/ImageSets/Segmentation/val.txt')
    img_list = set(read_imglist(val_set_dir)[:50])

    segmenter = assemble(cfg_fp, checkpoint)

    images = []
    for image in os.listdir(img_dir):
        if image not in img_list:
            continue
        images.append(get_image(img_dir + image, order='BGR'))

    predictions = segmenter(image=images)

    # segmenter.save_tensorrt_model()

    for image, prediction in zip(images, predictions):
        get_plot(image, prediction,
                 vis_mask=True, vis_contour=True,
                 inverse_color_channel=True, n_class=2, color_name='rainbow')


if __name__ == '__main__':
    main()
