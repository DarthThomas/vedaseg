import argparse
import os
import sys

from tqdm import tqdm

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
                        default='/home/yuhaoye/tmp/epoch_135.pth')
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
                   '/restricted_voc/ImageSets/Segmentation/trainaug.txt')  #
    # trainaug.txt
    img_list = read_imglist(val_set_dir)

    output_dir = '/media/yuhaoye/DATA7/datasets/x-ray/' \
                 'jinnan2_round2_train_20190401/' \
                 'jinnan2_round2_train_20190401/' \
                 'restricted_voc/train_infer_135/'

    os.makedirs(output_dir, exist_ok=True)
    segmenter = assemble(cfg_fp, checkpoint)

    images = []
    for image in tqdm(img_list, desc='Fetching images', dynamic_ncols=True):
        images.append(get_image(img_dir + image, order='BGR'))

    predictions = segmenter(image=images, thres=0.6)

    for idx, (image, prediction) in enumerate(tqdm(zip(images, predictions),
                                                   desc='Rendering images',
                                                   dynamic_ncols=True)):
        get_plot(image, prediction, vis_mask=True, vis_contour=True,
                 output_dir=(output_dir + img_list[idx]).replace('jpg', 'png'),
                 inverse_color_channel=True, n_class=2, color_name='autumn')

    # images = []
    # for image in img_list:
    #     images.append(get_image(img_dir + image, order='RGB'))
    #
    # for image in images:
    #     prediction = segmenter(image=image, thres=0.4)
    #     get_plot(image, prediction, vis_mask=True, vis_contour=True,
    #              inverse_color_channel=False, n_class=2, color_name='autumn')

    # for image in img_list:
    #     print(img_dir + image)
    #     prediction = segmenter(image=get_image(img_dir + image, order='RGB'),
    #                            thres=0.625)
    #     get_plot(image, prediction, vis_mask=True, vis_contour=True,
    #              inverse_color_channel=True, n_class=2, color_name='rainbow')


if __name__ == '__main__':
    main()
