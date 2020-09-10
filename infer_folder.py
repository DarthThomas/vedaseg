import argparse
import os

from tqdm import tqdm

from infer_template import get_image, get_plot
from vedaseg.assembler import assemble


def parse_args():
    parser = argparse.ArgumentParser(
        description='Use trained semantic segmenter')
    parser.add_argument('--config', help='train config file path',
                        default='configs/R50_161_A.py')
    parser.add_argument('--checkpoint', help='train config file path',
                        default='vedaseg/model/R50_161_A_E130.pth')
    parser.add_argument('--input_dir', help='folder for input images',
                        default='/media/yuhaoye/DATA7/datasets/kfc_data_temp/'
                                'hw_hstacked_1_100/mask_not_ignore/JPEGImages')
    parser.add_argument('--output_dir', help='folder for output renders',
                        default='/media/yuhaoye/DATA7/datasets/kfc_data_temp/'
                                'h_hw_100i_R50_161_A_E130')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_fp = args.config
    checkpoint = args.checkpoint
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    runner = assemble(cfg_fp, checkpoint)

    for image_dir in tqdm(os.listdir(input_dir)):
        image = get_image(os.path.join(input_dir, image_dir))
        target_dir = os.path.join(output_dir, image_dir)
        if '.jpg' in target_dir:
            target_dir = target_dir.replace('.jpg', '.png')
        # prediction = runner(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        prediction = runner(image=image)
        get_plot(image, prediction, vis_contour=True, output_dir=target_dir)


if __name__ == '__main__':
    main()
