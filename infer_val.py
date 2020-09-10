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
                        default='/DATA/home/tianhewang/work_spaces/seg/speed_up/speed_R50_161_ASPP_E150/epoch_130.pth')
    parser.add_argument('--input_dir', help='folder for input images',
                        default='/DATA/home/tianhewang/data/kfc_data/processed/all_class/ac_2_fc_rw/mask_not_ignore')
    parser.add_argument('--output_dir', help='folder for output renders',
                        default='/DATA/home/tianhewang/data/kfc_data/processed/all_class/ac_2_fc_rw/mask_not_ignore/'
                                'h_hw_100i_R50_161_A_E130')
    args = parser.parse_args()
    return args

def read_image_set(set_dir):
    res = set()
    with open(set_dir) as image_list:
        for line in image_list:
            res.add(line.strip())
    return res

def main():
    args = parse_args()
    cfg_fp = args.config
    checkpoint = args.checkpoint
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    image_set = read_image_set(os.path.join(input_dir,'ImageSets/Segmentation/val.txt'))
    val_dir = os.path.join(input_dir, 'JPEGImages')
    runner = assemble(cfg_fp, checkpoint)
    
    print(f"There is {len(os.listdir(val_dir))} images in total set, while {len(image_set)} in val set")

    for image_dir in tqdm(image_set):
        image_dir += '.jpg'
        image = get_image(os.path.join(val_dir, image_dir))
        target_dir = os.path.join(output_dir, image_dir)
        if '.jpg' in target_dir:
            target_dir = target_dir.replace('.jpg', '.png')
        prediction = runner(image=image)
        get_plot(image, prediction, vis_mask=True, vis_contour=True, output_dir=target_dir)


if __name__ == '__main__':
    main()
