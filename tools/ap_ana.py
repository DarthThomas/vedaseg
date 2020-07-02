import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                '../../vedaseg'))

from vedaseg.assembler import assemble


def parse_args():
    parser = argparse.ArgumentParser(description='Search for threshold')
    parser.add_argument('--config', help='config file path',
                        default='/media/yuhaoye/DATA7/temp_for_upload/vedaseg'
                                '/configs/ap_ana.py')
    parser.add_argument('--checkpoint', help='model checkpoint file path',
                        default='/home/yuhaoye/tmp/epoch_75.pth')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_fp = args.config
    checkpoint = args.checkpoint

    runner = assemble(cfg_fp, checkpoint, True)
    runner(ap_ana=True,
           conf_thresholds=np.arange(0.1, 1.0, 0.2),
           iou_thresholds=np.arange(0.1, 1.0, 0.2))
    # runner(ap_ana=np.arange(0.1, 1.0, 0.1))


if __name__ == '__main__':
    main()
