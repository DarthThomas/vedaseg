import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../vedaseg'))

from vedaseg.assemble import assemble


def parse_args():
    parser = argparse.ArgumentParser(description='Save segmentation results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='test checkpoint')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_fp = args.config
    checkpoint = args.checkpoint

    runner = assemble(cfg_fp, checkpoint, test_mode=True, infer_mode=True)
    runner()


if __name__ == '__main__':
    main()
