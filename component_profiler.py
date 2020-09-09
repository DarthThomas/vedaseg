import argparse
import os
import time
from collections import defaultdict

import numpy as np
import prettytable as pt
import torch
from tqdm import trange

from vedaseg import utils
from vedaseg.models.builder import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Use trained semantic segmenter')
    base_dir = '/media/yuhaoye/DATA7/temp_for_upload/vedaseg/configs/'
    parser.add_argument('--config', help='train config file path',
                        default=base_dir + 'd3p_resnet18BoNe_321_NoAspp.py')
    parser.add_argument('--batch_size', help='infer batch size', default=16)
    parser.add_argument('--trails', help='infer trail number', default=100)
    parser.add_argument('--skips', help='skip infer trail number', default=5)
    args = parser.parse_args()
    return args


def build_parts(cfg):
    model = build_model(cfg=cfg['model']).cuda().eval()
    backbone = model._modules['0']._modules['0']  # noqa
    enhance = model._modules['0']._modules['1']  # noqa
    neck = model._modules['1']  # noqa
    head = model._modules['2']  # noqa
    parts = dict(Backbone=backbone, Enhance=enhance, Neck=neck, Head=head)
    return parts


def profile_moel(args, input_size, parts, cost_record):
    for _ in trange(args.trails + args.skips,
                    dynamic_ncols=True,
                    desc=f'model profiling by parts',
                    unit='round',
                    unit_scale=True):
        fake_in = torch.rand(input_size).cuda()
        for name, part in parts.items():
            time_start = time.time()
            with torch.no_grad():
                torch.cuda.synchronize()
                bb_out = part(fake_in)
            torch.cuda.synchronize()
            if _ >= args.skips:
                cost_record[name].append(time.time() - time_start)
            fake_in = bb_out


def get_summery(args, cost_record):
    tb = pt.PrettyTable()
    tb.field_names = ['Part', 'Latency: mean +- std(s)']
    total = np.zeros(args.trails)
    for name, cost in cost_record.items():
        cost = np.array(cost)
        mean = np.mean(cost)
        std = np.std(cost)
        tb.add_row([name, f"{mean:.4f} +- {std:.5f}"])
        total += cost

    mean = np.mean(total)
    std = np.std(total)
    tb.add_row(['Total', f"{mean:.4f} +- {std:.5f}"])
    return tb


def main():
    args = parse_args()
    cfg_fp = args.config
    print(cfg_fp.split('/')[-1])
    cfg = utils.Config.fromfile(cfg_fp)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu_id']
    input_size = cfg.net_size

    tensor_size = [args.batch_size, 3, input_size, input_size]
    parts = build_parts(cfg)
    cost_record = defaultdict(list)

    profile_moel(args, tensor_size, parts, cost_record)
    tb = get_summery(args, cost_record)
    print(tb)


if __name__ == '__main__':
    main()
