import os

import torch
from torch import nn

from .. import utils
from ..datasets import build_dataset
from ..datasets.transforms.builder import build_transform
from ..loggers import build_logger
from ..models import build_model
from ..runner import build_runner


def assemble(cfg_fp, checkpoint='', verbose=False):
    step = 0
    cfg = utils.Config.fromfile(cfg_fp)

    # set gpu environment
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu_id']

    # set seed if not None
    seed = cfg.pop('seed')
    if seed is not None:
        utils.set_random_seed(seed)

    # 1. logging
    level = 'INFO' if verbose else 'WARNING'
    cfg['logger']['handlers'] = (dict(type='StreamHandler', level=level),)
    cfg['workdir'] = None
    logger = build_logger(cfg['logger'], dict(workdir=cfg['workdir']))

    step += 1
    logger.info(f'Assemble, Step {step}, Build Transformer')
    # 2. data
    ## 2.1 transformer
    infer_tf = build_transform(cfg['data']['infer']['transforms'])
    infer_dataset = build_dataset(cfg['data']['infer']['dataset'],
                                  dict(transform=infer_tf))

    step += 1
    logger.info(f'Assemble, Step {step}, Build Model')
    # 3. model
    model = build_model(cfg['model'])
    if torch.cuda.is_available():
        logger.info('Using GPU {}'.format(cfg['gpu_id']))
        gpu = True
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.cuda()
    else:
        logger.info('Using CPU')
        gpu = False

    step += 1
    logger.info(f'Assemble, Step {step}, Build Runner')
    # 4. runner
    runner = build_runner(
        cfg['runner'],
        dict(
            model=model.eval(),
            gpu=gpu,
            infer_dataset=infer_dataset,
            infer_tf=infer_tf,
            loader_setting=cfg['data']['infer']['loader_setting']
        )
    )

    cfg['resume'] = dict(checkpoint=checkpoint)

    runner.resume(**cfg['resume'])

    return runner
