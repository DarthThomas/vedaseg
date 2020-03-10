import os

import torch
from torch import nn

from vedaseg import utils
from vedaseg.loggers import build_logger
from vedaseg.datasets.transforms.builder import build_transform
from vedaseg.models import build_model
from vedaseg.runner import build_runner


def assemble(cfg_fp, checkpoint='', verbose=False):
    _, fullname = os.path.split(cfg_fp)
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
    cfg['logger']['handlers'] = (dict(type='StreamHandler', level=level), )
    cfg['workdir'] = None
    logger = build_logger(cfg['logger'], dict(workdir=cfg['workdir']))

    step += 1
    logger.info(f'Assemble, Step {step}, Build Transformer')
    # 2. data
    ## 2.1 transformer
    head_size = cfg['net_size']
    infer_tf = build_transform(cfg['data']['infer']['transforms'])

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
    # 7. runner
    runner = build_runner(
        cfg['runner'],
        dict(
            loader=loader,
            model=model,
            gpu=gpu,
            test_mode=test_mode,
            infer_mode=infer_mode,
            infer_tf=infer_tf,
            head_size=head_size  # TODO: read infer size from  model so that we don't need this kwarg
        )
    )

    if test_mode or infer_mode:
        cfg['resume'] = dict(checkpoint=checkpoint, resume_optimizer=False, resume_lr=False, resume_epoch=False)

    if cfg['resume']:
        runner.resume(**cfg['resume'])

    return runner
