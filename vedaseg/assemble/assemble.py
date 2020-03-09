import os

import torch
from torch import nn

from vedaseg import utils
from vedaseg.loggers import build_logger
from vedaseg.datasets import build_dataset
from vedaseg.datasets.transforms.builder import build_transform
from vedaseg.dataloaders import build_dataloader
from vedaseg.models import build_model
from vedaseg.criteria import build_criterion
from vedaseg.optims import build_optim
from vedaseg.lr_schedulers import build_lr_scheduler
from vedaseg.utils import MetricMeter
from vedaseg.runner import build_runner


def assemble(cfg_fp, checkpoint='', test_mode=False, infer_mode=False):
    _, fullname = os.path.split(cfg_fp)
    fname, ext = os.path.splitext(fullname)
    step = 0

    cfg = utils.Config.fromfile(cfg_fp)

    # set gpu environment
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu_id']

    # make workdir if not exist
    if not infer_mode:
        root_workdir = cfg.pop('root_workdir')
        cfg['workdir'] = os.path.join(root_workdir, fname)

        os.makedirs(cfg['workdir'], exist_ok=True)

    # set seed if not None
    seed = cfg.pop('seed')
    if seed is not None:
        utils.set_random_seed(seed)

    # 1. logging
    if infer_mode:
        cfg['logger']['handlers'] = (dict(type='StreamHandler', level='WARNING'), )
        cfg['workdir'] = None
    logger = build_logger(cfg['logger'], dict(workdir=cfg['workdir']))

    loader, infer_tf, infer_size = None, None, None
    if not infer_mode:
        step += 1
        logger.info(f'Assemble, Step {step}, Build Dataset')
        # 2. data
        ## 2.1 dataset
        train_tf = build_transform(cfg['data']['train']['transforms'])
        train_dataset = build_dataset(cfg['data']['train']['dataset'], dict(transform=train_tf))

        if cfg['data'].get('val'):
            val_tf = build_transform(cfg['data']['val']['transforms'])
            val_dataset = build_dataset(cfg['data']['val']['dataset'], dict(transform=val_tf))

        step += 1
        logger.info(f'Assemble, Step {step}, Build Dataloader')
        # 2.2 dataloader
        train_loader = build_dataloader(cfg['data']['train']['loader'], dict(dataset=train_dataset))
        loader = {'train': train_loader}
        if cfg['data'].get('val'):
            val_loader = build_dataloader(cfg['data']['val']['loader'], dict(dataset=val_dataset))
            loader['val'] = val_loader
    else:
        step += 1
        logger.info(f'Assemble, Step {step}, Build Transformer')
        # 2. data
        ## 2.1 transformer
        infer_size = cfg['net_size']
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

    criterion, optim, lr_scheduler = None, None, None
    if not infer_mode:
        step += 1
        logger.info(f'Assemble, Step {step}, Build Criterion')
        # 4. criterion
        criterion = build_criterion(cfg['criterion'])

        step += 1
        logger.info(f'Assemble, Step {step}, Build Optimizer')
        # 5. optim
        optim = build_optim(cfg['optimizer'], dict(params=model.parameters()))

        step += 1
        logger.info(f'Assemble, Step {step}, Build LR Scheduler')
        # 6. lr scheduler
        lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], dict(optimizer=optim, niter_per_epoch=len(train_loader)))

    step += 1
    logger.info(f'Assemble, Step {step}, Build Runner')
    # 7. runner
    runner = build_runner(
        cfg['runner'],
        dict(
            loader=loader,
            model=model,
            criterion=criterion,
            metric=None if infer_mode else MetricMeter(cfg['nclasses']),
            optim=optim,
            lr_scheduler=lr_scheduler,
            workdir=cfg['workdir'],
            gpu=gpu,
            test_cfg=cfg.get('test_cfg', None),
            test_mode=test_mode,
            infer_mode=infer_mode,
            infer_tf=infer_tf,
            infer_size=infer_size  # TODO: read infer size from  model so that we don't need this kwarg
        )
    )

    if test_mode or infer_mode:
        cfg['resume'] = dict(checkpoint=checkpoint, resume_optimizer=False, resume_lr=False, resume_epoch=False)

    if cfg['resume']:
        runner.resume(**cfg['resume'])

    return runner
