import os

import torch
from addict import Dict
from torch import nn

from vedaseg import utils
from vedaseg.criteria import build_criterion
from vedaseg.dataloaders import build_dataloader
from vedaseg.datasets import build_dataset
from vedaseg.datasets.transforms.builder import build_transform
from vedaseg.loggers import build_logger
from vedaseg.lr_schedulers import build_lr_scheduler
from vedaseg.models import build_model
from vedaseg.optims import build_optim
from vedaseg.runners import build_runner
from vedaseg.utils import MetricMeter

try:
    from volksdep.converters import load as volksdep_load
except ImportError as e:
    volksdep_load = None


def assemble(cfg_fp, checkpoint='', test_mode=False):
    step = 0
    # load config
    cfg = utils.Config.fromfile(cfg_fp)

    # set gpu environment
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu_id']

    # make workdir if not exist
    set_workdir(cfg_fp, cfg)  # here cfg['workdir'] is updated with new filename

    # set seed if not None
    utils.set_random_seed(cfg['seed']) if 'seed' in cfg else None

    # 1. logging
    logger = build_logger(cfg['logger'], dict(workdir=cfg['workdir']))

    # set working mode (training or inferencing)
    infer_mode = cfg['runner']['type'] == 'Inferencer'
    tensorrt_mode = False
    logger.info(f'Working in {"inference" if infer_mode else "training"} mode')

    logger.info(f'Assemble, Step {step}, Build Dataset & Dataloader')
    step += 1
    # 2. data
    dataset, transform, loader = assemble_dataset(cfg['data'],
                                                  infer_mode=infer_mode)

    logger.info(f'Assemble, Step {step}, Build Model')
    step += 1
    # 3. model
    model, gpu = assemble_model(cfg, logger)

    if not infer_mode:
        logger.info(f'Assemble, Step {step}, Build Criterion')
        step += 1
        # 4. criterion
        criterion = build_criterion(cfg['criterion'])

        logger.info(f'Assemble, Step {step}, Build Optimizer')
        step += 1
        # 5. optim
        optim = build_optim(cfg['optimizer'], dict(params=model.parameters()))

        logger.info(f'Assemble, Step {step}, Build LR Scheduler')
        step += 1
        # 6. lr scheduler
        lr_scheduler = build_lr_scheduler(
            cfg['lr_scheduler'],
            dict(optimizer=optim, niter_per_epoch=len(loader.train))
        )

        logger.info(f'Assemble, Step {step}, Build Runner')
        step += 1
        # 7. runners
        runner = build_runner(
            cfg['runner'],
            dict(
                loader=loader,
                model=model,
                criterion=criterion,
                metric=MetricMeter(cfg['nclasses']),
                optim=optim,
                lr_scheduler=lr_scheduler,
                workdir=cfg['workdir'],
                gpu=gpu,
                test_cfg=cfg.get('test_cfg', None),
                test_mode=test_mode,
            )
        )

    else:
        logger.info(f'Assemble, Step {step}, Build Runner')
        step += 1
        # 7. runners
        runner = build_runner(
            cfg['runner'],
            dict(
                model=model,
                workdir=cfg['workdir'],
                gpu=gpu,
                infer_dataset=dataset.infer,
                infer_tf=transform.infer,
                loader_setting=loader.infer,
            )
        )
        if 'tensor_rt' in cfg:
            trt_dir = cfg['tensor_rt'].get('trt_model', None)
            if trt_dir is not None:
                tensorrt_mode = True

    if not tensorrt_mode:
        if test_mode or infer_mode:
            cfg['resume'] = dict(checkpoint=checkpoint,
                                 resume_optimizer=False,
                                 resume_lr=False,
                                 resume_epoch=False)

        if cfg['resume']:
            runner.resume(**cfg['resume'])

    return runner


def assemble_dataset(cfg, infer_mode=False):
    dataset, transform, loader = Dict(), Dict(), Dict()
    for usage, setting in cfg.items():
        if infer_mode ^ (usage == 'infer'):
            continue

        transform[usage] = build_transform(setting['transforms'])
        dataset[usage] = build_dataset(setting['dataset'],
                                       dict(transform=transform[usage]))
        if infer_mode:
            loader[usage] = setting['loader']
        else:
            loader[usage] = build_dataloader(setting['loader'],
                                             dict(dataset=dataset[usage]))
    return dataset, transform, loader


def assemble_model(cfg, logger):
    if 'tensor_rt' in cfg:
        trt_dir = cfg['tensor_rt'].get('trt_model', None)
        if trt_dir is not None:
            if volksdep_load is None:
                msg = ('Volksdep is not available, please install newest '
                       'version from https://github.com/Media-Smart/volksdep')
                raise ImportError(msg)
            if not torch.cuda.is_available():
                raise AssertionError('No available GPU for TensorRT model')
            trt_model = volksdep_load(cfg['tensor_rt']['trt_model'])
            logger.info('Load TensorRT model from {}'.format(trt_dir))
            return trt_model, True

    model, gpu = build_model(cfg['model']), False
    if torch.cuda.is_available():
        logger.info('Using GPU {}'.format(cfg['gpu_id']))
        gpu = True
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.cuda()
        if cfg['runner']['type'] == 'infer':
            model.eval()
    else:
        logger.info('Using CPU')
    return model, gpu


def set_workdir(cfg_fp, cfg):
    _, fullname = os.path.split(cfg_fp)
    fname, _ = os.path.splitext(fullname)
    root_workdir = cfg.pop('root_workdir')
    cfg['workdir'] = os.path.join(root_workdir, fname)
    os.makedirs(cfg['workdir'], exist_ok=True)
