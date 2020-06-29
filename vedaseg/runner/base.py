import logging

import torch
from torch import nn

from vedaseg.models import build_model
from vedaseg.utils import Config
from vedaseg.utils.checkpoint import load_checkpoint
from .registry import RUNNERS

try:
    from volksdep.converters import load as volksdep_load
except ImportError as e:
    volksdep_load = None

logger = logging.getLogger()


@RUNNERS.register_module
class BaseRunner:
    """
    Base runner, provides basic runner utility such as build module
    """

    def __init__(self, cfg_fp, check_point_fp=None):
        self.build_step = 0
        self.epoch = None
        self.start_epoch = None
        self.iter = None
        self.lr = None
        self.model = None
        self.gpu_mode = True

        cfg = Config.fromfile(cfg_fp)

        self.prepare_model(cfg, check_point_fp=None)

        # self.loader = loader

    def __call__(self):
        pass

    def prepare_model_(self, cfg, check_point_fp=None):
        if torch.cuda.is_available():
            logger.info('Using GPU {}'.format(cfg['gpu_id']))

            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
            self.model.cuda()
            if cfg['runner']['type'] in ['infer', 'test']:
                self.model.eval()

        if cfg.get('use_trt_model') is not None:
            trt_dir = cfg['use_trt_model'].get('model_dir', None)
            if trt_dir is not None:
                if volksdep_load is None:
                    msg = ('Failed to import volksdep, please verify '
                           'installation or install latest version from: '
                           'https://github.com/Media-Smart/volksdep')
                    raise ImportError(msg)
                self.model = volksdep_load(trt_dir)
                logger.info(f'Load TensorRT model from {trt_dir}')
            else:
                raise ValueError('No TensorRT engine provided')
        else:
            self.model = build_model(cfg['model'])

            else:
            logger.info('Using CPU')
            self.gpu_mode = False

        resume_config = cfg['resume'] if 'resume' in cfg else dict()

        if check_point_fp is not None:
            resume_config.update(dict(checkpoint=check_point_fp, ))

        self.resume(**resume_config)


def prepare_model(self, cfg, check_point_fp=None):
    if cfg.get('use_trt_model') is not None:
        trt_dir = cfg['use_trt_model'].get('model_dir', None)
        if trt_dir is not None:
            if volksdep_load is None:
                msg = ('Failed to import volksdep, please verify '
                       'installation or install latest version from: '
                       'https://github.com/Media-Smart/volksdep')
                raise ImportError(msg)
            self.model = volksdep_load(trt_dir)
            logger.info(f'Load TensorRT model from {trt_dir}')
        else:
            raise ValueError('No TensorRT engine provided')
    else:
        self.model = build_model(cfg['model'])
        if torch.cuda.is_available():
            logger.info('Using GPU {}'.format(cfg['gpu_id']))
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
            self.model.cuda()
            if cfg['runner']['type'] in ['infer', 'test']:
                self.model.eval()
        else:
            logger.info('Using CPU')
            self.gpu_mode = False

        resume_config = cfg['resume'] if 'resume' in cfg else dict()

        if check_point_fp is not None:
            resume_config.update(dict(checkpoint=check_point_fp, ))

        self.resume(**resume_config)


def load_checkpoint(self, filename, map_location='cpu', strict=False):
    logger.info('Resume from %s', filename)
    return load_checkpoint(self.model, filename, map_location, strict,
                           logger)


def resume(self,
           checkpoint,
           resume_optimizer=False,
           resume_lr=False,
           resume_epoch=False,
           map_location='default'):
    if map_location == 'default':
        device_id = torch.cuda.current_device()
        map_location = lambda storage, loc: storage.cuda(device_id)  # noqa
    checkpoint = self.load_checkpoint(checkpoint, map_location=map_location)
    if 'optimizer' in checkpoint and resume_optimizer:
        self.optim.load_state_dict(checkpoint['optimizer'])
    if resume_epoch:
        self.start_epoch = self.epoch = checkpoint['meta']['epoch']
        self.iter = checkpoint['meta']['iter']
    if resume_lr:
        self.lr = checkpoint['meta']['lr']

# @staticmethod
# def set_workdir(cfg_fp, cfg):
#     _, fullname = osp.split(cfg_fp)
#     fname, _ = osp.splitext(fullname)
#     root_workdir = cfg.pop('root_workdir')
#     cfg['workdir'] = osp.join(root_workdir, fname)
#     os.makedirs(cfg['workdir'], exist_ok=True)
