from torchvision.transforms import ColorJitter

from vedaseg.utils import Registry

TRANSFORMS = Registry('transforms')
TRANSFORMS.register_module(ColorJitter)
