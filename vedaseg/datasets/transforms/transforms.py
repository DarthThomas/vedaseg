import logging
import random
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tt
from PIL import Image

from .base import BaseTransform
from .registry import TRANSFORMS

CV2_MODE = {
    'bilinear': cv2.INTER_LINEAR,
    'nearest': cv2.INTER_NEAREST,
    'cubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
}

CV2_BORDER_MODE = {
    'constant': cv2.BORDER_CONSTANT,
    'reflect': cv2.BORDER_REFLECT,
    'reflect101': cv2.BORDER_REFLECT101,
    'replicate': cv2.BORDER_REPLICATE,
}

logger = logging.getLogger()


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **kwargs):  # image, mask, detail, inverse=False
        state = deepcopy(kwargs)
        inverse = kwargs.get("inverse", False)
        if inverse:
            details = state.pop('details', None)
            assert details is not None, "Details not provided for inverse " \
                                        "transform."
            transforms = reversed(self.transforms)
            details = reversed(details)
            for t, detail in zip(transforms, details):
                state.update({'details': detail})
                state = t(**state)
            return self.unpack(state)
        else:
            details = state.get('details', [])
            assert len(details) == 0, f"Should start recording with an empty" \
                                      f" list while list {details} with " \
                                      f"length '{len(details)}' provided."
            transforms = self.transforms
            for t in transforms:
                state = t(**state)
            return self.unpack(state)

    @staticmethod
    def unpack(state):
        res = []
        for key in ['image', 'mask', 'details']:
            if state.get(key, None) is not None:
                res.append(state[key])
        if len(res) == 1:
            return res[0]
        return res


@TRANSFORMS.register_module
class FactorScale(BaseTransform):
    def __init__(self, scale_factor=1.0, mode='bilinear'):
        super().__init__()
        self.mode = mode
        self.scale_factor = scale_factor

    @staticmethod
    def check_scale_factor(scale_factor):
        if scale_factor == 0:
            raise ValueError('Encountered zero scale factor.')
        if scale_factor < 0.01 or scale_factor > 100:
            logger.warning(f"Encountered scale change larger than 100 "
                           f"with scale factor: {scale_factor}.")

    def image_forward(self, image, **kwargs):
        scale_factor = kwargs.get('scale_factor', self.scale_factor)
        self.check_scale_factor(scale_factor)

        if scale_factor == 1.0:
            return image

        new_h = int(image.shape[0] * scale_factor)
        new_w = int(image.shape[1] * scale_factor)

        if kwargs.get('details', None) is not None:
            info = {'shape_orig': image.shape[:2],
                    'scale_factor': self.scale_factor}
            self.transform_detail['image'].update(info)

        torch_image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        torch_image = F.interpolate(torch_image, size=(new_h, new_w),
                                    mode=self.mode, align_corners=True)
        new_image = torch_image.squeeze().permute(1, 2, 0).numpy()

        return new_image

    def mask_forward(self, mask, **kwargs):
        scale_factor = kwargs.get('scale_factor', self.scale_factor)
        self.check_scale_factor(scale_factor)

        if scale_factor == 1.0:
            return mask

        new_h = int(mask.shape[0] * scale_factor)
        new_w = int(mask.shape[1] * scale_factor)

        if kwargs.get('details', None) is not None:
            info = {'shape_orig': mask.shape[:2],
                    'scale_factor': self.scale_factor}
            self.transform_detail['mask'].update(info)

        torch_mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        torch_mask = F.interpolate(torch_mask, size=(new_h, new_w),
                                   mode='nearest')
        new_mask = torch_mask.squeeze().numpy()

        return new_mask

    def image_inverse(self, image, **kwargs):
        detail = self.load_detail(target='image', **kwargs)
        scale_factor = detail.get('scale_factor', 0)
        self.check_scale_factor(scale_factor)
        self.image_forward(image, scale_factor=1.0 / scale_factor)

    def mask_inverse(self, mask, **kwargs):
        detail = self.load_detail(target='mask', **kwargs)
        scale_factor = detail.get('scale_factor', 0)
        self.check_scale_factor(scale_factor)
        self.image_forward(mask, scale_factor=1.0 / scale_factor)


@TRANSFORMS.register_module
class SizeScale(FactorScale):
    def __init__(self, target_size, mode='bilinear'):
        self.target_size = target_size
        super().__init__(mode=mode)

    def get_scale_factor(self, long_edge):
        self.scale_factor = self.target_size / long_edge

    def image_forward(self, image, **kwargs):
        self.get_scale_factor(max(image.shape[:2]))
        super().image_forward(image, **kwargs)

    def mask_forward(self, mask, **kwargs):
        self.get_scale_factor(max(mask.shape[:2]))
        super().mask_forward(mask, **kwargs)


@TRANSFORMS.register_module
class RandomScale(FactorScale):
    def __init__(self, min_scale, max_scale, scale_step=0.0, mode='bilinear'):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_step = scale_step
        super().__init__(mode=mode)

    @staticmethod
    def get_scale_factor(min_scale, max_scale, scale_step):
        if min_scale == max_scale:
            return min_scale

        if scale_step == 0:
            return random.uniform(min_scale, max_scale)

        num_steps = int((max_scale - min_scale) / scale_step + 1)
        scale_factors = np.linspace(min_scale, max_scale, num_steps)
        scale_factor = np.random.choice(scale_factors).item()

        return scale_factor

    def update_params(self, ):
        self.scale_factor = self.get_scale_factor(self.min_scale,
                                                  self.max_scale,
                                                  self.scale_step)


@TRANSFORMS.register_module
class PadIfNeeded(BaseTransform):
    def __init__(self, height, width, image_value, mask_value):
        self.height = height
        self.width = width
        self.image_value = image_value
        self.mask_value = mask_value
        self.channel = len(image_value)
        super().__init__()

    def image_forward(self, image, **kwargs):
        h, w = image.shape[:2]

        assert h <= self.height and w <= self.width

        target_height = h + max(self.height - h, 0)
        target_width = w + max(self.width - w, 0)

        image_pad_value = np.reshape(np.array(self.image_value,
                                              dtype=image.dtype),
                                     [1, 1, self.channel])

        new_image = np.tile(image_pad_value, (target_height, target_width, 1))

        new_image[:h, :w, :] = image
        if kwargs.get('details', None) is not None:
            self.transform_detail['image'] = {'shape_orig': image.shape[:2]}

        return new_image

    def mask_forward(self, mask, **kwargs):
        h, w = mask.shape[:2]

        assert h <= self.height and w <= self.width

        target_height = h + max(self.height - h, 0)
        target_width = w + max(self.width - w, 0)

        mask_pad_value = np.reshape(np.array(self.mask_value,
                                             dtype=mask.dtype),
                                    [1, 1])

        new_mask = np.tile(mask_pad_value, (target_height, target_width))

        new_mask[:h, :w] = mask
        if kwargs.get('details', None) is not None:
            self.transform_detail['mask'] = {'shape_orig': mask.shape[:2]}
        return new_mask

    def image_inverse(self, image, **kwargs):
        detail = self.load_detail(target='image', **kwargs)

        shape = detail.get('shape_orig', None)
        assert shape is not None, 'No detail provided for inverse transform'
        h, w = shape
        return image[:h, :w, :]

    def mask_inverse(self, mask, **kwargs):
        detail = self.load_detail(target='mask', **kwargs)

        shape = detail.get('shape_orig', None)
        assert shape is not None, 'No detail provided for inverse transform'
        h, w = shape
        return mask[:h, :w]


@TRANSFORMS.register_module
class RandomCrop(PadIfNeeded):
    def __init__(self, height, width, image_value, mask_value):
        self.height = height
        self.width = width
        self.image_value = image_value
        self.mask_value = mask_value
        self.channel = len(image_value)
        self.crop_info = None
        super().__init__()

    def get_crop_info(self, image):
        h, w = image.shape[:2]
        target_height = h + max(self.height - h, 0)
        target_width = w + max(self.width - w, 0)
        y1 = int(random.uniform(0, target_height - self.height + 1))
        y2 = y1 + self.height
        x1 = int(random.uniform(0, target_width - self.width + 1))
        x2 = x1 + self.width
        self.crop_info = (y1, y2, x1, x2)

    def update_params(self, **kwargs):
        image = kwargs.get('image', None)
        mask = kwargs.get('mask', None)
        if image is not None:
            self.get_crop_info(image)
        elif mask is not None:
            self.get_crop_info(mask)
        else:
            raise ValueError('Neither image or mask was provided')

    def image_forward(self, image, **kwargs):
        y1, y2, x1, x2 = self.crop_info
        padded = super().image_forward(image, **kwargs)
        new_image = padded[y1:y2, x1:x2, :]
        if kwargs.get('details', None) is not None:
            info = {'shape_padded': image.shape[:2],
                    'crop_info': self.crop_info}
            self.transform_detail['image'].update(info)
        return new_image

    def mask_forward(self, mask, **kwargs):
        y1, y2, x1, x2 = self.crop_info
        padded = super().mask_forward(mask, **kwargs)
        new_mask = padded[y1:y2, x1:x2]
        if kwargs.get('details', None) is not None:
            info = {'shape_padded': mask.shape[:2],
                    'crop_info': self.crop_info}
            self.transform_detail['mask'].update(info)
        return new_mask

    def image_inverse(self, image, **kwargs):
        detail = self.load_detail(target='image', **kwargs)
        image_pad_value = np.reshape(np.array(self.image_value,
                                              dtype=image.dtype),
                                     [1, 1, self.channel])
        shape = detail.get('shape_padded', None)
        crop_info = detail.get('crop_info', None)
        for _ in [shape, crop_info]:
            assert _ is not None, 'No padded shape provided for inverse ' \
                                  'transform'
        h, w = shape
        y1, y2, x1, x2 = crop_info
        padded = np.tile(image_pad_value, (h, w, 1))
        padded[y1:y2, x1:x2, :] = image
        return padded

    def mask_inverse(self, mask, **kwargs):
        detail = self.load_detail(target='mask', **kwargs)
        mask_pad_value = np.reshape(np.array(self.mask_value,
                                             dtype=mask.dtype),
                                    [1, 1])
        shape = detail.get('shape_padded', None)
        crop_info = detail.get('crop_info', None)
        for _ in [shape, crop_info]:
            assert _ is not None, 'No padded shape provided for inverse ' \
                                  'transform'
        h, w = shape
        y1, y2, x1, x2 = crop_info

        padded = np.tile(mask_pad_value, (h, w))
        padded[y1:y2, x1:x2] = mask
        return padded


@TRANSFORMS.register_module
class HorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
        self.random_number = 0.0

    def update_params():
        self.random_number = random.random() 

    def image_forward(self, image, **kwargs):
        if self.random_number > self.p:
            image = cv2.flip(image, 1)
        if kwargs.get('details', None) is not None:
            info = {'flipped': self.random_number > self.p}
            self.transform_detail['image'].update(info)
        return image

    def mask_forward(self, mask, **kwargs):
        if self.random_number > self.p:
            mask = cv2.flip(mask, 1)
        if kwargs.get('details', None) is not None:
            info = {'flipped': self.random_number > self.p}
            self.transform_detail['mask'].update(info)
        return mask

    def image_inverse(self, image, **kwargs):
        detail = self.load_detail(target='image', **kwargs)
        flipped = detail.get('flipped', None)
        

        if self.random_number > self.p:
            image = cv2.flip(image, 1)
        if kwargs.get('details', None) is not None:
            info = {'flipped': self.random_number > self.p}
            self.transform_detail['image'].update(info)
        return image




    def __call__(self, image, mask):
        if random.random() > self.p:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        return image, mask


@TRANSFORMS.register_module
class RandomRotate:
    def __init__(self,
                 p=0.5,
                 degrees=30,
                 mode='bilinear',
                 border_mode='reflect101',
                 image_value=None,
                 mask_value=None):
        self.p = p
        self.degrees = (-degrees, degrees) if isinstance(degrees, (int, float)) else degrees
        self.mode = CV2_MODE[mode]
        self.border_mode = CV2_BORDER_MODE[border_mode]
        self.image_value = image_value
        self.mask_value = mask_value

    def __call__(self, image, mask):
        if random.random() < self.p:
            h, w, c = image.shape

            angle = random.uniform(*self.degrees)
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

            image = cv2.warpAffine(image,
                                   M=matrix,
                                   dsize=(w, h),
                                   flags=self.mode,
                                   borderMode=self.border_mode,
                                   borderValue=self.image_value)
            mask = cv2.warpAffine(mask,
                                  M=matrix,
                                  dsize=(w, h),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=self.border_mode,
                                  borderValue=self.mask_value)

        return image, mask


@TRANSFORMS.register_module
class GaussianBlur:
    def __init__(self, p=0.5, ksize=7):
        self.p = p
        self.ksize = (ksize, ksize) if isinstance(ksize, int) else ksize

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = cv2.GaussianBlur(image, ksize=self.ksize, sigmaX=0)

        return image, mask


@TRANSFORMS.register_module
class Normalize:
    def __init__(self,
                 mean=(123.675, 116.280, 103.530),
                 std=(58.395, 57.120, 57.375)):
        self.mean = mean
        self.std = std
        self.channel = len(mean)

    def __call__(self, image, mask):
        mean = np.reshape(np.array(self.mean, dtype=image.dtype),
                          [1, 1, self.channel])
        std = np.reshape(np.array(self.std, dtype=image.dtype),
                         [1, 1, self.channel])
        denominator = np.reciprocal(std, dtype=image.dtype)

        new_image = (image - mean) * denominator
        new_mask = mask

        return new_image, new_mask


@TRANSFORMS.register_module
class ColorJitter(tt.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness=brightness,
                         contrast=contrast,
                         saturation=saturation,
                         hue=hue)

    def __call__(self, image, mask=None):
        new_image = Image.fromarray(image.astype(np.uint8))
        new_image = super().__call__(new_image)
        new_image = np.array(new_image).astype(np.float32)
        return new_image, mask


@TRANSFORMS.register_module
class ToTensor:
    def __call__(self, image, mask):
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask)

        return image, mask
