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

        if kwargs.get('details', None) is not None:
            info = {'shape_orig': image.shape[:2],
                    'scale_factor': self.scale_factor}
            self.transform_detail['image'].update(info)

        if scale_factor == 1.0:
            return image

        new_h = int(image.shape[0] * scale_factor)
        new_w = int(image.shape[1] * scale_factor)
        torch_image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        torch_image = F.interpolate(torch_image, size=(new_h, new_w),
                                    mode=self.mode, align_corners=True)
        new_image = torch_image.squeeze().permute(1, 2, 0).numpy()

        return new_image

    def mask_forward(self, mask, **kwargs):
        scale_factor = kwargs.get('scale_factor', self.scale_factor)
        self.check_scale_factor(scale_factor)

        if kwargs.get('details', None) is not None:
            info = {'shape_orig': mask.shape[:2],
                    'scale_factor': self.scale_factor}
            self.transform_detail['mask'].update(info)

        if scale_factor == 1.0:
            return mask

        new_h = int(mask.shape[0] * scale_factor)
        new_w = int(mask.shape[1] * scale_factor)
        torch_mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        torch_mask = F.interpolate(torch_mask, size=(new_h, new_w),
                                   mode='nearest')
        new_mask = torch_mask.squeeze().numpy()

        return new_mask

    def image_inverse(self, image, **kwargs):
        detail = self.load_detail(target='image', **kwargs)
        shape_orig = detail.get('shape_orig', None)
        assert shape_orig is not None, f"Shape info not provided for transform"
        new_h, new_w = shape_orig

        torch_image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        torch_image = F.interpolate(torch_image, size=(new_h, new_w),
                                    mode=self.mode, align_corners=True)
        new_image = torch_image.squeeze().permute(1, 2, 0).numpy()

        return new_image

    def mask_inverse(self, mask, **kwargs):
        detail = self.load_detail(target='mask', **kwargs)
        shape_orig = detail.get('shape_orig', None)
        assert shape_orig is not None, f"Shape info not provided for transform"
        new_h, new_w = shape_orig

        torch_mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        torch_mask = F.interpolate(torch_mask, size=(new_h, new_w),
                                   mode='nearest')
        new_mask = torch_mask.squeeze().numpy()

        return new_mask


@TRANSFORMS.register_module
class SizeScale(FactorScale):
    def __init__(self, target_size, mode='bilinear'):
        self.target_size = target_size
        super().__init__(mode=mode)

    def get_scale_factor(self, data):
        if isinstance(self.target_size, int):
            scale_factor = self.target_size / max(data.shape[:2])
        else:
            scale_factor = None
        return scale_factor

    def update_params(self, **kwargs):
        image = kwargs.get('image', None)
        mask = kwargs.get('mask', None)
        recorded = False
        if image is not None:
            self.scale_factor = self.get_scale_factor(image)
            recorded = True
        if mask is not None:
            scale_factor = self.get_scale_factor(mask)
            if recorded:
                if scale_factor != self.scale_factor:
                    raise ValueError(f"Got different scale factor between "
                                     f"image: {self.scale_factor} and "
                                     f"mask: {scale_factor}.")
            else:
                self.scale_factor = scale_factor
            recorded = True
        if not recorded:
            raise ValueError('Neither image or mask was provided')

    def image_forward(self, image, **kwargs):
        if isinstance(self.target_size, int):
            return super().image_forward(image, **kwargs)

        if kwargs.get('details', None) is not None:
            info = {'shape_orig': image.shape[:2]}
            self.transform_detail['image'].update(info)

        torch_image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        torch_image = F.interpolate(torch_image, size=self.target_size,
                                    mode=self.mode, align_corners=True)
        new_image = torch_image.squeeze().permute(1, 2, 0).numpy()

        return new_image

    def mask_forward(self, mask, **kwargs):
        if isinstance(self.target_size, int):
            return super().mask_forward(mask, **kwargs)

        if kwargs.get('details', None) is not None:
            info = {'shape_orig': mask.shape[:2]}
            self.transform_detail['mask'].update(info)

        torch_mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        torch_mask = F.interpolate(torch_mask, size=self.target_size,
                                   mode='nearest')
        new_mask = torch_mask.squeeze().numpy()

        return new_mask


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

    def update_params(self, **kwargs):
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

    def check_hw(self, shape):
        h, w = shape
        if self.__class__.__name__ == 'PadIfNeeded':
            if h > self.height or w > self.width:
                raise ValueError(f"Original image shape: {shape}, already "
                                 f"larger than intended padding area: "
                                 f"{self.height, self.width}")

    def image_forward(self, image, **kwargs):
        h, w = image.shape[:2]
        self.check_hw(image.shape[:2])

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
        self.check_hw(mask.shape[:2])

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
        self.channel = len(image_value)
        self.crop_info = None
        self.shape_orig = None
        super().__init__(height=height,
                         width=width,
                         image_value=image_value,
                         mask_value=mask_value)

    def get_crop_info(self, image):
        h, w = image.shape[:2]
        target_height = h + max(self.height - h, 0)
        target_width = w + max(self.width - w, 0)
        y1 = int(random.uniform(0, target_height - self.height + 1))
        y2 = y1 + self.height
        x1 = int(random.uniform(0, target_width - self.width + 1))
        x2 = x1 + self.width
        self.crop_info = (y1, y2, x1, x2)
        self.shape_orig = h, w

    def update_params(self, **kwargs):
        image = kwargs.get('image', None)
        mask = kwargs.get('mask', None)
        if image is not None:
            self.get_crop_info(image)
        elif mask is not None:
            self.get_crop_info(mask)
        else:
            raise ValueError('Neither image or mask was provided')

    def shared_apply(self, data, target=None, details=None):
        y1, y2, x1, x2 = self.crop_info
        if target == 'image':
            padded = super().image_forward(data)
            new_data = padded[y1:y2, x1:x2, :]
        elif target == 'mask':
            padded = super().mask_forward(data)
            new_data = padded[y1:y2, x1:x2]
        else:
            raise ValueError(f"Unrecognized target:{target}.")
        return new_data

    def record_detail(self, target='image'):
        info = {'shape_orig': self.shape_orig,
                'crop_info': self.crop_info}
        self.transform_detail['image'].update(info)

    def image_forward(self, image, **kwargs):
        return self.apply_save(image, target='image', **kwargs)

    def mask_forward(self, mask, **kwargs):
        return self.apply_save(mask, target='mask', **kwargs)

    def image_inverse(self, image, **kwargs):
        detail = self.load_detail(target='image', **kwargs)
        image_pad_value = np.reshape(np.array(self.image_value,
                                              dtype=image.dtype),
                                     [1, 1, self.channel])
        shape = detail.get('shape_orig', None)
        crop_info = detail.get('crop_info', None)
        for _ in [shape, crop_info]:
            assert _ is not None, 'No padded shape provided for inverse ' \
                                  'transform'
        h, w = shape
        y1, y2, x1, x2 = crop_info
        padded = np.tile(image_pad_value, (y2 - y1 + h, x2 - x1 + w, 1))
        padded[y1:y2, x1:x2, :] = image
        return padded[:h, :w]

    def mask_inverse(self, mask, **kwargs):
        detail = self.load_detail(target='mask', **kwargs)
        mask_pad_value = np.reshape(np.array(self.mask_value,
                                             dtype=mask.dtype),
                                    [1, 1])
        shape = detail.get('shape_orig', None)
        crop_info = detail.get('crop_info', None)
        for _ in [shape, crop_info]:
            assert _ is not None, 'No padded shape provided for inverse ' \
                                  'transform'
        h, w = shape
        y1, y2, x1, x2 = crop_info
        padded = np.tile(mask_pad_value, (y2 - y1 + h, x2 - x1 + w))
        padded[y1:y2, x1:x2] = mask
        return padded[:h, :w]


@TRANSFORMS.register_module
class HorizontalFlip(BaseTransform):
    def __init__(self, p=0.5):
        self.p = p
        self.random_number = 0.0
        super().__init__()

    def update_params(self, **kwargs):
        self.random_number = random.random()

    def shared_apply(self, data, target=None, **kwargs):
        return cv2.flip(data, 1)

    def record_detail(self, target='image'):
        info = {'flipped': self.random_number < self.p}
        self.transform_detail[target].update(info)

    def apply_condition(self):
        return self.random_number < self.p

    def image_forward(self, image, details=None, **kwargs):
        return self.apply_save(image, 'image', details=details)

    def mask_forward(self, mask, details=None, **kwargs):
        return self.apply_save(mask, 'mask', details=details)

    def image_inverse(self, image, details=None, **kwargs):
        detail = self.load_detail(target='image', details=details)
        flipped = detail.get('flipped', None)
        assert flipped is not None, 'No flip info provided for transform.'
        if not flipped:
            return image
        return self.shared_apply(image, 'image')

    def mask_inverse(self, mask, details=None, **kwargs):
        detail = self.load_detail(target='mask', details=details)
        flipped = detail.get('flipped', None)
        assert flipped is not None, 'No flip info provided for transform.'
        if not flipped:
            return mask
        return self.shared_apply(mask, 'mask')


@TRANSFORMS.register_module
class RandomRotate(HorizontalFlip):
    def __init__(self,
                 p=0.5,
                 degrees=30,
                 mode='bilinear',
                 border_mode='reflect101',
                 image_value=None,
                 mask_value=None):
        self.degrees = degrees
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        self.mode = CV2_MODE[mode]
        self.border_mode = CV2_BORDER_MODE[border_mode]
        self.image_value = image_value
        self.mask_value = mask_value
        self.angle = 0
        super().__init__(p=p)

    def get_rotate_info(self):
        self.angle = random.uniform(*self.degrees)

    def update_params(self, **kwargs):
        self.random_number = random.random()
        if self.random_number < self.p:
            self.get_rotate_info()

    def shared_apply(self, data, target=None, details=None):
        h, w = data.shape[:2]
        flags = self.mode
        border_value = self.image_value
        if target == 'mask':
            flags = cv2.INTER_NEAREST
            border_value = self.mask_value

        angle = self.angle
        if details is not None:
            angle = details.get('angle', None)
            assert angle is not None, 'No rotate info provided for transform.'
            angle = -angle
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        data = cv2.warpAffine(data,
                              M=matrix,
                              dsize=(w, h),
                              flags=flags,
                              borderMode=self.border_mode,
                              borderValue=border_value)

        return data

    def record_detail(self, target='image'):
        info = {'rotated': self.apply_condition(),
                'angle': self.angle}
        self.transform_detail[target].update(info)

    def image_inverse(self, image, details=None, **kwargs):
        details = self.load_detail(target='image', details=details)
        rotated = details.get('rotated', None)
        assert rotated is not None, 'No rotate info provided for transform.'
        if not rotated:
            return image
        return self.apply_save(image, 'image', inverse=True, details=details)

    def mask_inverse(self, mask, details=None, **kwargs):
        details = self.load_detail(target='mask', details=details)
        rotated = details.get('rotated', None)
        assert rotated is not None, 'No rotate info provided for transform.'
        if not rotated:
            return mask
        return self.apply_save(mask, 'mask', inverse=True, details=details)


@TRANSFORMS.register_module
class GaussianBlur(BaseTransform):
    def __init__(self, p=0.5, ksize=7):
        self.p = p
        self.random_number = 0.0
        self.ksize = (ksize, ksize) if isinstance(ksize, int) else ksize
        super().__init__()

    def update_params(self, **kwargs):
        self.random_number = random.random()

    def shared_apply(self, data, target=None, **kwargs):
        return cv2.GaussianBlur(data, ksize=self.ksize, sigmaX=0)

    def record_detail(self, target='image'):
        info = {'blurred': self.apply_condition(),
                'kernel_size': self.ksize}
        self.transform_detail[target].update(info)

    def apply_condition(self):
        return self.random_number < self.p

    def image_forward(self, image, details=None, **kwargs):
        return self.apply_save(image, 'image', details=details)

    def image_inverse(self, image, details=None, **kwargs):
        logger.warning('Skipped inverse gaussian blur')
        return image


@TRANSFORMS.register_module
class Normalize(BaseTransform):
    def __init__(self,
                 mean=(123.675, 116.280, 103.530),
                 std=(58.395, 57.120, 57.375)):
        self.mean = mean
        self.std = std
        self.channel = len(mean)
        super().__init__()

    def shared_apply(self, data, target=None, details=None):
        if target == 'image':
            mean = np.reshape(np.array(self.mean, dtype=data.dtype),
                              [1, 1, self.channel])
            std = np.reshape(np.array(self.std, dtype=data.dtype),
                             [1, 1, self.channel])

            if details is not None:
                new_data = data * std + mean
            else:
                denominator = np.reciprocal(std, dtype=data.dtype)
                new_data = (data - mean) * denominator
            return new_data
        else:
            return data

    def record_detail(self, target='image'):
        info = {'Normalized': True}
        self.transform_detail[target].update(info)

    def image_forward(self, image, details=None, **kwargs):
        return self.apply_save(image, 'image', details=details)

    def image_inverse(self, image, details=None, **kwargs):
        return self.apply_save(image, 'image', inverse=True, details=details)


@TRANSFORMS.register_module
class ColorJitter(BaseTransform):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.jitter = tt.ColorJitter(brightness=brightness,
                                     contrast=contrast,
                                     saturation=saturation,
                                     hue=hue)
        super().__init__()

    def shared_apply(self, data, target=None, details=None):
        if target == 'image':
            new_image = Image.fromarray(data.astype(np.uint8))
            new_image = self.jitter(new_image)
            new_image = np.array(new_image).astype(np.float32)
            return new_image
        else:
            return data

    def record_detail(self, target='image'):
        info = {'Jittered': True}
        self.transform_detail[target].update(info)

    def image_forward(self, image, details=None, **kwargs):
        return self.apply_save(image, 'image', details=details)

    def image_inverse(self, image, details=None, **kwargs):
        logger.warning('Skipped inverse color jitter (not implemented yet)')
        return image


@TRANSFORMS.register_module
class ToTensor(BaseTransform):
    def shared_apply(self, data, target=None, details=None):
        new_data = torch.from_numpy(data)
        if target == 'image':
            return new_data.permute(2, 0, 1)
        return new_data

    def record_detail(self, target='image'):
        info = {'Tensorified': True}
        self.transform_detail[target].update(info)

    def image_forward(self, image, details=None, **kwargs):
        return self.apply_save(image, 'image', details=details)

    def mask_forward(self, mask, details=None, **kwargs):
        return self.apply_save(mask, 'mask', details=details)

    def image_inverse(self, image, details=None, **kwargs):
        new_image = image.permute(1, 2, 0)
        return new_image.cpu().numpy()

    def mask_inverse(self, mask, details=None, **kwargs):
        return mask.cpu().numpy()
