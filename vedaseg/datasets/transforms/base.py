import logging
from collections import defaultdict
from copy import deepcopy

VALID_TARGETS = ['image', 'mask']

logger = logging.getLogger()


class BaseTransform:

    def __init__(self):
        self.valid_targets = VALID_TARGETS
        self.transform_detail = defaultdict(dict)

    def __call__(self, **kwargs):  # image, mask, detail
        self.validate_kwargs(**kwargs)
        self.update_params(**kwargs)
        res = self.apply_to_targets(**kwargs)
        self.update_details(res)

        return res

    def validate_kwargs(self, **kwargs):
        for kwarg in kwargs:
            if kwarg.lower() in ['details', 'inverse']:
                continue
            if kwarg.lower() not in self.valid_targets:
                raise ValueError(f"Unsupported target '{kwarg}' provided, "
                                 f"currently support target "
                                 f"in {self.valid_targets}.")

    def update_details(self, res):
        inverse = res.get('inverse', False)
        details = res.get('details', None)
        if not inverse and details is not None:
            res['details'].append(self.transform_detail)

    def apply_to_targets(self, **kwargs):
        res = deepcopy(kwargs)
        inverse = res.get('inverse', False)
        details = res.get('details', None)

        for target in self.valid_targets:
            value = kwargs.get(target, None)
            if value is not None:
                target_func = self._get_target_func(target, inverse=inverse)
                res[target] = target_func(value, details=details)
        return res

    def _get_target_func(self, key, inverse=False):
        transform_key = f"{key}_{'inverse' if inverse else 'forward'}"
        target_function = self.targets.get(transform_key, lambda x, **p: x)
        return target_function

    @property
    def targets(self):
        return {
            "image_forward": self.image_forward,
            "mask_forward": self.mask_forward,
            "image_inverse": self.image_inverse,
            "mask_inverse": self.mask_inverse,
        }

    def load_detail(self, shared=True, target='image', **kwargs):
        detail = kwargs.get('details', None)
        assert detail is not None, 'No detail provided to inverse scale'
        img_detail = detail.get('image', None)
        msk_detail = detail.get('mask', None)

        if img_detail is None and msk_detail is None:
            raise ValueError('No detail provided to inverse scale')

        detail = img_detail if target == 'image' else msk_detail

        if not shared:
            assert detail is not None, f"No detail provided for {target}"

        if detail is None:
            logger.debug(f"Using shared detail to do inverse transform "
                         f"for{self.__class__.__name__}.")
            detail = msk_detail if target == 'image' else img_detail

        return detail

    def apply_condition(self):
        return True

    def apply_save(self, data, target, inverse=None, details=None):
        if not inverse:
            apply = self.apply_condition()
        else:
            apply = inverse

        if apply:
            if inverse:
                data = self.shared_apply(data, target, details=details)
            else:
                data = self.shared_apply(data, target, details=None)

        if inverse is None and details is not None:
            self.record_detail(target=target)
        return data

    def shared_apply(self, data, target='image', **kwargs):
        pass

    def record_detail(self, target='image'):
        pass

    def image_forward(self, image, details=None, **kwargs):
        return image

    def mask_forward(self, mask, details=None, **kwargs):
        return mask

    def image_inverse(self, image, details=None, **kwargs):
        return image

    def mask_inverse(self, mask, details=None, **kwargs):
        return mask

    def update_params(self, **kwargs):
        pass
