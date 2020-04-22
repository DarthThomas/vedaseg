from copy import deepcopy

VALID_TARGETS = ['image', 'mask']


class BaseTransform:

    def __init__(self):
        self.valid_targets = VALID_TARGETS
        self.transform_detail = {}

    def __call__(self, **kwargs):  # image, mask, detail
        self.validate_kwargs(**kwargs)
        self.update_params()
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

    def mask_forward(self, mask, **kwargs):
        return mask

    def image_forward(self, image, **kwargs):
        return image

    def image_inverse(self, mask, **kwargs):
        return mask

    def mask_inverse(self, image, **kwargs):
        return image

    def update_params(self):
        pass

    def reverse_params(self):
        pass

    def forward_params(self):
        pass
