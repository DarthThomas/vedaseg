from copy import deepcopy

VALID_TARGETS = ['image', 'mask']


class BaseTransform:

    def __init__(self):
        self.valid_targets = VALID_TARGETS
        self.transform_detail = {}

    def __call__(self, **kwargs):  # image, mask, detail
        self.validate_kwargs(**kwargs)
        self.update_params(**kwargs)
        res = self.apply_to_targets(**kwargs)
        self.update_details(res)

        return res

    def validate_kwargs(self, **kwargs):
        for kwarg in kwargs:
            if kwarg.lower() in ['details', 'reverse']:
                continue
            if kwarg.lower() not in self.valid_targets:
                raise ValueError(f"Unsupported target '{kwarg}' provided, "
                                 f"currently support target "
                                 f"in {self.valid_targets}.")

    def update_params(self, **kwargs):
        reverse = kwargs.get('reverse', False)
        details = kwargs.get('details', None)
        if reverse:
            assert details is not None, "Details not provided for reverse " \
                                        "transform."
            self.reverse_params()
        else:
            self.forward_params()

    def reverse_params(self):
        pass

    def forward_params(self):
        pass

    def update_details(self, res):
        reverse = res.get('reverse', False)
        details = res.get('details', None)
        if not reverse and details is not None:
            res['details'].append(self.transform_detail)

    def apply_to_targets(self, **kwargs):
        res = deepcopy(kwargs)

        for target in self.valid_targets:
            value = kwargs.get(target, None)
            if value is not None:
                target_func = self._get_target_func(target)
                res[target] = target_func(value)
        return res

    def _get_target_func(self, key):
        transform_key = key
        target_function = self.targets.get(transform_key, lambda x, **p: x)
        return target_function

    @property
    def targets(self):
        return {
            "image": self.image_apply,
            "mask": self.mask_apply,
        }

    def mask_apply(self, value):
        return value

    def image_apply(self, value):
        return value
