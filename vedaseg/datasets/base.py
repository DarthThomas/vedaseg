from torch.utils.data import Dataset

from .transforms.transforms import Compose


class BaseDataset(Dataset):
    """ BaseDataset
    """

    def __init__(self):
        self.transform = None

    def process(self, **kwargs):
        if self.transform:
            transforms = Compose(self.transform)
            return transforms(**kwargs)

        return kwargs
