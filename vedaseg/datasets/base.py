from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """ BaseDataset
    """

    def __init__(self):
        self.transform = None

    def process(self, **kwargs):
        res = None
        if self.transform:
            res = self.transform(**kwargs)

        return res
