import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy

from .registry import CRITERIA


@CRITERIA.register_module
class BalanceCrossEntropyLoss(nn.Module):
    """
    Balanced cross entropy loss.
    Shape:
        - Input: :math:`(N, 1, H, W)`
        - GT: :math:`(N, 1, H, W)`, same shape as the input
        - Mask: :math:`(N, H, W)`, same spatial shape as the input
        - Output: scalar.

    Examples::

        >>> m = nn.Sigmoid()
        >>> loss = nn.BCELoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(m(input), target)
        >>> output.backward()
    """

    def __init__(self, negative_ratio=3.0, eps=1e-6, negative_override=0.1,
                 ignore_index=255):
        super(BalanceCrossEntropyLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps
        self.negative_override = negative_override
        self.ignore_index = ignore_index

    def forward(self,
                pred: torch.Tensor,
                gt: torch.Tensor):
        """
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, 1, H, W)`, the target
        """
        valid_tensor = torch.ones_like(gt)
        valid_tensor[gt == self.ignore_index] = 0

        prob = pred.softmax(dim=1)
        _, mask = torch.max(prob, dim=1)

        positive = (valid_tensor * gt * mask).byte()
        negative = (valid_tensor * (1 - gt) * mask).byte()

        positive_count = int(positive.float().sum())
        negative_count = int(negative.float().sum() * self.negative_override)
        if positive_count >= 1:
            negative_count = min(int(negative.float().sum()),
                                 int(positive_count * self.negative_ratio))

        loss = binary_cross_entropy(prob[:, 1, :, :],
                                    gt,
                                    reduction='none')[:, 0, :, :]
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / \
                       (positive_count + negative_count + self.eps)
        return balance_loss
