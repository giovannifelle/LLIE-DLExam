from torch import nn

from .l1_loss import L1Loss
from .ssim_loss import SSIMLoss


class CombinedLoss(nn.Module):
    """Weighted combination of pixel and structural reconstruction losses."""

    def __init__(self, l1_weight=1.0, ssim_weight=0.2):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss()

    def forward(self, prediction, target, return_components=False):
        l1 = self.l1_loss(prediction, target)
        ssim = self.ssim_loss(prediction, target)
        total = self.l1_weight * l1 + self.ssim_weight * ssim

        if return_components:
            return total, {"l1": l1, "ssim": ssim}
        return total
