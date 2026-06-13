from torch import nn

from .color_constancy_loss import ColorConstancyLoss
from .l1_loss import L1Loss
from .ssim_loss import SSIMLoss


class CombinedLoss(nn.Module):
    """Weighted combination of reconstruction, structure, and color losses."""

    def __init__(self, l1_weight=1.0, ssim_weight=0.2, color_weight=0.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.color_weight = color_weight
        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss()
        self.color_loss = ColorConstancyLoss()

    def forward(self, prediction, target, return_components=False):
        l1 = self.l1_loss(prediction, target)
        ssim = self.ssim_loss(prediction, target)
        total = self.l1_weight * l1 + self.ssim_weight * ssim
        components = {"l1": l1, "ssim": ssim}

        if self.color_weight > 0:
            color = self.color_loss(prediction, target)
            total = total + self.color_weight * color
            components["color"] = color

        if return_components:
            return total, components
        return total
