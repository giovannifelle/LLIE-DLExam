from torch import nn


class L1Loss(nn.Module):
    """Pixel reconstruction loss for image enhancement."""

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, prediction, target):
        return self.loss(prediction, target)
