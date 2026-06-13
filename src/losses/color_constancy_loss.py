import torch
from torch import nn


class ColorConstancyLoss(nn.Module):
    """Match global per-image RGB channel statistics."""

    def forward(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target must have the same shape")
        if prediction.ndim != 4:
            raise ValueError(
                "ColorConstancyLoss expects tensors with shape "
                "(batch, channels, height, width)"
            )
        if prediction.shape[1] != 3:
            raise ValueError("ColorConstancyLoss expects RGB tensors with 3 channels")

        prediction_mean = prediction.mean(dim=(2, 3))
        target_mean = target.mean(dim=(2, 3))
        return torch.mean(torch.abs(prediction_mean - target_mean))
