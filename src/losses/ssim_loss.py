import torch
from torch import nn
from torch.nn import functional as F


class SSIMLoss(nn.Module):
    """Structural similarity loss computed with a local average window."""

    def __init__(self, window_size=11, data_range=1.0):
        super().__init__()
        self.window_size = window_size
        self.padding = window_size // 2
        self.c1 = (0.01 * data_range) ** 2
        self.c2 = (0.03 * data_range) ** 2

    def forward(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target must have the same shape")

        mean_prediction = self._local_average(prediction)
        mean_target = self._local_average(target)

        prediction_variance = self._local_average(prediction**2) - mean_prediction**2
        target_variance = self._local_average(target**2) - mean_target**2
        covariance = self._local_average(prediction * target) - mean_prediction * mean_target

        numerator = (2 * mean_prediction * mean_target + self.c1) * (
            2 * covariance + self.c2
        )
        denominator = (
            mean_prediction**2 + mean_target**2 + self.c1
        ) * (prediction_variance + target_variance + self.c2)

        ssim = numerator / denominator
        # Convert mean SSIM to a loss value (0 = identical images, 1 = maximum difference)
        return 1 - torch.clamp(ssim.mean(), 0.0, 1.0)

    def _local_average(self, image):
        """Compute the local mean of the image using average pooling over a sliding window."""
        return F.avg_pool2d(
            image,
            kernel_size=self.window_size,
            stride=1,
            padding=self.padding,
        )
