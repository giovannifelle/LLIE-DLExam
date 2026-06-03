import torch
from torch.nn import functional as F


def calculate_psnr(prediction, target, data_range=1.0):
    """Calculate the Peak Signal-to-Noise Ratio for a batch of images."""
    _check_matching_shapes(prediction, target)
    mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))

    # PSNR is computed per image before averaging the batch.
    # Identical images have an infinite value because their MSE is zero.
    psnr = torch.where(
        mse == 0,
        torch.tensor(float("inf"), device=mse.device),
        10 * torch.log10(data_range**2 / mse),
    )
    return psnr.mean()


def calculate_ssim(prediction, target, window_size=11, data_range=1.0):
    """Calculate the Structural Similarity Index for a batch of images."""
    _check_matching_shapes(prediction, target)
    padding = window_size // 2
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    # Local statistics compare image structure inside a small moving window.
    mean_prediction = _local_average(prediction, window_size, padding)
    mean_target = _local_average(target, window_size, padding)
    prediction_variance = (
        _local_average(prediction**2, window_size, padding) - mean_prediction**2
    )
    target_variance = _local_average(target**2, window_size, padding) - mean_target**2
    covariance = (
        _local_average(prediction * target, window_size, padding)
        - mean_prediction * mean_target
    )

    numerator = (2 * mean_prediction * mean_target + c1) * (
        2 * covariance + c2
    )
    denominator = (
        mean_prediction**2 + mean_target**2 + c1
    ) * (prediction_variance + target_variance + c2)

    ssim_per_image = (numerator / denominator).mean(dim=(1, 2, 3))
    return torch.clamp(ssim_per_image, 0.0, 1.0).mean()


class MetricCalculator:
    """Calculate the metrics selected in the project configuration."""

    FULL_REFERENCE_METRICS = {"PSNR", "SSIM"}
    NO_REFERENCE_METRICS = {"NIQE", "BRISQUE"}

    def __init__(self, metric_names, device="cpu"):
        self.metric_names = [name.upper() for name in metric_names]
        self.device = torch.device(device)
        self.no_reference_device = self._select_no_reference_device()
        self._no_reference_metrics = {}

        supported_metrics = self.FULL_REFERENCE_METRICS | self.NO_REFERENCE_METRICS
        unknown_metrics = set(self.metric_names) - supported_metrics
        if unknown_metrics:
            raise ValueError(f"Unsupported metrics: {sorted(unknown_metrics)}")

    @classmethod
    def from_config(cls, config, device="cpu"):
        return cls(config["evaluation"]["metrics"], device=device)

    def calculate(self, prediction, target=None, input_image=None):
        """Return available scores for a batch of enhanced images."""
        prediction = prediction.to(self.device)
        target = target.to(self.device) if target is not None else None
        input_image = input_image.to(self.device) if input_image is not None else None
        scores = {}

        # Paired datasets use PSNR and SSIM when a target is available.
        # No-reference metrics can also be computed on paired test datasets.
        for metric_name in self.metric_names:
            if metric_name == "PSNR" and target is not None:
                scores["PSNR"] = calculate_psnr(prediction, target).item()
            elif metric_name == "SSIM" and target is not None:
                scores["SSIM"] = calculate_ssim(prediction, target).item()
            elif metric_name in self.NO_REFERENCE_METRICS:
                metric = self._get_no_reference_metric(metric_name)
                prediction_score = metric(
                    prediction.to(self.no_reference_device)
                ).mean()
                scores[metric_name] = prediction_score.item()

                if input_image is not None:
                    input_score = metric(
                        input_image.to(self.no_reference_device)
                    ).mean()
                    scores[f"{metric_name}_input"] = input_score.item()
                    scores[f"{metric_name}_delta"] = (
                        prediction_score - input_score
                    ).item()

        return scores

    def _get_no_reference_metric(self, metric_name):
        # NIQE and BRISQUE are loaded only for unpaired evaluation.
        # This avoids loading their external models during training validation.
        if metric_name not in self._no_reference_metrics:
            try:
                import pyiqa
            except ImportError as error:
                raise ImportError(
                    "NIQE and BRISQUE require the optional pyiqa dependency. "
                    "Install it with: pip install pyiqa"
                ) from error

            # pyiqa expects RGB tensors with values in the [0, 1] range.
            self._no_reference_metrics[metric_name] = pyiqa.create_metric(
                metric_name.lower(),
                device=self.no_reference_device,
            )
        return self._no_reference_metrics[metric_name]

    def _select_no_reference_device(self):
        # NIQE uses float64 internally, which is not supported by Apple MPS.
        # The model can still run on MPS while these metrics are computed on CPU.
        if self.device.type == "mps":
            return torch.device("cpu")
        return self.device


def _check_matching_shapes(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target must have the same shape")
    if prediction.ndim != 4:
        raise ValueError("Metrics expect tensors with shape (batch, channels, height, width)")


def _local_average(image, window_size, padding):
    return F.avg_pool2d(
        image,
        kernel_size=window_size,
        stride=1,
        padding=padding,
    )
