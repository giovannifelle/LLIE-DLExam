import numpy as np
import torch
from PIL import Image


def image_to_tensor(image):
    # PyTorch expects channels first and values between 0 and 1.
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1)).contiguous()


class ResizeToTensor:
    """Apply deterministic RGB conversion, resize, and [0, 1] normalization."""

    def __init__(self, image_size):
        self.size = (image_size, image_size)

    def __call__(self, image):
        image = image.convert("RGB").resize(self.size, Image.Resampling.BILINEAR)
        return image_to_tensor(image)


class PairedResizeToTensor:
    def __init__(self, image_size):
        self.transform = ResizeToTensor(image_size)

    def __call__(self, low_image, high_image):
        return self.transform(low_image), self.transform(high_image)


def build_transforms(config):
    """Build transforms directly from the project configuration."""
    from .augmentation import PairedTrainTransform

    image_size = config["data"]["image_size"]
    augmentation = config["augmentation"]

    # Validation and test transforms must always give the same result.
    val_transform = PairedResizeToTensor(image_size)
    test_transform = ResizeToTensor(image_size)

    if augmentation["enabled"]:
        train_transform = PairedTrainTransform(
            image_size=image_size,
            crop_padding=augmentation["crop_padding"],
            horizontal_flip_probability=augmentation[
                "horizontal_flip_probability"
            ],
            brightness=augmentation["brightness"],
            contrast=augmentation["contrast"],
        )
    else:
        train_transform = val_transform

    return {
        "train": train_transform,
        "val": val_transform,
        "paired_test": val_transform,
        "unpaired_test": test_transform,
    }
