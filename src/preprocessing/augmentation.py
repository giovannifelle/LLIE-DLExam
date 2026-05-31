import random

from PIL import Image, ImageEnhance, ImageOps

from .transforms import image_to_tensor


class PairedTrainTransform:
    """Training augmentation that preserves alignment between paired images."""

    def __init__(
        self,
        image_size,
        crop_padding=32,
        horizontal_flip_probability=0.5,
        brightness=0.1,
        contrast=0.1,
    ):
        self.image_size = image_size
        self.resize_size = image_size + crop_padding
        self.flip_probability = horizontal_flip_probability
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, low_image, high_image):
        size = (self.resize_size, self.resize_size)
        low_image = low_image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
        high_image = high_image.convert("RGB").resize(size, Image.Resampling.BILINEAR)

        # Both images use the same crop because they represent the same scene.
        max_offset = self.resize_size - self.image_size
        left = random.randint(0, max_offset)
        top = random.randint(0, max_offset)
        box = (left, top, left + self.image_size, top + self.image_size)
        low_image = low_image.crop(box)
        high_image = high_image.crop(box)

        if random.random() < self.flip_probability:
            low_image = ImageOps.mirror(low_image)
            high_image = ImageOps.mirror(high_image)

        # Color changes are applied only to the input, not to the ground truth.
        low_image = self._jitter_low_light_image(low_image)
        return image_to_tensor(low_image), image_to_tensor(high_image)

    def _jitter_low_light_image(self, image):
        if self.brightness:
            factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if self.contrast:
            factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            image = ImageEnhance.Contrast(image).enhance(factor)
        return image
