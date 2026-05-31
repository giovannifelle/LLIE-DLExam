from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from src.preprocessing.split import load_split_file


class BaseDataset(Dataset):
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    @staticmethod
    def load_image(path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        with Image.open(path) as image:
            return image.convert("RGB")

    @classmethod
    def list_images(cls, directory):
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {directory}")
        return sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in cls.IMAGE_EXTENSIONS
        )

    @staticmethod
    def load_split(path):
        return load_split_file(path)
