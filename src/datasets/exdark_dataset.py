from pathlib import Path

from .base_dataset import BaseDataset


class ExDarkDataset(BaseDataset):
    """Load unpaired ExDark images and their directory-based categories."""

    def __init__(self, root, transform=None, split_file=None):
        self.root = Path(root)
        self.transform = transform
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")

        if split_file is None:
            raise ValueError("A split file is required for ExDark")

        # Each line contains a relative path such as Bicycle/2015_00001.png.
        self.samples = []
        for relative_path in self.load_split(split_file):
            image_path = self.root / relative_path
            if not image_path.is_file():
                raise FileNotFoundError(f"ExDark image not found: {image_path}")
            self.samples.append((image_path, image_path.parent.name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, category = self.samples[index]
        image = self.load_image(image_path)
        if self.transform:
            image = self.transform(image)
        return {"image": image, "category": category, "path": str(image_path)}
