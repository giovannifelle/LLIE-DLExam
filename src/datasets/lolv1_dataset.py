from pathlib import Path

from .base_dataset import BaseDataset


class LOLv1Dataset(BaseDataset):
    """Load the paired LOL-v1 eval15 cross-domain test set."""

    def __init__(self, root, transform=None):
        self.root = Path(root) / "eval15"
        self.transform = transform
        low_paths = self.list_images(self.root / "low")
        self.samples = [(path, self.root / "high" / path.name) for path in low_paths]
        for low_path, high_path in self.samples:
            if not high_path.is_file():
                raise FileNotFoundError(f"Invalid LOL-v1 pair: {low_path}, {high_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        low_path, high_path = self.samples[index]
        low_image = self.load_image(low_path)
        high_image = self.load_image(high_path)
        if self.transform:
            low_image, high_image = self.transform(low_image, high_image)
        return {"low": low_image, "high": high_image}
