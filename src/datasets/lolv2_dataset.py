from pathlib import Path

from .base_dataset import BaseDataset


class LOLv2Dataset(BaseDataset):
    """Load paired images from LOL-v2 Real Captured."""

    def __init__(self, root, split, transform=None, split_file=None):
        self.root = Path(root) / "Real_captured"
        self.split = split
        self.transform = transform

        if split in {"train", "val"}:
            # Train and validation samples come from the custom split files.
            if split_file is None:
                raise ValueError(f"A split file is required for LOL-v2 {split}")
            self.sample_ids = self.load_split(split_file)
            self.data_dir = self.root / "Train"
        elif split == "test":
            # The official test set is used completely.
            low_paths = self.list_images(self.root / "Test" / "Low")
            self.sample_ids = [path.stem.removeprefix("low") for path in low_paths]
            self.data_dir = self.root / "Test"
            print(f"LOL-v2 test low paths: {len(low_paths)}")
            print(f"LOL-v2 test sample IDs: {len(self.sample_ids)}")
        else:
            raise ValueError(f"Unsupported LOL-v2 split: {split}")

        self.samples = [
            (
                self.data_dir / "Low" / f"low{sample_id}.png",
                self.data_dir / "Normal" / f"normal{sample_id}.png",
            )
            for sample_id in self.sample_ids
        ]
        if split == "test":
            print(f"LOL-v2 test pairs: {len(self.samples)}")
        self._validate_pairs()

    def _validate_pairs(self):
        for low_path, high_path in self.samples:
            if not low_path.is_file() or not high_path.is_file():
                raise FileNotFoundError(f"Invalid LOL-v2 pair: {low_path}, {high_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        low_path, high_path = self.samples[index]
        low_image = self.load_image(low_path)
        high_image = self.load_image(high_path)
        if self.transform:
            low_image, high_image = self.transform(low_image, high_image)
        return {"low": low_image, "high": high_image}
