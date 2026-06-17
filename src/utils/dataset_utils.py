from pathlib import Path

from src.datasets import ExDarkDataset, LOLv1Dataset, LOLv2Dataset
from src.preprocessing import build_transforms


def build_test_datasets(config):
    """Create the datasets used for final paired and unpaired evaluation."""
    data_config = config["data"]
    raw_dir = Path(data_config["raw_dir"])
    splits_dir = Path(data_config["splits_dir"])
    transforms = build_transforms(config)

    return {
        "lolv2_test": LOLv2Dataset(
            root=raw_dir / "LOL-v2",
            split="test",
            transform=transforms["paired_test"],
        ),
        "lolv1_eval15": LOLv1Dataset(
            root=raw_dir / "lol_dataset",
            transform=transforms["paired_test"],
        ),
        "exdark": ExDarkDataset(
            root=raw_dir / "ExDark_dataset",
            transform=transforms["unpaired_test"],
            split_file=splits_dir / "ex_dark_split.txt",
        ),
    }
