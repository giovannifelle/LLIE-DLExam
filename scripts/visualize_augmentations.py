from pathlib import Path
import argparse
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import LOLv2Dataset
from src.preprocessing import build_transforms
from src.utils import (
    build_dataloader_generator,
    load_config,
    save_horizontal_panel,
    set_seed,
)


def main():
    args = parse_args()
    # This script shows the exact augmentation pipeline used during training.
    config = load_config(args.config)

    # Fixing the seed makes the selected examples and random transforms repeatable.
    set_seed(config["experiment"]["seed"])
    data_config = config["data"]
    raw_dir = Path(data_config["raw_dir"])
    splits_dir = Path(data_config["splits_dir"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transforms = build_transforms(config)
    generator = build_dataloader_generator(config["experiment"]["seed"])

    # Same train split and same train transform used in scripts/train.py.
    augmented_dataset = LOLv2Dataset(
        root=raw_dir / "LOL-v2",
        split="train",
        transform=transforms["train"],
        split_file=splits_dir / "lolv2_real_train.txt",
    )

    # This is only for comparison in the figure; it is not used during training.
    original_dataset = LOLv2Dataset(
        root=raw_dir / "LOL-v2",
        split="train",
        transform=transforms["val"],
        split_file=splits_dir / "lolv2_real_train.txt",
    )

    example_count = min(args.examples, len(original_dataset))

    # The examples are shuffled deterministically so the figure set is reproducible.
    shuffled_indices = torch.randperm(len(augmented_dataset), generator=generator)

    for output_index, dataset_index in enumerate(shuffled_indices[:example_count].tolist()):
        # Calling augmented_dataset[index] applies PairedTrainTransform exactly as training.
        original = original_dataset[dataset_index]
        augmented = augmented_dataset[dataset_index]
        panels = [
            ("VAL TRANSFORM LOW", original["low"]),
            ("TRAIN AUGMENTED LOW", augmented["low"]),
            ("VAL TRANSFORM GT", original["high"]),
            ("TRAIN AUGMENTED GT", augmented["high"]),
        ]
        save_horizontal_panel(panels, output_dir / f"augmentation_{output_index:03d}.png")

    print(f"Saved {example_count} augmentation examples to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Save examples of training augmentations.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default="data/processed/augmentation_examples",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
