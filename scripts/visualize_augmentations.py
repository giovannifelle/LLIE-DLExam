from pathlib import Path
import argparse
import random
import sys

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import LOLv2Dataset
from src.preprocessing import build_transforms
from scripts.train import build_dataloader_generator


def main():
    args = parse_args()
    with Path(args.config).open() as config_file:
        config = yaml.safe_load(config_file)

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
        save_figure(panels, output_dir / f"augmentation_{output_index:03d}.png")

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


def set_seed(seed):
    """Make the displayed augmentations reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_figure(panels, path):
    label_height = 28
    images = [(label, tensor_to_image(tensor)) for label, tensor in panels]
    width = images[0][1].width
    height = images[0][1].height
    figure = Image.new("RGB", (width * len(images), height + label_height), "white")
    draw = ImageDraw.Draw(figure)

    for index, (label, image) in enumerate(images):
        left = index * width
        figure.paste(image, (left, label_height))
        draw.text((left + 8, 7), label, fill="black")

    figure.save(path)


def tensor_to_image(tensor):
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    return Image.fromarray(array)


if __name__ == "__main__":
    main()
