from pathlib import Path
import argparse
import sys

import torch
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval import build_test_datasets, load_model
from scripts.train import select_device


def main():
    args = parse_args()
    with Path(args.config).open() as config_file:
        config = yaml.safe_load(config_file)

    device = select_device()
    print(f"Using device: {device}")
    model = load_model(config, args.checkpoint, device)
    datasets = build_test_datasets(config)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs") / config["experiment"]["name"] / "figures"
    )

    save_paired_figures(
        model,
        datasets["lolv2_test"],
        output_dir / "lolv2",
        device,
        
    )
    save_paired_figures(
        model,
        datasets["lolv1_eval15"],
        output_dir / "lolv1",
        device,
        
    )
    save_unpaired_figures(
        model,
        datasets["exdark"],
        output_dir / "exdark",
        device,
        
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Save qualitative model results.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="outputs/baseline/best_model.pt")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--examples", type=int, default=None)
    return parser.parse_args()


def save_paired_figures(model, dataset, output_dir, device):
    """Save low-light input, prediction, and ground truth comparisons."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            prediction = model(sample["low"].unsqueeze(0).to(device)).squeeze(0)
            panels = [
                ("LOW INPUT", sample["low"]),
                ("PREDICTION", prediction),
                ("GROUND TRUTH", sample["high"]),
            ]
            save_figure(panels, output_dir / f"{index:03d}.png")

    print(f"Saved paired figures to {output_dir}")


def save_unpaired_figures(model, dataset, output_dir, device):
    """Save low-light input and prediction comparisons for ExDark."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            prediction = model(sample["image"].unsqueeze(0).to(device)).squeeze(0)
            panels = [
                ("LOW INPUT", sample["image"]),
                ("PREDICTION", prediction),
            ]
            category = sample["category"].lower()
            save_figure(panels, output_dir / f"{index:03d}_{category}.png")

    print(f"Saved unpaired figures to {output_dir}")


def save_figure(panels, path):
    """Create one labeled horizontal image that is easy to include in the report."""
    images = [(label, tensor_to_image(tensor)) for label, tensor in panels]
    label_height = 28
    panel_width = images[0][1].width
    panel_height = images[0][1].height
    figure = Image.new(
        "RGB",
        (panel_width * len(images), panel_height + label_height),
        color="white",
    )
    draw = ImageDraw.Draw(figure)

    for index, (label, image) in enumerate(images):
        left = index * panel_width
        figure.paste(image, (left, label_height))
        draw.text((left + 8, 7), label, fill="black")

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.save(path)


def tensor_to_image(tensor):
    """Convert a normalized PyTorch tensor back to a Pillow RGB image."""
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    return Image.fromarray(array)


if __name__ == "__main__":
    main()
