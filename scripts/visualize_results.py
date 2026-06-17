from pathlib import Path
import argparse
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import (
    build_test_datasets,
    load_config,
    load_model,
    resolve_checkpoint_path,
    save_horizontal_panel,
    select_device,
)


def main():
    args = parse_args()
    # Qualitative figures use the same checkpoint and datasets as quantitative evaluation.
    config = load_config(args.config)

    device = select_device()
    print(f"Using device: {device}")
    checkpoint_path = resolve_checkpoint_path(config, args.checkpoint)
    model = load_model(config, checkpoint_path, device)
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
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def save_paired_figures(model, dataset, output_dir, device):
    """Save low-light input, prediction, and ground truth comparisons."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    # Paired datasets allow direct visual comparison with the ground truth.
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            prediction = model(sample["low"].unsqueeze(0).to(device)).squeeze(0)
            panels = [
                ("LOW INPUT", sample["low"]),
                ("PREDICTION", prediction),
                ("GROUND TRUTH", sample["high"]),
            ]
            save_horizontal_panel(panels, output_dir / f"{index:03d}.png")

    print(f"Saved paired figures to {output_dir}")


def save_unpaired_figures(model, dataset, output_dir, device):
    """Save low-light input and prediction comparisons for ExDark."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    # ExDark has no target image, so only input and prediction are shown.
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            prediction = model(sample["image"].unsqueeze(0).to(device)).squeeze(0)
            panels = [
                ("LOW INPUT", sample["image"]),
                ("PREDICTION", prediction),
            ]
            category = sample["category"].lower()
            save_horizontal_panel(panels, output_dir / f"{index:03d}_{category}.png")

    print(f"Saved unpaired figures to {output_dir}")


if __name__ == "__main__":
    main()
