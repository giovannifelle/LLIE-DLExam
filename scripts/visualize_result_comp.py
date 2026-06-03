from pathlib import Path
import argparse
import copy
import sys

import torch
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval import build_test_datasets, load_model
from scripts.train import select_device
from scripts.visualize_results import tensor_to_image


MODEL_NAME_BY_RUN = {
    "baseline": "UNet",
    "residual_unet": "ResidualUNet",
}


def main():
    args = parse_args()
    with Path(args.config).open() as config_file:
        base_config = yaml.safe_load(config_file)

    device = select_device()
    print(f"Using device: {device}")

    runs = discover_runs(Path(args.outputs_dir), args.runs)
    models = load_run_models(base_config, runs, device)
    datasets = build_test_datasets(base_config)
    output_dir = Path(args.output_dir)

    save_paired_comparisons(
        models=models,
        dataset=datasets["lolv2_test"],
        output_dir=output_dir / "lolv2",
        device=device,
        max_examples=args.examples,
    )
    save_paired_comparisons(
        models=models,
        dataset=datasets["lolv1_eval15"],
        output_dir=output_dir / "lolv1",
        device=device,
        max_examples=args.examples,
    )
    save_unpaired_comparisons(
        models=models,
        dataset=datasets["exdark"],
        output_dir=output_dir / "exdark",
        device=device,
        max_examples=args.examples,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save side-by-side qualitative comparisons for all output runs."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument(
        "--output-dir",
        default="outputs/comparisons/figures",
        help="Directory where comparison figures are saved.",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Optional run names to compare. Defaults to every run with best_model.pt.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=None,
        help="Optional maximum number of examples per dataset.",
    )
    return parser.parse_args()


def discover_runs(outputs_dir, requested_runs=None):
    if requested_runs:
        runs = [outputs_dir / run_name for run_name in requested_runs]
    else:
        runs = sorted(
            path
            for path in outputs_dir.iterdir()
            if path.is_dir() and (path / "best_model.pt").is_file()
        )

    missing = [path.name for path in runs if not (path / "best_model.pt").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing best_model.pt for runs: {missing}")
    if not runs:
        raise ValueError(f"No runs with best_model.pt found in {outputs_dir}")
    return runs


def load_run_models(base_config, runs, device):
    models = []
    for run_dir in runs:
        run_config = build_run_config(base_config, run_dir)
        checkpoint_path = run_dir / "best_model.pt"
        model = load_model(run_config, checkpoint_path, device)
        models.append((format_run_label(run_dir.name), model))
        print(
            "Loaded "
            f"{run_dir.name} as {run_config['model']['name']} "
            f"from {checkpoint_path}"
        )
    return models


def build_run_config(base_config, run_dir):
    config = copy.deepcopy(base_config)
    config["experiment"]["name"] = run_dir.name
    config["model"]["name"] = infer_model_name(run_dir.name, config["model"]["name"])
    return config


def infer_model_name(run_name, fallback_model_name):
    normalized_run_name = run_name.lower()
    if normalized_run_name in MODEL_NAME_BY_RUN:
        return MODEL_NAME_BY_RUN[normalized_run_name]
    if "residual" in normalized_run_name:
        return "ResidualUNet"
    if "baseline" in normalized_run_name:
        return "UNet"
    return fallback_model_name


def format_run_label(run_name):
    return run_name.replace("_", " ").title()


def save_paired_comparisons(models, dataset, output_dir, device, max_examples=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    limit = resolve_limit(dataset, max_examples)

    with torch.inference_mode():
        for index in range(limit):
            sample = dataset[index]
            panels = [("LOW INPUT", sample["low"])]
            for label, model in models:
                model.eval()
                prediction = model(sample["low"].unsqueeze(0).to(device)).squeeze(0)
                panels.append((label, prediction))
            panels.append(("GROUND TRUTH", sample["high"]))
            save_figure(panels, output_dir / f"{index:03d}.png")

    print(f"Saved paired comparison figures to {output_dir}")


def save_unpaired_comparisons(models, dataset, output_dir, device, max_examples=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    limit = resolve_limit(dataset, max_examples)

    with torch.inference_mode():
        for index in range(limit):
            sample = dataset[index]
            panels = [("LOW INPUT", sample["image"])]
            for label, model in models:
                model.eval()
                prediction = model(sample["image"].unsqueeze(0).to(device)).squeeze(0)
                panels.append((label, prediction))
            category = sample["category"].lower()
            save_figure(panels, output_dir / f"{index:03d}_{category}.png")

    print(f"Saved unpaired comparison figures to {output_dir}")


def resolve_limit(dataset, max_examples):
    if max_examples is None:
        return len(dataset)
    return min(len(dataset), max_examples)


def save_figure(panels, path):
    images = [(label, tensor_to_image(tensor)) for label, tensor in panels]
    label_height = 30
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
        draw.text((left + 8, 8), label, fill="black")

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.save(path)


if __name__ == "__main__":
    main()
