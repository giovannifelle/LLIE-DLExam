from pathlib import Path
import argparse
import json
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train import (
    build_dataloader_generator,
    build_model,
    seed_worker,
    select_device,
    set_seed,
)
from src.datasets import ExDarkDataset, LOLv1Dataset, LOLv2Dataset
from src.evaluation import Evaluator, MetricCalculator
from src.preprocessing import build_transforms


def main():
    args = parse_args()
    with Path(args.config).open() as config_file:
        config = yaml.safe_load(config_file)

    seed = config["experiment"]["seed"]
    set_seed(seed)
    dataloader_generator = build_dataloader_generator(seed)
    device = select_device()
    print(f"Using device: {device}")
    checkpoint_path = resolve_checkpoint_path(config, args.checkpoint)
    model = load_model(config, checkpoint_path, device)
    datasets = build_test_datasets(config)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs") / config["experiment"]["name"] / "metrics"
    )

    # Each protocol uses only the metrics that make sense for its available data.
    evaluation_protocols = {
        "lolv2_test": (datasets["lolv2_test"], ["PSNR", "SSIM", "NIQE"], True),
        "lolv1_eval15": (datasets["lolv1_eval15"], ["PSNR", "SSIM", "NIQE"], True),
        "exdark": (datasets["exdark"], ["NIQE", "BRISQUE"], False),
    }

    for name, (dataset, metric_names, is_paired) in evaluation_protocols.items():
        print(f"Dataset root: {dataset.root}")
        if hasattr(dataset, "data_dir"):
            print(f"Dataset data directory: {dataset.data_dir}")
        print(f"Evaluating {name}: {len(dataset)} images")
        dataloader = DataLoader(
            dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            worker_init_fn=seed_worker,
            generator=dataloader_generator,
        )
        evaluator = Evaluator(
            model=model,
            dataloader=dataloader,
            metric_calculator=MetricCalculator(metric_names, device=device),
            device=device,
        )
        results = (
            evaluator.evaluate_paired()
            if is_paired
            else evaluator.evaluate_unpaired()
        )
        save_metrics(results, output_dir / f"{name}.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained enhancement model.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def resolve_checkpoint_path(config, checkpoint_path):
    if checkpoint_path:
        return Path(checkpoint_path)
    return Path("outputs") / config["experiment"]["name"] / "best_model.pt"


def load_model(config, checkpoint_path, device):
    """Load the best model weights saved during training."""
    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def build_test_datasets(config):
    """Create the three datasets used to freeze the baseline results."""
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


def save_metrics(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(f"Metrics saved to {path}")


if __name__ == "__main__":
    main()
