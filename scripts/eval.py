from pathlib import Path
import argparse
import sys

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import Evaluator, MetricCalculator
from src.utils import (
    build_dataloader_generator,
    build_test_datasets,
    load_config,
    load_model,
    resolve_checkpoint_path,
    save_json,
    seed_worker,
    select_device,
    set_seed,
)


def main():
    args = parse_args()
    # Evaluation can reuse the default config or a config passed from the command line.
    config = load_config(args.config)

    # The same seed/device utilities are used here to keep execution consistent.
    seed = config["experiment"]["seed"]
    set_seed(seed)
    dataloader_generator = build_dataloader_generator(seed)
    device = select_device()
    print(f"Using device: {device}")
    checkpoint_path = resolve_checkpoint_path(config, args.checkpoint)
    model = load_model(config, checkpoint_path, device)

    # These are the three final datasets used in the report.
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
        # A fresh evaluator is created because each dataset uses different metrics.
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


def save_metrics(results, path):
    """Save one JSON file for each evaluated dataset."""
    save_json(results, path)
    print(f"Metrics saved to {path}")


if __name__ == "__main__":
    main()
