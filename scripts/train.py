from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import LOLv2Dataset
from src.evaluation import Evaluator, MetricCalculator
from src.losses import CombinedLoss
from src.preprocessing import build_transforms
from src.training import Trainer
from src.utils import (
    build_dataloader_generator,
    build_model,
    load_config,
    save_json,
    seed_worker,
    select_device,
    set_seed,
)


def main():
    # The training script is driven by the central YAML configuration.
    config = load_config("configs/config.yaml")

    # Fix all random sources before creating datasets, loaders, and the model.
    seed = config["experiment"]["seed"]
    set_seed(seed)
    dataloader_generator = build_dataloader_generator(seed)
    device = select_device()
    print(f"Using device: {device}")

    data_config = config["data"]
    transforms = build_transforms(config)
    raw_dir = Path(data_config["raw_dir"])
    splits_dir = Path(data_config["splits_dir"])

    # Training and validation both come from LOL-v2, but use different split files.
    train_dataset = LOLv2Dataset(
        root=raw_dir / "LOL-v2",
        split="train",
        transform=transforms["train"],
        split_file=splits_dir / "lolv2_real_train.txt",
    )
    val_dataset = LOLv2Dataset(
        root=raw_dir / "LOL-v2",
        split="val",
        transform=transforms["val"],
        split_file=splits_dir / "lolv2_real_val.txt",
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
        worker_init_fn=seed_worker,
        generator=dataloader_generator,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
        worker_init_fn=seed_worker,
        generator=dataloader_generator,
    )
    print(f"Dataset sizes: {len(train_dataset)} train, {len(val_dataset)} val")

    # Model, loss, and optimizer are built from the same config for every run.
    model = build_model(config)
    loss_function = CombinedLoss(
        l1_weight=config["loss"]["l1_weight"],
        ssim_weight=config["loss"]["ssim_weight"],
        color_weight=config["loss"].get("color_weight", 0.0),
    )
    optimizer = build_optimizer(config, model)

    # Validation only needs paired metrics. NIQE and BRISQUE are computed after training.
    metric_calculator = MetricCalculator(["PSNR", "SSIM"], device=device)
    evaluator = Evaluator(
        model=model,
        dataloader=val_dataloader,
        metric_calculator=metric_calculator,
        device=device,
        loss_function=loss_function,
    )

    # The Trainer handles epochs, checkpoints, and early stopping.
    output_dir = Path("outputs") / config["experiment"]["name"]
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        loss_function=loss_function,
        evaluator=evaluator,
        device=device,
        epochs=config["training"]["epochs"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
        checkpoint_path=output_dir / "best_model.pt",
        mixed_precision=config["training"]["mixed_precision"],
    )
    history = trainer.train()
    save_history(history, output_dir / "history.json")


def build_optimizer(config, model):
    """Build the optimizer used by the training loop."""
    training_config = config["training"]
    if training_config["optimizer"] != "AdamW":
        raise ValueError(f"Unsupported optimizer: {training_config['optimizer']}")

    return torch.optim.AdamW(
        model.parameters(),
        lr=training_config["lr"],
        weight_decay=training_config["weight_decay"],
    )


def save_history(history, path):
    """Save validation results to make the experiment easier to reproduce."""
    save_json(history, path)
    print(f"Training history saved to {path}")


if __name__ == "__main__":
    main()
