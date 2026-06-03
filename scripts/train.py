from pathlib import Path
import json
import random
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import LOLv2Dataset
from src.evaluation import Evaluator, MetricCalculator
from src.losses import CombinedLoss
from src.models import UNet
from src.preprocessing import build_transforms
from src.training import Trainer


def main():
    with Path("configs/config.yaml").open() as config_file:
        config = yaml.safe_load(config_file)

    seed = config["experiment"]["seed"]
    set_seed(seed)
    dataloader_generator = build_dataloader_generator(seed)
    device = select_device()
    print(f"Using device: {device}")

    data_config = config["data"]
    transforms = build_transforms(config)
    raw_dir = Path(data_config["raw_dir"])
    splits_dir = Path(data_config["splits_dir"])

    # Both datasets use the same LOL-v2 source directory but different split files.
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

    model = build_model(config)
    loss_function = CombinedLoss(
        l1_weight=config["loss"]["l1_weight"],
        ssim_weight=config["loss"]["ssim_weight"],
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


def set_seed(seed):
    """Set random seeds used by Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError as error:
        print(f"Warning: deterministic algorithms could not be enabled: {error}")


def seed_worker(worker_id):
    """Seed each DataLoader worker from PyTorch's worker-specific seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def select_device():
    """Prefer Apple Silicon, then CUDA, and finally CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(config):
    model_config = config["model"]
    if model_config["name"] != "UNet":
        raise ValueError(f"Unsupported model: {model_config['name']}")

    return UNet(
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        base_features=model_config["base_features"],
    )


def build_optimizer(config, model):
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2))
    print(f"Training history saved to {path}")


if __name__ == "__main__":
    main()
