from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config, load_json


def main():
    args = parse_args()
    # The experiment name decides the default history and output paths.
    config = load_config(args.config)

    experiment_name = config["experiment"]["name"]
    history_path = (
        Path(args.history)
        if args.history
        else Path("outputs") / experiment_name / "history.json"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs") / experiment_name / "history_figures"
    )

    history = load_history(history_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_loss(history, output_dir / "loss.png")
    plot_metric(history, "val_psnr", "Validation PSNR", "PSNR (dB)", output_dir / "psnr.png")
    plot_metric(history, "val_ssim", "Validation SSIM", "SSIM", output_dir / "ssim.png")
    print(f"History figures saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training history curves.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--history", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def load_history(path):
    """Load the training history saved by the training script."""
    return load_json(path)


def plot_loss(history, path):
    """Plot train and validation loss on the same figure."""
    pyplot = import_pyplot()
    epochs = get_epochs(history)
    train_loss = get_values(history, "train_loss")
    val_loss = get_values(history, "val_loss")

    pyplot.figure(figsize=(8, 5))
    pyplot.plot(epochs, train_loss, label="Train loss", linewidth=2)
    pyplot.plot(epochs, val_loss, label="Validation loss", linewidth=2)
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Loss")
    pyplot.title("Training and Validation Loss")
    pyplot.grid(True, alpha=0.3)
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.savefig(path, dpi=160)
    pyplot.close()


def plot_metric(history, key, title, ylabel, path):
    """Plot one validation metric across epochs."""
    pyplot = import_pyplot()
    epochs = get_epochs(history)
    values = get_values(history, key)

    pyplot.figure(figsize=(8, 5))
    pyplot.plot(epochs, values, linewidth=2)
    pyplot.xlabel("Epoch")
    pyplot.ylabel(ylabel)
    pyplot.title(title)
    pyplot.grid(True, alpha=0.3)
    pyplot.tight_layout()
    pyplot.savefig(path, dpi=160)
    pyplot.close()


def get_epochs(history):
    """Read epoch numbers from the saved history entries."""
    return [entry["epoch"] for entry in history]


def get_values(history, key):
    """Read a metric from every epoch and fail if it is missing."""
    missing_epochs = [entry["epoch"] for entry in history if key not in entry]
    if missing_epochs:
        raise KeyError(f"Missing {key} in epochs: {missing_epochs}")
    return [entry[key] for entry in history]


def import_pyplot():
    try:
        from matplotlib import pyplot
    except ImportError as error:
        raise ImportError(
            "visualize_history.py requires matplotlib. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from error
    return pyplot


if __name__ == "__main__":
    main()
