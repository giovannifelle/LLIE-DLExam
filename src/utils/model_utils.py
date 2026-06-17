from pathlib import Path

import torch

from src.models import ResidualUNet, UNet


def build_model(config):
    """Build the model selected in the project configuration."""
    model_config = config["model"]
    model_name = model_config["name"]
    print(f"Using model: {model_name}")

    if model_name == "UNet":
        return UNet(
            in_channels=model_config["in_channels"],
            out_channels=model_config["out_channels"],
            base_features=model_config["base_features"],
        )

    if model_name == "ResidualUNet":
        return ResidualUNet(
            in_channels=model_config["in_channels"],
            out_channels=model_config["out_channels"],
            base_features=model_config["base_features"],
        )

    raise ValueError(f"Unsupported model: {model_name}")


def resolve_checkpoint_path(config, checkpoint_path=None):
    """Use the explicit checkpoint path or the default one for the experiment."""
    if checkpoint_path:
        return Path(checkpoint_path)
    return Path("outputs") / config["experiment"]["name"] / "best_model.pt"


def load_model(config, checkpoint_path, device):
    """Load a trained model checkpoint and switch the model to evaluation mode."""
    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
