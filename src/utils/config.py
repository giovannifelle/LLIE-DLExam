from pathlib import Path

import yaml


def load_config(path):
    """Load a YAML configuration file from disk."""
    with Path(path).open() as config_file:
        return yaml.safe_load(config_file)
