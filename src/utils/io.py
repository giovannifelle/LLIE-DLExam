import json
from pathlib import Path


def save_json(data, path):
    """Save a Python object as an indented JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def load_json(path):
    """Load a JSON file and return its decoded content."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text())
