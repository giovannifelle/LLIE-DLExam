from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.split import (
    create_stratified_subset,
    create_train_val_split,
    save_split_file,
)


def main():
    with Path("configs/config.yaml").open() as config_file:
        config = yaml.safe_load(config_file)

    data_config = config["data"]
    low_dir = Path(data_config["raw_dir"]) / "LOL-v2" / "Real_captured" / "Train" / "Low"
    sample_ids = sorted(path.stem.removeprefix("low") for path in low_dir.glob("*.png"))

    # LOL-v2 does not provide a validation set, so we create one from its training images.
    train_ids, val_ids = create_train_val_split(
        sample_ids,
        val_fraction=data_config["val_fraction"],
        seed=config["experiment"]["seed"],
    )

    splits_dir = Path(data_config["splits_dir"])
    save_split_file(train_ids, splits_dir / "lolv2_real_train.txt")
    save_split_file(val_ids, splits_dir / "lolv2_real_val.txt")
    print(f"Created LOL-v2 Real Captured splits: {len(train_ids)} train, {len(val_ids)} val")

    # ExDark is large, so we keep a small balanced subset for the evaluation.
    exdark_dir = Path(data_config["raw_dir"]) / "ExDark_dataset"
    samples_by_category = {
        category_dir.name: [
            path.name
            for path in category_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        for category_dir in exdark_dir.iterdir()
        if category_dir.is_dir()
    }
    exdark_samples = create_stratified_subset(
        samples_by_category,
        samples_per_category=data_config["exdark_samples_per_category"],
        seed=config["experiment"]["seed"],
    )
    save_split_file(exdark_samples, splits_dir / "ex_dark_split.txt")
    print(
        f"Created ExDark split: {len(exdark_samples)} images "
        f"from {len(samples_by_category)} categories"
    )


if __name__ == "__main__":
    main()
