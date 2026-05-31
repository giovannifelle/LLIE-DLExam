from pathlib import Path
import random


def create_train_val_split(sample_ids, val_fraction=0.15, seed=42):
    """Return deterministic train and validation IDs."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    sample_ids = sorted(sample_ids)
    if len(sample_ids) < 2:
        raise ValueError("At least two samples are required to create a split")

    shuffled_ids = sample_ids.copy()
    # A fixed seed makes the experiment reproducible on different runs.
    random.Random(seed).shuffle(shuffled_ids)
    val_size = max(1, round(len(shuffled_ids) * val_fraction))
    val_ids = sorted(shuffled_ids[:val_size])
    train_ids = sorted(shuffled_ids[val_size:])
    return train_ids, val_ids


def create_stratified_subset(samples_by_category, samples_per_category, seed=42):
    """Return a deterministic subset with the same number of samples per category."""
    if samples_per_category < 1:
        raise ValueError("samples_per_category must be positive")

    random_generator = random.Random(seed)
    selected_samples = []
    for category in sorted(samples_by_category):
        samples = sorted(samples_by_category[category])
        if len(samples) < samples_per_category:
            raise ValueError(
                f"Category {category} has only {len(samples)} samples, "
                f"but {samples_per_category} were requested"
            )
        # The same number of images is selected from every ExDark category.
        selected_samples.extend(
            f"{category}/{sample}"
            for sample in sorted(random_generator.sample(samples, samples_per_category))
        )
    return selected_samples


def save_split_file(sample_ids, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{sample_id}\n" for sample_id in sample_ids))


def load_split_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]
