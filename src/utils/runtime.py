import random

import numpy as np
import torch


def set_seed(seed, deterministic=True):
    """Set the random seeds used by Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
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
    """Create a seeded PyTorch generator for reproducible DataLoader shuffling."""
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
