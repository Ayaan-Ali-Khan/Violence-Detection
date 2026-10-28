import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)

def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy inside a PyTorch DataLoader worker.

    Pass this function as ``worker_init_fn`` to a DataLoader. The worker's seed
    is derived from PyTorch's worker-specific initial seed.
    """
    del worker_id  # PyTorch passes the ID; initial_seed already varies by worker.
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_generator(seed: int = 42) -> torch.Generator:
    """Create a seeded generator for reproducible DataLoader sampling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator