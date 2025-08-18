import os
import random
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set seeds for reproducibility across numpy, random, and torch.

    Note: Perfect reproducibility is not guaranteed on all backends.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Apple MPS backend
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.manual_seed(seed)

