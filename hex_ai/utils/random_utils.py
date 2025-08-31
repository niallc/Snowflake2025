"""
Random seed utilities for deterministic behavior.
"""

import random
import numpy as np
import torch


def set_deterministic_seeds(seed: int) -> None:
    """
    Set all random seeds for deterministic behavior.
    
    Args:
        seed: Random seed to use for all random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
