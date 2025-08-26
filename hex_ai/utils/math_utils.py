"""
Mathematical utility functions for the hex_ai library.
"""

import numpy as np
from typing import Union


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax over 1D logits -> probs that sum to 1.
    
    Args:
        logits: Input logits array
        
    Returns:
        Softmax probabilities that sum to 1
        
    Raises:
        ValueError: If the softmax computation fails (e.g., invalid input)
    """
    m = np.max(logits)
    exps = np.exp(logits - m)
    s = np.sum(exps)
    if s <= 0 or not np.isfinite(s):
        raise ValueError(f"Invalid softmax input: sum={s}, logits={logits}")
    return exps / s
