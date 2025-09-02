"""
Mathematical utility functions for the hex_ai library.
"""

import numpy as np
from typing import Union


def validate_probabilities(probs: np.ndarray, context: str = "") -> None:
    """
    Validate that an array contains valid probabilities.
    
    Args:
        probs: Array to validate
        context: Optional context string for error messages
        
    Raises:
        ValueError: If probabilities are invalid
    """
    if not np.all(np.isfinite(probs)):
        raise ValueError(f"{context} Probabilities contain non-finite values: {probs}")
    
    if not np.all((probs >= -1e-9) & (probs <= 1 + 1e-9)):
        raise ValueError(f"{context} Probabilities contain values outside [0,1]: min={probs.min():.6f}, max={probs.max():.6f}")
    
    if not np.isclose(np.sum(probs), 1.0, atol=1e-6):
        raise ValueError(f"{context} Probabilities don't sum to 1: sum={np.sum(probs):.6f}")


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
    
    probs = exps / s
    
    # Validate the output probabilities
    validate_probabilities(probs, "Softmax output")
    
    return probs
