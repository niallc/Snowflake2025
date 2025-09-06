"""
Mathematical utility functions for the hex_ai library.
"""

import numpy as np
from typing import Union, List, Tuple


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


def policy_logits_to_legal_probs(policy_logits: np.ndarray, legal_moves: List[Tuple[int, int]], 
                                board_size: int) -> np.ndarray:
    """
    Convert policy logits to probabilities properly masked to legal moves only.
    
    This function ensures that probabilities sum to 1.0 over legal moves only by:
    1. Masking illegal moves to -inf
    2. Applying softmax to the masked logits
    3. Extracting probabilities for legal moves
    
    Args:
        policy_logits: Raw policy logits [K] where K = board_size * board_size
        legal_moves: List of (row, col) tuples for legal moves
        board_size: Size of the board (e.g., 13 for 13x13 Hex)
        
    Returns:
        Probabilities for legal moves only, summing to 1.0
        
    Raises:
        ValueError: If no legal moves provided or invalid inputs
    """
    if not legal_moves:
        raise ValueError("No legal moves provided")
    
    if len(policy_logits) != board_size * board_size:
        raise ValueError(f"Policy logits length {len(policy_logits)} doesn't match board size {board_size}x{board_size}")
    
    # Create legal indices for masking
    legal_indices = [row * board_size + col for row, col in legal_moves]
    
    # Mask illegal moves by setting their logits to -inf
    masked_logits = policy_logits.copy()
    legal_mask = np.zeros(len(masked_logits), dtype=bool)
    legal_mask[legal_indices] = True
    masked_logits[~legal_mask] = -np.inf
    
    # Apply softmax to masked logits to get proper probabilities over legal moves only
    policy_probs = softmax_np(masked_logits)
    
    # Extract probabilities for legal moves
    legal_probs = np.array([policy_probs[idx] for idx in legal_indices])
    
    return legal_probs
