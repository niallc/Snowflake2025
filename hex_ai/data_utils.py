"""
Data processing utilities for Hex AI.

This module contains functions for converting between different Hex data formats,
applying data augmentation, and preparing data for training. It provides a modern,
typed interface to the legacy conversion functions.

Key functions:
- File format conversion (trmph ↔ tensor)
- Data augmentation (rotation, reflection)
- Training data preparation
- Coordinate system conversions
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import logging
import re
import string

from .config import BOARD_SIZE, NUM_PLAYERS, TRMPH_EXTENSION, POLICY_OUTPUT_SIZE
from hex_ai.utils.format_conversion import (
    strip_trmph_preamble, split_trmph_moves, trmph_move_to_rowcol, parse_trmph_to_board,
    rowcol_to_trmph, tensor_to_rowcol, rowcol_to_tensor, tensor_to_trmph, trmph_to_tensor
)

logger = logging.getLogger(__name__)

# The trmph URL is:
# https://trmph.com/hex/board#13,a1b2c3
# We match up to the number (13) and then the comma.
TRMPH_BOARD_PATTERN = re.compile(r"#(\d+),")
LETTERS = string.ascii_lowercase


# ============================================================================
# File Format Conversion Functions
# ============================================================================

def load_trmph_file(file_path: str) -> List[str]:
    """
    Load a single .trmph file.
    
    Args:
        file_path: Path to the .trmph file
        
    Returns:
        List of trmph game strings (one per line)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Strip whitespace and filter out empty lines
        games = [line.strip() for line in lines if line.strip()]
        
        if not games:
            raise ValueError(f"Empty file: {file_path}")
        
        return games
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading file {file_path}: {e}")


def display_board(board: np.ndarray, format_type: str = "matrix") -> str:
    """
    Display a board in a human-readable format.
    
    Args:
        board: Board array (either 2-channel or single-channel)
        format_type: "matrix" or "visual"
        
    Returns:
        String representation of the board
    """
    if board.ndim == 3:
        # 2-channel format: convert to single channel
        board_2d = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        board_2d[board[0] == 1] = 1  # Blue pieces
        board_2d[board[1] == 1] = 2  # Red pieces
    else:
        board_2d = board
    
    if format_type == "matrix":
        return str(board_2d)
    
    elif format_type == "visual":
        # Create a visual representation
        symbols = {0: ".", 1: "B", 2: "R"}
        lines = []
        for row in range(BOARD_SIZE):
            line = " " * row  # Indent for hex shape
            for col in range(BOARD_SIZE):
                line += symbols[board_2d[row, col]] + " "
            lines.append(line)
        return "\n".join(lines)
    
    else:
        raise ValueError(f"Unknown format_type: {format_type}")


def strip_trmph_preamble(trmph_text: str) -> str:
    """
    Remove the preamble from a trmph string (e.g., 'http://...#13,a1b2c3' -> 'a1b2c3').
    """
    # Warning, though a trmph URL contains just the preamble and then moves
    # our data files also contain a space and then an integer 1 or 2, to indicate the winner.
    # This function strips the preamble to start with the move sequence and .search returns
    # the index of the comma after the board size.
    # When we return trmph_text[match.end():] we get the move sequence.
    # if called on a full line of input data, this would then include the space 
    # and the winner annotation.
    # More typically we will call this function on a single move sequence.
    match = TRMPH_BOARD_PATTERN.search(trmph_text)
    if not match:
        raise ValueError(f"No board preamble found in trmph string: {trmph_text}")
    return trmph_text[match.end():]


def split_trmph_moves(bare_moves: str) -> list[str]:
    """
    Split a bare trmph move string into a list of moves (e.g., 'a1b2c3' -> ['a1','b2','c3']).
    """
    moves = []
    i = 0
    while i < len(bare_moves):
        if bare_moves[i] not in LETTERS:
            raise ValueError(f"Expected letter at position {i} in {bare_moves}")
        j = i + 1
        while j < len(bare_moves) and bare_moves[j].isdigit():
            j += 1
        moves.append(bare_moves[i:j])
        i = j
    return moves


def trmph_move_to_rowcol(move: str, board_size: int = BOARD_SIZE) -> tuple[int, int]:
    """
    Convert a trmph move (e.g., 'a1') to (row, col) coordinates (0-indexed).
    """
    if len(move) < 2 or len(move) > 4:
        raise ValueError(f"Invalid trmph move: {move}")
    letter = move[0]
    number = int(move[1:])
    if letter not in LETTERS[:board_size]:
        raise ValueError(f"Invalid letter in move: {move}")
    if not (1 <= number <= board_size):
        raise ValueError(f"Invalid number in move: {move}")
    row = number - 1
    col = LETTERS.index(letter)
    return row, col


def parse_trmph_to_board(trmph_text: str, board_size: int = BOARD_SIZE, debug_info: str = "") -> np.ndarray:
    """
    Parse a trmph string to a board matrix.
    
    Args:
        trmph_text: Complete trmph string
        board_size: Size of the board
        debug_info: Optional debug information (e.g., line number)
        
    Returns:
        Board matrix with 0=empty, 1=blue, 2=red
    """
    # Strip preamble and get moves
    bare_moves = strip_trmph_preamble(trmph_text)
    moves = split_trmph_moves(bare_moves)
    
    # Initialize board
    board = np.zeros((board_size, board_size), dtype=np.int8)
    
    # Place moves on board
    for i, move in enumerate(moves):
        row, col = trmph_move_to_rowcol(move, board_size)
        
        # Check for duplicate moves
        if board[row, col] != 0:
            # For training data, skip duplicate moves instead of failing
            if "training" in debug_info.lower() or "game" in debug_info.lower():
                logger.warning(f"Skipping duplicate move '{move}' at {(row, col)} in training data")
                continue  # Skip this move and continue with the next
            else:
                # Enhanced debugging output for non-training contexts
                import traceback
                frame = traceback.extract_stack()[-2]  # Get calling frame
                logger.error(f"DUPLICATE MOVE DETECTED:")
                if debug_info:
                    logger.error(f"  {debug_info}")
                logger.error(f"  File: {frame.filename}")
                logger.error(f"  Line: {frame.lineno}")
                logger.error(f"  Function: {frame.name}")
                logger.error(f"  Move: '{move}' at position ({row}, {col})")
                logger.error(f"  Move index: {i}")
                logger.error(f"  Board value at position: {board[row, col]}")
                logger.error(f"  Full trmph string: {trmph_text}")
                logger.error(f"  All moves: {moves}")
                raise ValueError(f"Duplicate move '{move}' at {(row, col)} in {trmph_text}")
        
        # Place move (alternating players: blue=1, red=2)
        player = (i % 2) + 1
        board[row, col] = player
    
    return board



def rowcol_to_trmph(row: int, col: int, board_size: int = BOARD_SIZE) -> str:
    """
    Convert (row, col) coordinates to trmph move.
    
    Args:
        row: Row index (0-12)
        col: Column index (0-12)
        board_size: Size of the board
        
    Returns:
        Trmph move string
        
    Example:
        (0, 0) → "a1"
        (12, 12) → "m13"
    """
    if not (0 <= row < board_size) or not (0 <= col < board_size):
        raise ValueError(f"Invalid coordinates: ({row}, {col}) for board size {board_size}")
    
    letter = LETTERS[col]
    number = str(row + 1)
    return letter + number


def tensor_to_rowcol(tensor_pos: int) -> Tuple[int, int]:
    """
    Convert tensor position index to (row, col).
    
    Args:
        tensor_pos: Position in flattened tensor (0-168)
        
    Returns:
        (row, col) coordinates
    """
    if not (0 <= tensor_pos < BOARD_SIZE * BOARD_SIZE):
        raise ValueError(f"Invalid tensor position: {tensor_pos}")
    
    row = tensor_pos // BOARD_SIZE
    col = tensor_pos % BOARD_SIZE
    return row, col


def rowcol_to_tensor(row: int, col: int) -> int:
    """
    Convert (row, col) to tensor position index.
    
    Args:
        row: Row index (0-12)
        col: Column index (0-12)
        
    Returns:
        Position in flattened tensor (0-168)
    """
    if not (0 <= row < BOARD_SIZE) or not (0 <= col < BOARD_SIZE):
        raise ValueError(f"Invalid coordinates: ({row}, {col}) for board size {BOARD_SIZE}")
    
    return row * BOARD_SIZE + col


def tensor_to_trmph(tensor_pos: int, board_size: int = BOARD_SIZE) -> str:
    """
    Convert tensor position index to trmph move.
    
    Args:
        tensor_pos: Position in flattened tensor (0-168)
        board_size: Size of the board
        
    Returns:
        Trmph move string
        
    Example:
        0 → "a1"
        168 → "m13"
    """
    row, col = tensor_to_rowcol(tensor_pos)
    return rowcol_to_trmph(row, col, board_size)


def trmph_to_tensor(move: str, board_size: int = BOARD_SIZE) -> int:
    """
    Convert trmph move to tensor position index.
    
    Args:
        move: Trmph move (e.g., 'a1')
        board_size: Size of the board
        
    Returns:
        Position in flattened tensor (0-168)
    """
    row, col = trmph_move_to_rowcol(move, board_size)
    return rowcol_to_tensor(row, col)


# ============================================================================
# Data Augmentation Functions
# ============================================================================

def rotate_board_180(board: np.ndarray) -> np.ndarray:
    """
    Rotate board 180 degrees (no color swap).
    
    This preserves the logical game state by only rotating, not swapping colors.
    The board edges maintain their meaning (red edges vs blue edges).
    
    Args:
        board: Board array of shape (2, 13, 13) or (13, 13)
        
    Returns:
        Rotated board (no color swap)
    """
    if board.ndim == 3:
        # 2-channel format: (2, 13, 13)
        rotated = np.flip(board, axis=(1, 2))  # Rotate 180°
        return rotated
    else:
        # Single channel format: (13, 13) with values 0/1/2
        rotated = np.flip(board, axis=(0, 1))  # Rotate 180°
        return rotated


def reflect_board_long_diagonal(board: np.ndarray) -> np.ndarray:
    """
    Reflect board along the long diagonal (top-left to bottom-right) and swap colors.
    
    This preserves the logical game state under the swap rule.
    
    Args:
        board: Board array of shape (2, 13, 13) or (13, 13)
        
    Returns:
        Reflected and color-swapped board
    """
    if board.ndim == 3:
        # 2-channel format: (2, 13, 13)
        reflected = np.transpose(board, (0, 2, 1))  # Transpose
        # Swap channels (blue <-> red)
        reflected = reflected[::-1]
        return reflected
    else:
        # Single channel format: (13, 13) with values 0/1/2
        reflected = np.transpose(board)  # Transpose
        # Swap colors: 1 <-> 2
        reflected = np.where(reflected == 1, 2, np.where(reflected == 2, 1, reflected))
        return reflected


def reflect_board_short_diagonal(board: np.ndarray) -> np.ndarray:
    """
    Reflect board along the short diagonal (top-right to bottom-left) and swap colors.
    This is a true short-diagonal reflection:
      - Each (row, col) maps to (maxIndex - col, maxIndex - row)
      - Colors are swapped (blue <-> red)
    Args:
        board: Board array of shape (2, N, N) or (N, N)
    Returns:
        Reflected and color-swapped board
    """
    if board.ndim == 3:
        # 2-channel format: (2, N, N)
        N = board.shape[1]
        reflected = np.zeros_like(board)
        for row in range(N):
            for col in range(N):
                # Swap color channel
                reflected[1, N - 1 - col, N - 1 - row] = board[0, row, col]  # Blue -> Red
                reflected[0, N - 1 - col, N - 1 - row] = board[1, row, col]  # Red -> Blue
        return reflected
    else:
        # Single channel format: (N, N) with values 0/1/2
        N = board.shape[0]
        reflected = np.zeros_like(board)
        for row in range(N):
            for col in range(N):
                val = board[row, col]
                if val == 1:
                    reflected[N - 1 - col, N - 1 - row] = 2  # Blue -> Red
                elif val == 2:
                    reflected[N - 1 - col, N - 1 - row] = 1  # Red -> Blue
                else:
                    reflected[N - 1 - col, N - 1 - row] = 0
        return reflected

def create_augmented_boards(board: np.ndarray) -> list[np.ndarray]:
    """
    Create all 4 augmented versions of a board.
    
    Args:
        board: Board array of shape (2, 13, 13) or (13, 13)
        
    Returns:
        List of 4 boards: [original, rotated_180, reflected_long, reflected_short]
    """
    rotated = rotate_board_180(board)
    reflected_long = reflect_board_long_diagonal(board)
    reflected_short = reflect_board_short_diagonal(board)
    
    return [board, rotated, reflected_long, reflected_short]


def create_augmented_policies(policy: np.ndarray) -> list[np.ndarray]:
    """
    Create policy arrays corresponding to the 4 board augmentations.
    
    Args:
        policy: Policy array of shape (169,)
        
    Returns:
        List of 4 policies corresponding to the board augmentations
    """
    # Reshape policy to (13, 13) for easier manipulation
    policy_2d = policy.reshape(13, 13)
    
    # Create the 4 augmented policies
    rotated = np.flip(policy_2d, axis=(0, 1))  # 180° rotation
    reflected_long = np.transpose(policy_2d)  # Long diagonal reflection
    reflected_short = np.flip(np.transpose(policy_2d), axis=(0, 1))  # Short diagonal reflection
    
    # Reshape back to (169,)
    return [
        policy,
        rotated.reshape(169),
        reflected_long.reshape(169),
        reflected_short.reshape(169)
    ]


def augment_board(board: np.ndarray, policy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random data augmentation to a board and its corresponding policy.
    
    This function randomly selects one of 4 possible augmentations:
    - Original board
    - 180° rotation with color swap
    - Long diagonal reflection with color swap
    - Short diagonal reflection with color swap
    
    Args:
        board: Board array of shape (2, 13, 13)
        policy: Policy array of shape (169,)
        
    Returns:
        Tuple of (augmented_board, augmented_policy)
    """
    # Create all 4 augmented versions
    augmented_boards = create_augmented_boards(board)
    augmented_policies = create_augmented_policies(policy)
    
    # Randomly select one augmentation
    import random
    idx = random.randint(0, 3)
    
    return augmented_boards[idx], augmented_policies[idx]


# ============================================================================
# Training Data Preparation
# ============================================================================


def extract_training_examples_from_game(trmph_text: str, winner_from_file: str = None, debug_info: str = "") -> List[Tuple[np.ndarray, Optional[np.ndarray], float]]:
    """
    Extract training examples from a game using correct logic for two-headed networks.
    
    This function creates training examples from all positions in a game:
    - Position i (0 to M): Board state after i moves
    - Policy target: Next move (position i+1) if available, None for final position
    - Value target: Final game outcome for all positions
    
    For two-headed networks, we include all positions and handle missing policy targets
    in the loss function. This ensures both heads get trained on the same board states.
    
    Args:
        trmph_text: Complete trmph string
        winner_from_file: Winner from file data ("1" for blue, "2" for red)
        debug_info: Optional debug information
        
    Returns:
        List of (board_state, policy_target, value_target) tuples
        - board_state: Current board state (2, 13, 13)
        - policy_target: Next move as one-hot (169,) or None if no next move
        - value_target: Final game outcome (0.0 or 1.0)

    Limitations:
    - No data augmentation (will be added later)
    Raises:
        ValueError: If game has no moves (empty game) or missing winner data
    """
    try:
        # Parse moves from trmph string
        bare_moves = strip_trmph_preamble(trmph_text)
        moves = split_trmph_moves(bare_moves)

        # Reject empty games - they provide no useful training signal
        if not moves:
            logger.error(f"Empty game found - no moves to learn from: {trmph_text[:50]}...")
            if debug_info:
                logger.error(f"Debug info: {debug_info}")
            raise ValueError("Empty game - no moves to learn from")

        # Require winner data from file - no automatic calculation
        if winner_from_file is None:
            logger.error(f"Missing winner data for game: {trmph_text[:50]}...")
            if debug_info:
                logger.error(f"Debug info: {debug_info}")
            raise ValueError("Missing winner data - cannot determine training target")

        # Validate winner format
        if winner_from_file not in ["1", "2"]:
            logger.error(f"Invalid winner format '{winner_from_file}' for game: {trmph_text[:50]}...")
            if debug_info:
                logger.error(f"Debug info: {debug_info}")
            raise ValueError(f"Invalid winner format: {winner_from_file}")

        # Set value target from validated file data
        value_target = 1.0 if winner_from_file == "1" else 0.0

        training_examples = []

        # Iterate through all positions (0 to M) to create training examples
        # Position 0 is the empty board
        # Positions 1 to M are board states after each move
        for position in range(len(moves) + 1):
            board_state = create_board_from_moves(moves[:position])
            policy_target = None
            if position < len(moves):
                next_move = moves[position]
                policy_target = create_policy_target(next_move)
            training_examples.append((board_state, policy_target, value_target))

        return training_examples

    except Exception as e:
        # Re-raise ValueError (our validation errors) but catch other exceptions
        if isinstance(e, ValueError):
            raise
        logger.error(f"Failed to extract training examples from game {trmph_text[:50]}...: {e}")
        if debug_info:
            logger.error(f"Debug info: {debug_info}")
        raise ValueError(f"Failed to process game: {e}")


def create_board_from_moves(moves: List[str]) -> np.ndarray:
    """
    Create a board state from a list of moves.
    
    Args:
        moves: List of trmph moves (e.g., ['a1', 'b2', 'c3'])
        
    Returns:
        Board state of shape (2, 13, 13)
    """
    # Initialize board
    board_matrix = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    
    # Place moves on board
    for i, move in enumerate(moves):
        row, col = trmph_move_to_rowcol(move)
        # Alternating players: blue=1, red=2
        player = (i % 2) + 1
        board_matrix[row, col] = player
    
    # Convert to 2-channel format
    board_state = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    board_state[0] = (board_matrix == 1).astype(np.float32)  # Blue channel
    board_state[1] = (board_matrix == 2).astype(np.float32)  # Red channel
    
    return board_state


def create_policy_target(move: str) -> np.ndarray:
    """
    Create a policy target from a single move.
    
    Args:
        move: Trmph move (e.g., 'a1')
        
    Returns:
        Policy target of shape (169,) with 1.0 for the move, 0.0 elsewhere
    """
    # Convert move to tensor position
    row, col = trmph_move_to_rowcol(move)
    tensor_pos = rowcol_to_tensor(row, col)
    
    # Create one-hot policy target
    policy = np.zeros(POLICY_OUTPUT_SIZE, dtype=np.float32)
    policy[tensor_pos] = 1.0
    
    return policy

def validate_game(trmph_url: str, winner_indicator: str, line_info: str = "") -> Tuple[bool, str]:
    """
    Validate a single game for corruption.
    
    Args:
        trmph_url: The trmph URL string
        winner_indicator: The winner indicator string
        line_info: Optional line information for debugging
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Test if we can extract training examples from the game without errors
        training_examples = extract_training_examples_from_game(trmph_url, winner_indicator, line_info)
        return True, ""
    except Exception as e:
        return False, str(e)


# =========================================================================
# Player-to-move Channel Utility
# =========================================================================

from hex_ai.inference.board_utils import BLUE_PLAYER, RED_PLAYER

def get_player_to_move_from_board(board_2ch: np.ndarray, error_tracker=None) -> int:
    """
    Given a (2, N, N) board, return BLUE_PLAYER if it's blue's move, RED_PLAYER if it's red's move.
    Uses error tracking to handle invalid board states gracefully.
    Args:
        board_2ch: np.ndarray of shape (2, N, N), blue and red channels
        error_tracker: Optional BoardStateErrorTracker instance
    Returns:
        int: BLUE_PLAYER or RED_PLAYER
    """
    if board_2ch.shape[0] != 2:
        raise ValueError(f"Expected board with 2 channels, got shape {board_2ch.shape}")
    
    blue_count = int(np.sum(board_2ch[0]))
    red_count = int(np.sum(board_2ch[1]))
    
    if blue_count == red_count:
        return BLUE_PLAYER
    elif blue_count == red_count + 1:
        return RED_PLAYER
    else:
        # Invalid board state - use error tracking if available
        error_msg = f"Invalid board state: blue_count={blue_count}, red_count={red_count}. Board must have equal or one more blue than red."
        
        if error_tracker is not None:
            error_tracker.record_error(
                board_state=board_2ch,
                error_msg=error_msg,
                file_info=getattr(error_tracker, '_current_file', "Unknown"),
                sample_info=getattr(error_tracker, '_current_sample', "Unknown")
            )
            # Return a default value to continue processing
            # Assume it's blue's turn if we can't determine
            return BLUE_PLAYER
        else:
            # Fall back to original behavior if no error tracker
            raise ValueError(error_msg)
