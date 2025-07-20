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
import random
from datetime import datetime
import json


from .config import (
    BOARD_SIZE, NUM_PLAYERS, TRMPH_EXTENSION, POLICY_OUTPUT_SIZE, 
    BLUE_PLAYER, RED_PLAYER, BLUE_PIECE, RED_PIECE, EMPTY_PIECE,
    PIECE_ONEHOT, EMPTY_ONEHOT, BLUE_CHANNEL, RED_CHANNEL, PLAYER_CHANNEL
)
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
TRLETTERS = LETTERS[:BOARD_SIZE]


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
        # Convert one-hot encoded channels to N×N format
        board_2d[board[BLUE_CHANNEL] == PIECE_ONEHOT] = BLUE_PIECE  # Blue pieces
        board_2d[board[RED_CHANNEL] == PIECE_ONEHOT] = RED_PIECE   # Red pieces
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
        if bare_moves[i] not in TRLETTERS:
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
    if letter not in TRLETTERS:
        raise ValueError(f"Invalid letter in move: {move}")
    if not (1 <= number <= board_size):
        raise ValueError(f"Invalid number in move: {move}")
    row = number - 1
    col = LETTERS.index(letter)
    return row, col


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
        # Swap colors: BLUE_PIECE <-> RED_PIECE
        reflected = np.where(reflected == BLUE_PIECE, RED_PIECE,
                             np.where(reflected == RED_PIECE, BLUE_PIECE,
                                      reflected))
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
                if val == BLUE_PIECE:
                    reflected[N - 1 - col, N - 1 - row] = RED_PIECE  # Blue -> Red
                elif val == RED_PIECE:
                    reflected[N - 1 - col, N - 1 - row] = BLUE_PIECE  # Red -> Blue
                else:
                    reflected[N - 1 - col, N - 1 - row] = EMPTY_PIECE
        return reflected

def create_augmented_boards(board: np.ndarray) -> list[np.ndarray]:
    """
    Create 4 augmented board states from a single board.
    
    Args:
        board: 2-channel board state (2, N, N)
        
    Returns:
        List of 4 board states:
        - Original
        - 180° rotation (no color swap)
        - Long diagonal reflection + color swap
        - Short diagonal reflection + color swap
    """
    # Ensure input is contiguous
    board = np.ascontiguousarray(board)
    
    # Create augmented boards
    rotated = rotate_board_180(board)
    reflected_long = reflect_board_long_diagonal(board)
    reflected_short = reflect_board_short_diagonal(board)
    
    # Ensure all outputs are contiguous
    return [
        board.copy(),
        np.ascontiguousarray(rotated),
        np.ascontiguousarray(reflected_long),
        np.ascontiguousarray(reflected_short)
    ]


def create_augmented_policies(policy: np.ndarray) -> list[np.ndarray]:
    """
    Create policy labels for the 4 board augmentations.
    - For color-swapping symmetries, policy is transformed accordingly.
    """
    # Handle None policies (final moves with no next move)
    if policy is None:
        # Return 4 copies of zero policy for final moves
        zero_policy = np.zeros(169, dtype=np.float32)
        return [zero_policy.copy() for _ in range(4)]
    
    # Ensure input is contiguous
    policy = np.ascontiguousarray(policy)
    
    # Reshape to 2D for easier manipulation
    policy_2d = policy.reshape(13, 13)
    
    # Create augmented policies
    rotated = np.flip(policy_2d, axis=(0, 1))  # 180° rotation (no color swap)
    reflected_long = np.transpose(policy_2d)  # Long diagonal reflection
    reflected_short = np.zeros_like(policy_2d)  # Short diagonal reflection

    # Short diagonal reflection: (row, col) -> (N-1-col, N-1-row)
    for row in range(13):
        for col in range(13):
            reflected_short[12 - col, 12 - row] = policy_2d[row, col]
    
    # Ensure all outputs are contiguous
    return [
        policy.copy(),
        np.ascontiguousarray(rotated.reshape(169)),
        np.ascontiguousarray(reflected_long.reshape(169)),
        np.ascontiguousarray(reflected_short.reshape(169))
    ]


def create_augmented_values(value: float) -> list[float]:
    """
    Create value labels for the 4 board augmentations.
    - For color-swapping symmetries, swap the value (1.0 <-> 0.0).
    """
    return [
        value,          # Original
        value,          # 180° rotation (no color swap)
        1.0 - value,    # Long diagonal reflection + color swap
        1.0 - value     # Short diagonal reflection + color swap
    ]


def create_augmented_player_to_move(player_to_move: int) -> list[int]:
    """
    Create player-to-move values for the 4 board augmentations.
    - For color-swapping symmetries, swap the player (0 <-> 1).
    """
    return [
        player_to_move,          # Original
        player_to_move,          # 180° rotation (no color swap)
        1 - player_to_move,      # Long diagonal reflection + color swap
        1 - player_to_move       # Short diagonal reflection + color swap
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

def create_board_from_moves(moves: List[str]) -> np.ndarray:
    """
    Create a board state from a list of moves.
    
    This function uses parse_trmph_to_board internally to ensure consistent
    duplicate move handling and error checking.
    
    Args:
        moves: List of trmph moves (e.g., ['a1', 'b2', 'c3'])
        
    Returns:
        Board state of shape (2, 13, 13)
        
    Raises:
        ValueError: If duplicate moves are found (duplicate_action="exception")
    """
    # Convert moves list to TRMPH string format
    trmph_text = f"#13,{''.join(moves)}"
    
    # Use parse_trmph_to_board for consistent duplicate handling and error checking
    # We use "exception" since remove_repeated_moves should be called before this function
    board_nxn = parse_trmph_to_board(trmph_text, duplicate_action="exception")
    
    # Convert N×N format to 2-channel format for neural network training
    board_state = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    board_state[BLUE_CHANNEL] = (board_nxn == BLUE_PIECE).astype(np.float32)  # Blue channel
    board_state[RED_CHANNEL] = (board_nxn == RED_PIECE).astype(np.float32)   # Red channel
    
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

def create_augmented_example(board: np.ndarray, policy: np.ndarray, value: float) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """
    Create all 4 augmented examples from a single example.
    
    Args:
        board: 2-channel board state (2, N, N)
        policy: Policy target (169,)
        value: Value target (float)
        
    Returns:
        List of 4 tuples: [(board_2ch, policy, value), ...]
        - Original
        - 180° rotation (no color swap)
        - Long diagonal reflection + color swap
        - Short diagonal reflection + color swap
    """
    # Skip empty boards (no pieces to augment)
    if np.sum(board) == 0:
        return [(board, policy, value)] * 4
    
    # Create augmented components
    augmented_boards = create_augmented_boards(board)
    augmented_policies = create_augmented_policies(policy)
    augmented_values = create_augmented_values(value)
    
    # Create augmented examples
    augmented_examples = []
    for i in range(4):
        augmented_examples.append((
            augmented_boards[i],
            augmented_policies[i],
            augmented_values[i]
        ))
    
    return augmented_examples


def create_augmented_example_with_player_to_move(board: np.ndarray, policy: np.ndarray, value: float, error_tracker=None) -> list[tuple[np.ndarray, np.ndarray, float, int]]:
    """
    Create 4 augmented examples from a single board position.
    
    Args:
        board: Board state as (2, 13, 13) array
        policy: Policy target as (169,) array or None for final moves
        value: Value target as float
        error_tracker: Optional BoardStateErrorTracker for handling invalid board states
        
    Returns:
        List of 4 tuples: (augmented_board, augmented_policy, augmented_value, player_to_move)
    """
    # Handle None policies (final moves with no next move)
    if policy is None:
        policy = np.zeros(169, dtype=np.float32)
    
    # Create augmented boards, policies, values, and player-to-move
    augmented_boards = create_augmented_boards(board)
    augmented_policies = create_augmented_policies(policy)
    augmented_values = create_augmented_values(value)
    
    # Compute player-to-move from the board, then create augmented versions
    # Use error tracker to handle invalid board states gracefully
    player_to_move = get_player_to_move_from_board(board, error_tracker)
    augmented_player_to_move = create_augmented_player_to_move(player_to_move)
    
    # Combine into examples
    examples = []
    for i in range(4):
        examples.append((
            augmented_boards[i],
            augmented_policies[i], 
            augmented_values[i],
            augmented_player_to_move[i]
        ))
    
    return examples


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

def assign_value_sample_tiers(total_positions: int) -> List[int]:
    """
    Assign sampling tiers to positions in a game.
    
    Tiers:
    - 0: High priority (5 positions) - Always used for value training
    - 1: Medium priority (5 positions) - Usually used for value training  
    - 2: Low priority (10 positions) - Sometimes used for value training
    - 3: Very low priority (20+ positions) - Rarely used for value training
    
    This allows flexible control over how many positions per game are used
    for value network training while keeping all positions for policy training.
    """
    if total_positions <= 5:
        # Small games: all positions get tier 0
        return [0] * total_positions
    
    # Define positions per tier
    positions_per_tier = [5, 5, 10, max(0, total_positions - 20)]
    
    # Assign tiers
    tiers = []
    for tier, count in enumerate(positions_per_tier):
        if count > 0:
            tiers.extend([tier] * min(count, total_positions - len(tiers)))
    
    # Shuffle within each tier to avoid bias
    tier_groups = {}
    for i, tier in enumerate(tiers):
        if tier not in tier_groups:
            tier_groups[tier] = []
        tier_groups[tier].append(i)
    
    # Shuffle each tier group
    for tier in tier_groups:
        random.shuffle(tier_groups[tier])
    
    # Reconstruct tiers list
    result = [0] * total_positions
    for tier, indices in tier_groups.items():
        for idx in indices:
            result[idx] = tier
    
    return result


def remove_repeated_moves(moves: List[str]) -> List[str]:
    """
    Remove repeated moves and all subsequent moves from the game.
    
    Args:
        moves: List of TRMPH moves
        
    Returns:
        Cleaned list of moves with no repetitions
    """
    seen_moves = set()
    clean_moves = []
    
    for move in moves:
        if move in seen_moves:
            # Found repeated move - discard this and all subsequent moves
            logger.debug(f"Repeated move {move} found, discarding game from this point")
            break
        seen_moves.add(move)
        clean_moves.append(move)
    
    return clean_moves


def extract_training_examples_from_game(
    trmph_text: str, 
    winner_from_file: str = None,
    game_id: Tuple[int, int] = None,  # (file_idx, line_idx)
    include_trmph: bool = False,       # Whether to include full TRMPH string
    shuffle_positions: bool = True
) -> List[Dict]:
    """
    Extract training examples with comprehensive metadata and flexible sampling.
    
    Args:
        trmph_text: Complete TRMPH string
        winner_from_file: Winner from file data ("1" for blue, "2" for red)
        game_id: Tuple of (file_index, line_index) for tracking
        include_trmph: Whether to include full TRMPH string in metadata
        shuffle_positions: Whether to shuffle position order within game
        
    Returns:
        List of enhanced training examples with metadata
    """
    try:
        # Parse moves and validate
        bare_moves = strip_trmph_preamble(trmph_text)
        moves = split_trmph_moves(bare_moves)
        
        # Handle repeated moves
        moves = remove_repeated_moves(moves)
        
        if not moves:
            raise ValueError("Empty game after removing repeated moves")
        
        # Validate winner and convert to clear format
        if winner_from_file not in ["1", "2"]:
            raise ValueError(f"Invalid winner format: {winner_from_file}")
        
        # Convert winner format: "1"=BLUE, "2"=RED
        winner_clear = "BLUE" if winner_from_file == "1" else "RED"
        value_target = 0.0 if winner_from_file == "1" else 1.0  # BLUE=0.0, RED=1.0
        
        total_positions = len(moves) + 1
        
        # Assign sampling tiers
        value_sample_tiers = assign_value_sample_tiers(total_positions)
        
        # Create position indices (shuffle if requested)
        position_indices = list(range(total_positions))
        if shuffle_positions:
            random.shuffle(position_indices)
        
        training_examples = []
        
        for i, position in enumerate(position_indices):
            # Create board state
            board_state = create_board_from_moves(moves[:position])
            
            # Create policy target
            policy_target = None if position >= len(moves) else create_policy_target(moves[position])
            
            # Create metadata
            metadata = {
                'game_id': game_id,
                'position_in_game': position,
                'total_positions': total_positions,
                'value_sample_tier': value_sample_tiers[i],
                'winner': winner_clear  # Store as "BLUE" or "RED"
            }
            
            if include_trmph:
                metadata['trmph_game'] = trmph_text
            
            # Create example
            example = {
                'board': board_state,
                'policy': policy_target,
                'value': value_target,
                'metadata': metadata
            }
            
            training_examples.append(example)
        
        return training_examples
        
    except Exception as e:
        logger.error(f"Failed to extract training examples from game {trmph_text[:50]}...: {e}")
        raise ValueError(f"Failed to process game: {e}")


def extract_positions_range(
    trmph_text: str, 
    winner: str, 
    start_pos: int, 
    end_pos: int, 
    game_id: Tuple[int, int]
) -> Tuple[List[Dict], bool]:
    """
    Extract only positions in the specified range from a game.
    
    Args:
        trmph_text: Complete TRMPH string
        winner: Winner from file data ("1" or "2")
        start_pos: Starting position (inclusive)
        end_pos: Ending position (exclusive)
        game_id: Tuple of (file_index, line_index)
        
    Returns:
        List of training examples for the specified position range
    """
    try:
        # Parse moves
        bare_moves = strip_trmph_preamble(trmph_text)
        raw_moves = split_trmph_moves(bare_moves)        
        moves = remove_repeated_moves(raw_moves)
        repeat = False
        if len(raw_moves) != len(moves):
            repeat = True
        
        if not moves:
            return [], False
        
        # Validate winner
        if winner not in ["1", "2"]:
            raise ValueError(f"Invalid winner format: {winner}")
        
        winner_clear = "BLUE" if winner == "1" else "RED"
        value_target = 0.0 if winner == "1" else 1.0
        
        total_positions = len(moves) + 1
        examples = []
        
        # Extract positions in range
        for position in range(start_pos, min(end_pos, total_positions)):
            # Create board state
            board_state = create_board_from_moves(moves[:position])
            
            # Create policy target
            policy_target = None if position >= len(moves) else create_policy_target(moves[position])
            
            # Create metadata
            metadata = {
                'game_id': game_id,
                'position_in_game': position,
                'total_positions': total_positions,
                'value_sample_tier': 0,  # Default tier for range extraction
                'winner': winner_clear
            }
            
            # Create example
            example = {
                'board': board_state,
                'policy': policy_target,
                'value': value_target,
                'metadata': metadata
            }
            
            examples.append(example)
        
        return examples, repeat
        
    except Exception as e:
        logger.error(f"Failed to extract positions range from game: {e}")
        return [], False


def create_file_lookup_table(trmph_files: List[Path], output_dir: Path) -> Path:
    """
    Create a file lookup table mapping file indices to actual filenames.
    
    Args:
        trmph_files: List of TRMPH file paths
        output_dir: Directory to save the lookup table
        
    Returns:
        Path to the created lookup table file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lookup_file = output_dir / f"file_lookup_{timestamp}.json"
    
    file_mapping = {}
    for file_idx, file_path in enumerate(trmph_files):
        file_mapping[file_idx] = str(file_path)
    
    lookup_data = {
        'file_mapping': file_mapping,
        'created_at': datetime.now().isoformat(),
        'total_files': len(trmph_files),
        'format_version': '1.0'
    }
    
    with open(lookup_file, 'w') as f:
        json.dump(lookup_data, f, indent=2)
    
    logger.info(f"Created file lookup table: {lookup_file}")
    return lookup_file


# =========================================================================
# Player-to-move Channel Utility
# =========================================================================

from hex_ai.config import BLUE_PLAYER, RED_PLAYER

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
