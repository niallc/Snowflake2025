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
from typing import List, Tuple, Dict, Optional, Union, Any
import logging
import re
import string
import random
from datetime import datetime
import json

from hex_ai.enums import player_to_int
from .config import (
    BOARD_SIZE, NUM_PLAYERS, TRMPH_EXTENSION, POLICY_OUTPUT_SIZE,
    PIECE_ONEHOT, EMPTY_ONEHOT,
    TRMPH_BLUE_WIN, TRMPH_RED_WIN, TRAINING_BLUE_WIN, TRAINING_RED_WIN,
)
from hex_ai.enums import Piece, Channel, piece_to_char, channel_to_int
from hex_ai.value_utils import (
    trmph_winner_to_training_value, trmph_winner_to_clear_str,
    get_player_to_move_from_moves, Winner, Player
)
from hex_ai.utils.format_conversion import (
    strip_trmph_preamble, split_trmph_moves, trmph_move_to_rowcol, parse_trmph_to_board,
    rowcol_to_trmph, tensor_to_rowcol, rowcol_to_tensor, tensor_to_trmph, trmph_to_tensor
)
from hex_ai.utils.player_utils import get_player_to_move_from_board

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

def find_trmph_files(source_dirs: List[Path]) -> List[Tuple[Path, Path]]:
    """
    Find all .trmph files in the specified source directories.
    
    Args:
        source_dirs: List of source directories to search for .trmph files
        
    Returns:
        List of tuples: (source_dir, file_path)
    """
    all_files = []
    
    for source_dir in source_dirs:
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory {source_dir} does not exist - this is likely a configuration error")
            
        # Find all .trmph files recursively
        trmph_files = sorted(source_dir.rglob("*.trmph"))
        logger.info(f"Found {len(trmph_files)} .trmph files in {source_dir}")
        
        for file_path in trmph_files:
            all_files.append((source_dir, file_path))
    
    all_files.sort(key=lambda item: str(item[1]))
    logger.info(f"Total .trmph files found: {len(all_files)}")
    return all_files


def extract_games_from_file(file_path: Path) -> List[str]:
    """Extract game lines from a TRMPH file, skipping headers."""
    games = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Current training-corpus game lines start with #13, (board size + format)
            # Header lines start with just # (metadata)
            if line and line.startswith('#13,'):
                # Remove any inline comments (everything after # that's not part of the game)
                if ' # ' in line:
                    line = line.split(' # ')[0].strip()
                games.append(line)
    return games


def remove_duplicates(games: List[str]) -> List[str]:
    """Remove duplicate games while preserving order."""
    seen = set()
    unique_games = []
    
    for game in games:
        if game not in seen:
            seen.add(game)
            unique_games.append(game)
    
    logger.info(f"Removed {len(games) - len(unique_games)} duplicate games")
    return unique_games

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
        board_2d[board[channel_to_int(Channel.BLUE)] == PIECE_ONEHOT] = piece_to_char(Piece.BLUE)  # Blue pieces
        board_2d[board[channel_to_int(Channel.RED)] == PIECE_ONEHOT] = piece_to_char(Piece.RED)   # Red pieces
    else:
        board_2d = board
    
    if format_type == "matrix":
        return str(board_2d)
    
    elif format_type == "visual":
        # Create a visual representation
        symbols = {piece_to_char(Piece.EMPTY): ".", piece_to_char(Piece.BLUE): "B", piece_to_char(Piece.RED): "R"}
        lines = []
        for row in range(BOARD_SIZE):
            line = " " * row  # Indent for hex shape
            for col in range(BOARD_SIZE):
                piece = board_2d[row, col]
                line += symbols[piece] + " "
            lines.append(line)
        return "\n".join(lines)
    
    else:
        raise ValueError(f"Unknown format_type: {format_type}")



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
        reflected = np.where(reflected == piece_to_char(Piece.BLUE), piece_to_char(Piece.RED),
                             np.where(reflected == piece_to_char(Piece.RED), piece_to_char(Piece.BLUE),
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
                if val == piece_to_char(Piece.BLUE):
                    reflected[N - 1 - col, N - 1 - row] = piece_to_char(Piece.RED)  # Blue -> Red
                elif val == piece_to_char(Piece.RED):
                    reflected[N - 1 - col, N - 1 - row] = piece_to_char(Piece.BLUE)  # Red -> Blue
                else:
                    reflected[N - 1 - col, N - 1 - row] = piece_to_char(Piece.EMPTY)
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


def create_augmented_player_to_move(player_to_move: Player) -> list[Player]:
    """
    Create player-to-move values for the 4 board augmentations.
    - For color-swapping symmetries, swap the player (BLUE <-> RED).
    """
    return [
        player_to_move,                    # Original
        player_to_move,                    # 180° rotation (no color swap)
        Player.RED if player_to_move == Player.BLUE else Player.BLUE,  # Long diagonal reflection + color swap
        Player.RED if player_to_move == Player.BLUE else Player.BLUE   # Short diagonal reflection + color swap
    ]

# =====================
# Position Augmentation Generators
# =====================


def create_augmented_example_with_player_to_move(
    board: np.ndarray, policy: np.ndarray, value: float, error_tracker=None
) -> list[tuple[np.ndarray, np.ndarray, float, Player]]:
    """
    Create 4 augmented examples from a single board position, including player-to-move.
    
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
    
    # Create augmented boards, policies, values
    augmented_boards = create_augmented_boards(board)
    augmented_policies = create_augmented_policies(policy)
    augmented_values = create_augmented_values(value)
    
    # Compute player-to-move from the board, then create augmented versions
    player_to_move = get_player_to_move_from_board(board, error_tracker)
    # Keep as Player enum for augmentation
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
    board_nxn = parse_trmph_to_board(trmph_text)
    
    # Convert N×N format to 2-channel format for neural network training
    board_state = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    board_state[channel_to_int(Channel.BLUE)] = (board_nxn == piece_to_char(Piece.BLUE)).astype(np.float32)  # Blue channel
    board_state[channel_to_int(Channel.RED)] = (board_nxn == piece_to_char(Piece.RED)).astype(np.float32)   # Red channel
    
    return board_state


def create_policy_target(move: str) -> np.ndarray:
    """
    Create a policy target from a single move.
    
    Args:
        move: Trmph move (e.g., 'a1')
        
    Returns:
        Policy target of shape (BOARD_SIZE * BOARD_SIZE,) with 1.0 for the move, 0.0 elsewhere
    """
    # Convert move to tensor position
    row, col = trmph_move_to_rowcol(move)
    tensor_pos = rowcol_to_tensor(row, col)
    
    # Create one-hot policy target
    policy = np.zeros(POLICY_OUTPUT_SIZE, dtype=np.float32)
    policy[tensor_pos] = 1.0
    return policy

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


def remove_repeated_moves(moves: List[str]) -> Optional[List[str]]:
    """
    Check for repeated moves in the game.
    
    Args:
        moves: List of TRMPH moves
        
    Returns:
        Cleaned list of moves with no repetitions, or None if duplicates were found
    """
    seen_moves = set()
    clean_moves = []
    
    for move in moves:
        if move in seen_moves:
            # Found repeated move - skip this game entirely
            logger.warning(f"Duplicate move '{move}' found in game, skipping entire game")
            return None
        seen_moves.add(move)
        clean_moves.append(move)
    
    return clean_moves


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
# File Lookup Utilities
# =========================================================================

def get_filename_from_game_id_using_state(game_id: tuple, state_file_path: Path) -> str:
    """
    Get filename from game_id using processing state file.
    
    Args:
        game_id: Tuple of (file_idx, line_idx)
        state_file_path: Path to processing_state.json file
        
    Returns:
        Filename corresponding to the game_id
        
    Raises:
        FileNotFoundError: If state file doesn't exist
        ValueError: If file_idx is out of range
    """
    if not state_file_path.exists():
        raise FileNotFoundError(f"Processing state file not found: {state_file_path}")
    
    with open(state_file_path, 'r') as f:
        state_data = json.load(f)
    
    processed_files = state_data.get('processed_files', [])
    file_idx, line_idx = game_id
    
    if file_idx >= len(processed_files):
        raise ValueError(f"File index {file_idx} out of range (max: {len(processed_files)-1})")
    
    file_path = processed_files[file_idx]['file']
    return Path(file_path).name


# =========================================================================
# Player-to-move Channel Utility
# =========================================================================

from hex_ai.value_utils import Player

def get_valid_policy_target(policy, use_uniform: bool = False):
    """
    Convert a possibly-None policy to a valid vector for training or analysis.
    If policy is None (terminal position), returns either all zeros or uniform (1/N) vector.
    Args:
        policy: np.ndarray, torch.Tensor, or None
        use_uniform: If True, returns uniform (1/N) vector; else all zeros
    Returns:
        np.ndarray of shape (BOARD_SIZE * BOARD_SIZE,)
    """
    N = BOARD_SIZE * BOARD_SIZE
    if policy is None:
        if use_uniform:
            return np.full(N, 1.0 / N, dtype=np.float32)
        else:
            return np.zeros(N, dtype=np.float32)
    # If already a numpy array or tensor, just return as numpy
    if isinstance(policy, torch.Tensor):
        return policy.cpu().numpy()
    return np.array(policy, dtype=np.float32)


def preprocess_example_for_model(ex, use_uniform_policy: bool = False):
    """
    Convert a raw example dict to (board, policy, value) tensors for model input.
    - Adds player-to-move channel to board (from 2ch to 3ch)
    - Converts policy label using get_valid_policy_target
    Args:
        ex: dict with keys 'board', 'policy', 'value'
        use_uniform_policy: If True, use uniform policy for terminal positions
    Returns:
        board: torch.FloatTensor (3, BOARD_SIZE, BOARD_SIZE)
        policy: torch.FloatTensor (BOARD_SIZE * BOARD_SIZE,)
        value: torch.FloatTensor (scalar or shape (1,))
    """
    board_2ch = ex['board']
    player_to_move = get_player_to_move_from_board(board_2ch)
    # Convert Player enum to integer for tensor creation
    player_to_move_int = player_to_int(player_to_move)
    player_channel = np.full((1, BOARD_SIZE, BOARD_SIZE), float(player_to_move_int), dtype=board_2ch.dtype)
    board_3ch = np.concatenate([board_2ch, player_channel], axis=0)
    board = torch.tensor(board_3ch, dtype=torch.float32)
    policy = get_valid_policy_target(ex['policy'], use_uniform=use_uniform_policy)
    if isinstance(policy, torch.Tensor):
        policy = policy.detach().clone()
    else:
        policy = torch.tensor(policy, dtype=torch.float32)
    value = torch.tensor(ex['value'], dtype=torch.float32)
    return board, policy, value


def extract_training_examples_with_selector_from_game(
    trmph_text: str,
    winner_from_file: str,
    game_id: tuple,
    position_selector: str = "all",
    policy_train_mask: Optional[str] = None,
    policy_move_codes: Optional[str] = None,
    policy_search_targets: Optional[np.ndarray] = None,
    policy_target_source_codes: Optional[str] = None,
    policy_target_version: Optional[int] = None,
    include_trmph: bool = False,
    shuffle_positions: bool = True
) -> tuple[list, Optional[str]]:
    """
    Extract training examples with a selector for position type: 'all', 'final', or 'penultimate'.
    """
    try:
        bare_moves = strip_trmph_preamble(trmph_text)
        moves = remove_repeated_moves(split_trmph_moves(bare_moves))
        if moves is None:
            # Game had duplicate moves - skip entirely
            logger.warning(f"Game skipped due to duplicate moves: {trmph_text[:50]}...")
            return [], "duplicate_moves"
        if not moves:
            raise ValueError("Empty game after removing repeated moves")
        total_positions = len(moves) + 1

        if policy_train_mask is not None:
            if not isinstance(policy_train_mask, str):
                raise ValueError(
                    f"policy_train_mask must be str when provided, got {type(policy_train_mask)}"
                )
            if len(policy_train_mask) != len(moves):
                raise ValueError(
                    f"policy_train_mask length {len(policy_train_mask)} does not match move count {len(moves)}"
                )
            invalid_mask_bits = sorted(set(policy_train_mask) - {"0", "1"})
            if invalid_mask_bits:
                raise ValueError(
                    f"policy_train_mask contains invalid bits: {invalid_mask_bits}"
                )

        if policy_move_codes is not None:
            if not isinstance(policy_move_codes, str):
                raise ValueError(
                    f"policy_move_codes must be str when provided, got {type(policy_move_codes)}"
                )
            if len(policy_move_codes) != len(moves):
                raise ValueError(
                    f"policy_move_codes length {len(policy_move_codes)} does not match move count {len(moves)}"
                )
            invalid_codes = sorted(set(policy_move_codes) - {"V", "G", "C", "T"})
            if invalid_codes:
                raise ValueError(
                    f"policy_move_codes contains invalid values: {invalid_codes}"
                )

        policy_search_targets_arr: Optional[np.ndarray] = None
        if policy_search_targets is not None:
            policy_search_targets_arr = np.asarray(policy_search_targets, dtype=np.float32)
            if policy_search_targets_arr.ndim != 2:
                raise ValueError(
                    f"policy_search_targets must be 2D when provided, got shape {policy_search_targets_arr.shape}"
                )
            if policy_search_targets_arr.shape[0] != len(moves):
                raise ValueError(
                    "policy_search_targets row count must match move count "
                    f"(got {policy_search_targets_arr.shape[0]} vs {len(moves)})"
                )
            if policy_search_targets_arr.shape[1] != POLICY_OUTPUT_SIZE:
                raise ValueError(
                    "policy_search_targets column count must match policy output size "
                    f"(got {policy_search_targets_arr.shape[1]} vs {POLICY_OUTPUT_SIZE})"
                )
            if not np.isfinite(policy_search_targets_arr).all():
                raise ValueError("policy_search_targets contains non-finite values")
            if (policy_search_targets_arr < 0.0).any():
                raise ValueError("policy_search_targets contains negative probabilities")

            trainable_row_mask: Optional[List[bool]] = None
            if policy_train_mask is not None:
                trainable_row_mask = [bit == "1" for bit in policy_train_mask]
            elif policy_target_source_codes is not None:
                trainable_row_mask = [
                    code in {"V", "G", "T"} for code in policy_target_source_codes
                ]
            elif policy_move_codes is not None:
                trainable_row_mask = [code in {"V", "G", "T"} for code in policy_move_codes]

            if trainable_row_mask is not None:
                row_sums = policy_search_targets_arr.sum(axis=1, dtype=np.float64)
                bad_rows = [
                    idx
                    for idx, trainable in enumerate(trainable_row_mask)
                    if trainable and float(row_sums[idx]) <= 1e-12
                ]
                if bad_rows:
                    preview = ", ".join(str(idx) for idx in bad_rows[:8])
                    if len(bad_rows) > 8:
                        preview += f", ... ({len(bad_rows)} total)"
                    raise ValueError(
                        "policy_search_targets has non-positive probability mass on "
                        f"trainable rows at positions: {preview}"
                    )

        if policy_target_source_codes is not None:
            if not isinstance(policy_target_source_codes, str):
                raise ValueError(
                    "policy_target_source_codes must be str when provided, "
                    f"got {type(policy_target_source_codes)}"
                )
            if len(policy_target_source_codes) != len(moves):
                raise ValueError(
                    "policy_target_source_codes length must match move count "
                    f"(got {len(policy_target_source_codes)} vs {len(moves)})"
                )
            invalid_target_source_codes = sorted(
                set(policy_target_source_codes) - {"V", "G", "C", "T"}
            )
            if invalid_target_source_codes:
                raise ValueError(
                    "policy_target_source_codes contains invalid values: "
                    f"{invalid_target_source_codes}"
                )
            if policy_move_codes is not None and policy_target_source_codes != policy_move_codes:
                raise ValueError(
                    "policy_target_source_codes must match policy_move_codes when both are provided"
                )

        if policy_target_version is not None:
            if isinstance(policy_target_version, bool):
                raise ValueError("policy_target_version must be int, got bool")
            try:
                policy_target_version = int(policy_target_version)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "policy_target_version must be int when provided, "
                    f"got {type(policy_target_version)}"
                ) from exc
            if policy_target_version < 0:
                raise ValueError(
                    f"policy_target_version must be >= 0, got {policy_target_version}"
                )

        if winner_from_file not in [TRMPH_BLUE_WIN, TRMPH_RED_WIN]:
            raise ValueError(f"Invalid winner format: {winner_from_file}")
        winner_clear = trmph_winner_to_clear_str(winner_from_file)
        value_target = trmph_winner_to_training_value(winner_from_file)
        # Determine which positions to extract
        if position_selector == "all":
            position_indices = list(range(total_positions))
            if shuffle_positions:
                random.shuffle(position_indices)
        elif position_selector == "final":
            position_indices = [total_positions - 1]
        elif position_selector == "penultimate":
            if total_positions >= 2:
                position_indices = [total_positions - 2]
            else:
                return []  # No penultimate position
        else:
            raise ValueError(f"Unknown position_selector: {position_selector}")
        # Assign sample tiers (for compatibility, just use 0 for non-all)
        if position_selector == "all":
            value_sample_tiers = assign_value_sample_tiers(total_positions)
        else:
            value_sample_tiers = [0] * len(position_indices)
        training_examples = []
        for i, position in enumerate(position_indices):
            board_state = create_board_from_moves(moves[:position])
            is_terminal_position = position >= len(moves)
            if not is_terminal_position and policy_train_mask is not None:
                # Skip non-terminal positions that are explicitly masked out by provenance.
                # These are not useful policy targets and do not add independent value signal.
                # Note: terminal-winning move annotations (code 'T') map to trainable mask=1
                # and are therefore kept as normal policy targets.
                if policy_train_mask[position] != "1":
                    continue

            policy_target = None
            policy_search_target = None
            if not is_terminal_position:
                policy_target = create_policy_target(moves[position])
                if policy_search_targets_arr is not None:
                    policy_search_target = np.array(
                        policy_search_targets_arr[position], dtype=np.float32, copy=True
                    )
            player_to_move = get_player_to_move_from_moves(moves[:position])
            # Convert winner string to Winner enum
            winner_enum = Winner.BLUE if winner_clear == "BLUE" else Winner.RED if winner_clear == "RED" else None
            
            metadata = {
                'game_id': game_id,
                'position_in_game': position,
                'total_positions': total_positions,
                'value_sample_tier': value_sample_tiers[i],
                'winner': winner_enum
            }
            if policy_move_codes is not None and not is_terminal_position:
                metadata['policy_move_code'] = policy_move_codes[position]
            if policy_train_mask is not None and not is_terminal_position:
                metadata['policy_target_trainable'] = policy_train_mask[position] == "1"
            if policy_target_source_codes is not None and not is_terminal_position:
                metadata['policy_target_source_code'] = policy_target_source_codes[position]
            if policy_target_version is not None and not is_terminal_position:
                metadata['policy_target_version'] = policy_target_version
            if include_trmph:
                metadata['trmph_game'] = trmph_text
            example = {
                'board': board_state,
                'policy': policy_target,
                'policy_search_target': policy_search_target,
                'value': value_target,
                'player_to_move': player_to_move,
                'metadata': metadata
            }
            training_examples.append(example)
        return training_examples, None
    except Exception as e:
        logger.error(f"Failed to extract training examples from game {trmph_text[:50]}...: {e}")
        raise ValueError(f"Failed to process game: {e}")
