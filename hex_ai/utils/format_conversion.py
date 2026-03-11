"""
Format conversion utilities for Hex AI.

This module provides utilities for converting between different board representations
and coordinate systems used in the Hex AI project.
"""

import numpy as np
import torch
from typing import Tuple

from hex_ai.config import (
    BOARD_SIZE, PIECE_ONEHOT, TRMPH_BLUE_WIN, TRMPH_RED_WIN
)
from hex_ai.enums import Piece, Channel, piece_to_char, channel_to_int, player_to_int

import string
import logging
import re
logger = logging.getLogger(__name__)

LETTERS = string.ascii_lowercase

# TODO: ENUM MIGRATION - This module now uses Piece/Channel internally and keeps
# boundary types (TRMPH strings, numpy/tensors) for IO. Continue migrating
# upstream/downstream code to pass Enums in domain logic and convert only at boundaries.

# --- TRMPH/Move Conversion Functions (from data_utils.py) ---
def strip_trmph_preamble(trmph_text: str) -> str:
    """Strip TRMPH preamble if present, otherwise return input as-is.
    
    This allows the function to work with both full TRMPH strings (#13,a1b2)
    and already-normalized strings (a1b2).
    """
    match = re.compile(r"#(\d+),").search(trmph_text)
    if not match:
        # No preamble found - assume input is already normalized (bare moves).
        # Validation of move format happens downstream in split_trmph_moves().
        return trmph_text
    
    return trmph_text[match.end():]

def split_trmph_moves(bare_moves: str) -> list[str]:
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

def count_trmph_moves(trmph_text: str) -> int:
    """
    Count moves in a TRMPH string (with or without preamble).
    Returns 0 for empty/None input. Raises ValueError for invalid format.
    """
    if not trmph_text:
        return 0
    bare_moves = strip_trmph_preamble(trmph_text)
    if not bare_moves:
        return 0
    return len(split_trmph_moves(bare_moves))

def _validate_board_size(board_size: int) -> int:
    """Validate and normalize board size for coordinate conversion helpers."""
    size = int(board_size)
    if size <= 0:
        raise ValueError(f"Board size must be positive, got {board_size}")
    if size > len(LETTERS):
        raise ValueError(f"Board size {size} exceeds supported coordinate range ({len(LETTERS)})")
    return size

def trmph_move_to_rowcol(move: str, board_size: int = BOARD_SIZE) -> tuple[int, int]:
    board_size = _validate_board_size(board_size)
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

def trmph_to_moves(trmph_text: str, board_size: int = BOARD_SIZE) -> list[tuple[int, int]]:
    """
    Convert a TRMPH string to a list of (row, col) moves.
    
    Args:
        trmph_text: TRMPH string (can include preamble)
        board_size: Size of the board
        
    Returns:
        List of (row, col) tuples representing the moves
    """
    # Strip preamble and get moves
    bare_moves = strip_trmph_preamble(trmph_text)
    moves = split_trmph_moves(bare_moves)
    
    # Convert to (row, col) format
    rowcol_moves = []
    for move in moves:
        row, col = trmph_move_to_rowcol(move, board_size)
        rowcol_moves.append((row, col))
    
    return rowcol_moves

def parse_trmph_to_board(trmph_text: str, board_size: int = BOARD_SIZE) -> np.ndarray:
    """
    Parse a trmph string to a board matrix.
    
    Args:
        trmph_text: Complete trmph string
        board_size: Size of the board
        
    Returns:
        Board matrix with 'e'=empty, 'b'=blue, 'r'=red (character array)
        
    Raises:
        ValueError: If duplicate move found
    """
    # Strip preamble and get moves
    bare_moves = strip_trmph_preamble(trmph_text)
    moves = split_trmph_moves(bare_moves)
    
    # Initialize board with empty piece ('e')
    board = np.full((board_size, board_size), Piece.EMPTY.value, dtype='U1')
    
    # Place moves on board
    for i, move in enumerate(moves):
        row, col = trmph_move_to_rowcol(move, board_size)
        
        # Check for duplicate moves
        if board[row, col] != Piece.EMPTY.value:
            raise ValueError(f"Duplicate move '{move}' at ({row}, {col}) in {trmph_text}")
        
        # Place move (Alternating players. Piece colours are blue='b', red='r' for nxn boards)
        is_blue_turn = (i % 2) == 0
        # TODO: ENUM MIGRATION - This is a temporary fix to allow the board to be created with the Piece enum.
        #       In the future, we should use the Piece enum directly in the board creation.
        # TODO: Avoid this if / else style. Check for blue / red and raise an exception with any other value.
        board[row, col] = Piece.BLUE.value if is_blue_turn else Piece.RED.value
    
    return board


def rowcol_to_trmph(row: int, col: int, board_size: int = BOARD_SIZE) -> str:
    board_size = _validate_board_size(board_size)
    if not (0 <= row < board_size) or not (0 <= col < board_size):
        raise ValueError(f"Invalid coordinates: ({row}, {col}) for board size {board_size}")
    letter = LETTERS[col]
    number = str(row + 1)
    return letter + number

def tensor_to_rowcol(tensor_pos: int) -> Tuple[int, int]:
    return tensor_to_rowcol_with_size(tensor_pos, BOARD_SIZE)

def rowcol_to_tensor(row: int, col: int) -> int:
    return rowcol_to_tensor_with_size(row, col, BOARD_SIZE)

def rowcol_to_tensor_with_size(row: int, col: int, board_size: int) -> int:
    """Convert row, col coordinates to tensor index with specified board size."""
    board_size = _validate_board_size(board_size)
    if not (0 <= row < board_size) or not (0 <= col < board_size):
        raise ValueError(f"Invalid coordinates: ({row}, {col}) for board size {board_size}")
    return row * board_size + col

def tensor_to_rowcol_with_size(tensor_pos: int, board_size: int) -> Tuple[int, int]:
    """Convert tensor index to row, col coordinates with specified board size."""
    board_size = _validate_board_size(board_size)
    max_actions = board_size * board_size
    if not (0 <= tensor_pos < max_actions):
        raise ValueError(f"Invalid tensor position: {tensor_pos} for board size {board_size}")
    row = tensor_pos // board_size
    col = tensor_pos % board_size
    return row, col

def trmph_to_tensor(move: str, board_size: int = BOARD_SIZE) -> int:
    row, col = trmph_move_to_rowcol(move, board_size)
    return rowcol_to_tensor_with_size(row, col, board_size)

def tensor_to_trmph(tensor_pos: int, board_size: int = BOARD_SIZE) -> str:
    row, col = tensor_to_rowcol_with_size(tensor_pos, board_size)
    return rowcol_to_trmph(row, col, board_size)

# --- Board/Tensor Conversion Functions ---
def board_2nxn_to_nxn(board_2nxn: torch.Tensor) -> np.ndarray:
    """Convert 2×N×N tensor format to N×N array format."""
    if isinstance(board_2nxn, torch.Tensor):
        board_np = board_2nxn.detach().cpu().numpy()
    else:
        board_np = np.asarray(board_2nxn)
    if board_np.ndim != 3 or board_np.shape[0] != 2:
        raise ValueError(f"Expected shape (2, N, N), got {board_np.shape}")
    board_size_rows = int(board_np.shape[1])
    board_size_cols = int(board_np.shape[2])
    if board_size_rows <= 0 or board_size_cols <= 0:
        raise ValueError(f"Board dimensions must be positive, got {board_np.shape}")
    if board_size_rows != board_size_cols:
        raise ValueError(f"Expected square board shape, got {board_np.shape}")
    board_nxn = np.full((board_size_rows, board_size_cols), piece_to_char(Piece.EMPTY), dtype='U1')
    # Convert one-hot encoded channels to N×N format
    board_nxn[board_np[channel_to_int(Channel.BLUE)] == PIECE_ONEHOT] = piece_to_char(Piece.BLUE)
    board_nxn[board_np[channel_to_int(Channel.RED)] == PIECE_ONEHOT] = piece_to_char(Piece.RED)
    return board_nxn

def board_nxn_to_2nxn(board_nxn: np.ndarray) -> torch.Tensor:
    """Convert N×N array format to 2×N×N tensor format."""
    board_np = np.asarray(board_nxn)
    if board_np.ndim != 2:
        raise ValueError(f"Expected shape (N, N), got {board_np.shape}")
    board_size_rows = int(board_np.shape[0])
    board_size_cols = int(board_np.shape[1])
    if board_size_rows <= 0 or board_size_cols <= 0:
        raise ValueError(f"Board dimensions must be positive, got {board_np.shape}")
    if board_size_rows != board_size_cols:
        raise ValueError(f"Expected square board shape, got {board_np.shape}")
    board_2nxn = torch.zeros(2, board_size_rows, board_size_cols, dtype=torch.float32)
    # Convert N×N format to one-hot encoded channels
    board_2nxn[channel_to_int(Channel.BLUE)] = torch.from_numpy((board_np == piece_to_char(Piece.BLUE)).astype(np.float32))
    board_2nxn[channel_to_int(Channel.RED)] = torch.from_numpy((board_np == piece_to_char(Piece.RED)).astype(np.float32))
    return board_2nxn

def board_2nxn_to_3nxn(board_2nxn: torch.Tensor) -> torch.Tensor:
    """
    Convert a (2, N, N) board tensor to a (3, N, N) tensor by adding a player-to-move channel.
    The player-to-move channel is filled with Player.BLUE.value (0.0) or Player.RED.value (1.0) as float.
    Args:
        board_2nxn: torch.Tensor of shape (2, N, N)
    Returns:
        torch.Tensor of shape (3, N, N)
    """
    # TODO: Properly fix the circular import from putting the below at the top of this file.
    #       Decide which which of this and data_utils is the upstream dependency.
    from hex_ai.utils.player_utils import get_player_to_move_from_board

    if isinstance(board_2nxn, torch.Tensor):
        board_np = board_2nxn.detach().cpu().numpy()
    else:
        board_np = np.asarray(board_2nxn)
    if board_np.ndim != 3 or board_np.shape[0] != 2:
        raise ValueError(f"Expected shape (2, N, N), got {board_np.shape}")
    board_size_rows = int(board_np.shape[1])
    board_size_cols = int(board_np.shape[2])
    if board_size_rows <= 0 or board_size_cols <= 0:
        raise ValueError(f"Board dimensions must be positive, got {board_np.shape}")
    if board_size_rows != board_size_cols:
        raise ValueError(f"Expected square board shape, got {board_np.shape}")
    # For format conversion, we don't have error tracking context, so pass None
    # This will use the original behavior (raise exception for invalid boards)
    player_to_move = get_player_to_move_from_board(board_np, error_tracker=None)
    # Convert Player enum to integer for tensor creation
    player_to_move_int = player_to_int(player_to_move)
    player_channel = np.full((board_size_rows, board_size_cols), float(player_to_move_int), dtype=np.float32)
    # Add player-to-move channel as the third channel
    board_3ch = np.concatenate([board_np, player_channel[None, ...]], axis=0)
    return torch.from_numpy(board_3ch)

def board_nxn_to_3nxn(board_nxn: np.ndarray) -> torch.Tensor:
    """
    Convert a (N, N) board to a (3, N, N) tensor with player-to-move channel.
    Args:
        board_nxn: np.ndarray of shape (N, N)
    Returns:
        torch.Tensor of shape (3, N, N)
    """
    board_2nxn = board_nxn_to_2nxn(board_nxn)
    return board_2nxn_to_3nxn(board_2nxn)

def board_3nxn_to_nxn(board_3nxn: torch.Tensor) -> np.ndarray:
    """
    Convert a (3, N, N) tensor to a (N, N) string array format.
    Extracts the first two channels (blue and red) and converts to string representation.
    The third channel (player-to-move) is ignored.
    
    Args:
        board_3nxn: torch.Tensor of shape (3, N, N) or np.ndarray of shape (3, N, N)
        
    Returns:
        np.ndarray of shape (N, N) with 'e'=empty, 'b'=blue, 'r'=red
        
    Raises:
        ValueError: If input shape is not (3, N, N)
    """
    if isinstance(board_3nxn, torch.Tensor):
        board_np = board_3nxn.detach().cpu().numpy()
    else:
        board_np = np.asarray(board_3nxn)
    
    if board_np.ndim != 3 or board_np.shape[0] != 3:
        raise ValueError(f"Expected shape (3, N, N), got {board_np.shape}")
    board_size_rows = int(board_np.shape[1])
    board_size_cols = int(board_np.shape[2])
    if board_size_rows <= 0 or board_size_cols <= 0:
        raise ValueError(f"Board dimensions must be positive, got {board_np.shape}")
    if board_size_rows != board_size_cols:
        raise ValueError(f"Expected square board shape, got {board_np.shape}")
    
    # Extract blue and red channels
    blue_channel = board_np[channel_to_int(Channel.BLUE)]
    red_channel = board_np[channel_to_int(Channel.RED)]
    
    # Convert to N×N string format
    board_nxn = np.full((board_size_rows, board_size_cols), piece_to_char(Piece.EMPTY), dtype='U1')
    board_nxn[blue_channel == PIECE_ONEHOT] = piece_to_char(Piece.BLUE)
    board_nxn[red_channel == PIECE_ONEHOT] = piece_to_char(Piece.RED)
    
    return board_nxn

def normalize_game_input(text: str, board_size: int = BOARD_SIZE) -> str:
    if not text:
        return ""
        
    # Check for swap
    is_swap = "swap" in text.lower()
    
    # Clean up the text
    try:
        if text.startswith('#'):
             text = strip_trmph_preamble(text)
    except ValueError:
        pass

    # Remove copy-paste wrapper words and non-move markers
    clean_text = re.sub(r'\bmoves\b', '', text, flags=re.IGNORECASE)
    clean_text = re.sub(r'\bswap\b', '', clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'\bresign\b', '', clean_text, flags=re.IGNORECASE)
    
    # Remove move numbers (e.g. "1.", "10.")
    clean_text = re.sub(r'\b\d+\.', '', clean_text)
    
    # Remove all non-alphanumeric characters
    clean_text = re.sub(r'[^a-zA-Z0-9]', '', clean_text)
    
    # If it was a swap game, we need to transpose all moves EXCEPT the first one
    # The user wants the original move preserved, but the rest of the game reflected
    if is_swap:
        try:
            # Ensure lowercase for move processing
            moves = split_trmph_moves(clean_text.lower())
            processed_moves = []
            
            # Handle first move (keep original)
            if moves:
                processed_moves.append(moves[0])
                
            # Handle subsequent moves (transpose)
            for move in moves[1:]:
                row, col = trmph_move_to_rowcol(move, board_size)
                # Transpose: swap row and col
                processed_moves.append(rowcol_to_trmph(col, row, board_size))
                
            return "".join(processed_moves)
        except ValueError as e:
            # If parsing fails, return original cleaned text and let validation handle it
            logger.warning(f"Failed to transpose swap moves: {e}")
            
    # Convert to lowercase to ensure consistency with TRMPH format (e.g. a3, not A3)
    return clean_text.lower()
