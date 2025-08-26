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
logger = logging.getLogger(__name__)

LETTERS = string.ascii_lowercase

# TODO: ENUM MIGRATION - This module now uses Piece/Channel internally and keeps
# boundary types (TRMPH strings, numpy/tensors) for IO. Continue migrating
# upstream/downstream code to pass Enums in domain logic and convert only at boundaries.

# --- TRMPH/Move Conversion Functions (from data_utils.py) ---
def strip_trmph_preamble(trmph_text: str) -> str:
    match = __import__('re').compile(r"#(\d+),").search(trmph_text)
    if not match:
        raise ValueError(f"No board preamble found in trmph string: {trmph_text}")
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

def trmph_move_to_rowcol(move: str, board_size: int = BOARD_SIZE) -> tuple[int, int]:
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

def parse_trmph_to_board(trmph_text: str, board_size: int = BOARD_SIZE, duplicate_action: str = "exception") -> np.ndarray:
    """
    Parse a trmph string to a board matrix.
    
    Args:
        trmph_text: Complete trmph string
        board_size: Size of the board
        duplicate_action: How to handle duplicate moves ("exception" or "ignore")
        
    Returns:
        Board matrix with 'e'=empty, 'b'=blue, 'r'=red (character array)
        
    Raises:
        ValueError: If duplicate_action="exception" and duplicate move found
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
            if duplicate_action == "ignore":
                logger.warning(f"Skipping duplicate move '{move}' at {(row, col)} in {trmph_text}")
                break  # Do not process any moves after a duplicate.
            else:
                # Enhanced debugging output for non-training contexts
                import traceback
                frame = traceback.extract_stack()[-2]  # Get calling frame
                logger.error(f"DUPLICATE MOVE DETECTED:")
                logger.error(f"  File: {frame.filename}")
                logger.error(f"  Line: {frame.lineno}")
                logger.error(f"  Function: {frame.name}")
                logger.error(f"  Move: '{move}' at position ({row}, {col})")
                logger.error(f"  Move index: {i}")
                logger.error(f"  Board value at position: {board[row, col]}")
                logger.error(f"  Full trmph string: {trmph_text}")
                logger.error(f"  All moves: {moves}")
                raise ValueError(f"Duplicate move '{move}' at ({row}, {col}) in {trmph_text}")
        
        # Place move (Alternating players. Piece colours are blue='b', red='r' for nxn boards)
        is_blue_turn = (i % 2) == 0
        # TODO: ENUM MIGRATION - This is a temporary fix to allow the board to be created with the Piece enum.
        #       In the future, we should use the Piece enum directly in the board creation.
        # TODO: Avoid this if / else style. Check for blue / red and raise an exception with any other value.
        board[row, col] = Piece.BLUE.value if is_blue_turn else Piece.RED.value
    
    return board


def rowcol_to_trmph(row: int, col: int, board_size: int = BOARD_SIZE) -> str:
    if not (0 <= row < board_size) or not (0 <= col < board_size):
        raise ValueError(f"Invalid coordinates: ({row}, {col}) for board size {board_size}")
    letter = LETTERS[col]
    number = str(row + 1)
    return letter + number

def tensor_to_rowcol(tensor_pos: int) -> Tuple[int, int]:
    if not (0 <= tensor_pos < BOARD_SIZE * BOARD_SIZE):
        raise ValueError(f"Invalid tensor position: {tensor_pos}")
    row = tensor_pos // BOARD_SIZE
    col = tensor_pos % BOARD_SIZE
    return row, col

def rowcol_to_tensor(row: int, col: int) -> int:
    if not (0 <= row < BOARD_SIZE) or not (0 <= col < BOARD_SIZE):
        raise ValueError(f"Invalid coordinates: ({row}, {col}) for board size {BOARD_SIZE}")
    return row * BOARD_SIZE + col

def trmph_to_tensor(move: str, board_size: int = BOARD_SIZE) -> int:
    row, col = trmph_move_to_rowcol(move, board_size)
    return rowcol_to_tensor(row, col)

def tensor_to_trmph(tensor_pos: int, board_size: int = BOARD_SIZE) -> str:
    row, col = tensor_to_rowcol(tensor_pos)
    return rowcol_to_trmph(row, col, board_size)

# --- Board/Tensor Conversion Functions ---
def board_2nxn_to_nxn(board_2nxn: torch.Tensor) -> np.ndarray:
    """Convert 2×N×N tensor format to N×N array format."""
    if board_2nxn.shape != (2, BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"Expected shape (2, {BOARD_SIZE}, {BOARD_SIZE}), got {board_2nxn.shape}")
    board_nxn = np.full((BOARD_SIZE, BOARD_SIZE), piece_to_char(Piece.EMPTY), dtype='U1')
    # Convert one-hot encoded channels to N×N format
    board_nxn[board_2nxn[channel_to_int(Channel.BLUE)] == PIECE_ONEHOT] = piece_to_char(Piece.BLUE)
    board_nxn[board_2nxn[channel_to_int(Channel.RED)] == PIECE_ONEHOT] = piece_to_char(Piece.RED)
    return board_nxn

def board_nxn_to_2nxn(board_nxn: np.ndarray) -> torch.Tensor:
    """Convert N×N array format to 2×N×N tensor format."""
    if board_nxn.shape != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"Expected shape ({BOARD_SIZE}, {BOARD_SIZE}), got {board_nxn.shape}")
    board_2nxn = torch.zeros(2, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
    # Convert N×N format to one-hot encoded channels
    board_2nxn[channel_to_int(Channel.BLUE)] = torch.from_numpy((board_nxn == piece_to_char(Piece.BLUE)).astype(np.float32))
    board_2nxn[channel_to_int(Channel.RED)] = torch.from_numpy((board_nxn == piece_to_char(Piece.RED)).astype(np.float32))
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
        board_np = board_2nxn
    # For format conversion, we don't have error tracking context, so pass None
    # This will use the original behavior (raise exception for invalid boards)
    player_to_move = get_player_to_move_from_board(board_np, error_tracker=None)
    # Convert Player enum to integer for tensor creation
    player_to_move_int = player_to_int(player_to_move)
    player_channel = np.full((BOARD_SIZE, BOARD_SIZE), float(player_to_move_int), dtype=np.float32)
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

def parse_trmph_game_record(line: str) -> tuple[str, str]:
    """
    Parse a single line from a TRMPH file, returning (trmph_url, winner_indicator).
    Raises ValueError if the format is invalid or if legacy formats are detected.
    """
    line = line.strip()
    if not line:
        raise ValueError("Empty line")
    parts = line.split()
    if len(parts) != 2:
        raise ValueError(f"Invalid TRMPH game record format: {repr(line)}")
    trmph_url, winner_indicator = parts
    
    # Check for legacy formats and raise exceptions
    if winner_indicator == "1":
        raise ValueError(f"Legacy TRMPH_BLUE_WIN value ('1') detected in line: {repr(line)}. Use new format ('b') instead.")
    elif winner_indicator == "2":
        raise ValueError(f"Legacy TRMPH_RED_WIN value ('2') detected in line: {repr(line)}. Use new format ('r') instead.")
    
    if winner_indicator not in {TRMPH_BLUE_WIN, TRMPH_RED_WIN}:
        raise ValueError(f"Invalid winner indicator: {winner_indicator} in line: {repr(line)}")
    return trmph_url, winner_indicator 