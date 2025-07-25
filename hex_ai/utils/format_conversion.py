"""
Format conversion utilities for Hex AI.

NOTE: Data augmentation and .pkl.gz file processing is performed using the 
      2-channel (blue/red) format. The player-to-move (3rd) channel is added 
      at load time for training and inference, ensuring consistency with the 
      model input format.

This module centralizes all board, trmph, and tensor conversion functions.

Migrated from:
- hex_ai/data_utils.py
- hex_ai/inference/board_utils.py
"""

import torch
import numpy as np
from typing import Tuple
# Board value constants for N×N format
from hex_ai.config import (
    BOARD_SIZE, BLUE_PLAYER, RED_PLAYER, BLUE_PIECE, RED_PIECE, EMPTY_PIECE,
    PIECE_ONEHOT, EMPTY_ONEHOT, BLUE_CHANNEL, RED_CHANNEL, PLAYER_CHANNEL,
    TRMPH_BLUE_WIN, TRMPH_RED_WIN,
)
from hex_ai.value_utils import trmph_winner_to_training_value, trmph_winner_to_clear_str

import string
import logging
logger = logging.getLogger(__name__)

LETTERS = string.ascii_lowercase

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

def parse_trmph_to_board(trmph_text: str, board_size: int = BOARD_SIZE, duplicate_action: str = "exception") -> np.ndarray:
    """
    Parse a trmph string to a board matrix.
    
    Args:
        trmph_text: Complete trmph string
        board_size: Size of the board
        duplicate_action: How to handle duplicate moves ("exception" or "ignore")
        
    Returns:
        Board matrix with 0=empty, 1=blue, 2=red
        
    Raises:
        ValueError: If duplicate_action="exception" and duplicate move found
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
        if board[row, col] != EMPTY_PIECE:
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
                raise ValueError(f"Duplicate move '{move}' at {(row, col)} in {trmph_text}")
        
        # Place move (Alternating players. Piece colours are blue=1, red=2 for nxn boards)
        player = BLUE_PLAYER if (i % 2) == 0 else RED_PLAYER
        board[row, col] = BLUE_PIECE if player == BLUE_PLAYER else RED_PIECE
    
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

def tensor_to_trmph(tensor_pos: int, board_size: int = BOARD_SIZE) -> str:
    row, col = tensor_to_rowcol(tensor_pos)
    return rowcol_to_trmph(row, col, board_size)

def trmph_to_tensor(move: str, board_size: int = BOARD_SIZE) -> int:
    row, col = trmph_move_to_rowcol(move, board_size)
    return rowcol_to_tensor(row, col)

# --- Board/Tensor Conversion Functions (from board_utils.py) ---
def board_2nxn_to_nxn(board_2nxn: torch.Tensor) -> np.ndarray:
    if board_2nxn.shape != (2, BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"Expected shape (2, {BOARD_SIZE}, {BOARD_SIZE}), got {board_2nxn.shape}")
    board_nxn = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    # Convert one-hot encoded channels to N×N format
    board_nxn[board_2nxn[BLUE_CHANNEL] == PIECE_ONEHOT] = BLUE_PIECE
    board_nxn[board_2nxn[RED_CHANNEL] == PIECE_ONEHOT] = RED_PIECE
    return board_nxn

def board_nxn_to_2nxn(board_nxn: np.ndarray) -> torch.Tensor:
    if board_nxn.shape != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"Expected shape ({BOARD_SIZE}, {BOARD_SIZE}), got {board_nxn.shape}")
    board_2nxn = torch.zeros(2, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
    # Convert N×N format to one-hot encoded channels
    board_2nxn[BLUE_CHANNEL] = torch.from_numpy((board_nxn == BLUE_PIECE).astype(np.float32))
    board_2nxn[RED_CHANNEL] = torch.from_numpy((board_nxn == RED_PIECE).astype(np.float32))
    return board_2nxn

def board_2nxn_to_3nxn(board_2nxn: torch.Tensor) -> torch.Tensor:
    """
    Convert a (2, N, N) board tensor to a (3, N, N) tensor by adding a player-to-move channel.
    The player-to-move channel is filled with BLUE_PLAYER (0.0) or RED_PLAYER (1.0) as float.
    Args:
        board_2nxn: torch.Tensor of shape (2, N, N)
    Returns:
        torch.Tensor of shape (3, N, N)
    """
    import numpy as np
    from hex_ai.data_utils import get_player_to_move_from_board
    if isinstance(board_2nxn, torch.Tensor):
        board_np = board_2nxn.detach().cpu().numpy()
    else:
        board_np = board_2nxn
    # For format conversion, we don't have error tracking context, so pass None
    # This will use the original behavior (raise exception for invalid boards)
    player_to_move = get_player_to_move_from_board(board_np, error_tracker=None)
    player_channel = np.full((BOARD_SIZE, BOARD_SIZE), float(player_to_move), dtype=np.float32)
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
    Raises ValueError if the format is invalid.
    """
    line = line.strip()
    if not line:
        raise ValueError("Empty line")
    parts = line.split()
    if len(parts) != 2:
        raise ValueError(f"Invalid TRMPH game record format: {repr(line)}")
    trmph_url, winner_indicator = parts
    if not winner_indicator.isdigit() or winner_indicator not in {TRMPH_BLUE_WIN, TRMPH_RED_WIN}:
        raise ValueError(f"Invalid winner indicator: {winner_indicator} in line: {repr(line)}")
    return trmph_url, winner_indicator 