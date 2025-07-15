"""
Format conversion utilities for Hex AI.

This module centralizes all board, trmph, and tensor conversion functions.

Migrated from:
- hex_ai/data_utils.py
- hex_ai/inference/board_utils.py
"""

import torch
import numpy as np
from typing import Tuple
from hex_ai.config import BOARD_SIZE
import string

# Board value constants for N×N format
EMPTY = 0
BLUE = 1
RED = 2

# Player constants
BLUE_PLAYER = 0
RED_PLAYER = 1

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

def parse_trmph_to_board(trmph_text: str, board_size: int = BOARD_SIZE, debug_info: str = "") -> np.ndarray:
    bare_moves = strip_trmph_preamble(trmph_text)
    moves = split_trmph_moves(bare_moves)
    board = np.zeros((board_size, board_size), dtype=np.int8)
    for i, move in enumerate(moves):
        row, col = trmph_move_to_rowcol(move, board_size)
        if board[row, col] != 0:
            if "training" in debug_info.lower() or "game" in debug_info.lower():
                continue
            else:
                raise ValueError(f"Duplicate move '{move}' at {(row, col)} in {trmph_text}")
        player = (i % 2) + 1
        board[row, col] = player
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
    board_nxn[board_2nxn[0] == 1.0] = BLUE
    board_nxn[board_2nxn[1] == 1.0] = RED
    return board_nxn

def board_nxn_to_2nxn(board_nxn: np.ndarray) -> torch.Tensor:
    if board_nxn.shape != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"Expected shape ({BOARD_SIZE}, {BOARD_SIZE}), got {board_nxn.shape}")
    board_2nxn = torch.zeros(2, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
    board_2nxn[0] = torch.from_numpy((board_nxn == BLUE).astype(np.float32))
    board_2nxn[1] = torch.from_numpy((board_nxn == RED).astype(np.float32))
    return board_2nxn 