"""
Game engine for Hex AI.

This module provides the core game logic for Hex, including board representation,
move validation, winner detection, and game state management.

The winner detection uses an efficient array-based Union-Find (DSU) implementation
that provides O(1) connectivity checks and incremental updates, making it suitable
for high-frequency MCTS tree traversal.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from hex_ai.config import BOARD_SIZE, VERBOSE_LEVEL
from hex_ai.enums import Piece as PieceEnum, Player, Winner as WinnerEnum
from hex_ai.inference.board_utils import board_to_string, is_empty, place_piece
from hex_ai.utils.format_conversion import (
    board_nxn_to_2nxn, board_nxn_to_3nxn, rowcol_to_trmph, split_trmph_moves, trmph_move_to_rowcol
)
from hex_ai.value_utils import apply_move_to_tensor, get_top_k_legal_moves, sample_move_by_value, get_player_to_move_from_moves


# Edge coordinates for Union-Find
LEFT_EDGE = -1
RIGHT_EDGE = BOARD_SIZE
TOP_EDGE = -1
BOTTOM_EDGE = BOARD_SIZE

# Hex neighbor directions (same for all positions)
HEX_NEIGHBOR_DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (-1, 1)]

# Edge indices for DSU connectivity
EDGE_INDICES = {
    'red_left': BOARD_SIZE * BOARD_SIZE,
    'red_right': BOARD_SIZE * BOARD_SIZE + 1,
    'blue_top': BOARD_SIZE * BOARD_SIZE,
    'blue_bottom': BOARD_SIZE * BOARD_SIZE + 1
}

# Precomputed neighbor lookup table for efficient connectivity
# Each cell (r, c) maps to a list of up to 6 neighbor indices
NEIGHBORS: List[List[int]] = []

def _get_adjacent_positions(row: int, col: int) -> List[Tuple[int, int]]:
    """Get adjacent positions for a given (row, col) coordinate."""
    adjacent = []
    for dr, dc in HEX_NEIGHBOR_DIRECTIONS:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
            adjacent.append((new_row, new_col))
    return adjacent

def _initialize_neighbors():
    """Initialize the global NEIGHBORS lookup table using shared logic."""
    global NEIGHBORS
    NEIGHBORS = []
    
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            neighbors = []
            for nr, nc in _get_adjacent_positions(r, c):
                neighbors.append(nr * BOARD_SIZE + nc)
            NEIGHBORS.append(neighbors)

# Initialize the neighbor lookup table
_initialize_neighbors()

# Performance instrumentation
INIT_REBUILDS = 0

def rowcol_to_index(r: int, c: int) -> int:
    """Convert (row, col) to DSU index."""
    return r * BOARD_SIZE + c

def index_to_rowcol(idx: int) -> Tuple[int, int]:
    """Convert DSU index to (row, col)."""
    return idx // BOARD_SIZE, idx % BOARD_SIZE

def _get_edge_connections(row: int, col: int, piece_color: str) -> List[int]:
    """Get edge indices that a piece at (row, col) should connect to."""
    edge_connections = []
    
    if piece_color == PieceEnum.RED.value:
        if col == 0:  # Left edge
            edge_connections.append(EDGE_INDICES['red_left'])
        if col == BOARD_SIZE - 1:  # Right edge
            edge_connections.append(EDGE_INDICES['red_right'])
    else:  # BLUE
        if row == 0:  # Top edge
            edge_connections.append(EDGE_INDICES['blue_top'])
        if row == BOARD_SIZE - 1:  # Bottom edge
            edge_connections.append(EDGE_INDICES['blue_bottom'])
    
    return edge_connections

def _create_dsu_pair() -> Tuple['ArrayDSU', 'ArrayDSU']:
    """Create a pair of DSUs for red and blue connectivity tracking."""
    dsu_size = BOARD_SIZE * BOARD_SIZE + 2  # +2 for edge nodes
    return ArrayDSU(dsu_size), ArrayDSU(dsu_size)

# Efficient array-based Union-Find data structure for Hex connectivity
class ArrayDSU:
    """
    Array-based Union-Find data structure optimized for Hex game connectivity.
    
    Uses integer indices instead of namedtuples for maximum performance.
    Supports both regular operations and rollback for MCTS tree traversal.
    """
    
    def __init__(self, size: int):
        """
        Initialize DSU for given size.
        
        Args:
            size: Total number of elements (board cells + edge nodes)
        """
        self.size = size
        self.parent = list(range(size))
        self.size_array = [1] * size
        
    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """
        Union two sets by size.
        
        Returns:
            True if union was performed, False if already connected
        """
        x_root = self.find(x)
        y_root = self.find(y)
        
        if x_root == y_root:
            return False
        
        # Union by size
        if self.size_array[x_root] < self.size_array[y_root]:
            x_root, y_root = y_root, x_root
        
        self.parent[y_root] = x_root
        self.size_array[x_root] += self.size_array[y_root]
        return True
    
    def connected(self, x: int, y: int) -> bool:
        """Check if two elements are connected."""
        return self.find(x) == self.find(y)
    
    def copy(self) -> 'ArrayDSU':
        """Create a copy of this DSU."""
        new_dsu = ArrayDSU(self.size)
        new_dsu.parent = self.parent.copy()
        new_dsu.size_array = self.size_array.copy()
        return new_dsu


class DSURollback:
    """
    Union-Find with rollback capability for MCTS tree traversal.
    
    This allows efficient apply/undo operations by maintaining a history
    of all changes that can be reverted.
    """
    
    def __init__(self, size: int):
        self.size = size
        self.parent = list(range(size))
        self.size_array = [1] * size
        self.history: List[Tuple[int, int, int, int]] = []  # (child, old_parent, new_parent, old_size)
    
    def find(self, x: int) -> int:
        """Find root without path compression (for rollback compatibility)."""
        while self.parent[x] != x:
            x = self.parent[x]
        return x
    
    def union(self, x: int, y: int) -> bool:
        """
        Union two sets and record the operation for rollback.
        
        Returns:
            True if union was performed, False if already connected
        """
        x_root = self.find(x)
        y_root = self.find(y)
        
        if x_root == y_root:
            return False
        
        # Union by size
        if self.size_array[x_root] < self.size_array[y_root]:
            x_root, y_root = y_root, x_root
        
        # Record the operation for rollback
        self.history.append((y_root, y_root, x_root, self.size_array[x_root]))
        
        self.parent[y_root] = x_root
        self.size_array[x_root] += self.size_array[y_root]
        return True
    
    def connected(self, x: int, y: int) -> bool:
        """Check if two elements are connected."""
        return self.find(x) == self.find(y)
    
    def snapshot(self) -> int:
        """Create a snapshot and return the current history length."""
        return len(self.history)
    
    def rollback_to(self, snapshot: int) -> None:
        """Rollback to a previous snapshot."""
        while len(self.history) > snapshot:
            child, old_parent, new_parent, old_size = self.history.pop()
            self.parent[child] = old_parent
            self.size_array[new_parent] = old_size




@dataclass(init=False)
class HexGameState:
    """
    Represents the state of a Hex game using N×N character format.
    """
    board: np.ndarray = field(default_factory=lambda: np.full((BOARD_SIZE, BOARD_SIZE), PieceEnum.EMPTY.value, dtype='U1'))
    _current_player: Player = field(init=False, repr=False)  # No default - must be set in __init__
    move_history: List[Tuple[int, int]] = field(default_factory=list)
    game_over: bool = False
    _winner: Optional[WinnerEnum] = field(default=None, repr=False)
    
    # Operation stack for undo functionality
    _undo_stack: List[Dict[str, Any]] = field(default_factory=list, init=False)
    
    # Efficient connectivity tracking
    red_dsu: ArrayDSU = field(init=False, repr=False)
    blue_dsu: ArrayDSU = field(init=False, repr=False)

    def __init__(self, _current_player: Player, board: Optional[np.ndarray] = None,
                 move_history: Optional[List[Tuple[int, int]]] = None,
                 game_over: bool = False, winner: Optional[WinnerEnum] = None,
                 *, skip_initial_connectivity: bool = False):
        # Initialize board
        self.board = board if board is not None else np.full((BOARD_SIZE, BOARD_SIZE), PieceEnum.EMPTY.value, dtype='U1')
        # Initialize current player with compatibility for legacy int
        if not isinstance(_current_player, Player):
            raise TypeError(f"_current_player must be Player, got {type(_current_player)}")
        self._current_player = _current_player
        # Initialize other fields
        self.move_history = move_history.copy() if move_history is not None else []
        self.game_over = game_over
        # Winner internal storage uses Enum only
        self._winner = winner
        self._undo_stack = []
        
        # Initialize DSUs for connectivity tracking
        self.red_dsu, self.blue_dsu = _create_dsu_pair()
        
        # Build initial connectivity from board state (unless skipped for performance)
        if not skip_initial_connectivity:
            self._build_initial_connectivity()
    
    @property
    def board_2nxn(self) -> torch.Tensor:
        """Get board in 2×N×N format for compatibility with tests (legacy, do not use for inference)."""
        return board_nxn_to_2nxn(self.board)

    @property
    def current_player(self) -> int:
        """Expose current player as legacy int (0/1) for compatibility."""
        return int(self._current_player.value)

    @current_player.setter
    def current_player(self, value) -> None:
        """Set current player using Player enum only; no implicit int conversions."""
        if not isinstance(value, Player):
            raise TypeError(f"current_player must be Player, got {type(value)}")
        self._current_player = value

    @property
    def current_player_enum(self) -> Player:
        """Preferred: expose current player as Player enum."""
        return self._current_player

    @property
    def winner(self) -> Optional[WinnerEnum]:
        """Winner as Winner enum or None. Internally stores Winner enum."""
        return self._winner

    @winner.setter
    def winner(self, value: Optional[WinnerEnum]) -> None:
        """Accept Winner enum or None; fail fast on invalid inputs."""
        if value is None:
            self._winner = None
            return
        if isinstance(value, WinnerEnum):
            self._winner = value
            return
        raise TypeError(f"winner must be Winner enum or None; got {type(value)}")

    @property
    def winner_enum(self) -> Optional[WinnerEnum]:
        """Preferred: expose winner as Winner enum (or None if game not over)."""
        return self._winner

    def is_valid_move(self, row: int, col: int) -> bool:
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return False
        if self.game_over:
            return False
        return is_empty(self.board, row, col)

    def apply_move(self, row: int, col: int) -> None:
        """
        Apply a move in-place to the current state.
        
        This method mutates the current state instead of creating a new one,
        which is much more efficient for MCTS tree traversal.
        
        Args:
            row: Row index (0-12)
            col: Column index (0-12)
            
        Raises:
            ValueError: If move is invalid
        """
        if not self.is_valid_move(row, col):
            raise ValueError(f"Invalid move: ({row}, {col})")
        
        # Save state for undo
        undo_info = {
            'board': self.board.copy(),
            # Store enum, not legacy int, to satisfy strict setter on undo
            'current_player': self._current_player,
            'game_over': self.game_over,
            'winner': self.winner,
            'move': (row, col),
            'red_dsu': self.red_dsu.copy(),
            'blue_dsu': self.blue_dsu.copy()
        }
        self._undo_stack.append(undo_info)
        
        # Apply the move
        piece = PieceEnum.BLUE if self._current_player == Player.BLUE else PieceEnum.RED
        self.board = place_piece(self.board, row, col, piece)
        self.move_history.append((row, col))
        self._current_player = Player.RED if self._current_player == Player.BLUE else Player.BLUE
        
        # Connect only the new piece to its neighbors (optimized)
        self._connect_new_piece_only(row, col, piece)
        
        # Check for winner using efficient connectivity
        winner = self._check_winner_efficient()
        if winner:
            self.game_over = True
            self.winner = winner

    def undo_last(self) -> None:
        """
        Undo the last move applied to this state.
        
        This restores the state to before the last apply_move() call.
        
        Raises:
            ValueError: If no moves have been applied
        """
        if not self._undo_stack:
            raise ValueError("No moves to undo")
        
        # Restore state from undo stack
        undo_info = self._undo_stack.pop()
        self.board = undo_info['board']
        self.current_player = undo_info['current_player']
        self.game_over = undo_info['game_over']
        self.winner = undo_info['winner']
        self.red_dsu = undo_info['red_dsu']
        self.blue_dsu = undo_info['blue_dsu']
        
        # Remove the last move from history
        if self.move_history:
            self.move_history.pop()

    def fast_copy(self) -> 'HexGameState':
        """
        Create a fast copy of the current state.
        
        This is more efficient than deepcopy for MCTS when we need
        to materialize child nodes.
        
        Returns:
            A new HexGameState with the same state
        """
        new_state = HexGameState(
            board=self.board.copy(),
            _current_player=self._current_player,
            move_history=self.move_history.copy(),
            game_over=self.game_over,
            winner=self.winner
        )
        # Copy DSUs for connectivity tracking
        new_state.red_dsu = self.red_dsu.copy()
        new_state.blue_dsu = self.blue_dsu.copy()
        # Don't copy the undo stack - it's not needed for new states
        return new_state

    def _build_initial_connectivity(self) -> None:
        """Build initial connectivity from the current board state."""
        global INIT_REBUILDS
        INIT_REBUILDS += 1
        
        # Connect all existing pieces to their neighbors
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r, c] != PieceEnum.EMPTY.value:
                    self._connect_piece_to_neighbors(r, c)
    
    def _connect_piece_to_neighbors(self, row: int, col: int) -> None:
        """Connect a piece to its same-color neighbors using efficient DSU."""
        piece_idx = rowcol_to_index(row, col)
        piece_color = self.board[row, col]
        
        # Choose the appropriate DSU
        dsu = self.red_dsu if piece_color == PieceEnum.RED.value else self.blue_dsu
        
        # Connect to same-color neighbors
        for neighbor_idx in NEIGHBORS[piece_idx]:
            nr, nc = index_to_rowcol(neighbor_idx)
            if self.board[nr, nc] == piece_color:
                dsu.union(piece_idx, neighbor_idx)
        
        # Connect to edges if on board edge
        for edge_idx in _get_edge_connections(row, col, piece_color):
            dsu.union(piece_idx, edge_idx)
    
    def _connect_new_piece_only(self, row: int, col: int, piece: PieceEnum) -> None:
        """
        Connect only the new piece to its neighbors and edges.
        
        This is the optimized version that only connects the newly placed piece,
        avoiding the expensive full board scan.
        
        Args:
            row: Row index of the new piece
            col: Column index of the new piece  
            piece: The piece that was placed
        """
        piece_idx = rowcol_to_index(row, col)
        piece_color = piece.value
        
        # Choose the appropriate DSU
        dsu = self.red_dsu if piece_color == PieceEnum.RED.value else self.blue_dsu
        
        # Connect to same-color neighbors
        for neighbor_idx in NEIGHBORS[piece_idx]:
            nr, nc = index_to_rowcol(neighbor_idx)
            if self.board[nr, nc] == piece_color:
                dsu.union(piece_idx, neighbor_idx)
        
        # Connect to edges if on board edge
        for edge_idx in _get_edge_connections(row, col, piece_color):
            dsu.union(piece_idx, edge_idx)
    
    def _check_winner_efficient(self) -> Optional[WinnerEnum]:
        """Check for winner using efficient DSU connectivity."""
        # Check red win (horizontal connection)
        if self.red_dsu.connected(EDGE_INDICES['red_left'], EDGE_INDICES['red_right']):
            return WinnerEnum.RED
        
        # Check blue win (vertical connection)
        if self.blue_dsu.connected(EDGE_INDICES['blue_top'], EDGE_INDICES['blue_bottom']):
            return WinnerEnum.BLUE
        
        return None

    def fast_copy(self) -> 'HexGameState':
        """Create a fast copy of the state without rebuilding connectivity."""
        new_state = HexGameState(
            _current_player=self._current_player,
            board=self.board.copy(),
            move_history=self.move_history.copy(),
            game_over=self.game_over,
            winner=self._winner,
            skip_initial_connectivity=True  # Skip the expensive rebuild
        )
        # Copy DSUs from parent state
        new_state.red_dsu = self.red_dsu.copy()
        new_state.blue_dsu = self.blue_dsu.copy()
        return new_state

    def _create_child_state(self, row: int, col: int) -> 'HexGameState':
        """Create a child state with the given move applied using fast copy."""
        piece = PieceEnum.BLUE if self._current_player == Player.BLUE else PieceEnum.RED
        new_board = place_piece(self.board, row, col, piece)
        new_move_history = self.move_history + [(row, col)]
        
        # Create new state using fast copy to avoid connectivity rebuild
        new_state = HexGameState(
            _current_player=Player.RED if self._current_player == Player.BLUE else Player.BLUE,
            board=new_board,
            move_history=new_move_history,
            game_over=False,
            winner=None,
            skip_initial_connectivity=True  # Skip the expensive rebuild
        )
        
        # Copy DSUs from parent state
        new_state.red_dsu = self.red_dsu.copy()
        new_state.blue_dsu = self.blue_dsu.copy()
        
        return new_state

    def make_move(self, row: int, col: int) -> 'HexGameState':
        """Create a new state with the move applied using efficient DSU copy."""
        if not self.is_valid_move(row, col):
            raise ValueError(f"Invalid move: ({row}, {col})")
        
        # Create child state with the move applied (no connectivity rebuild)
        new_state = self._create_child_state(row, col)
        
        # Connect only the new piece to its neighbors (optimized)
        piece = PieceEnum.BLUE if self._current_player == Player.BLUE else PieceEnum.RED
        new_state._connect_new_piece_only(row, col, piece)
        
        # Check for winner using efficient connectivity
        winner = new_state._check_winner_efficient()
        if winner:
            new_state.game_over = True
            new_state.winner = winner
        
        return new_state



    def get_legal_moves(self) -> List[Tuple[int, int]]:
        return [(row, col)
                for row in range(BOARD_SIZE)
                for col in range(BOARD_SIZE)
                if self.is_valid_move(row, col)]

    def get_board_tensor(self) -> torch.Tensor:
        """Convert board to 3×N×N tensor for neural net input (with player-to-move channel)."""
        return board_nxn_to_3nxn(self.board)

    def to_trmph(self) -> str:
        """Convert game state to TRMPH format."""
        moves_str = ""
        for row, col in self.move_history:
            trmph_move = rowcol_to_trmph(row, col)
            moves_str += trmph_move
        
        return f"#13,{moves_str}"
    
    @classmethod
    def from_trmph(cls, trmph: str) -> 'HexGameState':
        """Create game state from TRMPH format using split_trmph_moves utility."""
        if not trmph.startswith("#13,"):
            raise ValueError("Invalid TRMPH format")
        moves_str = trmph[4:]  # Remove "#13," prefix
        moves = split_trmph_moves(moves_str)
        
        # First create an EMPTY BOARD, then add moves to it.
        # Blue always starts first, so we always begin with Player.BLUE
        # The moves will be applied in sequence (blue, red, blue, red, ...)
        state = make_empty_hex_state()
        
        # Apply all moves
        for move in moves:
            row, col = trmph_move_to_rowcol(move)
            state = state.make_move(row, col)
        return state

    def __str__(self) -> str:
        status = f"Player {'Blue' if self._current_player == Player.BLUE else 'Red'}'s turn"
        if self.game_over:
            winner_name = str(self.winner)
            status = f"Game over - {winner_name} wins!"
        return f"HexGameState(moves={len(self.move_history)}, {status})\n" + board_to_string(self.board)

    def __repr__(self) -> str:
        return self.__str__()


class HexGameEngine:
    """
    Game engine for Hex games.
    
    This class provides a high-level interface for game management,
    move validation, and state transitions.
    """
    
    def __init__(self, board_size: int = BOARD_SIZE):
        self.board_size = board_size
    
    def reset(self) -> HexGameState:
        """Start a new game."""
        return make_empty_hex_state()
    
    def make_move(self, state: HexGameState, row: int, col: int) -> HexGameState:
        """
        Make a move and return new state.
        
        Args:
            state: Current game state
            row: Row index (0-12)
            col: Column index (0-12)
            
        Returns:
            New game state after the move
        """
        return state.make_move(row, col)
    
    def is_valid_move(self, state: HexGameState, row: int, col: int) -> bool:
        """
        Check if a move is valid.
        
        Args:
            state: Current game state
            row: Row index (0-12)
            col: Column index (0-12)
            
        Returns:
            True if the move is valid
        """
        return state.is_valid_move(row, col)
    
    def get_legal_moves(self, state: HexGameState) -> List[Tuple[int, int]]:
        """
        Get all legal moves for a state.
        
        Args:
            state: Current game state
            
        Returns:
            List of (row, col) tuples for legal moves
        """
        return state.get_legal_moves()
    
    def get_winner(self, state: HexGameState) -> Optional[WinnerEnum]:
        """
        Get the winner of the game.
        
        Args:
            state: Current game state
            
        Returns:
            Winner enum or None if no winner
        """
        return state._check_winner_efficient()
    
    def is_game_over(self, state: HexGameState) -> bool:
        """
        Check if the game is over.
        
        Args:
            state: Current game state
            
        Returns:
            True if the game is over
        """
        return state.game_over 

def apply_move_to_state(state, row: int, col: int) -> 'HexGameState':
    """
    Apply a move to a HexGameState and return the new state.
    
    This is the primary function for applying moves to game states.
    It handles the move validation and state updates.
    
    Args:
        state: HexGameState instance
        row: Row index (0-12)
        col: Column index (0-12)
        
    Returns:
        New HexGameState with the move applied
        
    Raises:
        ValueError: If move is invalid
    """
    if not state.is_valid_move(row, col):
        raise ValueError(f"Invalid move: ({row}, {col})")
    
    # Use the existing make_move method which handles all the game logic
    return state.make_move(row, col)


def apply_move_to_state_trmph(state, trmph_move: str) -> 'HexGameState':
    """
    Apply a TRMPH move to a HexGameState and return the new state.
    
    This is a wrapper that converts TRMPH to row,col coordinates.
    
    Args:
        state: HexGameState instance
        trmph_move: TRMPH format move (e.g., "a1", "b2")
        
    Returns:
        New HexGameState with the move applied
        
    Raises:
        ValueError: If TRMPH move is invalid or move is invalid
    """
    
    try:
        row, col = trmph_move_to_rowcol(trmph_move)
        return apply_move_to_state(state, row, col)
    except Exception as e:
        raise ValueError(f"Invalid TRMPH move '{trmph_move}': {e}")


def apply_move_to_tensor_trmph(board_tensor: torch.Tensor, trmph_move: str, player: Player) -> torch.Tensor:
    """
    Apply a TRMPH move to a 3-channel board tensor and return the new tensor.
    
    This is a wrapper that converts TRMPH to row,col coordinates.
    
    Args:
        board_tensor: torch.Tensor of shape (3, BOARD_SIZE, BOARD_SIZE)
        trmph_move: TRMPH format move (e.g., "a1", "b2")
        player: Player making the move (Player enum)
        
    Returns:
        New board tensor with the move applied
        
    Raises:
        ValueError: If TRMPH move is invalid or move is invalid
    """
    
    try:
        row, col = trmph_move_to_rowcol(trmph_move)
        return apply_move_to_tensor(board_tensor, row, col, player)
    except Exception as e:
        raise ValueError(f"Invalid TRMPH move '{trmph_move}': {e}") 

def select_top_value_head_move(model, state, top_k=20, temperature=1.0):
    """
    Select a move by evaluating the value head on the top-k policy moves and sampling among them.
    Args:
        model: Model instance (must have .simple_infer() method)
        state: Game state (must have .board and .get_legal_moves())
        top_k: Number of top moves to consider
        temperature: Temperature for policy and value sampling
    Returns:
        (row, col) tuple for the selected move, or None if no legal moves
    """
    topk_moves = get_top_k_legal_moves(model, state, top_k=top_k, temperature=temperature)
    if not topk_moves:
        return None
    move_values = []
    for move in topk_moves:
        temp_state = apply_move_to_state(state, *move)
        _, value_signed = model.simple_infer(temp_state.board)
        move_values.append(value_signed)
    chosen_idx = sample_move_by_value(move_values, temperature)
    return topk_moves[chosen_idx]


def make_empty_hex_state() -> HexGameState:
    """Create an empty Hex game state with Blue as the starting player.
    
    This is the standard way to start a new game. Blue always goes first
    in Hex by convention.
    
    Returns:
        A new HexGameState with an empty board and Blue to play
    """
    return HexGameState(Player.BLUE) 