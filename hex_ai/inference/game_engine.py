"""
Game engine for Hex AI.

This module provides the core game logic for Hex, including board representation,
move validation, winner detection, and game state management.
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from collections import namedtuple

from hex_ai.config import (
    BOARD_SIZE, VERBOSE_LEVEL, EMPTY_PIECE
)
from hex_ai.enums import Player, Winner as WinnerEnum, Piece as PieceEnum
from hex_ai.inference.board_utils import (
    is_empty, place_piece, board_to_string
)
from hex_ai.utils.format_conversion import (
    board_nxn_to_2nxn, board_nxn_to_3nxn, rowcol_to_trmph, trmph_move_to_rowcol, split_trmph_moves
)
from hex_ai.value_utils import (
    get_top_k_legal_moves, sample_move_by_value, apply_move_to_tensor
)


# Edge coordinates for Union-Find
LEFT_EDGE = -1
RIGHT_EDGE = BOARD_SIZE
TOP_EDGE = -1
BOTTOM_EDGE = BOARD_SIZE

# Union-Find data structure for tracking connected components
Piece = namedtuple("Piece", ["row", "column", "colour"])

class UnionFind:
    """Union-Find data structure for tracking connected components in Hex."""
    
    def __init__(self):
        self.parents_ = dict()
        self.ranks_ = dict()
    
    def make_set(self, x):
        """Create a new set containing element x."""
        if x not in self.parents_:
            self.parents_[x] = None
            self.ranks_[x] = 0
    
    def find(self, x):
        """Find the root of the set containing x, with path compression."""
        assert x in self.parents_, f"Element {x} not found in UnionFind. Call make_set first."
        
        if self.parents_[x] is None:
            return x
        
        # Path compression
        self.parents_[x] = self.find(self.parents_[x])
        return self.parents_[x]
    
    def union(self, x, y):
        """Union the sets containing x and y."""
        x_root = self.find(x)
        y_root = self.find(y)
        
        if x_root == y_root:
            return  # Already in same set
        
        # Union by rank
        if self.ranks_[x_root] < self.ranks_[y_root]:
            self.parents_[x_root] = y_root
        elif self.ranks_[x_root] > self.ranks_[y_root]:
            self.parents_[y_root] = x_root
        else:
            # Same rank, arbitrarily make x_root the parent
            self.parents_[y_root] = x_root
            self.ranks_[x_root] += 1
    
    def are_connected(self, x, y):
        """Check if x and y are in the same connected component."""
        x_root = self.find(x)
        y_root = self.find(y)
        return x_root == y_root

@dataclass(init=False)
class HexGameState:
    """
    Represents the state of a Hex game using N×N character format.
    """
    board: np.ndarray = field(default_factory=lambda: np.full((BOARD_SIZE, BOARD_SIZE), EMPTY_PIECE, dtype='U1'))
    _current_player: Player = field(default=Player.BLUE, repr=False)
    move_history: List[Tuple[int, int]] = field(default_factory=list)
    game_over: bool = False
    _winner: Optional[WinnerEnum] = field(default=None, repr=False)
    
    # Operation stack for undo functionality
    _undo_stack: List[Dict[str, Any]] = field(default_factory=list, init=False)

    def __init__(self, board: Optional[np.ndarray] = None, current_player: Optional[int] = None,
                 _current_player: Optional[Player] = None, move_history: Optional[List[Tuple[int, int]]] = None,
                 game_over: bool = False, winner: Optional[str] = None):
        # Initialize board
        self.board = board if board is not None else np.full((BOARD_SIZE, BOARD_SIZE), EMPTY_PIECE, dtype='U1')
        # Initialize current player with compatibility for legacy int
        if _current_player is not None:
            if not isinstance(_current_player, Player):
                raise TypeError(f"_current_player must be Player, got {type(_current_player)}")
            self._current_player = _current_player
        elif current_player is not None:
            if current_player == Player.BLUE.value:
                self._current_player = Player.BLUE
            elif current_player == Player.RED.value:
                self._current_player = Player.RED
            else:
                raise ValueError(f"{current_player} is not a valid Player")
        else:
            self._current_player = Player.BLUE
        # Initialize other fields
        self.move_history = move_history.copy() if move_history is not None else []
        self.game_over = game_over
        # Winner internal storage uses Enum; accept both Enum and string for compatibility
        self._winner = None
        if winner is not None:
            if isinstance(winner, str):
                if winner == "blue":
                    self._winner = WinnerEnum.BLUE
                elif winner == "red":
                    self._winner = WinnerEnum.RED
                else:
                    raise ValueError(f"Invalid winner string: {winner}")
            elif isinstance(winner, WinnerEnum):
                self._winner = winner
            else:
                raise TypeError(f"winner must be str or Winner enum, got {type(winner)}")
        self._undo_stack = []
    
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
    def winner(self) -> Optional[str]:
        """Legacy-facing winner as 'blue'/'red' or None. Internally stores Winner enum."""
        if self._winner is None:
            return None
        return "blue" if self._winner == WinnerEnum.BLUE else "red"

    @winner.setter
    def winner(self, value: Optional[object]) -> None:
        """Accept 'blue'/'red' strings or Winner enum; fail fast on invalid inputs."""
        if value is None:
            self._winner = None
            return
        if isinstance(value, str):
            if value == "blue":
                self._winner = WinnerEnum.BLUE
                return
            if value == "red":
                self._winner = WinnerEnum.RED
                return
            raise ValueError(f"Invalid winner string: {value}")
        if isinstance(value, WinnerEnum):
            self._winner = value
            return
        raise TypeError(f"winner must be str, Winner enum, or None; got {type(value)}")

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
            'move': (row, col)
        }
        self._undo_stack.append(undo_info)
        
        # Apply the move
        color = "blue" if self._current_player == Player.BLUE else "red"
        self.board = place_piece(self.board, row, col, color)
        self.move_history.append((row, col))
        self._current_player = Player.RED if self._current_player == Player.BLUE else Player.BLUE
        
        # Check for winner
        winner = self._find_winner()
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
        # Don't copy the undo stack - it's not needed for new states
        return new_state

    def make_move(self, row: int, col: int) -> 'HexGameState':
        # TODO: PERFORMANCE CRITICAL - Replace deepcopy with apply/undo pattern
        # Current make_move() creates new HexGameState objects which is expensive for MCTS
        # IMPLEMENTATION PLAN (Phase 3.1):
        # 1) Add apply_move() method to HexGameState that mutates in place
        # 2) Add undo_last() method that restores previous state
        # 3) Update MCTS to use apply → encode → undo pattern for evaluations
        # 4) Use fast_copy() only when child nodes need to be materialized
        # Expected gain: 10-20x speedup in expansion phase
        if not self.is_valid_move(row, col):
            raise ValueError(f"Invalid move: ({row}, {col})")
        color = "blue" if self._current_player == Player.BLUE else "red"
        new_board = place_piece(self.board, row, col, color)
        new_move_history = self.move_history + [(row, col)]
        new_state = HexGameState(
            board=new_board,
            _current_player=Player.RED if self._current_player == Player.BLUE else Player.BLUE,
            move_history=new_move_history,
            game_over=False,
            winner=None
        )
        # Winner detection
        # TODO: Improve Efficiency: We currently find the winner from scratch after
        # every move. UnionFind is an efficient incremental algorithm and we should
        # update an existing UnionFind object instead of creating a new one.
        winner = new_state._find_winner()
        if winner:
            new_state.game_over = True
            new_state.winner = winner
        return new_state

    def _get_adjacent_positions(self, row: int, col: int) -> List[Tuple[int, int]]:
        adjacent = []
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (-1, 1)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                adjacent.append((new_row, new_col))
        return adjacent

    def _find_winner(self) -> Optional[str]:
        """Find the winner using Union-Find connected components."""
        connections = self._build_connections()
        
        # Check for red win (horizontal connection)
        red_left = Piece(row=0, column=LEFT_EDGE, colour="red")
        red_right = Piece(row=0, column=RIGHT_EDGE, colour="red")
        if connections.are_connected(red_left, red_right):
            return "red"
        
        # Check for blue win (vertical connection)
        blue_top = Piece(row=TOP_EDGE, column=0, colour="blue")
        blue_bottom = Piece(row=BOTTOM_EDGE, column=0, colour="blue")
        if connections.are_connected(blue_top, blue_bottom):
            return "blue"
        
        return None
    
    def _build_connections(self) -> UnionFind:
        """Build Union-Find connections for all pieces on the board."""
        connections = UnionFind()
        
        # Initialize special edge pieces
        red_left = Piece(row=0, column=LEFT_EDGE, colour="red")
        red_right = Piece(row=0, column=RIGHT_EDGE, colour="red")
        blue_top = Piece(row=TOP_EDGE, column=0, colour="blue")
        blue_bottom = Piece(row=BOTTOM_EDGE, column=0, colour="blue")
        
        connections.make_set(red_left)
        connections.make_set(red_right)
        connections.make_set(blue_top)
        connections.make_set(blue_bottom)
        if VERBOSE_LEVEL >= 4:
            print(f"[DEBUG] Initialized edge pieces: {red_left}, {red_right}, {blue_top}, {blue_bottom}")
        
        # Add all pieces on the board and connect them
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board[row, col] == PieceEnum.RED.value:
                    piece = Piece(row=row, column=col, colour="red")
                    connections.make_set(piece)
                    if VERBOSE_LEVEL >= 4:
                        print(f"[DEBUG] Make set for RED piece: {piece}")
                    self._connect_piece_to_neighbors(piece, connections)
                elif self.board[row, col] == PieceEnum.BLUE.value:
                    piece = Piece(row=row, column=col, colour="blue")
                    connections.make_set(piece)
                    if VERBOSE_LEVEL >= 4:
                        print(f"[DEBUG] Make set for BLUE piece: {piece}")
                    self._connect_piece_to_neighbors(piece, connections)
        
        # Explicitly connect edge pieces to board edge pieces (legacy logic)
        for col in range(BOARD_SIZE):
            # Blue: top and bottom rows
            if self.board[0, col] == PieceEnum.BLUE.value:
                piece = Piece(row=0, column=col, colour="blue")
                connections.union(piece, blue_top)
                if VERBOSE_LEVEL >= 4:
                    print(f"[DEBUG] Union BLUE top: {piece} <-> {blue_top}")
            if self.board[BOARD_SIZE-1, col] == PieceEnum.BLUE.value:
                piece = Piece(row=BOARD_SIZE-1, column=col, colour="blue")
                connections.union(piece, blue_bottom)
                if VERBOSE_LEVEL >= 4:
                    print(f"[DEBUG] Union BLUE bottom: {piece} <-> {blue_bottom}")
        for row in range(BOARD_SIZE):
            # Red: left and right columns
            if self.board[row, 0] == PieceEnum.RED.value:
                piece = Piece(row=row, column=0, colour="red")
                connections.union(piece, red_left)
                if VERBOSE_LEVEL >= 4:
                    print(f"[DEBUG] Union RED left: {piece} <-> {red_left}")
            if self.board[row, BOARD_SIZE-1] == PieceEnum.RED.value:
                piece = Piece(row=row, column=BOARD_SIZE-1, colour="red")
                connections.union(piece, red_right)
                if VERBOSE_LEVEL >= 4:
                    print(f"[DEBUG] Union RED right: {piece} <-> {red_right}")
        if VERBOSE_LEVEL >= 4:
            print("[DEBUG] Final Union-Find parents:")
            for k, v in connections.parents_.items():
                print(f"  {k} -> {v}")
        return connections
    
    def _connect_piece_to_neighbors(self, piece: Piece, connections: UnionFind):
        """Connect a piece to its same-color neighbors."""
        neighbors = self._get_same_color_neighbors(piece)
        if VERBOSE_LEVEL >= 4:
            print(f"[DEBUG] {piece} neighbors: {neighbors}")
        for neighbor in neighbors:
            if neighbor not in connections.parents_:
                connections.make_set(neighbor)
                if VERBOSE_LEVEL >= 4:
                    print(f"[DEBUG] Make set for neighbor: {neighbor}")
            connections.union(piece, neighbor)
            if VERBOSE_LEVEL >= 4:
                print(f"[DEBUG] Union: {piece} <-> {neighbor}")
    
    def _get_same_color_neighbors(self, piece: Piece) -> List[Piece]:
        """Get all same-color neighbors of a piece."""
        neighbors = []
        color = piece.colour
        expected_value = PieceEnum.RED.value if color == "red" else PieceEnum.BLUE.value
        
        # Get adjacent positions
        adjacent_positions = self._get_adjacent_positions(piece.row, piece.column)
        
        for adj_row, adj_col in adjacent_positions:
            if self._is_valid_position(adj_row, adj_col):
                # Check if this position has a piece of the same color
                if self.board[adj_row, adj_col] == expected_value:
                    neighbor = Piece(row=adj_row, column=adj_col, colour=color)
                    neighbors.append(neighbor)
            else:
                # Check if this connects to an edge piece
                edge_piece = self._get_edge_piece(adj_row, adj_col, color)
                if edge_piece:
                    neighbors.append(edge_piece)
        
        return neighbors
    
    def _is_valid_position(self, row: int, col: int) -> bool:
        """Check if a position is valid on the board."""
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE
    
    def _get_edge_piece(self, row: int, col: int, color: str) -> Optional[Piece]:
        """Get the appropriate edge piece for off-board positions."""
        if color == "red":
            if col == LEFT_EDGE:
                return Piece(row=0, column=LEFT_EDGE, colour="red")
            elif col == RIGHT_EDGE:
                return Piece(row=0, column=RIGHT_EDGE, colour="red")
        elif color == "blue":
            if row == TOP_EDGE:
                return Piece(row=TOP_EDGE, column=0, colour="blue")
            elif row == BOTTOM_EDGE:
                return Piece(row=BOTTOM_EDGE, column=0, colour="blue")
        return None

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
        state = cls()
        for move in moves:
            row, col = trmph_move_to_rowcol(move)
            state = state.make_move(row, col)
        return state

    def __str__(self) -> str:
        status = f"Player {'Blue' if self._current_player == Player.BLUE else 'Red'}'s turn"
        if self.game_over:
            status = f"Game over - {self.winner.title()} wins!"
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
        return HexGameState()
    
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
    
    def get_winner(self, state: HexGameState) -> Optional[str]:
        """
        Get the winner of the game.
        
        Args:
            state: Current game state
            
        Returns:
            "blue", "red", or None if no winner
        """
        return state._find_winner()
    
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


def apply_move_to_tensor_trmph(board_tensor: torch.Tensor, trmph_move: str, player: int) -> torch.Tensor:
    """
    Apply a TRMPH move to a 3-channel board tensor and return the new tensor.
    
    This is a wrapper that converts TRMPH to row,col coordinates.
    
    Args:
        board_tensor: torch.Tensor of shape (3, BOARD_SIZE, BOARD_SIZE)
        trmph_move: TRMPH format move (e.g., "a1", "b2")
        player: Player making the move (prefer Player enum; int accepted for backward compatibility)
        
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
        _, value_logit = model.simple_infer(temp_state.board)
        move_values.append(value_logit)
    chosen_idx = sample_move_by_value(move_values, temperature)
    return topk_moves[chosen_idx] 