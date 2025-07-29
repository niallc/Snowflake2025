"""
Game engine for Hex AI.

This module provides the core game logic for Hex, including board representation,
move validation, winner detection, and game state management.
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import namedtuple

from hex_ai.config import (
    BOARD_SIZE, BLUE_PLAYER, RED_PLAYER, BLUE_PIECE, RED_PIECE, VERBOSE_LEVEL, EMPTY_PIECE
)
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

@dataclass
class HexGameState:
    """
    Represents the state of a Hex game using N×N character format.
    """
    board: np.ndarray = field(default_factory=lambda: np.full((BOARD_SIZE, BOARD_SIZE), EMPTY_PIECE, dtype='U1'))
    current_player: int = BLUE_PLAYER
    move_history: List[Tuple[int, int]] = field(default_factory=list)
    game_over: bool = False
    winner: Optional[str] = None  # "blue", "red", or None
    
    @property
    def board_2nxn(self) -> torch.Tensor:
        """Get board in 2×N×N format for compatibility with tests (legacy, do not use for inference)."""
        return board_nxn_to_2nxn(self.board)

    def is_valid_move(self, row: int, col: int) -> bool:
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return False
        if self.game_over:
            return False
        return is_empty(self.board, row, col)

    def make_move(self, row: int, col: int) -> 'HexGameState':
        if not self.is_valid_move(row, col):
            raise ValueError(f"Invalid move: ({row}, {col})")
        color = "blue" if self.current_player == BLUE_PLAYER else "red"
        new_board = place_piece(self.board, row, col, color)
        new_move_history = self.move_history + [(row, col)]
        new_state = HexGameState(
            board=new_board,
            current_player=RED_PLAYER if self.current_player == BLUE_PLAYER else BLUE_PLAYER,
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
                if self.board[row, col] == RED_PIECE:
                    piece = Piece(row=row, column=col, colour="red")
                    connections.make_set(piece)
                    if VERBOSE_LEVEL >= 4:
                        print(f"[DEBUG] Make set for RED piece: {piece}")
                    self._connect_piece_to_neighbors(piece, connections)
                elif self.board[row, col] == BLUE_PIECE:
                    piece = Piece(row=row, column=col, colour="blue")
                    connections.make_set(piece)
                    if VERBOSE_LEVEL >= 4:
                        print(f"[DEBUG] Make set for BLUE piece: {piece}")
                    self._connect_piece_to_neighbors(piece, connections)
        
        # Explicitly connect edge pieces to board edge pieces (legacy logic)
        for col in range(BOARD_SIZE):
            # Blue: top and bottom rows
            if self.board[0, col] == BLUE_PIECE:
                piece = Piece(row=0, column=col, colour="blue")
                connections.union(piece, blue_top)
                if VERBOSE_LEVEL >= 4:
                    print(f"[DEBUG] Union BLUE top: {piece} <-> {blue_top}")
            if self.board[BOARD_SIZE-1, col] == BLUE_PIECE:
                piece = Piece(row=BOARD_SIZE-1, column=col, colour="blue")
                connections.union(piece, blue_bottom)
                if VERBOSE_LEVEL >= 4:
                    print(f"[DEBUG] Union BLUE bottom: {piece} <-> {blue_bottom}")
        for row in range(BOARD_SIZE):
            # Red: left and right columns
            if self.board[row, 0] == RED_PIECE:
                piece = Piece(row=row, column=0, colour="red")
                connections.union(piece, red_left)
                if VERBOSE_LEVEL >= 4:
                    print(f"[DEBUG] Union RED left: {piece} <-> {red_left}")
            if self.board[row, BOARD_SIZE-1] == RED_PIECE:
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
        expected_value = RED_PIECE if color == "red" else BLUE_PIECE
        
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
        status = f"Player {'Blue' if self.current_player == BLUE_PLAYER else 'Red'}'s turn"
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
        player: Player making the move (BLUE_PLAYER=0 or RED_PLAYER=1)
        
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
        model: Model instance (must have .infer() method)
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