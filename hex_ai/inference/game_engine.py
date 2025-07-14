"""
Game engine for Hex AI.

This module provides the core game logic, state management, and move validation
for Hex games. It integrates with the existing data_utils for coordinate conversions
and ports the Union-Find winner detection from legacy code.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from collections import namedtuple
import logging

from ..data_utils import (
    trmph_move_to_rowcol, rowcol_to_trmph, 
    tensor_to_rowcol, rowcol_to_tensor
)
from ..config import BOARD_SIZE, NUM_PLAYERS

logger = logging.getLogger(__name__)

# Union-Find implementation for winner detection
Piece = namedtuple("Piece", ["row", "column", "colour"])


class UnionFind:
    """
    Union-Find data structure for tracking connected pieces.
    
    This is ported from legacy_code/UnionFind.py and adapted for our use case.
    """
    
    def __init__(self):
        self.parents_ = {}
        self.ranks_ = {}
    
    def make_set(self, x):
        """Create a new set containing element x."""
        if x not in self.parents_:
            self.parents_[x] = None
            self.ranks_[x] = 0
    
    def find(self, x):
        """Find the root of the set containing element x."""
        if x not in self.parents_:
            raise ValueError(f"Element {x} not found in UnionFind")
        
        if self.parents_[x] is None:
            return x
        
        # Path compression
        self.parents_[x] = self.find(self.parents_[x])
        return self.parents_[x]
    
    def union(self, x, y):
        """Merge the sets containing elements x and y."""
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
            # Arbitrarily make one root the new parent
            self.parents_[y_root] = x_root
            self.ranks_[x_root] += 1
    
    def are_connected(self, x, y):
        """Check if elements x and y are in the same set."""
        try:
            x_root = self.find(x)
            y_root = self.find(y)
            return x_root == y_root
        except ValueError:
            return False


@dataclass
class HexGameState:
    """
    Represents the state of a Hex game.
    
    This class provides an immutable game state with methods for
    move validation, winner detection, and board representation.
    """
    
    # Board representation: (2, 13, 13) tensor
    # Channel 0: Blue pieces (1.0 where blue piece exists, 0.0 elsewhere)
    # Channel 1: Red pieces (1.0 where red piece exists, 0.0 elsewhere)
    board: torch.Tensor = field(default_factory=lambda: torch.zeros(2, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32))
    
    # Current player: 0=blue (first player), 1=red (second player)
    current_player: int = 0
    
    # Move history: list of (row, col) tuples
    move_history: List[Tuple[int, int]] = field(default_factory=list)
    
    # Union-Find for tracking connected pieces
    connections: UnionFind = field(default_factory=UnionFind)
    
    # Game state flags
    game_over: bool = False
    winner: Optional[str] = None  # "blue", "red", or None
    
    def __post_init__(self):
        """Initialize the game state after creation."""
        # Initialize special pieces for winner detection
        self._initialize_special_pieces()
        
        # Apply existing moves to connections
        self._update_connections_from_history()
    
    def _initialize_special_pieces(self):
        """Initialize the special edge pieces for winner detection."""
        # Special pieces representing connections to edges
        # These are "virtual" pieces that represent the board edges
        self.blue_top = Piece(row=-1, column=0, colour="blue")
        self.blue_bottom = Piece(row=BOARD_SIZE, column=0, colour="blue")
        self.red_left = Piece(row=0, column=-1, colour="red")
        self.red_right = Piece(row=0, column=BOARD_SIZE, colour="red")
        
        # Add special pieces to Union-Find
        self.connections.make_set(self.blue_top)
        self.connections.make_set(self.blue_bottom)
        self.connections.make_set(self.red_left)
        self.connections.make_set(self.red_right)
    
    def _update_connections_from_history(self):
        """Update connections based on existing move history."""
        for i, (row, col) in enumerate(self.move_history):
            player = i % 2  # 0=blue, 1=red
            colour = "blue" if player == 0 else "red"
            piece = Piece(row=row, column=col, colour=colour)
            
            # Add piece to Union-Find
            self.connections.make_set(piece)
            
            # Connect to adjacent pieces of same color
            self._connect_to_neighbors(piece)
    
    def _update_connections_from_board(self):
        """Update connections based on current board state."""
        # Clear existing connections (except special pieces)
        special_pieces = {self.blue_top, self.blue_bottom, self.red_left, self.red_right}
        new_connections = UnionFind()
        
        # Re-add special pieces
        for piece in special_pieces:
            new_connections.make_set(piece)
        
        # Add all pieces on the board
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board[0, row, col] == 1.0:  # Blue piece
                    piece = Piece(row=row, column=col, colour="blue")
                    new_connections.make_set(piece)
                elif self.board[1, row, col] == 1.0:  # Red piece
                    piece = Piece(row=row, column=col, colour="red")
                    new_connections.make_set(piece)
        
        # Connect all pieces to their neighbors
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board[0, row, col] == 1.0:  # Blue piece
                    piece = Piece(row=row, column=col, colour="blue")
                    self._connect_piece_to_neighbors(piece, new_connections)
                elif self.board[1, row, col] == 1.0:  # Red piece
                    piece = Piece(row=row, column=col, colour="red")
                    self._connect_piece_to_neighbors(piece, new_connections)
        
        self.connections = new_connections
        
        # Check for winner after updating connections
        winner = self._check_winner()
        if winner:
            self.game_over = True
            self.winner = winner
    
    def _connect_piece_to_neighbors(self, piece: Piece, connections: UnionFind):
        """Connect a piece to its adjacent pieces of the same color."""
        row, col, colour = piece.row, piece.column, piece.colour
        
        # Get adjacent positions
        adjacent_positions = self._get_adjacent_positions(row, col)
        
        for adj_row, adj_col in adjacent_positions:
            # Check if adjacent position has a piece of the same color
            if self._has_piece_at(adj_row, adj_col, colour):
                adj_piece = Piece(row=adj_row, column=adj_col, colour=colour)
                connections.make_set(adj_piece)
                connections.union(piece, adj_piece)
        
        # Connect to edge pieces if this piece is on an edge
        self._connect_piece_to_edges(piece, connections)
    
    def _connect_piece_to_edges(self, piece: Piece, connections: UnionFind):
        """Connect a piece to edge pieces if it's on an edge."""
        row, col, colour = piece.row, piece.column, piece.colour
        
        if colour == "red":
            # Red connects left to right
            if col == 0:  # Left edge
                connections.union(piece, self.red_left)
            if col == BOARD_SIZE - 1:  # Right edge
                connections.union(piece, self.red_right)
        elif colour == "blue":
            # Blue connects top to bottom
            if row == 0:  # Top edge
                connections.union(piece, self.blue_top)
            if row == BOARD_SIZE - 1:  # Bottom edge
                connections.union(piece, self.blue_bottom)
    
    def _connect_to_neighbors(self, piece: Piece):
        """Connect a piece to its adjacent pieces of the same color."""
        row, col, colour = piece.row, piece.column, piece.colour
        
        # Get adjacent positions
        adjacent_positions = self._get_adjacent_positions(row, col)
        
        for adj_row, adj_col in adjacent_positions:
            # Check if adjacent position has a piece of the same color
            if self._has_piece_at(adj_row, adj_col, colour):
                adj_piece = Piece(row=adj_row, column=adj_col, colour=colour)
                self.connections.make_set(adj_piece)
                self.connections.union(piece, adj_piece)
        
        # Connect to edge pieces if this piece is on an edge
        self._connect_to_edges(piece)
    
    def _connect_to_edges(self, piece: Piece):
        """Connect a piece to edge pieces if it's on an edge."""
        row, col, colour = piece.row, piece.column, piece.colour
        
        if colour == "red":
            # Red connects left to right
            if col == 0:  # Left edge
                self.connections.union(piece, self.red_left)
            if col == BOARD_SIZE - 1:  # Right edge
                self.connections.union(piece, self.red_right)
        elif colour == "blue":
            # Blue connects top to bottom
            if row == 0:  # Top edge
                self.connections.union(piece, self.blue_top)
            if row == BOARD_SIZE - 1:  # Bottom edge
                self.connections.union(piece, self.blue_bottom)
    
    def _get_adjacent_positions(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get all adjacent positions for a given position."""
        # Hex adjacency: 6 directions
        # Based on legacy code, adjacent positions are:
        # (row, col+1), (row, col-1)  # Same row
        # (row+1, col), (row-1, col)  # Same column  
        # (row+1, col-1), (row-1, col+1)  # Diagonal
        
        adjacent = []
        
        # Standard adjacent positions
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (-1, 1)]:
            new_row, new_col = row + dr, col + dc
            
            # Check if position is on board
            if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                adjacent.append((new_row, new_col))
        
        return adjacent
    
    def _has_piece_at(self, row: int, col: int, colour: str) -> bool:
        """Check if there's a piece of the given color at the specified position."""
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return False
        
        if colour == "blue":
            return self.board[0, row, col] == 1.0
        elif colour == "red":
            return self.board[1, row, col] == 1.0
        else:
            return False
    
    def is_valid_move(self, row: int, col: int) -> bool:
        """
        Check if a move is valid.
        
        Args:
            row: Row index (0-12)
            col: Column index (0-12)
            
        Returns:
            True if the move is valid, False otherwise
        """
        # Check if position is on board
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return False
        
        # Check if position is empty
        if self.board[0, row, col] == 1.0 or self.board[1, row, col] == 1.0:
            return False
        
        # Check if game is over
        if self.game_over:
            return False
        
        return True
    
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Get all legal moves for the current position."""
        legal_moves = []
        
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.is_valid_move(row, col):
                    legal_moves.append((row, col))
        
        return legal_moves
    
    def make_move(self, row: int, col: int) -> 'HexGameState':
        """
        Make a move and return a new game state.
        
        Args:
            row: Row index (0-12)
            col: Column index (0-12)
            
        Returns:
            New game state after the move
            
        Raises:
            ValueError: If the move is invalid
        """
        if not self.is_valid_move(row, col):
            raise ValueError(f"Invalid move: ({row}, {col})")
        
        # Create new board tensor
        new_board = self.board.clone()
        
        # Place the piece
        if self.current_player == 0:  # Blue
            new_board[0, row, col] = 1.0
            colour = "blue"
        else:  # Red
            new_board[1, row, col] = 1.0
            colour = "red"
        
        # Create new move history
        new_move_history = self.move_history + [(row, col)]
        
        # Create new connections
        new_connections = UnionFind()
        
        # Copy existing connections
        for piece, parent in self.connections.parents_.items():
            new_connections.parents_[piece] = parent
            new_connections.ranks_[piece] = self.connections.ranks_[piece]
        
        # Create new state
        new_state = HexGameState(
            board=new_board,
            current_player=1 - self.current_player,  # Switch players
            move_history=new_move_history,
            connections=new_connections,
            game_over=self.game_over,
            winner=self.winner
        )
        
        # Add the new piece to connections
        piece = Piece(row=row, column=col, colour=colour)
        new_connections.make_set(piece)
        
        # Connect to adjacent pieces of same color
        new_state._connect_to_neighbors(piece)
        
        # Check for winner
        winner = new_state._check_winner()
        if winner:
            new_state.game_over = True
            new_state.winner = winner
        
        return new_state
    
    def _check_winner(self) -> Optional[str]:
        """
        Check if there's a winner using Union-Find.
        
        Returns:
            "blue", "red", or None if no winner
        """
        # Check if red connects left to right
        if self.connections.are_connected(self.red_left, self.red_right):
            return "red"
        
        # Check if blue connects top to bottom
        if self.connections.are_connected(self.blue_top, self.blue_bottom):
            return "blue"
        
        return None
    
    def get_board_tensor(self) -> torch.Tensor:
        """
        Get the board as a tensor for model input.
        
        Returns:
            Board tensor of shape (2, 13, 13)
        """
        return self.board.clone()
    
    def to_trmph(self) -> str:
        """
        Convert the game state to TRMPH format.
        
        Returns:
            TRMPH string representation
        """
        moves = []
        for row, col in self.move_history:
            move = rowcol_to_trmph(row, col)
            moves.append(move)
        
        return f"#13,{''.join(moves)}"
    
    @classmethod
    def from_trmph(cls, trmph_text: str) -> 'HexGameState':
        """
        Create a game state from a TRMPH string.
        
        Args:
            trmph_text: TRMPH string representation (can be full URL or just moves)
            
        Returns:
            Game state
        """
        # Handle full URLs or just the moves part
        if trmph_text.startswith("http://"):
            # Extract the moves part from URL
            if "#13," in trmph_text:
                moves_text = trmph_text.split("#13,")[1]
            else:
                raise ValueError("Invalid TRMPH URL format")
        elif trmph_text.startswith("#13,"):
            moves_text = trmph_text[4:]  # Remove "#13," prefix
        else:
            # Assume it's just the moves
            moves_text = trmph_text
        
        moves = []
        
        # Parse moves (pairs of letter+number)
        i = 0
        while i < len(moves_text):
            if i + 1 >= len(moves_text):
                break
            
            letter = moves_text[i]
            if not letter.isalpha():
                raise ValueError(f"Expected letter at position {i}")
            
            # Find the number
            j = i + 1
            while j < len(moves_text) and moves_text[j].isdigit():
                j += 1
            
            move = moves_text[i:j]
            row, col = trmph_move_to_rowcol(move)
            moves.append((row, col))
            
            i = j
        
        # Create game state
        state = cls()
        
        # Apply moves
        for row, col in moves:
            state = state.make_move(row, col)
        
        return state
    
    def __str__(self) -> str:
        """String representation of the game state."""
        status = f"Player {'Blue' if self.current_player == 0 else 'Red'}'s turn"
        if self.game_over:
            status = f"Game over - {self.winner.title()} wins!"
        
        return f"HexGameState(moves={len(self.move_history)}, {status})"
    
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
        return state.winner
    
    def is_game_over(self, state: HexGameState) -> bool:
        """
        Check if the game is over.
        
        Args:
            state: Current game state
            
        Returns:
            True if the game is over
        """
        return state.game_over 