"""
Opening strategies for self-play training.

This module provides utilities for generating boards with specific opening moves
that are realistic for pie rule play. The goal is to ensure self-play games
explore the kinds of configurations that high-level play would focus on.
"""

import random
from typing import List, Optional, Tuple
from hex_ai.inference.game_engine import HexGameState
from hex_ai.utils.format_conversion import trmph_move_to_rowcol
from hex_ai.enums import Player


class OpeningStrategy:
    """Base class for opening strategies."""
    
    def __init__(self, board_size: int = 13):
        self.board_size = board_size
    
    def get_opening_move(self, game_index: int) -> Optional[Tuple[int, int]]:
        """
        Get the opening move for a given game index.
        
        Args:
            game_index: Index of the game (0-based)
            
        Returns:
            Tuple of (row, col) coordinates, or None for empty board
        """
        raise NotImplementedError
    
    def get_total_games(self) -> int:
        """Get the total number of games this strategy covers."""
        raise NotImplementedError


class PieRuleOpeningStrategy(OpeningStrategy):
    """
    Opening strategy that focuses on realistic pie rule openings.
    
    This strategy generates openings in the following order:
    1. Balanced moves: (a2-a13), (b5-b11), (b2-k2)
    2. Somewhat unbalanced moves: (a1, b12, c3)
    3. Network-chosen moves (empty board)
    4. Bad moves (b1, c1, d1, ..., l1) at reduced frequency
    """
    
    def __init__(self, board_size: int = 13, bad_move_frequency: float = 0.1):
        super().__init__(board_size)
        self.bad_move_frequency = bad_move_frequency
        
        # Define opening moves by category
        self.balanced_moves = []
        self.somewhat_unbalanced_moves = []
        self.bad_moves = []
        
        # Balanced moves: (a2-a13), (b5-b11), (b2-k2)
        # a2-a13: moves along the a-file
        for row in range(1, board_size):  # a2 to a13
            self.balanced_moves.append((row, 0))  # col 0 = 'a'
        
        # b5-b11: moves along the b-file, middle section
        for row in range(4, 11):  # b5 to b11
            self.balanced_moves.append((row, 1))  # col 1 = 'b'
        
        # b2-k2: moves along row 1 (2nd row), b to k
        for col in range(1, 11):  # b2 to k2
            self.balanced_moves.append((1, col))
        
        # Somewhat unbalanced moves
        self.somewhat_unbalanced_moves = [
            (0, 0),   # a1
            (11, 1),  # b12
            (2, 2),   # c3
        ]
        
        # Bad moves: b1, c1, d1, ..., l1 (edge moves that are too strong)
        for col in range(1, 12):  # b1 to l1
            self.bad_moves.append((0, col))
        
        # Calculate total games for each category
        self.balanced_games = len(self.balanced_moves)
        self.unbalanced_games = len(self.somewhat_unbalanced_moves)
        self.network_games = 2  # 2 games with network-chosen moves
        self.bad_games = int(len(self.bad_moves) * self.bad_move_frequency)
        
        # Total games this strategy covers
        self._total_games = (
            self.balanced_games + 
            self.unbalanced_games + 
            self.network_games + 
            self.bad_games
        )
    
    def get_opening_move(self, game_index: int) -> Optional[Tuple[int, int]]:
        """
        Get the opening move for a given game index.
        
        Args:
            game_index: Index of the game (0-based)
            
        Returns:
            Tuple of (row, col) coordinates, or None for empty board
        """
        # If beyond our strategy, cycle through the moves
        if game_index >= self._total_games:
            # Calculate which cycle we're in and the offset within that cycle
            cycle = game_index // self._total_games
            offset = game_index % self._total_games
            game_index = offset
        
        # Balanced moves first
        if game_index < self.balanced_games:
            return self.balanced_moves[game_index]
        
        game_index -= self.balanced_games
        
        # Somewhat unbalanced moves
        if game_index < self.unbalanced_games:
            return self.somewhat_unbalanced_moves[game_index]
        
        game_index -= self.unbalanced_games
        
        # Network-chosen moves (empty board)
        if game_index < self.network_games:
            return None
        
        game_index -= self.network_games
        
        # Bad moves (at reduced frequency)
        if game_index < self.bad_games:
            # Sample from bad moves based on frequency
            bad_move_index = int(game_index / self.bad_move_frequency)
            if bad_move_index < len(self.bad_moves):
                return self.bad_moves[bad_move_index]
        
        # Fallback: empty board
        return None
    
    def get_total_games(self) -> int:
        """Get the total number of games this strategy covers."""
        return self._total_games


class RandomOpeningStrategy(OpeningStrategy):
    """Opening strategy that randomly selects from a set of moves."""
    
    def __init__(self, moves: List[Tuple[int, int]], board_size: int = 13, 
                 empty_board_prob: float = 0.1):
        super().__init__(board_size)
        self.moves = moves
        self.empty_board_prob = empty_board_prob
    
    def get_opening_move(self, game_index: int) -> Optional[Tuple[int, int]]:
        """Get a random opening move."""
        if random.random() < self.empty_board_prob:
            return None
        return random.choice(self.moves)
    
    def get_total_games(self) -> int:
        """This strategy can be used for any number of games."""
        return float('inf')  # Infinite games


def create_board_with_opening(opening_move: Optional[Tuple[int, int]], 
                            board_size: int = 13) -> HexGameState:
    """
    Create a game state with a specific opening move.
    
    Args:
        opening_move: Tuple of (row, col) coordinates, or None for empty board
        board_size: Size of the board
        
    Returns:
        HexGameState with the opening move applied
    """
    state = make_empty_hex_state()
    
    if opening_move is not None:
        row, col = opening_move
        # Validate coordinates
        if not (0 <= row < board_size and 0 <= col < board_size):
            raise ValueError(f"Invalid coordinates ({row}, {col}) for board size {board_size}")
        
        # Apply the opening move
        state = state.make_move(row, col)
    
    return state


def get_trmph_opening_move(opening_move: Optional[Tuple[int, int]], 
                          board_size: int = 13) -> Optional[str]:
    """
    Convert opening move coordinates to TRMPH format.
    
    Args:
        opening_move: Tuple of (row, col) coordinates, or None
        board_size: Size of the board
        
    Returns:
        TRMPH move string (e.g., "a2"), or None for empty board
    """
    if opening_move is None:
        return None
    
    row, col = opening_move
    from hex_ai.utils.format_conversion import rowcol_to_trmph
    return rowcol_to_trmph(row, col, board_size)


def create_pie_rule_strategy(board_size: int = 13, bad_move_frequency: float = 0.1) -> PieRuleOpeningStrategy:
    """
    Create a pie rule opening strategy.
    
    Args:
        board_size: Size of the board
        bad_move_frequency: Frequency of bad moves (0.0 to 1.0)
        
    Returns:
        PieRuleOpeningStrategy instance
    """
    return PieRuleOpeningStrategy(board_size, bad_move_frequency)
