#!/usr/bin/env python3
"""
Client for communicating with the SF18 (Snowflake 2018) Hex game server.

This module provides a clean interface for requesting moves from the older
neural network-powered Hex game server via HTTP API.
"""

import requests
import json
import time
import logging
from typing import Optional, Dict, Any, Tuple
from hex_ai.config import BOARD_SIZE
from hex_ai.enums import Player
from hex_ai.utils.format_conversion import trmph_to_moves, rowcol_to_trmph, trmph_move_to_rowcol

logger = logging.getLogger(__name__)

# Delay between SF18 server requests (seconds)
# Set to 0 for local server, increase for remote/internet servers
SF18_REQUEST_DELAY = 0.0


class SF18Client:
    """Client for communicating with the SF18 Hex game server."""
    
    def __init__(self, server_url: str = "http://localhost:8088", timeout: int = 30):
        """
        Initialize the SF18 client.
        
        Args:
            server_url: Base URL of the SF18 Hex game server
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip('/')
        self.api_endpoint = f"{self.server_url}/api/move"
        self.timeout = timeout
    
    def get_move(self, moves: str, size: int = BOARD_SIZE, difficulty: int = 9, 
                 include_metadata: bool = False) -> Dict[str, Any]:
        """
        Request a move from the SF18 server.
        
        Args:
            moves: TRMPH format move sequence (e.g., "a13d3k10h6g6k11l10g7j4")
            size: Board size (default: 13)
            difficulty: Difficulty level 1-10 (default: 9 for strong play)
            include_metadata: Whether to include additional move information
            
        Returns:
            dict: Response containing move, winner, and optionally metadata
            
        Raises:
            requests.RequestException: If the request fails
            ValueError: If the response contains an error
        """
        payload = {
            "moves": moves,
            "size": size,
            "difficulty": difficulty,
            "include_metadata": include_metadata
        }
        
        try:
            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            if 'error' in result:
                raise ValueError(f"SF18 server error: {result['error']}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            raise requests.RequestException(f"Failed to connect to SF18 server: {e}")
    
    def is_server_running(self) -> bool:
        """
        Check if the SF18 server is running and accessible.
        
        Returns:
            bool: True if server is accessible, False otherwise
        """
        try:
            response = requests.get(f"{self.server_url}/", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_move_from_state(self, state, difficulty: int = 9) -> Tuple[int, int]:
        """
        Get a move from the SF18 server given a game state.
        
        Args:
            state: HexGameState object representing current board position
            difficulty: Difficulty level for SF18 (default: 9)
            
        Returns:
            Tuple of (row, col) representing the move
            
        Raises:
            requests.RequestException: If the request fails
            ValueError: If the response contains an error
        """
        # Convert state to TRMPH format using the built-in method
        trmph_str = state.to_trmph()
        # Remove the "#13," prefix to get just the moves
        moves_str = trmph_str[4:] if trmph_str.startswith("#13,") else trmph_str
        
        # Get move from server
        result = self.get_move(moves_str, difficulty=difficulty, include_metadata=True)
        
        if result['move'] is None:
            raise ValueError("SF18 server returned no move (game over)")
        
        # Convert TRMPH move to row, col
        move_trmph = result['move']
        try:
            row, col = trmph_move_to_rowcol(move_trmph, BOARD_SIZE)
            return (row, col)
        except Exception as e:
            raise ValueError(f"Invalid move format from SF18: {move_trmph} - {e}")
    
    
    def play_complete_game(self, initial_moves: str = "", max_moves: int = 100, 
                          difficulty: int = 9) -> Tuple[str, str]:
        """
        Play a complete game using only the SF18 server.
        
        Args:
            initial_moves: Starting moves in TRMPH format
            max_moves: Maximum number of moves to make
            difficulty: Difficulty level
            
        Returns:
            Tuple of (final_moves, winner) where winner is "blue", "red", or "no winner"
        """
        moves = initial_moves
        
        for move_num in range(max_moves):
            try:
                result = self.get_move(moves, difficulty=difficulty)
                
                if result['move'] is None or result['winner'] != "no winner":
                    return moves, result['winner']
                
                next_move = result['move']
                moves += next_move
                
                # Small delay to avoid overwhelming the server
                if SF18_REQUEST_DELAY > 0:
                    time.sleep(SF18_REQUEST_DELAY)
                
            except Exception as e:
                logger.error(f"Error on move {move_num + 1}: {e}")
                raise
        
        # Game didn't finish within max_moves
        return moves, "no winner"


class SF18Player:
    """
    Player wrapper for SF18 that integrates with the existing tournament system.
    
    This class provides a consistent interface that can be used alongside
    SF25 model players in tournaments.
    """
    
    def __init__(self, name: str, client: SF18Client, difficulty: int = 9):
        """
        Initialize SF18 player.
        
        Args:
            name: Name for this player instance
            client: SF18Client instance for communication
            difficulty: Difficulty level for SF18 (1-10)
        """
        self.name = name
        self.client = client
        self.difficulty = difficulty
        self.temperature = None  # SF18 doesn't use temperature, but needed for compatibility
        self.strategy_type = "sf18"  # Strategy type for compatibility
        self.model_path = "SF18"  # Model path for compatibility
    
    def get_move(self, state) -> Tuple[int, int]:
        """
        Get a move from this SF18 player.
        
        Args:
            state: HexGameState object representing current board position
            
        Returns:
            Tuple of (row, col) representing the move
        """
        return self.client.get_move_from_state(state, difficulty=self.difficulty)
    
    def __str__(self) -> str:
        return f"SF18Player({self.name}, difficulty={self.difficulty})"
    
    def __repr__(self) -> str:
        return self.__str__()
