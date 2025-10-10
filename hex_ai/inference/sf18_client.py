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
                 include_metadata: bool = False, max_retries: int = 3) -> Dict[str, Any]:
        """
        Request a move from the SF18 server with retry logic.
        
        Args:
            moves: TRMPH format move sequence (e.g., "a13d3k10h6g6k11l10g7j4")
            size: Board size (default: 13)
            difficulty: Difficulty level 1-10 (default: 9 for strong play)
            include_metadata: Whether to include additional move information
            max_retries: Maximum number of retry attempts for server errors
            
        Returns:
            dict: Response containing move, winner, and optionally metadata
            
        Raises:
            requests.RequestException: If the request fails after all retries
            ValueError: If the response contains an error
        """
        payload = {
            "moves": moves,
            "size": size,
            "difficulty": difficulty,
            "include_metadata": include_metadata
        }
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    self.api_endpoint,
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Log successful requests for debugging
                if 'metadata' in result:
                    metadata = result['metadata']
                    move_num = metadata.get('move_number', 'unknown')
                    player = metadata.get('player', 'unknown')
                    logger.debug(f"SF18 move: {result.get('move', 'none')} (move #{move_num}, {player})")
                
                if 'error' in result:
                    # Enhanced error reporting from SF18 server
                    error_msg = result['error']
                    error_type = result.get('error_type', 'unknown')
                    debug_info = result.get('debug_info', {})
                    request_id = debug_info.get('request_id', 'unknown')
                    
                    logger.error(f"SF18 server error (ID: {request_id}): {error_msg}")
                    logger.error(f"Error type: {error_type}")
                    logger.error(f"Debug info: {debug_info}")
                    
                    raise ValueError(f"SF18 server error [{error_type}] (ID: {request_id}): {error_msg}")
                
                return result
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                
                # Don't retry 400 errors (Bad Request) - these are client errors
                if hasattr(e, 'response') and e.response is not None:
                    status_code = e.response.status_code
                    if status_code == 400:
                        logger.error(f"SF18 server validation error (400): {e}")
                        # Try to get error details from response body
                        try:
                            error_response = e.response.json()
                            if 'error' in error_response:
                                error_msg = error_response['error']
                                error_type = error_response.get('error_type', 'validation_error')
                                debug_info = error_response.get('debug_info', {})
                                request_id = debug_info.get('request_id', 'unknown')
                                logger.error(f"SF18 validation error (ID: {request_id}): {error_msg}")
                                logger.error(f"Error type: {error_type}")
                                logger.error(f"Debug info: {debug_info}")
                        except:
                            pass
                        raise ValueError(f"SF18 validation error: {e}")
                    elif status_code >= 500:
                        # Retry 500+ errors (server errors)
                        if attempt < max_retries:
                            logger.warning(f"SF18 server error {status_code} (attempt {attempt + 1}/{max_retries + 1}): {e}")
                            time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                        else:
                            logger.error(f"SF18 server error {status_code} after {max_retries + 1} attempts: {e}")
                    else:
                        # Don't retry other client errors (401, 403, etc.)
                        logger.error(f"SF18 client error {status_code}: {e}")
                        raise ValueError(f"SF18 client error {status_code}: {e}")
                else:
                    # Network/connection errors - retry
                    if attempt < max_retries:
                        logger.warning(f"SF18 connection error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    else:
                        logger.error(f"SF18 connection error after {max_retries + 1} attempts: {e}")
        
        # If we get here, all retries failed
        raise requests.RequestException(f"Failed to connect to SF18 server after {max_retries + 1} attempts: {last_exception}")
    
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
    
    def test_error_handling(self) -> None:
        """
        Test the enhanced error handling by sending invalid requests.
        This helps verify that the SF18 server is providing detailed error information.
        """
        logger.info("Testing SF18 server error handling...")
        
        # Test invalid moves
        try:
            result = self.get_move("invalid_moves", size=13, difficulty=5)
            logger.warning("Expected error for invalid moves, but got success")
        except ValueError as e:
            logger.info(f"✓ Got expected error for invalid moves: {e}")
        
        # Test invalid difficulty
        try:
            result = self.get_move("a1", size=13, difficulty=99)
            logger.warning("Expected error for invalid difficulty, but got success")
        except ValueError as e:
            logger.info(f"✓ Got expected error for invalid difficulty: {e}")
        
        logger.info("SF18 error handling test complete")
    
    def get_move_from_state(self, state, difficulty: int = 9, max_retries: int = 3) -> Tuple[int, int]:
        """
        Get a move from the SF18 server given a game state with retry logic.
        
        Args:
            state: HexGameState object representing current board position
            difficulty: Difficulty level for SF18 (default: 9)
            max_retries: Maximum number of retry attempts for server errors
            
        Returns:
            Tuple of (row, col) representing the move
            
        Raises:
            requests.RequestException: If the request fails after all retries
            ValueError: If the response contains an error
        """
        # Convert state to TRMPH format using the built-in method
        trmph_str = state.to_trmph()
        # Remove the "#13," prefix to get just the moves
        moves_str = trmph_str[4:] if trmph_str.startswith("#13,") else trmph_str
        
        # Get move from server with retry logic
        result = self.get_move(moves_str, difficulty=difficulty, include_metadata=True, max_retries=max_retries)
        
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
