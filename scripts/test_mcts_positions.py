#!/usr/bin/env python3
"""
Test MCTS on specific positions provided in trmph format.

This script allows testing MCTS on specific board positions to debug
the _select_child_with_puct function and other MCTS components.
"""

import sys
import logging
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hex_ai.inference.mcts import NeuralMCTS
from hex_ai.inference.game_engine import HexGameState
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.inference.model_config import get_model_path
from hex_ai.inference.board_display import display_hex_board
from hex_ai.utils.format_conversion import parse_trmph_to_board
from hex_ai.config import BLUE_PLAYER, RED_PLAYER, BLUE_PIECE, RED_PIECE, EMPTY_PIECE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_state_from_trmph(trmph_text: str) -> HexGameState:
    """
    Create a HexGameState from a trmph format string.
    
    Args:
        trmph_text: Trmph format string (e.g., "https://trmph.com/hex/board#13,g1a7g2b7...")
        
    Returns:
        HexGameState with the board position set
    """
    # Parse the board from trmph format
    board = parse_trmph_to_board(trmph_text)
    
    # Create a new game state
    state = HexGameState()
    
    # Apply the moves to the state
    for row in range(board.shape[0]):
        for col in range(board.shape[1]):
            if board[row, col] == BLUE_PIECE:
                state.board[row, col] = BLUE_PIECE  # Use string value
            elif board[row, col] == RED_PIECE:
                state.board[row, col] = RED_PIECE   # Use string value
            # EMPTY_PIECE is already the default
    
    # Use the state board directly for display (now expects string values)
    display_board = state.board
    
    # Determine current player based on number of moves
    blue_count = (board == BLUE_PIECE).sum()
    red_count = (board == RED_PIECE).sum()
    
    if blue_count > red_count:
        state.current_player = RED_PLAYER  # Red's turn next
    else:
        state.current_player = BLUE_PLAYER  # Blue's turn next
    
    # Check if game is over by finding winner
    winner = state._find_winner()
    if winner:
        state.game_over = True
        state.winner = winner
    
    return state, display_board


def test_mcts_on_position(model_path: str, trmph_position: str, num_simulations: int = 100, verbose: int = 0):
    """
    Test MCTS on a specific position.
    
    Args:
        model_path: Path to the model checkpoint
        trmph_position: Position in trmph format
        num_simulations: Number of MCTS simulations to run
        verbose: Verbosity level (0=quiet, 1=basic, 2=detailed, 3=debug, 4=extreme debug)
    """
    logger.info(f"Testing MCTS on position: {trmph_position}")
    logger.info(f"Number of simulations: {num_simulations}")
    
    # Load model
    try:
        model = SimpleModelInference(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False
    
    # Create MCTS engine
    mcts = NeuralMCTS(model=model, exploration_constant=1.4, verbose=verbose)
    logger.info("MCTS engine created")
    
    # Create state from trmph position
    try:
        state, display_board = create_state_from_trmph(trmph_position)
        logger.info("Position loaded successfully")
    except Exception as e:
        logger.error(f"Failed to parse position: {e}")
        return False
    
    # Display the position
    print("\nCurrent position:")
    display_hex_board(display_board)
    print(f"Player to move: {'Blue' if state.current_player == BLUE_PLAYER else 'Red'}")
    print(f"Game over: {state.game_over}")
    if state.game_over:
        print(f"Winner: {state.winner}")
    
    # Run MCTS search
    logger.info("Starting MCTS search...")
    root = mcts.search(state, num_simulations)
    
    # Get search statistics
    stats = mcts.get_search_statistics()
    logger.info(f"Search completed: {stats}")
    
    # Display search results
    print(f"\nMCTS Search Results ({num_simulations} simulations):")
    print(f"Total time: {stats.get('total_time', 0):.2f}s")
    print(f"Simulations per second: {stats.get('simulations_per_second', 0):.1f}")
    print(f"Total inferences: {stats.get('total_inferences', 0)}")
    
    # Display move statistics
    print("\nMove statistics:")
    sorted_moves = sorted(root.children.items(), key=lambda x: x[1].visits, reverse=True)
    for i, (move, child) in enumerate(sorted_moves[:10]):
        print(f"  {i+1}. Move {move}: visits={child.visits}, mean_value={child.mean_value:.4f}")
    
    # Select move
    selected_move = mcts.select_move(root, temperature=0)  # Deterministic
    print(f"\nSelected move (deterministic): {selected_move}")
    
    # Test stochastic selection
    selected_move_stoch = mcts.select_move(root, temperature=1.0)  # Stochastic
    print(f"Selected move (stochastic): {selected_move_stoch}")
    
    return True


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MCTS on specific positions")
    parser.add_argument("--model_path", help="Path to model checkpoint", default=get_model_path("current_best"))
    parser.add_argument("--position", type=str, 
                       default="https://trmph.com/hex/board#13,g1a7g2b7g3c7g4d7g5e7g6f7g8h7g9i7g10j7g11k7g12l7g13m7",
                       help="Position in trmph format")
    parser.add_argument("--simulations", type=int, default=5, help="Number of MCTS simulations")
    parser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2, 3, 4], 
                       help="Verbosity level (0=quiet, 1=basic, 2=detailed, 3=debug, 4=extreme debug)")
    
    args = parser.parse_args()
    
    success = test_mcts_on_position(args.model_path, args.position, args.simulations, args.verbose)
    
    if success:
        logger.info("MCTS position test completed successfully")
        return 0
    else:
        logger.error("MCTS position test failed")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 