#!/usr/bin/env python3
"""
Integration test for MCTS implementation.

This script tests the MCTS implementation with a real model to ensure it works
correctly for move selection in actual games.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hex_ai.inference.mcts import NeuralMCTS
from hex_ai.inference.game_engine import HexGameState
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.inference.board_display import display_hex_board
from hex_ai.config import BLUE_PLAYER, RED_PLAYER

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_mcts_with_model(model_path: str, num_simulations: int = 100):
    """
    Test MCTS with a real model.
    
    Args:
        model_path: Path to the model checkpoint
        num_simulations: Number of MCTS simulations to run
    """
    logger.info(f"Testing MCTS with model: {model_path}")
    logger.info(f"Number of simulations: {num_simulations}")
    
    # Load model
    try:
        model = SimpleModelInference(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False
    
    # Create MCTS engine
    mcts = NeuralMCTS(model=model, exploration_constant=1.4)
    logger.info("MCTS engine created")
    
    # Create initial game state
    state = HexGameState()
    logger.info("Initial game state created")
    
    # Display initial board
    print("\nInitial board:")
    display_hex_board(state.board)
    
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
    for move, child in root.children.items():
        print(f"  Move {move}: visits={child.visits}, mean_value={child.mean_value:.4f}")
    
    # Select move
    selected_move = mcts.select_move(root, temperature=0)  # Deterministic
    print(f"\nSelected move (deterministic): {selected_move}")
    
    # Test stochastic selection
    selected_move_stoch = mcts.select_move(root, temperature=1.0)  # Stochastic
    print(f"Selected move (stochastic): {selected_move_stoch}")
    
    # Make the move
    new_state = state.make_move(*selected_move)
    print(f"\nAfter move {selected_move}:")
    display_hex_board(new_state.board)
    
    return True


def test_mcts_progression(model_path: str, num_moves: int = 3):
    """
    Test MCTS over multiple moves in a game.
    
    Args:
        model_path: Path to the model checkpoint
        num_moves: Number of moves to play
    """
    logger.info(f"Testing MCTS progression with {num_moves} moves")
    
    # Load model and create MCTS
    model = SimpleModelInference(model_path)
    mcts = NeuralMCTS(model=model, exploration_constant=1.4)
    
    # Start game
    state = HexGameState()
    
    for move_num in range(num_moves):
        print(f"\n{'='*50}")
        print(f"Move {move_num + 1}")
        print(f"{'='*50}")
        
        # Display current board
        print("Current board:")
        display_hex_board(state.board)
        print(f"Player to move: {'Blue' if state.current_player == BLUE_PLAYER else 'Red'}")
        
        # Run MCTS search
        root = mcts.search(state, num_simulations=50)  # Fewer simulations for speed
        
        # Display top moves
        print("\nTop moves:")
        sorted_moves = sorted(root.children.items(), key=lambda x: x[1].visits, reverse=True)
        for i, (move, child) in enumerate(sorted_moves[:5]):
            print(f"  {i+1}. Move {move}: visits={child.visits}, value={child.mean_value:.4f}")
        
        # Select move
        selected_move = mcts.select_move(root, temperature=0)
        print(f"\nSelected move: {selected_move}")
        
        # Make move
        state = state.make_move(*selected_move)
        
        # Check if game is over
        if state.game_over:
            print(f"\nGame over! Winner: {state.winner}")
            break
    
    return True


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MCTS integration")
    parser.add_argument("model_path", help="Path to model checkpoint")
    parser.add_argument("--simulations", type=int, default=100, help="Number of MCTS simulations")
    parser.add_argument("--moves", type=int, default=3, help="Number of moves to play in progression test")
    parser.add_argument("--progression", action="store_true", help="Run progression test")
    
    args = parser.parse_args()
    
    if args.progression:
        success = test_mcts_progression(args.model_path, args.moves)
    else:
        success = test_mcts_with_model(args.model_path, args.simulations)
    
    if success:
        logger.info("MCTS integration test completed successfully")
        return 0
    else:
        logger.error("MCTS integration test failed")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 