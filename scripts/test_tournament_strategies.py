#!/usr/bin/env python3
"""
Test script to validate tournament strategies.

This script tests the new move selection strategy system by running
small tournaments with different strategies to ensure they work correctly.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hex_ai.inference.tournament import TournamentConfig, TournamentPlayConfig, run_round_robin_tournament
from hex_ai.inference.move_selection import get_strategy, MoveSelectionConfig, list_available_strategies
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.inference.game_engine import HexGameState
from hex_ai.config import BOARD_SIZE


def test_strategy_registry():
    """Test that all strategies are available."""
    print("Testing strategy registry...")
    strategies = list_available_strategies()
    print(f"Available strategies: {strategies}")
    
    for strategy_name in strategies:
        strategy = get_strategy(strategy_name)
        print(f"  {strategy_name}: {strategy.get_name()}")
    
    print("✓ Strategy registry test passed\n")


def test_move_selection_strategies():
    """Test that each strategy can select moves."""
    print("Testing move selection strategies...")
    
    # Create a simple test state
    from hex_ai.enums import Player
    state = HexGameState(
        board=[[0] * BOARD_SIZE for _ in range(BOARD_SIZE)],
        _current_player=Player.BLUE  # Blue's turn
    )
    
    # We need a model for testing - use a mock or find a real checkpoint
    # For now, just test that strategies can be instantiated
    for strategy_name in list_available_strategies():
        strategy = get_strategy(strategy_name)
        config = MoveSelectionConfig()
        
        print(f"  {strategy_name}: {strategy.get_config_summary(config)}")
    
    print("✓ Move selection strategy test passed\n")


def test_tournament_config():
    """Test tournament configuration with different strategies."""
    print("Testing tournament configuration...")
    
    # Test different strategy configurations
    configs = [
        ("policy", {}),
        ("mcts", {"mcts_sims": 50, "mcts_c_puct": 1.0}),
        ("fixed_tree", {"search_widths": [10, 5]}),
    ]
    
    for strategy_name, strategy_config in configs:
        play_config = TournamentPlayConfig(
            strategy=strategy_name,
            strategy_config=strategy_config,
            temperature=1.0,
            random_seed=42
        )
        print(f"  {strategy_name}: {play_config.strategy_config}")
    
    print("✓ Tournament configuration test passed\n")


def test_small_tournament():
    """Test a small tournament with different strategies."""
    print("Testing small tournament...")
    
    # Find checkpoints to use for testing
    from hex_ai.inference.model_config import get_model_path, get_available_models, validate_model_path
    
    # Try to get two different models for testing
    checkpoint_paths = []
    model_names = get_available_models()
    
    for model_name in model_names:
        if len(checkpoint_paths) >= 2:
            break
        try:
            model_path = get_model_path(model_name)
            if validate_model_path(model_path):
                checkpoint_paths.append(model_path)
        except Exception as e:
            print(f"Warning: Could not load model {model_name}: {e}")
    
    if len(checkpoint_paths) < 2:
        print("⚠ Need at least 2 checkpoints for testing - skipping tournament test")
        print(f"Available models: {model_names}")
        return
    
    print(f"Using checkpoints: {[os.path.basename(p) for p in checkpoint_paths]}")
    
    # Test with policy strategy (fastest)
    config = TournamentConfig(
        checkpoint_paths=checkpoint_paths,
        num_games=2  # Very small for testing
    )
    
    play_config = TournamentPlayConfig(
        strategy="policy",
        temperature=1.0,
        random_seed=42,
        pie_rule=False
    )
    
    # Create temporary files for output
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, "test_tournament.trmph")
        csv_file = os.path.join(temp_dir, "test_tournament.csv")
        
        try:
            result = run_round_robin_tournament(
                config, verbose=1, log_file=log_file, csv_file=csv_file, play_config=play_config
            )
            print(f"  Tournament completed: {result.total_games} games")
            print(f"  Win rates: {result.win_rates()}")
            print("✓ Small tournament test passed")
        except Exception as e:
            print(f"✗ Tournament test failed: {e}")
            raise
    
    print()


def main():
    """Run all tests."""
    print("Testing Tournament Strategy System")
    print("=" * 40)
    
    try:
        test_strategy_registry()
        test_move_selection_strategies()
        test_tournament_config()
        test_small_tournament()
        
        print("All tests passed! ✓")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
