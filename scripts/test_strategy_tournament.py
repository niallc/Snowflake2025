#!/usr/bin/env python3
"""
Test script for the strategy tournament system.

This script runs small strategy tournaments to validate that the system works correctly.
"""

import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_strategy_tournament import (
    StrategyConfig, parse_strategy_configs, run_strategy_tournament
)
from hex_ai.inference.tournament import TournamentPlayConfig
from hex_ai.inference.model_config import get_model_path, validate_model_path


def test_strategy_parsing():
    """Test strategy configuration parsing."""
    print("Testing strategy configuration parsing...")
    
    # Test basic strategies
    strategies = ["policy", "mcts_100", "fixed_tree_13_8"]
    configs = parse_strategy_configs(strategies, None, None)
    
    assert len(configs) == 3
    assert configs[0].name == "policy"
    assert configs[0].strategy_type == "policy"
    assert configs[1].name == "mcts_100"
    assert configs[1].strategy_type == "mcts"
    assert configs[1].config["mcts_sims"] == 100
    assert configs[2].name == "fixed_tree_13_8"
    assert configs[2].strategy_type == "fixed_tree"
    assert configs[2].config["search_widths"] == [13, 8]
    
    print("✓ Strategy parsing test passed")
    
    # Test with command line overrides
    mcts_sims = [50, 200]
    search_widths = ["10,5", "20,10"]
    strategies = ["mcts_100", "mcts_200", "fixed_tree_13_8", "fixed_tree_20_10"]
    configs = parse_strategy_configs(strategies, mcts_sims, search_widths)
    
    assert configs[0].config["mcts_sims"] == 50
    assert configs[1].config["mcts_sims"] == 200
    assert configs[2].config["search_widths"] == [10, 5]
    assert configs[3].config["search_widths"] == [20, 10]
    
    print("✓ Strategy override test passed\n")


def test_small_strategy_tournament():
    """Test a small strategy tournament."""
    print("Testing small strategy tournament...")
    
    # Find a model to use for testing
    try:
        model_path = get_model_path("current_best")
        if not validate_model_path(model_path):
            print("⚠ Current best model not found, trying previous best...")
            model_path = get_model_path("previous_best")
            if not validate_model_path(model_path):
                print("⚠ No valid models found for testing - skipping tournament test")
                return
    except Exception as e:
        print(f"⚠ Could not load model: {e}")
        return
    
    print(f"Using model: {os.path.basename(model_path)}")
    
    # Test with policy vs MCTS (fastest combination)
    strategies = ["policy", "mcts_50"]  # Small number of simulations for speed
    configs = parse_strategy_configs(strategies, None, None)
    
    # Create base configuration
    base_config = TournamentPlayConfig(
        temperature=1.0,
        random_seed=42,
        pie_rule=False  # Disable pie rule for simpler testing
    )
    
    # Run small tournament
    result = run_strategy_tournament(
        model_path=model_path,
        strategy_configs=configs,
        num_games=2,  # Very small for testing
        base_play_config=base_config,
        verbose=1
    )
    
    # Check results
    assert result.total_games == 4  # 2 games per pair, 1 pair = 4 total games
    win_rates = result.win_rates()
    assert len(win_rates) == 2
    assert "policy" in win_rates
    assert "mcts_50" in win_rates
    
    print("✓ Small tournament test passed")
    print(f"  Win rates: {win_rates}")
    print()


def test_strategy_config_creation():
    """Test StrategyConfig creation and play config generation."""
    print("Testing StrategyConfig creation...")
    
    # Create a strategy config
    config = StrategyConfig("mcts_100", "mcts", {"mcts_sims": 100, "mcts_c_puct": 1.5})
    
    # Test string representation
    assert str(config) == "mcts_100(mcts)"
    
    # Test play config generation
    base_config = TournamentPlayConfig(temperature=1.0, random_seed=42)
    play_config = config.get_play_config(base_config)
    
    assert play_config.strategy == "mcts"
    assert play_config.strategy_config["mcts_sims"] == 100
    assert play_config.strategy_config["mcts_c_puct"] == 1.5
    assert play_config.temperature == 1.0
    assert play_config.random_seed == 42
    
    print("✓ StrategyConfig test passed\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("STRATEGY TOURNAMENT SYSTEM TESTS")
    print("=" * 60)
    
    try:
        test_strategy_parsing()
        test_strategy_config_creation()
        test_small_strategy_tournament()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
