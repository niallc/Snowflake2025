#!/usr/bin/env python3
"""
Test script to demonstrate MCTS early termination feature.
"""

import sys
import os
import time
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hex_ai.inference.game_engine import HexGameEngine, HexGameState
from hex_ai.inference.mcts import (
    BaselineMCTS, BaselineMCTSConfig, 
    create_mcts_config
)
from hex_ai.inference.model_wrapper import ModelWrapper
from hex_ai.inference.model_config import get_model_path
from hex_ai.utils.format_conversion import trmph_to_moves


def test_early_termination():
    """Test early termination with different configurations."""
    
    # Use a game state where RED is in a commanding position
    # This should trigger high win probability for RED
    red_is_winning = "#13,a2g7f6e8d7c9b8b10h8i6j7k5l6l4"
    
    # Parse the game
    moves = trmph_to_moves(red_is_winning)
    
    # Create game state (use all moves to get the commanding position)
    engine = HexGameEngine()
    state = HexGameState()
    
    # Play all moves to get the commanding position
    for move in moves:
        state = state.make_move(move[0], move[1])
    
    print(f"🎮 Testing early termination on RED commanding position after {len(moves)} moves")
    print(f"🎮 Current player: {state.current_player}")
    print(f"🎮 Legal moves: {len(state.get_legal_moves())}")
    
    # Load model
    model_path = get_model_path("current_best")
    model = ModelWrapper(model_path, device="cpu")  # Use CPU for testing
    
    # Test different configurations
    configs = [
        ("Standard MCTS (no early termination)", BaselineMCTSConfig(sims=200)),
        ("Self-play config (aggressive)", create_mcts_config("selfplay", sims=200, early_termination_threshold=0.85)),
        ("Tournament config (conservative)", create_mcts_config("tournament", sims=200, early_termination_threshold=0.95)),
        ("Fast self-play config (very aggressive)", create_mcts_config("fast_selfplay", sims=200, early_termination_threshold=0.8)),
    ]
    
    results = []
    
    for name, config in configs:
        print(f"\n🧪 Testing: {name}")
        print(f"   Simple termination: {config.enable_early_termination}")
        if config.enable_early_termination:
            print(f"   Threshold: {config.early_termination_threshold} (stops when win prob ≥{config.early_termination_threshold} OR ≤{1-config.early_termination_threshold})")
        
        # Run MCTS
        mcts = BaselineMCTS(engine, model, config)
        start_time = time.perf_counter()
        result = mcts.run(state, verbose=2)
        end_time = time.perf_counter()
        
        # Get move and win probability
        move = result.move
        win_prob = result.win_probability
        
        # Record results
        results.append({
            'name': name,
            'time': end_time - start_time,
            'simulations': result.stats['total_simulations'],
            'early_termination': result.stats.get('early_termination_occurred', False),
            'win_probability': win_prob,
            'selected_move': move
        })
        
        print(f"   Time: {end_time - start_time:.3f}s")
        print(f"   Simulations: {result.stats['total_simulations']}")
        print(f"   Simple termination occurred: {result.stats.get('early_termination_occurred', False)}")
        print(f"   Win probability: {win_prob:.3f}")
        print(f"   Selected move: {move}")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"{'Configuration':<40} {'Time (s)':<10} {'Sims':<8} {'Term':<8} {'Win Prob':<10}")
    print("-" * 80)
    
    for result in results:
        term_info = "Yes" if result['early_termination'] else "No"
        print(f"{result['name']:<40} {result['time']:<10.3f} {result['simulations']:<8} "
              f"{term_info:<8} {result['win_probability']:<10.3f}")
    
    # Calculate speedup
    baseline_time = results[0]['time']
    print(f"\n🚀 Speedup compared to standard MCTS:")
    for result in results[1:]:
        speedup = baseline_time / result['time']
        print(f"   {result['name']}: {speedup:.2f}x faster")


def test_early_termination_on_different_positions():
    """Test early termination on different game positions."""
    
    # Test positions with different characteristics
    test_positions = [
        ("RED commanding position", "#13,a2g7f6e8d7c9b8b10h8i6j7k5l6l4"),
        ("BLUE commanding position", "#13,a2g7i6j4h5i3g4h2f3g1e2j7h8g10i9h11j10i12k11j13l12"),
        ("Early game", "#13,a1g1a7g2b7g3c7g4d7g5e7g6f7"),
    ]
    
    engine = HexGameEngine()
    model_path = get_model_path("current_best")
    model = ModelWrapper(model_path, device="cpu")
    
    # Use fast self-play config for testing
    config = create_mcts_config("fast_selfplay", sims=100, early_termination_threshold=0.8)
    
    print(f"\n🎯 Testing early termination on different game positions:")
    print(f"Config: {config.early_termination_threshold} threshold")
    
    for position_name, trmph_game in test_positions:
        moves = trmph_to_moves(trmph_game)
        state = HexGameState()
        
        for move in moves:
            state = state.make_move(move[0], move[1])
        
        print(f"\n📍 {position_name} (move {len(moves)})")
        
        # Run MCTS
        mcts = BaselineMCTS(engine, model, config)
        start_time = time.perf_counter()
        result = mcts.run(state, verbose=1)
        end_time = time.perf_counter()
        
        win_prob = result.win_probability
        
        print(f"   Time: {end_time - start_time:.3f}s")
        print(f"   Simulations: {result.stats['total_simulations']}")
        print(f"   Simple termination: {result.stats.get('early_termination_occurred', False)}")
        print(f"   Win probability: {win_prob:.3f}")


if __name__ == "__main__":
    print("🧪 Testing MCTS Confidence-Based Termination Feature")
    print("=" * 50)
    
    try:
        test_early_termination()
        test_early_termination_on_different_positions()
        
        print(f"\n✅ Simple termination testing completed!")
        print(f"\n💡 Usage tips:")
        print(f"   - Use create_mcts_config('selfplay', ...) for fast game generation")
        print(f"   - Use create_mcts_config('tournament', ...) for high-quality play")
        print(f"   - Use create_mcts_config('fast_selfplay', ...) for maximum speed")
        print(f"   - Adjust early_termination_threshold based on your needs")
        print(f"   - Feature stops when neural network is very confident")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
