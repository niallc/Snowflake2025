#!/usr/bin/env python3
"""
Test script for deterministic tournament functionality.

This script validates that:
1. Opening extraction works correctly
2. Deterministic games produce consistent results
3. The tournament system handles deterministic play properly
"""

import os
import sys
import tempfile
import random
import numpy as np
from typing import List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_deterministic_tournament import (
    OpeningPosition, StrategyConfig, parse_strategy_configs,
    extract_openings_from_trmph_file, generate_diverse_openings,
    play_deterministic_game
)
from hex_ai.inference.model_config import get_model_path
from hex_ai.inference.model_cache import preload_tournament_models, get_model_cache


def create_test_trmph_file() -> str:
    """Create a temporary TRMPH file with test games."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.trmph', delete=False) as f:
        # Create some simple test games
        test_games = [
            "http://www.trmph.com/hex/board#13,i10f12k4j6e9g5j4i5l5k5e6d10i6j5f7e10f9f4g4f5d5d6f10e11h5h11g11g12i11h10f11e12i9h9l4j8k7k6l6k8i8j3h8f8e8j12h12h13g13f2i4l2h3h1i1h2i2h4i3g8h6g6i7 b",
            "http://www.trmph.com/hex/board#13,b13f5k4k3i4h8g6i5f8g7f7d11c11d10g9f9b10h4h5j2j3k2i3c8i2i1j1c10g8f12b11d8e11c12g10d13c13d12h11g11e12e13h10g13i12i11h12i13j12h13k11j13l12k13m12k12l11l13m13 b",
            "http://www.trmph.com/hex/board#13,a13f8g7h6e6f4h8g6i6h7i7i5g5d9e9e8k4j5l5k6k5j6l6k7l7k3j3k8l8k9l9k10l10k12m12m11k11j12j11i12l12l11i10i11j10j4m2m1l2l1k2l4l3k1i2j2i3i1j1g11g10h12h11g12h9d12c11e11e10d11d10i9h10f10f9g9g8b13b12 b",
        ]
        for game in test_games:
            f.write(game + '\n')
        return f.name


def test_opening_extraction():
    """Test that opening extraction works correctly."""
    print("Testing opening extraction...")
    
    # Create test file
    test_file = create_test_trmph_file()
    
    try:
        # Extract openings
        openings = extract_openings_from_trmph_file(test_file, opening_length=7, max_openings=10)
        
        # Verify results
        assert len(openings) > 0, "No openings extracted"
        assert all(len(opening.moves) == 7 for opening in openings), "Wrong opening length"
        assert all(opening.source_game.startswith(os.path.basename(test_file)) for opening in openings), "Wrong source game"
        
        print(f"✓ Extracted {len(openings)} openings successfully")
        
        # Test TRMPH conversion
        for opening in openings[:2]:  # Test first 2
            trmph_str = opening.get_trmph_string()
            assert trmph_str.startswith("#13,"), "Invalid TRMPH format"
            print(f"✓ TRMPH conversion: {trmph_str[:50]}...")
            
    finally:
        os.unlink(test_file)


def test_strategy_parsing():
    """Test strategy configuration parsing."""
    print("\nTesting strategy parsing...")
    
    # Test different strategy types
    strategies = ["policy", "mcts_100", "fixed_tree_13_8"]
    configs = parse_strategy_configs(strategies, None, None)
    
    assert len(configs) == 3, "Wrong number of configs"
    assert configs[0].strategy_type == "policy", "Wrong policy config"
    assert configs[1].strategy_type == "mcts", "Wrong MCTS config"
    assert configs[1].config["mcts_sims"] == 100, "Wrong MCTS sims"
    assert configs[2].strategy_type == "fixed_tree", "Wrong fixed_tree config"
    assert configs[2].config["search_widths"] == [13, 8], "Wrong search widths"
    
    print("✓ Strategy parsing works correctly")


def test_deterministic_consistency():
    """Test that deterministic games produce consistent results."""
    print("\nTesting deterministic consistency...")
    
    # Get a model
    try:
        model_path = get_model_path("current_best")
        preload_tournament_models([model_path])
        model_cache = get_model_cache()
        model = model_cache.get_simple_model(model_path)
    except Exception as e:
        print(f"⚠ Skipping deterministic test (no model available): {e}")
        return
    
    # Create simple opening
    opening = OpeningPosition([(6, 6), (6, 7), (7, 6), (7, 7), (8, 6), (8, 7), (9, 6)], "test")
    
    # Create strategies
    strategy_a = StrategyConfig("policy", "policy", {})
    strategy_b = StrategyConfig("policy", "policy", {})
    
    # Play the same game multiple times
    results = []
    for i in range(3):
        result = play_deterministic_game(model, strategy_a, strategy_b, opening)
        results.append(result)
    
    # Verify consistency
    first_result = results[0]
    for i, result in enumerate(results[1:], 1):
        assert result['winner_strategy'] == first_result['winner_strategy'], f"Inconsistent winner in run {i}"
        assert result['trmph_str'] == first_result['trmph_str'], f"Inconsistent game in run {i}"
    
    print("✓ Deterministic games produce consistent results")


def test_diverse_openings():
    """Test diverse opening generation."""
    print("\nTesting diverse opening generation...")
    
    # Create test files
    test_files = []
    for i in range(3):
        test_file = create_test_trmph_file()
        test_files.append(test_file)
    
    try:
        # Generate diverse openings
        openings = generate_diverse_openings(test_files, opening_length=7, target_count=10)
        
        # Verify results
        assert len(openings) > 0, "No diverse openings generated"
        assert len(openings) <= 10, "Too many openings generated"
        
        # Check diversity (simple check: different sources)
        sources = set(opening.source_game for opening in openings)
        assert len(sources) > 1, "Openings not diverse enough"
        
        print(f"✓ Generated {len(openings)} diverse openings from {len(sources)} sources")
        
    finally:
        for test_file in test_files:
            os.unlink(test_file)


def main():
    """Run all tests."""
    print("Running deterministic tournament tests...\n")
    
    # Set random seed for reproducible tests
    random.seed(42)
    np.random.seed(42)
    
    try:
        test_opening_extraction()
        test_strategy_parsing()
        test_deterministic_consistency()
        test_diverse_openings()
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
