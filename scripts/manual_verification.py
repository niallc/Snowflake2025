#!/usr/bin/env python3
"""
Manual verification script to check data flow in a 3x2 search tree.

This script provides manual verification tools to check that our batched inference
implementation is working correctly. It's designed for human review and debugging.

USAGE:
    python scripts/manual_verification.py

OUTPUT INTERPRETATION:
    The script will output detailed information about:
    1. Initial game state and legal moves
    2. Policy predictions for current position
    3. Top 3 policy moves with probabilities
    4. Value predictions for each child position
    5. Batched search results
    6. Verification checks (move legality, value reasonableness, etc.)

EXPECTED BEHAVIOR:
    - All moves should be legal
    - Values should be between -1.0 and 1.0
    - Best move should be among top 3 policy moves
    - Game should not end after one move
    - Batched results should match individual results

DEBUGGING:
    If you see unexpected behavior:
    1. Check that moves are legal (in legal_moves list)
    2. Verify values are reasonable (between -1.0 and 1.0)
    3. Ensure game state updates correctly after moves
    4. Compare batched vs individual results for discrepancies
"""

import sys
import os
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hex_ai.inference.game_engine import HexGameState
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.inference.fixed_tree_search import minimax_policy_value_search_with_batching, PositionCollector
from hex_ai.value_utils import get_top_k_moves_with_probs

def manual_verification_3x2():
    """
    Manually verify data flow in a 3x2 search tree.
    
    This function demonstrates the complete flow of a single move generation:
    1. Start with empty board
    2. Get policy for current position
    3. Find top 3 policy moves
    4. Evaluate each child position
    5. Run batched minimax search
    6. Verify results make sense
    
    This is useful for:
    - Understanding the search process
    - Debugging unexpected behavior
    - Verifying move legality and value reasonableness
    - Checking that batched inference works correctly
    """
    print("=== Manual Verification: 3x2 Search Tree ===")
    
    # Initialize model
    model = SimpleModelInference(
        "checkpoints/hyperparameter_tuning/loss_weight_sweep_exp0_bs256_98f719_20250724_233408/epoch2_mini16.pt.gz"
    )
    
    # Create initial state
    state = HexGameState()
    print(f"Initial state: {state.to_trmph()}")
    print(f"Current player: {state.current_player}")
    
    # Get legal moves
    legal_moves = state.get_legal_moves()
    print(f"Legal moves: {len(legal_moves)}")
    
    # Get policy for current position
    policy, value = model.simple_infer(state.board)
    print(f"Current position value: {value:.3f}")
    
    # Get top 3 policy moves
    policy_top_moves = get_top_k_moves_with_probs(policy, legal_moves, state.board.shape[0], k=3, temperature=1.0)
    print(f"Top 3 policy moves:")
    for i, (move, prob) in enumerate(policy_top_moves):
        print(f"  {i+1}. {move} - prob: {prob:.3f}")
    
    # Create child states for top 3 moves
    child_states = []
    child_values = []
    for move, prob in policy_top_moves:
        child_state = state.make_move(move[0], move[1])
        child_states.append(child_state)
        _, child_value = model.simple_infer(child_state.board)
        child_values.append(child_value)
        print(f"  Child {move}: value = {child_value:.3f}")
    
    # Now test the batched search
    print(f"\n=== Testing Batched Search ===")
    
    # Run batched search
    best_move, best_value = minimax_policy_value_search_with_batching(
        state=state,
        model=model,
        widths=[3, 2],
        temperature=1.0
    )
    
    print(f"Batched search result:")
    print(f"  Best move: {best_move}")
    print(f"  Best value: {best_value:.3f}")
    
    # Verify the result makes sense
    print(f"\n=== Verification ===")
    
    # Check if best move is among top 3 policy moves
    best_move_in_top3 = any(move == best_move for move, _ in policy_top_moves)
    print(f"Best move in top 3 policy moves: {best_move_in_top3}")
    
    # Check if value is reasonable
    print(f"Best value reasonable: {-1.0 <= best_value <= 1.0}")
    
    # Check if move is legal
    best_move_legal = best_move in legal_moves
    print(f"Best move is legal: {best_move_legal}")
    
    # Apply the move and check game state
    new_state = state.make_move(best_move[0], best_move[1])
    print(f"After move: {new_state.to_trmph()}")
    print(f"Game over: {new_state.game_over}")
    print(f"Winner: {new_state.winner}")
    
    print(f"\n✅ Manual verification completed!")

def test_position_collector_manual():
    """
    Manually test PositionCollector with known positions.
    
    This function tests the PositionCollector class directly to ensure:
    1. It can handle multiple policy requests
    2. Results are correctly mapped to callbacks
    3. Batched results match individual results
    
    This is useful for:
    - Debugging PositionCollector issues
    - Verifying callback mechanism works
    - Ensuring no data corruption in batching
    """
    print(f"\n=== Manual PositionCollector Test ===")
    
    # Initialize model
    model = SimpleModelInference(
        "checkpoints/hyperparameter_tuning/loss_weight_sweep_exp0_bs256_98f719_20250724_233408/epoch2_mini16.pt.gz"
    )
    
    # Create test positions
    state1 = HexGameState()
    state2 = state1.make_move(6, 6)  # Center
    state3 = state2.make_move(7, 7)  # Another center
    
    positions = [state1, state2, state3]
    
    # Get individual results for comparison
    individual_results = []
    for state in positions:
        policy, value = model.simple_infer(state.board)
        individual_results.append((policy, value))
        print(f"Individual {state.to_trmph()}: value = {value:.3f}")
    
    # Get batched results
    collector = PositionCollector(model)
    
    batched_results = []
    def make_callback(index):
        def callback(policy):
            batched_results.append((index, policy))
        return callback
    
    for i, state in enumerate(positions):
        collector.request_policy(state.board, make_callback(i))
    
    print(f"Processing {len(collector.policy_requests)} policy requests...")
    collector.process_batches()
    
    # Compare results
    print(f"Batched results: {len(batched_results)}")
    
    for i, (index, policy) in enumerate(batched_results):
        individual_policy = individual_results[index][0]
        diff = np.abs(policy - individual_policy).max()
        print(f"Position {index}: max difference = {diff}")
        assert diff < 1e-6, f"Difference too large: {diff}"
    
    print(f"✅ PositionCollector manual test passed!")

def interactive_debug_mode():
    """
    Interactive debug mode for manual inspection.
    
    This function allows you to:
    1. Create custom board positions
    2. Test specific moves
    3. Inspect intermediate results
    4. Debug specific issues
    
    Usage:
        Run this function and follow the prompts to test specific scenarios.
    """
    print(f"\n=== Interactive Debug Mode ===")
    print("This mode allows you to test specific scenarios.")
    print("Enter 'q' to quit, or follow the prompts.")
    
    model = SimpleModelInference(
        "checkpoints/hyperparameter_tuning/loss_weight_sweep_exp0_bs256_98f719_20250724_233408/epoch2_mini16.pt.gz"
    )
    
    while True:
        print(f"\nOptions:")
        print(f"1. Test empty board")
        print(f"2. Test custom position")
        print(f"3. Test specific move")
        print(f"q. Quit")
        
        choice = input("Enter choice: ").strip()
        
        if choice == 'q':
            break
        elif choice == '1':
            state = HexGameState()
            print(f"Empty board: {state.to_trmph()}")
            policy, value = model.simple_infer(state.board)
            print(f"Policy shape: {policy.shape}, Value: {value:.3f}")
        elif choice == '2':
            trmph = input("Enter TRMPH string (e.g., #13,g7): ").strip()
            try:
                state = HexGameState.from_trmph(trmph)
                print(f"Position: {state.to_trmph()}")
                print(f"Legal moves: {len(state.get_legal_moves())}")
                policy, value = model.simple_infer(state.board)
                print(f"Value: {value:.3f}")
            except Exception as e:
                print(f"Error: {e}")
        elif choice == '3':
            trmph = input("Enter TRMPH string: ").strip()
            move_input = input("Enter move (row,col): ").strip()
            try:
                state = HexGameState.from_trmph(trmph)
                row, col = map(int, move_input.split(','))
                new_state = state.make_move(row, col)
                print(f"After move: {new_state.to_trmph()}")
                print(f"Game over: {new_state.game_over}")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Invalid choice")

if __name__ == "__main__":
    print("Manual verification script for batched inference implementation")
    print("=" * 60)
    
    # Run basic verification
    manual_verification_3x2()
    test_position_collector_manual()
    
    # Ask if user wants interactive mode
    print(f"\n" + "=" * 60)
    print("Basic verification completed successfully!")
    print("Would you like to run interactive debug mode? (y/n)")
    
    choice = input().strip().lower()
    if choice == 'y':
        interactive_debug_mode()
    
    print(f"\n🎉 All manual verifications completed successfully!")
    print(f"\nNext steps for manual verification:")
    print(f"1. Check that all moves are legal")
    print(f"2. Verify values are reasonable (-1.0 to 1.0)")
    print(f"3. Ensure game states update correctly")
    print(f"4. Compare with individual inference results")
    print(f"5. Test with different search widths and temperatures")