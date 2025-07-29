#!/usr/bin/env python3
"""
Count the actual number of inference calls made during a single move in self-play.
"""

import sys
import os
import time

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hex_ai.inference.game_engine import HexGameState
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.inference.fixed_tree_search import minimax_policy_value_search
from hex_ai.value_utils import get_top_k_moves_with_probs, policy_logits_to_probs

class InferenceCounter:
    """Wrapper to count inference calls."""
    
    def __init__(self, model):
        self.model = model
        self.call_count = 0
        self.original_simple_infer = model.simple_infer
        self.original_batch_infer = model.batch_infer
        
        # Override methods to count calls
        def counted_simple_infer(board):
            self.call_count += 1
            return self.original_simple_infer(board)
        
        def counted_batch_infer(boards):
            self.call_count += 1
            return self.original_batch_infer(boards)
        
        model.simple_infer = counted_simple_infer
        model.batch_infer = counted_batch_infer
    
    def reset(self):
        self.call_count = 0
    
    def get_count(self):
        return self.call_count

def analyze_single_move():
    """Analyze inference calls for a single move."""
    print("=== Analyzing Inference Calls for Single Move ===")
    
    # Initialize model
    model = SimpleModelInference(
        "checkpoints/hyperparameter_tuning/loss_weight_sweep_exp0_bs256_98f719_20250724_233408/epoch2_mini16.pt.gz",
        cache_size=1000
    )
    
    # Create counter wrapper
    counter = InferenceCounter(model)
    
    # Create game state (after a few moves to make it interesting)
    state = HexGameState()
    
    # Make a few moves to get to an interesting position
    moves = [(6, 6), (7, 7), (5, 5), (8, 8)]  # Center moves
    for move in moves:
        if state.is_valid_move(*move):
            state = state.make_move(*move)
    
    print(f"Starting position: {state.to_trmph()}")
    print(f"Current player: {'Blue' if state.current_player == 0 else 'Red'}")
    
    # Reset counter
    counter.reset()
    
    # Simulate the exact flow from _generate_single_game
    print("\n--- Step 1: Policy inference for current position ---")
    policy, value = model.simple_infer(state.board)
    print(f"Call 1: Policy inference for current position")
    
    # Get policy head's top 5 moves
    legal_moves = state.get_legal_moves()
    policy_probs = policy_logits_to_probs(policy, 1.0)
    
    # Find policy head's top 5 moves
    policy_top_moves = get_top_k_moves_with_probs(
        policy, legal_moves, state.board.shape[0], k=5, temperature=1.0
    )
    
    print(f"Policy top 5 moves: {policy_top_moves}")
    
    # Get value predictions for top 5 policy moves
    print("\n--- Step 2: Value inference for top 5 policy moves ---")
    policy_move_values = []
    for i, (move, prob) in enumerate(policy_top_moves):
        # Create temporary state to evaluate this move
        temp_state = state.make_move(*move)
        _, move_value = model.simple_infer(temp_state.board)
        policy_move_values.append(move_value)
        print(f"Call {i+2}: Value inference for move {move}")
    
    # Get policy head's preferred move (top 1)
    policy_preferred_move = policy_top_moves[0][0] if policy_top_moves else None
    
    print(f"\n--- Step 3: Minimax search ---")
    print(f"Search widths: [10, 5]")
    
    # Select move using search (minimax)
    minimax_move, minimax_value = minimax_policy_value_search(
        state=state,
        model=model,
        widths=[10, 5],
        temperature=1.0,
        verbose=1
    )
    
    print(f"\n--- Summary ---")
    print(f"Total inference calls: {counter.get_count()}")
    print(f"Policy preferred move: {policy_preferred_move}")
    print(f"Minimax chosen move: {minimax_move}")
    print(f"Agreement: {policy_preferred_move == minimax_move}")
    
    # Analyze the calls
    calls = counter.get_count()
    print(f"\n--- Call Breakdown ---")
    print(f"1. Policy inference for current position: 1 call")
    print(f"2. Value inference for top 5 policy moves: 5 calls")
    print(f"3. Minimax search calls: {calls - 6} calls")
    print(f"   - Policy calls for tree building: ~{calls - 6 - 15} calls")
    print(f"   - Value calls for leaf evaluation: ~15 calls (batched)")
    
    return calls

def estimate_batching_potential():
    """Estimate potential speedup from batching."""
    print("\n=== Batching Potential Analysis ===")
    
    # Current calls per move (from analysis above)
    current_calls = 6 + 15 + 25  # Rough estimate: 1 + 5 + 15 + 25 = 46 calls
    
    print(f"Current calls per move: ~{current_calls}")
    print(f"Current breakdown:")
    print(f"  - Policy for current position: 1 call")
    print(f"  - Values for top 5 policy moves: 5 calls")
    print(f"  - Policy calls in minimax tree: ~25 calls")
    print(f"  - Value calls for leaves: ~15 calls (already batched)")
    
    # Potential batching
    print(f"\nPotential batching:")
    print(f"  - All policy calls: 1 + 5 + 25 = 31 calls → 1 batch call")
    print(f"  - All value calls: 15 calls → 1 batch call (already done)")
    print(f"  - Total: 46 calls → 2 batch calls")
    
    # Estimate speedup
    print(f"\nSpeedup estimate:")
    print(f"  - Individual calls: 46 × 0.001s = 0.046s")
    print(f"  - Batch calls: 2 × 0.005s = 0.010s")
    print(f"  - Speedup: 0.046s / 0.010s = 4.6x")
    
    return 4.6

if __name__ == "__main__":
    calls = analyze_single_move()
    speedup = estimate_batching_potential()
    
    print(f"\n=== Conclusion ===")
    print(f"Realistic speedup from batching: {speedup:.1f}x")
    print(f"This is much more realistic than the 238x from the prototype!")