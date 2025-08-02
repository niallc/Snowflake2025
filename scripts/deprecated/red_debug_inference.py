#!/usr/bin/env python3
"""
Debug script to test inference pipeline step by step.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.data_utils import create_board_from_moves, preprocess_example_for_model, get_player_to_move_from_board
from hex_ai.utils import format_conversion as fc
from hex_ai.config import (
    BLUE_PLAYER, RED_PLAYER, BLUE_CHANNEL, RED_CHANNEL,
    TRAINING_BLUE_WIN, TRAINING_RED_WIN
)

# TODO: Check carefully the use of BLUE* and RED* constants.
#       to make sure they are used in the right contexts.
#       Typically 0=Blue, 1=Red but trmph winner is 1=Blue, 2=Red
#       and on nxn boards also 1=Blue, 2=Red.
#       Also on tensors passed to the network all pieces are 1, the player is set by channel.

def debug_blue_win_position():
    """Debug the blue win position step by step."""
    
    # The blue win position
    trmph = "https://trmph.com/hex/board#13,g1a7g2b7g3c7g4d7g5e7g6f7g8h7g9i7g10j7g11k7g12l7g13m7g7"
    
    print("=== DEBUGGING BLUE WIN POSITION ===")
    print(f"TRMPH: {trmph}")
    
    # Step 1: Parse TRMPH to moves
    bare_moves = fc.strip_trmph_preamble(trmph)
    moves = fc.split_trmph_moves(bare_moves)
    print(f"\n1. Parsed moves: {moves}")
    print(f"   Number of moves: {len(moves)}")
    
    # Step 2: Create board using same function as training
    board_2ch = create_board_from_moves(moves)
    print(f"\n2. Board shape: {board_2ch.shape}")
    print(f"   Blue pieces: {np.sum(board_2ch[BLUE_CHANNEL])}")  # Use BLUE_CHANNEL constant
    print(f"   Red pieces: {np.sum(board_2ch[RED_CHANNEL])}")    # Use RED_CHANNEL constant
    
    # Step 3: Test finished position detection
    infer = SimpleModelInference("checkpoints/final_only/loss_weight_sweep_exp0_do0_pw0.2_794e88_20250723_230725/epoch1_mini3.pt", device="mps")
    is_finished, winner = infer._is_finished_position(board_2ch)
    print(f"\n3. Finished position detection:")
    print(f"   Is finished: {is_finished}")
    print(f"   Winner: {winner} ({'BLUE' if winner == BLUE_PLAYER else 'RED' if winner == RED_PLAYER else 'UNKNOWN'})")
    
    # Step 4: Test player-to-move logic
    normal_player = get_player_to_move_from_board(board_2ch)
    print(f"\n4. Player-to-move logic:")
    print(f"   Normal logic (loser): {normal_player} ({'BLUE' if normal_player == BLUE_PLAYER else 'RED'})")
    print(f"   Finished logic (winner): {winner} ({'BLUE' if winner == BLUE_PLAYER else 'RED'})")
    
    # Step 5: Create board with correct player channel
    board_3ch = infer._create_board_with_correct_player_channel(board_2ch)
    print(f"\n5. Final board shape: {board_3ch.shape}")
    print(f"   Player-to-move channel value: {board_3ch[2, 0, 0].item()}")
    
    # Step 6: COMPREHENSIVE COMPARISON - Training vs Inference
    print(f"\n6. COMPREHENSIVE COMPARISON - Training vs Inference")
    
    # Method A: Use exact same preprocessing as training
    example = {
        'board': board_2ch,
        'policy': None,  # Final position, no next move
        'value': TRAINING_BLUE_WIN  # Blue win = 0.0
    }
    board_3ch_training, policy_tensor, value_tensor = preprocess_example_for_model(example, use_uniform_policy=False)
    print(f"   Method A (Training pipeline):")
    print(f"     Board shape: {board_3ch_training.shape}")
    print(f"     Player-to-move channel: {board_3ch_training[2, 0, 0].item()}")
    print(f"     Value tensor: {value_tensor}")
    
    # Method B: Use custom finished position logic
    board_3ch_inference = infer._create_board_with_correct_player_channel(board_2ch)
    print(f"   Method B (Inference pipeline):")
    print(f"     Board shape: {board_3ch_inference.shape}")
    print(f"     Player-to-move channel: {board_3ch_inference[2, 0, 0].item()}")
    
    # Compare the tensors
    tensors_match = torch.allclose(board_3ch_training, board_3ch_inference, atol=1e-6)
    print(f"   Tensors match: {tensors_match}")
    
    if not tensors_match:
        print(f"   DIFFERENCE DETECTED!")
        diff = torch.abs(board_3ch_training - board_3ch_inference)
        print(f"   Max difference: {torch.max(diff).item()}")
        print(f"   Difference in player channel: {torch.max(diff[2]).item()}")
    
    # Step 7: Run inference with both methods
    print(f"\n7. Running inference with both methods...")
    
    # Method A inference
    if infer.is_legacy:
        input_tensor_a = board_3ch_training[:2]
    else:
        input_tensor_a = board_3ch_training
    policy_logits_a, value_logit_a = infer.model.predict(input_tensor_a)
    value_a = torch.sigmoid(value_logit_a).item()
    
    # Method B inference
    if infer.is_legacy:
        input_tensor_b = board_3ch_inference[:2]
    else:
        input_tensor_b = board_3ch_inference
    policy_logits_b, value_logit_b = infer.model.predict(input_tensor_b)
    value_b = torch.sigmoid(value_logit_b).item()
    
    print(f"   Method A (Training pipeline) result: {value_a:.4f}")
    print(f"   Method B (Inference pipeline) result: {value_b:.4f}")
    print(f"   Results match: {abs(value_a - value_b) < 1e-6}")
    
    # Step 8: Final inference using the CLI method
    print(f"\n8. Final inference using CLI method...")
    policy_probs, value, raw_value = infer.infer(trmph)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Expected value (blue win): {TRAINING_BLUE_WIN}")  # Use TRAINING_BLUE_WIN constant
    print(f"Model output value: {value:.4f}")
    print(f"Raw value logit: {raw_value:.4f}")
    print(f"Model interpretation: {value*100:.1f}% blue wins")
    
    # Step 9: Check if this matches training data
    print(f"\n=== TRAINING DATA CHECK ===")
    print(f"Training data for blue win positions should have value={TRAINING_BLUE_WIN}")
    print(f"Model should learn to output ~{TRAINING_BLUE_WIN} for blue wins")
    print(f"Current output: {value:.4f} - this is {'CORRECT' if value < 0.1 else 'INCORRECT'}")
    
    return value, raw_value

if __name__ == "__main__":
    debug_blue_win_position() 