#!/usr/bin/env python3
"""
Manual verification script for minimax search.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHKPT_DIR = "checkpoints/hyperparameter_tuning/loss_weight_sweep_exp0_bs256_98f719_20250724_233408"
FILE_NAME="epoch2_mini10.pt"
MODEL_PATH = os.path.join(CHKPT_DIR, FILE_NAME)

from hex_ai.inference.fixed_tree_search import (
    build_search_tree, evaluate_leaf_nodes, minimax_backup,
    print_tree_structure, print_all_terminal_nodes
)
from hex_ai.inference.game_engine import HexGameState
from hex_ai.inference.simple_model_inference import SimpleModelInference

def main():
    # Load your model here
    model = SimpleModelInference(MODEL_PATH)
    
    # Create starting position
    state = HexGameState()
    print(f"Starting state: player {state.current_player}")
    
    # Build tree with small search
    widths = [5, 3, 2]
    temperature = 0.0  # Deterministic policy
    
    print(f"Building tree with widths {widths}, temperature {temperature}...")
    root = build_search_tree(state, model, widths, temperature)
    
    print("\n=== TREE STRUCTURE ===")
    print_tree_structure(root)
    
    print("\n=== EVALUATING LEAVES ===")
    evaluate_leaf_nodes([root], model)
    
    print("\n=== ALL TERMINAL NODES ===")
    terminals = print_all_terminal_nodes(root)
    
    print("\n=== BACKUP VALUES ===")
    final_value = minimax_backup(root)
    print(f"Final root value: {final_value}")
    print(f"Best move: {root.best_move}")
    
    print("\n=== MANUAL VERIFICATION ===")
    print("1. Check each terminal node value")
    print("2. Work backwards through the tree")
    print("3. Verify the chosen move matches expected value")

if __name__ == "__main__":
    main()
