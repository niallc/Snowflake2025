#!/usr/bin/env python3
"""
Manual verification of minimax search with small tree.
Search widths [2,2,2] produces only 8 terminal nodes for manual inspection.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from hex_ai.inference.fixed_tree_search import (
    minimax_policy_value_search,
    MinimaxSearchNode,
    build_search_tree,
    evaluate_leaf_nodes,
    minimax_backup
)
from hex_ai.inference.game_engine import HexGameState
from hex_ai.config import BOARD_SIZE


def print_tree_structure(node: MinimaxSearchNode, indent=0):
    """Print the complete tree structure with all nodes."""
    print("  " * indent + f"Node: depth={node.depth}, player={'Blue' if node.state.current_player == 0 else 'Red'}, "
          f"maximizing={node.is_maximizing}, value={node.value}, path={node.path}")
    
    for move, child in node.children.items():
        print("  " * indent + f"Move {move}:")
        print_tree_structure(child, indent + 1)


def print_all_terminal_nodes(root: MinimaxSearchNode):
    """Print all terminal nodes with their values."""
    print("\n=== ALL TERMINAL NODES ===")
    
    def collect_terminals(node: MinimaxSearchNode):
        if not node.children:  # Terminal node
            print(f"Terminal: path={node.path}, value={node.value}")
            return [node]
        terminals = []
        for child in node.children.values():
            terminals.extend(collect_terminals(child))
        return terminals
    
    terminals = collect_terminals(root)
    print(f"Total terminal nodes: {len(terminals)}")
    return terminals


def manual_verification_test():
    """Test with search widths [2,2,2] for manual verification."""
    print("=== Manual Verification Test ===")
    print("Search widths: [2, 2, 2]")
    print("Expected: 2 * 2 * 2 = 8 terminal nodes")
    
    # Create starting position
    state = HexGameState()
    print(f"\nStarting state:")
    print(f"Current player: {state.current_player} ({'Blue' if state.current_player == 0 else 'Red'})")
    
    # We'll need a real model here - for now, create a placeholder
    # TODO: Replace with actual model loading
    print("\nNOTE: Need to load actual model for this test")
    print("Model should be loaded and passed to the functions below")
    
    # Build tree with small search
    widths = [2, 2, 2]
    print(f"\nBuilding tree with widths {widths}...")
    
    # This would need a real model:
    # root = build_search_tree(state, model, widths, temperature=0.0)
    # evaluate_leaf_nodes([root], model)
    # final_value = minimax_backup(root)
    
    print("\nTree structure would be printed here...")
    print("All terminal nodes would be listed here...")
    print("Manual verification of best move would be done here...")
    
    return None


def create_verification_script():
    """Create a script template for manual verification."""
    script_content = '''#!/usr/bin/env python3
"""
Manual verification script for minimax search.
Replace MODEL_PATH with actual model path.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hex_ai.inference.fixed_tree_search import (
    build_search_tree, evaluate_leaf_nodes, minimax_backup,
    print_tree_structure, print_all_terminal_nodes
)
from hex_ai.inference.game_engine import HexGameState
from hex_ai.inference.simple_model_inference import SimpleModelInference

def main():
    # Load your model here
    MODEL_PATH = "path/to/your/model.pth"  # Replace with actual path
    model = SimpleModelInference(MODEL_PATH)
    
    # Create starting position
    state = HexGameState()
    print(f"Starting state: player {state.current_player}")
    
    # Build tree with small search
    widths = [2, 2, 2]
    temperature = 0.0  # Deterministic policy
    
    print(f"Building tree with widths {widths}, temperature {temperature}...")
    root = build_search_tree(state, model, widths, temperature)
    
    print("\\n=== TREE STRUCTURE ===")
    print_tree_structure(root)
    
    print("\\n=== EVALUATING LEAVES ===")
    evaluate_leaf_nodes([root], model)
    
    print("\\n=== ALL TERMINAL NODES ===")
    terminals = print_all_terminal_nodes(root)
    
    print("\\n=== BACKUP VALUES ===")
    final_value = minimax_backup(root)
    print(f"Final root value: {final_value}")
    print(f"Best move: {root.best_move}")
    
    print("\\n=== MANUAL VERIFICATION ===")
    print("1. Check each terminal node value")
    print("2. Work backwards through the tree")
    print("3. Verify the chosen move matches expected value")

if __name__ == "__main__":
    main()
'''
    
    with open("manual_verification_script.py", "w") as f:
        f.write(script_content)
    
    print("Created manual_verification_script.py")
    print("Edit MODEL_PATH and run to test with your actual model")


if __name__ == "__main__":
    print("Minimax Manual Verification Test")
    print("=" * 50)
    
    manual_verification_test()
    create_verification_script()
    
    print("\n" + "=" * 50)
    print("Next steps:")
    print("1. Edit manual_verification_script.py with your model path")
    print("2. Run the script to see all 8 terminal nodes")
    print("3. Manually verify the expected best move")
    print("4. Check if the algorithm picks the correct move") 