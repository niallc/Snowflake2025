#!/usr/bin/env python3
"""
Generate a markdown file showing the training data extracted from a sample Hex game.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from hex_ai.data_utils import extract_training_examples_from_game, tensor_to_rowcol, display_board

# Use a game with asymmetric moves to better show the board state
TRMPH_LINK = "http://www.trmph.com/hex/board#13,a1b2c3a4"
WINNER = "1"  # "1" for blue, "2" for red
OUTPUT_MD = "sample_game_training_output.md"

# Helper to get move from policy target
def policy_to_move(policy):
    if policy is None:
        return "(None)"
    idx = np.argmax(policy)
    # Use the consistent coordinate conversion from data_utils
    from hex_ai.data_utils import tensor_to_trmph
    move = tensor_to_trmph(idx)
    return f"{move} (tensor idx {idx})"

def main():
    examples = extract_training_examples_from_game(TRMPH_LINK, WINNER)
    
    with open(OUTPUT_MD, "w") as f:
        f.write(f"# Training Data Extraction for Sample Game\n\n")
        f.write(f"**Original trmph link:** [{TRMPH_LINK}]({TRMPH_LINK})\n\n")
        f.write(f"**Winner:** {'Blue' if WINNER == '1' else 'Red'}\n\n")
        
        for i, (board, policy, value) in enumerate(examples):
            f.write(f"## Position {i}\n\n")
            f.write("### Board State\n")
            f.write("```\n")
            
            # Use the actual display_board function to get proper output
            board_display = display_board(board, format_type="visual")
            f.write(board_display + "\n")
            
            f.write("```\n\n")
            f.write(f"**Policy target:** {policy_to_move(policy)}\n\n")
            f.write(f"**Value target:** {value}\n\n")
            
            # Add some analysis
            if policy is not None:
                f.write(f"**Analysis:** From this position, the model should predict the next move.\n\n")
            else:
                f.write(f"**Analysis:** This is the final position - no next move to predict.\n\n")
    
    print(f"Wrote output to {OUTPUT_MD}")
    print(f"Game: {TRMPH_LINK}")
    print(f"Winner: {'Blue' if WINNER == '1' else 'Red'}")
    print(f"Generated {len(examples)} training examples")

if __name__ == "__main__":
    main() 