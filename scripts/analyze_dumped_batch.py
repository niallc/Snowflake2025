import pickle
import torch
import numpy as np
import argparse
from hex_ai.utils import format_conversion as fc
from hex_ai.inference.board_display import display_hex_board

try:
    from IPython import embed as ipy_embed
except ImportError:
    ipy_embed = None

def main(file_path, max_samples=6, interactive=True):
    """
    Load and analyze a dumped training batch for value head debugging.
    Prints up to max_samples, returns a dict with raw and reformatted data.
    If interactive=True, drops into an interactive session at the end.
    """
    with open(file_path, 'rb') as f:
        batch = pickle.load(f)
    boards = batch['boards']  # shape: (batch, 3, N, N)
    policies = batch['policies']
    values = batch['values']

    boards_np = boards.numpy() if isinstance(boards, torch.Tensor) else np.array(boards)
    policies_np = policies.numpy() if isinstance(policies, torch.Tensor) else np.array(policies)
    values_np = values.numpy() if isinstance(values, torch.Tensor) else np.array(values)

    N = boards_np.shape[-1]
    print(f"Loaded batch: {boards_np.shape[0]} samples, board size {N}x{N}")

    reformatted = []
    for i in range(min(max_samples, boards_np.shape[0])):
        board3 = boards_np[i]  # (3, N, N)
        policy = policies_np[i]  # (169,)
        value = values_np[i][0] if values_np[i].shape else values_np[i]
        player_channel = board3[2]
        unique_players = np.unique(player_channel)
        # For display, use only the first two channels
        board2 = board3[:2]
        board_nxn = fc.board_2nxn_to_nxn(torch.tensor(board2))
        # Policy target
        policy_idx = np.argmax(policy)
        row, col = fc.tensor_to_rowcol(policy_idx)
        trmph_move = fc.tensor_to_trmph(policy_idx)
        print(f"\n=== Sample {i} ===")
        print(f"Value label: {value}")
        print(f"Player-to-move channel unique values: {unique_players}")
        print(f"Policy target: index {policy_idx}, (row,col)=({row},{col}), trmph={trmph_move}")
        print("Board:")
        display_hex_board(board_nxn, highlight_move=(row, col))
        reformatted.append({
            'board3': board3,
            'board2': board2,
            'board_nxn': board_nxn,
            'policy': policy,
            'policy_idx': policy_idx,
            'row': row,
            'col': col,
            'trmph_move': trmph_move,
            'value': value,
            'player_channel': player_channel,
            'unique_players': unique_players,
        })

    all_data = {
        'boards_np': boards_np,
        'policies_np': policies_np,
        'values_np': values_np,
        'reformatted': reformatted,
    }

    print("\n--- Interactive analysis ready!")
    print("Returned object: all_data (dict with keys: 'boards_np', 'policies_np', 'values_np', 'reformatted')")
    print("Suggested commands:")
    print("  all_data['reformatted'][0]  # Inspect first sample summary")
    print("  all_data['boards_np'].shape  # Raw batch shape")
    print("  all_data['reformatted'][0]['board_nxn']  # Numpy board for sample 0")
    if interactive:
        local_vars = dict(**all_data)
        if ipy_embed is not None:
            ipy_embed(user_ns=local_vars)
        else:
            import code
            code.interact(local=local_vars)
    return all_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze dumped training batch for value head debugging.")
    parser.add_argument('--file', type=str, required=True, help='Path to dumped batch pickle file')
    parser.add_argument('--max', type=int, default=6, help='Number of samples to print')
    args = parser.parse_args()
    main(args.file, max_samples=args.max, interactive=True) 