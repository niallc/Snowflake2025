import numpy as np
from typing import List, Tuple, Optional

# Assume HexGameState and SimpleModelInference are imported from the appropriate modules
from hex_ai.inference.game_engine import HexGameState


def minimax_policy_value_search(
    state: HexGameState,
    model,
    widths: List[int],
    batch_size: int = 1000,
    use_alpha_beta: bool = True,
) -> Tuple[Tuple[int, int], float]:
    """
    Fixed-width, fixed-depth minimax search with alpha-beta pruning and batch evaluation at the leaves.

    Args:
        state: HexGameState (current position)
        model: SimpleModelInference (must support batch inference)
        widths: List of ints, e.g. [20, 10, 10, 5] (width at each ply)
        batch_size: Max batch size for evaluation
        use_alpha_beta: Whether to use alpha-beta pruning

    Returns:
        best_move: (row, col) tuple for the best move at the root
        value: estimated value for the root position
    """
    # Helper: get top-k legal moves by policy
    def get_topk_moves(state, k):
        policy_probs, _ = model.infer(state.board)
        legal_moves = state.get_legal_moves()
        move_indices = [row * state.board.shape[0] + col for row, col in legal_moves]
        legal_policy = np.array([policy_probs[idx] for idx in move_indices])
        if len(legal_policy) == 0:
            return []
        topk_idx = np.argsort(legal_policy)[::-1][:k]
        return [legal_moves[i] for i in topk_idx]

    # Recursive minimax with alpha-beta and batch leaf collection
    def minimax(state, depth, maximizing, alpha, beta, leaf_states, path):
        if state.game_over or depth == len(widths):
            leaf_states.append((state, path))
            return None  # Value will be filled in batch
        k = widths[depth]
        moves = get_topk_moves(state, k)
        if not moves:
            leaf_states.append((state, path))
            return None
        best_value = -np.inf if maximizing else np.inf
        best_move = None
        for move in moves:
            child = state.make_move(*move)
            child_path = path + [move]
            val = minimax(child, depth + 1, not maximizing, alpha, beta, leaf_states, child_path)
            # Value will be filled in after batch eval
            if val is not None:
                if maximizing:
                    if val > best_value:
                        best_value = val
                        best_move = move
                    if use_alpha_beta:
                        alpha = max(alpha, best_value)
                        if beta <= alpha:
                            break
                else:
                    if val < best_value:
                        best_value = val
                        best_move = move
                    if use_alpha_beta:
                        beta = min(beta, best_value)
                        if beta <= alpha:
                            break
        return best_value

    # Collect all leaf states
    leaf_states = []  # List of (state, path)
    root = state
    maximizing = (state.current_player == 0)  # Blue maximizes, Red minimizes
    minimax(root, 0, maximizing, -np.inf, np.inf, leaf_states, [])

    # Batch evaluate all leaves
    boards = [leaf[0].board for leaf in leaf_states]
    values = []
    for i in range(0, len(boards), batch_size):
        batch = boards[i:i+batch_size]
        # Model must support batch inference
        batch_results = [model.infer(board)[1] for board in batch]
        values.extend(batch_results)

    # Assign values back to leaves
    for idx, (_, path) in enumerate(leaf_states):
        leaf_states[idx] = (values[idx], path)

    # Now, walk the tree again to propagate values up
    def minimax_value(depth, maximizing, path):
        # Find all leaves with this path prefix
        matching = [v for v, p in leaf_states if p[:depth] == path]
        if depth == len(widths) or not matching:
            return matching[0] if matching else 0.0
        k = widths[depth]
        # For each possible move at this depth, get value
        move_values = []
        for move in set(p[depth] for _, p in leaf_states if len(p) > depth and p[:depth] == path):
            v = minimax_value(depth + 1, not maximizing, path + [move])
            move_values.append((move, v))
        if not move_values:
            return 0.0
        if maximizing:
            return max(v for _, v in move_values)
        else:
            return min(v for _, v in move_values)

    # At root, pick the move with best value
    move_values = []
    for move in set(p[0] for _, p in leaf_states if p):
        v = minimax_value(1, not maximizing, [move])
        move_values.append((move, v))
    if not move_values:
        return None, 0.0
    if maximizing:
        best_move, best_value = max(move_values, key=lambda x: x[1])
    else:
        best_move, best_value = min(move_values, key=lambda x: x[1])
    return best_move, best_value 