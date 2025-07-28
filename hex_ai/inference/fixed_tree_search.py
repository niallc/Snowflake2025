import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
import torch

# Assume HexGameState and SimpleModelInference are imported from the appropriate modules
from hex_ai.inference.game_engine import HexGameState
from hex_ai.value_utils import temperature_scaled_softmax
from hex_ai.config import BLUE_PLAYER, RED_PLAYER

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MinimaxSearchNode:
    """Represents a node in the minimax search tree for easier debugging and testing."""
    
    def __init__(self, state: HexGameState, depth: int, path: List[Tuple[int, int]] = None):
        self.state = state
        self.depth = depth
        self.path = path or []
        self.children: Dict[Tuple[int, int], MinimaxSearchNode] = {}
        self.value: Optional[float] = None
        self.best_move: Optional[Tuple[int, int]] = None
        self.is_maximizing: bool = (state.current_player == BLUE_PLAYER)  # Blue (0) maximizes, Red (1) minimizes
        
    def __str__(self):
        return f"Node(depth={self.depth}, player={'Blue' if self.state.current_player == 0 else 'Red'}, " \
               f"maximizing={self.is_maximizing}, value={self.value}, path={self.path})"


def get_topk_moves(state: HexGameState, model, k: int, 
                   temperature: float = 1.0) -> List[Tuple[int, int]]:
    """
    Sample k moves from the policy distribution with temperature scaling.
    
    Args:
        state: Current game state
        model: Model for inference
        k: Number of moves to sample
        temperature: Temperature for sampling (0.0 = deterministic top-k, higher = more random)
        
    Returns:
        List of k sampled moves
    """
    policy_logits, _ = model.infer(state.board)
    policy_probs = temperature_scaled_softmax(policy_logits, temperature)
    legal_moves = state.get_legal_moves()
    
    if not legal_moves:
        return []
    
    # Convert moves to indices and get probabilities
    # TODO: Is legal_policy checking for legal moves? 
    #       Legal moves have value 0, where is this being checked?
    # TODO: This logic should be refactored into a reusable utility function.
    move_indices = [row * state.board.shape[0] + col for row, col in legal_moves]
    legal_policy = np.array([policy_probs[idx] for idx in move_indices])
    
    # Normalize legal move probabilities to sum to 1
    legal_policy_sum = np.sum(legal_policy)
    if legal_policy_sum > 0:
        legal_policy = legal_policy / legal_policy_sum
    else:
        # If all probabilities are 0, use uniform distribution
        legal_policy = np.ones(len(legal_moves)) / len(legal_moves)
    
    # Sample k moves from the policy distribution
    if temperature == 0.0 or len(legal_moves) <= k:
        # Deterministic: take top-k moves
        topk_idx = np.argsort(legal_policy)[::-1][:k]
        sampled_moves = [legal_moves[i] for i in topk_idx]
    else:
        # Stochastic: sample k moves weighted by policy probabilities
        sampled_indices = np.random.choice(len(legal_moves), size=k, replace=False, p=legal_policy)
        sampled_moves = [legal_moves[i] for i in sampled_indices]
    
    # Debug logging removed for production code
    return sampled_moves


def build_search_tree(
    root_state: HexGameState, 
    model, 
    widths: List[int], 
    temperature: float = 1.0
) -> MinimaxSearchNode:
    """Build the complete search tree up to the specified depths."""
    
    def build_node(state: HexGameState, depth: int, path: List[Tuple[int, int]]) -> MinimaxSearchNode:
        node = MinimaxSearchNode(state, depth, path)
        
        # Stop if game is over or we've reached max depth
        if state.game_over or depth >= len(widths):
            return node
        
        # Get top moves for this depth
        k = widths[depth] if depth < len(widths) else 1
        moves = get_topk_moves(state, model, k, temperature)
        
        # Build children
        for move in moves:
            child_state = state.make_move(*move)
            child_path = path + [move]
            child_node = build_node(child_state, depth + 1, child_path)
            node.children[move] = child_node
            
        return node
    
    return build_node(root_state, 0, [])


def convert_model_logit_to_minimax_value(value_logit: float, root_player: int) -> float:
    """
    Convert a raw model value logit to a minimax-friendly value from the root player's perspective.
    
    The model outputs raw logits representing log(p/(1-p)) where p is the probability of Red winning.
    This function:
    1. Applies sigmoid to convert logit to probability: sigmoid(logit) = p
    2. Converts to root player's perspective for minimax search
    
    Args:
        value_logit: Raw logit from model's value head (unbounded)
        root_player: BLUE_PLAYER (0) for Blue, RED_PLAYER (1) for Red (the player whose perspective we want)
        
    Returns:
        Minimax value in range [-1, 1] where:
        - Positive values are good for the root player
        - Negative values are bad for the root player
        - 0.0 represents neutral/equal chances
        
    Raises:
        ValueError: If root_player is not BLUE_PLAYER or RED_PLAYER
    """
    # Validate root_player
    if root_player not in (BLUE_PLAYER, RED_PLAYER):
        raise ValueError(f"root_player must be BLUE_PLAYER ({BLUE_PLAYER}) or RED_PLAYER ({RED_PLAYER}), got {root_player}")
    
    # Step 1: Convert logit to probability using sigmoid
    # TODO: Need a more complete explanation of why this gives the probability of *Red* winning.
    #       The value network was trained, Niall thought, to predict the probability of Blue winning.
    prob_red_win = torch.sigmoid(torch.tensor(value_logit)).item()
    
    # Step 2: Convert to root player's perspective
    if root_player == BLUE_PLAYER:  # Root player is Blue
        # For Blue: positive values = Blue wins (good), negative values = Red wins (bad)
        # Convert from Red's win probability to Blue's perspective
        return 1.0 - 2.0 * prob_red_win  # Maps [0,1] to [1,-1]
    else:  # Root player is Red
        # For Red: negative values = Red wins (good), positive values = Blue wins (bad)
        # Convert from Red's win probability to Red's perspective
        return 2.0 * prob_red_win - 1.0  # Maps [0,1] to [-1,1]


def evaluate_leaf_nodes(nodes: List[MinimaxSearchNode], 
    model, batch_size: int = 1000, root_player: int = None
) -> None:
    """Batch evaluate all leaf nodes from the root player's perspective."""
    leaf_nodes = []
    
    def collect_leaves(node: MinimaxSearchNode):
        if not node.children:  # Leaf node
            leaf_nodes.append(node)
        else:
            for child in node.children.values():
                collect_leaves(child)
    
    # Collect all leaf nodes
    for node in nodes:
        collect_leaves(node)
    
    # Batch evaluate
    boards = [node.state.board for node in leaf_nodes]
    values = []
    
    for i in range(0, len(boards), batch_size):
        batch = boards[i:i+batch_size]
        # TODO: This looks like it's passing the boards one at a time.
        #       The reason for batching is that networks are faster when batching.
        #       Might want an infer_batch method.
        batch_results = [model.infer(board)[1] for board in batch]
        values.extend(batch_results)
    
    # Assign values to leaf nodes, converting to root player's perspective
    for node, value in zip(leaf_nodes, values):
        # Convert raw model logit to minimax-friendly value
        node.value = convert_model_logit_to_minimax_value(value, root_player)
        
        # Debug logging with intermediate values for clarity
        prob_red_win = torch.sigmoid(torch.tensor(value)).item()
        logger.debug(f"Leaf node {node.path}: raw_logit={value:.4f}, prob_red={prob_red_win:.4f}, converted_value={node.value:.4f}")




def minimax_backup(node: MinimaxSearchNode) -> float:
    """
    Backup values from leaves to root using minimax algorithm.
    Temperature is already applied during move sampling in get_topk_moves(),
    so we always choose the best move deterministically here.
    """
    if node.value is not None:  # Leaf node
        return node.value
    
    if not node.children:  # No children (shouldn't happen if we built tree correctly)
        raise RuntimeError("minimax_backup called on a node with no value and no children (invalid tree structure)")
    
    # Recursively get values from children
    child_values = []
    for move, child in node.children.items():
        child_value = minimax_backup(child)
        child_values.append((move, child_value))
        logger.debug(f"Child {move} of {node.path}: value = {child_value}")
    
    # Since all values are now from the root player's perspective,
    # we always maximize (choose the best move for the root player)
    # Temperature is already applied during move sampling, so choose best deterministically
    best_move, best_value = max(child_values, key=lambda x: x[1])
    # Debug logging removed for production code
    
    node.value = best_value
    node.best_move = best_move
    return best_value


def print_tree_structure(node: MinimaxSearchNode, indent=0):
    """Print the complete tree structure with all nodes."""
    print(
        "  " * indent
        + f"Node: depth={node.depth}, "
        + f"player={'Blue' if node.state.current_player == BLUE_PLAYER else 'Red'}, "
        + f"maximizing={node.is_maximizing}, "
        + f"value={node.value}, "
        + f"path={node.path}"
    )
    
    for move, child in node.children.items():
        print("  " * indent + f"Move {move}:")
        print_tree_structure(child, indent + 1)


def print_all_terminal_nodes(root: MinimaxSearchNode):
    """Print all terminal nodes with their values."""
    print("\n=== ALL TERMINAL NODES ===")
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    def collect_terminals(node: MinimaxSearchNode):
        if not node.children:  # Terminal node
            val = node.value
            if val is not None:
                if val > 0:
                    val_str = f"{GREEN}{val:.4g}{RESET}"
                elif val < 0:
                    val_str = f"{RED}{val:.4g}{RESET}"
                else:
                    val_str = f"{val:.4g}"
            else:
                val_str = "None"
            print(f"Terminal: path={node.path}, value= {val_str}")
            return [node]
        terminals = []
        for child in node.children.values():
            terminals.extend(collect_terminals(child))
        return terminals

    terminals = collect_terminals(root)
    print(f"Total terminal nodes: {len(terminals)}")
    return terminals


def minimax_policy_value_search(
    state: HexGameState,
    model,
    widths: List[int],
    batch_size: int = 1000,
    use_alpha_beta: bool = True,
    temperature: float = 1.0,
    debug: bool = False,
    return_tree: bool = False
) -> Tuple[Tuple[int, int], float, Optional[MinimaxSearchNode]]:
    """
    Fixed-width, fixed-depth minimax search with alpha-beta pruning and batch evaluation at the leaves.

    Args:
        state: HexGameState (current position)
        model: SimpleModelInference (must support batch inference)
        widths: List of ints, e.g. [20, 10, 10, 5] (width at each ply)
        batch_size: Max batch size for evaluation
        use_alpha_beta: Whether to use alpha-beta pruning (not implemented in new version yet)
        temperature: Policy temperature for move selection (default 1.0)
        debug: Whether to enable debug logging
        return_tree: Whether to return the search tree for debugging

    Returns:
        best_move: (row, col) tuple for the best move at the root
        value: estimated value for the root position
        root: search tree root node (if return_tree=True, otherwise None)
    """
    if debug:
        logger.setLevel(logging.DEBUG)
    
    logger.info(f"Starting minimax search with widths {widths}, temperature {temperature}")
    logger.info(f"Root state: player {state.current_player} ({'Blue' if state.current_player == 0 else 'Red'})")
    
    # Build the search tree
    root = build_search_tree(state, model, widths, temperature)
    
    # Evaluate all leaf nodes from the root player's perspective
    evaluate_leaf_nodes([root], model, batch_size, state.current_player)
    
    # Backup values to root (temperature already applied during move sampling)
    root_value = minimax_backup(root)
    
    logger.info(f"Search complete: best move = {root.best_move}, value = {root_value}")
    
    if return_tree:
        return root.best_move, root_value, root
    else:
        return root.best_move, root_value 