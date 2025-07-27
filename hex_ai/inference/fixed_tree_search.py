import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

# Assume HexGameState and SimpleModelInference are imported from the appropriate modules
from hex_ai.inference.game_engine import HexGameState
from hex_ai.value_utils import temperature_scaled_softmax  # <-- Add this import

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
        self.is_maximizing: bool = (state.current_player == 0)  # Blue (0) maximizes, Red (1) minimizes
        
    def __str__(self):
        return f"Node(depth={self.depth}, player={'Blue' if self.state.current_player == 0 else 'Red'}, " \
               f"maximizing={self.is_maximizing}, value={self.value}, path={self.path})"


def get_topk_moves(state: HexGameState, model, k: int, 
                   temperature: float = 1.0) -> List[Tuple[int, int]]:
    """Get top-k legal moves by policy probability."""
    policy_logits, _ = model.infer(state.board)
    policy_probs = temperature_scaled_softmax(policy_logits, temperature)
    legal_moves = state.get_legal_moves()
    
    if not legal_moves:
        return []
    
    # Convert moves to indices and get probabilities
    move_indices = [row * state.board.shape[0] + col for row, col in legal_moves]
    legal_policy = np.array([policy_probs[idx] for idx in move_indices])
    
    # Get top-k moves
    topk_idx = np.argsort(legal_policy)[::-1][:k]
    return [legal_moves[i] for i in topk_idx]


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


def evaluate_leaf_nodes(nodes: List[MinimaxSearchNode], model, batch_size: int = 1000, root_player: int = None) -> None:
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
        batch_results = [model.infer(board)[1] for board in batch]
        values.extend(batch_results)
    
    # Assign values to leaf nodes, converting to root player's perspective
    for node, value in zip(leaf_nodes, values):
        # The model outputs probability of Red winning (1.0 = Red wins, 0.0 = Blue wins)
        # We need to convert this to a value from the root player's perspective
        if root_player == 0:  # Root player is Blue
            # For Blue, positive values are good (Blue wins), negative are bad (Red wins)
            # Convert from Red's win probability to Blue's perspective
            node.value = 1.0 - 2.0 * value  # Maps [0,1] to [1,-1]
        else:  # Root player is Red
            # For Red, negative values are good (Red wins), positive are bad (Blue wins)
            # Convert from Red's win probability to Red's perspective
            node.value = 2.0 * value - 1.0  # Maps [0,1] to [-1,1]
        
        logger.debug(f"Leaf node {node.path}: raw_value={value:.4f}, converted_value={node.value:.4f}")


def minimax_backup(node: MinimaxSearchNode) -> float:
    """Backup values from leaves to root using minimax algorithm."""
    if node.value is not None:  # Leaf node
        return node.value
    
    if not node.children:  # No children (shouldn't happen if we built tree correctly)
        return 0.0
    
    # Recursively get values from children
    child_values = []
    for move, child in node.children.items():
        child_value = minimax_backup(child)
        child_values.append((move, child_value))
        logger.debug(f"Child {move} of {node.path}: value = {child_value}")
    
    # Since all values are now from the root player's perspective,
    # we always maximize (choose the best move for the root player)
    best_move, best_value = max(child_values, key=lambda x: x[1])
    logger.debug(f"Node {node.path}: best move = {best_move}, value = {best_value}")
    
    node.value = best_value
    node.best_move = best_move
    return best_value


def print_tree_structure(node: MinimaxSearchNode, indent=0):
    """Print the complete tree structure with all nodes."""
    print(
        "  " * indent
        + f"Node: depth={node.depth}, "
        + f"player={'Blue' if node.state.current_player == 0 else 'Red'}, "
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
    debug: bool = False
) -> Tuple[Tuple[int, int], float]:
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

    Returns:
        best_move: (row, col) tuple for the best move at the root
        value: estimated value for the root position
    """
    if debug:
        logger.setLevel(logging.DEBUG)
    
    logger.info(f"Starting minimax search with widths {widths}, temperature {temperature}")
    logger.info(f"Root state: player {state.current_player} ({'Blue' if state.current_player == 0 else 'Red'})")
    
    # Build the search tree
    root = build_search_tree(state, model, widths, temperature)
    
    # Evaluate all leaf nodes from the root player's perspective
    evaluate_leaf_nodes([root], model, batch_size, state.current_player)
    
    # Backup values to root
    root_value = minimax_backup(root)
    
    logger.info(f"Search complete: best move = {root.best_move}, value = {root_value}")
    
    return root.best_move, root_value 