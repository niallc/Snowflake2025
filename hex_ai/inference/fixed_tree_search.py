import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
import logging
import torch

# Assume HexGameState and SimpleModelInference are imported from the appropriate modules
from hex_ai.inference.game_engine import HexGameState
from hex_ai.value_utils import temperature_scaled_softmax, get_top_k_moves_with_probs
from hex_ai.config import BLUE_PLAYER, RED_PLAYER

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionCollector:
    """Collects board positions during tree building for batch processing."""
    
    def __init__(self, model):
        self.model = model
        self.policy_requests = []  # List of (board, callback) tuples
        self.value_requests = []   # List of (board, callback) tuples
        
    def request_policy(self, board, callback: Callable):
        """Add a policy request to be processed later."""
        self.policy_requests.append((board, callback))
        
    def request_value(self, board, callback: Callable):
        """Add a value request to be processed later."""
        self.value_requests.append((board, callback))
    
    def process_batches(self):
        """Process all collected positions in batches."""
        # Process policy requests
        if self.policy_requests:
            boards = [req[0] for req in self.policy_requests]
            policies, _ = self.model.batch_infer(boards)
            for (board, callback), policy in zip(self.policy_requests, policies):
                callback(policy)
            self.policy_requests.clear()
        
        # Process value requests
        if self.value_requests:
            boards = [req[0] for req in self.value_requests]
            _, values = self.model.batch_infer(boards)
            for (board, callback), value in zip(self.value_requests, values):
                callback(value)
            self.value_requests.clear()


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
    policy_logits, _ = model.simple_infer(state.board)
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


def get_topk_moves_from_policy(policy_logits: np.ndarray, state: HexGameState, k: int, 
                              temperature: float = 1.0) -> List[Tuple[int, int]]:
    """
    Get top-k moves from policy logits (for use with batched inference).
    
    Args:
        policy_logits: Raw policy logits
        state: Current game state
        k: Number of moves to sample
        temperature: Temperature for sampling
        
    Returns:
        List of k sampled moves
    """
    legal_moves = state.get_legal_moves()
    if not legal_moves:
        return []
    
    # Use existing utility function
    moves_with_probs = get_top_k_moves_with_probs(
        policy_logits, legal_moves, state.board.shape[0], k, temperature
    )
    
    # Extract just the moves (not the probabilities)
    return [move for move, _ in moves_with_probs]


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


def build_search_tree_with_collection(
    root_state: HexGameState, 
    model, 
    widths: List[int], 
    temperature: float = 1.0,
    collector: PositionCollector = None
) -> MinimaxSearchNode:
    """Build search tree while collecting positions for batch processing."""
    
    def build_node(state: HexGameState, depth: int, path: List[Tuple[int, int]]) -> MinimaxSearchNode:
        node = MinimaxSearchNode(state, depth, path)
        
        # Stop if game is over or we've reached max depth
        if state.game_over or depth >= len(widths):
            return node
        
        # Get top moves for this depth
        k = widths[depth] if depth < len(widths) else 1
        
        if collector is not None:
            # Collect policy request instead of immediate inference
            def policy_callback(policy_logits):
                # Use policy to get top moves
                moves = get_topk_moves_from_policy(policy_logits, state, k, temperature)
                # Build children
                for move in moves:
                    child_state = state.make_move(*move)
                    child_path = path + [move]
                    child_node = build_node(child_state, depth + 1, child_path)
                    node.children[move] = child_node
            
            collector.request_policy(state.board, policy_callback)
        else:
            # Fallback to original behavior
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
    
    The value head predicts Red's win probability because Red wins are labeled as 1.0 in training.
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
    # The value head predicts Red's win probability because Red wins are labeled as 1.0 in training.
    # sigmoid(value_logit) gives the probability that Red wins.
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
        # Use efficient batch inference instead of individual calls
        _, batch_values = model.batch_infer(batch)
        values.extend(batch_values)
    
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
    print("  " * indent + f"Node: depth={node.depth}, player={'Blue' if node.state.current_player == 0 else 'Red'}, "
          f"maximizing={node.is_maximizing}, value={node.value}, path={node.path}")
    
    for move, child in node.children.items():
        print("  " * indent + f"Move {move}:")
        print_tree_structure(child, indent + 1)


def print_all_terminal_nodes(root: MinimaxSearchNode):
    """Print all terminal nodes for manual verification."""
    terminals = []
    
    def collect_terminals(node: MinimaxSearchNode):
        if not node.children:  # Terminal node
            terminals.append(node)
        else:
            for child in node.children.values():
                collect_terminals(child)
    
    collect_terminals(root)
    
    print(f"Found {len(terminals)} terminal nodes:")
    for i, node in enumerate(terminals):
        print(f"  {i+1}. Path: {node.path}, Value: {node.value}")
    
    return terminals


def minimax_policy_value_search(
    state: HexGameState,
    model,
    widths: List[int],
    batch_size: int = 1000,
    use_alpha_beta: bool = True,
    temperature: float = 1.0,
    debug: bool = False,
    return_tree: bool = False,
    verbose: int = 0
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
        verbose: Verbosity level (0: silent, 1+: show info logs)

    Returns:
        best_move: (row, col) tuple for the best move at the root
        value: estimated value for the root position
        root: search tree root node (if return_tree=True, otherwise None)
    """
    if debug:
        logger.setLevel(logging.DEBUG)
    elif verbose <= 1:
        logger.setLevel(logging.WARNING)  # Suppress INFO logs when not verbose
    else:
        logger.setLevel(logging.INFO)
    
    if verbose >= 2:
        logger.info(f"Starting minimax search with widths {widths}, temperature {temperature}")
        logger.info(f"Root state: player {state.current_player} ({'Blue' if state.current_player == 0 else 'Red'})")
    
    # Build the search tree
    root = build_search_tree(state, model, widths, temperature)
    
    # Evaluate all leaf nodes from the root player's perspective
    evaluate_leaf_nodes([root], model, batch_size, state.current_player)
    
    # Backup values to root (temperature already applied during move sampling)
    root_value = minimax_backup(root)
    
    if verbose >= 2:
        logger.info(f"Search complete: best move = {root.best_move}, value = {root_value}")
    
    if return_tree:
        return root.best_move, root_value, root
    else:
        return root.best_move, root_value


def minimax_policy_value_search_with_batching(
    state: HexGameState,
    model,
    widths: List[int],
    batch_size: int = 1000,
    use_alpha_beta: bool = True,
    temperature: float = 1.0,
    debug: bool = False,
    return_tree: bool = False,
    verbose: int = 0
) -> Tuple[Tuple[int, int], float, Optional[MinimaxSearchNode]]:
    """
    Fixed-width, fixed-depth minimax search with batched inference for all policy calls.

    Args:
        state: HexGameState (current position)
        model: SimpleModelInference (must support batch inference)
        widths: List of ints, e.g. [20, 10, 10, 5] (width at each ply)
        batch_size: Max batch size for evaluation
        use_alpha_beta: Whether to use alpha-beta pruning (not implemented in new version yet)
        temperature: Policy temperature for move selection (default 1.0)
        debug: Whether to enable debug logging
        return_tree: Whether to return the search tree for debugging
        verbose: Verbosity level (0: silent, 1+: show info logs)

    Returns:
        best_move: (row, col) tuple for the best move at the root
        value: estimated value for the root position
        root: search tree root node (if return_tree=True, otherwise None)
    """
    if debug:
        logger.setLevel(logging.DEBUG)
    elif verbose <= 1:
        logger.setLevel(logging.WARNING)  # Suppress INFO logs when not verbose
    else:
        logger.setLevel(logging.INFO)
    
    if verbose >= 2:
        logger.info(f"Starting batched minimax search with widths {widths}, temperature {temperature}")
        logger.info(f"Root state: player {state.current_player} ({'Blue' if state.current_player == 0 else 'Red'})")
    
    # Create position collector for batched inference
    collector = PositionCollector(model)
    
    # Build the search tree with position collection
    root = build_search_tree_with_collection(state, model, widths, temperature, collector)
    
    # Process all collected batches
    collector.process_batches()
    
    # Evaluate all leaf nodes from the root player's perspective
    evaluate_leaf_nodes([root], model, batch_size, state.current_player)
    
    # Backup values to root (temperature already applied during move sampling)
    root_value = minimax_backup(root)
    
    if verbose >= 2:
        logger.info(f"Batched search complete: best move = {root.best_move}, value = {root_value}")
    
    if return_tree:
        return root.best_move, root_value, root
    else:
        return root.best_move, root_value 