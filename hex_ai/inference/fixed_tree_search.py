import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
import logging
import torch
import psutil
import sys
import threading
import time

# Assume HexGameState and SimpleModelInference are imported from the appropriate modules
from hex_ai.inference.game_engine import HexGameState
from hex_ai.value_utils import temperature_scaled_softmax, get_top_k_moves_with_probs, sample_moves_from_policy
from hex_ai.enums import Player

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Memory monitoring constants
MAX_POSITIONS = 10_000_000  # Exit if we try to collect more than 10M positions
MEMORY_WARNINGS = [2, 6, 10, 14, 18]  # GB thresholds for warnings
MEMORY_EXIT_THRESHOLD = 20  # GB threshold for exit


class PositionCollector:
    """
    Collects board positions for batched inference to improve efficiency.
    
    This class is designed to be used in a single-threaded context where:
    1. Multiple positions are collected during tree building
    2. All positions are processed in iterative batches (waves)
    3. Results are distributed back to the appropriate callbacks
    
    HOW IT WORKS:
    1. During tree building, instead of immediately calling model.infer() for each position,
       we collect the board and a callback function using request_policy()
    2. After the tree structure is built, we call process_batches() iteratively which:
       - Processes current wave of policy requests in batches
       - Calls model.batch_infer() for the current wave
       - Calls the appropriate callback for each result
       - New requests from callbacks are queued for the next wave
    3. This continues until no more requests are pending (iterative batching)
    
    BENEFITS:
    - Reduces GPU kernel launches from O(n) to O(1) where n is the number of positions
    - Better GPU utilization through larger batch sizes
    - Reduced overhead from individual inference calls
    
    CONCERNS:
    - Memory usage: All requests are held in memory until processed
    - Thread safety: Not thread-safe (single-threaded usage only)
    - Complexity: Callback mechanism can be error-prone
    
    USAGE PATTERN:
        collector = PositionCollector(model)
        # Build tree while collecting requests
        root = build_search_tree_with_collection(state, model, widths, collector)
        # Process all collected requests in iterative batches
        while collector.has_pending_requests():
            collector.process_batches()
    """
    
    def __init__(self, model, verbose: int = 0):
        """
        Initialize the position collector.
        
        Args:
            model: Model instance that supports batch_infer() method
            verbose: Verbosity level for debugging output
            
        Raises:
            RuntimeError: If called from a multi-threaded context
        """
        # Thread safety check: allow single worker thread but prevent multi-threading
        current_thread = threading.current_thread()
        main_thread = threading.main_thread()
        
        # Allow if we're in the main thread OR if we're in a single worker thread
        # (ThreadPoolExecutor with max_workers=1 creates one worker thread)
        if current_thread is main_thread:
            # Main thread - always allowed
            pass
        elif current_thread.name.startswith('ThreadPoolExecutor-'):
            # Check if this is a single worker thread (ThreadPoolExecutor-0_0)
            # or a multi-threaded worker (ThreadPoolExecutor-0_1, ThreadPoolExecutor-0_2, etc.)
            if '_0' in current_thread.name and current_thread.name.endswith('_0'):
                # Single worker thread - acceptable
                pass
            else:
                # Multi-threaded worker - not allowed
                error_msg = (
                    "CRITICAL ERROR: PositionCollector is not thread-safe and must be used only in the main thread.\n"
                    "This is a GPU-limited process that benefits from batching, not multi-threading.\n"
                    f"Current thread: {current_thread.name}\n"
                    f"Main thread: {main_thread.name}\n"
                    "The SelfPlayEngine is trying to use multi-threading, but PositionCollector requires single-threaded usage.\n"
                    "Please set --num_workers=1 in the self-play script to disable multi-threading."
                )
                print(error_msg, file=sys.stderr)
                sys.exit(1)
        else:
            # Other thread types - not allowed
            error_msg = (
                "CRITICAL ERROR: PositionCollector is not thread-safe and must be used only in the main thread.\n"
                "This is a GPU-limited process that benefits from batching, not multi-threading.\n"
                f"Current thread: {current_thread.name}\n"
                f"Main thread: {main_thread.name}\n"
                "The SelfPlayEngine is trying to use multi-threading, but PositionCollector requires single-threaded usage.\n"
                "Please set --num_workers=1 in the self-play script to disable multi-threading."
            )
            print(error_msg, file=sys.stderr)
            sys.exit(1)
        
        self.model = model
        self.verbose = verbose
        # Store (board, callback, metadata) tuples for deferred processing
        self.policy_requests = []  # List of (board, callback, metadata) tuples
        self._memory_warnings_shown = set()  # Track which warnings we've already shown
        self._creation_thread = threading.current_thread()  # Track creation thread
        self._last_memory_check_time = 0  # Track when we last checked memory
        self._memory_check_cooldown = 2.0  # Don't check memory more than once every 5 seconds
        
        # Add counters and timing for iterative batching
        self.total_policy_requests = 0
        self.total_value_requests = 0
        self.iteration_index = 0
        self.policy_batches_processed = 0
        self.policy_items_processed = 0
        self.last_batch_size = 0
        self.eval_time_sec = 0.0
        
    def has_pending_requests(self) -> bool:
        """Check if there are any pending policy requests."""
        return bool(self.policy_requests)
    
    def _check_thread_safety(self, method_name: str):
        """Check if method is called from the correct thread."""
        current_thread = threading.current_thread()
        
        # Allow if we're in the creation thread OR if we're in a single worker thread
        if current_thread is self._creation_thread:
            # Same thread as creation - always allowed
            pass
        elif (current_thread.name.startswith('ThreadPoolExecutor-') and 
              self._creation_thread.name.startswith('ThreadPoolExecutor-') and
              '_0' in current_thread.name and current_thread.name.endswith('_0') and
              '_0' in self._creation_thread.name and self._creation_thread.name.endswith('_0')):
            # Both are single worker threads - acceptable
            pass
        else:
            # Different threads - not allowed
            error_msg = (
                f"CRITICAL ERROR: PositionCollector.{method_name}() called from wrong thread.\n"
                f"Current thread: {current_thread.name}\n"
                f"Creation thread: {self._creation_thread.name}\n"
                "PositionCollector is not thread-safe.\n"
                "Please set --num_workers=1 in the self-play script to disable multi-threading."
            )
            print(error_msg, file=sys.stderr)
            sys.exit(1)
    
    def _check_memory_usage(self):
        """
        Check current memory usage and warn/exit if thresholds are exceeded.
        
        This method implements rate limiting to prevent flooding the logs with
        repeated warnings. Memory checks are throttled to once every 5 seconds,
        but exit threshold checks always run for safety.
        
        Returns:
            str: Status message indicating what was checked or why it was skipped
            
        Raises:
            RuntimeError: If memory usage exceeds exit threshold
        """
        current_time = time.time()
        
        process = psutil.Process()
        memory_gb = process.memory_info().rss / (1024**3)
        
        # Always check exit threshold for safety
        if memory_gb >= MEMORY_EXIT_THRESHOLD:
            logger.error(f"Memory usage {memory_gb:.1f}GB exceeds exit threshold {MEMORY_EXIT_THRESHOLD}GB")
            raise RuntimeError(f"Memory usage {memory_gb:.1f}GB exceeds safety limit of {MEMORY_EXIT_THRESHOLD}GB")
        
        # Rate limiting: only check warnings if enough time has passed since last check
        if current_time - self._last_memory_check_time < self._memory_check_cooldown:
            time_since_last = current_time - self._last_memory_check_time
            return f"Memory check skipped (checked {time_since_last:.1f}s ago, cooldown: {self._memory_check_cooldown}s)"
        
        self._last_memory_check_time = current_time
        
        # Check for warnings (rate limited)
        warnings_shown = 0
        for threshold in MEMORY_WARNINGS:
            if memory_gb >= threshold and threshold not in self._memory_warnings_shown:
                logger.warning(f"Memory usage is {memory_gb:.1f}GB (threshold: {threshold}GB)")
                self._memory_warnings_shown.add(threshold)
                warnings_shown += 1
        
        if warnings_shown > 0:
            return f"Memory check completed: {memory_gb:.1f}GB, {warnings_shown} new warning(s) shown"
        else:
            return f"Memory check completed: {memory_gb:.1f}GB, no new warnings"
    
    def _check_position_limit(self):
        """
        Check if we're approaching the position limit.
        
        Raises:
            RuntimeError: If position count exceeds limit
        """
        total_positions = len(self.policy_requests)
        if total_positions >= MAX_POSITIONS:
            raise RuntimeError(f"Position count {total_positions} exceeds limit of {MAX_POSITIONS}")
        
        # Warn at 80% of limit
        if total_positions >= MAX_POSITIONS * 0.8:
            logger.warning(f"Position count {total_positions} is approaching limit of {MAX_POSITIONS}")
    
    def _validate_and_add_request(self, board, callback: Callable, metadata: Optional[Dict], method_name: str):
        """Common validation and request addition logic."""
        self._check_thread_safety(method_name)
        self._check_memory_usage()
        self._check_position_limit()
        
        metadata = metadata or {}
        self.policy_requests.append((board, callback, metadata))
        
    def request_policy(self, board, callback: Callable, metadata: Optional[Dict] = None):
        """
        Add a policy request to be processed later.
        
        Args:
            board: Board state to get policy for
            callback: Function to call with policy logits when available
                     Signature: callback(policy_logits: np.ndarray, metadata: Dict) -> None
            metadata: Optional metadata for validation (e.g., {'game_id': 1, 'depth': 2, 'path': [(1,1), (2,2)]})
            
        Raises:
            RuntimeError: If called from a different thread than the constructor
        """
        self._validate_and_add_request(board, callback, metadata, "request_policy")
        self.total_policy_requests += 1
    
    def process_batches(self):
        """
        Process all collected positions in batches.
        
        This method:
        1. Extracts all boards from policy requests
        2. Calls model.batch_infer() once for all policy requests
        3. Calls each policy callback with its corresponding result and metadata
        4. Clears all request lists after processing
        
        Note: This method assumes the model.batch_infer() returns results
        in the same order as the input boards.
        
        Raises:
            RuntimeError: If model.batch_infer() fails or callback errors occur
        """
        self._check_thread_safety("process_batches")
        
        if not self.policy_requests:
            return

        # Wave index (for logs)
        wave = self.iteration_index
        self.iteration_index += 1

        # Snapshot current wave; DO NOT process items that callbacks add later
        snapshot = list(self.policy_requests)  # [(board, callback, meta), ...]
        self.policy_requests.clear()

        boards = [req[0] for req in snapshot]
        callbacks = [req[1] for req in snapshot]
        metadata_list = [req[2] for req in snapshot]

        self.last_batch_size = len(boards)

        t0 = time.perf_counter()
        policies, _ = self.model.batch_infer(boards)
        t1 = time.perf_counter()

        self.eval_time_sec += (t1 - t0)
        self.policy_batches_processed += 1
        self.policy_items_processed += len(boards)

        # sanity
        if len(policies) != len(boards):
            raise RuntimeError(f"Model returned {len(policies)} policies for {len(boards)} boards")

        # Dispatch callbacks
        for policy, cb, meta in zip(policies, callbacks, metadata_list):
            cb(policy, meta)

        if self.verbose >= 1:
            logger.info(
                f"🔍 BATCH {wave}: processed {len(boards)} policy items "
                f"(cum {self.policy_items_processed}); time {t1 - t0:.4f}s"
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about collected requests and memory usage.
        
        Returns:
            Dictionary with counts of pending requests and memory info
        """
        process = psutil.Process()
        memory_gb = process.memory_info().rss / (1024**3)
        
        # Get current memory check status
        current_time = time.time()
        time_since_last = current_time - self._last_memory_check_time
        memory_check_status = "ready" if time_since_last >= self._memory_check_cooldown else f"cooldown ({time_since_last:.1f}s remaining)"
        
        return {
            'policy_requests': len(self.policy_requests),
            'total_requests': len(self.policy_requests),
            'memory_gb': memory_gb,
            'memory_warnings_shown': len(self._memory_warnings_shown),
            'last_memory_check_time': self._last_memory_check_time,
            'memory_check_cooldown': self._memory_check_cooldown,
            'memory_check_status': memory_check_status
        }


class MinimaxSearchNode:
    """Represents a node in the minimax search tree for easier debugging and testing."""
    
    def __init__(self, state: HexGameState, depth: int, path: List[Tuple[int, int]] = None):
        self.state = state
        self.depth = depth
        self.path = path or []
        self.children: Dict[Tuple[int, int], MinimaxSearchNode] = {}
        self.value: Optional[float] = None
        self.best_move: Optional[Tuple[int, int]] = None
        self.is_maximizing: bool = (state.current_player_enum == Player.BLUE)
        
    def __str__(self):
        current_player_enum = self.state.current_player_enum
        return (
            f"Node(depth={self.depth}, player={'Blue' if current_player_enum == Player.BLUE else 'Red'}, "
            f"maximizing={self.is_maximizing}, value={self.value}, path={self.path})"
        )


def get_topk_moves(state: HexGameState, model, k: int, 
                   temperature: float = 1.0) -> List[Tuple[Tuple[int, int], float]]:
    """
    Sample k moves from the policy distribution with temperature scaling.
    
    Args:
        state: Current game state
        model: Model for inference
        k: Number of moves to sample
        temperature: Temperature for sampling (0.0 = deterministic top-k, higher = more random)
        
    Returns:
        List of ((row, col), probability) tuples for k sampled moves
    """
    policy_logits, _ = model.simple_infer(state.board)
    legal_moves = state.get_legal_moves()
    
    # Use the core sampling function
    return sample_moves_from_policy(policy_logits, legal_moves, state.board.shape[0], k, temperature)


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
    
    # Use the core sampling function directly
    moves_with_probs = sample_moves_from_policy(
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
        moves_with_probs = get_topk_moves(state, model, k, temperature)
        
        # Build children
        for move, _ in moves_with_probs:
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
    collector: PositionCollector = None,
    game_id: Optional[int] = None,
    verbose: int = 0
) -> MinimaxSearchNode:
    """
    Build search tree while collecting positions for batch processing.
    
    This function builds a minimax search tree but defers all neural network
    inference calls to be processed later in batches. This is the key function
    that enables the batching optimization.
    
    HOW IT WORKS:
    1. Recursively builds the tree structure (nodes and edges)
    2. Instead of immediately calling model.infer() for each position, it:
       - Creates a callback function that will handle the policy result
       - Registers the board + callback with the PositionCollector
       - Continues building the tree structure
    3. The callback function (when called later) will:
       - Extract moves from the policy logits
       - Create child nodes for each move
       - Recursively build the subtree for each child
    
    KEY INSIGHT:
    The tree structure is built "lazily" - we know the shape of the tree
    (which moves to explore) but don't have the actual policy values yet.
    The callbacks allow us to complete the tree building once the batched
    inference results are available.
    
    VALIDATION:
    Each position request includes metadata (game_id, depth, path) that is
    passed back with the result to ensure correct routing of batched results.
    
    Args:
        root_state: Starting game state
        model: Neural network model
        widths: List of search widths at each depth
        temperature: Temperature for move sampling
        collector: PositionCollector instance for batching (if None, falls back to immediate inference)
        game_id: Optional game identifier for metadata validation
        
    Returns:
        Root node of the search tree (children may not be fully populated until process_batches() is called)
    """
    
    def build_node(state: HexGameState, depth: int, path: List[Tuple[int, int]], verbose: int = 0) -> MinimaxSearchNode:
        """
        Recursively build a node in the search tree.
        
        This function creates the tree structure but defers policy inference
        when a collector is provided. The actual move selection happens later
        when the batched policy results are processed.
        """
        node = MinimaxSearchNode(state, depth, path)
        
        # DEBUG: Trace tree building process
        if verbose >= 2:
            logger.info(f"🔍 BUILD NODE DEBUG: Building node at depth {depth}, path: {path}")
            logger.info(f"🔍 BUILD NODE DEBUG: Game over: {state.game_over}, depth >= len(widths): {depth >= len(widths)}")
            logger.info(f"🔍 BUILD NODE DEBUG: Current widths: {widths}, max depth: {len(widths)}")
        
        # Stop if game is over or we've reached max depth
        if state.game_over or depth >= len(widths):
            if verbose >= 2:
                logger.info(f"🔍 BUILD NODE DEBUG: Stopping at depth {depth} - game_over: {state.game_over}, max_depth: {depth >= len(widths)}")
            return node
        
        # Get top moves for this depth
        k = widths[depth] if depth < len(widths) else 1
        
        if collector is not None:
            # DEFERRED INFERENCE: Collect policy request instead of immediate inference
            def policy_callback(policy_logits, metadata):
                """
                Callback function that will be called when policy results are available.
                
                This function:
                1. Extracts the top-k moves from the policy logits
                2. Creates child nodes for each move
                3. Recursively builds the subtree for each child
                
                Note: This callback is called AFTER the tree structure is built,
                so we can safely modify the node's children.
                
                Args:
                    policy_logits: Policy logits from batched inference
                    metadata: Validation metadata to ensure correct routing
                """
                # DEBUG: Trace callback execution
                if verbose >= 2:
                    logger.info(f"🔍 CALLBACK DEBUG: Policy callback called for depth {depth}, path: {path}")
                    logger.info(f"🔍 CALLBACK DEBUG: Policy logits shape: {policy_logits.shape if hasattr(policy_logits, 'shape') else 'no shape'}")
                    logger.info(f"🔍 CALLBACK DEBUG: Metadata: {metadata}")
                
                # Validate metadata to ensure this result belongs to this node
                expected_metadata = {
                    'game_id': game_id,
                    'depth': depth,
                    'path': path.copy()
                }
                if metadata != expected_metadata:
                    raise RuntimeError(
                        f"Metadata mismatch in policy callback. Expected {expected_metadata}, got {metadata}"
                    )
                
                # Use policy to get top moves from the already-computed policy logits
                moves = get_topk_moves_from_policy(policy_logits, state, k, temperature)
                
                if verbose >= 2:
                    logger.info(f"🔍 CALLBACK DEBUG: Got {len(moves)} moves: {moves}")
                
                # Build children for each move
                for move in moves:
                    child_state = state.make_move(*move)
                    child_path = path + [move]
                    if verbose >= 2:
                        logger.info(f"🔍 CALLBACK DEBUG: Building child for move {move} at depth {depth + 1}")
                    child_node = build_node(child_state, depth + 1, child_path, verbose)
                    node.children[move] = child_node
                
                if verbose >= 2:
                    logger.info(f"🔍 CALLBACK DEBUG: Created {len(node.children)} children for depth {depth}")
            
            # Create metadata for validation
            metadata = {
                'game_id': game_id,
                'depth': depth,
                'path': path.copy()
            }
            
            # Register the board + callback + metadata for later batch processing
            if verbose >= 2:
                logger.info(f"🔍 POLICY REQUEST DEBUG: Registering policy request for depth {depth}, path: {path}")
            collector.request_policy(state.board, policy_callback, metadata)
            
        else:
            # FALLBACK: Immediate inference (original behavior)
            moves_with_probs = get_topk_moves(state, model, k, temperature)
            
                        # Build children immediately
            for move, _ in moves_with_probs:
                child_state = state.make_move(*move)
                child_path = path + [move]
                child_node = build_node(child_state, depth + 1, child_path)
                node.children[move] = child_node
            
        return node
    
    return build_node(root_state, 0, [], verbose)


def expected_nodes_by_depth(widths):
    """Calculate expected number of nodes at each depth for given search widths."""
    # depth 0: 1 node (root)
    out = [1]
    for w in widths:
        out.append(out[-1] * w)
    return out  # len = len(widths) + 1


def convert_model_logit_to_minimax_value(value_logit: float, root_player: Player) -> float:
    """
    Convert a raw model value logit to a minimax-friendly value from the root player's perspective.
    
    The value head predicts Red's win probability because Red wins are labeled as 1.0 in training.
    The model outputs raw logits representing log(p/(1-p)) where p is the probability of Red winning.
    This function:
    1. Applies sigmoid to convert logit to probability: sigmoid(logit) = p
    2. Converts to root player's perspective for minimax search
    
    Args:
        value_logit: Raw logit from model's value head (unbounded)
        root_player: Player whose perspective we want (Player enum only)
        
    Returns:
        Minimax value in range [-1, 1] where:
        - Positive values are good for the root player
        - Negative values are bad for the root player
        - 0.0 represents neutral/equal chances
        
    Raises:
        TypeError: If root_player is not a Player enum
    """
    # Normalize root_player to Player enum (Enum-only API)
    if not isinstance(root_player, Player):
        raise TypeError(f"root_player must be Player, got {type(root_player)}")
    root_player_enum = root_player
    
    # Step 1: Convert logit to probability using sigmoid
    # The value head predicts Red's win probability because Red wins are labeled as 1.0 in training.
    # sigmoid(value_logit) gives the probability that Red wins.
    prob_red_win = torch.sigmoid(torch.tensor(value_logit)).item()
    
    # Step 2: Convert to root player's perspective
    if root_player_enum == Player.BLUE:  # Root player is Blue
        # For Blue: positive values = Blue wins (good), negative values = Red wins (bad)
        # Convert from Red's win probability to Blue's perspective
        return 1.0 - 2.0 * prob_red_win  # Maps [0,1] to [1,-1]
    else:  # Root player is Red
        # For Red: negative values = Red wins (good), positive values = Blue wins (bad)
        # Convert from Red's win probability to Red's perspective
        return 2.0 * prob_red_win - 1.0  # Maps [0,1] to [-1,1]


def evaluate_leaf_nodes(
    nodes: List[MinimaxSearchNode],
    model,
    batch_size: int = 1000,
    root_player: Player | None = None,
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
        t0 = time.perf_counter()
        _, batch_values = model.batch_infer(batch)
        t1 = time.perf_counter()
        logger.info(
            f"NN_BATCH values batch_index={i // batch_size} size={len(batch)} wall_ms={(t1 - t0)*1e3:.2f}"
        )
        
        # DEBUG: Analyze the returned values to verify they're real
        if len(batch_values) > 0:
            import numpy as np
            batch_values_np = np.array(batch_values)
            logger.info(f"🔍 VALUE ANALYSIS DEBUG: Batch {i//batch_size + 1} with {len(batch_values)} values")
            logger.info(f"🔍 VALUE ANALYSIS DEBUG: Min: {batch_values_np.min():.6f}, Max: {batch_values_np.max():.6f}")
            logger.info(f"🔍 VALUE ANALYSIS DEBUG: Mean: {batch_values_np.mean():.6f}, Std: {batch_values_np.std():.6f}")
            logger.info(f"🔍 VALUE ANALYSIS DEBUG: First 5 values: {batch_values[:5]}")
            logger.info(f"🔍 VALUE ANALYSIS DEBUG: Last 5 values: {batch_values[-5:]}")
        
        values.extend(batch_values)
    
    # Assign values to leaf nodes, converting to root player's perspective
    for node, value in zip(leaf_nodes, values):
        # Convert raw model logit to minimax-friendly value
        # If root_player not provided, use the game's current player at root of search
        effective_root_player = root_player if root_player is not None else nodes[0].state.current_player_enum
        node.value = convert_model_logit_to_minimax_value(value, effective_root_player)
        
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
    player_enum = node.state.current_player_enum
    print(
        "  " * indent
        + f"Node: depth={node.depth}, player={'Blue' if player_enum == Player.BLUE else 'Red'}, "
        f"maximizing={node.is_maximizing}, value={node.value}, path={node.path}"
    )
    
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
        root_player_enum = state.current_player_enum
        logger.info(f"Root state: player {state.current_player} ({'Blue' if root_player_enum == Player.BLUE else 'Red'})")
    
    # Build the search tree
    root = build_search_tree(state, model, widths, temperature)
    
    # Evaluate all leaf nodes from the root player's perspective
    evaluate_leaf_nodes([root], model, batch_size, state.current_player_enum)
    
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
) -> Tuple[Tuple[int, int], float, Optional[MinimaxSearchNode], Dict[str, Any]]:
    """
    Fixed-width, fixed-depth minimax search with batched inference for all policy calls.

    This function implements the complete batching workflow for minimax search.
    It demonstrates the key optimization: instead of making O(n) individual
    neural network calls during tree building, it makes O(1) batched calls.

    BATCHING WORKFLOW:
    1. Create PositionCollector to manage deferred inference requests
    2. Build search tree structure while collecting all policy requests
    3. Process all collected policy requests in iterative batches (waves)
    4. Evaluate leaf nodes (also batched)
    5. Perform minimax backup to compute final values

    PERFORMANCE BENEFITS:
    - Tree building: O(n) individual calls → O(waves) batched calls
    - Leaf evaluation: Already batched in evaluate_leaf_nodes()
    - Total inference calls: O(n) → O(waves) for policy + O(1) for values
    - Iterative batching ensures all depths are properly explored
    
    VALIDATION:
    - Each position request includes metadata for validation
    - Results are validated to ensure correct routing back to the right node
    - Fail-fast on any metadata mismatches

    Args:
        state: HexGameState (current position)
        model: SimpleModelInference (must support batch inference)
        widths: List of ints, e.g. [20, 10, 10, 5] (width at each ply)
        batch_size: Max batch size for evaluation (used for leaf nodes)
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
        root_player_enum = state.current_player_enum
        logger.info(f"Root state: player {state.current_player} ({'Blue' if root_player_enum == Player.BLUE else 'Red'})")
    
    # Generate unique game ID for metadata validation
    game_id = id(state)  # Use object ID as unique identifier
    
    # STEP 1: Create position collector for batched inference
    collector = PositionCollector(model, verbose=verbose)
    
    # Check memory usage at the start of the search
    collector._check_memory_usage()
    
    # STEP 2: Build the search tree with position collection
    # This creates the tree structure but defers all policy inference
    root = build_search_tree_with_collection(state, model, widths, temperature, collector, game_id=game_id, verbose=verbose)
    
    if verbose >= 2:
        stats = collector.get_stats()
        logger.info(f"Tree built with {stats['policy_requests']} policy requests collected")
        logger.info(f"Memory usage: {stats['memory_gb']:.1f}GB")
    
    # STEP 3: Process all collected batches iteratively
    # This is where the actual neural network inference happens
    # Process in waves until no more requests are pending
    t0_all = time.perf_counter()
    max_waves = max(2, len(widths) * 2)  # generous safety cap
    waves = 0
    while collector.has_pending_requests():
        collector.process_batches()
        waves += 1
        if waves >= max_waves:
            logger.warning(
                f"🔍 TREE SEARCH WARNING: exceeded max_waves={max_waves} "
                f"with pending requests still present; aborting further waves."
            )
            break
    t1_all = time.perf_counter()
    
    # Check memory usage after processing batches
    collector._check_memory_usage()
    
    if verbose >= 2:
        logger.info("All policy batches processed, tree is now complete")
    
    # Log evaluation summary
    logger.info(
        f"🔍 EVAL SUMMARY: {collector.policy_items_processed} policy evals "
        f"in {collector.eval_time_sec:.4f}s NN time; "
        f"{t1_all - t0_all:.4f}s total policy phase; "
        f"{collector.policy_batches_processed} waves; "
        f"avg batch size { (collector.policy_items_processed / max(1, collector.policy_batches_processed)) :.2f}"
    )
    
    # DEBUG: Count total positions in tree and analyze structure (AFTER callbacks are processed)
    def count_positions(node):
        count = 1  # Count this node
        for child in node.children.values():
            count += count_positions(child)
        return count
    
    def analyze_tree_structure(node, depth=0):
        """Analyze tree structure to understand depth distribution"""
        if depth == 0:
            # Root level
            return {
                'depth_0': 1,
                'depth_1': len(node.children),
                'depth_2': sum(len(child.children) for child in node.children.values()),
                'max_depth': max([analyze_tree_structure(child, depth + 1)['max_depth'] for child in node.children.values()], default=0)
            }
        else:
            # Recursive case
            if not node.children:
                return {'max_depth': depth}
            else:
                child_max_depths = [analyze_tree_structure(child, depth + 1)['max_depth'] for child in node.children.values()]
                return {'max_depth': max(child_max_depths) if child_max_depths else depth}
    
    total_positions = count_positions(root)
    tree_analysis = analyze_tree_structure(root)
    

    
    # Depth accounting: verify expected breadth per depth
    exp = expected_nodes_by_depth(widths)
    act = [tree_analysis.get(f'depth_{d}', 0) for d in range(len(exp))]
    logger.info(f"🔍 TREE STATS: expected nodes by depth: {exp}")
    logger.info(f"🔍 TREE STATS: actual nodes by depth:   {act}")

    # If depth under-explored:
    for d in range(1, len(exp)):  # ignore root
        if act[d] < (0.9 * exp[d]):  # threshold configurable
            logger.warning(
                f"🔍 TREE SEARCH WARNING: depth {d} has {act[d]} nodes; expected ~{exp[d]}"
            )
    
    # Additional debugging: Check if the tree is actually being built to depth 2
    if tree_analysis['depth_2'] == 0 and len(widths) > 1:
        logger.warning(f"🔍 TREE SEARCH WARNING: Tree has 0 positions at depth 2, but search widths {widths} should explore depth 2")
        logger.warning(f"🔍 TREE SEARCH WARNING: This suggests the tree building is not working correctly")
    
    # STEP 4: Evaluate all leaf nodes from the root player's perspective
    # This is also batched for efficiency
    if verbose >= 1:
        logger.info(f"🔍 LEAF EVALUATION DEBUG: Starting leaf node evaluation")
    evaluate_leaf_nodes([root], model, batch_size, state.current_player_enum)
    if verbose >= 1:
        logger.info(f"🔍 LEAF EVALUATION DEBUG: Completed leaf node evaluation")
    
    # STEP 5: Backup values to root (temperature already applied during move sampling)
    root_value = minimax_backup(root)
    
    if verbose >= 2:
        logger.info(f"Batched search complete: best move = {root.best_move}, value = {root_value}")
    
    # Create search statistics
    search_stats = {
        'policy_items_processed': collector.policy_items_processed,
        'policy_batches_processed': collector.policy_batches_processed,
        'policy_nn_time': collector.eval_time_sec,
        'policy_total_time': t1_all - t0_all,
        'avg_policy_batch': collector.policy_items_processed / max(1, collector.policy_batches_processed),
        'total_positions': total_positions,
        'tree_analysis': tree_analysis
    }
    
    if return_tree:
        return root.best_move, root_value, root, search_stats
    else:
        return root.best_move, root_value, None, search_stats 


def test_iterative_batching_sanity():
    """
    Quick sanity test for iterative batching.
    
    This test verifies that:
    1. The tree is built to the expected depth
    2. The correct number of policy evaluations are performed
    3. The iterative batching works correctly
    
    Expected with widths=[13,8]:
    - 1 root node (depth 0)
    - 13 children at depth 1
    - 13*8 = 104 grandchildren at depth 2
    - Total: 1 + 13 + 104 = 118 policy evaluations
    """
    from hex_ai.inference.simple_model_inference import SimpleModelInference
    from hex_ai.training_utils import get_device
    
    print("Testing iterative batching sanity...")
    
    # Create a simple model (you'll need to provide a valid model path)
    try:
        model = SimpleModelInference("checkpoints/hyperparameter_tuning/pipeline_20250805_162626/pipeline_sweep_exp0__99914b_20250805_162626/epoch4_mini32.pt.gz", device=get_device())
    except Exception as e:
        print(f"Could not load model: {e}")
        print("Skipping iterative batching test")
        return
    
    # Create test state
    state = HexGameState()
    widths = [13, 8]
    
    # Run search
    best_move, best_value, _, search_stats = minimax_policy_value_search_with_batching(
        state=state,
        model=model,
        widths=widths,
        temperature=1.0,
        verbose=2
    )
    
    # Verify results
    expected_policy_evals = 1 + 13  # root + depth1 (depth2 nodes are leaves, no policy needed)
    actual_policy_evals = search_stats['policy_items_processed']
    
    print(f"Expected policy evaluations: {expected_policy_evals}")
    print(f"Actual policy evaluations: {actual_policy_evals}")
    print(f"Policy evaluations match: {actual_policy_evals == expected_policy_evals}")
    
    # Check depth exploration
    tree_analysis = search_stats['tree_analysis']
    depth_2_count = tree_analysis.get('depth_2', 0)
    expected_depth_2 = 13 * 8
    
    print(f"Expected depth 2 nodes: {expected_depth_2}")
    print(f"Actual depth 2 nodes: {depth_2_count}")
    print(f"Depth 2 exploration correct: {depth_2_count == expected_depth_2}")
    
    # Check that we have the right number of waves
    expected_waves = 2  # wave 0: root, wave 1: depth1
    actual_waves = search_stats['policy_batches_processed']
    print(f"Expected waves: {expected_waves}")
    print(f"Actual waves: {actual_waves}")
    print(f"Wave count correct: {actual_waves == expected_waves}")
    
    if (actual_policy_evals == expected_policy_evals and 
        depth_2_count == expected_depth_2 and 
        actual_waves == expected_waves):
        print("✅ Iterative batching sanity test PASSED")
    else:
        print("❌ Iterative batching sanity test FAILED")
        print("This indicates the tree building is not working correctly")


if __name__ == "__main__":
    test_iterative_batching_sanity() 