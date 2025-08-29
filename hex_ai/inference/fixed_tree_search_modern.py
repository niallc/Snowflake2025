"""
Modernized Fixed Tree Search Implementation

This module provides a clean, maintainable implementation of fixed-width minimax search
that follows the same patterns as the MCTS implementation. It eliminates the complex
PositionCollector system and provides a simple, efficient API.
"""

from __future__ import annotations

import time
import logging
import psutil
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable
import numpy as np

from hex_ai.inference.game_engine import HexGameState
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.value_utils import temperature_scaled_softmax, get_top_k_moves_with_probs, sample_moves_from_policy
from hex_ai.enums import Player
from hex_ai.utils.timing import MCTSTimingTracker

logger = logging.getLogger(__name__)


@dataclass
class FixedTreeSearchConfig:
    """Configuration for fixed-width minimax search."""
    
    # Search parameters
    search_widths: List[int]  # e.g., [20, 10, 5] for 3-ply search
    temperature: float = 1.0  # Temperature for move sampling
    
    # Performance parameters
    batch_size: int = 1000  # Batch size for neural network evaluation
    enable_early_termination: bool = True
    early_termination_threshold: float = 0.95  # Win probability threshold
    
    # Memory management
    max_positions: int = 1_000_000  # Maximum positions to evaluate
    
    # Validation
    def __post_init__(self):
        if not self.search_widths:
            raise ValueError("search_widths cannot be empty")
        if any(w <= 0 for w in self.search_widths):
            raise ValueError("All search widths must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 <= self.early_termination_threshold <= 1:
            raise ValueError("early_termination_threshold must be between 0 and 1")


@dataclass(frozen=True)
class FixedTreeSearchResult:
    """Complete result of a fixed tree search."""
    move: Tuple[int, int]  # The selected move
    value: float  # Estimated value for the root position
    stats: Dict[str, Any]  # Performance statistics
    tree_data: Dict[str, Any]  # Tree information for analysis
    early_termination_info: Optional[Dict[str, Any]]  # Early termination details
    win_probability: float  # Win probability for current player


@dataclass
class MinimaxNode:
    """Node in the minimax search tree."""
    state: HexGameState
    depth: int
    move: Optional[Tuple[int, int]] = None  # Move that led to this node
    parent: Optional[MinimaxNode] = None
    children: List[MinimaxNode] = field(default_factory=list)
    policy: Optional[np.ndarray] = None
    value: Optional[float] = None
    best_move: Optional[Tuple[int, int]] = None
    is_terminal: bool = False
    terminal_value: Optional[float] = None


class SimpleBatchProcessor:
    """Simple, efficient batch processing for neural network evaluation."""
    
    def __init__(self, model: SimpleModelInference, batch_size: int):
        self.model = model
        self.batch_size = batch_size
        self.pending_evaluations: List[Tuple[MinimaxNode, Callable]] = []
    
    def add_evaluation(self, node: MinimaxNode, callback: Callable):
        """Add a node for evaluation."""
        self.pending_evaluations.append((node, callback))
    
    def process_batch(self):
        """Process current batch of evaluations."""
        if not self.pending_evaluations:
            return
        
        # Take current batch
        batch = self.pending_evaluations[:self.batch_size]
        self.pending_evaluations = self.pending_evaluations[self.batch_size:]
        
        # Extract nodes and callbacks
        nodes = [item[0] for item in batch]
        callbacks = [item[1] for item in batch]
        
        # Run batch inference with timing
        boards = [node.state.board for node in nodes]
        start_time = time.time()
        policies, values = self.model.batch_infer(boards)
        nn_time = time.time() - start_time
        
        # Call callbacks with results
        for (policy, value), (node, callback) in zip(zip(policies, values), batch):
            callback(node, policy, value)
        
        # Return timing info for statistics
        return len(nodes), nn_time
    
    def has_pending_requests(self) -> bool:
        """Check if there are pending evaluation requests."""
        return len(self.pending_evaluations) > 0


class FixedTreeSearch:
    """Fixed-width minimax search with modern API."""
    
    def __init__(self, model: SimpleModelInference, config: FixedTreeSearchConfig):
        self.model = model
        self.config = config
        self.timing_tracker = MCTSTimingTracker()
        self.stats = {
            'total_positions': 0,
            'total_evaluations': 0,
            'early_terminations': 0,
            'tree_depth': 0,
            'tree_width': 0,
            'policy_evaluations': 0,
            'value_evaluations': 0,
            'policy_batches': 0,
            'value_batches': 0,
            'policy_nn_time': 0.0,
            'value_nn_time': 0.0,
            'tree_building_time': 0.0,
            'leaf_evaluation_time': 0.0,
            'backup_time': 0.0,
            'memory_usage_mb': 0.0,
        }
    
    def run(self, state: HexGameState, verbose: int = 0) -> FixedTreeSearchResult:
        """Run fixed tree search and return complete result."""
        start_time = time.time()
        
        # Set logging level based on verbosity
        if verbose >= 4:
            logger.setLevel(logging.DEBUG)
        elif verbose >= 2:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
        
        # Only log basic info at verbose >= 2
        if verbose >= 2:
            logger.info(f"Starting fixed tree search with widths {self.config.search_widths}, temperature {self.config.temperature}")
            root_player_enum = state.current_player_enum
            logger.info(f"Root state: player {state.current_player} ({'Blue' if root_player_enum == Player.BLUE else 'Red'})")
        
        # Check memory usage at start (only warn if high)
        self._check_memory_usage(verbose)
        
        # Check for early termination conditions
        early_termination_info = self._check_early_termination(state)
        if early_termination_info:
            self.stats['early_terminations'] += 1
            if verbose >= 1:
                logger.info(f"Early termination: {early_termination_info['reason']}")
            
            return FixedTreeSearchResult(
                move=early_termination_info['move'],
                value=early_termination_info['value'],
                stats=self.stats,
                tree_data={'early_termination': True},
                early_termination_info=early_termination_info,
                win_probability=early_termination_info['win_probability']
            )
        
        # Build search tree
        tree_start = time.time()
        root = self._build_search_tree(state, verbose)
        self.stats['tree_building_time'] = time.time() - tree_start
        
        # Collect all leaf nodes and evaluate them
        leaf_start = time.time()
        leaf_nodes = self._collect_leaf_nodes(root)
        self._evaluate_leaf_nodes(leaf_nodes, verbose)
        self.stats['leaf_evaluation_time'] = time.time() - leaf_start
        
        # Perform minimax backup
        backup_start = time.time()
        root_value = self._minimax_backup(root)
        self.stats['backup_time'] = time.time() - backup_start
        
        # Extract best move
        best_move = root.best_move
        if best_move is None:
            raise RuntimeError("No best move found in search tree")
        
        # Calculate win probability
        win_probability = (root_value + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1]
        
        # Update statistics
        self.stats['total_positions'] = self._count_nodes(root)
        self.stats['tree_depth'] = len(self.config.search_widths)
        self.stats['tree_width'] = max(self.config.search_widths)
        
        elapsed_time = time.time() - start_time
        self.stats['elapsed_time'] = elapsed_time
        
        # Update memory usage
        self.stats['memory_usage_mb'] = self._get_memory_usage_mb()
        
        # Log minimal completion info at verbose >= 1 (just a dot for tournaments)
        if verbose >= 1:
            print(".", end="", flush=True)
        
        # Log detailed results at verbose >= 2
        if verbose >= 2:
            logger.info(f"Search complete: move={best_move}, value={root_value:.3f}, "
                       f"time={elapsed_time:.2f}s, positions={self.stats['total_positions']}")
        
        if verbose >= 3:
            self._log_detailed_stats()
            self._log_tree_analysis(root)
        
        return FixedTreeSearchResult(
            move=best_move,
            value=root_value,
            stats=self.stats,
            tree_data=self._extract_tree_data(root),
            early_termination_info=None,
            win_probability=win_probability
        )
    
    def _check_early_termination(self, state: HexGameState) -> Optional[Dict[str, Any]]:
        """Check if search should terminate early."""
        if not self.config.enable_early_termination:
            return None
        
        # Check for terminal moves
        if state.game_over:
            winner = state.winner_enum
            if winner == state.current_player_enum:
                return {
                    'reason': 'terminal_win',
                    'move': None,  # Game is already over
                    'value': 1.0,
                    'win_probability': 1.0
                }
            else:
                return {
                    'reason': 'terminal_loss',
                    'move': None,  # Game is already over
                    'value': -1.0,
                    'win_probability': 0.0
                }
        
        # Check neural network confidence
        policy, value = self.model.simple_infer(state.board)
        win_probability = (value + 1.0) / 2.0
        
        if win_probability >= self.config.early_termination_threshold:
            # High confidence win - select best move from policy
            top_moves = get_top_k_moves_with_probs(policy, 1)
            best_move = top_moves[0][0]
            return {
                'reason': 'high_confidence',
                'move': best_move,
                'value': value,
                'win_probability': win_probability
            }
        
        return None
    
    def _build_search_tree(self, state: HexGameState, verbose: int) -> MinimaxNode:
        """Build the search tree structure."""
        root = MinimaxNode(state=state, depth=0)
        
        # Create batch processor for efficient evaluation
        batch_processor = SimpleBatchProcessor(self.model, self.config.batch_size)
        
        # Build tree level by level
        current_level = [root]
        for depth, width in enumerate(self.config.search_widths):
            if verbose >= 3:
                logger.info(f"Building level {depth + 1} with width {width}")
            
            # First, get policies for all nodes at current level
            policy_nodes = []
            for node in current_level:
                if not node.is_terminal:
                    policy_nodes.append(node)
                    batch_processor.add_evaluation(node, self._set_node_policy)
            
            # Process batch to get all policies
            while batch_processor.has_pending_requests():
                batch_size, nn_time = batch_processor.process_batch()
                self.stats['policy_evaluations'] += batch_size
                self.stats['policy_batches'] += 1
                self.stats['policy_nn_time'] += nn_time
            
            if verbose >= 3:
                logger.info(f"Level {depth + 1}: {len(policy_nodes)} policy evaluations")
            
            # Now expand all nodes with their policies
            next_level = []
            for node in current_level:
                if node.is_terminal:
                    continue
                
                # Expand children
                children = self._expand_node(node, width)
                node.children = children
                next_level.extend(children)
            
            current_level = next_level
            if not current_level:
                break
        
        return root
    
    def _set_node_policy(self, node: MinimaxNode, policy: np.ndarray, value: float):
        """Set policy and value for a node (callback for batch processing)."""
        node.policy = policy
        node.value = value
    
    def _expand_node(self, node: MinimaxNode, width: int) -> List[MinimaxNode]:
        """Expand a node by sampling moves from its policy."""
        if node.policy is None:
            raise RuntimeError("Node policy not set before expansion")
        
        # Sample moves from policy
        legal_moves = node.state.get_legal_moves()
        moves_with_probs = sample_moves_from_policy(node.policy, legal_moves, 13, width, self.config.temperature)
        moves = [move for move, prob in moves_with_probs]
        
        children = []
        for move in moves:
            # Apply move to create child state
            child_state = node.state.fast_copy()
            child_state.apply_move(*move)
            
            # Check if child is terminal
            is_terminal = child_state.game_over
            terminal_value = None
            if is_terminal:
                winner = child_state.winner_enum
                if winner == node.state.current_player_enum:
                    terminal_value = 1.0
                elif winner is not None:
                    terminal_value = -1.0
                else:
                    terminal_value = 0.0  # Draw
            
            child = MinimaxNode(
                state=child_state,
                depth=node.depth + 1,
                move=move,
                parent=node,
                is_terminal=is_terminal,
                terminal_value=terminal_value
            )
            children.append(child)
        
        return children
    
    def _collect_leaf_nodes(self, node: MinimaxNode) -> List[MinimaxNode]:
        """Collect all leaf nodes in the tree."""
        leaf_nodes = []
        if not node.children:
            leaf_nodes.append(node)
        else:
            for child in node.children:
                leaf_nodes.extend(self._collect_leaf_nodes(child))
        return leaf_nodes
    
    def _evaluate_leaf_nodes(self, nodes: List[MinimaxNode], verbose: int):
        """Evaluate all leaf nodes using batch processing."""
        if verbose >= 3:
            logger.info(f"Evaluating {len(nodes)} leaf nodes")
        
        batch_processor = SimpleBatchProcessor(self.model, self.config.batch_size)
        
        # Collect all leaf nodes that need evaluation
        leaf_nodes = []
        for node in nodes:
            if not node.children and not node.is_terminal and node.value is None:
                leaf_nodes.append(node)
                batch_processor.add_evaluation(node, self._set_node_value)
        
        if verbose >= 3:
            logger.info(f"Found {len(leaf_nodes)} leaf nodes to evaluate")
        
        # Process all evaluations
        while batch_processor.has_pending_requests():
            batch_size, nn_time = batch_processor.process_batch()
            self.stats['value_evaluations'] += batch_size
            self.stats['value_batches'] += 1
            self.stats['value_nn_time'] += nn_time
    
    def _set_node_value(self, node: MinimaxNode, policy: np.ndarray, value: float):
        """Set value for a node (callback for batch processing)."""
        node.value = value
    
    def _minimax_backup(self, node: MinimaxNode) -> float:
        """Perform minimax backup from leaf nodes to root."""
        if node.is_terminal:
            return node.terminal_value
        
        if node.children:
            # Internal node - backup from children
            # In minimax, we alternate between maximizing and minimizing players
            # The root player wants to maximize, so odd depths minimize, even depths maximize
            if node.depth % 2 == 0:
                # Maximizing player (root player's turn)
                best_value = float('-inf')
                best_move = None
                for child in node.children:
                    child_value = self._minimax_backup(child)
                    if child_value > best_value:
                        best_value = child_value
                        best_move = child.move
            else:
                # Minimizing player (opponent's turn)
                best_value = float('inf')
                best_move = None
                for child in node.children:
                    child_value = self._minimax_backup(child)
                    if child_value < best_value:
                        best_value = child_value
                        best_move = child.move
            
            node.value = best_value
            node.best_move = best_move
            return best_value
        else:
            # Leaf node - use neural network value
            if node.value is None:
                raise RuntimeError("Leaf node value not set")
            return node.value
    
    def _count_nodes(self, node: MinimaxNode) -> int:
        """Count total nodes in the tree."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _check_memory_usage(self, verbose: int):
        """Check memory usage and log warnings if needed."""
        memory_mb = self._get_memory_usage_mb()
        if verbose >= 3:
            logger.info(f"Memory usage: {memory_mb:.1f}MB")
        
        # Warn if memory usage is high (always warn, regardless of verbose level)
        if memory_mb > 1000:  # 1GB
            logger.warning(f"High memory usage: {memory_mb:.1f}MB")
    
    def _log_detailed_stats(self):
        """Log detailed performance statistics."""
        stats = self.stats
        logger.info("=== DETAILED SEARCH STATISTICS ===")
        logger.info(f"Total positions: {stats['total_positions']}")
        logger.info(f"Policy evaluations: {stats['policy_evaluations']} in {stats['policy_batches']} batches")
        logger.info(f"Value evaluations: {stats['value_evaluations']} in {stats['value_batches']} batches")
        logger.info(f"Early terminations: {stats['early_terminations']}")
        logger.info(f"Tree depth: {stats['tree_depth']}, max width: {stats['tree_width']}")
        logger.info(f"Memory usage: {stats['memory_usage_mb']:.1f}MB")
        
        # Timing breakdown
        total_time = stats['elapsed_time']
        if total_time > 0:
            logger.info("=== TIMING BREAKDOWN ===")
            logger.info(f"Tree building: {stats['tree_building_time']:.3f}s ({(stats['tree_building_time']/total_time)*100:.1f}%)")
            logger.info(f"Leaf evaluation: {stats['leaf_evaluation_time']:.3f}s ({(stats['leaf_evaluation_time']/total_time)*100:.1f}%)")
            logger.info(f"Backup: {stats['backup_time']:.3f}s ({(stats['backup_time']/total_time)*100:.1f}%)")
            logger.info(f"Policy NN time: {stats['policy_nn_time']:.3f}s")
            logger.info(f"Value NN time: {stats['value_nn_time']:.3f}s")
            
            # Performance metrics
            if stats['policy_evaluations'] > 0:
                policy_rate = stats['policy_evaluations'] / stats['policy_nn_time'] if stats['policy_nn_time'] > 0 else 0
                logger.info(f"Policy eval rate: {policy_rate:.1f} evals/sec")
            if stats['value_evaluations'] > 0:
                value_rate = stats['value_evaluations'] / stats['value_nn_time'] if stats['value_nn_time'] > 0 else 0
                logger.info(f"Value eval rate: {value_rate:.1f} evals/sec")
    
    def _extract_tree_data(self, root: MinimaxNode) -> Dict[str, Any]:
        """Extract tree information for analysis."""
        # Analyze tree structure by depth
        depth_counts = self._analyze_tree_by_depth(root)
        expected_counts = self._calculate_expected_nodes()
        
        return {
            'total_nodes': self._count_nodes(root),
            'max_depth': len(self.config.search_widths),
            'search_widths': self.config.search_widths.copy(),
            'depth_analysis': {
                'actual': depth_counts,
                'expected': expected_counts,
                'coverage': [actual / expected if expected > 0 else 0 for actual, expected in zip(depth_counts, expected_counts)]
            }
        }
    
    def _analyze_tree_by_depth(self, root: MinimaxNode) -> List[int]:
        """Analyze tree structure to count nodes at each depth."""
        depth_counts = [0] * (len(self.config.search_widths) + 1)
        
        def count_at_depth(node: MinimaxNode, depth: int):
            if depth < len(depth_counts):
                depth_counts[depth] += 1
            for child in node.children:
                count_at_depth(child, depth + 1)
        
        count_at_depth(root, 0)
        return depth_counts
    
    def _calculate_expected_nodes(self) -> List[int]:
        """Calculate expected number of nodes at each depth based on search widths."""
        expected = [1]  # Root node
        for width in self.config.search_widths:
            expected.append(expected[-1] * width)
        return expected
    
    def _log_tree_analysis(self, root: MinimaxNode):
        """Log detailed tree structure analysis."""
        depth_counts = self._analyze_tree_by_depth(root)
        expected_counts = self._calculate_expected_nodes()
        
        logger.info("=== TREE STRUCTURE ANALYSIS ===")
        logger.info(f"Expected nodes by depth: {expected_counts}")
        logger.info(f"Actual nodes by depth:   {depth_counts}")
        
        # Check for under-exploration
        for depth in range(1, len(expected_counts)):
            if depth < len(depth_counts):
                actual = depth_counts[depth]
                expected = expected_counts[depth]
                coverage = actual / expected if expected > 0 else 0
                logger.info(f"Depth {depth}: {actual}/{expected} nodes ({coverage:.1%} coverage)")
                
                if coverage < 0.9:
                    logger.warning(f"Depth {depth} under-explored: {coverage:.1%} coverage")
            else:
                logger.warning(f"Depth {depth}: 0 nodes (expected {expected_counts[depth]})")
        
        # Check for terminal nodes
        terminal_count = sum(1 for node in self._collect_leaf_nodes(root) if node.is_terminal)
        total_leaves = len(self._collect_leaf_nodes(root))
        if total_leaves > 0:
            logger.info(f"Terminal nodes: {terminal_count}/{total_leaves} ({terminal_count/total_leaves:.1%})")


def run_fixed_tree_search(
    state: HexGameState,
    model: SimpleModelInference,
    config: FixedTreeSearchConfig,
    verbose: int = 0
) -> FixedTreeSearchResult:
    """Run fixed tree search with modern API."""
    searcher = FixedTreeSearch(model, config)
    return searcher.run(state, verbose)


def create_fixed_tree_config(
    search_widths: List[int],
    temperature: float = 1.0,
    batch_size: int = 1000,
    enable_early_termination: bool = True,
    early_termination_threshold: float = 0.95
) -> FixedTreeSearchConfig:
    """Create a fixed tree search configuration."""
    return FixedTreeSearchConfig(
        search_widths=search_widths,
        temperature=temperature,
        batch_size=batch_size,
        enable_early_termination=enable_early_termination,
        early_termination_threshold=early_termination_threshold
    )
