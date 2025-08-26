"""
Move selection strategies for tournament play.

This module provides a clean interface for different move selection strategies,
allowing tournaments to easily compare MCTS, fixed tree search, and policy-based selection.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from hex_ai.inference.game_engine import HexGameState, HexGameEngine
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.value_utils import select_policy_move
from hex_ai.inference.fixed_tree_search import minimax_policy_value_search
from hex_ai.inference.mcts import BaselineMCTS, BaselineMCTSConfig, create_mcts_config
from hex_ai.inference.model_cache import get_model_cache


@dataclass
class MoveSelectionConfig:
    """Configuration for move selection strategies."""
    temperature: float = 1.0
    # For MCTS
    mcts_sims: int = 200
    mcts_c_puct: float = 1.5
    mcts_dirichlet_alpha: float = 0.3
    mcts_dirichlet_eps: float = 0.25
    # For fixed tree search
    search_widths: Optional[list] = None
    # For policy-based selection
    policy_top_k: Optional[int] = None


class MoveSelectionStrategy(ABC):
    """Abstract base class for move selection strategies."""
    
    @abstractmethod
    def select_move(self, state: HexGameState, model: SimpleModelInference, 
                   config: MoveSelectionConfig) -> Tuple[int, int]:
        """Select a move for the given state and model."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable name for this strategy."""
        pass
    
    @abstractmethod
    def get_config_summary(self, config: MoveSelectionConfig) -> str:
        """Return a summary of the configuration for this strategy."""
        pass


class PolicyBasedStrategy(MoveSelectionStrategy):
    """Move selection using direct policy sampling."""
    
    def select_move(self, state: HexGameState, model: SimpleModelInference, 
                   config: MoveSelectionConfig) -> Tuple[int, int]:
        return select_policy_move(state, model, config.temperature)
    
    def get_name(self) -> str:
        return "policy"
    
    def get_config_summary(self, config: MoveSelectionConfig) -> str:
        return f"policy(t={config.temperature})"


class FixedTreeSearchStrategy(MoveSelectionStrategy):
    """Move selection using fixed-width minimax search."""
    
    def select_move(self, state: HexGameState, model: SimpleModelInference, 
                   config: MoveSelectionConfig) -> Tuple[int, int]:
        if not config.search_widths:
            raise ValueError("FixedTreeSearchStrategy requires search_widths configuration")
        
        move, _ = minimax_policy_value_search(
            state, model, config.search_widths, temperature=config.temperature
        )
        return move
    
    def get_name(self) -> str:
        return "fixed_tree"
    
    def get_config_summary(self, config: MoveSelectionConfig) -> str:
        if not config.search_widths:
            return "fixed_tree(no_widths)"
        return f"fixed_tree(widths={config.search_widths}, t={config.temperature})"


class MCTSStrategy(MoveSelectionStrategy):
    """Move selection using MCTS."""
    
    def select_move(self, state: HexGameState, model: SimpleModelInference, 
                   config: MoveSelectionConfig) -> Tuple[int, int]:
        # Create MCTS configuration optimized for tournament play
        mcts_config = create_mcts_config("tournament",
            sims=config.mcts_sims,
            early_termination_threshold=0.95  # Conservative early termination for quality
        )
        
        # Create required components
        engine = HexGameEngine()
        model_cache = get_model_cache()
        model_wrapper = model_cache.get_wrapper_model(model.checkpoint_path)
        
        # Run MCTS and select move
        mcts = BaselineMCTS(engine, model_wrapper, mcts_config)
        mcts.run(state, verbose=0)  # Quiet mode for tournaments
        return mcts.pick_move(state, temperature=config.temperature)
    
    def get_name(self) -> str:
        return "mcts"
    
    def get_config_summary(self, config: MoveSelectionConfig) -> str:
        return f"mcts(sims={config.mcts_sims}, c_puct={config.mcts_c_puct}, t={config.temperature})"


# Strategy registry
STRATEGY_REGISTRY = {
    "policy": PolicyBasedStrategy(),
    "fixed_tree": FixedTreeSearchStrategy(),
    "mcts": MCTSStrategy(),
}


def get_strategy(strategy_name: str) -> MoveSelectionStrategy:
    """Get a move selection strategy by name."""
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[strategy_name]


def list_available_strategies() -> list:
    """List all available move selection strategies."""
    return list(STRATEGY_REGISTRY.keys())
