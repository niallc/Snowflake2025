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
from hex_ai.inference.fixed_tree_search import run_fixed_tree_search, create_fixed_tree_config
from hex_ai.inference.mcts import BaselineMCTS, BaselineMCTSConfig, create_mcts_config
from hex_ai.inference.model_cache import get_model_cache
from hex_ai.config import (
    DEFAULT_GUMBEL_SIM_THRESHOLD,
    DEFAULT_GUMBEL_CANDIDATE_POWER_SCALE,
    DEFAULT_GUMBEL_CANDIDATE_POWER_RATE,
    DEFAULT_GUMBEL_CANDIDATE_POWER_OFFSET,
    DEFAULT_GUMBEL_CANDIDATE_MIN,
    DEFAULT_GUMBEL_CANDIDATE_MAX,
    DEFAULT_C_PUCT,
    DEFAULT_MCTS_SIMS,
    DEFAULT_MCTS_DIRICHLET_ALPHA,
    DEFAULT_MCTS_DIRICHLET_EPS,
    DEFAULT_GUMBEL_C_VISIT,
    DEFAULT_GUMBEL_C_SCALE,
    DEFAULT_GUMBEL_USE_GUMBEL_IN_FINAL_EVAL
)


@dataclass
class MoveSelectionConfig:
    """Configuration for move selection strategies."""
    temperature: float  # No default - must be specified
    # For MCTS
    mcts_sims: int = DEFAULT_MCTS_SIMS
    mcts_c_puct: float = DEFAULT_C_PUCT
    mcts_dirichlet_alpha: float = DEFAULT_MCTS_DIRICHLET_ALPHA
    mcts_dirichlet_eps: float = DEFAULT_MCTS_DIRICHLET_EPS
    batch_size: Optional[int] = None  # Override default batch size for MCTS
    # For Gumbel AlphaZero root selection
    enable_gumbel_root_selection: bool = True  # Enable Gumbel-AlphaZero root selection
    gumbel_sim_threshold: int = DEFAULT_GUMBEL_SIM_THRESHOLD  # Use Gumbel selection when sims <= this threshold
    gumbel_c_visit: float = DEFAULT_GUMBEL_C_VISIT  # Gumbel-AlphaZero c_visit parameter
    gumbel_c_scale: float = DEFAULT_GUMBEL_C_SCALE  # Gumbel-AlphaZero c_scale parameter
    gumbel_m_candidates: Optional[int] = None  # Number of candidates to consider (None for auto)
    # Gumbel candidate scaling parameters (power-law scaling)
    gumbel_candidate_power_scale: float = DEFAULT_GUMBEL_CANDIDATE_POWER_SCALE  # Scale factor for power-law candidate scaling
    gumbel_candidate_power_rate: float = DEFAULT_GUMBEL_CANDIDATE_POWER_RATE  # Rate (exponent) for power-law candidate scaling
    gumbel_candidate_power_offset: float = DEFAULT_GUMBEL_CANDIDATE_POWER_OFFSET  # Offset for power-law candidate scaling
    gumbel_candidate_min: int = DEFAULT_GUMBEL_CANDIDATE_MIN  # Minimum number of candidates
    gumbel_candidate_max: int = DEFAULT_GUMBEL_CANDIDATE_MAX  # Maximum number of candidates
    # Gumbel ranking stabilization parameters
    gumbel_use_gumbel_in_final_eval: bool = DEFAULT_GUMBEL_USE_GUMBEL_IN_FINAL_EVAL  # Remove Gumbel noise in final evaluation
    # For fixed tree search
    search_widths: Optional[list] = None
    # For policy-based selection
    policy_top_k: Optional[int] = None


class MoveSelectionStrategy(ABC):
    """Abstract base class for move selection strategies."""
    
    @abstractmethod
    def select_move(self, state: HexGameState, model: SimpleModelInference, 
                   config: MoveSelectionConfig, verbose: int = 0) -> Tuple[int, int]:
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
                   config: MoveSelectionConfig, verbose: int = 0) -> Tuple[int, int]:
        return select_policy_move(state, model, config.temperature)
    
    def get_name(self) -> str:
        return "policy"
    
    def get_config_summary(self, config: MoveSelectionConfig) -> str:
        return f"policy(t={config.temperature})"


class FixedTreeSearchStrategy(MoveSelectionStrategy):
    """Move selection using fixed-width minimax search."""
    
    def select_move(self, state: HexGameState, model: SimpleModelInference, 
                   config: MoveSelectionConfig, verbose: int = 0) -> Tuple[int, int]:
        if not config.search_widths:
            raise ValueError("FixedTreeSearchStrategy requires search_widths configuration")
        
        # Create modern config and run search
        search_config = create_fixed_tree_config(
            search_widths=config.search_widths,
            temperature=config.temperature,
            batch_size=1000  # Default batch size
        )
        result = run_fixed_tree_search(state, model, search_config, verbose)
        return result.move
    
    def get_name(self) -> str:
        return "fixed_tree"
    
    def get_config_summary(self, config: MoveSelectionConfig) -> str:
        if not config.search_widths:
            return "fixed_tree(no_widths)"
        return f"fixed_tree(widths={config.search_widths}, t={config.temperature})"


class MCTSStrategy(MoveSelectionStrategy):
    """Move selection using MCTS."""

    def __init__(self, verbose: int = 0):
        self.verbose = verbose

    def select_move(self, state: HexGameState, model: SimpleModelInference, 
                   config: MoveSelectionConfig, verbose: int = 0) -> Tuple[int, int]:
        # Create MCTS configuration optimized for tournament play
        # Pass all parameters through create_mcts_config for consistency
        mcts_config = create_mcts_config("tournament",
            sims=config.mcts_sims,
            confidence_termination_threshold=0.95,  # Conservative confidence termination for quality
            c_puct=config.mcts_c_puct,  # Pass the c_puct parameter from strategy config
            dirichlet_alpha=config.mcts_dirichlet_alpha,  # Pass the dirichlet_alpha parameter
            dirichlet_eps=config.mcts_dirichlet_eps,  # Pass the dirichlet_eps parameter
            enable_depth_discounting=False,  # Disable depth discounting for tournament play
            enable_gumbel_root_selection=config.enable_gumbel_root_selection,  # Pass the gumbel parameter from strategy config
            # Pass all Gumbel parameters through create_mcts_config for consistency
            gumbel_sim_threshold=config.gumbel_sim_threshold,
            gumbel_c_visit=config.gumbel_c_visit,
            gumbel_c_scale=config.gumbel_c_scale,
            gumbel_m_candidates=config.gumbel_m_candidates,
            gumbel_candidate_log_base=config.gumbel_candidate_log_base,
            gumbel_candidate_log_offset=config.gumbel_candidate_log_offset,
            gumbel_candidate_min=config.gumbel_candidate_min,
            gumbel_candidate_max=config.gumbel_candidate_max
        )
        
        # Override batch size if specified in config
        if config.batch_size is not None:
            mcts_config.batch_cap = config.batch_size
        
        # Set temperature from tournament configuration
        # For deterministic tournaments, we want to use the same temperature for all moves
        # rather than temperature decay, so set both start and end to the same value
        mcts_config.temperature_start = config.temperature
        mcts_config.temperature_end = config.temperature
        
        # Disable Dirichlet noise for deterministic tournaments
        mcts_config.add_root_noise = False
        
        
        # Create required components
        engine = HexGameEngine()
        model_cache = get_model_cache()
        model_wrapper = model_cache.get_wrapper_model(model.checkpoint_path)
        
        # Run MCTS and select move
        mcts = BaselineMCTS(engine, model_wrapper, mcts_config)
        
        # DEBUG: Print MCTS configuration to confirm what's actually being used
        if verbose >= 5:
            print(f"[MCTS DEBUG] add_root_noise={mcts_config.add_root_noise}, dirichlet_alpha={mcts_config.dirichlet_alpha}, dirichlet_eps={mcts_config.dirichlet_eps}")
        
        result = mcts.run(state, verbose=verbose)  # Use passed verbose parameter
        return result.move
    
    def get_name(self) -> str:
        return "mcts"
    
    def get_config_summary(self, config: MoveSelectionConfig) -> str:
        batch_info = f", batch={config.batch_size}" if config.batch_size is not None else ""
        gumbel_info = ""
        if config.enable_gumbel_root_selection:
            gumbel_info = f", gumbel(c_visit={config.gumbel_c_visit}, c_scale={config.gumbel_c_scale}, m={config.gumbel_m_candidates})"
        return f"mcts(sims={config.mcts_sims}, c_puct={config.mcts_c_puct}, t={config.temperature}{batch_info}{gumbel_info})"


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
