"""
Shared utilities for strategy configuration parsing.

This module consolidates the duplicate strategy parsing code that was previously
scattered across multiple tournament scripts.
"""

import os
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from hex_ai.config import (
    DEFAULT_C_PUCT, DEFAULT_MCTS_SIMS, DEFAULT_GUMBEL_C_SCALE, BOARD_SIZE,
    DEFAULT_GUMBEL_CANDIDATE_POWER_SCALE, DEFAULT_GUMBEL_CANDIDATE_POWER_RATE, DEFAULT_GUMBEL_CANDIDATE_POWER_OFFSET
)
from hex_ai.inference.tournament_parameters import (
    TournamentParameterConfig, TournamentModelConfig, UnifiedTournamentConfig
)


@dataclass
class StrategyConfig:
    """Configuration for a single strategy."""
    
    name: str
    strategy_type: str
    config: Dict[str, Any]
    model_path: str  # Model checkpoint path for this strategy
    original_name: str = None  # Original name before parameter modifications
    temperature: float = None  # Strategy-specific temperature (None for global fallback)
    
    def __post_init__(self):
        if self.original_name is None:
            self.original_name = self.name
    
    def __str__(self) -> str:
        return f"{self.name}({self.strategy_type})"



def create_strategy_configs_from_unified_config(unified_config: UnifiedTournamentConfig) -> List[StrategyConfig]:
    """
    Create StrategyConfig objects from a UnifiedTournamentConfig.
    
    This function provides a bridge between the new unified parameter system
    and the existing StrategyConfig class.
    
    Args:
        unified_config: The unified tournament configuration
    
    Returns:
        List of StrategyConfig objects
    
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate the unified config first
    unified_config.validate()
    
    configs = []
    num_strategies = len(unified_config.strategies)
    
    for i in range(num_strategies):
        participant_label = f"strategy_{i}"
        participant_config = unified_config.get_participant_config(participant_label, i)
        
        # Create strategy config
        strategy_name = unified_config.strategies[i]
        strategy_type = _determine_strategy_type(strategy_name)
        
        # Build config dictionary
        config_dict = {}
        
        # Add strategy-specific parameters
        if strategy_type == "mcts":
            config_dict["mcts_sims"] = participant_config["mcts_sims"]
            config_dict["mcts_c_puct"] = participant_config.get("c_puct", DEFAULT_C_PUCT)
            config_dict["batch_size"] = participant_config.get("batch_size", 64)
            
            # Add Gumbel parameters if specified
            if "enable_gumbel" in participant_config:
                config_dict["enable_gumbel_root_selection"] = participant_config["enable_gumbel"]
            if "gumbel_sim_threshold" in participant_config:
                config_dict["gumbel_sim_threshold"] = participant_config["gumbel_sim_threshold"]
            if "gumbel_candidate_power_scale" in participant_config:
                config_dict["gumbel_candidate_power_scale"] = participant_config["gumbel_candidate_power_scale"]
            if "gumbel_candidate_power_rate" in participant_config:
                config_dict["gumbel_candidate_power_rate"] = participant_config["gumbel_candidate_power_rate"]
            if "gumbel_candidate_power_offset" in participant_config:
                config_dict["gumbel_candidate_power_offset"] = participant_config["gumbel_candidate_power_offset"]
            if "gumbel_progressive_widening" in participant_config:
                config_dict["gumbel_progressive_widening"] = participant_config["gumbel_progressive_widening"]
            if "gumbel_batch_scaling_factor" in participant_config:
                config_dict["gumbel_batch_scaling_factor"] = participant_config["gumbel_batch_scaling_factor"]
            if "gumbel_c_scale" in participant_config:
                config_dict["gumbel_c_scale"] = participant_config["gumbel_c_scale"]
        elif strategy_type == "policy":
            # Policy strategies need minimal config - just temperature for consistency
            # The temperature is also stored as a separate field on StrategyConfig
            pass  # Temperature is handled separately, policy strategies don't need other params
        
        
        # Create StrategyConfig
        strategy_config = StrategyConfig(
            name=strategy_name,
            strategy_type=strategy_type,
            config=config_dict,
            model_path=participant_config["model_path"],
            original_name=strategy_name,
            temperature=participant_config["temperature"]
        )
        
        configs.append(strategy_config)
    
    return configs


def _determine_strategy_type(strategy_name: str) -> str:
    """
    Determine the strategy type from a strategy name.
    
    Args:
        strategy_name: Name of the strategy
    
    Returns:
        Strategy type ("policy" or "mcts")
    
    Raises:
        ValueError: If strategy name is invalid
    """
    if strategy_name == "policy":
        return "policy"
    elif strategy_name == "mcts":
        return "mcts"
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def create_unified_config_from_args(
    strategies: List[str],
    models: Optional[List[str]] = None,
    model_paths: Optional[List[str]] = None,
    mcts_sims: Optional[Union[int, List[int]]] = None,
    temperatures: Optional[Union[float, List[float]]] = None,
    batch_sizes: Optional[Union[int, List[int]]] = None,
    c_pucts: Optional[Union[float, List[float]]] = None,
    enable_gumbel: Optional[Union[bool, List[bool]]] = None,
    gumbel_sim_thresholds: Optional[Union[int, List[int]]] = None,
    gumbel_candidate_power_scales: Optional[Union[float, List[float]]] = None,
    gumbel_candidate_power_rates: Optional[Union[float, List[float]]] = None,
    gumbel_candidate_power_offsets: Optional[Union[float, List[float]]] = None,
    gumbel_progressive_widening: Optional[Union[bool, List[bool]]] = None,
    gumbel_batch_scaling_factors: Optional[Union[float, List[float]]] = None,
    gumbel_c_scales: Optional[Union[float, List[float]]] = None,
    num_games: int = 10,
    board_size: int = BOARD_SIZE,
    random_seed: Optional[int] = None,
    pie_rule: bool = False
) -> UnifiedTournamentConfig:
    """
    Create a UnifiedTournamentConfig from command line arguments.
    
    This function provides a convenient way to create the unified config
    from the various argument formats used by tournament scripts.
    
    Args:
        strategies: List of strategy names
        models: Optional list of model registry names
        model_paths: Optional list of direct model paths
        mcts_sims: MCTS simulation count(s)
        temperatures: Temperature value(s)
        batch_sizes: Batch size value(s)
        c_pucts: C_PUCT value(s)
        enable_gumbel: Gumbel enable flag(s)
        gumbel_sim_thresholds: Gumbel simulation threshold(s)
        gumbel_candidate_power_scales: Gumbel candidate power scale(s)
        gumbel_candidate_power_rates: Gumbel candidate power rate(s)
        gumbel_candidate_power_offsets: Gumbel candidate power offset(s)
        gumbel_progressive_widening: Gumbel progressive widening flag(s)
        gumbel_batch_scaling_factors: Gumbel batch scaling factor(s)
        gumbel_c_scales: Gumbel c_scale parameter(s)
        num_games: Number of games per pair
        board_size: Board size
        random_seed: Random seed
        pie_rule: Whether to use pie rule
    
    Returns:
        UnifiedTournamentConfig object
    
    Raises:
        ValueError: If arguments are invalid
    """
    num_strategies = len(strategies)
    
    # Create model configuration
    if model_paths:
        model_config = TournamentModelConfig(model_paths=model_paths)
    elif models:
        model_config = TournamentModelConfig(model_names=models)
    else:
        model_config = TournamentModelConfig()  # Use default model
    
    # Create parameter configurations
    mcts_sims_config = TournamentParameterConfig(
        default_value=DEFAULT_MCTS_SIMS,  # Default MCTS simulations
        per_strategy_values=to_list_if_needed(mcts_sims, num_strategies)
    )
    
    temperatures_config = TournamentParameterConfig(
        default_value=0.0,  # Default temperature
        per_strategy_values=to_list_if_needed(temperatures, num_strategies)
    )
    
    # Optional parameter configurations
    batch_sizes_config = None
    if batch_sizes is not None:
        batch_sizes_config = TournamentParameterConfig(
            default_value=64,  # Default batch size
            per_strategy_values=to_list_if_needed(batch_sizes, num_strategies)
        )
    
    c_pucts_config = None
    if c_pucts is not None:
        c_pucts_config = TournamentParameterConfig(
            default_value=DEFAULT_C_PUCT,  # Default C_PUCT
            per_strategy_values=to_list_if_needed(c_pucts, num_strategies)
        )
    
    enable_gumbel_config = None
    if enable_gumbel is not None:
        enable_gumbel_config = TournamentParameterConfig(
            default_value=False,  # Default Gumbel disabled
            per_strategy_values=to_list_if_needed(enable_gumbel, num_strategies)
        )
    
    gumbel_sim_thresholds_config = None
    if gumbel_sim_thresholds is not None:
        gumbel_sim_thresholds_config = TournamentParameterConfig(
            default_value=200,  # Default Gumbel threshold
            per_strategy_values=to_list_if_needed(gumbel_sim_thresholds, num_strategies)
        )
    
    gumbel_candidate_power_scales_config = None
    if gumbel_candidate_power_scales is not None:
        gumbel_candidate_power_scales_config = TournamentParameterConfig(
            default_value=DEFAULT_GUMBEL_CANDIDATE_POWER_SCALE,
            per_strategy_values=to_list_if_needed(gumbel_candidate_power_scales, num_strategies)
        )
    
    gumbel_candidate_power_rates_config = None
    if gumbel_candidate_power_rates is not None:
        gumbel_candidate_power_rates_config = TournamentParameterConfig(
            default_value=DEFAULT_GUMBEL_CANDIDATE_POWER_RATE,
            per_strategy_values=to_list_if_needed(gumbel_candidate_power_rates, num_strategies)
        )
    
    gumbel_candidate_power_offsets_config = None
    if gumbel_candidate_power_offsets is not None:
        gumbel_candidate_power_offsets_config = TournamentParameterConfig(
            default_value=DEFAULT_GUMBEL_CANDIDATE_POWER_OFFSET,
            per_strategy_values=to_list_if_needed(gumbel_candidate_power_offsets, num_strategies)
        )
    
    gumbel_progressive_widening_config = None
    if gumbel_progressive_widening is not None:
        gumbel_progressive_widening_config = TournamentParameterConfig(
            default_value=False,  # Default progressive widening disabled
            per_strategy_values=to_list_if_needed(gumbel_progressive_widening, num_strategies)
        )
    
    gumbel_batch_scaling_factors_config = None
    if gumbel_batch_scaling_factors is not None:
        gumbel_batch_scaling_factors_config = TournamentParameterConfig(
            default_value=1.0,  # Default scaling factor
            per_strategy_values=to_list_if_needed(gumbel_batch_scaling_factors, num_strategies)
        )
    
    gumbel_c_scales_config = None
    if gumbel_c_scales is not None:
        gumbel_c_scales_config = TournamentParameterConfig(
            default_value=DEFAULT_GUMBEL_C_SCALE,  # Default c_scale from config
            per_strategy_values=to_list_if_needed(gumbel_c_scales, num_strategies)
        )
    
    return UnifiedTournamentConfig(
        models=model_config,
        strategies=strategies,
        mcts_sims=mcts_sims_config,
        temperatures=temperatures_config,
        batch_sizes=batch_sizes_config,
        c_pucts=c_pucts_config,
        enable_gumbel=enable_gumbel_config,
        gumbel_sim_thresholds=gumbel_sim_thresholds_config,
        gumbel_candidate_power_scales=gumbel_candidate_power_scales_config,
        gumbel_candidate_power_offsets=gumbel_candidate_power_offsets_config,
        gumbel_progressive_widening=gumbel_progressive_widening_config,
        gumbel_batch_scaling_factors=gumbel_batch_scaling_factors_config,
        gumbel_c_scales=gumbel_c_scales_config,
        num_games=num_games,
        board_size=board_size,
        random_seed=random_seed,
        pie_rule=pie_rule
    )


def to_list_if_needed(
    value: Optional[Union[Any, List[Any]]], 
    num_strategies: int,
    parameter_name: str = "parameter",
    original_values: Optional[List[Any]] = None,
    strategy_names: Optional[List[str]] = None,
    exit_on_error: bool = True
) -> Optional[List[Any]]:
    """
    Convert a single value to a list if needed, or return None if value is None.
    
    Args:
        value: Single value or list of values
        num_strategies: Number of strategies (used for validation)
        parameter_name: Name of the parameter for error messages (e.g., "models", "mcts_sims")
        original_values: Original values before processing (for error display)
        strategy_names: List of strategy names (for error display)
        exit_on_error: Whether to call sys.exit(1) on validation error (default: True)
    
    Returns:
        List of values or None
    
    Raises:
        ValueError: If list length doesn't match number of strategies and exit_on_error=False
    """
    if value is None:
        return None
    
    if isinstance(value, list):
        if len(value) == 1:
            # Single value in list - apply to all strategies
            return [value[0]] * num_strategies
        elif len(value) != num_strategies:
            error_msg = f"List length ({len(value)}) must match number of strategies ({num_strategies})"
            if original_values is not None:
                error_msg += f"\n  Provided {parameter_name}: {original_values}"
            if strategy_names is not None:
                error_msg += f"\n  Strategies: {strategy_names}"
            error_msg += f"\n  Tip: You can provide a single {parameter_name} to use for all strategies"
            
            if exit_on_error:
                print(f"ERROR: {error_msg}")
                import sys
                sys.exit(1)
            else:
                raise ValueError(error_msg)
        return value
    else:
        # Single value - apply to all strategies
        return [value] * num_strategies
