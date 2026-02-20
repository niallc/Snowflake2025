"""
Tournament parameter configuration system.

This module provides a systematic approach to handling tournament parameters,
replacing the inconsistent parameter handling that was previously scattered
across multiple tournament scripts.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import os

from hex_ai.config import BOARD_SIZE
from hex_ai.inference.model_config import get_model_path, validate_model_path, get_available_models


@dataclass
class TournamentParameterConfig:
    """
    Configuration for a tournament parameter with multiple specification methods.
    
    This class provides a consistent interface for parameters that can be specified as:
    1. A global default value
    2. A per-strategy vector (applied in order)
    3. A per-participant mapping (by participant label)
    
    The priority order is: per_participant_mapping > per_strategy_values > default_value
    """
    
    def __init__(self, 
                 default_value: Union[int, float],
                 per_strategy_values: Optional[List[Union[int, float]]] = None,
                 per_participant_mapping: Optional[Dict[str, Union[int, float]]] = None):
        self.default_value = default_value
        self.per_strategy_values = per_strategy_values
        self.per_participant_mapping = per_participant_mapping
    
    def get_value_for_participant(self, participant_label: str, strategy_index: int) -> Union[int, float]:
        """
        Get the parameter value for a specific participant.
        
        Args:
            participant_label: Label/name of the participant
            strategy_index: Index of the strategy in the strategy list (0-based)
        
        Returns:
            The parameter value for this participant
        
        Priority order:
        1. per_participant_mapping (if participant_label exists)
        2. per_strategy_values (if strategy_index is valid)
        3. default_value (fallback)
        """
        # Priority 1: Per-participant mapping
        if self.per_participant_mapping and participant_label in self.per_participant_mapping:
            return self.per_participant_mapping[participant_label]
        
        # Priority 2: Per-strategy values
        elif self.per_strategy_values and strategy_index < len(self.per_strategy_values):
            return self.per_strategy_values[strategy_index]
        
        # Priority 3: Default value
        else:
            return self.default_value
    
    def validate(self, num_strategies: int, participant_labels: List[str]) -> None:
        """
        Validate the parameter configuration.
        
        Args:
            num_strategies: Number of strategies in the tournament
            participant_labels: List of participant labels
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate per_strategy_values length
        if self.per_strategy_values and len(self.per_strategy_values) != num_strategies:
            raise ValueError(
                f"per_strategy_values length ({len(self.per_strategy_values)}) "
                f"must match number of strategies ({num_strategies})"
            )
        
        # Validate per_participant_mapping keys
        if self.per_participant_mapping:
            invalid_labels = set(self.per_participant_mapping.keys()) - set(participant_labels)
            if invalid_labels:
                raise ValueError(
                    f"per_participant_mapping contains invalid participant labels: {invalid_labels}"
                )


@dataclass
class TournamentModelConfig:
    """
    Configuration for tournament models with systematic handling.
    
    This class provides a consistent interface for model specification that can be:
    1. A default model from the model registry
    2. A list of model registry names
    3. A list of direct model paths
    4. A per-participant mapping
    """
    
    def __init__(self,
                 default_model: str = "best",
                 model_names: Optional[List[str]] = None,
                 model_paths: Optional[List[str]] = None,
                 per_participant_models: Optional[Dict[str, str]] = None):
        self.default_model = default_model
        self.model_names = model_names
        self.model_paths = model_paths
        self.per_participant_models = per_participant_models
    
    def get_model_path_for_participant(self, participant_label: str, strategy_index: int) -> str:
        """
        Get the model path for a specific participant.
        
        Args:
            participant_label: Label/name of the participant
            strategy_index: Index of the strategy in the strategy list (0-based)
        
        Returns:
            The model path for this participant
        
        Priority order:
        1. per_participant_models (if participant_label exists)
        2. model_paths (if strategy_index is valid)
        3. model_names (if strategy_index is valid, resolved through registry)
        4. default_model (resolved through registry)
        """
        # Priority 1: Per-participant models
        if self.per_participant_models and participant_label in self.per_participant_models:
            model_spec = self.per_participant_models[participant_label]
            # Try to resolve as model registry name first
            try:
                return get_model_path(model_spec)
            except ValueError as e:
                # Model registry lookup failed - this is a configuration error
                raise ValueError(
                    f"Model specification '{model_spec}' for participant '{participant_label}' "
                    f"is not a valid model registry name. "
                    f"Available models: {get_available_models()}. "
                    f"Original error: {e}"
                )
        
        # Priority 2: Direct model paths
        elif self.model_paths and strategy_index < len(self.model_paths):
            model_path = self.model_paths[strategy_index]
            if not os.path.exists(model_path):
                raise ValueError(f"Model path does not exist: {model_path}")
            return model_path
        
        # Priority 3: Model registry names
        elif self.model_names and strategy_index < len(self.model_names):
            return get_model_path(self.model_names[strategy_index])
        
        # Priority 4: Default model
        else:
            return get_model_path(self.default_model)
    
    def validate(self, num_strategies: int, participant_labels: List[str]) -> None:
        """
        Validate the model configuration.
        
        Args:
            num_strategies: Number of strategies in the tournament
            participant_labels: List of participant labels
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate model_names length
        if self.model_names and len(self.model_names) != num_strategies:
            raise ValueError(
                f"model_names length ({len(self.model_names)}) "
                f"must match number of strategies ({num_strategies})"
            )
        
        # Validate model_paths length
        if self.model_paths and len(self.model_paths) != num_strategies:
            raise ValueError(
                f"model_paths length ({len(self.model_paths)}) "
                f"must match number of strategies ({num_strategies})"
            )
        
        # Validate per_participant_models keys
        if self.per_participant_models:
            invalid_labels = set(self.per_participant_models.keys()) - set(participant_labels)
            if invalid_labels:
                raise ValueError(
                    f"per_participant_models contains invalid participant labels: {invalid_labels}"
                )
        
        # Validate that all model specifications are valid
        for i in range(num_strategies):
            try:
                model_path = self.get_model_path_for_participant(participant_labels[i], i)
                if not validate_model_path(model_path):
                    raise ValueError(f"Invalid model path: {model_path}")
            except Exception as e:
                raise ValueError(f"Model validation failed for strategy {i}: {e}")


@dataclass
class UnifiedTournamentConfig:
    """
    Unified configuration for all tournament parameters.
    
    This class consolidates all tournament parameters into a single, consistent interface.
    """
    
    def __init__(self,
                 # Model configuration
                 models: TournamentModelConfig,
                 
                 # Strategy configuration
                 strategies: List[str],
                 
                 # Parameter configurations
                 mcts_sims: TournamentParameterConfig,
                 temperatures: TournamentParameterConfig,
                 batch_sizes: Optional[TournamentParameterConfig] = None,
                 c_pucts: Optional[TournamentParameterConfig] = None,
                 enable_gumbel: Optional[TournamentParameterConfig] = None,
                 gumbel_sim_thresholds: Optional[TournamentParameterConfig] = None,
                 gumbel_candidate_power_scales: Optional[TournamentParameterConfig] = None,
                 gumbel_candidate_power_rates: Optional[TournamentParameterConfig] = None,
                 gumbel_candidate_power_offsets: Optional[TournamentParameterConfig] = None,
                 gumbel_progressive_widening: Optional[TournamentParameterConfig] = None,
                 gumbel_batch_scaling_factors: Optional[TournamentParameterConfig] = None,
                 gumbel_c_scales: Optional[TournamentParameterConfig] = None,
                 
                 # Tournament settings
                 num_games: int = 10,
                 board_size: int = BOARD_SIZE,
                 random_seed: Optional[int] = None,
                 pie_rule: bool = False):
        
        self.models = models
        self.strategies = strategies
        self.mcts_sims = mcts_sims
        self.temperatures = temperatures
        self.batch_sizes = batch_sizes
        self.c_pucts = c_pucts
        self.enable_gumbel = enable_gumbel
        self.gumbel_sim_thresholds = gumbel_sim_thresholds
        self.gumbel_candidate_power_scales = gumbel_candidate_power_scales
        self.gumbel_candidate_power_rates = gumbel_candidate_power_rates
        self.gumbel_candidate_power_offsets = gumbel_candidate_power_offsets
        self.gumbel_progressive_widening = gumbel_progressive_widening
        self.gumbel_batch_scaling_factors = gumbel_batch_scaling_factors
        self.gumbel_c_scales = gumbel_c_scales
        self.num_games = num_games
        self.board_size = board_size
        self.random_seed = random_seed
        self.pie_rule = pie_rule
    
    def validate(self) -> None:
        """
        Validate the entire tournament configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        num_strategies = len(self.strategies)
        participant_labels = [f"strategy_{i}" for i in range(num_strategies)]
        
        # Validate all parameter configurations
        self.models.validate(num_strategies, participant_labels)
        self.mcts_sims.validate(num_strategies, participant_labels)
        self.temperatures.validate(num_strategies, participant_labels)
        
        if self.batch_sizes:
            self.batch_sizes.validate(num_strategies, participant_labels)
        if self.c_pucts:
            self.c_pucts.validate(num_strategies, participant_labels)
        if self.enable_gumbel:
            self.enable_gumbel.validate(num_strategies, participant_labels)
        if self.gumbel_sim_thresholds:
            self.gumbel_sim_thresholds.validate(num_strategies, participant_labels)
        if self.gumbel_candidate_power_scales:
            self.gumbel_candidate_power_scales.validate(num_strategies, participant_labels)
        if self.gumbel_candidate_power_rates:
            self.gumbel_candidate_power_rates.validate(num_strategies, participant_labels)
        if self.gumbel_candidate_power_offsets:
            self.gumbel_candidate_power_offsets.validate(num_strategies, participant_labels)
        if self.gumbel_progressive_widening:
            self.gumbel_progressive_widening.validate(num_strategies, participant_labels)
        if self.gumbel_batch_scaling_factors:
            self.gumbel_batch_scaling_factors.validate(num_strategies, participant_labels)
        
        # Validate basic tournament settings
        if self.num_games <= 0:
            raise ValueError(f"num_games must be positive, got {self.num_games}")
        if self.board_size <= 0:
            raise ValueError(f"board_size must be positive, got {self.board_size}")
    
    def get_participant_config(self, participant_label: str, strategy_index: int) -> Dict[str, Any]:
        """
        Get the complete configuration for a specific participant.
        
        Args:
            participant_label: Label/name of the participant
            strategy_index: Index of the strategy in the strategy list (0-based)
        
        Returns:
            Dictionary containing all configuration parameters for this participant
        """
        config = {
            'model_path': self.models.get_model_path_for_participant(participant_label, strategy_index),
            'strategy': self.strategies[strategy_index],
            'mcts_sims': self.mcts_sims.get_value_for_participant(participant_label, strategy_index),
            'temperature': self.temperatures.get_value_for_participant(participant_label, strategy_index),
        }
        
        # Add optional parameters if they exist
        if self.batch_sizes:
            config['batch_size'] = self.batch_sizes.get_value_for_participant(participant_label, strategy_index)
        if self.c_pucts:
            config['c_puct'] = self.c_pucts.get_value_for_participant(participant_label, strategy_index)
        if self.enable_gumbel:
            config['enable_gumbel'] = self.enable_gumbel.get_value_for_participant(participant_label, strategy_index)
        if self.gumbel_sim_thresholds:
            config['gumbel_sim_threshold'] = self.gumbel_sim_thresholds.get_value_for_participant(participant_label, strategy_index)
        if self.gumbel_candidate_power_scales:
            config['gumbel_candidate_power_scale'] = self.gumbel_candidate_power_scales.get_value_for_participant(participant_label, strategy_index)
        if self.gumbel_candidate_power_rates:
            config['gumbel_candidate_power_rate'] = self.gumbel_candidate_power_rates.get_value_for_participant(participant_label, strategy_index)
        if self.gumbel_candidate_power_offsets:
            config['gumbel_candidate_power_offset'] = self.gumbel_candidate_power_offsets.get_value_for_participant(participant_label, strategy_index)
        if self.gumbel_progressive_widening:
            config['gumbel_progressive_widening'] = self.gumbel_progressive_widening.get_value_for_participant(participant_label, strategy_index)
        if self.gumbel_batch_scaling_factors:
            config['gumbel_batch_scaling_factor'] = self.gumbel_batch_scaling_factors.get_value_for_participant(participant_label, strategy_index)
        if self.gumbel_c_scales:
            config['gumbel_c_scale'] = self.gumbel_c_scales.get_value_for_participant(participant_label, strategy_index)
        
        return config
