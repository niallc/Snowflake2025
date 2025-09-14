"""
Unified logging and configuration utilities for all script types.

This module provides consistent logging, configuration printing, and results analysis
across tournament and selfplay scripts to reduce duplication and improve maintainability.
"""

import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

from hex_ai.system_utils import get_git_commit_info
from hex_ai.utils.gumbel_utils import generate_gumbel_summary_from_configs, generate_gumbel_summary_from_params


@dataclass
class ScriptConfig:
    """Configuration container for script logging."""
    script_type: str  # "tournament", "deterministic_tournament", "selfplay"
    models: List[str]
    strategies: List[str]
    num_games: int
    strategy_config: Dict[str, Any]
    temperatures: Union[float, List[float], Dict[str, float]]
    random_seed: int
    pie_rule: bool = False
    opening_length: Optional[int] = None
    opening_strategy: Optional[str] = None
    batch_size: Optional[int] = None
    batch_sizes: Optional[List[int]] = None
    cache_size: Optional[int] = None
    mcts_sims: Optional[int] = None
    c_puct: Optional[float] = None
    enable_gumbel: Optional[bool] = None
    gumbel_sim_threshold: Optional[int] = None
    gumbel_c_visit: Optional[float] = None
    gumbel_c_scale: Optional[float] = None
    gumbel_candidate_log_base: Optional[float] = None
    gumbel_candidate_log_offset: Optional[float] = None
    gumbel_m_candidates: Optional[int] = None
    search_widths: Optional[List[int]] = None
    output_dir: Optional[str] = None
    checkpoint_dirs: Optional[List[str]] = None
    default_checkpoint_dir: Optional[str] = None
    temperature_end: Optional[float] = None
    no_batched_inference: Optional[bool] = None


class ConfigurationPrinter:
    """Base class for configuration printing strategies."""
    
    def print_header(self, config: ScriptConfig) -> None:
        """Print the script type header."""
        script_title = {
            "tournament": "Tournament Configuration",
            "deterministic_tournament": "Deterministic Tournament Configuration", 
            "selfplay": "Self-Play Configuration"
        }.get(config.script_type, "Script Configuration")
        print(f"\n{script_title}:")
    
    def print_models(self, config: ScriptConfig) -> None:
        """Print model/participant information."""
        raise NotImplementedError
    
    def print_games(self, config: ScriptConfig) -> None:
        """Print game configuration."""
        raise NotImplementedError
    
    def print_strategy(self, config: ScriptConfig) -> None:
        """Print strategy information."""
        raise NotImplementedError
    
    def print_specific_config(self, config: ScriptConfig) -> None:
        """Print script-specific configuration."""
        pass  # Default: no specific config
    
    def print_common_config(self, config: ScriptConfig, include_git: bool, include_timestamp: bool, include_gumbel: bool) -> None:
        """Print common configuration elements."""
        # Print checkpoint directories if available
        if config.checkpoint_dirs:
            print(f"  Checkpoint directories: {config.checkpoint_dirs}")
        elif config.default_checkpoint_dir:
            print(f"  Checkpoint directory: {config.default_checkpoint_dir}")
        
        # Print temperature information
        if isinstance(config.temperatures, dict):
            print(f"  Temperature (per participant):")
            for name, temp in config.temperatures.items():
                print(f"    {name}: {temp}")
        elif isinstance(config.temperatures, list):
            print(f"  Temperatures: {config.temperatures}")
        else:
            print(f"  Temperature: {config.temperatures}")
        
        # Print pie rule information (tournaments only)
        if config.script_type != "selfplay":
            print(f"  Pie rule: {config.pie_rule}")
        
        # Print random seed
        print(f"  Random seed: {config.random_seed}")
        
        # Print Gumbel configuration summary
        if include_gumbel:
            gumbel_summary = _generate_gumbel_summary(config)
            if gumbel_summary:
                print(f"  {gumbel_summary}")
        
        # Print timestamp
        if include_timestamp:
            timestamp = datetime.now()
            print(f"  Run time: {timestamp.strftime('%Y-%m-%d %H:%M')}")
        
        # Print git commit information
        if include_git:
            git_info = get_git_commit_info()
            print(f"  Git: {git_info['status']}")
        
        print()


class TournamentPrinter(ConfigurationPrinter):
    """Printer for tournament configurations."""
    
    def print_models(self, config: ScriptConfig) -> None:
        print(f"  Participants: {len(config.models)}")
        for i, model in enumerate(config.models):
            strategy_info = config.strategies[i] if i < len(config.strategies) else "unknown"
            print(f"    {strategy_info} ({os.path.basename(model)})")
    
    def print_games(self, config: ScriptConfig) -> None:
        print(f"  Number of games per pair: {config.num_games}")
    
    def print_strategy(self, config: ScriptConfig) -> None:
        # Extract strategy type from first strategy (assuming all use same type)
        strategy_type = "unknown"
        if config.strategies:
            first_strategy = config.strategies[0]
            if "policy" in first_strategy:
                strategy_type = "policy"
            elif "mcts" in first_strategy:
                strategy_type = "mcts"
            elif "fixed_tree" in first_strategy:
                strategy_type = "fixed_tree"
        print(f"  Strategy: {strategy_type}")
        if config.strategy_config:
            print(f"  Strategy config: {config.strategy_config}")


class DeterministicTournamentPrinter(TournamentPrinter):
    """Printer for deterministic tournament configurations."""
    
    def print_specific_config(self, config: ScriptConfig) -> None:
        if config.opening_length:
            print(f"  Opening length: {config.opening_length} moves")
        if config.batch_sizes:
            print(f"  Batch sizes: {config.batch_sizes}")
        if config.c_puct:
            print(f"  C_PUCT values: {config.c_puct}")
        print(f"  Dirichlet noise: alpha=0.3, eps=0.25 (MCTS default)")
        print(f"  Root noise: disabled (add_root_noise=False)")


class SelfplayPrinter(ConfigurationPrinter):
    """Printer for selfplay configurations."""
    
    def print_models(self, config: ScriptConfig) -> None:
        print(f"  Model: {os.path.basename(config.models[0]) if config.models else 'unknown'}")
    
    def print_games(self, config: ScriptConfig) -> None:
        print(f"  Games: {config.num_games}")
    
    def print_strategy(self, config: ScriptConfig) -> None:
        print(f"  Search method: MCTS ({config.mcts_sims} simulations)")
        if config.c_puct:
            print(f"  C_PUCT: {config.c_puct}")
    
    def print_specific_config(self, config: ScriptConfig) -> None:
        if config.temperature_end:
            print(f"  Temperature decay: {config.temperatures} -> {config.temperature_end}")
        if config.opening_strategy:
            print(f"  Opening strategy: {config.opening_strategy}")
        if config.batch_size:
            print(f"  Batch size: {config.batch_size}")
        if config.cache_size:
            print(f"  Cache size: {config.cache_size}")
        print(f"  Batched inference: {not config.no_batched_inference if hasattr(config, 'no_batched_inference') else 'unknown'}")


def print_script_configuration(
    config: ScriptConfig,
    include_git: bool = True,
    include_timestamp: bool = True,
    include_gumbel: bool = True
) -> None:
    """
    Print unified configuration information for all script types.
    
    Args:
        config: ScriptConfig object containing all configuration information
        include_git: Whether to include git commit information
        include_timestamp: Whether to include timestamp information
        include_gumbel: Whether to include Gumbel configuration summary
    """
    # Select appropriate printer strategy
    printer_map = {
        "tournament": TournamentPrinter(),
        "deterministic_tournament": DeterministicTournamentPrinter(),
        "selfplay": SelfplayPrinter()
    }
    
    printer = printer_map.get(config.script_type, ConfigurationPrinter())
    
    # Print configuration using strategy pattern
    printer.print_header(config)
    printer.print_models(config)
    printer.print_games(config)
    printer.print_strategy(config)
    printer.print_specific_config(config)
    printer.print_common_config(config, include_git, include_timestamp, include_gumbel)


def _generate_gumbel_summary(config: ScriptConfig) -> Optional[str]:
    """Generate Gumbel configuration summary for the given config."""
    if config.script_type == "selfplay":
        # For selfplay, use the parameter-based summary
        return generate_gumbel_summary_from_params(
            mcts_sims=config.mcts_sims,
            enable_gumbel=config.enable_gumbel,
            gumbel_sim_threshold=config.gumbel_sim_threshold,
            strategy_name="selfplay"
        )
    else:
        # For tournaments, we need to extract strategy configs
        # This is a simplified version - in practice, you'd pass the actual StrategyConfig objects
        if config.enable_gumbel and config.mcts_sims:
            summary_parts = []
            summary_parts.append(f"Gumbel enabled (sims≤{config.gumbel_sim_threshold})")
            if config.gumbel_c_visit:
                summary_parts.append(f"c_visit={config.gumbel_c_visit}")
            if config.gumbel_c_scale:
                summary_parts.append(f"c_scale={config.gumbel_c_scale}")
            if config.gumbel_candidate_log_base:
                summary_parts.append(f"log_base={config.gumbel_candidate_log_base}")
            if config.gumbel_candidate_log_offset:
                summary_parts.append(f"log_offset={config.gumbel_candidate_log_offset}")
            if config.gumbel_m_candidates:
                summary_parts.append(f"m_candidates={config.gumbel_m_candidates}")
            return f"Gumbel configuration: {', '.join(summary_parts)}"
        elif not config.enable_gumbel:
            return "Gumbel configuration: disabled"
    
    return None


def print_script_results(
    script_type: str,
    results: Any,
    config: ScriptConfig,
    output_files: Optional[Dict[str, str]] = None
) -> None:
    """
    Print unified results analysis for all script types.
    
    Args:
        script_type: Type of script ("tournament", "deterministic_tournament", "selfplay")
        results: Results object (TournamentResult, DeterministicTournamentResult, or selfplay results)
        config: ScriptConfig object
        output_files: Optional dict of output file paths
    """
    if script_type == "selfplay":
        _print_selfplay_results(results, config, output_files)
    elif script_type in ["tournament", "deterministic_tournament"]:
        _print_tournament_results(results, config, output_files)
    else:
        print(f"\n{script_type.title()} Complete!")
        if output_files:
            for file_type, file_path in output_files.items():
                print(f"  {file_type.title()}: {file_path}")


def _print_selfplay_results(results: Any, config: ScriptConfig, output_files: Optional[Dict[str, str]] = None) -> None:
    """Print selfplay-specific results."""
    print(f"\n=== Generation Complete ===")
    
    if hasattr(results, '__len__') and len(results) > 0:
        # Winner distribution
        winners = [game.get('winner', 'unknown') for game in results]
        red_wins = winners.count('r')
        blue_wins = winners.count('b')
        print(f"Winner distribution: Red {red_wins}, Blue {blue_wins}")
        print(f"Red win rate: {red_wins / len(results):.1%}")
    
    # Performance statistics (if available)
    if hasattr(results, 'get_performance_stats'):
        stats = results.get_performance_stats()
        if 'model' in stats:
            model_stats = stats['model']
            print(f"\n=== Model Performance ===")
            print(f"Total inferences: {model_stats.get('total_inferences', 0)}")
            print(f"Cache hit rate: {model_stats.get('cache', {}).get('hit_rate', 0):.1%}")
            print(f"Average batch size: {model_stats.get('avg_batch_size', 0):.1f}")
            print(f"Throughput: {model_stats.get('throughput', 0):.1f} boards/s")
    
    # Output files
    if output_files:
        print(f"\nSaved files:")
        for file_type, file_path in output_files.items():
            print(f"  {file_type.title()}: {file_path}")


def _print_tournament_results(results: Any, config: ScriptConfig, output_files: Optional[Dict[str, str]] = None) -> None:
    """Print tournament-specific results."""
    print(f"\n{config.script_type.replace('_', ' ').title()} Complete!")
    
    # Use existing tournament analysis if available
    if hasattr(results, 'win_rates') and hasattr(results, 'elo_ratings'):
        from hex_ai.utils.tournament_stats import print_comprehensive_tournament_analysis
        
        # Extract participant temperatures if available
        participant_temperatures = None
        if isinstance(config.temperatures, dict):
            participant_temperatures = config.temperatures
        
        print_comprehensive_tournament_analysis(results, participant_temperatures)
    
    # Print timing summary if available
    if hasattr(results, 'print_timing_summary'):
        results.print_timing_summary()
    
    # Output files
    if output_files:
        print(f"\nResults saved to:")
        for file_type, file_path in output_files.items():
            print(f"  {file_type.title()}: {file_path}")


def create_script_config_from_args(
    script_type: str,
    args: Any,
    models: List[str],
    strategies: List[str],
    strategy_config: Dict[str, Any],
    temperatures: Union[float, List[float], Dict[str, float]],
    **kwargs
) -> ScriptConfig:
    """
    Create a ScriptConfig from command line arguments and parsed data.
    
    Args:
        script_type: Type of script
        args: Parsed command line arguments
        models: List of model paths
        strategies: List of strategy names
        strategy_config: Strategy configuration dict
        temperatures: Temperature configuration
        **kwargs: Additional configuration parameters
    
    Returns:
        ScriptConfig object
    """
    return ScriptConfig(
        script_type=script_type,
        models=models,
        strategies=strategies,
        num_games=getattr(args, 'num_games', getattr(args, 'num_openings', 100)),
        strategy_config=strategy_config,
        temperatures=temperatures,
        random_seed=getattr(args, 'seed', 0),
        pie_rule=getattr(args, 'pie_rule', not getattr(args, 'no_pie_rule', False)),
        opening_length=getattr(args, 'opening_length', None),
        opening_strategy=getattr(args, 'opening_strategy', None),
        batch_size=getattr(args, 'batch_size', None),
        cache_size=getattr(args, 'cache_size', None),
        mcts_sims=getattr(args, 'mcts_sims', None),
        c_puct=getattr(args, 'c_puct', getattr(args, 'mcts_c_puct', None)),
        enable_gumbel=getattr(args, 'enable_gumbel', not getattr(args, 'disable_gumbel', False)),
        gumbel_sim_threshold=getattr(args, 'gumbel_sim_threshold', None),
        gumbel_c_visit=getattr(args, 'gumbel_c_visit', None),
        gumbel_c_scale=getattr(args, 'gumbel_c_scale', None),
        gumbel_candidate_log_base=getattr(args, 'gumbel_candidate_log_base', None),
        gumbel_candidate_log_offset=getattr(args, 'gumbel_candidate_log_offset', None),
        gumbel_m_candidates=getattr(args, 'gumbel_m_candidates', None),
        search_widths=getattr(args, 'search_widths', None),
        output_dir=getattr(args, 'output_dir', None),
        checkpoint_dirs=getattr(args, 'checkpoint_dirs', None),
        default_checkpoint_dir=getattr(args, 'default_checkpoint_dir', None),
        **kwargs
    )
