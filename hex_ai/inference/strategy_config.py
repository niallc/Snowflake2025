"""
Shared utilities for strategy configuration parsing.

This module consolidates the duplicate strategy parsing code that was previously
scattered across multiple tournament scripts.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class StrategyConfig:
    """Configuration for a single strategy."""
    
    name: str
    strategy_type: str
    config: Dict[str, Any]
    
    def __str__(self) -> str:
        return f"{self.name}({self.strategy_type})"


def parse_strategy_configs(strategies: List[str], mcts_sims: Optional[List[int]] = None, 
                          search_widths: Optional[List[str]] = None, batch_sizes: Optional[List[int]] = None) -> List[StrategyConfig]:
    """
    Parse strategy configurations from command line arguments.
    
    This function handles the parsing of strategy names and optional parameter overrides.
    It's used by multiple tournament scripts to avoid code duplication.
    
    Args:
        strategies: List of strategy names (e.g., ["policy", "mcts_100", "fixed_tree_13_8"])
        mcts_sims: Optional list of MCTS simulation counts to override strategy names
        search_widths: Optional list of search width strings (e.g., ["13,8", "20,10"])
        batch_sizes: Optional list of batch sizes for MCTS strategies (e.g., [64, 128, 256])
    
    Returns:
        List of StrategyConfig objects
    
    Raises:
        ValueError: If strategy names are invalid or parameter counts don't match
    """
    configs = []
    
    for strategy_name in strategies:
        # Parse strategy name to determine type and parameters
        if strategy_name == "policy":
            configs.append(StrategyConfig("policy", "policy", {}))
        
        elif strategy_name.startswith("mcts_"):
            # Extract simulation count from name (e.g., "mcts_100" -> 100)
            try:
                sims = int(strategy_name.split("_")[1])
                configs.append(StrategyConfig(
                    strategy_name, "mcts", 
                    {"mcts_sims": sims, "mcts_c_puct": 1.5}
                ))
            except (IndexError, ValueError):
                raise ValueError(f"Invalid MCTS strategy name: {strategy_name}. Expected format: mcts_<sims>")
        
        elif strategy_name.startswith("fixed_tree_"):
            # Extract widths from name (e.g., "fixed_tree_13_8" -> [13, 8])
            try:
                parts = strategy_name.split("_")[2:]
                widths = [int(w) for w in parts]
                configs.append(StrategyConfig(
                    strategy_name, "fixed_tree", 
                    {"search_widths": widths}
                ))
            except (IndexError, ValueError):
                raise ValueError(f"Invalid fixed_tree strategy name: {strategy_name}. Expected format: fixed_tree_<width1>_<width2>_...")
        
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # Override with command line parameters if provided
    if mcts_sims:
        mcts_configs = [c for c in configs if c.strategy_type == "mcts"]
        if len(mcts_sims) != len(mcts_configs):
            raise ValueError(f"Number of MCTS simulation counts ({len(mcts_sims)}) must match number of MCTS strategies ({len(mcts_configs)})")
        
        mcts_idx = 0
        for config in configs:
            if config.strategy_type == "mcts":
                config.config["mcts_sims"] = mcts_sims[mcts_idx]
                mcts_idx += 1
    
    if search_widths:
        tree_configs = [c for c in configs if c.strategy_type == "fixed_tree"]
        if len(search_widths) != len(tree_configs):
            raise ValueError(f"Number of search width sets ({len(search_widths)}) must match number of fixed_tree strategies ({len(tree_configs)})")
        
        tree_idx = 0
        for config in configs:
            if config.strategy_type == "fixed_tree":
                # Parse widths string (e.g., "13,8" -> [13, 8])
                widths = [int(w.strip()) for w in search_widths[tree_idx].split(",")]
                config.config["search_widths"] = widths
                tree_idx += 1
    
    if batch_sizes:
        mcts_configs = [c for c in configs if c.strategy_type == "mcts"]
        if len(batch_sizes) != len(mcts_configs):
            raise ValueError(f"Number of batch sizes ({len(batch_sizes)}) must match number of MCTS strategies ({len(mcts_configs)})")
        
        batch_idx = 0
        for config in configs:
            if config.strategy_type == "mcts":
                config.config["batch_size"] = batch_sizes[batch_idx]
                # Update strategy name to include batch size for unique identification
                config.name = f"{config.name}_b{batch_sizes[batch_idx]}"
                batch_idx += 1
    
    return configs


def validate_strategy_configs(configs: List[StrategyConfig]) -> None:
    """
    Validate strategy configurations.
    
    Args:
        configs: List of strategy configurations to validate
    
    Raises:
        ValueError: If configurations are invalid
    """
    for config in configs:
        if config.strategy_type == "mcts":
            if "mcts_sims" not in config.config:
                raise ValueError(f"MCTS strategy {config.name} missing mcts_sims parameter")
            if config.config["mcts_sims"] <= 0:
                raise ValueError(f"MCTS strategy {config.name} has invalid mcts_sims: {config.config['mcts_sims']}")
        
        elif config.strategy_type == "fixed_tree":
            if "search_widths" not in config.config:
                raise ValueError(f"Fixed tree strategy {config.name} missing search_widths parameter")
            if not config.config["search_widths"]:
                raise ValueError(f"Fixed tree strategy {config.name} has empty search_widths")
            for width in config.config["search_widths"]:
                if width <= 0:
                    raise ValueError(f"Fixed tree strategy {config.name} has invalid search width: {width}")


def get_strategy_summary(configs: List[StrategyConfig]) -> str:
    """
    Get a human-readable summary of strategy configurations.
    
    Args:
        configs: List of strategy configurations
    
    Returns:
        String summary of the configurations
    """
    summaries = []
    for config in configs:
        if config.strategy_type == "policy":
            summaries.append("policy")
        elif config.strategy_type == "mcts":
            sims = config.config.get("mcts_sims", "unknown")
            summaries.append(f"mcts_{sims}")
        elif config.strategy_type == "fixed_tree":
            widths = config.config.get("search_widths", [])
            width_str = "_".join(str(w) for w in widths)
            summaries.append(f"fixed_tree_{width_str}")
        else:
            summaries.append(str(config))
    
    return ", ".join(summaries)
