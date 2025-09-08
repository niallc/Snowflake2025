"""
Shared utilities for handling model paths and configurations in tournaments.

This module provides utilities to resolve model paths from command line arguments,
validate model configurations, and generate unique labels for tournament participants.
These utilities are shared between different tournament types to maintain DRY principles.

This module consolidates model-specific logic that was previously duplicated between
tournament_utils.py and model_tournament_utils.py.
"""

import os
import sys
from typing import List, Dict, Tuple, Optional

from hex_ai.inference.model_config import get_model_path, get_model_dir, validate_model_path


def resolve_model_paths(
    models_arg: Optional[str],
    model_dirs_arg: Optional[str], 
    default_model_dir: str,
    default_models: List[str]
) -> List[str]:
    """
    Resolve model paths from command line arguments.
    
    This function handles the logic for building full model paths from:
    - Model names (resolved via model config)
    - Model directories (optional)
    - Default configurations
    
    Args:
        models_arg: Comma-separated model names from --models argument
        model_dirs_arg: Comma-separated model directories from --model-dirs argument
        default_model_dir: Default directory to use if no directories specified
        default_models: Default model names to use if no models specified
        
    Returns:
        List of full model checkpoint paths
        
    Raises:
        SystemExit: If argument configuration is invalid
    """
    if models_arg:
        # User provided specific model names
        model_names = [name.strip() for name in models_arg.split(',')]
        
        # Build full paths for model directories
        if model_dirs_arg:
            model_dirs = [dir_name.strip() for dir_name in model_dirs_arg.split(',')]
        else:
            # Use the default model directory
            model_dirs = [default_model_dir]

        # Build model paths
        if len(model_dirs) == 1:
            # Single directory for all models
            base_dir = model_dirs[0]
            model_paths = []
            for model_name in model_names:
                try:
                    # Try to resolve as a model config name first
                    model_path = get_model_path(model_name)
                    model_paths.append(model_path)
                except ValueError:
                    # If not found in model config, treat as filename in the directory
                    model_path = os.path.join(base_dir, model_name)
                    model_paths.append(model_path)
        elif len(model_dirs) == len(model_names):
            # One directory per model
            model_paths = []
            for dir_name, model_name in zip(model_dirs, model_names):
                try:
                    # Try to resolve as a model config name first
                    model_path = get_model_path(model_name)
                    model_paths.append(model_path)
                except ValueError:
                    # If not found in model config, treat as filename in the directory
                    model_path = os.path.join(dir_name, model_name)
                    model_paths.append(model_path)
        else:
            print(f"ERROR: Number of model directories ({len(model_dirs)}) must be 1 or match the number of models ({len(model_names)})")
            print(f"  Provided model names: {model_names}")
            print(f"  Provided model directories: {model_dirs}")
            sys.exit(1)
    else:
        # Use defaults from the default model directory
        model_paths = []
        for model_name in default_models:
            try:
                model_path = get_model_path(model_name)
                model_paths.append(model_path)
            except ValueError:
                # If not found in model config, treat as filename in the default directory
                model_path = os.path.join(default_model_dir, model_name)
                model_paths.append(model_path)

    return model_paths


def validate_participant_paths(model_paths: List[str], participant_type: str = "model") -> None:
    """
    Validate that model paths exist and there are enough for a tournament.
    
    Args:
        model_paths: List of model checkpoint file paths
        participant_type: Type of participant for error messages (e.g., "model", "checkpoint")
        
    Raises:
        SystemExit: If validation fails
    """
    # Validate that we have at least 2 models for a meaningful tournament
    if len(model_paths) < 2:
        print(f"ERROR: Need at least 2 unique {participant_type}s for a tournament.")
        print(f"  Provided {participant_type}s: {[os.path.basename(p) for p in model_paths]}")
        sys.exit(1)

    # Check that all model paths exist before proceeding
    missing_paths = [p for p in model_paths if not os.path.isfile(p)]
    if missing_paths:
        print(f"\nERROR: The following {participant_type} files do not exist:")
        for p in missing_paths:
            print(f"  {p}")
        print(f"\nPlease check that the {participant_type} files exist and the paths are correct.")
        sys.exit(1)


def validate_model_paths(model_paths: List[str]) -> None:
    """
    Validate that model paths exist and there are enough for a tournament.
    
    This is a convenience wrapper around validate_participant_paths.
    Maintained for backward compatibility.
    
    Args:
        model_paths: List of model checkpoint file paths
        
    Raises:
        SystemExit: If validation fails
    """
    validate_participant_paths(model_paths, "model")


def generate_participant_labels(model_paths: List[str], label_prefix: str = "Model") -> Tuple[List[str], Dict[str, str]]:
    """
    Generate unique participant labels for model paths, handling duplicates.
    
    This function creates unique labels for each model to handle cases where
    the same model file might be used multiple times in a tournament.
    
    Args:
        model_paths: List of model checkpoint file paths
        label_prefix: Prefix for duplicate labels (e.g., "Model", "Player")
        
    Returns:
        Tuple of (participant_labels, label_to_model_mapping)
        - participant_labels: List of unique labels for each participant
        - label_to_model_mapping: Dict mapping labels to actual model paths
    """
    participant_labels = []
    label_to_model = {}
    model_usage = {}
    
    for model_path in model_paths:
        if model_path in model_usage:
            # This is a duplicate - create a unique participant label
            model_usage[model_path] += 1
            participant_label = f"{label_prefix}{model_usage[model_path]}_{os.path.basename(model_path)}"
        else:
            # First occurrence - use just the filename for consistency
            model_usage[model_path] = 1
            participant_label = os.path.basename(model_path)
        
        participant_labels.append(participant_label)
        label_to_model[participant_label] = model_path
    
    return participant_labels, label_to_model


def generate_model_labels(model_paths: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Generate unique model labels for model paths, handling duplicates.
    
    This is a convenience wrapper around generate_participant_labels with "Model" prefix.
    Maintained for backward compatibility.
    
    Args:
        model_paths: List of model checkpoint file paths
        
    Returns:
        Tuple of (model_labels, label_to_model_mapping)
    """
    return generate_participant_labels(model_paths, "Model")


def print_duplicate_participant_info(participant_labels: List[str], model_paths: List[str], participant_type: str = "model") -> None:
    """
    Print information about duplicate participants and participant labels.
    
    Args:
        participant_labels: List of participant labels
        model_paths: List of model paths
        participant_type: Type of participant for display (e.g., "model", "checkpoint")
    """
    unique_paths = list(dict.fromkeys(model_paths))
    if len(unique_paths) != len(model_paths):
        print(f"INFO: Duplicate {participant_type}s detected. Using unique {participant_type} labels:")
        for label, path in zip(participant_labels, model_paths):
            if label != path:
                print(f"  {label} -> {os.path.basename(path)}")
            else:
                print(f"  {os.path.basename(path)}")
        print()


def print_duplicate_model_info(model_labels: List[str], model_paths: List[str]) -> None:
    """
    Print information about duplicate models and model labels.
    
    This is a convenience wrapper around print_duplicate_participant_info.
    Maintained for backward compatibility.
    
    Args:
        model_labels: List of model labels
        model_paths: List of model paths
    """
    print_duplicate_participant_info(model_labels, model_paths, "model")


def print_model_tournament_configuration(
    model_labels: List[str],
    model_paths: List[str],
    num_games: int,
    strategy: str,
    strategy_config: Dict,
    temperature: float,
    random_seed: int,
    model_dirs: Optional[List[str]] = None,
    default_model_dir: Optional[str] = None
) -> None:
    """
    Print model tournament configuration in a user-friendly format.
    
    Args:
        model_labels: List of model labels
        model_paths: List of model paths
        num_games: Number of games per pair
        strategy: Tournament strategy
        strategy_config: Strategy configuration
        temperature: Temperature for move selection
        random_seed: Random seed
        model_dirs: Optional model directories
        default_model_dir: Default model directory
    """
    print(f"Model Tournament Configuration:")
    print(f"  Participants: {len(model_labels)}")
    for label, path in zip(model_labels, model_paths):
        if label != path:
            print(f"    {label} ({os.path.basename(path)})")
        else:
            print(f"    {os.path.basename(path)}")
    
    if model_dirs:
        print(f"  Model directories: {model_dirs}")
    elif default_model_dir:
        print(f"  Model directory: {default_model_dir}")
    
    print(f"  Number of games per pair: {num_games}")
    print(f"  Strategy: {strategy}")
    print(f"  Strategy config: {strategy_config}")
    print(f"  Temperature: {temperature}")
    print(f"  Random seed: {random_seed}")


def get_model_label_for_path(
    model_path: str, 
    config_model_paths: List[str], 
    config_model_labels: List[str]
) -> str:
    """
    Get the model label for a specific model path.
    
    Args:
        model_path: The model path to look up
        config_model_paths: List of model paths from config
        config_model_labels: List of model labels from config
        
    Returns:
        The model label for the given model path
        
    Raises:
        ValueError: If model path is not found in config
    """
    try:
        index = config_model_paths.index(model_path)
        return config_model_labels[index]
    except ValueError:
        raise ValueError(f"Model path {model_path} not found in tournament configuration")
