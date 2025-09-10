"""
Central configuration for model checkpoint paths.

This module provides a single source of truth for model checkpoint paths
used throughout the project. This makes it easy to update to new best models
and ensures consistency across all scripts and modules.
"""

import os
from typing import List, Optional, Dict, Any
from pathlib import Path

# Base directory for all checkpoints
CHECKPOINTS_BASE_DIR = "checkpoints"

# Current best model configuration
# Temporary honourable mention models: sep6th.../epoch4_mini135.pt.gz, sep6th.../epoch4_mini40.pt.gz

# # Temporary state for retraining:
CURRENT_BEST_MODEL_DIR = "hyperparameter_tuning/pipeline_20250910_080404/pipeline_sweep_exp0__99914b_20250910_080404/"
CURRENT_BEST_MODEL_FILE = "epoch6_mini44.pt.gz"
# CURRENT_BEST_MODEL_DIR = "hyperparameter_tuning/pipeline_20250909_195315/pipeline_sweep_exp0__99914b_20250909_195315/"
# CURRENT_BEST_MODEL_FILE = "epoch5_mini39.pt.gz"
# CURRENT_BEST_MODEL_DIR = "sep6th_extraValueLayer/pipeline_20250906_82558/pipeline_sweep_exp0__99914b_20250906_182558"
# CURRENT_BEST_MODEL_FILE = "epoch4_mini123.pt.gz"
CURRENT_BEST_MODEL_PATH = os.path.join(CHECKPOINTS_BASE_DIR, CURRENT_BEST_MODEL_DIR, CURRENT_BEST_MODEL_FILE)

# Previous best model (kept for comparison/testing)
PREVIOUS_BEST_MODEL_DIR = "aug28th_extraValueLayer/loss_weight_sweep_exp0__99914b_20250828_183718"
PREVIOUS_BEST_MODEL_FILE = "epoch2_mini201.pt.gz"
PREVIOUS_BEST_MODEL_PATH = os.path.join(CHECKPOINTS_BASE_DIR, PREVIOUS_BEST_MODEL_DIR, PREVIOUS_BEST_MODEL_FILE)

# Legacy models (for historical comparison)
LEGACY_MODELS = {}

# Central registry of all model IDs and their paths
# This ensures all parts of the code use the same model IDs and paths
MODEL_REGISTRY = {
    "model1": CURRENT_BEST_MODEL_PATH,
    "model2": PREVIOUS_BEST_MODEL_PATH,
    "current_best": CURRENT_BEST_MODEL_PATH,
    "previous_best": PREVIOUS_BEST_MODEL_PATH,
}

def get_model_path(model_name: str = "current_best") -> str:
    """
    Get the full path to a model checkpoint.
    
    Args:
        model_name: Name of the model to get path for. Options:
            - "current_best": Latest best model
            - "previous_best": Previous best model
            - "model1": Alias for current_best
            - "model2": Alias for previous_best
            - Any key from LEGACY_MODELS
    
    Returns:
        Full path to the model checkpoint file
    """
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]
    elif model_name in LEGACY_MODELS:
        model_info = LEGACY_MODELS[model_name]
        return os.path.join(CHECKPOINTS_BASE_DIR, model_info["dir"], model_info["file"])
    else:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(MODEL_REGISTRY.keys())}, {list(LEGACY_MODELS.keys())}")

def get_model_dir(model_name: str = "current_best") -> str:
    """
    Get the directory containing a model checkpoint.
    
    Args:
        model_name: Name of the model to get directory for
    
    Returns:
        Directory path containing the model checkpoint
    """
    model_path = get_model_path(model_name)
    return os.path.dirname(model_path)

def get_available_models() -> List[str]:
    """
    Get list of available model names.
    
    Returns:
        List of model names that can be used with get_model_path()
    """
    return list(MODEL_REGISTRY.keys()) + list(LEGACY_MODELS.keys())

def validate_model_path(model_path: str) -> bool:
    """
    Check if a model checkpoint file exists.
    
    Args:
        model_path: Path to model checkpoint file
    
    Returns:
        True if file exists, False otherwise
    """
    return os.path.isfile(model_path)

def get_default_model_paths() -> dict:
    """
    Get default model paths for web app and other components.
    
    Returns:
        Dictionary with model1 and model2 paths
    """
    return {
        "model1": get_model_path("model1"),
        "model2": get_model_path("model2")
    }

def get_model_info(model_id: str) -> Dict[str, Any]:
    """
    Get comprehensive information about a model.
    
    Args:
        model_id: Model identifier (e.g., "model1", "current_best")
    
    Returns:
        Dictionary with model information including path, filename, etc.
    """
    model_path = get_model_path(model_id)
    filename = os.path.basename(model_path)
    directory = os.path.dirname(model_path)
    
    return {
        "id": model_id,
        "path": model_path,
        "filename": filename,
        "directory": directory,
        "exists": validate_model_path(model_path),
        "relative_path": os.path.relpath(model_path, CHECKPOINTS_BASE_DIR) if model_path.startswith(CHECKPOINTS_BASE_DIR) else model_path
    }

def get_all_model_info() -> List[Dict[str, Any]]:
    """
    Get information about all available models.
    
    Returns:
        List of model info dictionaries
    """
    return [get_model_info(model_id) for model_id in get_available_models()]

def register_model(model_id: str, model_path: str) -> None:
    """
    Register a new model in the central registry.
    
    Args:
        model_id: Unique identifier for the model
        model_path: Path to the model checkpoint file
    """
    if model_id in MODEL_REGISTRY:
        raise ValueError(f"Model ID '{model_id}' already exists in registry")
    
    MODEL_REGISTRY[model_id] = model_path

def is_valid_model_id(model_id: str) -> bool:
    """
    Check if a model ID is valid (exists in registry).
    
    Args:
        model_id: Model identifier to check
    
    Returns:
        True if model_id is valid, False otherwise
    """
    return model_id in MODEL_REGISTRY or model_id in LEGACY_MODELS

def get_normalized_path(model_path: str) -> str:
    """
    Get a normalized (absolute) path for consistent caching.
    
    Args:
        model_path: Path to normalize
    
    Returns:
        Normalized absolute path
    """
    return os.path.abspath(os.path.normpath(model_path))

# Convenience variables for backward compatibility
DEFAULT_MODEL_PATH = CURRENT_BEST_MODEL_PATH
DEFAULT_MODEL_DIR = os.path.join(CHECKPOINTS_BASE_DIR, CURRENT_BEST_MODEL_DIR) 