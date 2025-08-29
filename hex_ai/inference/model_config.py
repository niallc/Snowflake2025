"""
Central configuration for model checkpoint paths.

This module provides a single source of truth for model checkpoint paths
used throughout the project. This makes it easy to update to new best models
and ensures consistency across all scripts and modules.
"""

import os
from typing import List, Optional

# Base directory for all checkpoints
CHECKPOINTS_BASE_DIR = "checkpoints"

# Current best model configuration
CURRENT_BEST_MODEL_DIR = "aug28th_extraValueLayer/loss_weight_sweep_exp0__99914b_20250828_183718"
CURRENT_BEST_MODEL_FILE = "epoch2_mini100.pt.gz"
CURRENT_BEST_MODEL_PATH = os.path.join(CHECKPOINTS_BASE_DIR, CURRENT_BEST_MODEL_DIR, CURRENT_BEST_MODEL_FILE)

# Previous best model (kept for comparison/testing)
PREVIOUS_BEST_MODEL_DIR = "aug28th_extraValueLayer/loss_weight_sweep_exp0__99914b_20250828_183718"
PREVIOUS_BEST_MODEL_FILE = "epoch1_mini201.pt.gz"
PREVIOUS_BEST_MODEL_PATH = os.path.join(CHECKPOINTS_BASE_DIR, PREVIOUS_BEST_MODEL_DIR, PREVIOUS_BEST_MODEL_FILE)

# Legacy models (for historical comparison)
LEGACY_MODELS = {}

def get_model_path(model_name: str = "current_best") -> str:
    """
    Get the full path to a model checkpoint.
    
    Args:
        model_name: Name of the model to get path for. Options:
            - "current_best": Latest best model
            - "previous_best": Previous best model
            - Any key from LEGACY_MODELS
    
    Returns:
        Full path to the model checkpoint file
    """
    if model_name == "current_best":
        return CURRENT_BEST_MODEL_PATH
    elif model_name == "previous_best":
        return PREVIOUS_BEST_MODEL_PATH
    elif model_name in LEGACY_MODELS:
        model_info = LEGACY_MODELS[model_name]
        return os.path.join(CHECKPOINTS_BASE_DIR, model_info["dir"], model_info["file"])
    else:
        raise ValueError(f"Unknown model name: {model_name}. Available: current_best, previous_best, {list(LEGACY_MODELS.keys())}")

def get_model_dir(model_name: str = "current_best") -> str:
    """
    Get the directory containing a model checkpoint.
    
    Args:
        model_name: Name of the model to get directory for
    
    Returns:
        Directory path containing the model checkpoint
    """
    if model_name == "current_best":
        return os.path.join(CHECKPOINTS_BASE_DIR, CURRENT_BEST_MODEL_DIR)
    elif model_name == "previous_best":
        return os.path.join(CHECKPOINTS_BASE_DIR, PREVIOUS_BEST_MODEL_DIR)
    elif model_name in LEGACY_MODELS:
        model_info = LEGACY_MODELS[model_name]
        return os.path.join(CHECKPOINTS_BASE_DIR, model_info["dir"])
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def get_available_models() -> List[str]:
    """
    Get list of available model names.
    
    Returns:
        List of model names that can be used with get_model_path()
    """
    return ["current_best", "previous_best"] + list(LEGACY_MODELS.keys())

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
        "model1": CURRENT_BEST_MODEL_PATH,
        "model2": PREVIOUS_BEST_MODEL_PATH
    }

# Convenience variables for backward compatibility
DEFAULT_MODEL_PATH = CURRENT_BEST_MODEL_PATH
DEFAULT_MODEL_DIR = os.path.join(CHECKPOINTS_BASE_DIR, CURRENT_BEST_MODEL_DIR) 