"""
Central configuration for model checkpoint paths.

This module provides a single source of truth for model checkpoint paths
used throughout the project. Models are organized by generation numbers,
where each generation represents a directory of models from a training run.
The current best model is derived from the highest generation number.

To add a new generation:
1. Add a new entry to MODEL_GENERATIONS with the next generation number
2. Format: {generation_number: {"dir": "path/to/dir", "models": ["file1.pt.gz", "file2.pt.gz"]}}
3. The first model in the list is considered the primary model for that generation
"""

import os
import re
from typing import List, Optional, Dict, Any, Tuple, TYPE_CHECKING
from pathlib import Path

from hex_ai.inference.knockout_tournament import TournamentParticipant

# Base directory for all checkpoints
CHECKPOINTS_BASE_DIR = "checkpoints"

# Model generations: Each generation represents a directory of models from a training run
# Keys are generation numbers (integers starting from 1)
# Values contain directory path and list of model files of interest
# The current best model is derived from the highest generation number
MODEL_GENERATIONS: Dict[int, Dict[str, Any]] = {
    # 1: {
    #     "dir": "hyperparameter_tuning/loss_weight_sweep_exp0__99914b_20250917_192629",
    #     "models": ["epoch6_mini90.pt.gz", "epoch7_mini105.pt.gz"]
    # },
    2: {
        "dir": "hyperparameter_tuning/pipeline_20250921_095250/pipeline_sweep_exp0__99914b_20250921_095250",
        "models": ["epoch9_mini12.pt.gz", "epoch8_mini1.pt.gz"]
    },
    3: {
        "dir": "hyperparameter_tuning/pipeline_20250922_071957/pipeline_sweep_exp0__99914b_20250922_072446",
        "models": ["epoch11_mini15.pt.gz"]
    },
    4: {
        "dir": "hyperparameter_tuning/pipeline_20250923_072259/pipeline_sweep_exp0__99914b_20250923_073100",
        "models": ["epoch13_mini5.pt.gz"]
    },
    5: {
        "dir": "hyperparameter_tuning/pipeline_20250924_151002/pipeline_sweep_exp0__99914b_20250924_151002",
        "models": ["epoch14_mini34.pt.gz"]
    },
    6: {
        "dir": "hyperparameter_tuning/pipeline_20250926_003151/pipeline_sweep_0",
        "models": ["epoch18_mini30.pt.gz"]
    },
    7: {
        "dir": "hyperparameter_tuning/pipeline_20250929_142959/pipeline_sweep_0",
        "models": ["epoch15_mini24.pt.gz", "epoch13_mini19.pt.gz", "epoch14_mini9.pt.gz", "epoch14_mini18.pt.gz"]
    },
    8: {
        "dir": "hyperparameter_tuning/pipeline_20251003_205950/",
        "models": ["epoch19_mini6.pt.gz", "epoch18_mini23.pt.gz", "epoch19_mini7.pt.gz", "epoch19_mini5.pt.gz", "epoch17_mini7.pt.gz", "epoch16_mini7.pt.gz", "epoch19_mini19.pt.gz"]
    },
    9: {
        "dir": "hyperparameter_tuning/pipeline_20251005_123332/",
        "models": ["epoch23_mini35.pt.gz", "epoch22_mini15.pt.gz", "epoch22_mini19.pt.gz", "epoch21_mini11.pt.gz"]
    },
    10: {
        "dir": "hyperparameter_tuning/pipeline_20251009_082022/",
        "models": ["epoch30_mini24.pt.gz", "epoch31_mini32.pt.gz"]
    },
    11: {
        "dir": "hyperparameter_tuning/pipeline_20251011_102118/",
        "models": ["epoch37_mini47.pt.gz"]
    },
    12: {
        "dir": "hyperparameter_tuning/pipeline_20251013_181038/",
        "models": ["epoch43_mini34.pt.gz"]
    },
    13: {
        "dir": "hyperparameter_tuning/pipeline_20251016_184020/",
        "models": ["epoch48_mini11.pt.gz"]
    },
    14: {
        "dir": "hyperparameter_tuning/pipeline_20251018_152819/",
        "models": ["epoch49_mini57.pt.gz"]
    },
    15: {
        "dir": "hyperparameter_tuning/pipeline_20251020_171353/",
        "models": ["epoch51_mini30.pt.gz", "epoch50_mini52.pt.gz"]
    },
    16: {
        "dir": "hyperparameter_tuning/pipeline_20251021_104711/",
        "models": ["epoch55_mini13.pt.gz"]
    },
    17: {
        "dir": "hyperparameter_tuning/pipeline_20251023_124549/",
        "models": ["epoch60_mini37.pt.gz", "epoch60_mini44.pt.gz", "epoch59_mini15.pt.gz"]
    },
    18: {
        "dir": "hyperparameter_tuning/pipeline_20251025_210806/",
        "models": ["epoch64_mini35.pt.gz"]
    },
    19: {
        "dir": "hyperparameter_tuning/pipeline_20251027_065000/",
        "models": ["epoch64_mini20.pt.gz", "epoch65_mini33.pt.gz"]
    },
    20: {
        "dir": "hyperparameter_tuning/pipeline_20251029_212557/",
        "models": ["epoch68_mini15.pt.gz", "epoch70_mini36.pt.gz", "epoch69_mini10.pt.gz",  "epoch67_mini4.pt.gz"]
    },
    21: {
        "dir": "hyperparameter_tuning/pipeline_20251102_235603/",
        "models": ["epoch72_mini7.pt.gz", "epoch72_mini26.pt.gz", "epoch70_mini59.pt.gz"]
    },
    22: {
        "dir": "hyperparameter_tuning/pipeline_20251105_220508/",
        "models": ["epoch74_mini13.pt.gz"]
    },
    23: {
        "dir": "hyperparameter_tuning/pipeline_20251106_124025/",
        "models": ["epoch76_mini33.pt.gz"]
    },
    24: {
        "dir": "hyperparameter_tuning/pipeline_20251108_134726/",
        "models": ["epoch76_mini41.pt.gz"]
    },
    25: {
        "dir": "hyperparameter_tuning/pipeline_20251110_100611/",
        "models": ["epoch79_mini26.pt.gz", "epoch80_mini1", "epoch77_mini13.pt.gz"]
    },
    26: {
        "dir": "hyperparameter_tuning/pipeline_20251112_205851/",
        "models": ["epoch82_mini11.pt.gz", "epoch80_mini25.pt.gz"]
    },
    27: {
        "dir": "hyperparameter_tuning/pipeline_20251114_230638/",
        "models": ["epoch86_mini32.pt.gz", "epoch86_mini30.pt.gz"]
    },
    28: {
        "dir": "hyperparameter_tuning/pipeline_20251117_195834/",
        "models": ["epoch89_mini20.pt.gz", "epoch88_mini28.pt.gz"]
    },
    29: {
        "dir": "hyperparameter_tuning/pipeline_20251119_175908/",
        "models": ["epoch92_mini37.pt.gz"]
    },
    30: {
        "dir": "hyperparameter_tuning/pipeline_20251124_130617/",
        "models": ["epoch95_mini51.pt.gz", "epoch95_mini40.pt.gz"]
    },
    31: {
        "dir": "hyperparameter_tuning/pipeline_20251126_213922/",
        "models": ["epoch102_mini24.pt.gz"]
    },
    32: {
        "dir": "hyperparameter_tuning/pipeline_20251130_122123/",
        "models": ["epoch104_mini65.pt.gz", "epoch104_mini52.pt.gz", "epoch103_mini43.pt.gz"]
    },
    33: {
        "dir": "hyperparameter_tuning/pipeline_20251203_110951/",
        "models": ["epoch106_mini57.pt.gz", "epoch106_mini61.pt.gz"]
    },
    34: {
        "dir": "hyperparameter_tuning/pipeline_20251206_004859/",
        "models": ["epoch109_mini14.pt.gz", "epoch108_mini22.pt.gz"]
    },
    35: {
        "dir": "hyperparameter_tuning/pipeline_20251208_090537/",
        "models": ["epoch111_mini47.pt.gz"]
    },
    36: {
        "dir": "hyperparameter_tuning/pipeline_20251210_124135/",
        "models": ["epoch114_mini5.pt.gz", "epoch112_mini45.pt.gz"]
    },
    37: {
        "dir": "hyperparameter_tuning/pipeline_20251212_145108/",
        "models": ["epoch117_mini18.pt.gz"]
    }

}

def _get_current_generation() -> int:
    """Get the highest generation number (current best model generation)."""
    if not MODEL_GENERATIONS:
        raise ValueError("MODEL_GENERATIONS is empty")
    return max(MODEL_GENERATIONS.keys())


def _get_current_best_model() -> Tuple[str, str]:
    """Get the directory and primary model file from the highest generation."""
    gen = _get_current_generation()
    gen_data = MODEL_GENERATIONS[gen]
    return gen_data["dir"], gen_data["models"][0]


def _get_previous_best_model() -> Optional[Tuple[str, str]]:
    """Get the directory and primary model file from the second highest generation."""
    if len(MODEL_GENERATIONS) < 2:
        return None
    sorted_gens = sorted(MODEL_GENERATIONS.keys())
    gen = sorted_gens[-2]  # Second highest
    gen_data = MODEL_GENERATIONS[gen]
    return gen_data["dir"], gen_data["models"][0]


# Derive current best model from highest generation
_current_dir, _current_file = _get_current_best_model()
CURRENT_BEST_MODEL_DIR = _current_dir
CURRENT_BEST_MODEL_FILE = _current_file

CURRENT_BEST_MODEL_PATH = os.path.join(CHECKPOINTS_BASE_DIR, CURRENT_BEST_MODEL_DIR, CURRENT_BEST_MODEL_FILE)

# Previous best model (kept for comparison/testing)
# Derived from the second highest generation
_previous_best = _get_previous_best_model()
if _previous_best is not None:
    PREVIOUS_BEST_MODEL_DIR, PREVIOUS_BEST_MODEL_FILE = _previous_best
    PREVIOUS_BEST_MODEL_PATH = os.path.join(CHECKPOINTS_BASE_DIR, PREVIOUS_BEST_MODEL_DIR, PREVIOUS_BEST_MODEL_FILE)
else:
    # Fallback if there's only one generation
    PREVIOUS_BEST_MODEL_DIR = CURRENT_BEST_MODEL_DIR
    PREVIOUS_BEST_MODEL_FILE = CURRENT_BEST_MODEL_FILE
    PREVIOUS_BEST_MODEL_PATH = CURRENT_BEST_MODEL_PATH

# Fallback model configuration (used when current best model is unavailable)
FALLBACK_MODEL_DIR = "hyperparameter_tuning/pipeline_20251023_124549/"
FALLBACK_MODEL_FILE = "epoch60_mini37.pt.gz"
FALLBACK_MODEL_PATH = os.path.join(CHECKPOINTS_BASE_DIR, FALLBACK_MODEL_DIR, FALLBACK_MODEL_FILE)

# Simple model configuration (used to generate lower ELO play, rather than best-possible play)
SIMPLE_MODEL_DIR = "hyperparameter_tuning/pipeline_20250921_095250/pipeline_sweep_exp0__99914b_20250921_095250"
SIMPLE_MODEL_FILE = "epoch8_mini1.pt.gz"
SIMPLE_MODEL_PATH = os.path.join(CHECKPOINTS_BASE_DIR, SIMPLE_MODEL_DIR, SIMPLE_MODEL_FILE)
# Legacy models (for historical comparison)
LEGACY_MODELS = {}

# Central registry of all model IDs and their paths
# This ensures all parts of the code use the same model IDs and paths
MODEL_REGISTRY = {
    "best": CURRENT_BEST_MODEL_PATH,
    "model2": PREVIOUS_BEST_MODEL_PATH,
    "previous_best": PREVIOUS_BEST_MODEL_PATH,
    "fallback": FALLBACK_MODEL_PATH,
    "simple": SIMPLE_MODEL_PATH,
}

def get_model_path(model_name: str = "best") -> str:
    """
    Get the full path to a model checkpoint.
    
    Args:
        model_name: Name of the model to get path for. Options:
            - "best": Latest best model (preferred)
            - "previous_best": Previous best model
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

def get_model_dir(model_name: str = "best") -> str:
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
        Dictionary with best and model2 paths
    """
    return {
        "best": get_model_path("best"),
        "model2": get_model_path("model2")
    }

def get_model_info(model_id: str) -> Dict[str, Any]:
    """
    Get comprehensive information about a model.
    
    Args:
        model_id: Model identifier (e.g., "best", "simple", "previous_best")
    
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

def get_model_path_with_fallback(model_name: str = "best") -> str:
    """
    Get the full path to a model checkpoint with fallback support.
    
    If the requested model doesn't exist, falls back to the fallback model.
    This is useful for web applications that need to stay online even when
    the current best model is temporarily unavailable.
    
    Args:
        model_name: Name of the model to get path for. Options:
            - "best": Latest best model (with fallback, preferred)
            - "fallback": Direct access to fallback model
            - Other models: No fallback, returns as-is
    
    Returns:
        Full path to the model checkpoint file (or fallback if needed)
    """
    # Get the primary model path
    primary_path = get_model_path(model_name)
    
    # For best, check if file exists and fallback if needed
    if model_name in ["best"]:
        if not validate_model_path(primary_path):
            # Primary model doesn't exist, use fallback
            fallback_path = get_model_path("fallback")
            if validate_model_path(fallback_path):
                return fallback_path
            else:
                # Even fallback doesn't exist, return primary path (will cause error)
                return primary_path
    
    return primary_path

def get_available_model_with_fallback(model_name: str = "best") -> str:
    """
    Get an available model path, trying primary first, then fallback.
    
    This function is designed for web applications that need to gracefully
    handle model unavailability by falling back to a known working model.
    
    Args:
        model_name: Name of the model to get path for
    
    Returns:
        Full path to an available model checkpoint file
    
    Raises:
        FileNotFoundError: If neither primary nor fallback model exists
    """
    # Try primary model first
    try:
        primary_path = get_model_path(model_name)
        if validate_model_path(primary_path):
            return primary_path
    except (ValueError, KeyError):
        pass  # Model name not found, try fallback
    
    # Try fallback model
    try:
        fallback_path = get_model_path("fallback")
        if validate_model_path(fallback_path):
            return fallback_path
    except (ValueError, KeyError):
        pass  # Fallback model not found
    
    # Neither model exists
    raise FileNotFoundError(f"Neither primary model '{model_name}' nor fallback model is available")

# Convenience variables for backward compatibility
DEFAULT_MODEL_PATH = CURRENT_BEST_MODEL_PATH
DEFAULT_MODEL_DIR = os.path.join(CHECKPOINTS_BASE_DIR, CURRENT_BEST_MODEL_DIR)


def get_all_model_participants_from_generations(knockout_config: Dict[str, Any]) -> List['TournamentParticipant']:
    """
    Convert all models in MODEL_GENERATIONS into TournamentParticipant objects.
    
    This function iterates through all entries in MODEL_GENERATIONS and creates
    TournamentParticipant objects for each model file, suitable for use in
    knockout tournaments.
    
    Args:
        knockout_config: Configuration dictionary for knockout tournament strategy.
            Should contain MCTS parameters like "mcts_sims", "enable_gumbel_root_selection", etc.
            Will be merged with each participant's strategy_config.
    
    Returns:
        List of TournamentParticipant objects, one for each model in MODEL_GENERATIONS
    
    Example:
        >>> config = {"mcts_sims": 220, "enable_gumbel_root_selection": True, "temperature": 1.0}
        >>> participants = get_all_model_participants_from_generations(config)
        >>> len(participants)  # Total number of models across all generations
    """
    from hex_ai.inference.knockout_tournament import TournamentParticipant
    
    participants = []
    checkpoint_pattern = re.compile(r'epoch(\d+)_mini(\d+)(?:\.pt\.gz)?$')
    
    for gen_num, gen_data in sorted(MODEL_GENERATIONS.items()):
        gen_dir = gen_data["dir"]
        model_files = gen_data["models"]
        
        for model_file in model_files:
            # Ensure model_file has .pt.gz extension if not present
            if not model_file.endswith('.pt.gz'):
                model_file = model_file + '.pt.gz'
            
            # Create full path to model
            model_path = os.path.join(CHECKPOINTS_BASE_DIR, gen_dir, model_file)
            
            # Extract epoch and mini from filename for naming
            match = checkpoint_pattern.match(model_file)
            if match:
                epoch = int(match.group(1))
                mini = int(match.group(2))
                # Create descriptive name: gen{gen}_epoch{epoch}_mini{mini}
                participant_name = f"gen{gen_num}_epoch{epoch}_mini{mini}"
            else:
                # Fallback if pattern doesn't match
                model_name_base = model_file.replace('.pt.gz', '')
                participant_name = f"gen{gen_num}_{model_name_base}"
            
            # Create strategy config for this participant
            strategy_config = {
                "strategy": "mcts",
                "model_path": model_path,
                **knockout_config
            }
            
            # Create metadata
            metadata = {
                "generation": gen_num,
                "model_file": model_file,
                "model_dir": gen_dir,
                "model_path": model_path
            }
            
            # Add epoch/mini to metadata if extracted
            if match:
                metadata["epoch"] = epoch
                metadata["mini"] = mini
            
            participant = TournamentParticipant(
                name=participant_name,
                strategy_config=strategy_config,
                metadata=metadata
            )
            
            participants.append(participant)
    
    return participants 