"""
Snowflake2025 Hex AI Training Framework

A modern PyTorch implementation of a Hex AI training system, revamping the 2018 'Snowflake' project 
to create a stronger Hex AI using current best practices.

This module validates the environment and fails fast if setup is incorrect.
"""

import os
import sys
from pathlib import Path

# Version info
__version__ = "2025.1.0"
__author__ = "Snowflake2025 Team"

# Core modules that should be available
__all__ = [
    "models",
    "training_orchestration", 
    "inference",
    "selfplay",
    "data_processing",
    "system_utils",
    "training_utils",
    "value_utils",
]

def _validate_environment():
    """Validate the environment and fail fast if setup is incorrect."""
    # Check virtual environment
    venv_path = os.environ.get("VIRTUAL_ENV", "")
    if not venv_path:
        raise ImportError(
            "hex_ai requires hex_ai_env virtual environment.\n"
            "Activate it first: source hex_ai_env/bin/activate"
        )
    
    if "hex_ai_env" not in venv_path:
        raise ImportError(
            f"hex_ai requires hex_ai_env virtual environment.\n"
            f"Current environment: {venv_path}\n"
            f"Activate correct environment: source hex_ai_env/bin/activate"
        )
    
    # Check PYTHONPATH
    project_root = Path(__file__).parent.parent
    python_path = os.environ.get("PYTHONPATH", "")
    
    if not python_path:
        raise ImportError(
            "hex_ai requires PYTHONPATH to include project root.\n"
            "Set it first: export PYTHONPATH=."
        )
    
    # Check if project root is in Python's sys.path (which includes PYTHONPATH)
    # Accept either the absolute project root path or "." (relative path)
    project_root_str = str(project_root)
    current_dir_str = str(Path.cwd())
    
    path_is_valid = (
        project_root_str in sys.path or
        current_dir_str in sys.path or
        "." in python_path.split(os.pathsep) if python_path else False
    )
    
    if not path_is_valid:
        raise ImportError(
            f"hex_ai requires project root in Python path.\n"
            f"Current PYTHONPATH: {python_path}\n"
            f"Expected project root: {project_root_str} or .\n"
            f"Set it first: export PYTHONPATH=."
        )

# Validate environment on import
_validate_environment()

# Import core modules - let import errors propagate
from . import models
from . import training_orchestration
from . import inference
from . import selfplay

 