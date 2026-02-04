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

def _discover_repo_root() -> Path | None:
    """
    Best-effort discovery of the repository root.
    
    We prefer to support standard Python packaging (e.g. `pip install -e .`) rather than
    requiring `PYTHONPATH=.`. When running from a source checkout, we can identify the
    repo root by locating a `pyproject.toml` (preferred) or `.git` marker.
    
    Returns:
        Path to repo root if detected, else None.
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").is_file() or (parent / ".git").exists():
            return parent
    return None

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

    # Prefer standard packaging over PYTHONPATH hacks.
    # If we're running from a source checkout, ensure the repo root is importable so that
    # `import hex_ai` behaves consistently across CLIs, tests, and modules.
    repo_root = _discover_repo_root()
    if repo_root is not None:
        repo_root_str = str(repo_root)
        cwd_str = str(Path.cwd().resolve())
        repo_root_on_path = repo_root_str in sys.path
        running_from_repo_root = cwd_str == repo_root_str
        if not (repo_root_on_path or running_from_repo_root):
            raise ImportError(
                "hex_ai could not confirm the repository root is importable.\n"
                f"Detected repo root: {repo_root_str}\n"
                f"Current working directory: {cwd_str}\n"
                "\n"
                "Recommended setup (once per venv):\n"
                "  pip install -r requirements.txt\n"
                "  pip install -e .\n"
                "\n"
                "Or run commands from the repository root."
            )

# Validate environment on import
_validate_environment()

 