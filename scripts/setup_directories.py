#!/usr/bin/env python3
"""
Setup script to create required directories for Snowflake2025.

This script creates all the directories that the code expects to exist,
with appropriate warnings for first-time setup.
"""

import os
import sys
from pathlib import Path

def create_required_directories():
    """Create all required directories with appropriate warnings."""
    
    # Required directories
    directories = [
        "checkpoints",
        "checkpoints/bookkeeping", 
        "checkpoints/hyperparameter_tuning",
        "data",
        "data/tournament_play",
        "data/sf25", 
        "data/twoNetGames",
        "data/web_games",
        "data/processed",
        "data/collected",
        "data/cleaned",
        "logs",
        "temp"
    ]
    
    print("=" * 60)
    print("SNOWFLAKE2025 DIRECTORY SETUP")
    print("=" * 60)
    print()

    created_dirs = []
    existing_dirs = []
    
    for directory in directories:
        dir_path = Path(directory)
        if dir_path.exists():
            existing_dirs.append(directory)
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(directory)
    
    if created_dirs:
        print("✅ Created directories:")
        for directory in created_dirs:
            print(f"   - {directory}")
        print()
    
    if existing_dirs:
        print("📁 Directories already exist:")
        for directory in existing_dirs:
            print(f"   - {directory}")
        print()
    
    print("📋 Next steps:")
    print("1. Get the latest model file from Niall Cardin (niallc@gmail.com)")
    print("2. Place it in the checkpoints/ directory")
    print("3. Update hex_ai/inference/model_config.py with the model path")
    print("4. For training, get training data from Niall or generate using self-play")
    print()
    print("🔍 Check your system capabilities:")
    print("   python scripts/check_device.py")
    print()
    print("🎮 To start playing:")
    print("   source hex_ai_env/bin/activate")
    print("   PYTHONPATH=. python -m hex_ai.web.app --port 5001")
    print()
    print("🏋️ To start training:")
    print("   source hex_ai_env/bin/activate") 
    print("   PYTHONPATH=. python scripts/training_pipeline.py --help")

if __name__ == "__main__":
    create_required_directories()
