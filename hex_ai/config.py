"""
Configuration constants and settings for the Hex AI project.

This module contains all the configuration constants used throughout the project,
including model parameters, data paths, and training settings.
"""

import os
from pathlib import Path
import torch

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LEGACY_CODE_DIR = PROJECT_ROOT / "legacy_code"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Global configuration settings

# Verbose logging levels:
# 0: Critical issues and errors
# 1: Important info and warnings
# 2: Detailed info (default for development)
# 3: Very detailed debug info
# 4: Extremely verbose debug info
VERBOSE_LEVEL = 0

# Training configuration
BOARD_SIZE = 13 
NUM_PLAYERS = 2 
POLICY_OUTPUT_SIZE = BOARD_SIZE * BOARD_SIZE  # 169 for 13x13 board
VALUE_OUTPUT_SIZE = 1

# Default hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 10

# Loss weights for standardized comparison
POLICY_LOSS_WEIGHT = 0.14
VALUE_LOSS_WEIGHT = 0.86

# Model architecture
RESNET_DEPTH = 18  # ResNet-18 for initial implementation
INITIAL_CHANNELS = 64
CHANNEL_PROGRESSION = [64, 128, 256, 512]  # Standard ResNet progression

# Data augmentation
ROTATION_AUGMENTATION = True
REFLECTION_AUGMENTATION = True

# Logging
WANDB_PROJECT_NAME = "hex-ai-2025"
LOG_INTERVAL = 100  # Log every N batches

# File extensions
TRMPH_EXTENSION = ".trmph"
NUMPY_EXTENSION = ".npy"
PICKLE_EXTENSION = ".pkl" 