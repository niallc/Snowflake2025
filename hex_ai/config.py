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
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Global configuration settings

# Verbose logging levels:
# 0: Critical issues and errors
# 1: Important info and warnings
# 2: Detailed info (default for development)
# 3: Very detailed debug info
# 4: Extremely verbose debug info
VERBOSE_LEVEL = 2

# Training configuration
BOARD_SIZE = 13 
NUM_PLAYERS = 2 
POLICY_OUTPUT_SIZE = BOARD_SIZE * BOARD_SIZE  # 169 for 13x13 board
VALUE_OUTPUT_SIZE = 1

# Player and piece constants
# Player constants (for player-to-move channel and game logic)
BLUE_PLAYER = 0
RED_PLAYER = 1

# Piece constants (for N×N board representation)
BLUE_PIECE = "b"  # Blue pieces on the board (character representation)
RED_PIECE = "r"   # Red pieces on the board (character representation)
EMPTY_PIECE = "e" # Empty positions (character representation)

# Legacy numeric constants for backward compatibility (will be removed)
LEGACY_BLUE_PIECE = 0
LEGACY_RED_PIECE = 1
LEGACY_EMPTY_PIECE = -1

# One-hot encoded board constants (for 2N×N and 3N×N formats)
PIECE_ONEHOT = 1      # Value for occupied positions in one-hot encoding
EMPTY_ONEHOT = 0      # Value for empty positions in one-hot encoding
BLUE_CHANNEL = 0      # Channel index for blue pieces
RED_CHANNEL = 1       # Channel index for red pieces
PLAYER_CHANNEL = 2    # Channel index for player-to-move (3N×N format)

# Model input channel counts
MODEL_CHANNELS = 3 # Models expect 3-channel input (BLUE_CHANNEL, RED_CHANNEL, PLAYER_CHANNEL)

# Winner format mapping
# TRMPH format: "b" = BLUE win, "r" = RED win (was "1" and "2")
# Training format: BLUE = 0.0, RED = 1.0 (unchanged)
# Enhanced metadata: "BLUE" or "RED" (clear text)
TRMPH_BLUE_WIN = "b"
TRMPH_RED_WIN = "r"
TRAINING_BLUE_WIN = 0.0
TRAINING_RED_WIN = 1.0

# Default hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 128
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