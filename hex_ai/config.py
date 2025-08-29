"""
Configuration constants and settings for the Hex AI project.

This module contains all the configuration constants used throughout the project,
including model parameters, data paths, and training settings.
"""

from pathlib import Path
import torch
from hex_ai.enums import Player, Piece, Channel

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

# One-hot encoded board constants (for 2N×N and 3N×N formats)
PIECE_ONEHOT = 1      # Value for occupied positions in one-hot encoding
EMPTY_ONEHOT = 0      # Value for empty positions in one-hot encoding

# Deprecated semantic constants – prefer hex_ai.enums usage throughout the codebase.
# Player int aliases removed; use Player enum everywhere internally.

BLUE_PIECE = Piece.BLUE.value
RED_PIECE = Piece.RED.value
EMPTY_PIECE = Piece.EMPTY.value

BLUE_CHANNEL = Channel.BLUE.value
RED_CHANNEL = Channel.RED.value
PLAYER_CHANNEL = Channel.PLAYER_TO_MOVE.value

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

# TRMPH format constants
TRMPH_PREFIX = "#13,"

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

# MCTS inference defaults
DEFAULT_BATCH_CAP = 64  # Default batch size for neural network evaluation
DEFAULT_C_PUCT = 1.5    # Default PUCT exploration constant

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