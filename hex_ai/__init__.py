"""
Hex AI - Modern PyTorch implementation of the 2018 TensorFlow Hex AI

This package contains the core components for training a modern Hex AI:
- Dataset handling for .trmph game files
- Two-headed ResNet model architecture
- Training utilities and configuration
"""

__version__ = "0.1.0"
__author__ = "Snowflake2025 Team"

# Core imports
from .models import TwoHeadedResNet, ResNetBlock

from .config import *

__all__ = [
    "TwoHeadedResNet",
    "ResNetBlock",
] 