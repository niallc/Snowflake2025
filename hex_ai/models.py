"""
Model architecture for the Hex AI.

This module contains the neural network architectures used for the Hex AI,
including the main TwoHeadedResNet model and supporting components.

The architecture follows a two-headed design:
- Policy head: Predicts move probabilities for each board position
- Value head: Predicts the probability of winning from the current position
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .config import (
    BOARD_SIZE, NUM_PLAYERS, POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE,
    INITIAL_CHANNELS, CHANNEL_PROGRESSION, RESNET_DEPTH
)


class ResNetBlock(nn.Module):
    """
    Standard ResNet block with two convolutional layers and residual connection.
    
    This block implements the basic building block of ResNet architectures,
    with batch normalization and ReLU activations. It handles both regular
    residual connections and projection shortcuts when dimensions change.
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # TODO: Implement ResNet block
        # Should contain:
        # - Two sequences of Conv2d -> BatchNorm2d -> ReLU
        # - Residual connection handling
        # - Projection shortcut when dimensions change
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass
        pass


class TwoHeadedResNet(nn.Module):
    """
    Two-headed ResNet architecture for Hex AI.
    
    This model uses a ResNet backbone with two separate heads:
    - Policy head: Predicts move probabilities (169 outputs for 13x13 board)
    - Value head: Predicts win probability (1 output)
    
    The architecture follows modern best practices with:
    - Global average pooling after the ResNet body
    - Separate linear layers for policy and value heads
    - Batch normalization and proper initialization
    """
    
    def __init__(self, resnet_depth: int = RESNET_DEPTH):
        super().__init__()
        
        # Input layer: Convert board representation to initial features
        # Input shape: (batch_size, 2, 13, 13) for two players
        # TODO: Implement initial convolution layer
        
        # ResNet body
        # TODO: Implement ResNet layers based on depth
        
        # Global average pooling
        # TODO: Implement adaptive average pooling
        
        # Policy head: Predict move probabilities
        # TODO: Implement policy head (linear layer to 169 outputs)
        
        # Value head: Predict win probability  
        # TODO: Implement value head (linear layer to 1 output)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using modern best practices."""
        # TODO: Implement weight initialization
        pass
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the two-headed ResNet.
        
        Args:
            x: Input tensor of shape (batch_size, 2, 13, 13)
            
        Returns:
            Tuple of (policy_logits, value_logit):
            - policy_logits: Shape (batch_size, 169)
            - value_logit: Shape (batch_size, 1)
        """
        # TODO: Implement forward pass
        # Should return both policy and value outputs
        pass


def create_model(model_type: str = "resnet18") -> TwoHeadedResNet:
    """
    Factory function to create a model instance.
    
    Args:
        model_type: Type of model to create (currently only "resnet18")
        
    Returns:
        Initialized model instance
    """
    if model_type == "resnet18":
        return TwoHeadedResNet(resnet_depth=18)
    else:
        raise ValueError(f"Unknown model type: {model_type}") 