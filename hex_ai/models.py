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
import math

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
        
        # Main path: two conv layers with batch norm and ReLU
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection (identity or projection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add shortcut connection
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out


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
    - Mixed precision support
    """
    
    def __init__(self, resnet_depth: int = RESNET_DEPTH, dropout_prob: float = 0.1):
        super().__init__()
        self.resnet_depth = resnet_depth
        
        # Input layer: Convert board representation to initial features
        # Input shape: (batch_size, 3, 13, 13) for two players + player-to-move channel
        self.input_conv = nn.Conv2d(3, INITIAL_CHANNELS, 
                                   kernel_size=5, stride=1, padding=2, bias=False)
        self.input_bn = nn.BatchNorm2d(INITIAL_CHANNELS)
        
        # ResNet body: 4 stages with different channel counts
        self.layer1 = self._make_layer(INITIAL_CHANNELS, CHANNEL_PROGRESSION[0], 
                                     blocks=2, stride=1)  # 64 channels
        self.layer2 = self._make_layer(CHANNEL_PROGRESSION[0], CHANNEL_PROGRESSION[1], 
                                     blocks=2, stride=2)  # 128 channels
        self.layer3 = self._make_layer(CHANNEL_PROGRESSION[1], CHANNEL_PROGRESSION[2], 
                                     blocks=2, stride=2)  # 256 channels
        self.layer4 = self._make_layer(CHANNEL_PROGRESSION[2], CHANNEL_PROGRESSION[3], 
                                     blocks=2, stride=2)  # 512 channels
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # Policy head: Predict move probabilities
        self.policy_head = nn.Linear(CHANNEL_PROGRESSION[3], POLICY_OUTPUT_SIZE)
        
        # Value head: Predict win probability
        self.value_head = nn.Linear(CHANNEL_PROGRESSION[3], VALUE_OUTPUT_SIZE)
        
        # Initialize weights using modern best practices
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                   blocks: int, stride: int) -> nn.Sequential:
        """Create a layer of ResNet blocks."""
        layers = []
        
        # First block may have different stride
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        
        # Remaining blocks have stride=1
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights using modern best practices."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming initialization for conv layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize batch norm layers
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the two-headed ResNet.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 13, 13)
            
        Returns:
            Tuple of (policy_logits, value_logit):
            - policy_logits: Shape (batch_size, 169)
            - value_logit: Shape (batch_size, 1)
        """
        # Input convolution
        x = F.relu(self.input_bn(self.input_conv(x)))
        
        # ResNet body
        x = self.layer1(x)  # (batch_size, 64, 13, 13)
        x = self.layer2(x)  # (batch_size, 128, 7, 7)
        x = self.layer3(x)  # (batch_size, 256, 4, 4)
        x = self.layer4(x)  # (batch_size, 512, 2, 2)
        
        # Global average pooling
        x = self.global_pool(x)  # (batch_size, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (batch_size, 512)
        # Dropout
        x = self.dropout(x)
        # Policy head
        policy_logits = self.policy_head(x)  # (batch_size, 169)
        # Value head
        value_logit = self.value_head(x)  # (batch_size, 1)
        return policy_logits, value_logit


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


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module) -> str:
    """
    Get a summary of the model architecture.
    
    Args:
        model: PyTorch model
        
    Returns:
        String summary of the model
    """
    total_params = count_parameters(model)
    
    summary = f"""
Model Summary:
==============
Total Parameters: {total_params:,}
Model Type: {model.__class__.__name__}

Architecture:
- Input: (batch_size, 3, 13, 13)
- ResNet Body: 4 stages with {CHANNEL_PROGRESSION} channels
- Global Average Pooling
- Policy Head: {POLICY_OUTPUT_SIZE} outputs
- Value Head: {VALUE_OUTPUT_SIZE} outputs

Output:
- Policy Logits: (batch_size, 169)
- Value Logit: (batch_size, 1)
"""
    return summary 