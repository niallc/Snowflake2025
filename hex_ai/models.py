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
    - Value head: Predicts Red's win probability (1 output)
    
    The value head predicts Red's win probability because Red wins are labeled as 1.0 in training.
    The output is a value in [-1, 1] range with tanh activation that should be converted to [0, 1] probability.
    
    The architecture follows modern best practices with:
    - Global average pooling after the ResNet body
    - Separate linear layers for policy and value heads
    - Batch normalization and proper initialization
    - Mixed precision support
    - Enhanced value head with hidden layer and optional bottleneck
    """
    
    def __init__(self, resnet_depth: int = RESNET_DEPTH, dropout_prob: float = 0.1, 
                 use_value_bottleneck: bool = True):
        super().__init__()
        self.resnet_depth = resnet_depth
        self.use_value_bottleneck = use_value_bottleneck
        
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
        
        # Enhanced value head with hidden layer and optional bottleneck
        if use_value_bottleneck:
            # 1x1 bottleneck convolution to reduce channels before pooling
            self.value_pre = nn.Sequential(
                nn.Conv2d(CHANNEL_PROGRESSION[3], 32, kernel_size=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
            # Value head with hidden layer
            self.value_head = nn.Sequential(
                nn.Linear(32, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),  # Light regularization
                nn.Linear(256, 1)
            )
        else:
            # Value head with hidden layer (no bottleneck)
            self.value_head = nn.Sequential(
                nn.Linear(CHANNEL_PROGRESSION[3], 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),  # Light regularization
                nn.Linear(256, 1)
            )
        
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
    
    def forward_shared(self, x: torch.Tensor) -> torch.Tensor:
        """Run the shared trunk up to the penultimate representation."""
        x = F.relu(self.input_bn(self.input_conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the two-headed ResNet.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 13, 13)
            
        Returns:
            Tuple of (policy_logits, value_signed):
            - policy_logits: Shape (batch_size, 169)
            - value_signed: Shape (batch_size, 1) - Signed value in [-1,1] range (tanh-activated)
        """
        # Shared trunk
        features = self.forward_shared(x)
        
        # Policy head path
        policy_features = self.global_pool(features)
        policy_features = policy_features.view(policy_features.size(0), -1)
        policy_features = self.dropout(policy_features)
        policy_logits = self.policy_head(policy_features)  # (batch_size, 169)
        
        # Value head path
        if self.use_value_bottleneck:
            # Apply 1x1 bottleneck convolution
            value_features = self.value_pre(features)
            # Global average pooling
            value_features = value_features.mean(dim=(2, 3))  # GAP
        else:
            # Standard global average pooling
            value_features = self.global_pool(features)
            value_features = value_features.view(value_features.size(0), -1)
        
        value_signed = torch.tanh(self.value_head(value_features))  # (batch_size, 1)
        
        return policy_logits, value_signed

    @torch.no_grad()
    def forward_value_only(self, x: torch.Tensor) -> torch.Tensor:
        """Value-only inference path for faster leaf evaluation."""
        features = self.forward_shared(x)
        
        if self.use_value_bottleneck:
            value_features = self.value_pre(features)
            value_features = value_features.mean(dim=(2, 3))  # GAP
        else:
            value_features = self.global_pool(features)
            value_features = value_features.view(value_features.size(0), -1)
        
        return torch.tanh(self.value_head(value_features))


def create_model(model_type: str = "resnet18", use_value_bottleneck: bool = True) -> TwoHeadedResNet:
    """
    Factory function to create a model instance.
    
    Args:
        model_type: Type of model to create (currently only "resnet18")
        use_value_bottleneck: Whether to use 1x1 bottleneck in value head
        
    Returns:
        Initialized model instance
    """
    if model_type == "resnet18":
        return TwoHeadedResNet(resnet_depth=18, use_value_bottleneck=use_value_bottleneck)
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
    
    # Check if model has the new value head structure
    has_bottleneck = hasattr(model, 'use_value_bottleneck') and model.use_value_bottleneck
    value_head_desc = "Enhanced (bottleneck + hidden layer)" if has_bottleneck else "Enhanced (hidden layer only)"
    
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
- Value Head: {value_head_desc} ({VALUE_OUTPUT_SIZE} outputs)

Output:
- Policy Logits: (batch_size, 169)
- Value Signed: (batch_size, 1) with tanh activation ([-1,1] range)
"""
    return summary 