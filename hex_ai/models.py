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


class GlobalPoolingResidualBlock(nn.Module):
    """
    Residual block with both local conv and global pooling re-injection,
    inspired by KataGo's gpool blocks.
    
    This block applies local convolutions like a standard ResNet block,
    but also computes global features through pooling and reinjects them
    as a bias to all spatial locations. This allows the network to propagate
    global board knowledge to all locations.
    """
    
    def __init__(self, channels: int, gpool_channels: int = 16):
        super().__init__()
        
        # Local conv path (same as standard ResNet block)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Global pooling path
        self.gconv = nn.Conv2d(channels, gpool_channels, kernel_size=1, bias=False)
        self.gbn = nn.BatchNorm2d(gpool_channels)
        self.fc = nn.Linear(gpool_channels, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Global pooling path
        g = F.relu(self.gbn(self.gconv(x)))   # (B, gpool_channels, H, W)
        g = g.mean(dim=(2, 3))                # (B, gpool_channels)
        g = self.fc(g).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        
        # Inject global bias
        out = out + g
        
        # Residual connection
        out = F.relu(out + x)
        return out


class PolicyHead(nn.Module):
    """
    Policy head with global pooling bias injection.
    
    This head computes both local features and global board context,
    then combines them before producing move logits. The global pooling
    allows the policy to consider whole-board balance when selecting moves.
    """
    
    def __init__(self, trunk_channels: int, board_size: int, gpool_channels: int = 16):
        super().__init__()
        self.board_size = board_size
        
        # Local conv path
        self.conv1 = nn.Conv2d(trunk_channels, trunk_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(trunk_channels)
        
        # Local → logits
        self.conv2 = nn.Conv2d(trunk_channels, 1, kernel_size=1, bias=False)
        
        # Global pooling bias
        self.gconv = nn.Conv2d(trunk_channels, gpool_channels, kernel_size=1, bias=False)
        self.gbn = nn.BatchNorm2d(gpool_channels)
        self.fc = nn.Linear(gpool_channels, trunk_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local features
        local = F.relu(self.bn1(self.conv1(x)))
        
        # Global pooling path
        g = F.relu(self.gbn(self.gconv(x)))      # (B, gpool_channels, H, W)
        g = g.mean(dim=(2, 3))                   # (B, gpool_channels)
        g = self.fc(g).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        
        # Inject global bias
        local = local + g
        
        # Final conv → logits
        p = self.conv2(local)  # (B, 1, H, W)
        return p.flatten(1)    # (B, H*W)


class TwoHeadedResNet(nn.Module):
    """
    Two-headed ResNet architecture for Hex AI, inspired by KataGo.
    
    This model uses a ResNet backbone with two separate heads:
    - Policy head: Predicts move probabilities (169 outputs for 13x13 board)
    - Value head: Predicts Red's win probability (1 output)
    
    The value head predicts Red's win probability because Red wins are labeled as 1.0 in training.
    The output is a value in [-1, 1] range with tanh activation that should be converted to [0, 1] probability.
    
    Key improvements from KataGo:
    - Flattened channel progression (constant trunk_channels instead of growing)
    - Mixed plain and global pooling residual blocks
    - Policy head with global pooling bias injection
    - Enhanced value head with hidden layer and optional bottleneck
    """
    
    def __init__(self, num_blocks: int = 10, trunk_channels: int = 128, 
                 dropout_prob: float = 0.1, use_value_bottleneck: bool = True):
        super().__init__()
        self.num_blocks = num_blocks
        self.trunk_channels = trunk_channels
        self.use_value_bottleneck = use_value_bottleneck
        
        # Input layer: Convert board representation to initial features
        # Input shape: (batch_size, 3, 13, 13) for two players + player-to-move channel
        self.input_conv = nn.Conv2d(3, trunk_channels, 
                                   kernel_size=5, stride=1, padding=2, bias=False)
        self.input_bn = nn.BatchNorm2d(trunk_channels)
        
        # Trunk: mix plain and gpool blocks with constant channel count
        blocks = []
        for i in range(num_blocks):
            if i % 3 == 2:  # every 3rd block is a gpool block
                blocks.append(GlobalPoolingResidualBlock(trunk_channels))
            else:
                blocks.append(ResNetBlock(trunk_channels, trunk_channels))
        self.trunk = nn.Sequential(*blocks)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout layer (kept for value path only)
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # Policy head with global pooling bias injection
        self.policy_head = PolicyHead(trunk_channels, BOARD_SIZE)
        
        # Enhanced value head with hidden layer and optional bottleneck
        if use_value_bottleneck:
            # 1x1 bottleneck convolution to reduce channels before pooling
            self.value_pre = nn.Sequential(
                nn.Conv2d(trunk_channels, 32, kernel_size=1, bias=False),
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
                nn.Linear(trunk_channels, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),  # Light regularization
                nn.Linear(256, 1)
            )
        
        # Initialize weights using modern best practices
        self._initialize_weights()
    
    
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
        trunk_out = self.trunk(x)
        return trunk_out

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
        trunk_out = self.forward_shared(x)
        
        # Policy head with global pooling bias injection
        policy_logits = self.policy_head(trunk_out)
        
        # Value head path
        if self.use_value_bottleneck:
            # Apply 1x1 bottleneck convolution
            value_features = self.value_pre(trunk_out)
            # Global average pooling
            value_features = value_features.mean(dim=(2, 3))  # GAP
        else:
            # Standard global average pooling
            value_features = self.global_pool(trunk_out)
            value_features = value_features.view(value_features.size(0), -1)
        
        value_signed = torch.tanh(self.value_head(value_features))  # (batch_size, 1)
        
        return policy_logits, value_signed

    @torch.no_grad()
    def forward_value_only(self, x: torch.Tensor) -> torch.Tensor:
        """Value-only inference path for faster leaf evaluation."""
        trunk_out = self.forward_shared(x)
        
        if self.use_value_bottleneck:
            value_features = self.value_pre(trunk_out)
            value_features = value_features.mean(dim=(2, 3))  # GAP
        else:
            value_features = self.global_pool(trunk_out)
            value_features = value_features.view(value_features.size(0), -1)
        
        return torch.tanh(self.value_head(value_features))


def create_model(model_type: str = "katago_inspired", use_value_bottleneck: bool = True, 
                num_blocks: int = 10, trunk_channels: int = 128) -> TwoHeadedResNet:
    """
    Factory function to create a model instance.
    
    Args:
        model_type: Type of model to create ("katago_inspired" or "resnet18" for backward compatibility)
        use_value_bottleneck: Whether to use 1x1 bottleneck in value head
        num_blocks: Number of residual blocks in the trunk
        trunk_channels: Number of channels in the trunk (constant throughout)
        
    Returns:
        Initialized model instance
    """
    if model_type == "katago_inspired":
        return TwoHeadedResNet(num_blocks=num_blocks, trunk_channels=trunk_channels, 
                              use_value_bottleneck=use_value_bottleneck)
    elif model_type == "resnet18":
        # Backward compatibility - use old parameters
        return TwoHeadedResNet(num_blocks=8, trunk_channels=128, 
                              use_value_bottleneck=use_value_bottleneck)
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
    
    # Check if model has the new architecture
    has_bottleneck = hasattr(model, 'use_value_bottleneck') and model.use_value_bottleneck
    has_trunk_channels = hasattr(model, 'trunk_channels')
    has_num_blocks = hasattr(model, 'num_blocks')
    
    if has_trunk_channels and has_num_blocks:
        # New KataGo-inspired architecture
        value_head_desc = "Enhanced (bottleneck + hidden layer)" if has_bottleneck else "Enhanced (hidden layer only)"
        gpool_blocks = model.num_blocks // 3
        plain_blocks = model.num_blocks - gpool_blocks
        
        summary = f"""
Model Summary:
==============
Total Parameters: {total_params:,}
Model Type: {model.__class__.__name__} (KataGo-inspired)

Architecture:
- Input: (batch_size, 3, 13, 13)
- Trunk: {model.num_blocks} blocks with constant {model.trunk_channels} channels
  * {plain_blocks} plain ResNet blocks
  * {gpool_blocks} global pooling blocks (every 3rd block)
- Policy Head: Global pooling bias injection preserving 13x13 spatial structure
- Value Head: {value_head_desc} with GAP ({VALUE_OUTPUT_SIZE} outputs)

Output:
- Policy Logits: (batch_size, 169) - row-major flattened from 13x13
- Value Signed: (batch_size, 1) with tanh activation ([-1,1] range)
"""
    else:
        # Legacy architecture
        value_head_desc = "Enhanced (bottleneck + hidden layer)" if has_bottleneck else "Enhanced (hidden layer only)"
        
        summary = f"""
Model Summary:
==============
Total Parameters: {total_params:,}
Model Type: {model.__class__.__name__} (Legacy)

Architecture:
- Input: (batch_size, 3, 13, 13)
- ResNet Body: 4 stages with {CHANNEL_PROGRESSION} channels (no downsampling)
- Policy Head: Convolutional (1x1 convs) preserving 13x13 spatial structure
- Value Head: {value_head_desc} with GAP ({VALUE_OUTPUT_SIZE} outputs)

Output:
- Policy Logits: (batch_size, 169) - row-major flattened from 13x13
- Value Signed: (batch_size, 1) with tanh activation ([-1,1] range)
"""
    return summary 