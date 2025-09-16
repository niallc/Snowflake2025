"""
Model architecture for the Hex AI.

This module contains the neural network architectures used for the Hex AI,
including the main TwoHeadedResNet model and supporting components.

The architecture follows a two-headed design:
- Policy head: Predicts move probabilities for each board position
- Value head: Predicts the probability of winning from the current position

TODO: Remaining Updates Needed
==============================

The KataGo-inspired architecture is now implemented. Remaining tasks:

1. CHECKPOINT COMPATIBILITY:
   - Old checkpoints are not compatible with new architecture
   - Need migration strategy or version handling for existing checkpoints
   - New model has different parameter count and structure

2. TESTING UPDATES:
   - Update existing model tests to provide move_stage parameter
   - Add tests for move_stage computation edge cases
   - Test checkpoint loading/saving with new architecture

3. DOCUMENTATION UPDATES:
   - Update model documentation with new API
   - Document move_stage computation and value ranges
   - Update training guides with new loss functions

4. PERFORMANCE MONITORING:
   - Benchmark performance impact of new architecture
   - Monitor memory usage changes in production
   - Compare training stability with new loss functions
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

# Numerical stability constant for log-cosh loss
MAX_LOG_COSH_INPUT_ABS = 20.0
# Rationale: log(cosh(x)) loses precision for |x| > 20. Beyond this point,
# the function becomes essentially linear and may cause numerical issues.

# =============================================================================
# POLICY HEAD STABILITY CONFIGURATION
# =============================================================================

class PolicyHeadStabilityConfig:
    """
    Configuration for policy head stability mechanisms.
    
    This class centralizes all the hyperparameters and thresholds used to prevent
    the systematic value explosion that was occurring in the policy head final layer.
    
    The key insight is that the policy head final conv layer was accumulating gradients
    without proper normalization, causing logits to grow from ~0.5 to >20.0 over ~500 batches.
    """
    
    # Initialization
    FINAL_LAYER_SCALING = 0.1
    """Scaling factor for policy head final conv layer initialization.
    Standard practice for policy networks to prevent extreme initial logits."""
    
    # Weight decay
    FINAL_LAYER_WEIGHT_DECAY_FACTOR = 2.0
    """Multiplier for weight decay on policy head final layer.
    Higher weight decay prevents gradient accumulation without harming performance."""
    
    # Monitoring thresholds
    GRADIENT_NORM_WARNING_THRESHOLD = 5.0
    """Gradient norm threshold for early instability detection.
    Lower than global threshold to catch policy head issues early."""
    
    WEIGHT_MAGNITUDE_WARNING_THRESHOLD = 2.0
    """Weight magnitude threshold for early instability detection.
    Helps detect when initialization scaling is being overwhelmed."""
    
    # Layer normalization
    LAYER_NORM_EPS = 1e-5
    """Epsilon for layer normalization numerical stability.
    Standard value to prevent division by zero."""

# Global instance for easy access
POLICY_HEAD_CONFIG = PolicyHeadStabilityConfig()


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
        local = F.relu(self.bn1(self.conv1(x)))
        local = self.bn2(self.conv2(local))
        
        # Global pooling path
        g = F.relu(self.gbn(self.gconv(x)))   # (B, gpool_channels, H, W)
        g = g.mean(dim=(2, 3))                # (B, gpool_channels)
        g = self.fc(g).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        
        # Combine: residual connection + global bias
        out = F.relu(local + x + g)
        return out


class PolicyHead(nn.Module):
    """
    Policy head with global pooling bias injection and stability mechanisms.
    
    This head computes both local features and global board context,
    then combines them before producing move logits. The global pooling
    allows the policy to consider whole-board balance when selecting moves.
    
    Stability improvements:
    - Layer normalization before final conv layer to prevent magnitude growth
    - Enhanced monitoring for gradient and weight magnitudes
    - Specialized initialization for the final conv layer
    """
    
    def __init__(self, trunk_channels: int, board_size: int, gpool_channels: int = 16):
        super().__init__()
        self.board_size = board_size
        self.trunk_channels = trunk_channels
        
        # Local conv path
        self.conv1 = nn.Conv2d(trunk_channels, trunk_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(trunk_channels)
        
        # Layer normalization before final conv to prevent magnitude growth
        # This is critical for preventing the systematic value explosion
        self.layer_norm = nn.LayerNorm(trunk_channels, eps=POLICY_HEAD_CONFIG.LAYER_NORM_EPS)
        
        # Local → logits (final conv layer with special handling)
        self.conv2 = nn.Conv2d(trunk_channels, 1, kernel_size=1, bias=False)
        
        # Global pooling bias
        self.gconv = nn.Conv2d(trunk_channels, gpool_channels, kernel_size=1, bias=False)
        self.gbn = nn.BatchNorm2d(gpool_channels)
        self.fc = nn.Linear(gpool_channels, trunk_channels)
        
        # Initialize the final conv layer with conservative scaling
        self._initialize_final_layer()
            
    def _initialize_final_layer(self):
        """
        Initialize the final conv layer with conservative scaling to prevent
        extreme logits during training.
        """
        # Use He initialization with very conservative scaling
        # He initialization is better for ReLU networks
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        
        with torch.no_grad():
            # Scale down weights to prevent extreme logits during training
            # The layer normalization provides the main stability, this is just a safety factor
            self.conv2.weight *= POLICY_HEAD_CONFIG.FINAL_LAYER_SCALING
    
    def _monitor_stability(self, local_features: torch.Tensor, batch_idx: int = None):
        """
        Monitor gradient and weight magnitudes for early detection of instability.
        
        Args:
            local_features: Features before final conv layer
            batch_idx: Current batch index for logging
        """
        # Monitor weight magnitudes in final conv layer
        weight_magnitude = torch.abs(self.conv2.weight).max().item()
        if weight_magnitude > POLICY_HEAD_CONFIG.WEIGHT_MAGNITUDE_WARNING_THRESHOLD:
            print(f"WARNING: Policy head final layer weight magnitude {weight_magnitude:.3f} "
                  f"exceeds threshold {POLICY_HEAD_CONFIG.WEIGHT_MAGNITUDE_WARNING_THRESHOLD}")
        
        # Monitor feature magnitudes before final conv
        feature_magnitude = torch.abs(local_features).max().item()
        if feature_magnitude > 10.0:  # Arbitrary threshold for feature monitoring
            print(f"WARNING: Policy head feature magnitude {feature_magnitude:.3f} "
                  f"is large (batch {batch_idx})")
    
    def forward(self, x: torch.Tensor, batch_idx: int = None) -> torch.Tensor:
        # Local features
        local = F.relu(self.bn1(self.conv1(x)))
        
        # Global pooling path
        g = F.relu(self.gbn(self.gconv(x)))      # (B, gpool_channels, H, W)
        g = g.mean(dim=(2, 3))                   # (B, gpool_channels)
        g = self.fc(g).unsqueeze(-1).unsqueeze(-1)  # (B, trunk_channels, 1, 1)
        
        # Inject global bias
        local = local + g
        
        # Apply layer normalization to prevent magnitude growth
        # Reshape for layer norm: (B, C, H, W) -> (B, H, W, C) -> (B*H*W, C)
        B, C, H, W = local.shape
        local_reshaped = local.permute(0, 2, 3, 1).contiguous().view(-1, C)
        local_normalized = self.layer_norm(local_reshaped)
        # Reshape back: (B*H*W, C) -> (B, H, W, C) -> (B, C, H, W)
        local_normalized = local_normalized.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        # Monitor stability (only in training mode to avoid overhead)
        if self.training and batch_idx is not None and batch_idx % 50 == 0:
            self._monitor_stability(local_normalized, batch_idx)
        
        # Final conv → logits
        p = self.conv2(local_normalized)  # (B, 1, H, W)
        return p.flatten(1)    # (B, H*W)


class ValueHead(nn.Module):
    """
    KataGo-inspired value head with stage conditioning and multi-output ensemble:
    - 1x1 bottleneck (32 ch) -> GAP
    - concat move_stage scalar
    - LayerNorm on pooled vector
    - MLP: FC -> ReLU -> FC -> ReLU
    - K parallel scalars -> learned linear combination -> tanh
    
    This design helps with value head saturation by:
    1. Conditioning on game stage to reduce end-game noise in early positions
    2. Using K parallel outputs with learned combination (cheap ensemble effect)
    3. LayerNorm and extra hidden layer for stability and expressiveness
    4. Initialization to produce ~0 at start (avoid early tanh saturation)
    """
    
    def __init__(self, in_channels: int, bottleneck_channels: int = 32,
                 hidden_dim: int = 256, k_outputs: int = 4):
        super().__init__()
        self.k_outputs = k_outputs
        
        # 1x1 bottleneck convolution
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
        )
        
        # After GAP we concat move_stage -> dim = bottleneck_channels + 1
        self.norm = nn.LayerNorm(bottleneck_channels + 1)
        
        # MLP with two hidden layers
        self.mlp = nn.Sequential(
            nn.Linear(bottleneck_channels + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        
        # K parallel outputs and learned linear combination
        self.out_k = nn.Linear(hidden_dim // 2, k_outputs)  # pre-tanh K scalars
        self.comb = nn.Linear(k_outputs, 1, bias=False)     # learned linear comb
        
        # Initialize to start with ~average and near-zero outputs
        nn.init.constant_(self.out_k.weight, 0.0)
        nn.init.constant_(self.out_k.bias, 0.0)
        with torch.no_grad():
            self.comb.weight.fill_(1.0 / k_outputs)
    
    def forward(self, trunk_feats: torch.Tensor, move_stage: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the value head.
        
        Args:
            trunk_feats: (B, C, H, W) - trunk features
            move_stage: (B,) - normalized move number in [0,1] range
            
        Returns:
            torch.Tensor: (B, 1) - value prediction in [-1,1] range
        """
        # trunk_feats: (B,C,H,W), move_stage: (B,) in [0,1]
        x = self.pre(trunk_feats)         # (B, Bn, H, W)
        x = x.mean(dim=(2, 3))            # GAP -> (B, Bn)
        x = torch.cat([x, move_stage.unsqueeze(1)], dim=1)  # (B, Bn+1)
        x = self.norm(x)
        h = self.mlp(x)                   # (B, hidden/2)
        k_vals = self.out_k(h)            # (B, K)
        v = self.comb(k_vals)             # (B, 1)
        return torch.tanh(v)              # (B, 1)


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
    
    def __init__(self, num_blocks: int = 10, trunk_channels: int = 128):
        super().__init__()
        self.num_blocks = num_blocks
        self.trunk_channels = trunk_channels
        
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
        
        # Policy head with global pooling bias injection
        self.policy_head = PolicyHead(trunk_channels, BOARD_SIZE)
        
        # New KataGo-inspired value head with stage conditioning
        self.value_head = ValueHead(
            in_channels=trunk_channels,
            bottleneck_channels=32,
            hidden_dim=256,
            k_outputs=4,
        )
        
        # Initialize weights using modern best practices
        self._initialize_weights()
    
    
    def _initialize_weights(self):
        """Initialize model weights using modern best practices."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Skip policy head final conv layer - it has special initialization
                if (hasattr(self, 'policy_head') and 
                    hasattr(self.policy_head, 'conv2') and 
                    m is self.policy_head.conv2):
                    continue  # Skip this layer - it's already initialized in PolicyHead
                
                # Kaiming initialization for conv layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize batch norm layers
                # For ResNet v2 stability: zero-init the last BN gamma in each residual block
                # This makes each block start as an identity function
                if self._is_last_bn_in_residual_block(m):
                    nn.init.constant_(m.weight, 0)  # Zero-init for ResNet v2 stability
                else:
                    nn.init.constant_(m.weight, 1)  # Standard init for other BN layers
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                # Initialize layer norm layers
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Policy head final layer initialization is handled in PolicyHead._initialize_final_layer()
        # This ensures proper initialization with the new stability mechanisms
    
    def _is_last_bn_in_residual_block(self, bn_module: nn.BatchNorm2d) -> bool:
        """
        Check if a BatchNorm module is the last BN in a residual block.
        
        For ResNet v2 stability, we zero-initialize the last BN gamma in each
        residual block so that each block starts as an identity function.
        
        Args:
            bn_module: The BatchNorm module to check
            
        Returns:
            True if this is the last BN in a residual block, False otherwise
        """
        # Find the module name to check if it's the last BN in a block
        for name, module in self.named_modules():
            if module is bn_module:
                # Check if this is the second BN in a residual block
                # Pattern: trunk.X.bn2 or similar for the last BN in each block
                if 'trunk' in name and name.endswith('.bn2'):
                    return True
                # Also check for shortcut BNs (though they're less common in our architecture)
                if 'shortcut' in name and name.endswith('.1'):  # shortcut.1 is usually the BN
                    return True
                break
        return False
    
    def forward_shared(self, x: torch.Tensor) -> torch.Tensor:
        """Run the shared trunk up to the penultimate representation."""
        x = F.relu(self.input_bn(self.input_conv(x)))
        trunk_out = self.trunk(x)
        return trunk_out

    def forward(self, x: torch.Tensor, move_stage: torch.Tensor, batch_idx: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the two-headed ResNet.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 13, 13)
            move_stage: Normalized move number tensor of shape (batch_size,) in [0,1] range
            batch_idx: Current batch index for monitoring (optional)
            
        Returns:
            Tuple of (policy_logits, value_signed):
            - policy_logits: Shape (batch_size, 169)
            - value_signed: Shape (batch_size, 1) - Signed value in [-1,1] range (tanh-activated)
        """
        # Shared trunk
        trunk_out = self.forward_shared(x)
        
        # Policy head with global pooling bias injection and stability monitoring
        policy_logits = self.policy_head(trunk_out, batch_idx)
        
        # Value head with stage conditioning
        value_signed = self.value_head(trunk_out, move_stage)
        
        return policy_logits, value_signed

    def get_policy_head_final_layer_params(self):
        """
        Get parameters for the policy head final conv layer with higher weight decay.
        
        Returns:
            List of parameters that should have higher weight decay applied
        """
        return [self.policy_head.conv2.weight]
    
    def get_policy_head_other_params(self):
        """
        Get parameters for the policy head excluding the final conv layer.
        
        Returns:
            List of parameters that should have normal weight decay applied
        """
        other_params = []
        for name, param in self.policy_head.named_parameters():
            if name != 'conv2.weight':  # Exclude final conv layer
                other_params.append(param)
        return other_params

    @torch.no_grad()
    def forward_value_only(self, x: torch.Tensor, move_stage: torch.Tensor) -> torch.Tensor:
        """
        Value-only inference path for faster leaf evaluation.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 13, 13)
            move_stage: Normalized move number tensor of shape (batch_size,) in [0,1] range
            
        Returns:
            torch.Tensor: Value prediction of shape (batch_size, 1) in [-1,1] range
        """
        trunk_out = self.forward_shared(x)
        return self.value_head(trunk_out, move_stage)


def create_model(model_type: str = "katago_inspired", 
                num_blocks: int = 10, trunk_channels: int = 128) -> TwoHeadedResNet:
    """
    Factory function to create a model instance.
    
    Args:
        model_type: Type of model to create (only "katago_inspired" supported)
        num_blocks: Number of residual blocks in the trunk
        trunk_channels: Number of channels in the trunk (constant throughout)
        
    Returns:
        Initialized model instance
    """
    if model_type == "katago_inspired":
        return TwoHeadedResNet(num_blocks=num_blocks, trunk_channels=trunk_channels)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Only 'katago_inspired' is supported. "
            f"Legacy model types are no longer supported. Please update your code to use 'katago_inspired'."
        )


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
    has_trunk_channels = hasattr(model, 'trunk_channels')
    has_num_blocks = hasattr(model, 'num_blocks')
    has_value_head = hasattr(model, 'value_head') and hasattr(model.value_head, 'k_outputs')
    
    if has_trunk_channels and has_num_blocks and has_value_head:
        # New KataGo-inspired architecture with enhanced value head
        gpool_blocks = model.num_blocks // 3
        plain_blocks = model.num_blocks - gpool_blocks
        k_outputs = model.value_head.k_outputs
        
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
- Value Head: Stage-conditioned multi-output ensemble ({k_outputs} outputs) with LayerNorm

Output:
- Policy Logits: (batch_size, 169) - row-major flattened from 13x13
- Value Signed: (batch_size, 1) with tanh activation ([-1,1] range)
- Requires move_stage input: (batch_size,) in [0,1] range
"""
    elif has_trunk_channels and has_num_blocks:
        # KataGo-inspired architecture without enhanced value head
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
- Value Head: Standard with GAP ({VALUE_OUTPUT_SIZE} outputs)

Output:
- Policy Logits: (batch_size, 169) - row-major flattened from 13x13
- Value Signed: (batch_size, 1) with tanh activation ([-1,1] range)
"""
    else:
        # Legacy architecture
        summary = f"""
Model Summary:
==============
Total Parameters: {total_params:,}
Model Type: {model.__class__.__name__} (Legacy)

Architecture:
- Input: (batch_size, 3, 13, 13)
- ResNet Body: 4 stages with {CHANNEL_PROGRESSION} channels (no downsampling)
- Policy Head: Convolutional (1x1 convs) preserving 13x13 spatial structure
- Value Head: Standard with GAP ({VALUE_OUTPUT_SIZE} outputs)

Output:
- Policy Logits: (batch_size, 169) - row-major flattened from 13x13
- Value Signed: (batch_size, 1) with tanh activation ([-1,1] range)
"""
    return summary


def log_cosh_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Log-cosh loss function for value head training.
    
    This loss function is more robust to outliers than MSE and provides
    better gradient behavior for value prediction tasks.
    
    Args:
        pred: Predicted values of shape (batch_size, 1)
        target: Target values of shape (batch_size, 1)
        
    Returns:
        torch.Tensor: Scalar loss value
"""
    x = pred - target
    loss = torch.mean(torch.log(torch.cosh(x + 1e-12)))
    
    # CRITICAL: Check for NaN in log-cosh loss
    if torch.isnan(loss):
        raise RuntimeError(
            f"NaN detected in log_cosh_loss! "
            f"pred range: [{pred.min().item():.6f}, {pred.max().item():.6f}], "
            f"target range: [{target.min().item():.6f}, {target.max().item():.6f}], "
            f"x range: [{x.min().item():.6f}, {x.max().item():.6f}]. "
            f"This indicates numerical instability in the value loss computation."
        )
    
    # Check for extreme values that may lead to numerical issues
    x_max_abs = torch.abs(x).max().item()
    if x_max_abs > MAX_LOG_COSH_INPUT_ABS:  # log(cosh(20)) ≈ 20, beyond this we lose precision
        raise RuntimeError(
            f"Extreme values detected in log_cosh_loss! "
            f"x max_abs: {x_max_abs:.6f}, pred range: [{pred.min().item():.6f}, {pred.max().item():.6f}], "
            f"target range: [{target.min().item():.6f}, {target.max().item():.6f}]. "
            f"These values may cause numerical instability in subsequent computations."
        )
    
    return loss


def compute_value_loss(pred: torch.Tensor, target: torch.Tensor, 
                      smooth: float = 0.95) -> torch.Tensor:
    """
    Compute value loss with label smoothing using log-cosh loss.
    
    Args:
        pred: Predicted values of shape (batch_size, 1) in [-1,1] range
        target: Target values of shape (batch_size, 1) in [-1,1] range
        smooth: Label smoothing factor (0.95 means clamp to [-0.95, 0.95])
        
    Returns:
        torch.Tensor: Scalar loss value
    """
    # Apply label smoothing: clamp target values to reduce noise
    z = torch.clamp(target, -smooth, smooth)
    return log_cosh_loss(pred, z)


def compute_move_stage(board: torch.Tensor) -> torch.Tensor:
    """
    Compute move_stage from board state for the new value head.
    
    Move stage represents the normalized number of stones on the board,
    ranging from 0.0 (empty board) to 1.0 (full board).
    
    Args:
        board: Board tensor of shape (batch_size, 3, height, width) or (3, height, width)
               where channels are [blue_channel, red_channel, player_channel]
               
    Returns:
        torch.Tensor: Move stage tensor of shape (batch_size,) in [0,1] range
    """
    # Handle single board case
    if board.dim() == 3:
        board = board.unsqueeze(0)
    
    # Binarize each plane before counting (matches data pipeline logic)
    blue = (board[:, 0] > 0).float()
    red = (board[:, 1] > 0).float()
    stones_on_board = (blue + red).sum(dim=(1, 2))
    
    # Normalize by total board size
    board_area = board.shape[2] * board.shape[3]  # height * width
    move_stage = stones_on_board / float(board_area)
    
    return move_stage


def is_new_architecture(model: nn.Module) -> bool:
    """
    Check if a model uses the new KataGo-inspired architecture.
    
    Args:
        model: PyTorch model to check
        
    Returns:
        bool: True if model uses new architecture, False otherwise
    """
    return (hasattr(model, 'value_head') and 
            hasattr(model.value_head, 'k_outputs') and
            hasattr(model, 'trunk_channels') and
            hasattr(model, 'num_blocks'))




def monitor_policy_head_gradients(model: nn.Module, batch_idx: int = None):
    """
    Monitor gradient norms in the policy head for early detection of instability.
    
    Args:
        model: The neural network model
        batch_idx: Current batch index for logging
    """
    if not hasattr(model, 'policy_head') or not hasattr(model.policy_head, 'conv2'):
        return
    
    # Check if gradients exist
    if model.policy_head.conv2.weight.grad is None:
        return
    
    # Compute gradient norm for policy head final layer
    grad_norm = torch.norm(model.policy_head.conv2.weight.grad).item()
    
    if grad_norm > POLICY_HEAD_CONFIG.GRADIENT_NORM_WARNING_THRESHOLD:
        print(f"WARNING: Policy head final layer gradient norm {grad_norm:.3f} "
              f"exceeds threshold {POLICY_HEAD_CONFIG.GRADIENT_NORM_WARNING_THRESHOLD} "
              f"(batch {batch_idx})")
    
    # Also monitor weight magnitude
    weight_magnitude = torch.abs(model.policy_head.conv2.weight).max().item()
    if weight_magnitude > POLICY_HEAD_CONFIG.WEIGHT_MAGNITUDE_WARNING_THRESHOLD:
        print(f"WARNING: Policy head final layer weight magnitude {weight_magnitude:.3f} "
              f"exceeds threshold {POLICY_HEAD_CONFIG.WEIGHT_MAGNITUDE_WARNING_THRESHOLD} "
              f"(batch {batch_idx})") 