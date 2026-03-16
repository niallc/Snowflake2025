"""
Model architecture for the Hex AI.

This module contains the neural network architectures used for the Hex AI,
including the main TwoHeadedResNet model and supporting components.

The architecture follows a two-headed design:
- Policy head: Predicts move probabilities for each board position
- Value head: Predicts a signed win value from the current position

Active architecture design references:
- write_ups/Current_Network_vs_KataGo_Gumbel_2026-03-13.md
- write_ups/Training_Architecture_Change_Discussion_2026-03-13.md
- docs/value_head_specification.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple

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
    # TODO: Revisit this value, it may be way too strong from debugging exploding policy values.
    FINAL_LAYER_SCALING = 0.1
    """Scaling factor for policy head final conv layer initialization.
    Standard practice for policy networks to prevent extreme initial logits."""
    
    # Weight decay
    # TODO: Revisit this value, it may be way too strong from debugging exploding policy values.
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
    LAYER_NORM_EPS = 5e-7
    """Epsilon for layer normalization numerical stability.
    Standard value to prevent division by zero."""

# Global instance for easy access
POLICY_HEAD_CONFIG = PolicyHeadStabilityConfig()


def _mean_max_pool2d(x: torch.Tensor) -> torch.Tensor:
    """Return concatenated mean/max pooled channel statistics."""
    mean_pooled = x.mean(dim=(2, 3))
    max_pooled = torch.amax(x, dim=(2, 3))
    return torch.cat([mean_pooled, max_pooled], dim=1)


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


class BottleneckResNetBlock(nn.Module):
    """
    Constant-width residual block with a narrower internal bottleneck.

    This keeps full board resolution while reducing the expensive 3x3 compute,
    which is the main efficiency lesson we want to test from the Gumbel appendix.
    """

    def __init__(self, channels: int, bottleneck_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, bottleneck_channels, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(
            bottleneck_channels, channels, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += x
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
    
    Key stability improvements:
    - Learnable gate g_alpha starts at 0, making the block identity at initialization
    - No ReLU before global pooling to keep pooled vector zero-mean
    - BN in global path ensures zero-mean at initialization
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
        
        # NEW: learnable gate, starts closed (initialized to 0)
        self.g_alpha = nn.Parameter(torch.zeros(1))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local path
        local = F.relu(self.bn1(self.conv1(x)))
        local = self.bn2(self.conv2(local))
        
        # Global path (NOTE: no ReLU here to keep zero-mean)
        g = self.gbn(self.gconv(x))                 # (B, gpool_channels, H, W)
        g = g.mean(dim=(2, 3))                      # (B, gpool_channels)
        g = self.fc(g).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        
        # Combine: residual + gated global bias
        out = F.relu(x + local + self.g_alpha * g)
        return out


class BottleneckGlobalPoolingResidualBlock(nn.Module):
    """
    Gpool residual block with a bottlenecked local path.

    The local path is compute-reduced while the global bias path stays aligned
    with the current live architecture so we change one main variable at a time.
    """

    def __init__(
        self,
        channels: int,
        bottleneck_channels: int,
        gpool_channels: int = 16,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels, bottleneck_channels, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(
            bottleneck_channels, channels, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(channels)

        self.gconv = nn.Conv2d(channels, gpool_channels, kernel_size=1, bias=False)
        self.gbn = nn.BatchNorm2d(gpool_channels)
        self.fc = nn.Linear(gpool_channels, channels)
        self.g_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = F.relu(self.bn1(self.conv1(x)))
        local = F.relu(self.bn2(self.conv2(local)))
        local = self.bn3(self.conv3(local))

        g = self.gbn(self.gconv(x))
        g = g.mean(dim=(2, 3))
        g = self.fc(g).unsqueeze(-1).unsqueeze(-1)

        out = F.relu(x + local + self.g_alpha * g)
        return out


class BottleneckPooledBiasResidualBlock(nn.Module):
    """
    Bottlenecked pooled-bias residual block using mean+max global statistics.

    This keeps the block board-size-friendly and cheaper than a full 3x3
    residual block while still injecting whole-board context.
    """

    def __init__(self, channels: int, bottleneck_channels: int):
        super().__init__()
        self.mix1_a = nn.Conv2d(
            channels, bottleneck_channels, kernel_size=1, bias=False
        )
        self.bn_a = nn.BatchNorm2d(bottleneck_channels)
        self.mix1_b = nn.Conv2d(
            channels, bottleneck_channels, kernel_size=1, bias=False
        )
        self.bn_b = nn.BatchNorm2d(bottleneck_channels)
        self.fc = nn.Linear(bottleneck_channels * 2, bottleneck_channels)
        self.mix1_out = nn.Conv2d(
            bottleneck_channels, channels, kernel_size=1, bias=False
        )
        self.bn_out = nn.BatchNorm2d(channels)
        self.g_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.bn_a(self.mix1_a(x))
        b = self.bn_b(self.mix1_b(x))
        bias = self.fc(_mean_max_pool2d(b)).unsqueeze(-1).unsqueeze(-1)
        out = F.relu(a + self.g_alpha * bias)
        out = self.bn_out(self.mix1_out(out))
        return F.relu(x + out)


class PolicyHeadBase(nn.Module):
    """Shared policy-head stability helpers."""

    def _initialize_final_layer(self):
        """
        Initialize the final conv layer with conservative scaling to prevent
        extreme logits during training.
        """
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        with torch.no_grad():
            self.conv2.weight *= POLICY_HEAD_CONFIG.FINAL_LAYER_SCALING

    def _monitor_stability(self, local_features: torch.Tensor, batch_idx: int = None):
        """
        Monitor gradient and weight magnitudes for early detection of instability.

        Args:
            local_features: Features before final conv layer
            batch_idx: Current batch index for logging
        """
        weight_magnitude = torch.abs(self.conv2.weight).max().item()
        if weight_magnitude > POLICY_HEAD_CONFIG.WEIGHT_MAGNITUDE_WARNING_THRESHOLD:
            print(f"WARNING: Policy head final layer weight magnitude {weight_magnitude:.3f} "
                  f"exceeds threshold {POLICY_HEAD_CONFIG.WEIGHT_MAGNITUDE_WARNING_THRESHOLD}")

        feature_magnitude = torch.abs(local_features).max().item()
        if feature_magnitude > 10.0:
            print(f"WARNING: Policy head feature magnitude {feature_magnitude:.3f} "
                  f"is large (batch {batch_idx})")


class PolicyHead(PolicyHeadBase):
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
        
        # Learnable gate for global bias, starts closed (initialized to 0)
        self.g_alpha = nn.Parameter(torch.zeros(1))
        
        # Initialize the final conv layer with conservative scaling
        self._initialize_final_layer()

    def forward(self, x: torch.Tensor, batch_idx: int = None) -> torch.Tensor:
        # Local features
        local = F.relu(self.bn1(self.conv1(x)))
        
        # Global pooling path
        g = F.relu(self.gbn(self.gconv(x)))      # (B, gpool_channels, H, W)
        g = g.mean(dim=(2, 3))                   # (B, gpool_channels)
        g = self.fc(g).unsqueeze(-1).unsqueeze(-1)  # (B, trunk_channels, 1, 1)
        
        # Inject gated global bias
        local = local + self.g_alpha * g
        
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


class PooledBiasPolicyHead(PolicyHeadBase):
    """Lightweight 1x1 pooled-bias policy head using mean+max statistics."""

    def __init__(
        self,
        trunk_channels: int,
        board_size: int,
        policy_channels: int = 96,
        gpool_channels: int = 32,
    ):
        super().__init__()
        self.board_size = board_size
        self.trunk_channels = trunk_channels
        self.policy_channels = policy_channels
        self.gpool_channels = gpool_channels

        self.conv1 = nn.Conv2d(
            trunk_channels, policy_channels, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(policy_channels)
        self.gconv = nn.Conv2d(
            trunk_channels, gpool_channels, kernel_size=1, bias=False
        )
        self.gbn = nn.BatchNorm2d(gpool_channels)
        self.fc = nn.Linear(gpool_channels * 2, policy_channels)
        self.combine_bn = nn.BatchNorm2d(policy_channels)
        self.conv2 = nn.Conv2d(policy_channels, 1, kernel_size=1, bias=False)
        self.g_alpha = nn.Parameter(torch.zeros(1))

        self._initialize_final_layer()

    def forward(self, x: torch.Tensor, batch_idx: int = None) -> torch.Tensor:
        local = self.bn1(self.conv1(x))

        g = self.gbn(self.gconv(x))
        bias = self.fc(_mean_max_pool2d(g)).unsqueeze(-1).unsqueeze(-1)

        local = F.relu(self.combine_bn(local + self.g_alpha * bias))

        if self.training and batch_idx is not None and batch_idx % 50 == 0:
            self._monitor_stability(local, batch_idx)

        return self.conv2(local).flatten(1)


class ValueHead(nn.Module):
    """
    KataGo-inspired value head with stage conditioning and multi-output ensemble:
    - 1x1 bottleneck (32 ch) -> pooled board statistics
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
    
    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int = 32,
        hidden_dim: int = 256,
        k_outputs: int = 4,
        pool_mode: str = "mean",
    ):
        super().__init__()
        self.k_outputs = k_outputs
        self.pool_mode = pool_mode
        if self.pool_mode not in {"mean", "mean_max"}:
            raise ValueError(
                f"Unsupported value-head pool_mode {self.pool_mode!r}. "
                "Expected 'mean' or 'mean_max'."
            )
        pooled_dim = bottleneck_channels
        if self.pool_mode == "mean_max":
            pooled_dim *= 2
        
        # 1x1 bottleneck convolution
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
        )
        
        # After pooling we concat move_stage.
        self.norm = nn.LayerNorm(pooled_dim + 1)
        
        # MLP with two hidden layers
        self.mlp = nn.Sequential(
            nn.Linear(pooled_dim + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        
        # K parallel outputs and learned linear combination
        self.out_k = nn.Linear(hidden_dim // 2, k_outputs)  # pre-tanh K scalars
        self.comb = nn.Linear(k_outputs, 1, bias=False)     # learned linear comb
        self.reset_output_layer_initialization()

    def reset_output_layer_initialization(self) -> None:
        """Restore the intended near-neutral value-output initialization."""
        nn.init.constant_(self.out_k.weight, 0.0)
        nn.init.constant_(self.out_k.bias, 0.0)
        with torch.no_grad():
            self.comb.weight.fill_(1.0 / self.k_outputs)
    
    def forward(self, trunk_feats: torch.Tensor, move_stage: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the value head.
        
        Args:
            trunk_feats: (B, C, H, W) - trunk features
            move_stage: (B,) - normalized move number in [0,1] range
            
        Returns:
            torch.Tensor: (B, 1) - value prediction in [-1,1] range
        """
        x = self.pre(trunk_feats)
        if self.pool_mode == "mean_max":
            x = _mean_max_pool2d(x)
        else:
            x = x.mean(dim=(2, 3))
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
    - Policy head: Predicts move probabilities over the configured board area
    - Value head: Predicts Red-referenced signed value (1 output)
    
    Red wins are labeled as 1.0 in training, and the runtime value head emits a
    signed value in [-1, 1] with tanh activation. Convert at API boundaries if
    a probability is needed.
    
    Key improvements from KataGo:
    - Flattened channel progression (constant trunk_channels instead of growing)
    - Mixed plain and global pooling residual blocks
    - Policy head with global pooling bias injection
    - Enhanced value head with hidden layer and optional bottleneck
    """
    
    def __init__(
        self,
        num_blocks: int = 7,
        trunk_channels: int = 128,
        board_size: int = BOARD_SIZE,
    ):
        super().__init__()
        self.model_type = "katago_inspired"
        self.num_blocks = num_blocks
        self.trunk_channels = trunk_channels
        self.board_size = int(board_size)
        if self.board_size <= 0:
            raise ValueError(f"board_size must be positive, got {board_size}")
        self.global_block_indices = tuple(i for i in range(num_blocks) if i % 3 == 2)
        
        # Input layer: Convert board representation to initial features
        # Input shape: (batch_size, 3, board_size, board_size) for two players
        # plus player-to-move channel.
        self.input_conv = nn.Conv2d(3, trunk_channels, 
                                   kernel_size=5, stride=1, padding=2, bias=False)
        self.input_bn = nn.BatchNorm2d(trunk_channels)
        
        # Trunk: mix plain and gpool blocks with constant channel count
        blocks = []
        for i in range(num_blocks):
            if i in self.global_block_indices:
                blocks.append(GlobalPoolingResidualBlock(trunk_channels))
            else:
                blocks.append(ResNetBlock(trunk_channels, trunk_channels))
        self.trunk = nn.Sequential(*blocks)
        
        # Policy head with global pooling bias injection
        self.policy_head = PolicyHead(trunk_channels, self.board_size)
        
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

        if hasattr(self, "value_head") and hasattr(self.value_head, "reset_output_layer_initialization"):
            self.value_head.reset_output_layer_initialization()

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
                # Pattern: trunk.X.bn2 / trunk.X.bn3 for the last BN in each block
                if 'trunk' in name and (
                    name.endswith('.bn2')
                    or name.endswith('.bn3')
                    or name.endswith('.bn_out')
                ):
                    return True
                # Also check for shortcut BNs (though they're less common in our architecture)
                if 'shortcut' in name and name.endswith('.1'):  # shortcut.1 is usually the BN
                    return True
                break
        return False
    
    def forward_shared(self, x: torch.Tensor) -> torch.Tensor:
        """Run the shared trunk up to the penultimate representation."""
        board_rows = int(x.shape[-2])
        board_cols = int(x.shape[-1])
        if board_rows != board_cols:
            raise ValueError(
                f"Expected square boards, got shape {tuple(x.shape)}"
            )
        if board_rows != self.board_size:
            raise ValueError(
                f"Input board size {board_rows} does not match model board_size "
                f"{self.board_size}"
            )
        x = F.relu(self.input_bn(self.input_conv(x)))
        trunk_out = self.trunk(x)
        return trunk_out

    def forward(self, x: torch.Tensor, move_stage: torch.Tensor, batch_idx: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the two-headed ResNet.
        
        Args:
            x: Input tensor of shape (batch_size, 3, board_size, board_size)
            move_stage: Normalized move number tensor of shape (batch_size,) in [0,1] range
            batch_idx: Current batch index for monitoring (optional)
            
        Returns:
            Tuple of (policy_logits, value_signed):
            - policy_logits: Shape (batch_size, board_size * board_size)
            - value_signed: Shape (batch_size, 1) - Signed value in [-1,1] range (tanh-activated)
        """
        # Shared trunk
        trunk_out = self.forward_shared(x)
        
        # Policy head with global pooling bias injection and stability monitoring
        policy_logits = self.policy_head(trunk_out, batch_idx)
        
        # Value head with stage conditioning
        value_signed = self.value_head(trunk_out, move_stage)
        
        return policy_logits, value_signed

    def forward_from_boards(
        self,
        boards: torch.Tensor,
        *,
        move_stage: torch.Tensor | None = None,
        batch_idx: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the family-standard forward pass from board tensors."""
        if move_stage is None:
            move_stage = compute_move_stage(boards)
        return self(boards, move_stage, batch_idx=batch_idx)

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

    def _separate_params_by_weight_decay(
        self,
        params: List[nn.Parameter],
    ) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """
        Split parameters into weight-decayed and exempt groups.

        Keeping this model-owned avoids trainer coupling to exact head/trunk
        structure when new families are introduced.
        """
        parameter_names = {id(param): name for name, param in self.named_parameters()}
        weight_decay_params: List[nn.Parameter] = []
        no_weight_decay_params: List[nn.Parameter] = []

        for param in params:
            param_name = parameter_names.get(id(param))
            if param_name is None:
                raise RuntimeError(
                    f"CRITICAL BUG: Could not find parameter name for parameter {param}. "
                    "All optimizer-group parameters must map back to named_parameters()."
                )

            if any(
                norm_type in param_name
                for norm_type in ["bn", "norm", "batch_norm", "layer_norm"]
            ) or param_name.endswith(".bias"):
                no_weight_decay_params.append(param)
            else:
                weight_decay_params.append(param)

        return weight_decay_params, no_weight_decay_params

    def build_optimizer_param_groups(
        self,
        *,
        learning_rate: float,
        weight_decay: float,
        value_learning_rate_factor: float,
        value_weight_decay_factor: float,
    ) -> List[Dict[str, Any]]:
        """
        Build optimizer parameter groups for this model family.
        """
        policy_final_params = self.get_policy_head_final_layer_params()
        policy_other_params = self.get_policy_head_other_params()

        value_head_params = list(self.value_head.parameters())
        excluded_param_ids = {
            id(param)
            for param in policy_final_params + policy_other_params + value_head_params
        }
        trunk_params = [
            param for param in self.parameters() if id(param) not in excluded_param_ids
        ]

        trunk_weight_decay, trunk_no_weight_decay = self._separate_params_by_weight_decay(
            trunk_params
        )
        (
            policy_other_weight_decay,
            policy_other_no_weight_decay,
        ) = self._separate_params_by_weight_decay(policy_other_params)
        (
            policy_final_weight_decay,
            policy_final_no_weight_decay,
        ) = self._separate_params_by_weight_decay(policy_final_params)
        (
            value_head_weight_decay,
            value_head_no_weight_decay,
        ) = self._separate_params_by_weight_decay(value_head_params)

        param_groups: List[Dict[str, Any]] = []

        if trunk_weight_decay:
            param_groups.append(
                {
                    "params": trunk_weight_decay,
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                }
            )
        if trunk_no_weight_decay:
            param_groups.append(
                {
                    "params": trunk_no_weight_decay,
                    "lr": learning_rate,
                    "weight_decay": 0.0,
                }
            )
        if policy_other_weight_decay:
            param_groups.append(
                {
                    "params": policy_other_weight_decay,
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                }
            )
        if policy_other_no_weight_decay:
            param_groups.append(
                {
                    "params": policy_other_no_weight_decay,
                    "lr": learning_rate,
                    "weight_decay": 0.0,
                }
            )
        if policy_final_weight_decay:
            param_groups.append(
                {
                    "params": policy_final_weight_decay,
                    "lr": learning_rate,
                    "weight_decay": weight_decay * 2.0,
                }
            )
        if policy_final_no_weight_decay:
            param_groups.append(
                {
                    "params": policy_final_no_weight_decay,
                    "lr": learning_rate,
                    "weight_decay": 0.0,
                }
            )
        if value_head_weight_decay:
            param_groups.append(
                {
                    "params": value_head_weight_decay,
                    "lr": learning_rate * value_learning_rate_factor,
                    "weight_decay": weight_decay * value_weight_decay_factor,
                }
            )
        if value_head_no_weight_decay:
            param_groups.append(
                {
                    "params": value_head_no_weight_decay,
                    "lr": learning_rate * value_learning_rate_factor,
                    "weight_decay": 0.0,
                }
            )

        return param_groups

    @torch.no_grad()
    def forward_value_only(self, x: torch.Tensor, move_stage: torch.Tensor) -> torch.Tensor:
        """
        Value-only inference path for faster leaf evaluation.
        
        Args:
            x: Input tensor of shape (batch_size, 3, board_size, board_size)
            move_stage: Normalized move number tensor of shape (batch_size,) in [0,1] range
            
        Returns:
            torch.Tensor: Value prediction of shape (batch_size, 1) in [-1,1] range
        """
        trunk_out = self.forward_shared(x)
        return self.value_head(trunk_out, move_stage)

    @torch.no_grad()
    def forward_value_only_from_boards(
        self,
        boards: torch.Tensor,
        *,
        move_stage: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run value-only inference from raw board tensors."""
        if move_stage is None:
            move_stage = compute_move_stage(boards)
        return self.forward_value_only(boards, move_stage)


class TwoHeadedBottleneckResNet(TwoHeadedResNet):
    """
    Bottleneck-trunk variant of the current KataGo-inspired family.

    It keeps the same stem, heads, full board resolution, and periodic global
    pooling pattern, but makes the trunk blocks cheaper so we can test deeper
    or wider challengers without paying the full cost of plain full-width blocks.
    """

    def __init__(
        self,
        num_blocks: int = 9,
        trunk_channels: int = 128,
        board_size: int = BOARD_SIZE,
    ):
        super().__init__(
            num_blocks=num_blocks,
            trunk_channels=trunk_channels,
            board_size=board_size,
        )
        self.model_type = "katago_bottleneck"
        self.bottleneck_channels = max(1, trunk_channels // 2)

        blocks = []
        for i in range(num_blocks):
            if i in self.global_block_indices:
                blocks.append(
                    BottleneckGlobalPoolingResidualBlock(
                        trunk_channels,
                        bottleneck_channels=self.bottleneck_channels,
                    )
                )
            else:
                blocks.append(
                    BottleneckResNetBlock(
                        trunk_channels,
                        bottleneck_channels=self.bottleneck_channels,
                    )
                )
        self.trunk = nn.Sequential(*blocks)
        self._initialize_weights()


class TwoHeadedBottleneckPoolResNet(TwoHeadedResNet):
    """
    Bottleneck family with sparse pooled-bias trunk blocks and cheaper pooled heads.
    """

    def __init__(
        self,
        num_blocks: int = 13,
        trunk_channels: int = 224,
        board_size: int = BOARD_SIZE,
    ):
        if num_blocks < 9:
            raise ValueError(
                "katago_bottleneck_pool requires num_blocks >= 9 so pooled-bias "
                "blocks can remain at 1-indexed positions 4 and 9."
            )
        super().__init__(
            num_blocks=num_blocks,
            trunk_channels=trunk_channels,
            board_size=board_size,
        )
        self.model_type = "katago_bottleneck_pool"
        self.bottleneck_channels = max(1, trunk_channels // 2)
        self.global_block_indices = (3, 8)

        blocks = []
        for i in range(num_blocks):
            if i in self.global_block_indices:
                blocks.append(
                    BottleneckPooledBiasResidualBlock(
                        trunk_channels,
                        bottleneck_channels=self.bottleneck_channels,
                    )
                )
            else:
                blocks.append(
                    BottleneckResNetBlock(
                        trunk_channels,
                        bottleneck_channels=self.bottleneck_channels,
                    )
                )
        self.trunk = nn.Sequential(*blocks)
        self.policy_head = PooledBiasPolicyHead(
            trunk_channels,
            self.board_size,
            policy_channels=96,
            gpool_channels=32,
        )
        self.value_head = ValueHead(
            in_channels=trunk_channels,
            bottleneck_channels=32,
            hidden_dim=256,
            k_outputs=4,
            pool_mode="mean_max",
        )
        self._initialize_weights()


class TwoHeadedBottleneckPool3x3PolicyResNet(TwoHeadedBottleneckPoolResNet):
    """
    Bottleneck-pool trunk/value family with the older richer 3x3 policy head.

    This is a practical head-only A/B against the lighter pooled-bias head while
    keeping the same trunk block layout and value-head semantics.
    """

    def __init__(
        self,
        num_blocks: int = 13,
        trunk_channels: int = 224,
        board_size: int = BOARD_SIZE,
    ):
        super().__init__(
            num_blocks=num_blocks,
            trunk_channels=trunk_channels,
            board_size=board_size,
        )
        self.model_type = "katago_bottleneck_pool_3x3_policy"
        self.policy_head = PolicyHead(trunk_channels, self.board_size)
        self._initialize_weights()


def create_model(
    model_type: str = "katago_inspired",
    num_blocks: int | None = None,
    trunk_channels: int | None = None,
    board_size: int = BOARD_SIZE,
) -> nn.Module:
    """
    Factory function to create a model instance.
    
    Args:
        model_type: Type of model to create
        num_blocks: Number of residual blocks in the trunk. Uses family defaults
            when omitted.
        trunk_channels: Number of channels in the trunk. Uses family defaults
            when omitted.
        board_size: Board size the model is configured to accept.
        
    Returns:
        Initialized model instance
    """
    model_defaults = {
        "katago_inspired": (7, 128),
        "katago_bottleneck": (9, 128),
        "katago_bottleneck_pool": (13, 224),
        "katago_bottleneck_pool_3x3_policy": (13, 224),
    }
    if model_type not in model_defaults:
        raise ValueError(
            f"Unknown model type: {model_type}. Supported model types are "
            "'katago_inspired', 'katago_bottleneck', 'katago_bottleneck_pool', "
            "and 'katago_bottleneck_pool_3x3_policy'."
        )

    default_blocks, default_trunk_channels = model_defaults[model_type]
    if num_blocks is None:
        num_blocks = default_blocks
    if trunk_channels is None:
        trunk_channels = default_trunk_channels

    if model_type == "katago_inspired":
        return TwoHeadedResNet(
            num_blocks=num_blocks,
            trunk_channels=trunk_channels,
            board_size=board_size,
        )
    if model_type == "katago_bottleneck":
        return TwoHeadedBottleneckResNet(
            num_blocks=num_blocks,
            trunk_channels=trunk_channels,
            board_size=board_size,
        )
    if model_type == "katago_bottleneck_pool_3x3_policy":
        return TwoHeadedBottleneckPool3x3PolicyResNet(
            num_blocks=num_blocks,
            trunk_channels=trunk_channels,
            board_size=board_size,
        )
    return TwoHeadedBottleneckPoolResNet(
        num_blocks=num_blocks,
        trunk_channels=trunk_channels,
        board_size=board_size,
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
        board_size = int(getattr(model, "board_size", BOARD_SIZE))
        policy_output_size = board_size * board_size
        global_block_indices = tuple(getattr(model, "global_block_indices", ()))
        global_block_count = len(global_block_indices)
        plain_blocks = model.num_blocks - global_block_count
        k_outputs = model.value_head.k_outputs
        bottleneck_channels = getattr(model, "bottleneck_channels", None)
        model_variant = f"{model.__class__.__name__} ({getattr(model, 'model_type', 'unknown')})"
        trunk_line = (
            f"- Trunk: {model.num_blocks} blocks with constant {model.trunk_channels} channels "
            f"and {bottleneck_channels} bottleneck channels"
            if bottleneck_channels is not None
            else f"- Trunk: {model.num_blocks} blocks with constant {model.trunk_channels} channels"
        )
        value_pool_mode = getattr(model.value_head, "pool_mode", "mean")
        value_pool_label = "mean + max" if value_pool_mode == "mean_max" else "global mean"
        if global_block_indices:
            global_block_label = (
                f"{global_block_count} pooled-bias blocks at 1-indexed positions "
                f"{[idx + 1 for idx in global_block_indices]}"
                if getattr(model, "model_type", "") == "katago_bottleneck_pool"
                else f"{global_block_count} global pooling blocks (every 3rd block)"
            )
        else:
            global_block_label = "0 global-context blocks"
        policy_head = getattr(model, "policy_head", None)
        if isinstance(policy_head, PooledBiasPolicyHead):
            policy_head_label = "1x1 pooled-bias policy head"
        elif isinstance(policy_head, PolicyHead):
            policy_head_label = "3x3 + global-bias + LayerNorm policy head"
        else:
            policy_head_label = f"{type(policy_head).__name__} policy head"
        spatial_shape = f"{board_size}x{board_size} (configured model size)"
        summary = f"""
Model Summary:
==============
Total Parameters: {total_params:,}
Model Type: {model_variant}

Architecture:
- Input: (batch_size, 3, {board_size}, {board_size}) [{spatial_shape}]
{trunk_line}
  * {plain_blocks} plain ResNet blocks
  * {global_block_label}
- Policy Head: {policy_head_label}
- Value Head: Stage-conditioned multi-output ensemble ({k_outputs} outputs) with {value_pool_label} pooling

Output:
- Policy Logits: (batch_size, {policy_output_size}) - row-major flattened from configured board size
- Value Signed: (batch_size, 1) with tanh activation ([-1,1] range)
- Requires move_stage input: (batch_size,) in [0,1] range
"""
    elif has_trunk_channels and has_num_blocks:
        # KataGo-inspired architecture without enhanced value head
        board_size = int(getattr(model, "board_size", BOARD_SIZE))
        policy_output_size = board_size * board_size
        global_block_indices = tuple(getattr(model, "global_block_indices", ()))
        global_block_count = len(global_block_indices)
        plain_blocks = model.num_blocks - global_block_count
        bottleneck_channels = getattr(model, "bottleneck_channels", None)
        model_variant = f"{model.__class__.__name__} ({getattr(model, 'model_type', 'unknown')})"
        trunk_line = (
            f"- Trunk: {model.num_blocks} blocks with constant {model.trunk_channels} channels "
            f"and {bottleneck_channels} bottleneck channels"
            if bottleneck_channels is not None
            else f"- Trunk: {model.num_blocks} blocks with constant {model.trunk_channels} channels"
        )
        spatial_shape = f"{board_size}x{board_size} (configured model size)"
        summary = f"""
Model Summary:
==============
Total Parameters: {total_params:,}
Model Type: {model_variant}

Architecture:
- Input: (batch_size, 3, {board_size}, {board_size}) [{spatial_shape}]
{trunk_line}
  * {plain_blocks} plain ResNet blocks
  * {global_block_count} global pooling blocks (every 3rd block)
- Policy Head: Global pooling bias injection preserving board spatial structure
- Value Head: Standard with GAP ({VALUE_OUTPUT_SIZE} outputs)

Output:
- Policy Logits: (batch_size, {policy_output_size}) - row-major flattened from configured board size
- Value Signed: (batch_size, 1) with tanh activation ([-1,1] range)
"""
    else:
        # Legacy architecture
        board_size = int(getattr(model, "board_size", BOARD_SIZE))
        policy_output_size = board_size * board_size
        spatial_shape = f"{board_size}x{board_size} (configured model size)"
        summary = f"""
Model Summary:
==============
Total Parameters: {total_params:,}
Model Type: {model.__class__.__name__} (Legacy)

Architecture:
- Input: (batch_size, 3, {board_size}, {board_size}) [{spatial_shape}]
- ResNet Body: 4 stages with {CHANNEL_PROGRESSION} channels (no downsampling)
- Policy Head: Convolutional (1x1 convs) preserving board spatial structure
- Value Head: Standard with GAP ({VALUE_OUTPUT_SIZE} outputs)

Output:
- Policy Logits: (batch_size, {policy_output_size}) - row-major flattened from configured board size
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
    move_stage = stones_on_board / board_area
    
    return move_stage


def is_new_architecture(model: nn.Module) -> bool:
    """
    Check whether a model implements the current multi-architecture runtime API.
    
    Args:
        model: PyTorch model to check
        
    Returns:
        bool: True if the model exposes the required runtime helpers, False otherwise
    """
    return all(
        callable(getattr(model, method_name, None))
        for method_name in (
            "forward_from_boards",
            "forward_value_only_from_boards",
            "build_optimizer_param_groups",
        )
    )
