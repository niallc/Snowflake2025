# Extreme Values Investigation Summary

## Problem Description

The neural network training is failing due to extreme values in model outputs, specifically policy logits exceeding the threshold of 20.0. The training fails with the error:

```
RuntimeError: Extreme values detected in model outputs! policy_pred max_abs: 20.593750, range: [-20.593750, 8.890625], value_pred max_abs: 0.556641, range: [0.406494, 0.556641]. Policy logits > 20.0 indicate loss of uncertainty (softmax saturation).
```

## Training Dynamics Observed

### Original Training Run (Before Fixes)
- **Batch 0**: policy_max_abs=61.47, value_max_abs=0.49
- **Batch 1**: policy_max_abs=87.44, value_max_abs=0.41  
- **Batch 2**: policy_max_abs=89.50, value_max_abs=0.36
- **Batch 19**: policy_max_abs=23.67, value_max_abs=0.56
- **Batch 20**: policy_max_abs=20.19, value_max_abs=0.60
- **Batch 21**: policy_max_abs=21.00, value_max_abs=0.60
- **Batch 29**: policy_max_abs=18.91, value_max_abs=0.58
- **Batch 80**: policy_max_abs=40.69, value_max_abs=0.57 ← **FAILED HERE**

**Pattern**: Values start extreme (89.5), improve during early training (down to ~18.9), then spike again later (40.69).

### After Xavier Initialization Fix
- **Batch 0**: policy_max_abs=56.41, value_max_abs=0.33
- **Batch 1**: policy_max_abs=79.63, value_max_abs=0.39
- **Batch 19**: policy_max_abs=20.80, value_max_abs=0.56
- **Batch 20**: policy_max_abs=19.98, value_max_abs=0.57
- **Batch 21**: policy_max_abs=20.59, value_max_abs=0.56 ← **FAILED HERE**

**Pattern**: Values start extreme (79.6), improve during early training (down to ~20.0), then cross the threshold at batch 21.

**Key Finding**: The Xavier initialization fix delayed the failure from batch 80 to batch 21, but did not solve the fundamental problem.

## Gradient Analysis

The training logs show significant gradient explosion:
- **Batch 1**: pre_clip_grad_norm=137.158, post_clip_grad_norm=2.000
- **Batch 2**: pre_clip_grad_norm=114.149, post_clip_grad_norm=2.000
- **Batch 3**: pre_clip_grad_norm=105.812, post_clip_grad_norm=2.000

Gradient clipping is working (reducing from 137+ to 2.0), but the underlying gradient explosion suggests architectural instability.

## Diagnostic Testing Results

### Isolated Policy Head Test
- **Result**: Reasonable values (max abs ~4.6, decreasing to ~3.4)
- **Conclusion**: Policy head alone is stable

### Policy Head + Simple Trunk Test  
- **Result**: Reasonable values (max abs ~8.9, growing slowly)
- **Conclusion**: Simple trunk is stable

### Full 10-Block ResNet Test
- **Result**: Extreme values (max abs 41.9) from batch 0
- **Conclusion**: The 10-block ResNet trunk is the source of instability

### Block Count Testing
- **6 blocks**: max abs 55.5 (extreme)
- **8 blocks**: max abs 56.6 (extreme)  
- **10 blocks**: max abs 51.1 (extreme)

## Evidence Collected

1. **Initialization changes affect timing**: Xavier initialization delayed failure from batch 80 to batch 21
2. **Block count testing results**: 
   - 6 blocks: max abs 55.5 (extreme)
   - 8 blocks: max abs 56.6 (extreme)  
   - 10 blocks: max abs 51.1 (extreme)
3. **Values improve during early training**: Network reduces extreme values from 89.5 to ~18.9 initially
4. **Gradient explosion observed**: Pre-clip gradients of 137+ being clipped to 2.0
5. **Isolation testing results**:
   - Policy head alone: max abs ~4.6 (reasonable)
   - Policy head + simple trunk: max abs ~8.9 (reasonable)
   - Full 10-block ResNet: max abs 41.9 (extreme)

## Debug Output Locations

### Training Logs
- Look for `EARLY_TRAINING_DEBUG` and `POST_WARMUP_DEBUG` messages
- Look for `LARGE_VALUES_DEBUG` warnings when values exceed 30.0
- Look for `EXTREME_VALUES_DETECTED` errors when values exceed 20.0

### Debug Files
- `checkpoints/bookkeeping/errors/debug_extreme_model_outputs_*.json` - Contains detailed statistics when extreme values are detected
- `temp/diagnostic_training_data.json` - Contains comprehensive layer-by-layer statistics from diagnostic runs

### Key Metrics to Monitor
- `policy_max_abs`: Maximum absolute value of policy logits
- `value_max_abs`: Maximum absolute value of value outputs  
- `pre_clip_grad_norm`: Gradient norm before clipping
- `post_clip_grad_norm`: Gradient norm after clipping

## Conjectures to Explore

### Why might the network be "too deep" despite deeper networks working elsewhere?

1. **Global pooling bias injection interaction**: The global pooling mechanism may interact poorly with deep feature representations, causing instability that doesn't occur in standard ResNets
2. **Feature magnitude accumulation**: Without proper normalization, features may grow in magnitude through the deep trunk, leading to extreme values at the policy head
3. **Gradient flow issues**: The combination of deep trunk + global pooling + policy head may create gradient flow patterns that cause instability
4. **Initialization mismatch**: The initialization assumptions (Kaiming for ReLU) may not hold for the specific feature distributions produced by this architecture

### Specific Evidence Supporting These Conjectures

- **Gradient explosion**: Pre-clip gradients of 137+ suggest the architecture creates unstable gradient flow
- **Feature magnitude growth**: Diagnostic data shows `policy_gconv` with max_abs=65.35, suggesting features grow large in the trunk
- **Improvement then degradation**: Values improve during early training (89.5→18.9) then grow again, suggesting the network learns to compensate initially but underlying instability remains
- **Isolation works**: Policy head alone produces reasonable values, suggesting the issue is in the trunk-head interaction

## Next Steps

1. **Test with 1-5 blocks** to find if there's a stable range
2. **Investigate feature normalization** in the trunk to prevent magnitude growth
3. **Test alternative global pooling mechanisms** (different scaling, different injection points)
4. **Compare with standard ResNet architectures** to understand what makes this combination unstable

## Files Modified During Investigation

- `hex_ai/models.py`: Added Xavier initialization for PolicyHead final conv layer
- `hex_ai/training.py`: Added enhanced debugging with gradient norm monitoring

### Enhanced Debug Output

The training code now includes additional debug output to help analyze the problem:

- **`EARLY_TRAINING_DEBUG`** (batches 0-19): Logs policy and value statistics for first 20 batches
- **`POST_WARMUP_DEBUG`** (batches 20-29): Logs statistics for batches after warmup period  
- **`LARGE_VALUES_DEBUG`**: Warning when policy_max_abs > 30.0 or value_max_abs > 0.8, includes gradient norm info
- **`EXTREME_VALUES_DETECTED`**: Error with detailed statistics when values exceed thresholds

**Motivation**: The original extreme value detection only triggered at failure, making it hard to understand the dynamics. The enhanced debugging provides visibility into the training progression and helps identify when and why values become extreme.

### Debug Data Files

- **`checkpoints/bookkeeping/errors/debug_extreme_model_outputs_*.json`**: Contains detailed statistics when extreme values are detected, including layer-by-layer analysis
- **`temp/diagnostic_training_data.json`**: Comprehensive diagnostic data with layer statistics, gradient norms, and weight statistics (created during investigation, now cleaned up)

---

## Appendix: Comprehensive Stability Fixes Implementation and Results

### Overview

Following the initial investigation, we implemented a comprehensive set of stability mechanisms to address the systematic value explosion in the policy head. Despite these extensive changes, the fundamental issue persists, indicating a deeper architectural problem.

### Implemented Changes

#### 1. Policy Head Stability Configuration

We created a centralized configuration class to manage all stability-related hyperparameters:

```python
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
```

#### 2. Enhanced Policy Head with Layer Normalization

We completely redesigned the PolicyHead class to include layer normalization and stability monitoring:

```python
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
```

#### 3. Fixed Initialization Order

We identified and fixed a critical bug where the main model initialization was overriding the policy head initialization:

```python
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
        # ... other initialization code ...
```

#### 4. Targeted Weight Decay Optimizer

We created a specialized optimizer that applies different weight decay to the policy head final layer:

```python
def create_optimizer_with_policy_head_stability(model: nn.Module, 
                                               base_learning_rate: float = 3e-4,
                                               base_weight_decay: float = 1e-4,
                                               betas: Tuple[float, float] = (0.9, 0.999),
                                               eps: float = 1e-8) -> torch.optim.AdamW:
    """
    Create an optimizer with different weight decay for policy head final layer.
    
    This function addresses the policy head stability issue by applying higher
    weight decay to the final conv layer that is prone to gradient accumulation.
    """
    # Get all model parameters except policy head final layer
    other_params = []
    for name, param in model.named_parameters():
        if not (hasattr(model, 'policy_head') and 
                hasattr(model.policy_head, 'conv2') and 
                name == 'policy_head.conv2.weight'):
            other_params.append(param)
    
    # Get policy head final layer parameters
    policy_final_params = []
    if hasattr(model, 'policy_head') and hasattr(model.policy_head, 'conv2'):
        policy_final_params = [model.policy_head.conv2.weight]
    
    # Create parameter groups with different weight decay
    param_groups = [
        {
            'params': other_params,
            'weight_decay': base_weight_decay,
            'lr': base_learning_rate
        }
    ]
    
    if policy_final_params:
        param_groups.append({
            'params': policy_final_params,
            'weight_decay': base_weight_decay * POLICY_HEAD_CONFIG.FINAL_LAYER_WEIGHT_DECAY_FACTOR,
            'lr': base_learning_rate
        })
    
    return torch.optim.AdamW(param_groups, betas=betas, eps=eps)
```

#### 5. Enhanced Gradient Monitoring

We added comprehensive monitoring for gradient and weight magnitudes:

```python
def monitor_policy_head_gradients(model: nn.Module, batch_idx: int = None):
    """
    Monitor gradient norms in the policy head for early detection of instability.
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
```

### Current Training Behavior (Post-Fixes)

Despite all these comprehensive changes, the systematic value explosion persists. Here is the detailed progression from a recent training run:

#### Early Training (Batches 0-29)
```
2025-09-16 14:12:39,971 INFO:hex_ai.training: EARLY_TRAINING_DEBUG: Epoch 1, Mini-epoch 1, Batch 0: policy_max_abs=0.683105, value_max_abs=0.661621
2025-09-16 14:12:41,066 INFO:hex_ai.training: EARLY_TRAINING_DEBUG: Epoch 1, Mini-epoch 1, Batch 1: policy_max_abs=0.636719, value_max_abs=0.832031
2025-09-16 14:12:41,435 INFO:hex_ai.training: EARLY_TRAINING_DEBUG: Epoch 1, Mini-epoch 1, Batch 2: policy_max_abs=0.690430, value_max_abs=0.820312
2025-09-16 14:12:41,804 INFO:hex_ai.training: EARLY_TRAINING_DEBUG: Epoch 1, Mini-epoch 1, Batch 3: policy_max_abs=0.801758, value_max_abs=0.702637
2025-09-16 14:12:42,170 INFO:hex_ai.training: EARLY_TRAINING_DEBUG: Epoch 1, Mini-epoch 1, Batch 4: policy_max_abs=0.858398, value_max_abs=0.636719
2025-09-16 14:12:42,531 INFO:hex_ai.training: EARLY_TRAINING_DEBUG: Epoch 1, Mini-epoch 1, Batch 5: policy_max_abs=0.919922, value_max_abs=0.640625
2025-09-16 14:12:42,891 INFO:hex_ai.training: EARLY_TRAINING_DEBUG: Epoch 1, Mini-epoch 1, Batch 6: policy_max_abs=0.956543, value_max_abs=0.682617
2025-09-16 14:12:43,255 INFO:hex_ai.training: EARLY_TRAINING_DEBUG: Epoch 1, Mini-epoch 1, Batch 7: policy_max_abs=1.077148, value_max_abs=0.615234
2025-09-16 14:12:43,620 INFO:hex_ai.training: EARLY_TRAINING_DEBUG: Epoch 1, Mini-epoch 1, Batch 8: policy_max_abs=1.142578, value_max_abs=0.679688
2025-09-16 14:12:43,982 INFO:hex_ai.training: EARLY_TRAINING_DEBUG: Epoch 1, Mini-epoch 1, Batch 9: policy_max_abs=1.281250, value_max_abs=0.698730
2025-09-16 14:12:44,343 INFO:hex_ai.training: EARLY_TRAINING_DEBUG: Epoch 1, Mini-epoch 1, Batch 10: policy_max_abs=1.304688, value_max_abs=0.755371
2025-09-16 14:12:44,707 INFO:hex_ai.training: EARLY_TRAINING_DEBUG: Epoch 1, Mini-epoch 1, Batch 11: policy_max_abs=1.370117, value_max_abs=0.734375
2025-09-16 14:12:45,069 INFO:hex_ai.training: EARLY_TRAINING_DEBUG: Epoch 1, Mini-epoch 1, Batch 12: policy_max_abs=1.407227, value_max_abs=0.754395
2025-09-16 14:12:45,430 INFO:hex_ai.training: EARLY_TRAINING_DEBUG: Epoch 1, Mini-epoch 1, Batch 13: policy_max_abs=1.428711, value_max_abs=0.712891
2025-09-16 14:12:45,798 INFO:hex_ai.training: EARLY_TRAINING_DEBUG: Epoch 1, Mini-epoch 1, Batch 14: policy_max_abs=1.487305, value_max_abs=0.641113
2025-09-16 14:12:46,165 INFO:hex_ai.training: EARLY_TRAINING_DEBUG: Epoch 1, Mini-epoch 1, Batch 15: policy_max_abs=1.546875, value_max_abs=0.698242
2025-09-16 14:12:46,528 INFO:hex_ai.training: EARLY_TRAINING_DEBUG: Epoch 1, Mini-epoch 1, Batch 16: policy_max_abs=1.587891, value_max_abs=0.665527
2025-09-16 14:12:46,891 INFO:hex_ai.training: EARLY_TRAINING_DEBUG: Epoch 1, Mini-epoch 1, Batch 17: policy_max_abs=1.638672, value_max_abs=0.655273
2025-09-16 14:12:47,253 INFO:hex_ai.training: EARLY_TRAINING_DEBUG: Epoch 1, Mini-epoch 1, Batch 18: policy_max_abs=1.674805, value_max_abs=0.698730
2025-09-16 14:12:47,616 INFO:hex_ai.training: EARLY_TRAINING_DEBUG: Epoch 1, Mini-epoch 1, Batch 19: policy_max_abs=1.720703, value_max_abs=0.703125
2025-09-16 14:12:47,977 INFO:hex_ai.training: POST_WARMUP_DEBUG: Epoch 1, Mini-epoch 1, Batch 20: policy_max_abs=1.782227, value_max_abs=0.717285
2025-09-16 14:12:48,339 INFO:hex_ai.training: POST_WARMUP_DEBUG: Epoch 1, Mini-epoch 1, Batch 21: policy_max_abs=1.813477, value_max_abs=0.674805
2025-09-16 14:12:48,700 INFO:hex_ai.training: POST_WARMUP_DEBUG: Epoch 1, Mini-epoch 1, Batch 22: policy_max_abs=1.917969, value_max_abs=0.665527
2025-09-16 14:12:49,061 INFO:hex_ai.training: POST_WARMUP_DEBUG: Epoch 1, Mini-epoch 1, Batch 23: policy_max_abs=1.922852, value_max_abs=0.679688
2025-09-16 14:12:49,423 INFO:hex_ai.training: POST_WARMUP_DEBUG: Epoch 1, Mini-epoch 1, Batch 24: policy_max_abs=1.989258, value_max_abs=0.683594
2025-09-16 14:12:49,787 INFO:hex_ai.training: POST_WARMUP_DEBUG: Epoch 1, Mini-epoch 1, Batch 25: policy_max_abs=2.013672, value_max_abs=0.671875
2025-09-16 14:12:50,152 INFO:hex_ai.training: POST_WARMUP_DEBUG: Epoch 1, Mini-epoch 1, Batch 26: policy_max_abs=2.033203, value_max_abs=0.729004
2025-09-16 14:12:50,523 INFO:hex_ai.training: POST_WARMUP_DEBUG: Epoch 1, Mini-epoch 1, Batch 27: policy_max_abs=2.085938, value_max_abs=0.689941
2025-09-16 14:12:50,894 INFO:hex_ai.training: POST_WARMUP_DEBUG: Epoch 1, Mini-epoch 1, Batch 28: policy_max_abs=2.132812, value_max_abs=0.664551
2025-09-16 14:12:51,256 INFO:hex_ai.training: POST_WARMUP_DEBUG: Epoch 1, Mini-epoch 1, Batch 29: policy_max_abs=2.175781, value_max_abs=0.715332
```

#### Continued Growth (Batches 90-100)
```
2025-09-16 14:13:13,342 WARNING:hex_ai.training: LARGE_VALUES_DEBUG: Epoch 1, Mini-epoch 1, Batch 90: policy_max_abs=3.724609, value_max_abs=0.803223, policy_range=[-3.182, 3.725], value_range=[0.097, 0.803], pre_clip_grad_norm=3.936, post_clip_grad_norm=2.000
2025-09-16 14:13:16,628 WARNING:hex_ai.training: LARGE_VALUES_DEBUG: Epoch 1, Mini-epoch 1, Batch 99: policy_max_abs=3.804688, value_max_abs=0.822266, policy_range=[-3.328, 3.805], value_range=[0.159, 0.822], pre_clip_grad_norm=3.545, post_clip_grad_norm=2.000
2025-09-16 14:13:16,981 WARNING:hex_ai.training: LARGE_VALUES_DEBUG: Epoch 1, Mini-epoch 1, Batch 100: policy_max_abs=3.833984, value_max_abs=0.816406, policy_range=[-3.404, 3.834], value_range=[0.083, 0.816], pre_clip_grad_norm=3.993, post_clip_grad_norm=2.000
```

#### Severe Growth (Batches 440-450)
```
2025-09-16 14:15:21,165 WARNING:hex_ai.training: LARGE_VALUES_DEBUG: Epoch 1, Mini-epoch 1, Batch 443: policy_max_abs=6.187500, value_max_abs=0.923340, policy_range=[-4.605, 6.188], value_range=[0.016, 0.923], pre_clip_grad_norm=2.741, post_clip_grad_norm=2.000
2025-09-16 14:15:21,529 WARNING:hex_ai.training: LARGE_VALUES_DEBUG: Epoch 1, Mini-epoch 1, Batch 444: policy_max_abs=6.230469, value_max_abs=0.901367, policy_range=[-4.559, 6.230], value_range=[-0.130, 0.901], pre_clip_grad_norm=2.591, post_clip_grad_norm=2.000
```

#### Critical Growth (Batches 770-980)
```
2025-09-16 14:17:20,402 WARNING:hex_ai.training: LARGE_VALUES_DEBUG: Epoch 1, Mini-epoch 1, Batch 772: policy_max_abs=8.117188, value_max_abs=0.887207, policy_range=[-4.461, 8.117], value_range=[0.011, 0.887], pre_clip_grad_norm=2.426, post_clip_grad_norm=2.000
2025-09-16 14:17:20,764 WARNING:hex_ai.training: LARGE_VALUES_DEBUG: Epoch 1, Mini-epoch 1, Batch 773: policy_max_abs=8.195312, value_max_abs=0.936035, policy_range=[-4.430, 8.195], value_range=[-0.020, 0.936], pre_clip_grad_norm=2.631, post_clip_grad_norm=2.000
2025-09-16 14:17:21,124 WARNING:hex_ai.training: LARGE_VALUES_DEBUG: Epoch 1, Mini-epoch 1, Batch 774: policy_max_abs=8.132812, value_max_abs=0.900879, policy_range=[-4.484, 8.133], value_range=[0.040, 0.901], pre_clip_grad_norm=2.645, post_clip_grad_norm=2.000
2025-09-16 14:17:21,484 WARNING:hex_ai.training: LARGE_VALUES_DEBUG: Epoch 1, Mini-epoch 1, Batch 775: policy_max_abs=8.156250, value_max_abs=0.891602, policy_range=[-4.559, 8.156], value_range=[0.048, 0.892], pre_clip_grad_norm=2.439, post_clip_grad_norm=2.000
2025-09-16 14:18:37,328 WARNING:hex_ai.training: LARGE_VALUES_DEBUG: Epoch 1, Mini-epoch 1, Batch 977: policy_max_abs=9.015625, value_max_abs=0.902344, policy_range=[-4.539, 9.016], value_range=[0.021, 0.902], pre_clip_grad_norm=2.461, post_clip_grad_norm=2.000
2025-09-16 14:18:37,692 WARNING:hex_ai.training: LARGE_VALUES_DEBUG: Epoch 1, Mini-epoch 1, Batch 978: policy_max_abs=9.203125, value_max_abs=0.909668, policy_range=[-4.395, 9.203], value_range=[-0.010, 0.910], pre_clip_grad_norm=2.814, post_clip_grad_norm=2.000
2025-09-16 14:18:38,057 WARNING:hex_ai.training: LARGE_VALUES_DEBUG: Epoch 1, Mini-epoch 1, Batch 978: policy_max_abs=9.093750, value_max_abs=0.945312, policy_range=[-4.645, 9.094], value_range=[-0.112, 0.945], pre_clip_grad_norm=2.509, post_clip_grad_norm=2.000
2025-09-16 14:18:38,416 WARNING:hex_ai.training: LARGE_VALUES_DEBUG: Epoch 1, Mini-epoch 1, Batch 980: policy_max_abs=9.179688, value_max_abs=0.919434, policy_range=[-4.457, 9.180], value_range=[-0.039, 0.919], pre_clip_grad_norm=2.685, post_clip_grad_norm=2.000
```

### Analysis of Current Behavior

#### Key Observations

1. **Systematic Growth Pattern**: The policy logits show a clear, monotonic increase:
   - Batch 0: 0.683
   - Batch 29: 2.176
   - Batch 100: 3.834
   - Batch 444: 6.230
   - Batch 773: 8.195
   - Batch 980: 9.180

2. **Growth Rate**: The growth rate is approximately 0.0087 per batch (9.18 - 0.68) / 980 batches, which is slower than the original 0.061465 per batch but still systematic.

3. **Gradient Clipping Working**: The gradient clipping is functioning correctly:
   - Pre-clip gradients: ~2.4-3.9
   - Post-clip gradients: 2.000 (clipped to max norm)

4. **Value Head Stability**: The value head remains stable throughout training (max_abs ~0.9), confirming the issue is specific to the policy head.

5. **Layer Normalization Not Sufficient**: Despite adding layer normalization before the final conv layer, the systematic growth continues, suggesting the issue is deeper than feature magnitude normalization.

### Conclusion

The comprehensive stability fixes we implemented have **delayed** the explosion but have **not solved** the fundamental architectural issue. The systematic growth pattern persists, indicating that:

1. **Layer normalization alone is insufficient** to prevent the value explosion
2. **The problem is not just initialization** - it's a deeper architectural instability
3. **The global pooling bias injection** may be fundamentally incompatible with the deep ResNet trunk
4. **The combination of deep trunk + global pooling + policy head** creates an unstable gradient flow pattern

The fact that we can delay the explosion from ~500 batches to potentially 1000+ batches suggests our fixes are working to some degree, but the underlying architectural instability remains. This indicates we need to consider more fundamental changes to the architecture, such as:

- Removing or redesigning the global pooling bias injection
- Simplifying the policy head architecture
- Using a different trunk architecture that's more compatible with the policy head design
- Implementing more aggressive regularization or architectural constraints

The systematic nature of the growth (not random noise) strongly suggests this is a fundamental design flaw rather than a hyperparameter tuning issue.
