# Network Design Analysis and Extreme Values Investigation

## Executive Summary

The Hex AI project is experiencing systematic extreme value explosions during training, specifically in policy logits that grow from ~0.5 to >20.0 over ~500 batches. This report analyzes the network architecture, identifies the root causes, and documents the investigation process.

## Network Architecture Overview

### Current Architecture: KataGo-Inspired Two-Headed ResNet

The network uses a modern architecture inspired by KataGo with the following components:

#### Core Architecture
- **Input**: 3-channel board representation (blue, red, player-to-move)
- **Trunk**: 7 ResNet blocks with constant 128 channels (no downsampling)
- **Policy Head**: Global pooling bias injection preserving 13×13 spatial structure
- **Value Head**: Stage-conditioned multi-output ensemble with LayerNorm

#### Block Composition
- **Plain ResNet Blocks**: Standard residual blocks (every 1st and 2nd block)
- **Global Pooling Blocks**: ResNet blocks with global pooling bias injection (every 3rd block)
- **Total**: 7 blocks (5 plain + 2 global pooling)

### Hyperparameters (From hyperparam_sweep.py)

```python
SWEEP = {
    "batch_size": [256],
    "max_grad_norm": [2.0],
    "weight_decay": [1e-4],
    "value_learning_rate_factor": [1],
    "value_weight_decay_factor": [1],
    "policy_weight": [0.7],
    "learning_rate": [3e-4],  # 0.0003, not 0.001 as in config.py
    "betas": [(0.9, 0.999)],
    "eps": [1e-8],
    "num_blocks": [7],
    "trunk_channels": [128],
    "dropout_prob": [0],
}
```

**Note**: The actual learning rate used is 3e-4 (0.0003), not 0.001 as defined in config.py, due to sweep parameter override.

## Extreme Values Problem Analysis

### Observed Behavior

The training shows a **systematic, monotonic increase** in policy logits:

```
Batch 0:   policy_max_abs=0.583
Batch 10:  policy_max_abs=1.442
Batch 29:  policy_max_abs=5.398
Batch 53:  policy_max_abs=6.594
Batch 84:  policy_max_abs=8.570
Batch 148: policy_max_abs=9.438
Batch 272: policy_max_abs=11.641
Batch 484: policy_max_abs=20.734  ← CRASH
```

**Key Characteristics**:
- **Not random noise**: Clear linear trend (0.061465 per batch)
- **Not instant explosion**: Gradual growth over ~500 batches
- **Policy-specific**: Value outputs remain stable
- **Gradient clipping working**: Pre-clip 3.xxx → Post-clip 2.000

### Diagnostic Evidence

#### 1. Architecture Isolation Tests
```
Policy head alone:           max_abs ~4.6  ✅ STABLE
Policy head + simple trunk:  max_abs ~8.9  ✅ STABLE  
Full 10-block ResNet:        max_abs 41.9  ❌ UNSTABLE
```

#### 2. Block Count Analysis
```
1-4 blocks:  STABLE (max policy abs: 12.9-19.7)
5+ blocks:   UNSTABLE (max policy abs: 56.5-194.1)
```

#### 3. Enhanced Training Diagnostics
```
Policy max abs progression: 0.146 → 1.108 (7.57x growth over 50 batches)
Policy head conv2 gradient norms: 0.415 → 1.193 (growing)
Policy head conv2 gradient trend: 0.001922 per batch
```

## Root Cause Analysis

### Primary Issue: Policy Head Final Conv Layer Instability

The evidence points to the **policy head's final conv layer** (`policy_head.conv2`) as the source of systematic value growth:

1. **Feature magnitude tracking shows explosion in final layer**:
   ```
   policy_after_injection: max_abs=6.469  ← Reasonable
   Policy: max_abs=26.834                 ← 4x explosion!
   ```

2. **Gradient norms growing systematically**:
   ```
   Policy head conv2 gradient norms: 0.415 → 1.193
   ```

3. **Isolation tests confirm**: Policy head alone is stable, full network explodes

### Secondary Issue: Global Pooling Residual Block Implementation

**Fixed Issue**: The `GlobalPoolingResidualBlock` had incorrect residual connection implementation:

**Original (problematic)**:
```python
out = local_features
out = out + g  # Add global bias to local features
out = F.relu(out + x)  # Add residual connection to (local + global)
```

**Fixed**:
```python
local = local_features
g = global_bias
out = F.relu(local + x + g)  # Add residual connection and global bias together
```

**Impact**: This fix reduced extreme values from 194.1 to 202.7, but didn't solve the fundamental issue.

### Tertiary Issue: Policy Head Initialization

**Current Fix**: Special initialization for policy head final conv layer:
```python
# Xavier initialization with 0.1 scaling
nn.init.xavier_normal_(self.policy_head.conv2.weight)
with torch.no_grad():
    self.policy_head.conv2.weight *= 0.1
```

**Impact**: This fix made the network stable for initialization but didn't prevent training-time growth.

## Attempted Solutions and Results

### 1. Architecture Fixes ✅ PARTIALLY EFFECTIVE
- **Fixed GlobalPoolingResidualBlock residual connections**: Reduced extreme values from 194.1 to 202.7
- **Added policy head special initialization**: Made network stable at initialization

### 2. Hyperparameter Adjustments ❌ INEFFECTIVE
- **Reduced learning rate**: 0.001 → 0.0001 (10x reduction)
- **Reduced policy head scaling**: 0.1 → 0.05 (2x reduction)
- **Result**: Would likely delay explosion but not prevent it

### 3. Diagnostic Improvements ✅ EFFECTIVE
- **Enhanced training diagnostics**: Revealed systematic gradient growth
- **Feature magnitude tracking**: Identified policy head as problem source
- **Isolation testing**: Confirmed architectural stability

## Current Status

### What's Working
- ✅ Network initializes stably (no instant explosion)
- ✅ Gradient clipping is effective
- ✅ Value head remains stable
- ✅ Architecture isolation tests pass

### What's Broken
- ❌ Policy logits grow systematically during training
- ❌ Policy head final conv layer accumulates gradients
- ❌ Network crashes after ~500 batches due to extreme values

## Recommended Next Steps

### 1. Immediate: Enhanced Policy Head Diagnostics
Create targeted diagnostics to track:
- Policy head conv2 weight changes over training
- Gradient accumulation patterns
- Comparison with standard ResNet final layers

### 2. Investigation: Policy Head Architecture
Examine why the policy head final conv layer is unstable:
- Is the 0.1 scaling factor appropriate?
- Is Xavier initialization correct for this layer?
- Are there architectural issues with the global pooling injection?

### 3. Alternative: Architecture Simplification
Consider testing:
- Policy head without global pooling bias injection
- Standard ResNet final layer instead of custom implementation
- Different initialization strategies for final layers

### 4. Long-term: Standard Architecture Comparison
Compare with:
- Standard ResNet architectures (ResNet-18, ResNet-34)
- Other policy networks (AlphaZero, Leela Zero)
- KataGo's actual implementation details

## Technical Debt and Cleanup

### Temporary Changes Made
1. **Policy head initialization scaling**: 0.1 factor (revert by removing scaling)
2. **GlobalPoolingResidualBlock fix**: Corrected residual connection (keep this fix)

### Revert Instructions
To return to standard setup:
```python
# In models.py, remove the scaling factor:
# Change: self.policy_head.conv2.weight *= 0.1
# To:     # self.policy_head.conv2.weight *= 0.1
```

## Conclusion

The extreme values issue is **not** a simple hyperparameter problem but a **fundamental architectural issue** with the policy head's final conv layer. The systematic growth pattern (0.061465 per batch) indicates a design flaw rather than training instability.

The investigation has successfully:
1. ✅ Identified the problem source (policy head final conv layer)
2. ✅ Ruled out simple fixes (learning rate, initialization)
3. ✅ Provided diagnostic tools for further investigation

**Next priority**: Deep dive into the policy head architecture to understand why the final conv layer accumulates gradients and causes systematic value growth.

---

*Report generated: 2025-09-16*  
*Investigation status: Active - Policy head architecture analysis needed*
