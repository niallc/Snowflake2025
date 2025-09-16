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
