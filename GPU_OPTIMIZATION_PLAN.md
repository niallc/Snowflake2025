# GPU Optimization Plan for M1 MacBook Pro

## Current Status

The hyperparameter tuning is currently running on CPU. PyTorch supports Apple's Metal Performance Shaders (MPS) for GPU acceleration on M1 Macs, which could provide **3-5x speedup** for training.

## Implementation Plan

### Phase 1: Enable MPS GPU Support (Quick Win)

#### 1.1 Update Device Detection
**File:** `hex_ai/config.py`
```python
# Current
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Updated
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
```

#### 1.2 Update Training Scripts
**Files:** `hex_ai/training.py`, `hyperparameter_tuning_v2.py`
- Replace `device="cuda"` with `device="mps"`
- Update mixed precision logic for MPS
- Remove `pin_memory=True` (not supported on MPS)

#### 1.3 Test GPU Acceleration
```bash
# Quick test script
python -c "
import torch
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'Device: {torch.device(\"mps\" if torch.backends.mps.is_available() else \"cpu\")}')
x = torch.randn(1000, 1000)
if torch.backends.mps.is_available():
    x = x.to('mps')
    print(f'GPU tensor shape: {x.shape}')
"
```

### Phase 2: Performance Optimization

#### 2.1 Batch Size Optimization
**Goal:** Find optimal batch size for M1 GPU memory
- Test batch sizes: 32, 64, 128, 256
- Monitor memory usage and training speed
- Find sweet spot between speed and memory efficiency

#### 2.2 Mixed Precision Training
**Goal:** Enable automatic mixed precision for MPS
```python
# In training.py
if device == "mps":
    # Use torch.autocast for MPS
    with torch.autocast(device_type="mps"):
        # Forward pass
```

#### 2.3 Data Loading Optimization
**Goal:** Optimize data pipeline for GPU
- Increase `num_workers` for DataLoader
- Use `prefetch_factor` for better pipelining
- Profile data loading bottlenecks

### Phase 3: Hyperparameter Tuning with GPU

#### 3.1 GPU-Specific Experiments
**New experiments to test:**
- Larger batch sizes (128, 256, 512)
- Different learning rates (GPU can handle higher rates)
- Memory-efficient model variants

#### 3.2 Performance Monitoring
**Metrics to track:**
- Training time per epoch
- GPU memory usage
- GPU utilization
- Data transfer overhead

### Phase 4: Advanced Optimizations

#### 4.1 Model Architecture Tweaks
- Test different ResNet depths (18, 34, 50)
- Experiment with attention mechanisms
- Try different activation functions

#### 4.2 Memory Management
- Gradient checkpointing for larger models
- Dynamic batch sizing
- Memory-efficient optimizers

## Implementation Steps

### Step 1: Quick GPU Test (5 minutes)
```bash
# Test MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"

# Test basic GPU operations
python -c "
import torch
if torch.backends.mps.is_available():
    x = torch.randn(1000, 1000).to('mps')
    y = torch.randn(1000, 1000).to('mps')
    z = torch.mm(x, y)
    print('GPU test successful')
"
```

### Step 2: Update Configuration (10 minutes)
- Update `hex_ai/config.py` to use MPS
- Test with current hyperparameter tuning
- Verify no errors

### Step 3: Performance Benchmarking (30 minutes)
- Run speed comparison: CPU vs GPU
- Test different batch sizes
- Profile memory usage

### Step 4: Enhanced Hyperparameter Tuning (2-3 hours)
- Run new GPU-optimized experiments
- Test larger models and batch sizes
- Compare results with CPU baseline

## Expected Benefits

### Speed Improvements
- **3-5x faster training** on GPU vs CPU
- **Larger batch sizes** possible (128-256 vs 32-64)
- **Faster model evaluation** during validation

### Memory Considerations
- **M1 GPU memory:** ~8GB shared with system
- **Optimal batch size:** Likely 128-256 for this model
- **Memory monitoring:** Essential to avoid OOM errors

### Quality Improvements
- **Better convergence** with larger batch sizes
- **More stable gradients** with GPU precision
- **Faster experimentation** cycle

## Risk Mitigation

### Potential Issues
1. **MPS compatibility:** Some PyTorch operations may not work on MPS
2. **Memory limits:** Need to monitor GPU memory usage
3. **Numerical differences:** GPU vs CPU may have slight precision differences

### Solutions
1. **Fallback to CPU:** If MPS fails, automatically use CPU
2. **Memory monitoring:** Add GPU memory tracking
3. **Reproducibility:** Set random seeds for consistent results

## Success Metrics

### Performance Targets
- **Training speed:** 3-5x faster than CPU
- **Memory efficiency:** <6GB GPU memory usage
- **Stability:** No crashes or OOM errors

### Quality Targets
- **Convergence:** Same or better loss curves
- **Reproducibility:** Consistent results across runs
- **Reliability:** 99%+ uptime during training

## Next Steps

1. **Immediate:** Test MPS availability and basic operations
2. **This week:** Update configuration and run GPU-optimized hyperparameter tuning
3. **Next week:** Analyze results and implement advanced optimizations

## Resources

- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [Apple Metal Performance Shaders](https://developer.apple.com/metal/pytorch/)
- [M1 Mac GPU Optimization](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/)
