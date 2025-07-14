# Parallel Hyperparameter Tuning Analysis & Plan

## **Current State & Context**

### **What We Have:**
- **Main script**: `hyperparameter_tuning_gpu_large.py` - runs 6 experiments sequentially
- **Batch size**: Recently increased from 64 → 512 (8x improvement)
- **Resource utilization**: ~20% CPU, ~60% memory (16GB/32GB), though this may be dominated by other processes.
- **Training speed**: ~15 seconds per batch update (100 batches of size 512), epochs complete in minutes
- **System**: 10-core Mac with 32GB RAM, Apple MPS GPU

### **Key Files:**
- `hyperparameter_tuning_gpu_large.py` - main tuning script
- `hex_ai/training_utils.py` - experiment runner functions
- `hex_ai/training.py` - training loop with memory monitoring
- `monitor_resources.py` - resource utilization tracking
- `analyze_tuning_results.py` - results analysis

### **Recent Optimizations Made:**
1. **Batch size increased** from 64 to 512
2. **Memory monitoring** added to training loop
3. **Resource monitoring script** created for real-time tracking

## **Parallel Experiment Complexity Analysis**

### **Race Conditions & Conflicts:**

#### **1. File System Conflicts (HIGH RISK)**
- **Problem**: Multiple processes writing to same directories
- **Current**: Each experiment creates `checkpoints/experiment_name/`
- **Risk**: Process A creates directory, Process B tries to create same directory
- **Solution**: Unique experiment names with timestamps + process IDs

#### **2. Data Loading Conflicts (MEDIUM RISK)**
- **Problem**: Multiple processes loading same data files simultaneously
- **Current**: Each experiment loads data independently
- **Risk**: File system contention, memory pressure
- **Solution**: Shared data loading or separate data splits

#### **3. GPU Memory Conflicts (HIGH RISK)**
- **Problem**: Multiple processes trying to use MPS GPU simultaneously
- **Current**: Single process uses GPU
- **Risk**: GPU memory exhaustion, process crashes
- **Solution**: GPU memory monitoring, process coordination

#### **4. Analysis Result Conflicts (LOW RISK)**
- **Problem**: Multiple processes writing analysis files with same names
- **Current**: Analysis runs after all experiments complete
- **Risk**: Overwriting plots, CSV files
- **Solution**: Unique output directories per parallel run

### **Directory Structure Conflicts:**

#### **Current Structure:**
```
checkpoints/
  hex_ai_hyperparam_tuning_v3_500k_samples_20250714_073022/
    balanced_weights/
    balanced_high_weight_decay/
    policy_heavy/
    ...
```

#### **Parallel Structure (Proposed):**
```
checkpoints/
  parallel_tuning_20250714_073022/
    process_1/
      balanced_weights/
      balanced_high_weight_decay/
    process_2/
      policy_heavy/
      policy_intermediate/
    process_3/
      balanced_no_dropout/
      balanced_high_lr/
    overall_results.json
```

## **Implementation Plan**

### **Phase 1: Safe Parallel Implementation (2-3 hours)**

#### **1.1 Process Isolation**
- **Unique experiment names**: `{base_name}_process_{pid}_{timestamp}`
- **Separate result directories**: Each process gets its own subdirectory
- **Independent data loading**: Each process loads its own data subset

#### **1.2 Resource Management**
- **GPU memory monitoring**: Track usage, limit concurrent GPU processes
- **Memory limits**: Cap total memory usage across processes
- **Process coordination**: Simple semaphore for GPU access

#### **1.3 Error Handling**
- **Process crash recovery**: If one fails, others continue
- **Resource cleanup**: Ensure GPU memory is freed on crash
- **Result aggregation**: Collect results from successful processes only

### **Phase 2: Analysis Integration (1 hour)**

#### **2.1 Result Collection**
- **Aggregate results**: Combine results from all parallel processes
- **Conflict resolution**: Handle duplicate experiment names
- **Progress tracking**: Monitor completion of all processes

#### **2.2 Analysis Updates**
- **Update analysis script**: Handle new directory structure
- **Plot generation**: Create combined plots from all processes
- **Summary reports**: Aggregate statistics across all experiments

## **Recommendations**

### **Immediate (Next 30 minutes):**
1. **Try larger batch size first**: Increase to 1024 or 2048
   - **Risk**: Low (just memory monitoring)
   - **Gain**: 2-3x speedup per experiment
   - **Time**: 5 minutes to test

2. **If batch size works well**: Implement basic parallel experiments
   - **Approach**: 2-3 processes, separate directories
   - **Risk**: Medium (need careful resource management)
   - **Gain**: 2-3x total time reduction
   - **Time**: 2-3 hours to implement safely

### **Implementation Strategy:**

#### **Option A: Conservative (Recommended)**
- **2 parallel processes** with batch size 1024
- **Separate data splits** for each process
- **Simple process coordination**
- **Expected speedup**: 3-4x total time

#### **Option B: Aggressive**
- **3 parallel processes** with batch size 2048
- **Shared data loading** with coordination
- **Advanced resource management**
- **Expected speedup**: 4-6x total time

### **Risk Mitigation:**
1. **Start with 2 processes** to validate approach
2. **Monitor GPU memory** closely
3. **Implement graceful degradation** (fall back to sequential if parallel fails)
4. **Keep backup of current working script**

## **Technical Details**

### **Current Experiment Configuration:**
```python
BATCH_SIZE = 512
NUM_EPOCHS = 10
TARGET_EXAMPLES = 500000
EXPERIMENTS = 6  # balanced_weights, balanced_high_weight_decay, etc.
```

### **Resource Monitoring Commands:**
```bash
# Monitor system resources
python monitor_resources.py --duration 10 --interval 5

# Check specific process
ps aux | grep hyperparameter_tuning

# Monitor GPU usage (if available)
nvidia-smi  # for CUDA
system_profiler SPDisplaysDataType  # for MPS
```

### **Key Functions to Modify:**
1. `run_hyperparameter_tuning()` in `training_utils.py`
2. `run_hyperparameter_experiment()` in `training_utils.py`
3. `analyze_tuning_results.py` for result aggregation

## **Next Steps**

1. **Test batch size 1024** first (5 minutes)
2. **If successful**: Implement 2-process parallel version
3. **Monitor carefully** for first parallel run
4. **Scale up** to 3 processes if resources allow

**Recommendation**: Start with Option A (conservative approach) to validate the parallel concept, then scale up if successful.

## **Files to Create/Modify**

### **New Files Needed:**
- `parallel_tuning.py` - main parallel implementation
- `process_coordinator.py` - resource management
- `result_aggregator.py` - combine results from parallel processes

### **Files to Modify:**
- `hyperparameter_tuning_gpu_large.py` - add parallel option
- `analyze_tuning_results.py` - handle parallel result structure
- `hex_ai/training_utils.py` - add parallel experiment runner

### **Backup Strategy:**
- Keep current working script as `hyperparameter_tuning_gpu_large_backup.py`
- Use git branches for parallel development
- Test parallel version on small dataset first

---

*Report generated: 2025-07-14*
*Current status: Batch size 512 working well, ready for parallel implementation* 