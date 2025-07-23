# Value Head Debugging Summary and Next Steps

**Note:** See also [value_net_overfitting_plan.md](value_net_overfitting_plan.md) (focused on now resolved overfitting) and [debugging_value_head_performance.md](debugging_value_head_performance.md) (older, detailed debugging log).

**Date:** 23rd July 2025
**Status:** Active Investigation  
**Priority:** High

## Executive Summary

The Hex AI value head is experiencing severe performance issues despite extensive debugging efforts. The value head shows poor generalization and appears to perform no better than random guessing on simple positions, while the policy head learns effectively.

## Current Understanding

### What We Know
1. **Value Loss ~0.2**: This indicates predictions are off by ~0.45 on average, suggesting the model is making predictions in the 0.3-0.7 range (close to random)
2. **Data Quality Verified**: Manual inspection confirms correct value labels, player-to-move channel, and data augmentation
3. **Policy Head Works**: Policy loss shows gradual improvement, indicating the network can extract useful features
4. **Data Shuffling Implemented**: Thorough shuffling has been applied to prevent game fingerprinting
5. **Extensive Hyperparameter Tuning**: Tested various learning rates, weight decay, loss weights, dropout, and gradient clipping

### What We've Tried
1. **Data Ordering**: Confirmed that ordered data leads to very low training loss (fingerprinting)
2. **Data Shuffling**: Implemented thorough shuffling, but value loss remains ~0.2
3. **Manual Position Testing**: Verified poor performance on simple positions using `simple_inference_cli.py`
4. **Hyperparameter Sweeps**: Tested various combinations without significant improvement
5. **Loss Separation**: Confirmed policy loss improves while value loss plateaus after ~10k samples

## Root Cause Hypotheses

### Primary Hypothesis: Value Head Architecture
- **Current**: Single linear layer (512→1)
- **Issue**: May be too simple or too complex for the task
- **Evidence**: Value loss plateaus early, suggesting architectural limitations

### Secondary Hypothesis: Training Dynamics
- **Issue**: Value head may be receiving insufficient or incorrect gradients
- **Evidence**: Policy head learns while value head doesn't, despite shared features

### Tertiary Hypothesis: Data Distribution
- **Issue**: Value prediction may be inherently harder than policy prediction
- **Evidence**: Need to analyze performance across different game stages

## New Analysis Tools Added

### 1. Gradient Monitoring (`GradientMonitor`)
- Tracks gradient norms for policy head, value head, and shared layers
- Helps identify vanishing/exploding gradients
- Lightweight logging without TensorBoard dependency

### 2. Activation Monitoring (`ActivationMonitor`)
- Monitors activations at key layers (value head, policy head, layer4, global_pool)
- Tracks activation statistics (mean, std, min, max)
- Helps identify saturation or dead neurons

### 3. Value Head Analyzer (`ValueHeadAnalyzer`)
- Analyzes value predictions across different position types (early/mid/late game)
- Tests simple known positions (empty board, winning positions)
- Provides detailed performance breakdown

## Recent Debugging Tools and Scripts (2025-07)

To support value head debugging and analysis, the following tools and scripts have been added:

- **GradientMonitor**: Tracks and logs gradient norms for policy head, value head, and shared layers during training. Helps diagnose vanishing/exploding gradients and imbalance between heads.
- **ActivationMonitor**: Monitors activations at key layers (value head, policy head, etc.) to detect saturation or dead neurons.
- **ValueHeadAnalyzer**: Provides methods to analyze value predictions by position type, test on simple positions, and summarize performance.
- **train_with_monitoring.py**: Training script that integrates gradient and activation monitoring, and saves analysis results.
- **analyze_value_by_game_stage.py**: Script to analyze value head performance across early, mid, and late game positions, reporting both strict and classification accuracy, and generating visualizations.
- **train_on_easy_positions.py**: Script to train and evaluate the value head on only final or penultimate positions, to test if the network can learn simple cases.
- **preprocess_example_for_model**: Central utility in hex_ai/data_utils.py to standardize preprocessing of examples for model input.

These tools are designed to make value head debugging more systematic, reproducible, and insightful.

## Proposed Next Steps

### Phase 1: Enhanced Monitoring (Week 1)
1. **Integrate Monitoring Tools**: Add gradient and activation monitoring to training loop
2. **Baseline Analysis**: Run current model with monitoring to establish baseline
3. **Position Type Analysis**: Analyze value performance across game stages

### Phase 2: Architecture Experiments (Week 2)
1. **Value Head Complexity**: Test different value head architectures:
   - 512→64→1 (with ReLU and dropout)
   - 512→32→1 (simpler)
   - 512→128→64→1 (more complex)
2. **Separate Feature Extraction**: Test value head with reduced feature channels (512→64→1)

### Phase 3: Targeted Training (Week 3)
1. **Easy Position Training**: Train on final positions only (no player turn dependency)
2. **Penultimate Position Training**: Train on penultimate + final positions
3. **Curriculum Learning**: Start with simple positions, gradually increase complexity

### Phase 4: Advanced Techniques (Week 4+)
1. **Multi-task Learning**: Add auxiliary tasks to help value head
2. **Ensemble Methods**: Train multiple value heads
3. **Self-play Data**: Generate additional training data

## Implementation Plan

### Immediate Actions (This Week)
1. **Add Monitoring to Training**: Integrate `GradientMonitor` and `ActivationMonitor` into training loop
2. **Run Baseline Analysis**: Collect gradient and activation data from current model
3. **Position Type Analysis**: Implement game stage analysis using `ValueHeadAnalyzer`

### Next Week
1. **Test Value Head Architectures**: Implement and test different value head designs
2. **Easy Position Training**: Set up training on final positions only
3. **Analyze Results**: Compare performance across different approaches

## Success Criteria

### Short-term (1-2 weeks)
- [ ] Value head achieves >70% accuracy on simple winning positions
- [ ] Training and validation loss curves show convergence
- [ ] Gradient analysis shows healthy gradient flow to value head

### Medium-term (3-4 weeks)
- [ ] Value head performance improves across all game stages
- [ ] Model can correctly evaluate obvious winning positions
- [ ] No evidence of overfitting or underfitting

## Key Questions to Answer

1. **Are gradients flowing to the value head?** (Gradient monitoring)
2. **Are activations healthy?** (Activation monitoring)
3. **Does performance vary by game stage?** (Position type analysis)
4. **Is the value head architecture appropriate?** (Architecture experiments)
5. **Can the model learn simple positions?** (Easy position training)

## Technical Notes

### Value Loss Interpretation
- MSE loss of 0.2 means predictions are off by √0.2 ≈ 0.45 on average
- This suggests predictions in 0.3-0.7 range (close to random)
- For binary classification (0.0 vs 1.0), this is poor performance

### Monitoring Integration
- Monitoring tools are lightweight and don't require TensorBoard
- Can be easily integrated into existing training loop
- Provides real-time feedback during training

### Data Availability
- 1.2M games available (4.8M positions with augmentation)
- Sufficient for focused experiments on specific position types
- Can create targeted datasets for different training approaches

## References

- **Current Model**: `hex_ai/models.py` - TwoHeadedResNet with simple value head
- **Training Code**: `hex_ai/training.py` - Trainer class with monitoring hooks
- **Data Pipeline**: `hex_ai/data_pipeline.py` - StreamingSequentialShardDataset
- **Inference**: `scripts/simple_inference_cli.py` - Manual position testing
- **Recent Plan & Tools**: `write_ups/value_head_debugging_summary.md` - **(this document, most up-to-date)**
- **Older Plan**: `write_ups/value_net_overfitting_plan.md`
- **Older Debugging Log**: `write_ups/debugging_value_head_performance.md`

---

## Updates Log

### 2025-01-XX
- Added gradient and activation monitoring tools (`GradientMonitor`, `ActivationMonitor`, `ValueHeadAnalyzer`)
- Created comprehensive analysis plan and summary document
- Implemented game stage analysis script (`analyze_value_by_game_stage.py`)
- Implemented easy position training script (`train_on_easy_positions.py`)
- Created monitored training script (`train_with_monitoring.py`)
- Identified key next steps and success criteria 

## Questions / Follow-ups

- **Policy label for terminal positions:**
  - Currently, terminal positions (no next move) use an all-zeros policy vector.
  - Though PolicyValueLoss actually already handles this via masking, with:
    - policy_loss = torch.tensor(0.0, device=policy_pred.device, requires_grad=True)
  - So this seems to be ok.

## Latest Analysis Results and Observations (23rd July 2025)

- **Summary of Results:**
  - None of the recent models (across all tested hyperparameters and checkpoints) are predicting well at any stage of the game.
  - Value head predictions are poor even on simple/final positions (e.g., fully-connected boards with a clear winner).
  - This is true even for checkpoints that have seen 5M unaugmented (20M augmented) examples.
  - This is surprising because:
    1. The policy network learns much faster and with less data.
    2. Previous projects achieved reasonable value head performance with less data.
    3. Intuitively, the network should be able to correlate with very easy/final positions after this much training.
  - The different models do produce different value predictions, and training loss does decrease somewhat after training starts, indicating that gradients are flowing and some learning is happening—but not in a way that leads to meaningful value prediction.

- **Current Hypothesis:**
  - There is still a fundamental issue with value head training, data, or architecture that is preventing the network from learning even the simplest value correlations.

- **Next Steps (Proposed):**
  - **Train on only-final-positions:**
    - Restrict the training set to only final positions (where the winner is known and the board is fully connected).
    - This experiment will isolate whether the value head can learn the simplest possible value mapping.
    - If the value head cannot learn this, the problem is almost certainly in the data pipeline, label assignment, or value head architecture.
    - If it can learn this, the issue may be with more complex positions, label noise, or the interaction between policy and value heads.
  - **Why this is promising:**
    - This is a minimal, controlled diagnostic. It should be easy for the network to achieve high accuracy on final positions if the pipeline is correct.
    - The outcome will provide a clear signal about where to look next (data/labels vs. model/training dynamics).
    - The winner of final does not depend on whose move it is, so it helps us confirm whether model training problems are related to the network knowing whose turn it is (likely not, as you'd get some predictive power not even knowin whose turn it is).

- **Action:**
  - Run the "train on only-final-positions" experiment and analyze the results.
  - Depending on the outcome, either focus on data/label debugging or move on to more complex diagnostic experiments.