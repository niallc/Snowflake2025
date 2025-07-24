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

## Targeted Value Head Sanity Check: Final Positions Only (2025-07-23)

### Motivation
The value head should at least be able to learn trivial value targets if the data and architecture are not fundamentally broken. To test this, we constructed a dataset containing only the final positions from each game (where the winner is known and the value target is unambiguous), and trained the value head on this data.

### Implementation
- **Data Extraction:**
  - Refactored the data processing pipeline to allow extraction of only the final (or penultimate) positions from each game.
  - Added a `position_selector` argument to the batch processor and CLI, so we can easily generate datasets of only final positions.
  - Output files are written to a new directory (e.g., `data/processed/final_only/`).

### Result
- **Value Loss Drops Rapidly:**
  - When training on the final-positions-only dataset, the value head loss drops extremely quickly and reaches excellent values (near zero) within a few epochs.
  - This confirms that the value head and training pipeline are capable of learning trivial value targets when the problem is easy and unambiguous.
- **Interpretation:**
  - The value head is not fundamentally broken; it can learn when the data is simple.
  - The persistent high value loss on the full dataset is likely due to the increased complexity/ambiguity of intermediate positions, not a bug in the value head or data pipeline.

### Next Steps
- Proceed to test on penultimate positions and more complex subsets to further probe the limits of the value head.
- Use these results to inform further architectural or data-centric debugging.

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

### 2025-07-23
- Ran monitoring-enabled training with 500k examples and augmentation
- Value head still fails to learn, predictions remain near 0.5–0.7
- Gradients and activations are healthy
   - [Batch 2450] Gradient norms - Policy: 0.089085, Value: 0.283956, Shared: 0.041630
- Fallback to StreamingAugmentedProcessedDataset for monitoring
- Next steps: focus on direct inspection of value labels and predictions

### 2025-07-23: Gradient and Activation Monitoring — Investigation Complete

After fixing the monitoring code to log during the actual training loop, we confirmed:
- **Value head gradients are dynamic and healthy:**
    - Example values: 0.31, 1.12, 0.29, 0.51, 0.64, ... (first batch: 50.92, then quickly stabilizes)
    - Gradients fluctuate batch-to-batch, with no sign of vanishing or exploding
- **Value head activations are also dynamic:**
    - Example means: -2.47 (first batch), 0.49, 0.50, 0.51, ...
    - Activations vary across batches, with plausible mean and std values
- **No evidence of dead neurons, stuck optimizer, or broken gradient flow**

**Conclusion:**
- The value head is being updated and is not dead or stuck.
- The root cause of poor value head learning is almost certainly in the data/label pipeline or loss/target wiring, not in the optimizer, gradients, or network health.

**Next steps:**
- Focus on direct inspection of value labels, label distribution, and loss/target correctness.
- Consider running a minimal “sanity check” experiment with a trivial value head and a tiny dataset to confirm the pipeline end-to-end.

**Status:**
- Gradient and activation monitoring investigation is complete and marked as resolved.

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

## Training vs Inference Discrepancy Investigation (2025-07-23, 11pm+)

### Key Finding: Training Loss vs Inference Performance Mismatch

**Observation**: Despite achieving very low training loss (~0.0001) on the final-positions-only dataset, the saved model produces ~50% value estimates during inference on the same types of positions.

**Evidence**:
- Training loss drops extremely quickly to near-zero values
- Model checkpoints show excellent training metrics
- Inference on clear blue-win final positions yields ~49.4% instead of expected ~0%
- Multiple checkpoints tested with consistent ~50% output
- Policy head works correctly during inference

### Systematic Investigation Results

**Preprocessing Pipeline Verification**:
- Refactored `hex_ai/inference/simple_model_inference.py` to use exact same preprocessing as training
- Replaced magic numbers with constants from `hex_ai/config.py` for clarity
- Created comprehensive debugging in `scripts/debug_inference.py`
- Verified that training and inference produce identical input tensors
- Confirmed that both methods produce identical model outputs

**Player-to-Move Channel Consistency**:
- Identified and fixed player-to-move channel inconsistency for finished positions
- Training data has player-to-move set to loser (next player)
- Modified inference to match training: set player-to-move to loser for finished positions
- This ensures input consistency between training and inference

**Root Cause Analysis**:
The issue is **not** in the preprocessing pipeline - we're feeding the model exactly the same input it was trained on, but still getting unexpected outputs.

### Current Hypotheses

1. **Input Replication Issue**: We may not be replicating the network inputs properly despite identical tensors
2. **Model Saving Issue**: Unlikely, as policy head works fine during inference
3. **Hidden Corruption**: There may be subtle differences between training and inference environments that we haven't identified

### Next Steps: Systematic Training-to-Inference Debugging

**Plan**: Since the model achieves very low training loss but produces unexpected inference results, we need to:

1. **Capture Training State**: Extract the exact input tensors, model state, and network outputs during training when loss is very low
2. **Reproduce Training Conditions**: Load the exact model state and feed the exact same inputs that produced the low loss
3. **Gradual Modification Test**: Systematically change one aspect at a time to identify what breaks the prediction
4. **Root Cause Identification**: Pinpoint whether the issue is in model architecture, training data, or loss function

**Implementation Strategy**:
- Add detailed logging to training loop to capture input/output pairs
- Create debugging utilities to load specific checkpoints and reproduce exact conditions
- Build systematic testing framework to isolate variables
- Document findings to prevent similar issues in future

**Expected Outcome**: Identify the exact point where training and inference diverge, providing clear guidance for fixing the value head training or inference pipeline.

## TODO: Clean up temporary batch dumping logic (2025-07-23)

- Temporary debug code has been added to `train_on_batches` in `hex_ai/training.py` to dump the first batch of epoch 0, mini_epoch 0 for value head debugging.
- This includes device, cpu, and numpy forms of the tensors for maximum flexibility during inference debugging.
- Once the value head inference/training pipeline is verified, this code should be removed or refactored.
- See also: inference script and tests to be written for this pipeline.