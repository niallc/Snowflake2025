# Debugging Value Head Performance in Hex AI

**Note:** This document is now superseded by [value_head_debugging_summary.md](value_head_debugging_summary.md), which contains the most up-to-date plan and toolset for value head debugging. See also [value_net_overfitting_plan.md](value_net_overfitting_plan.md) for related work.

## Context

Recent experiments show that the value head of the Hex AI ResNet model is not learning as expected:
- Training loss drops quickly, but validation loss remains high (overfitting symptoms).
- Even on simple or terminal positions, the value head's predictions are not reliably correlated with the true winner.
- The policy head appears to learn as expected, so the network is capable of extracting useful features from the data.
- Data augmentation and dataset shuffling (to avoid game fingerprinting) have reduced overfitting, but value head performance is still poor.

## Hypotheses

1. **Data/Label Issues**: The value label may be incorrectly assigned, corrupted, or misaligned during processing or augmentation.
2. **Model Architecture Issues**: The value head may be missing a critical layer, or the input channels may be misused.
3. **Training Pipeline Issues**: The loss function, optimizer, or batching may be mishandling the value head or its gradients.
4. **Insufficient Signal**: The value head may need more/different data, but this is less likely given the scale of data and policy head success.

## Debugging Plan

### 1. **Manual Inspection of Data Passed to the Model**
- **Goal:** Ensure that the 3-channel input and value labels are correct just before entering the model.
- **Steps:**
  1. Write a script to extract and print (or visualize) 10 random (board, policy, value) tuples from the `StreamingProcessedDataset` and `StreamingAugmentedProcessedDataset`.
  2. For each sample, print:
     - The board (as a 3x13x13 tensor, with channel meanings explained).
     - The value label.
     - The player-to-move channel (should be 0.0 for blue, 1.0 for red).
     - The metadata (if available: winner, position in game, etc.).
  3. For terminal positions, check that the value label matches the winner and that the board is a valid end state.
  4. For augmented samples, check that color-swapping symmetries also swap the value label (see `create_augmented_values` in `hex_ai/data_utils.py`).

### 2. **Automated Consistency Checks on a Large Sample**
- **Goal:** Systematically check for label or data corruption at scale.
- **Steps:**
  1. Write a script to iterate over 10,000+ samples from the dataset.
  2. For each sample, check:
     - The value label is in [0.0, 1.0].
     - The board is a valid Hex board (no illegal piece counts, etc.).
     - The player-to-move channel is consistent with the board state.
     - For augmented samples, the value label is swapped when colors are swapped.
  3. Log and summarize any inconsistencies.

### 3. **Model Architecture Review**
- **Goal:** Ensure the value head is correctly implemented and receives the right features.
- **Steps:**
  1. Review `hex_ai/models.py`:
     - Confirm the input layer expects 3 channels.
     - Confirm the value head is a linear layer on top of the global pooled features.
     - Check for missing activations, normalization, or other architectural issues.
  2. Compare with standard architectures for similar tasks (AlphaZero, etc.).
  3. Optionally, try adding a nonlinearity (e.g., ReLU or Tanh) to the value head output if not present.

### 4. **Training Pipeline and Loss Function Review**
- **Goal:** Ensure the value loss is computed and backpropagated correctly.
- **Steps:**
  1. Review `PolicyValueLoss` in `hex_ai/training.py`:
     - Confirm that value loss is always computed as MSE between predicted and true value.
     - Confirm that the value target is always a float in [0, 1].
     - Check for any masking or skipping of value loss.
  2. Check optimizer parameter groups: value head should not be excluded or have a learning rate of zero.
  3. Check for gradient flow to the value head (e.g., by inspecting gradients after a backward pass).

### 5. **Sanity-Check Inference on Known Positions**
- **Goal:** Ensure the trained model produces sensible value predictions on simple, known boards.
- **Steps:**
  1. Use the inference code (`hex_ai/inference/simple_model_inference.py`) to run the model on:
     - An empty board (should be ~0.5 if the model is unconfident).
     - A board with a clear win for blue (should be close to 1.0).
     - A board with a clear win for red (should be close to 0.0).
  2. If the value head fails these tests, the issue is likely in data, architecture, or training.

### 6. **Check for Data Leakage or Label Leakage**
- **Goal:** Ensure that the value head is not learning to memorize game IDs or other artifacts.
- **Steps:**
  1. Confirm that shuffling and augmentation prevent multiple positions from the same game appearing in the same batch or shard.
  2. Check that no game-unique features (e.g., game ID, move number) are present in the input.

### 7. **Compare with Policy Head**
- **Goal:** Understand why the policy head learns but the value head does not.
- **Steps:**
  1. Compare the gradients and activations for both heads during training.
  2. Check if the value head is underfitting (very small gradients, stuck predictions) or overfitting (memorizing training data).

### 8. **Experiment with Loss Weights and Learning Rates**
- **Goal:** Test if the value head is under- or over-emphasized during training.
- **Steps:**
  1. Try increasing/decreasing the value loss weight.
  2. Try increasing the value head's learning rate.
  3. Observe effects on training/validation loss and value predictions.

## References to Code
- Data pipeline: `hex_ai/data_pipeline.py`, `hex_ai/data_utils.py`, `hex_ai/data_processing.py`
- Model: `hex_ai/models.py`
- Training: `hex_ai/training.py`
- Inference: `hex_ai/inference/simple_model_inference.py`, `hex_ai/inference/model_wrapper.py`
- Data format: `hex_ai/data_formats.md`
- **See also:** [value_net_overfitting_plan.md](value_net_overfitting_plan.md) (**newest, most up-to-date plan**)
- **See also:** [value_head_debugging_summary.md](value_head_debugging_summary.md) (more recent summary)

## Next Steps
- Start with manual inspection (Step 1) and automated checks (Step 2).
- If data and labels are correct, proceed to model and training pipeline review.
- Document findings and update this plan as new issues or hypotheses arise. 

---

## Initial Automated Review Findings

### Value Label Assignment
- Value labels are assigned as 0.0 (blue win) or 1.0 (red win) in `hex_ai/data_utils.py`:
  ```python
  value_target = 0.0 if winner_from_file == "1" else 1.0  # BLUE=0.0, RED=1.0
  ```
- Augmentation swaps the value label for color-swapping symmetries:
  ```python
  def create_augmented_values(value: float) -> list[float]:
      return [
          value,          # Original
          value,          # 180° rotation (no color swap)
          1.0 - value,    # Long diagonal reflection + color swap
          1.0 - value     # Short diagonal reflection + color swap
      ]
  ```
- Data pipeline ensures value is always a float and checks for type errors.

### Model Value Head
- The value head is a linear layer on top of global pooled features:
  ```python
  self.value_head = nn.Linear(CHANNEL_PROGRESSION[3], VALUE_OUTPUT_SIZE)
  ...
  value_logit = self.value_head(x)  # (batch_size, 1)
  ```
- No activation is applied in the model; sigmoid is applied at inference:
  ```python
  value = torch.sigmoid(value_logit).item()  # Probability red wins
  ```
- Loss is MSE between value logit and true value:
  ```python
  value_loss = self.value_loss(value_pred.squeeze(), value_target.squeeze())
  ```

### CLI and Sanity Checks
- The CLI script (`scripts/simple_inference_cli.py`) runs inference and prints the value as a probability.
- There are scripts for inspecting batches, extracting samples, and overfitting tiny datasets (see `scripts/inspect_training_batch.py`, `scripts/extract_error_sample_from_pkl.py`, `scripts/overfit_tiny_dataset.py`).

### Next Steps
- Use `scripts/inspect_training_batch.py` or similar to check random samples for label/data consistency.
- Use the CLI to run inference on known positions and add results to this document.
- Try overfitting a tiny dataset to check if the value head can learn at all.

### Example CLI Usage

```sh
resCollDir="checkpoints/hyperparameter_tuning/"
resDirTag="shuffled_sweep_run_0_learning_rate0.001_batch_size256_max_grad_norm20_dropout_prob0_weight_decay0.0001_value_learning_rate_factor0.2_value_weight_decay_factor3.0_20250721_064254/"
resDir=${resCollDir}${resDirTag}
realGameB="http://www.trmph.com/hex/board#13,a6i2d10d9f8e9g9g10i9h9i8h8i7j4g6g7f7h6g8f10h7i10j10j11h10g4e5e4f4g12i11g2h2g3h4h13g11f12f11e12l11k12h12g13l12k10i12h3j3i4i3h5g5j2k2j12i13h11j9f3d5k1l1"
PYTHONPATH=. python scripts/simple_inference_cli.py \
    --trmph ${realGameB} \
    --model_dir ${resDir} \
    --model_file best_model.pt \
    --device mps
```
This runs inference on a known blue win and prints the value head’s prediction. 

---

## Manual Inspection of Dumped Training Batch

To directly inspect the data passed to the network, we:
- Added a debug block to `Trainer.train_epoch` to dump the first batch of epoch 0 to `analysis/debugging/value_head_performance/batch0_epoch0.pkl`.
- Created `scripts/analyze_dumped_batch.py` to load and display samples from this batch, using the project’s display and conversion utilities.

**Usage example:**
```python
from scripts.analyze_dumped_batch import main
all_data = main("analysis/debugging/value_head_performance/batch0_epoch0.pkl", max_samples=6, interactive=False)
```
This prints a summary of the first 6 samples and returns a dictionary with both raw and reformatted data for further interactive analysis.

### Visual Inspection Findings
- **Shuffling:** The 5th position in the batch is completely different from the first 4, confirming that shuffling is working as intended.
- **Augmentation:** For each original position, the two reflections swap the value label and player-to-move channel, while the rotation does not. This matches the intended augmentation logic.
- **Player-to-move:** The player-to-move channel appears correct and changes appropriately with augmentation.
- **Value label:** The value label is swapped as expected for color-swapping augmentations.

**Conclusion:**
- The data pipeline, shuffling, and augmentation all appear correct on visual/manual inspection.
- No further automated checks are planned at this stage, as existing code and tests already validate the main invariants.

If further issues or hypotheses arise, additional checks or scripts can be added as needed. 