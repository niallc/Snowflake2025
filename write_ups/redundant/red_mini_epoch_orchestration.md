# Mini-Epoch Orchestration Design for Hex AI Training Pipeline

## Motivation

With large datasets (e.g., 97M+ samples, augmented 4x), it is desirable to:
- Run validation and checkpointing more frequently than once per full epoch.
- Monitor training progress and catch issues early.
- Avoid long periods between checkpoints (risk of lost progress).
- Maintain optimizer and model state across the entire training run for efficiency and convergence.

## Requirements

- **Mini-Epochs:** Support running validation and checkpointing every N batches (mini-epoch), not just at epoch boundaries.
- **Stateful Training:** Keep a single Trainer/model/optimizer instance for the whole run (do not restart from scratch each mini-epoch).
- **No Data Duplication:** Do not re-implement batching or chunking logic; leverage existing DataLoader and StreamingAugmentedProcessedDataset.
- **Configurable Frequency:** Allow user to set mini-epoch size (e.g., every 200k samples, or every 500-1000 batches).
- **Separation of Concerns:** Keep mini-epoch orchestration logic outside the Trainer class for modularity and testability.
- **Robustness:** Ensure correct handling of end-of-epoch and partial mini-epochs.
- **Logging/Monitoring:** Log validation/checkpointing events with batch/sample counts for traceability.

## High-Level Design

### 1. Core Concepts
- **Batch:** A set of examples processed in one forward/backward pass (e.g., 256 positions).
- **Mini-Epoch:** A fixed number of batches (e.g., 500-1000) after which validation/checkpointing is triggered.
- **Epoch:** A full pass over the training dataset.

### 2. Components
- **Trainer:** Handles model, optimizer, and training/validation logic for one or more epochs. Remains unchanged except for a new `train_on_batches` method.
- **MiniEpochOrchestrator (Wrapper):**
  - Accepts a Trainer, DataLoader, and mini-epoch size (in batches).
  - Iterates over the DataLoader, processing batches in groups of N (mini-epoch).
  - After each mini-epoch, calls Trainer.validate() and handles checkpointing.
  - Handles end-of-epoch and partial mini-epochs gracefully.

### 3. Control Flow
1. For each epoch:
    - For each mini-epoch (N batches):
        - Call `Trainer.train_on_batches(mini_epoch_batches)`
        - Call `Trainer.validate()`
        - Save checkpoint
    - At end of epoch, ensure any remaining batches are processed and checkpointed.

## Implementation Plan

### 1. Extend Trainer
- Add `train_on_batches(self, batch_iterable)` method:
    - Processes a list or iterator of batches (forward, backward, optimizer step, loss tracking).
    - Returns training metrics for the mini-epoch.
- (Optionally) Refactor `train_epoch` to use `train_on_batches` internally for code reuse.

### 2. Implement MiniEpochOrchestrator
- Class or function that:
    - Accepts Trainer, DataLoader, mini-epoch size, num_epochs, etc.
    - For each epoch, iterates over DataLoader in chunks of N batches.
    - After each chunk, calls validation and checkpointing.
    - Logs progress (batches processed, samples seen, validation loss, checkpoint path).
    - Handles end-of-epoch and partial mini-epochs.

### 3. Configuration
- Allow mini-epoch size to be set via config or CLI (e.g., `mini_epoch_batches=500`).
- Default to a reasonable value (e.g., 500-1000 batches, or enough to cover ~200k samples).

### 4. Testing
- Unit test `train_on_batches` with synthetic data.
- Test orchestrator with a small dataset to ensure correct validation/checkpointing frequency and state continuity.

## Tricky Implementation Considerations

- **Partial Mini-Epochs:**
    - At the end of an epoch, the last mini-epoch may have fewer than N batches. Ensure these are still processed and validated.
- **DataLoader Exhaustion:**
    - Use `StopIteration` to detect end of DataLoader and handle gracefully.
- **State Continuity:**
    - Do not re-instantiate Trainer/model/optimizer between mini-epochs. Only checkpoint and optionally reload if resuming after interruption.
- **Validation/Checkpoint Frequency:**
    - Avoid excessive checkpointing (disk usage) or validation (runtime cost). Choose a frequency that balances safety and efficiency.
- **Logging:**
    - Log batch/sample counts, validation losses, and checkpoint paths for traceability.
- **Compatibility:**
    - Ensure orchestrator works with both single-worker and multi-worker DataLoader setups.
- **Interrupt Handling:**
    - Optionally, support graceful shutdown and resume from last checkpoint.

## Example Pseudocode

```python
class MiniEpochOrchestrator:
    def __init__(self, trainer, train_loader, val_loader, mini_epoch_batches, num_epochs):
        self.trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.mini_epoch_batches = mini_epoch_batches
        self.num_epochs = num_epochs

    def run(self):
        for epoch in range(self.num_epochs):
            batch_iter = iter(self.train_loader)
            batch_count = 0
            while True:
                mini_epoch_batches = []
                try:
                    for _ in range(self.mini_epoch_batches):
                        mini_epoch_batches.append(next(batch_iter))
                        batch_count += 1
                except StopIteration:
                    pass  # End of epoch
                if not mini_epoch_batches:
                    break
                self.trainer.train_on_batches(mini_epoch_batches)
                val_metrics = self.trainer.validate()
                self.trainer.save_checkpoint(...)
                # Log progress, handle early stopping, etc.
                if len(mini_epoch_batches) < self.mini_epoch_batches:
                    break  # End of epoch
```

## Summary

This design enables robust, maintainable, and idiomatic mini-epoch orchestration for large-scale training. It leverages existing batching/chunking logic, maintains optimizer/model state, and provides flexible, testable control over validation and checkpointing frequency. 