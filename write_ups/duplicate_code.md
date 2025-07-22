# Duplicate Code in Hex AI Project

This document logs known areas of code duplication in the project, with brief notes on their differences and any important context.

---

## 1. Dataset Classes

### StreamingAugmentedProcessedDataset
- Loads as many files as needed to reach the sample limit, storing all examples in memory.
- Supports random access (`__getitem__`, `__len__`).
- Skips files with errors (logs a warning, continues).
- Used in the original pipeline.

### StreamingSequentialShardDataset
- Loads one shard (file) at a time, yielding examples sequentially (never more than one shard in memory).
- Strictly sequential access (`__iter__` only, no random access).
- Fails loudly on any file error (no silent skipping).
- Designed for large-scale, memory-efficient, and robust streaming.
- More detailed progress logging when `verbose=True`.

**Key difference:**
- The first is for random access and in-memory loading; the second is for streaming, sequential, memory-efficient loading.

---

## 2. Trainer Methods

### Trainer.train
- The original training loop method.
- Handles epochs, batching, and validation internally.
- May be less modular or flexible for advanced orchestration.

### Trainer.train_on_batches
- A newer or alternative training loop.
- Designed for use with mini-epoch orchestrators or more granular control.
- May allow for more flexible interruption, logging, or integration with orchestration logic.

**Key difference:**
- `train` is a monolithic, self-contained loop; `train_on_batches` is more modular and designed for orchestration.

---

**Note:**
- These duplications may be temporary as the codebase is modernized and refactored. This document should be updated as further consolidation or cleanup occurs. 