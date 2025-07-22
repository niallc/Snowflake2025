# Chunk Loading in PyTorch Dataset: Streaming Sequential Shard Design (2024-07)

## Motivation

- Our data is pre-shuffled and stored in files ("shards"), each a random sample of the full dataset.
- **Key requirement:** Never load more than 1-2 shards (whole files) into memory at once, to support arbitrarily large datasets and minimize memory usage.
- **Access pattern:** Strictly sequential (no random access), for simplicity, efficiency, and correctness.

## Final Design

### 1. **Streaming, Sequential Shard Loading**
- The dataset keeps at most 1-2 shards (files) in memory at any time.
- Each shard is loaded fully into memory (typical size: 100k-200k examples, easily fits in RAM).
- When the current shard is exhausted, it is released from memory and the next shard is loaded.
- The DataLoader yields training batches (e.g., of size 256) from the current shard.
- No attempt is made to load all data into memory at once.

### 2. **Augmentation**
- Augmentation is applied on-the-fly as before, to each batch or example as needed.
- Only the current shard's data is ever held in memory.

### 3. **Player Channel**
- The player-to-move channel is always added, even when augmentation is disabled, ensuring model compatibility.

### 4. **Validation and Training**
- Both training and validation datasets use the same loader, with augmentation enabled for training and disabled for validation.

### 5. **Strict Error Handling**
- All error handling is strict: any corrupted or missing file causes the run to fail immediately, to avoid masking bugs or data issues.
- Silent failures are not acceptable. This is not a datacenter-scale distributed system; every file should be perfect.

### 6. **Testing**
- The test suite must ensure correct sequential loading, augmentation, error handling, and model compatibility.

## Example Usage

```python
files = discover_processed_files("data/processed/shuffled")
ds = StreamingSequentialShardDataset(files, max_examples_unaugmented=16000, enable_augmentation=True)
loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)
```

## __getitem__ vs __iter__ Design Decision

- **__getitem__ (random access):**
  - Pros: Compatible with standard PyTorch DataLoader, supports random access.
  - Cons: Requires complex logic to map indices to files and offsets, and to efficiently load only the needed shard. Random access is not needed for our use case and introduces major complexity and inefficiency.
- **__iter__ (sequential access):**
  - Pros: Simple, natural fit for our strictly sequential, streaming use case. Easy to reason about, maintain, and test. No need to map indices or support random access.
  - Cons: Not compatible with DataLoader's random sampling or shuffling, but this is not needed for our pipeline.
- **Recommendation:** Use `__iter__` and a custom DataLoader or generator for strictly sequential streaming. This is the simplest, most robust, and most maintainable approach for our requirements.

## Terminology
- **Shard:** A single pre-shuffled data file (e.g., 100k-200k examples).
- **Batch:** The set of examples passed to the network at each training step (e.g., 256 examples).
- **Chunk:** (If used) A portion of a shard held in memory at once. For now, we load whole shards at a time.

## What We No Longer Support (and Why)

- **No random access:** Only sequential streaming is supported. Random access is explicitly not supported and should not be implemented.
- **No loading all data into memory:** The loader must never attempt to load all data at once.
- **No runtime shuffling:** All shuffling is done during preprocessing.

## Summary

This design is:
- Simple, robust, and easy to maintain
- Efficient for pre-shuffled, file-based datasets
- Fully compatible with PyTorch DataLoader and model requirements (with sequential-only access)
- Extensively tested for all relevant edge cases
- Strict in error handling: any data error causes immediate failure

If future requirements change (e.g., truly massive datasets, or a need for random access or runtime shuffling), the design can be extended. For now, this approach is optimal for our project. 