# Chunk Loading in PyTorch Dataset: Final Sequential Streaming Design

## Motivation

For this project, our data is pre-shuffled and stored in files such that each file is already a random sample of the full dataset. Our goal is to maximize simplicity, efficiency, and maintainability by reading data sequentially, with no runtime shuffling or chunking logic.

## Final Design

- **Sequential Loading:**
  - The dataset loads as few files as needed to reach the requested number of unaugmented examples (`max_examples_unaugmented`).
  - Examples are read in order from the files provided.
  - All loaded examples are stored in a list in memory.

- **Augmentation:**
  - If augmentation is enabled, each base example yields 4 augmented examples (rotations/reflections, etc.).
  - Augmentation is applied on-the-fly in `__getitem__`.

- **Player Channel:**
  - The player-to-move channel is always added, even when augmentation is disabled, ensuring model compatibility.

- **Validation and Training:**
  - Both training and validation datasets use the same loader, with augmentation enabled for training and disabled for validation.

- **Error Handling:**
  - Corrupted or missing files are skipped with a warning, but valid examples from other files are loaded.

- **Testing:**
  - The test suite ensures correct sequential loading, augmentation, error handling, and model compatibility.

## Example Usage

```python
# Load up to 16,000 examples from a shuffled data directory
files = discover_processed_files("data/processed/shuffled")
ds = StreamingAugmentedProcessedDataset(files, max_examples_unaugmented=16000, enable_augmentation=True)
```

## What We No Longer Support (and Why)

- **No Random Access:**
  - The dataset is optimized for sequential access. While `__getitem__` works for any index, the design and performance are for sequential streaming.
  - This is a deliberate choice: our training and validation always iterate sequentially, and random access is not needed for our use case.

- **No Runtime Shuffling:**
  - We do not shuffle files or examples at runtime. All shuffling is done during preprocessing, and each file is already a random sample.
  - This avoids unnecessary complexity and IO, and is safe because our data is pre-randomized.

- **No Chunking Logic:**
  - We do not implement chunked loading or chunk boundaries. All loaded examples are kept in a single list, and batching is handled by the PyTorch DataLoader.
  - This is efficient for our file sizes and use case, and avoids the complexity of chunk management.

- **No Multi-worker or Prefetching:**
  - The current design is single-threaded and loads all needed examples into memory at once. For our data sizes, this is fast and simple.
  - If we ever need to scale to much larger datasets, we can revisit chunking or streaming logic.

## Summary

This design is:
- Simple, robust, and easy to maintain
- Efficient for pre-shuffled, file-based datasets
- Fully compatible with PyTorch DataLoader and model requirements
- Extensively tested for all relevant edge cases

If future requirements change (e.g., truly massive datasets, or a need for random access or runtime shuffling), the design can be extended. For now, this approach is optimal for our project. 