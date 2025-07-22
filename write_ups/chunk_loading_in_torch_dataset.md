# Chunk Loading in PyTorch Dataset: Sequential, Simple, and Standard

## Motivation

For datasets that are already shuffled during preprocessing, the simplest, most efficient, and most robust approach is to read data sequentially. This avoids unnecessary complexity, IO, and memory usage, and is fully supported by PyTorch and standard ML practice when the data is already randomized.

## Requirements

- **Sequential Access:** Read examples in order from one or more files, with no additional shuffling.
- **Chunked Access (Optional):** If needed, support chunked reading for efficiency, but always in sequential order.
- **No Unnecessary Complexity:** Do not shuffle files or examples at runtime if the data is already randomized.
- **Simplicity and Maintainability:** Keep the code as simple, idiomatic, and testable as possible.
- **Extensible:** If shuffling is ever needed, it can be added as an explicit, optional feature.

## High-Level Design

### 1. Sequential Example Loading

- At initialization, select the minimum number of files needed to reach the requested number of examples (`max_examples`).
- Read examples sequentially from these files, stopping when the limit is reached.
- Store all loaded examples in a list (or other simple structure).
- No shuffling is performed at any stage.

### 2. PyTorch Dataset Interface

- Implement `__len__` to return the number of loaded examples.
- Implement `__getitem__` to return the `idx`-th example from the list.
- Use `DataLoader(..., shuffle=False)` to preserve sequential access.

### 3. (Optional) Chunking

- If chunked access is needed for efficiency, implement chunking as a simple slice over the loaded list.
- Always process chunks in sequential order.

## Pseudocode/Code Sketch

```python
class SequentialDataset(torch.utils.data.Dataset):
    def __init__(self, data_files, max_examples=None):
        self.examples = []
        for file_path in data_files:
            with gzip.open(file_path, 'rb') as f:
                data = pickle.load(f)
            for ex in data['examples']:
                if max_examples is not None and len(self.examples) >= max_examples:
                    break
                self.examples.append(ex)
            if max_examples is not None and len(self.examples) >= max_examples:
                break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]  # Sequential access
```

## Best Practices

- **No shuffling unless explicitly requested.**
- **Short, focused functions:** Each function should do one thing.
- **Testability:** The dataset can be tested with a few small files and known data.
- **Extensible:** If shuffling is ever needed, add it as an explicit, opt-in feature.

## Error Handling

- If a file is missing or corrupted, log a warning and skip it.
- If an index is out of range, raise `IndexError`.

## Extensibility

- **Shuffling:** If needed, add a `shuffle=True` option that shuffles the loaded examples after reading.
- **Prefetching:** For very large datasets, implement chunked or streaming reading, but always in sequential order unless shuffling is explicitly requested.

## Summary

For pre-shuffled datasets, the most robust, efficient, and maintainable approach is to read examples sequentially from disk, with no additional shuffling. This is fully supported by PyTorch and is the recommended baseline for all such use cases. Only add shuffling if and when it is truly needed. 