# Chunk Loading in PyTorch Dataset: Clean Design

## Motivation

When working with large datasets, it is efficient to load data in chunks (batches of samples) rather than all at once. This is especially important for streaming or disk-based datasets. The goal is to design a clean, idiomatic, and maintainable chunked dataset for PyTorch, with clear separation of concerns and robust error handling.

## Requirements

- **Chunked Access:** Map a global index to a chunk number and an index within that chunk.
- **On-Demand Loading:** Load the appropriate chunk into memory only when needed.
- **Separation of Concerns:** Keep chunk mapping, chunk loading, and data transformation as separate, testable units.
- **No Code Smells:** Avoid long functions, mid-function imports, and tightly coupled logic.
- **Extensible:** Easy to add features like shuffling, prefetching, or multi-worker support.

## High-Level Design

### 1. Index Mapping

- Given a global index `idx`, compute:
  - `chunk_number = idx // chunk_size`
  - `index_in_chunk = idx % chunk_size`

### 2. Chunk Loading

- If the requested chunk is not already loaded, call a utility function to load it.
- The chunk loader should:
  - Determine which files and which examples are needed for the chunk.
  - Load and transform only those examples.
  - Store the loaded chunk and its number for future accesses.

### 3. Data Transformation

- All data transformation (e.g., augmentation, tensorization) should be handled in a dedicated function, not inline in the chunk loader.

### 4. Error Handling

- Handle missing/corrupted files gracefully.
- If a chunk cannot be loaded, raise a clear error or return a fallback.

## Pseudocode/Code Sketch

```python
class ChunkedDataset(torch.utils.data.Dataset):
    def __init__(self, data_files, chunk_size, ...):
        self.data_files = data_files
        self.chunk_size = chunk_size
        self.current_chunk = None
        self.current_chunk_number = None
        # ... other init ...

    def __len__(self):
        # Return total number of samples (augmented if needed)
        ...

    def __getitem__(self, idx):
        chunk_number, index_in_chunk = self._map_index(idx)
        if self.current_chunk_number != chunk_number:
            self.current_chunk = self._load_chunk(chunk_number)
            self.current_chunk_number = chunk_number
        return self.current_chunk[index_in_chunk]

    def _map_index(self, idx):
        chunk_number = idx // self.chunk_size
        index_in_chunk = idx % self.chunk_size
        return chunk_number, index_in_chunk

    def _load_chunk(self, chunk_number):
        # Determine which files/examples are needed for this chunk
        # Load and transform only those examples
        # Return a list of samples for this chunk
        ...

    def _transform_example(self, example):
        # Apply augmentation, tensorization, etc.
        ...
```

## Best Practices

- **No mid-function imports:** All imports at the top of the file.
- **Short, focused functions:** Each function should do one thing.
- **Clear state management:** Track which chunk is loaded and its number.
- **Testability:** Each utility function should be independently testable.
- **Extensible:** Easy to add shuffling, prefetching, etc.

## Error Handling

- If a file is missing or corrupted, log a warning and skip it.
- If a chunk cannot be filled (e.g., not enough data), return a partial chunk or raise an error.
- If an index is out of range, raise `IndexError`.

## Extensibility

- **Shuffling:** Shuffle file list or chunk order at epoch start.
- **Prefetching:** Add a background thread or process to pre-load the next chunk.
- **Multi-worker:** Ensure chunk loading is thread/process safe.

## Implementation and Testing Progress (July 2024)

### Multi-file Chunking
- The dataset now robustly supports multi-file chunking.
- Tests create temporary .pkl.gz files on the fly (using pytest's `tmp_path`), each with unique examples.
- The test verifies that all examples are seen in the correct order, even when chunk boundaries cross file boundaries.
- This approach is memory-efficient, does not pollute the repo, and is robust to future changes in the data format.

### Handling `policy=None`
- The dataset now handles `policy=None` by converting it to a zero vector of the correct shape (`(BOARD_SIZE * BOARD_SIZE,)`), matching the real data format on disk.
- This is tested with a dedicated test that creates a temporary file with a `policy=None` example and verifies the output tensor is all zeros and the correct shape.

### Test Suite Best Practices
- All tests use temporary files and valid data, avoiding patching/mocking and error threshold triggers.
- Tests are robust, memory-efficient, and easy to maintain.
- For future contributors:
  - Use `tmp_path` or `TemporaryDirectory` for test data.
  - Always match the real data format (e.g., numpy arrays, correct shapes).
  - Avoid static test data files unless absolutely necessary.
  - Prefer direct, public interface tests over patching or internal logic tests.

### Current State
- The codebase now has a solid, maintainable foundation for chunked dataset loading and augmentation.
- All key edge cases (multi-file, chunk boundaries, policy=None) are covered by robust, idiomatic tests.
- This design and test approach should be used as a reference for future extensions or refactors.

## Summary

This design provides a clean, idiomatic, and maintainable approach to chunked data loading in PyTorch. By separating index mapping, chunk loading, and data transformation, and by following best practices, we ensure robust, testable, and extensible code. 