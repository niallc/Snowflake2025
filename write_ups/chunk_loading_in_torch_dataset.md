# Chunk Loading in PyTorch Dataset: Clean, Sequential Design

## Motivation

For large datasets stored across multiple files, it is efficient and maintainable to load data in sequential chunks, minimizing memory usage and disk reads. The goal is to design a PyTorch dataset that loads at most two files at a time, supports chunked access, and is easy to extend and test.

## Requirements

- **Chunked Access:** Map a global index to a chunk number and an index within that chunk.
- **Efficient File Loading:** For each chunk, load only the files needed for that chunk—never more than two at a time.
- **Sequential Reading:** Read records in order, leveraging the fact that files are already shuffled and can be processed sequentially.
- **Separation of Concerns:** Keep chunk mapping, chunk loading, and data transformation as separate, testable units.
- **Extensible:** Easy to add features like shuffling within chunks, prefetching, or multi-worker support.

## High-Level Design

### 1. File and Example Indexing

- At initialization, record the number of examples in each file.
- Maintain a mapping from global example index to (file_idx, example_idx) using cumulative counts.
- No need to build a global index map of all examples.

### 2. Chunk Mapping

- Given a global index `idx`, compute:
  - `chunk_number = idx // chunk_size`
  - `index_in_chunk = idx % chunk_size`
- For a given chunk, determine which files and which example indices are needed.
- If a chunk crosses a file boundary, load both files.

### 3. Chunk Loading

- When a chunk is requested:
  - Identify the files and the range of examples needed from each.
  - Load each file only once per chunk.
  - Extract all required examples from each file and concatenate them to form the chunk.
- At most two files are loaded into memory at any time.

### 4. Data Transformation

- All data transformation (e.g., augmentation, tensorization) is handled in a dedicated function, not inline in the chunk loader.

### 5. Error Handling

- Handle missing/corrupted files gracefully.
- If a chunk cannot be loaded, raise a clear error or return a fallback.

## Pseudocode/Code Sketch

```python
class ChunkedDataset(torch.utils.data.Dataset):
    def __init__(self, data_files, chunk_size, ...):
        self.data_files = data_files
        self.chunk_size = chunk_size
        self.file_example_counts = [count_examples(f) for f in data_files]
        self.cumulative_counts = ... # cumulative sum for fast lookup
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
        # Load each file only once, extract all required examples
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
- **Extensible:** Easy to add shuffling within chunks, prefetching, etc.

## Error Handling

- If a file is missing or corrupted, log a warning and skip it.
- If a chunk cannot be filled (e.g., not enough data), return a partial chunk or raise an error.
- If an index is out of range, raise `IndexError`.

## Extensibility

- **Shuffling:** Shuffle file list at epoch start, or shuffle within chunks if needed.
- **Prefetching:** Add a background thread or process to pre-load the next chunk.
- **Multi-worker:** Ensure chunk loading is thread/process safe.

## Implementation and Testing Progress (July 2024)

### Multi-file Chunking
- The dataset supports efficient multi-file chunking, loading at most two files at a time.
- Tests create temporary .pkl.gz files on the fly (using pytest's `tmp_path`), each with unique examples.
- The test verifies that all examples are seen in the correct order, even when chunk boundaries cross file boundaries.
- This approach is memory-efficient, does not pollute the repo, and is robust to future changes in the data format.

### Handling `policy=None`
- The dataset handles `policy=None` by converting it to a zero vector of the correct shape (`(BOARD_SIZE * BOARD_SIZE,)`), matching the real data format on disk.
- This is tested with a dedicated test that creates a temporary file with a `policy=None` example and verifies the output tensor is all zeros and the correct shape.

### Test Suite Best Practices
- All tests use temporary files and valid data, avoiding patching/mocking and error threshold triggers.
- Tests are robust, memory-efficient, and easy to maintain.
- For future contributors:
  - Use `tmp_path` or `TemporaryDirectory` for test data.
  - Always match the real data format (e.g., numpy arrays, correct shapes).
  - Avoid static test data files unless absolutely necessary.
  - Prefer direct, public interface tests over patching or internal logic tests.

### Error Handling and Shuffling (July 2024)
- Added tests for error handling: the dataset logs warnings and skips corrupted or missing files, but still yields all valid examples from good files.
- Added tests for shuffling: with `shuffle_files=False`, the order is stable; with `shuffle_files=True`, the order can change across runs.

### Current Test Coverage
- The dataset is robustly tested for:
  - Multi-file chunking (at most two files in memory)
  - Augmentation (including value label flipping)
  - `policy=None` handling
  - Error handling for corrupted/missing files
  - Shuffling of file order
- The test suite is idiomatic, robust, and provides strong coverage for real-world usage, even if not every edge case is exhaustively tested.
- Future contributors are encouraged to extend tests as new features or edge cases arise.

### Current State
- The codebase now has a solid, maintainable foundation for chunked dataset loading and augmentation.
- All key edge cases (multi-file, chunk boundaries, policy=None) are covered by robust, idiomatic tests.
- This design and test approach should be used as a reference for future extensions or refactors.

## Summary

This design provides a clean, idiomatic, and maintainable approach to chunked data loading in PyTorch. By separating index mapping, chunk loading, and data transformation, and by following best practices, we ensure robust, testable, and extensible code. 