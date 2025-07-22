# =============================================================================
# IMPORTANT: To run this test, set PYTHONPATH to the project root!
# Example:
#     PYTHONPATH=. pytest tests/test_streaming_augmented_processed_dataset.py
# This is required so that 'import hex_ai' works correctly.
# =============================================================================
"""
Tests for StreamingAugmentedProcessedDataset.

Expected behavior:
- The dataset takes a list of data files, each containing N base examples.
- If augmentation is enabled, each base example yields 4 augmented examples, so the dataset length is N*4 (where N is the number of base examples, up to max_examples_unaugmented).
- Indexing (ds[i]) returns the i-th augmented example, with 0 <= i < N*4.
- Accessing ds[i] for i >= N*4 raises IndexError.
- max_examples_unaugmented limits the number of base (unaugmented) examples, not the number of augmented examples.
- The dataset supports chunking, shuffling, and multiple files, and should always yield all augmentations for all base examples.
- Board data should only contain 0s and 1s; any other value is invalid.

Each test below documents what it is testing, what is passed, and what is expected.
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch
from pathlib import Path
from collections import Counter

from hex_ai.data_pipeline import StreamingAugmentedProcessedDataset
from hex_ai.config import BOARD_SIZE

import time
import functools


def timed_test(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"[TIMING] {func.__name__} took {duration:.3f} seconds")
        return result
    return wrapper


# ---
# Test: Integration with real file, tensorization, and chunking
# 1. Tests StreamingAugmentedProcessedDataset (class)
# 2. Tests integration with real data file, tensorization, chunking, and augmentation
# 3. Passes a real test file with known examples, chunk_size=2, max_examples_unaugmented=2, augmentation enabled
# 4. Expects each example to yield 4 augmentations, all returned as tensors with correct shapes
#    (ds[0]..ds[7] valid, ds[8] raises IndexError)
# ---
@timed_test
def test_real_file_tensorization_and_chunking():
    import gzip, pickle, os
    # Use the provided small test file
    file_path = Path("tests/small_shuffled_test.pkl.gz")
    assert file_path.exists(), "Test data file does not exist."
    # Patch augmentation to return 4 augmentations per example
    with patch("hex_ai.data_utils.create_augmented_example_with_player_to_move") as mock_aug:
        mock_aug.side_effect = lambda board, policy, value, error_tracker: [
            (np.array(board), np.array(policy), float(value), 0),
            (np.array(board), np.array(policy), float(value), 1),
            (np.array(board), np.array(policy), float(value), 0),
            (np.array(board), np.array(policy), float(value), 1),
        ]
        ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=2, max_examples_unaugmented=2, enable_augmentation=True, verbose=True)
        # Each example should yield 4 augmentations, so indices 0-3 for first, 4-7 for second
        for i in range(8):
            board, policy, value = ds[i]
            assert isinstance(board, torch.Tensor)
            assert isinstance(policy, torch.Tensor)
            assert isinstance(value, torch.Tensor)
            assert board.shape[0] == 3  # 3 channels
            assert board.shape[1] == BOARD_SIZE and board.shape[2] == BOARD_SIZE
            assert policy.shape == (BOARD_SIZE * BOARD_SIZE,)
            assert value.shape == (1,)

# ---
# Test: Off-by-one and boundary indexing with various chunk sizes
# 1. Tests that all valid indices up to max_examples_unaugmented*4-1 succeed, and max_examples_unaugmented*4 raises IndexError
# 2. Tests with chunk sizes that do not evenly divide the number of base examples
# ---
@timed_test
def test_off_by_one_and_boundary_indexing(tmp_path):
    """
    For a file with 3 base examples and augmentation enabled, test that indices 0-11 are valid and 12 raises IndexError.
    Repeat for chunk_size=1, 2, 3, 4.
    """
    import gzip, pickle
    from hex_ai.config import BOARD_SIZE
    from unittest.mock import patch
    start_time = time.time()
    # Create a file with 3 examples
    file_path = tmp_path / "boundary_test.pkl.gz"
    examples = []
    for i in range(3):
        board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        board[0, 0, 0] = i
        examples.append({'board': board, 'policy': np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 'value': float(i)})
    with gzip.open(file_path, "wb") as f:
        pickle.dump({"examples": examples}, f)
    for chunk_size in [1, 2, 3, 4]:
        print(f"[TEST] Starting chunk_size={chunk_size}")
        with patch("hex_ai.data_utils.create_augmented_example_with_player_to_move") as mock_aug:
            mock_aug.side_effect = lambda board, policy, value, error_tracker: [
                (np.array(board), np.array(policy), float(value), 0),
                (np.array(board), np.array(policy), float(value), 1),
                (np.array(board), np.array(policy), float(value), 0),
                (np.array(board), np.array(policy), float(value), 1),
            ]
            ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=chunk_size, max_examples_unaugmented=3, enable_augmentation=True)
            # All valid indices
            for i in range(12):
                board, policy, value = ds[i]
                assert board.shape[0] == 3
            import pytest
            with pytest.raises(IndexError):
                _ = ds[12]
        print(f"[TEST] Finished chunk_size={chunk_size}")
    print(f"[TEST] test_off_by_one_and_boundary_indexing finished in {time.time() - start_time:.2f} seconds")

# ---
# Test: Model integration
# 1. Tests StreamingAugmentedProcessedDataset (class) and TwoHeadedResNet (model)
# 2. Tests that dataset output is compatible with model input
# 3. Passes a real test file, chunk_size=2, max_examples_unaugmented=2, augmentation enabled
# 4. Expects a batch of 4 boards, policies, values to pass through the model without error
# ---
@timed_test
def test_streaming_dataset_model_integration():
    from hex_ai.models import TwoHeadedResNet
    file_path = Path("tests/small_shuffled_test.pkl.gz")
    ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=2, max_examples_unaugmented=2, enable_augmentation=True)
    model = TwoHeadedResNet()
    # Get a batch
    boards, policies, values = zip(*(ds[i] for i in range(4)))
    boards = torch.stack(boards)
    policies = torch.stack(policies)
    values = torch.stack(values)
    # Pass through model
    policy_logits, value_logit = model(boards)
    assert policy_logits.shape == (boards.shape[0], policies.shape[1])
    assert value_logit.shape == (boards.shape[0], 1)

# ---
# Test: Player channel correctness
# 1. Tests StreamingAugmentedProcessedDataset (class)
# 2. Tests that the player channel in the board tensor is always 0 or 1
# 3. Passes a real test file, chunk_size=2, max_examples_unaugmented=2, augmentation enabled
# 4. Expects all player channel values to be 0 or 1
# ---
@timed_test
def test_player_channel_correctness():
    file_path = Path("tests/small_shuffled_test.pkl.gz")
    ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=2, max_examples_unaugmented=2, enable_augmentation=True)
    for i in range(4):
        board, _, _ = ds[i]
        player_channel = board[2]
        assert torch.all((player_channel == 0) | (player_channel == 1)), "Player channel should be 0 or 1"

# ---
# Test: Empty board should be augmented (4x)
# 1. Tests StreamingAugmentedProcessedDataset (class)
# 2. Tests that even empty boards yield 4 augmentations
# 3. Passes a file with a single empty board, chunk_size=1, max_examples_unaugmented=1, augmentation enabled
# 4. Expects ds[0]..ds[3] valid, ds[4] raises IndexError
# ---
@timed_test
def test_empty_board_handling(tmp_path):
    import gzip, pickle
    from hex_ai.config import BOARD_SIZE
    # Create a .pkl.gz file with an empty board example
    file_path = tmp_path / "empty_board_test.pkl.gz"
    empty_board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    example = {'board': empty_board, 'policy': np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 'value': 1.0}
    with gzip.open(file_path, "wb") as f:
        pickle.dump({"examples": [example]}, f)
    ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=1, max_examples_unaugmented=1, enable_augmentation=True)
    # Should return 4 augmentations for ds[0]..ds[3]
    for i in range(4):
        board, policy, value = ds[i]
        assert isinstance(board, torch.Tensor)
        assert board.shape[0] == 2 or board.shape[0] == 3  # Accept 2 or 3 channels
    # Should raise IndexError for ds[4] (no more augmentations)
    import pytest
    with pytest.raises(IndexError):
        _ = ds[4]

# ---
# Test: Non-empty boards are augmented (4x)
# 1. Tests StreamingAugmentedProcessedDataset (class)
# 2. Tests that each non-empty board yields 4 augmentations
# 3. Passes a file with 2 non-empty boards, chunk_size=10, augmentation enabled
# 4. Expects 8 augmentations (4 per board), order doesn't matter
# ---
@timed_test
def test_get_augmented_example_logic_all_non_empty(tmp_path):
    import gzip, pickle
    from hex_ai.config import BOARD_SIZE
    from unittest.mock import patch
    # Create a .pkl.gz file with 2 non-empty examples
    file_path = tmp_path / "aug_logic_non_empty_test.pkl.gz"
    examples = []
    for i in range(2):
        board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        board[0, 0, 0] = i + 1  # Ensure non-empty
        examples.append({'board': board, 'policy': np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 'value': float(i)})
    with gzip.open(file_path, "wb") as f:
        pickle.dump({"examples": examples}, f)
    with patch("hex_ai.data_utils.create_augmented_example_with_player_to_move") as mock_aug:
        mock_aug.side_effect = lambda board, policy, value, error_tracker: [
            (np.array(board), np.array(policy), float(value), 0),
            (np.array(board), np.array(policy), float(value), 1),
            (np.array(board), np.array(policy), float(value), 0),
            (np.array(board), np.array(policy), float(value), 1),
        ]
        ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=10, max_examples_unaugmented=None, enable_augmentation=True)
        values = []
        i = 0
        while True:
            try:
                board, policy, value = ds[i]
                values.append(board[0, 0, 0].item())
                i += 1
            except IndexError:
                break
        # Should be four 1s and four 2s (order doesn't matter due to shuffling)
        assert sorted(values) == [1, 1, 1, 1, 2, 2, 2, 2]

# ---
# Test: Augmentation disabled returns only original examples
# 1. Tests StreamingAugmentedProcessedDataset (class)
# 2. Tests that disabling augmentation returns only the original examples
# 3. Passes a real test file, chunk_size=2, max_examples_unaugmented=2, augmentation disabled
# 4. Expects ds[0]..ds[1] valid, ds[2] raises IndexError
# ---
@timed_test
def test_non_augmented_path():
    file_path = Path("tests/small_shuffled_test.pkl.gz")
    ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=2, max_examples_unaugmented=2, enable_augmentation=False)
    # Should return only the original examples (no augmentation)
    for i in range(2):
        board, policy, value = ds[i]
        assert isinstance(board, torch.Tensor)
        assert isinstance(policy, torch.Tensor)
        assert isinstance(value, torch.Tensor)
    import pytest
    # Should raise IndexError after the original examples are exhausted
    with pytest.raises(IndexError):
        _ = ds[2] 

# ---
# Test: Non-augmented path logic
# 1. Tests StreamingAugmentedProcessedDataset (class)
# 2. Tests that non-augmented path returns all base examples
# 3. Passes a file with 3 examples, chunk_size=10, max_examples_unaugmented=3, augmentation disabled
# 4. Expects ds[0]..ds[2] valid, ds[3] raises IndexError
# ---
@timed_test
def test_get_non_augmented_example_logic(tmp_path):
    import gzip, pickle
    from hex_ai.config import BOARD_SIZE
    # Create a .pkl.gz file with 3 examples
    file_path = tmp_path / "non_aug_logic_test.pkl.gz"
    examples = []
    for i in range(3):
        board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        board[0, 0, 0] = i
        examples.append({'board': board, 'policy': np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 'value': float(i)})
    with gzip.open(file_path, "wb") as f:
        pickle.dump({"examples": examples}, f)
    ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=10, max_examples_unaugmented=3, enable_augmentation=False)
    # Collect all board[0,0,0] values using the public interface
    seen = set()
    i = 0
    while True:
        try:
            board, policy, value = ds[i]
            seen.add(board[0, 0, 0].item())
            i += 1
        except IndexError:
            break
    assert seen == {0, 1, 2}

# ---
# Test: Augmented example logic (mixed empty/non-empty)
# 1. Tests StreamingAugmentedProcessedDataset (class)
# 2. Tests that both empty and non-empty boards yield 4 augmentations each
# 3. Passes a file with 1 empty and 1 non-empty board, chunk_size=10, augmentation enabled
# 4. Expects 8 augmentations (4 per board), order doesn't matter
# ---
@timed_test
def _run_augmented_example_logic_test(tmp_path):
    import gzip, pickle
    from hex_ai.config import BOARD_SIZE
    from unittest.mock import patch
    # Create a .pkl.gz file with 2 examples (one empty, one non-empty)
    file_path = tmp_path / "aug_logic_test.pkl.gz"
    examples = []
    for i in range(2):
        board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        board[0, 0, 0] = i
        examples.append({'board': board, 'policy': np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 'value': float(i)})
    with gzip.open(file_path, "wb") as f:
        pickle.dump({"examples": examples}, f)
    # Print what was written
    print("[TEST DEBUG] Written examples:")
    for idx, ex in enumerate(examples):
        print(f"  Example {idx}: board sum={ex['board'].sum()}, board[0,0,0]={ex['board'][0,0,0]}, value={ex['value']}")
    # Patch augmentation to return 4 augmentations per example
    with patch("hex_ai.data_utils.create_augmented_example_with_player_to_move") as mock_aug:
        mock_aug.side_effect = lambda board, policy, value, error_tracker: [
            (np.array(board), np.array(policy), float(value), 0),
            (np.array(board), np.array(policy), float(value), 1),
            (np.array(board), np.array(policy), float(value), 0),
            (np.array(board), np.array(policy), float(value), 1),
        ]
        ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=10, max_examples_unaugmented=None, enable_augmentation=True, verbose=True)
        values = []
        boards = []
        i = 0
        while True:
            try:
                board, policy, value = ds[i]
                values.append(board[0, 0, 0].item())
                boards.append(board.detach().cpu().numpy())
                i += 1
            except IndexError:
                break
        print(f"[TEST DEBUG] values: {values}")
        print(f"[TEST DEBUG] Counter: {Counter(int(v) for v in values)})")
        for idx, b in enumerate(boards):
            print(f"[TEST DEBUG] Board {idx} sum: {b.sum()}, board[0,0,0]: {b[0,0,0]}")
        # Now, every example (even empty) yields 4 augmentations
        c = Counter(int(v) for v in values)
        assert c[1] == 4, f"Expected 4 augmentations of non-empty board, got {c[1]}"
        assert c[0] == 4, f"Expected 4 augmentations of empty board, got {c[0]}"
        assert len(values) == 8, f"Expected 8 total examples, got {len(values)}"

@timed_test
def test_get_augmented_example_logic(tmp_path):
    _run_augmented_example_logic_test(tmp_path)

# ---
# Test: Augmentation value label logic (multiset comparison)
# 1. Tests StreamingAugmentedProcessedDataset (class)
# 2. Tests that value labels are correctly handled in augmentation
# 3. Passes a file with 2 non-empty boards, each with value 0.0 or 1.0, chunk_size=10, augmentation enabled
# 4. Expects two 0s and two 1s per base example, order doesn't matter
# ---
@timed_test
def test_augmentation_value_label(tmp_path):
    import gzip, pickle
    from hex_ai.config import BOARD_SIZE
    from unittest.mock import patch
    from collections import Counter
    # Create a .pkl.gz file with 2 non-empty examples, one with value 0.0, one with value 1.0
    file_path = tmp_path / "aug_logic_value_label_test.pkl.gz"
    examples = []
    for i, v in enumerate([0.0, 1.0]):
        board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        board[0, 0, 0] = i + 1  # Ensure non-empty
        examples.append({'board': board, 'policy': np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 'value': v})
    with gzip.open(file_path, "wb") as f:
        pickle.dump({"examples": examples}, f)
    with patch("hex_ai.data_utils.create_augmented_example_with_player_to_move") as mock_aug:
        # Simulate two 0s and two 1s per base example
        def aug_fn(board, policy, value, error_tracker):
            return [
                (np.array(board), np.array(policy), 0.0, 0),
                (np.array(board), np.array(policy), 0.0, 1),
                (np.array(board), np.array(policy), 1.0, 0),
                (np.array(board), np.array(policy), 1.0, 1),
            ]
        mock_aug.side_effect = aug_fn
        ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=10, max_examples_unaugmented=None, enable_augmentation=True)
        values = []
        i = 0
        while True:
            try:
                board, policy, value = ds[i]
                values.append(value.item())
                i += 1
            except IndexError:
                break
        # For each base example, should get two 0s and two 1s (order doesn't matter)
        assert Counter(int(v) for v in values) == Counter([0, 0, 1, 1, 0, 0, 1, 1]) 

# ---
# Test: Chunk boundaries
# 1. Tests StreamingAugmentedProcessedDataset (class)
# 2. Tests that all augmentations are accessible across chunk boundaries
# 3. Passes 2 files, each with 2 examples, chunk_size=2, max_examples_unaugmented=4, augmentation enabled
# 4. Expects 16 augmentations (4 per example), order doesn't matter
# ---
@timed_test
def test_chunk_boundaries(tmp_path):
    """
    Test accessing the last augmentation of the last example in a chunk, and the first in the next chunk.
    Should not assume order; check that all expected values are present.
    """
    import gzip, pickle
    from unittest.mock import patch
    from hex_ai.config import BOARD_SIZE
    # Create 2 files, each with 2 examples
    for file_num in range(2):
        file_path = tmp_path / f"chunk_boundary_{file_num}.pkl.gz"
        examples = []
        for i in range(2):
            board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
            board[0, 0, 0] = i + file_num * 2
            examples.append({'board': board, 'policy': np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 'value': float(i + file_num * 2)})
        with gzip.open(file_path, "wb") as f:
            pickle.dump({"examples": examples}, f)
    files = [tmp_path / f"chunk_boundary_{i}.pkl.gz" for i in range(2)]
    with patch("hex_ai.data_utils.create_augmented_example_with_player_to_move") as mock_aug:
        mock_aug.side_effect = lambda board, policy, value, error_tracker: [
            (np.array(board), np.array(policy), float(value), 0),
            (np.array(board), np.array(policy), float(value), 1),
            (np.array(board), np.array(policy), float(value), 0),
            (np.array(board), np.array(policy), float(value), 1),
        ]
        ds = StreamingAugmentedProcessedDataset(files, chunk_size=2, max_examples_unaugmented=4, enable_augmentation=True)
        # Collect all values
        values = []
        i = 0
        while True:
            try:
                board, policy, value = ds[i]
                values.append(board[0, 0, 0].item())
                i += 1
            except IndexError:
                break
        # Should be four 0s, four 1s, four 2s, four 3s (order doesn't matter)
        c = Counter(values)
        assert c[0.0] == 4
        assert c[1.0] == 4
        assert c[2.0] == 4
        assert c[3.0] == 4
        assert len(values) == 16

# ---
# Test: Multiple files and shuffling
# 1. Tests StreamingAugmentedProcessedDataset (class)
# 2. Tests that all augmentations are accessible across multiple files, with and without shuffling
# 3. Passes 3 files, each with 1 example, chunk_size=1, max_examples_unaugmented=3, augmentation enabled
# 4. Expects 12 augmentations (4 per example), order doesn't matter
# ---
@timed_test
def test_multiple_files_and_shuffling(tmp_path):
    """
    Test dataset with multiple files and shuffling enabled/disabled. Do not assume order; check set of values.
    """
    import gzip, pickle
    from unittest.mock import patch
    from hex_ai.config import BOARD_SIZE
    # Create 3 files, each with 1 example
    files = []
    for i in range(3):
        file_path = tmp_path / f"shuffle_{i}.pkl.gz"
        board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        board[0, 0, 0] = i
        example = {'board': board, 'policy': np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 'value': float(i)}
        with gzip.open(file_path, "wb") as f:
            pickle.dump({"examples": [example]}, f)
        files.append(file_path)
    with patch("hex_ai.data_utils.create_augmented_example_with_player_to_move") as mock_aug:
        mock_aug.side_effect = lambda board, policy, value, error_tracker: [
            (np.array(board), np.array(policy), float(value), 0),
            (np.array(board), np.array(policy), float(value), 1),
            (np.array(board), np.array(policy), float(value), 0),
            (np.array(board), np.array(policy), float(value), 1),
        ]
        # No shuffle
        ds = StreamingAugmentedProcessedDataset(files, chunk_size=1, max_examples_unaugmented=3, enable_augmentation=True, shuffle_files=False)
        values = []
        i = 0
        while True:
            try:
                board, policy, value = ds[i]
                values.append(board[0, 0, 0].item())
                i += 1
            except IndexError:
                break
        c = Counter(values)
        assert c[0.0] == 4
        assert c[1.0] == 4
        assert c[2.0] == 4
        assert len(values) == 12
        # Shuffle
        ds2 = StreamingAugmentedProcessedDataset(files, chunk_size=1, max_examples_unaugmented=3, enable_augmentation=True, shuffle_files=True)
        values2 = []
        i = 0
        while True:
            try:
                board, policy, value = ds2[i]
                values2.append(board[0, 0, 0].item())
                i += 1
            except IndexError:
                break
        c2 = Counter(values2)
        assert c2[0.0] == 4
        assert c2[1.0] == 4
        assert c2[2.0] == 4
        assert len(values2) == 12

# ---
# Test: Edge cases (empty file, only empty boards, one example)
# 1. Tests StreamingAugmentedProcessedDataset (class)
# 2. Tests edge cases: empty file, file with only empty boards, file with one example
# 3. Passes files with these cases, chunk_size and max_examples_unaugmented as appropriate, augmentation enabled
# 4. Expects correct number of augmentations or IndexError as appropriate
# ---
@timed_test
def test_edge_cases(tmp_path):
    """
    Test edge cases: empty file, file with only empty boards, file with one example. Do not assume order.
    """
    import gzip, pickle
    from unittest.mock import patch
    from hex_ai.config import BOARD_SIZE
    # Empty file
    file_path = tmp_path / "empty_file.pkl.gz"
    with gzip.open(file_path, "wb") as f:
        pickle.dump({"examples": []}, f)
    ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=1, max_examples_unaugmented=0, enable_augmentation=True)
    with pytest.raises(IndexError):
        _ = ds[0]
    # File with only empty boards
    file_path2 = tmp_path / "only_empty.pkl.gz"
    empty_board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    examples = [{'board': empty_board, 'policy': np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 'value': 1.0} for _ in range(2)]
    with gzip.open(file_path2, "wb") as f:
        pickle.dump({"examples": examples}, f)
    ds2 = StreamingAugmentedProcessedDataset([file_path2], chunk_size=2, max_examples_unaugmented=2, enable_augmentation=True)
    values2 = []
    for i in range(8):
        board, policy, value = ds2[i]
        values2.append(board[0, 0, 0].item())
    c2 = Counter(values2)
    assert c2[0.0] == 8
    with pytest.raises(IndexError):
        _ = ds2[8]
    # File with one example
    file_path3 = tmp_path / "one_example.pkl.gz"
    board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    board[0, 0, 0] = 42
    example = {'board': board, 'policy': np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 'value': 42.0}
    with gzip.open(file_path3, "wb") as f:
        pickle.dump({"examples": [example]}, f)
    ds3 = StreamingAugmentedProcessedDataset([file_path3], chunk_size=1, max_examples_unaugmented=1, enable_augmentation=True)
    values3 = []
    for i in range(4):
        board, policy, value = ds3[i]
        values3.append(board[0, 0, 0].item())
    c3 = Counter(values3)
    assert c3[42.0] == 4
    with pytest.raises(IndexError):
        _ = ds3[4]

# ---
# Test: __len__ NotImplementedError
# 1. Tests StreamingAugmentedProcessedDataset (class)
# 2. Tests that __len__ raises NotImplementedError if max_examples_unaugmented is not set
# 3. Passes a real test file, chunk_size=1, augmentation enabled, no max_examples_unaugmented
# 4. Expects NotImplementedError
# ---
@timed_test
def test_not_implemented_len():
    """
    Test that __len__ raises NotImplementedError if max_examples_unaugmented is not set.
    """
    from hex_ai.config import BOARD_SIZE
    file_path = Path("tests/small_shuffled_test.pkl.gz")
    ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=1, enable_augmentation=True)
    import pytest
    with pytest.raises(NotImplementedError):
        _ = len(ds)

# ---
# Test: Augmentation error handling
# 1. Tests StreamingAugmentedProcessedDataset (class)
# 2. Tests that errors in augmentation are propagated and logged
# 3. Passes a file with one example, chunk_size=1, max_examples_unaugmented=1, augmentation enabled, and a mock that raises
# 4. Expects RuntimeError when augmentation fails
# ---
@timed_test
def test_augmentation_error_handling(tmp_path):
    """
    Test that errors in augmentation are propagated and logged.
    """
    import gzip, pickle
    from unittest.mock import patch
    from hex_ai.config import BOARD_SIZE
    file_path = tmp_path / "aug_error.pkl.gz"
    board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    example = {'board': board, 'policy': np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 'value': 1.0}
    with gzip.open(file_path, "wb") as f:
        pickle.dump({"examples": [example]}, f)
    with patch("hex_ai.data_utils.create_augmented_example_with_player_to_move") as mock_aug:
        mock_aug.side_effect = RuntimeError("Augmentation failed!")
        ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=1, max_examples_unaugmented=1, enable_augmentation=True)
        with pytest.raises(RuntimeError):
            _ = ds[0] 

# ---
# Test: Augmentation handles None policy labels (terminal positions)
# 1. Tests create_augmented_example_with_player_to_move (function)
# 2. Passes a mock board and policy=None (terminal position)
# 3. Expects all returned policy tensors to be all zeros (handled at augmentation stage)
# 4. Documents that this is the expected handling for terminal positions
# ---
@timed_test
def test_create_augmented_example_with_player_to_move_handles_none_policy():
    """
    Test that create_augmented_example_with_player_to_move handles policy=None by returning a zero policy tensor for all augmentations.
    This is the expected handling for terminal positions (no legal moves left).
    The current implementation handles None policy labels at the data augmentation stage.
    """
    import numpy as np
    from hex_ai.data_utils import create_augmented_example_with_player_to_move
    from hex_ai.config import BOARD_SIZE, POLICY_OUTPUT_SIZE
    # Create a mock board (2, BOARD_SIZE, BOARD_SIZE)
    board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    value = 1.0
    policy = None  # Terminal position
    augmented = create_augmented_example_with_player_to_move(board, policy, value)
    assert len(augmented) == 4
    for aug_board, aug_policy, aug_value, aug_player in augmented:
        assert aug_policy.shape == (POLICY_OUTPUT_SIZE,)
        assert np.allclose(aug_policy, 0), "Policy should be all zeros for terminal positions"
        assert isinstance(aug_board, np.ndarray)
        assert isinstance(aug_value, float)
        assert aug_board.shape[1:] == (BOARD_SIZE, BOARD_SIZE)

# ---
# Test: Non-augmented path mimics augmentation handling of None policy labels
# 1. Tests StreamingAugmentedProcessedDataset (class)
# 2. Passes a file with a single example with policy=None (terminal position)
# 3. Expects the output policy tensor to be all zeros
# 4. Documents that the non-augmented path now mimics the augmentation path for consistency
# ---
@timed_test
def test_get_non_augmented_example_converts_none_policy_to_zeros(tmp_path):
    """
    Test that the non-augmented/validation path converts a None policy label to a zero tensor (terminal position),
    matching the augmented path. The output policy tensor should be all zeros.
    The current implementation handled None policy labels at the data augmentation stage, and the non-augmented path now mimics this behavior for consistency.
    """
    import gzip, pickle
    import numpy as np
    from hex_ai.data_pipeline import StreamingAugmentedProcessedDataset
    from hex_ai.config import BOARD_SIZE, POLICY_OUTPUT_SIZE
    # Create a .pkl.gz file with one example with policy=None
    file_path = tmp_path / "none_policy_non_augmented.pkl.gz"
    example = {
        'board': np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
        'policy': None,  # Should be converted to zeros
        'value': 1.0
    }
    with gzip.open(file_path, "wb") as f:
        pickle.dump({"examples": [example]}, f)
    ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=1, max_examples_unaugmented=1, enable_augmentation=False)
    board, policy, value = ds[0]
    # Check that the policy tensor is all zeros
    assert np.allclose(policy.numpy(), 0), "Policy tensor should be all zeros for terminal positions"
    assert board.shape[0] == 2 or board.shape[0] == 3
    assert value.shape == (1,)

# ---
# Test: _validate_example_format raises on None policy labels
# 1. Tests _validate_example_format (function)
# 2. Passes a mock example with policy=None (should never happen in real data)
# 3. Expects a ValueError to be raised
# 4. Documents that None policy labels should already have been handled upstream, and this test ensures an exception is raised if not
# ---
@timed_test
def test_validate_example_format_raises_on_none_policy():
    """
    Test that _validate_example_format raises ValueError if policy=None.
    This should never happen in real data and indicates a bug upstream (should be handled during preprocessing or augmentation).
    This test ensures that if a None policy label slips through, it is caught and surfaced as an error.
    """
    import numpy as np
    from hex_ai.data_pipeline import _validate_example_format
    from hex_ai.config import BOARD_SIZE
    example = {
        'board': np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
        'policy': None,  # Should trigger error
        'value': 1.0
    }
    import pytest
    with pytest.raises(ValueError, match="Policy target is None"):
        _validate_example_format(example, filename="<test>") 

# =====================
# Modern tests for sequential, chunked, single-pass design
# =====================
import tempfile
import shutil

def create_test_file(tmp_path, examples):
    import gzip, pickle
    file_path = tmp_path / "test_file.pkl.gz"
    with gzip.open(file_path, "wb") as f:
        pickle.dump({"examples": examples}, f)
    return file_path

DEBUG_TEST = True  # <<< DEBUG: Set to False to disable debug prints

# ---
def test_sequential_access_enforced(tmp_path):
    """
    Test that only sequential access is allowed; random access raises NotImplementedError.
    """
    import numpy as np
    from hex_ai.config import BOARD_SIZE
    from hex_ai.data_pipeline import StreamingAugmentedProcessedDataset
    # Create a file with 2 examples
    examples = []
    for i in range(2):
        board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        examples.append({'board': board, 'policy': np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 'value': float(i)})
    file_path = create_test_file(tmp_path, examples)
    ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=2, max_examples_unaugmented=2, enable_augmentation=False)
    # First access is fine
    _ = ds[0]
    # Second access is fine
    _ = ds[1]
    # Out of order access should fail
    import pytest
    with pytest.raises(NotImplementedError):
        _ = ds[0]

# ---
def test_chunk_loading_and_boundaries(tmp_path):
    """
    Test that chunk boundaries are handled correctly and all examples are returned.
    """
    import numpy as np
    from hex_ai.config import BOARD_SIZE
    from hex_ai.data_pipeline import StreamingAugmentedProcessedDataset
    # Create 4 examples, chunk_size=2
    examples = []
    for i in range(4):
        board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        board[0, 0, 0] = i
        examples.append({'board': board, 'policy': np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 'value': float(i)})
    file_path = create_test_file(tmp_path, examples)
    ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=2, max_examples_unaugmented=4, enable_augmentation=False)
    seen = []
    for i in range(4):
        board, policy, value = ds[i]
        val = board[0, 0, 0].item()
        seen.append(val)
        if DEBUG_TEST:
            print(f"[DEBUG-CHUNK-BOUNDARIES] idx={i}, board[0,0,0]={val}")
    # The dataset should yield the examples in order: 0, 1, 2, 3
    if DEBUG_TEST:
        print(f"[DEBUG-CHUNK-BOUNDARIES] seen={seen}")
    assert seen == [0, 1, 2, 3]
    import pytest
    with pytest.raises(IndexError):
        _ = ds[4]

# ---
def test_augmentation_enabled(tmp_path):
    """
    Test that with augmentation enabled, each base example yields 4 augmented examples.
    """
    import numpy as np
    from hex_ai.config import BOARD_SIZE
    from unittest.mock import patch
    from hex_ai.data_pipeline import StreamingAugmentedProcessedDataset
    # Create 1 example
    board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    example = {'board': board, 'policy': np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 'value': 1.0}
    file_path = create_test_file(tmp_path, [example])
    with patch("hex_ai.data_utils.create_augmented_example_with_player_to_move") as mock_aug:
        mock_aug.side_effect = lambda board, policy, value, error_tracker: [
            (np.array(board), np.array(policy), float(value), 0),
            (np.array(board), np.array(policy), float(value), 1),
            (np.array(board), np.array(policy), float(value), 0),
            (np.array(board), np.array(policy), float(value), 1),
        ]
        ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=4, max_examples_unaugmented=1, enable_augmentation=True)
        for i in range(4):
            board, policy, value = ds[i]
            assert isinstance(board, torch.Tensor)
            assert isinstance(policy, torch.Tensor)
            assert isinstance(value, torch.Tensor)
        import pytest
        with pytest.raises(IndexError):
            _ = ds[4]

# ---
def test_empty_file_edge_case(tmp_path):
    """
    Test that an empty file (no examples) raises IndexError on first access.
    """
    from hex_ai.data_pipeline import StreamingAugmentedProcessedDataset
    file_path = create_test_file(tmp_path, [])
    ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=1, max_examples_unaugmented=0, enable_augmentation=True)
    import pytest
    with pytest.raises(IndexError):
        _ = ds[0]

# ---
def test_tensorization_and_output_shape(tmp_path):
    """
    Test that output is always (board, policy, value) as torch tensors with correct shapes.
    """
    import numpy as np
    from hex_ai.config import BOARD_SIZE
    from hex_ai.data_pipeline import StreamingAugmentedProcessedDataset
    # Create 1 example
    board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    example = {'board': board, 'policy': np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 'value': 1.0}
    file_path = create_test_file(tmp_path, [example])
    ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=1, max_examples_unaugmented=1, enable_augmentation=False)
    board, policy, value = ds[0]
    assert isinstance(board, torch.Tensor)
    assert isinstance(policy, torch.Tensor)
    assert isinstance(value, torch.Tensor)
    assert board.shape[0] == 2 or board.shape[0] == 3
    assert policy.shape == (BOARD_SIZE * BOARD_SIZE,)
    assert value.shape == (1,)

# ---
def test_error_handling_bad_file(tmp_path):
    """
    Test that a file with missing 'examples' key is handled gracefully (raises IndexError on access).
    """
    import gzip, pickle
    from hex_ai.data_pipeline import StreamingAugmentedProcessedDataset
    file_path = tmp_path / "bad_file.pkl.gz"
    with gzip.open(file_path, "wb") as f:
        pickle.dump({"not_examples": []}, f)
    ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=1, max_examples_unaugmented=1, enable_augmentation=False)
    import pytest
    with pytest.raises(IndexError):
        _ = ds[0]

# ===================== 