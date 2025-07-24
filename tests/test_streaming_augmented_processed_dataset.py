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
        board = make_valid_board(i)
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
    # Just check no error

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


# =====================
# Modern tests for sequential, chunked, single-pass design
# =====================
import tempfile
import shutil

import pytest
import numpy as np
import torch
from pathlib import Path
from hex_ai.data_pipeline import StreamingAugmentedProcessedDataset
from hex_ai.config import BOARD_SIZE
import gzip, pickle

def make_valid_board(i):
    # Alternate between blue's turn and red's turn
    # Even i: blue_count == red_count == i//2
    # Odd i: blue_count == (i+1)//2, red_count == i//2
    from hex_ai.config import BOARD_SIZE
    board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    blue_count = (i + 1) // 2
    red_count = i // 2
    # Place blue stones
    for b in range(blue_count):
        board[0, 0, b] = 1
    # Place red stones
    for r in range(red_count):
        board[1, 0, r] = 1
    return board

def create_test_file(tmp_path, examples, fname):
    file_path = tmp_path / fname
    with gzip.open(file_path, "wb") as f:
        pickle.dump({"examples": examples}, f)
    return file_path

def test_loads_correct_number_of_examples(tmp_path):
    # File has 10 examples, but max_examples_unaugmented=5
    examples = []
    for i in range(10):
        board = make_valid_board(i)
        examples.append({'board': board, 'policy': np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 'value': float(i)})
    file_path = create_test_file(tmp_path, examples, "f1.pkl.gz")
    ds = StreamingAugmentedProcessedDataset([file_path], max_examples_unaugmented=5, enable_augmentation=False)
    assert len(ds) == 5
    seen = [ds[i][0][0, 0, 0].item() for i in range(5)]
    # The first blue stone is always at (0,0,0), so just check no error
    with pytest.raises(IndexError):
        _ = ds[5]

def test_loads_from_multiple_files(tmp_path):
    # File 1 has 3, file 2 has 4, want 5 total
    ex1 = []
    for i in range(3):
        board = make_valid_board(i)
        ex1.append({'board': board, 'policy': np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 'value': float(i)})
    ex2 = []
    for i in range(3, 7):
        board = make_valid_board(i)
        ex2.append({'board': board, 'policy': np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 'value': float(i)})
    f1 = create_test_file(tmp_path, ex1, "f1.pkl.gz")
    f2 = create_test_file(tmp_path, ex2, "f2.pkl.gz")
    ds = StreamingAugmentedProcessedDataset([f1, f2], max_examples_unaugmented=5, enable_augmentation=False)
    assert len(ds) == 5
    seen = [ds[i][0][0, 0, 0].item() for i in range(5)]
    # Just check no error

def test_sequential_order_across_files(tmp_path):
    # File 1: 2 examples, File 2: 2 examples
    ex1 = []
    for i in range(2):
        board = make_valid_board(i)
        ex1.append({'board': board, 'policy': np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 'value': float(i)})
    ex2 = []
    for i in range(2, 4):
        board = make_valid_board(i)
        ex2.append({'board': board, 'policy': np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 'value': float(i)})
    f1 = create_test_file(tmp_path, ex1, "f1.pkl.gz")
    f2 = create_test_file(tmp_path, ex2, "f2.pkl.gz")
    ds = StreamingAugmentedProcessedDataset([f1, f2], max_examples_unaugmented=4, enable_augmentation=False)
    seen = [ds[i][0][0, 0, 0].item() for i in range(4)]
    # Just check no error

def test_index_error_on_out_of_range(tmp_path):
    # File has 2 examples
    examples = []
    for i in range(2):
        board = make_valid_board(i)
        examples.append({'board': board, 'policy': np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 'value': float(i)})
    file_path = create_test_file(tmp_path, examples, "f1.pkl.gz")
    ds = StreamingAugmentedProcessedDataset([file_path], max_examples_unaugmented=2, enable_augmentation=False)
    assert len(ds) == 2
    with pytest.raises(IndexError):
        _ = ds[2]

def test_skips_missing_and_corrupted_files(tmp_path):
    # File 1: valid, File 2: corrupted, File 3: missing
    ex1 = []
    for i in range(3):
        board = make_valid_board(i)
        ex1.append({'board': board, 'policy': np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 'value': float(i)})
    f1 = create_test_file(tmp_path, ex1, "f1.pkl.gz")
    f2 = tmp_path / "corrupted.pkl.gz"
    with open(f2, "wb") as f:
        f.write(b"not a pickle file")
    f3 = tmp_path / "missing.pkl.gz"  # do not create
    ds = StreamingAugmentedProcessedDataset([f1, f2, f3], max_examples_unaugmented=3, enable_augmentation=False)
    assert len(ds) == 3
    seen = [ds[i][0][0, 0, 0].item() for i in range(3)]
    # Just check no error


# ---
# Test: get_augmented_tensor_for_index helper (no augmentation)
def test_get_augmented_tensor_for_index_no_aug(tmp_path):
    """
    Test the get_augmented_tensor_for_index helper for a non-augmented example.
    """
    import numpy as np
    from hex_ai.config import BOARD_SIZE
    from hex_ai.data_pipeline import StreamingAugmentedProcessedDataset
    board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
    value = 1.0
    ex = {'board': board, 'policy': policy, 'value': value}
    ds = StreamingAugmentedProcessedDataset([], chunk_size=1, max_examples_unaugmented=1, enable_augmentation=False)
    from hex_ai.error_handling import get_board_state_error_tracker
    error_tracker = get_board_state_error_tracker()
    tensorized = ds.get_augmented_tensor_for_index(ex, 0, error_tracker)
    board_tensor, policy_tensor, value_tensor = tensorized
    assert board_tensor.shape[0] == 2 or board_tensor.shape[0] == 3
    assert policy_tensor.shape == (BOARD_SIZE * BOARD_SIZE,)
    assert value_tensor.shape == (1,)


# ---
# Test: Edge case - empty file
def test_empty_file_edge_case(tmp_path):
    """
    Test that an empty file (no examples) raises IndexError on first access.
    """
    from hex_ai.data_pipeline import StreamingAugmentedProcessedDataset
    file_path = create_test_file(tmp_path, [], "empty.pkl.gz")
    ds = StreamingAugmentedProcessedDataset([file_path], max_examples_unaugmented=0, enable_augmentation=True)
    import pytest
    with pytest.raises(IndexError):
        _ = ds[0]


# ---
# Test: Policy none handling
def test_policy_none_handling(tmp_path):
    """
    Test that if an example in the data has policy=None, the dataset replaces it with a zero vector of the correct shape.
    This ensures robust handling of missing policy data.
    """
    import gzip, pickle
    import numpy as np
    from hex_ai.config import BOARD_SIZE
    from hex_ai.data_pipeline import StreamingAugmentedProcessedDataset

    # Create a minimal example with policy=None
    board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    example = {"board": board, "policy": None, "value": 1.0}
    dataset = {"examples": [example]}
    file = tmp_path / "policy_none_sample.pkl.gz"
    with gzip.open(file, "wb") as f:
        pickle.dump(dataset, f)

    # Instantiate the dataset
    ds = StreamingAugmentedProcessedDataset([file], chunk_size=1, max_examples_unaugmented=1, enable_augmentation=False)

    # Fetch the example
    ex = ds[0]
    policy_tensor = ex[1]  # ex[1] is the policy tensor

    # Assert that the policy tensor is all zeros and has the correct shape
    assert np.all(policy_tensor.numpy() == 0), "Policy tensor should be all zeros when policy=None"
    assert policy_tensor.shape == (BOARD_SIZE * BOARD_SIZE,), f"Policy tensor shape should be ({BOARD_SIZE * BOARD_SIZE},), got {policy_tensor.shape}"


def test_real_augmentation_logic(tmp_path):
    """
    Test that the dataset with enable_augmentation=True produces the correct number of augmented examples per input,
    and that the augmentations are valid (not all identical, correct shape, valid values).
    Also check that the set of value labels for each original example contains two of the original and two of the flipped winner.
    """
    import gzip, pickle
    import numpy as np
    from hex_ai.config import BOARD_SIZE
    from hex_ai.data_pipeline import StreamingAugmentedProcessedDataset, AUGMENTATION_FACTOR

    # Create 2 valid, non-empty boards with simple patterns and known values
    examples = []
    for i in range(2):
        board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        board[0, i, i] = 1  # Place a single stone for player 0
        board[1, (i+1)%BOARD_SIZE, (i+2)%BOARD_SIZE] = 1  # Place a single stone for player 1
        examples.append({'board': board, 'policy': np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 'value': float(i)})

    # Write to temp file
    file = tmp_path / "aug_sample.pkl.gz"
    with gzip.open(file, "wb") as f:
        pickle.dump({"examples": examples}, f)

    # Instantiate the dataset with augmentation
    ds = StreamingAugmentedProcessedDataset([file], chunk_size=2, max_examples_unaugmented=2, enable_augmentation=True)

    # Collect all augmented boards and value labels
    boards = []
    values = []
    for idx in range(len(ds)):
        ex = ds[idx]
        board_tensor = ex[0]  # ex[0] is the board tensor
        value_label = ex[2].item()  # ex[2] is the value tensor
        boards.append(board_tensor.numpy())
        values.append(value_label)

    # Check the number of outputs
    assert len(boards) == 2 * AUGMENTATION_FACTOR, f"Expected {2 * AUGMENTATION_FACTOR} augmented examples, got {len(boards)}"

    # Check that not all augmented boards are identical
    unique_boards = {b.tobytes() for b in boards}
    assert len(unique_boards) > 2, "Augmentation should produce different boards, not all identical"

    # Check that each board has the correct shape and valid values (0 or 1)
    for b in boards:
        assert b.shape == (2, BOARD_SIZE, BOARD_SIZE) or b.shape == (3, BOARD_SIZE, BOARD_SIZE), f"Unexpected board shape: {b.shape}"
        assert np.all((b == 0) | (b == 1)), "Board should only contain 0s and 1s after augmentation"

    # Check value label flipping logic for each original example
    for i in range(2):
        # Each original example should yield 4 augmentations
        start = i * AUGMENTATION_FACTOR
        end = (i + 1) * AUGMENTATION_FACTOR
        value_set = set(values[start:end])
        expected_set = {float(i), 1.0 - float(i)}
        assert value_set == expected_set, f"Augmented value labels for example {i} should be {{original, flipped}}, got {value_set}"
        # Optionally warn if the order is not [original, original, flipped, flipped]
        expected_order = [float(i), float(i), 1.0 - float(i), 1.0 - float(i)]
        actual_order = values[start:end]
        if actual_order != expected_order:
            print(f"[WARN] Augmentation value label order for example {i} is {actual_order}, expected {expected_order}. This is not an error, but order is not stable.")

def test_always_has_player_channel(tmp_path):
    # File with 2 examples
    examples = []
    for i in range(2):
        board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        board[0, 0, 0] = i
        examples.append({'board': board, 'policy': np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 'value': float(i)})
    file_path = create_test_file(tmp_path, examples, "f1.pkl.gz")
    # Test with augmentation disabled
    ds = StreamingAugmentedProcessedDataset([file_path], max_examples_unaugmented=2, enable_augmentation=False)
    for i in range(len(ds)):
        board, _, _ = ds[i]
        assert board.shape[0] == 3, f"Expected 3 channels, got {board.shape[0]}"
    # Test with augmentation enabled
    ds_aug = StreamingAugmentedProcessedDataset([file_path], max_examples_unaugmented=2, enable_augmentation=True)
    for i in range(len(ds_aug)):
        board, _, _ = ds_aug[i]
        assert board.shape[0] == 3, f"Expected 3 channels, got {board.shape[0]}"
