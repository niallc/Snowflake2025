import pytest
import numpy as np
import torch
from unittest.mock import patch
from pathlib import Path

from hex_ai.data_pipeline import StreamingAugmentedProcessedDataset
from hex_ai.config import BOARD_SIZE


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
        ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=2, max_examples=2, enable_augmentation=True, verbose=True)
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


def test_epoch_restart_and_chunking():
    import pytest
    file_path = Path("tests/small_shuffled_test.pkl.gz")
    assert file_path.exists(), "Test data file does not exist."
    with patch("hex_ai.data_utils.create_augmented_example_with_player_to_move") as mock_aug:
        mock_aug.side_effect = lambda board, policy, value, error_tracker: [
            (np.array(board), np.array(policy), float(value), 0),
            (np.array(board), np.array(policy), float(value), 1),
            (np.array(board), np.array(policy), float(value), 0),
            (np.array(board), np.array(policy), float(value), 1),
        ]
        ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=1, max_examples=2, enable_augmentation=True, verbose=True)
        # Exhaust the dataset, then trigger an epoch restart
        for i in range(4):  # Only access the valid indices
            _ = ds[i]
        # Next call should raise IndexError
        with pytest.raises(IndexError):
            _ = ds[4] 

def test_streaming_dataset_model_integration():
    from hex_ai.models import TwoHeadedResNet
    file_path = Path("tests/small_shuffled_test.pkl.gz")
    ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=2, max_examples=2, enable_augmentation=True)
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

def test_player_channel_correctness():
    file_path = Path("tests/small_shuffled_test.pkl.gz")
    ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=2, max_examples=2, enable_augmentation=True)
    for i in range(4):
        board, _, _ = ds[i]
        player_channel = board[2]
        assert torch.all((player_channel == 0) | (player_channel == 1)), "Player channel should be 0 or 1"

# ---
# Test: Empty board should not be augmented
# Expectation: Only one (non-augmented) example is returned for an empty board, even if augmentation is enabled.
# Any further access should raise IndexError. This matches the dataset logic: empty boards are not augmented.
# ---
def test_empty_board_handling(tmp_path):
    import gzip, pickle
    from hex_ai.config import BOARD_SIZE
    # Create a .pkl.gz file with an empty board example
    file_path = tmp_path / "empty_board_test.pkl.gz"
    empty_board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    example = {'board': empty_board, 'policy': np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 'value': 1.0}
    with gzip.open(file_path, "wb") as f:
        pickle.dump({"examples": [example]}, f)
    ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=1, max_examples=1, enable_augmentation=True)
    # Should return a single example for ds[0]
    board, policy, value = ds[0]
    assert isinstance(board, torch.Tensor)
    assert board.shape[0] == 2 or board.shape[0] == 3  # Accept 2 or 3 channels
    # Should raise IndexError for ds[1] (no augmentations for empty board)
    # This is by design: empty boards are not augmented.
    import pytest
    with pytest.raises(IndexError):
        _ = ds[1] 

# ---
# Test: Non-empty boards are augmented (4x examples per base example)
# Expectation: For each non-empty board, 4 augmented versions are returned if augmentation is enabled.
# The order is not guaranteed due to shuffling, so we use multiset (Counter) comparisons.
# ---
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
        ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=10, max_examples=None, enable_augmentation=True)
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
# Expectation: If enable_augmentation=False, only the original examples are returned, regardless of board content.
# ---
def test_non_augmented_path():
    file_path = Path("tests/small_shuffled_test.pkl.gz")
    ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=2, max_examples=2, enable_augmentation=False)
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
    ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=10, max_examples=3, enable_augmentation=False)
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

def test_get_augmented_example_logic(tmp_path):
    import gzip, pickle
    from hex_ai.config import BOARD_SIZE
    from unittest.mock import patch
    from collections import Counter
    # Create a .pkl.gz file with 2 examples
    file_path = tmp_path / "aug_logic_test.pkl.gz"
    examples = []
    for i in range(2):
        board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        board[0, 0, 0] = i
        examples.append({'board': board, 'policy': np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 'value': float(i)})
    with gzip.open(file_path, "wb") as f:
        pickle.dump({"examples": examples}, f)
    # Patch augmentation to return 4 augmentations per example
    with patch("hex_ai.data_utils.create_augmented_example_with_player_to_move") as mock_aug:
        mock_aug.side_effect = lambda board, policy, value, error_tracker: [
            (np.array(board), np.array(policy), float(value), 0),
            (np.array(board), np.array(policy), float(value), 1),
            (np.array(board), np.array(policy), float(value), 0),
            (np.array(board), np.array(policy), float(value), 1),
        ]
        # Set max_examples=None to process all base examples and get all augmentations
        ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=10, max_examples=None, enable_augmentation=True)
        # Collect all board[0,0,0] values using the public interface
        values = []
        i = 0
        while True:
            try:
                board, policy, value = ds[i]
                values.append(board[0, 0, 0].item())
                i += 1
            except IndexError:
                break
        # Should be one 0 (empty board, not augmented) and four 1s (non-empty, augmented)
        assert Counter(int(v) for v in values) == Counter([0, 1, 1, 1, 1])

# ---
# Test: Augmentation value label logic (multiset comparison)
# Expectation: For each base example, should get two 0s and two 1s (order doesn't matter)
# ---
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
        ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=10, max_examples=None, enable_augmentation=True)
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