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