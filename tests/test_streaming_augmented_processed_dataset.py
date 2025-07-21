import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from pathlib import Path

from hex_ai.data_pipeline import StreamingAugmentedProcessedDataset, PLAYER_CHANNEL

class DummyDataset(StreamingAugmentedProcessedDataset):
    def __init__(self, data_files, fake_examples, **kwargs):
        super().__init__(data_files, **kwargs)
        self.fake_examples = fake_examples
        self.data_files = data_files
        self.shuffle_files = kwargs.get('shuffle_files', False)
        self.max_examples = kwargs.get('max_examples', None)
        self.enable_augmentation = kwargs.get('enable_augmentation', True)
        self.effective_max_examples = self.max_examples * 4 if self.enable_augmentation and self.max_examples is not None else self.max_examples
        self.epoch_file_list = self._get_shuffled_file_list()
        self.current_file_idx = 0
        self.current_example_idx = 0
        self.total_examples_loaded = 0

    def _load_next_chunk(self):
        # Simulate loading a chunk from the current file
        self.current_chunk = self.fake_examples[self.epoch_file_list[self.current_file_idx]]

# Helper to create a fake example
def make_example(empty=False):
    if empty:
        board = np.zeros((3, 2, 2), dtype=np.float32)
    else:
        board = np.ones((3, 2, 2), dtype=np.float32)
    policy = np.array([0.5, 0.5], dtype=np.float32)
    value = np.float32(1.0)
    return (torch.from_numpy(board), torch.from_numpy(policy), torch.tensor(value))

@pytest.fixture
def fake_files():
    return ["file1", "file2", "file3"]

@pytest.fixture
def fake_examples(fake_files):
    # file1: non-empty, file2: empty, file3: non-empty
    return {
        "file1": make_example(empty=False),
        "file2": make_example(empty=True),
        "file3": make_example(empty=False),
    }

def test_no_file_repeats_within_epoch(fake_files, fake_examples):
    ds = DummyDataset(fake_files, fake_examples, shuffle_files=False, max_examples=3, enable_augmentation=False)
    seen = set()
    for i in range(len(fake_files)):
        ex = ds[i]
        seen.add(ds.epoch_file_list[ds.current_file_idx-1])
    assert seen == set(fake_files)

def test_shuffling_changes_order(fake_files, fake_examples):
    orders = []
    for _ in range(5):
        ds = DummyDataset(fake_files, fake_examples, shuffle_files=True, max_examples=3, enable_augmentation=False)
        orders.append(tuple(ds.epoch_file_list))
    # At least one order should be different
    assert len(set(orders)) > 1

def test_sample_limit_and_epoch_restart(fake_files, fake_examples):
    ds = DummyDataset(fake_files, fake_examples, shuffle_files=False, max_examples=2, enable_augmentation=False)
    with patch("builtins.print") as mock_print:
        for i in range(5):
            ds.__getitem__(i)
        # Should print the restart message at least once
        restart_msgs = [call for call in mock_print.call_args_list if "starting next epoch" in str(call)]
        assert len(restart_msgs) >= 1

def test_getitem_augmentation_logic(fake_files, fake_examples):
    # Patch the augmentation function to control output
    with patch("hex_ai.data_utils.create_augmented_example_with_player_to_move") as mock_aug:
        # Simulate 4 augmentations
        mock_aug.return_value = [
            (np.ones((2,2,2)), np.array([0.5,0.5]), 1.0, 1),
            (np.ones((2,2,2)), np.array([0.5,0.5]), 1.0, 1),
            (np.ones((2,2,2)), np.array([0.5,0.5]), 1.0, 1),
            (np.ones((2,2,2)), np.array([0.5,0.5]), 1.0, 1),
        ]
        ds = DummyDataset(fake_files, fake_examples, shuffle_files=False, max_examples=3, enable_augmentation=True)
        # Non-empty board returns 4 augmentations
        ex = ds[0]
        assert isinstance(ex, list)
        assert len(ex) == 4
    # Now test empty board returns just the original
    empty_example = make_example(empty=True)
    ds = DummyDataset(["file2"], {"file2": empty_example}, shuffle_files=False, max_examples=1, enable_augmentation=True)
    with patch("hex_ai.data_utils.create_augmented_example_with_player_to_move") as mock_aug:
        ex = ds[0]
        assert isinstance(ex, tuple) or (isinstance(ex, list) and len(ex) == 1) 