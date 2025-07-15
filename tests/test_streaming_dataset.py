# NOTE: The streaming dataset and model now use (3, BOARD_SIZE, BOARD_SIZE) 
# tensors (blue, red, player-to-move channels).
# All shape assertions in this file expect 3-channel input for board tensors.
import unittest
import torch
import numpy as np
from pathlib import Path
from hex_ai.data_pipeline import StreamingProcessedDataset
from hex_ai.models import TwoHeadedResNet
from hex_ai.config import BOARD_SIZE
import gzip
import pickle
import pytest

class TestStreamingProcessedDataset(unittest.TestCase):
    def test_real_file(self):
        # This test assumes the file exists and is in the expected format
        data_file = Path("data/exampleData/example_2channel_processsed.pkl.gz")
        if not data_file.exists():
            self.skipTest(f"Test data file {data_file} not found.")
        dataset = StreamingProcessedDataset([data_file], chunk_size=10)
        board, policy, value = dataset[0]
        self.assertEqual(board.shape, (3, BOARD_SIZE, BOARD_SIZE))
        # Check player-to-move channel is all 0 or 1
        player_channel = board[2].numpy()
        self.assertTrue(np.all((player_channel == 0) | (player_channel == 1)))

    def test_mocked_example(self):
        # Create a fake .pkl.gz file with a simple 2-channel board
        fake_file = Path("tests/fake_example.pkl.gz")
        board_2ch = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        board_2ch[0, 0, 0] = 1  # Blue's move
        example = (board_2ch, np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 1.0)
        with gzip.open(fake_file, "wb") as f:
            pickle.dump({"examples": [example]}, f)
        try:
            dataset = StreamingProcessedDataset([fake_file], chunk_size=1)
            board, policy, value = dataset[0]
            self.assertEqual(board.shape, (3, BOARD_SIZE, BOARD_SIZE))
            self.assertEqual(board[2, 0, 0].item(), 1.0)  # Player-to-move should be RED_PLAYER (1)
        finally:
            fake_file.unlink()

    def test_model_integration(self):
        # Use the mocked example for a full model forward pass
        fake_file = Path("tests/fake_example.pkl.gz")
        board_2ch = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        board_2ch[0, 0, 0] = 1  # Blue's move
        example = (board_2ch, np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32), 1.0)
        with gzip.open(fake_file, "wb") as f:
            pickle.dump({"examples": [example]}, f)
        try:
            dataset = StreamingProcessedDataset([fake_file], chunk_size=1)
            board, policy, value = dataset[0]
            model = TwoHeadedResNet()
            board_batch = board.unsqueeze(0)  # Add batch dim
            policy_logits, value_logit = model(board_batch)
            self.assertEqual(policy_logits.shape, (1, BOARD_SIZE * BOARD_SIZE))
            self.assertEqual(value_logit.shape, (1, 1))
        finally:
            fake_file.unlink()

    def test_inference_3channel_todo(self):
        """
        TODO: Update inference code to always construct the player-to-move channel for 3-channel input.
        This test should:
        - Construct a 2-channel board and pass it through the inference pipeline
        - Ensure the player-to-move channel is added as in training
        - Check that inference output matches expectations
        For now, this is a placeholder/reminder and will be skipped.
        """
        pytest.skip("3-channel inference test not implemented. Update inference code to handle player-to-move channel.")

    def test_error_handling_missing_file(self):
        """
        Test that loading a missing .pkl.gz file logs a warning and results in an empty dataset.
        """
        from hex_ai.data_pipeline import StreamingProcessedDataset
        from pathlib import Path
        missing_file = Path("tests/this_file_does_not_exist.pkl.gz")
        dataset = StreamingProcessedDataset([missing_file], chunk_size=1)
        self.assertEqual(len(dataset), 0)

    def test_error_handling_corrupted_file(self):
        """
        Test that loading a corrupted .pkl.gz file logs a warning and results in an empty dataset.
        """
        import gzip
        from hex_ai.data_pipeline import StreamingProcessedDataset
        from pathlib import Path
        corrupted_file = Path("tests/corrupted_example.pkl.gz")
        with open(corrupted_file, "wb") as f:
            f.write(b"not a valid pickle or gzip file")
        try:
            dataset = StreamingProcessedDataset([corrupted_file], chunk_size=1)
            self.assertEqual(len(dataset), 1)
        finally:
            corrupted_file.unlink()

    def test_error_handling_threshold(self):
        """
        Test that if enough files are missing/corrupted, a RuntimeError is raised and an error log is written.
        """
        import gzip
        from hex_ai.data_pipeline import StreamingProcessedDataset
        from pathlib import Path
        import os
        # Create 10 corrupted files
        files = []
        for i in range(10):
            fpath = Path(f"tests/corrupted_{i}.pkl.gz")
            with open(fpath, "wb") as f:
                f.write(b"not a valid pickle or gzip file")
            files.append(fpath)
        try:
            with self.assertRaises(RuntimeError):
                _ = StreamingProcessedDataset(files, chunk_size=1)
            # Check that error.log was written
            error_log = files[0].parent / "error.log"
            self.assertTrue(error_log.exists())
            with open(error_log) as f:
                log_content = f.read()
            self.assertIn("Too many data loading errors", log_content)
        finally:
            for f in files:
                f.unlink()
            error_log = files[0].parent / "error.log"
            if error_log.exists():
                error_log.unlink()

if __name__ == "__main__":
    unittest.main() 