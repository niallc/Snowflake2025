# =============================================================================
# IMPORTANT: To run this test, set PYTHONPATH to the project root!
# Example:
#     PYTHONPATH=. pytest tests/test_mini_epoch_orchestrator.py
# This is required so that 'import hex_ai' works correctly.
# =============================================================================

import torch
from torch.utils.data import DataLoader, TensorDataset
from hex_ai.mini_epoch_orchestrator import MiniEpochOrchestrator
import os
import tempfile

class MockTrainer:
    def __init__(self):
        self.train_calls = []
        self.validate_calls = 0
        self.checkpoints = []

    def train_on_batches(self, batch_iterable, epoch=None, mini_epoch=None):
        # Record the number of batches in this mini-epoch
        self.train_calls.append(len(list(batch_iterable)))
        return {'total_loss': 1.0, 'policy_loss': 0.5, 'value_loss': 0.5}

    def validate(self):
        self.validate_calls += 1
        return {'total_loss': 0.9, 'policy_loss': 0.4, 'value_loss': 0.5}

    def save_checkpoint(self, path, train_metrics, val_metrics):
        self.checkpoints.append(path)

# ---
# Test: Mini-epoch splitting and validation frequency
# 1. Tests MiniEpochOrchestrator (class) with a hand-rolled MockTrainer
# 2. Passes a DataLoader with 13 batches, mini-epoch size 5, 2 epochs
# 3. Expects train_on_batches to be called with [5,5,3,5,5,3] (3 mini-epochs per epoch)
# 4. Expects validate to be called after each mini-epoch (6 times total)
# ---
def test_mini_epoch_orchestrator_basic():
    """Test that mini-epoch splitting and validation frequency are correct for multiple epochs."""
    loader = make_loader(num_batches=13, batch_size=1)
    trainer = MockTrainer()
    orchestrator = MiniEpochOrchestrator(
        trainer=trainer,
        train_loader=loader,
        val_loader=loader,
        mini_epoch_batches=5,
        num_epochs=2,
        checkpoint_dir=None,
        log_interval=1
    )
    orchestrator.run()
    assert trainer.train_calls == [5,5,3,5,5,3]
    assert trainer.validate_calls == 6

# ---
# Test: Partial final mini-epoch handling
# 1. Tests MiniEpochOrchestrator (class) with a hand-rolled MockTrainer
# 2. Passes a DataLoader with 7 batches, mini-epoch size 3, 1 epoch
# 3. Expects train_on_batches to be called with [3,3,1] (last mini-epoch is partial)
# 4. Expects validate to be called after each mini-epoch (3 times total)
# ---
def test_mini_epoch_orchestrator_partial_final_mini_epoch():
    """Test that partial (non-full) final mini-epochs are handled correctly."""
    loader = make_loader(num_batches=7, batch_size=1)
    trainer = MockTrainer()
    orchestrator = MiniEpochOrchestrator(
        trainer=trainer,
        train_loader=loader,
        val_loader=loader,
        mini_epoch_batches=3,
        num_epochs=1,
        checkpoint_dir=None,
        log_interval=1
    )
    orchestrator.run()
    assert trainer.train_calls == [3,3,1]
    assert trainer.validate_calls == 3

# ---
# Helper: Create a DataLoader with a given number of batches
# ---
def make_loader(num_batches, batch_size=1):
    data = torch.randn(num_batches * batch_size, 2, 2)
    targets = torch.zeros(num_batches * batch_size, 2)
    dataset = TensorDataset(data, targets, targets)
    return DataLoader(dataset, batch_size=batch_size)

# ---
# Test: Checkpointing logic
# 1. Tests that save_checkpoint is called with expected paths
# 2. Uses a temp directory for checkpoint_dir
# ---
def test_mini_epoch_orchestrator_checkpointing():
    """Test that save_checkpoint is called with expected paths for each mini-epoch."""
    loader = make_loader(num_batches=6, batch_size=1)
    trainer = MockTrainer()
    with tempfile.TemporaryDirectory() as tmpdir:
        orchestrator = MiniEpochOrchestrator(
            trainer=trainer,
            train_loader=loader,
            val_loader=loader,
            mini_epoch_batches=2,
            num_epochs=1,
            checkpoint_dir=tmpdir,
            log_interval=1
        )
        orchestrator.run()
        # 6 batches, mini-epochs of 2: expect 3 mini-epochs, so 3 checkpoints
        assert len(trainer.checkpoints) == 3
        for i, path in enumerate(trainer.checkpoints):
            assert os.path.basename(path) == f"epoch1_mini{i+1}.pt"

# ---
# Test: log_interval logic
# 1. Tests that orchestrator runs correct number of mini-epochs regardless of log_interval
# ---
def test_mini_epoch_orchestrator_log_interval():
    """Test that log_interval does not affect mini-epoch execution/count."""
    loader = make_loader(num_batches=8, batch_size=1)
    trainer = MockTrainer()
    orchestrator = MiniEpochOrchestrator(
        trainer=trainer,
        train_loader=loader,
        val_loader=loader,
        mini_epoch_batches=3,
        num_epochs=1,
        checkpoint_dir=None,
        log_interval=2
    )
    orchestrator.run()
    # 8 batches, mini-epochs of 3: expect 3,3,2
    assert trainer.train_calls == [3,3,2]
    assert trainer.validate_calls == 3

# ---
# Test: No validation (val_loader=None)
# 1. Tests that validate is not called if val_loader is None
# ---
def test_mini_epoch_orchestrator_no_validation():
    """Test that no validation is performed if val_loader is None."""
    loader = make_loader(num_batches=5, batch_size=1)
    trainer = MockTrainer()
    orchestrator = MiniEpochOrchestrator(
        trainer=trainer,
        train_loader=loader,
        val_loader=None,
        mini_epoch_batches=2,
        num_epochs=1,
        checkpoint_dir=None,
        log_interval=1
    )
    orchestrator.run()
    assert trainer.validate_calls == 0
    assert trainer.train_calls == [2,2,1]

# ---
# Test: Different batch sizes
# 1. Tests that mini-epoch logic is based on batches, not samples
# ---
def test_mini_epoch_orchestrator_batch_size_gt_1():
    """Test that mini-epoch logic is based on batches, not samples (batch_size > 1)."""
    loader = make_loader(num_batches=6, batch_size=2)
    trainer = MockTrainer()
    orchestrator = MiniEpochOrchestrator(
        trainer=trainer,
        train_loader=loader,
        val_loader=loader,
        mini_epoch_batches=2,
        num_epochs=1,
        checkpoint_dir=None,
        log_interval=1
    )
    orchestrator.run()
    # 6 batches, mini-epochs of 2: expect 3 mini-epochs
    assert trainer.train_calls == [2,2,2]
    assert trainer.validate_calls == 3

# ---
# Test: Zero batches
# 1. Tests that no training or validation is called if there are no batches
# ---
def test_mini_epoch_orchestrator_zero_batches():
    """Test that no training or validation is performed if there are zero batches."""
    loader = make_loader(num_batches=0, batch_size=1)
    trainer = MockTrainer()
    orchestrator = MiniEpochOrchestrator(
        trainer=trainer,
        train_loader=loader,
        val_loader=loader,
        mini_epoch_batches=2,
        num_epochs=1,
        checkpoint_dir=None,
        log_interval=1
    )
    orchestrator.run()
    assert trainer.train_calls == []
    assert trainer.validate_calls == 0 