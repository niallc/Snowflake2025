import torch
from torch.utils.data import DataLoader, TensorDataset
from hex_ai.mini_epoch_orchestrator import MiniEpochOrchestrator

class MockTrainer:
    def __init__(self):
        self.train_calls = []
        self.validate_calls = 0
        self.checkpoints = []

    def train_on_batches(self, batch_iterable):
        # Record the number of batches in this mini-epoch
        self.train_calls.append(len(list(batch_iterable)))
        return {'total_loss': 1.0, 'policy_loss': 0.5, 'value_loss': 0.5}

    def validate(self):
        self.validate_calls += 1
        return {'total_loss': 0.9, 'policy_loss': 0.4, 'value_loss': 0.5}

    def save_checkpoint(self, path, train_metrics, val_metrics):
        self.checkpoints.append(path)

def make_loader(num_batches, batch_size=1):
    # Create a DataLoader with num_batches batches
    data = torch.randn(num_batches * batch_size, 2, 2)
    targets = torch.zeros(num_batches * batch_size, 2)
    dataset = TensorDataset(data, targets, targets)
    return DataLoader(dataset, batch_size=batch_size)

def test_mini_epoch_orchestrator_basic():
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
    # 13 batches per epoch, mini-epochs of 5: expect 3 mini-epochs per epoch (5+5+3)
    assert trainer.train_calls == [5,5,3,5,5,3]
    # Validation should be called after each mini-epoch
    assert trainer.validate_calls == 6

def test_mini_epoch_orchestrator_partial_final_mini_epoch():
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
    # 7 batches, mini-epochs of 3: expect 3,3,1
    assert trainer.train_calls == [3,3,1]
    assert trainer.validate_calls == 3 