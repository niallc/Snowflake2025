import pytest
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from hex_ai.data_pipeline import StreamingAugmentedProcessedDataset
from hex_ai.training import Trainer
from hex_ai.models import TwoHeadedResNet
from hex_ai.mini_epoch_orchestrator import MiniEpochOrchestrator

# ---
# Integration Test: MiniEpochOrchestrator + Trainer + TwoHeadedResNet + real data file
# 1. Loads a real small test file (tests/small_shuffled_test.pkl.gz) using StreamingAugmentedProcessedDataset
# 2. Runs a real Trainer and MiniEpochOrchestrator for 1 epoch, mini-epoch size 2
# 3. Asserts that training completes, model parameters change, and at least one checkpoint is created
# 4. Skips the test if the test file does not exist
# ---
@pytest.mark.skipif(not Path('tests/small_shuffled_test.pkl.gz').exists(), reason="Test data file not found.")
def test_integration_mini_epoch_real_data(tmp_path):
    """
    Integration test: Run MiniEpochOrchestrator + Trainer + TwoHeadedResNet on a real small test file.
    Asserts that training completes, model parameters change, and checkpoints are created.
    """
    file_path = Path('tests/small_shuffled_test.pkl.gz')
    ds = StreamingAugmentedProcessedDataset([file_path], chunk_size=2, max_examples_unaugmented=4, enable_augmentation=True)
    loader = DataLoader(ds, batch_size=2)
    model = TwoHeadedResNet()
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        learning_rate=0.01,
        device='cpu',
        enable_system_analysis=False,
        enable_csv_logging=False,
        experiment_name='integration_test'
    )
    # Save initial model params for comparison
    initial_params = [p.clone().detach() for p in model.parameters()]
    orchestrator = MiniEpochOrchestrator(
        trainer=trainer,
        train_loader=loader,
        val_loader=loader,
        mini_epoch_batches=2,
        num_epochs=1,
        checkpoint_dir=tmp_path,
        log_interval=1
    )
    orchestrator.run()
    # Assert model parameters have changed (training happened)
    changed = any(not torch.equal(p0, p1) for p0, p1 in zip(initial_params, model.parameters()))
    assert changed, "Model parameters did not change after training."
    # Assert checkpoints were created
    checkpoint_files = list(tmp_path.glob('*.pt'))
    assert len(checkpoint_files) > 0, "No checkpoints were created." 