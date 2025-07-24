import os
import pytest
import torch
import numpy as np
from hex_ai.inference.model_wrapper import ModelWrapper

BATCH_FILE = "analysis/debugging/value_head_performance/batch0_epoch0_test.pkl"  # Update to a real file for actual test
CHECKPOINT = "checkpoints/hyperparameter_tuning/loss_weight_sweep_exp2_do0_pw0.001_f537d4_20250722_211936/epoch1_mini15.pt"  # Update to a real file for actual test

@pytest.mark.skipif(not os.path.exists(BATCH_FILE) or not os.path.exists(CHECKPOINT), reason="Test data or checkpoint not found.")
def test_infer_on_training_batch_smoke():
    import pickle
    with open(BATCH_FILE, 'rb') as f:
        batch = pickle.load(f)
    boards = batch['boards_cpu']
    values = batch['values_cpu']
    model = ModelWrapper(CHECKPOINT, device="cpu")
    with torch.no_grad():
        _, value_logits = model.batch_predict(boards)
        value_probs = torch.sigmoid(value_logits).squeeze(-1).numpy()
        targets = values.squeeze(-1).numpy() if values.ndim > 1 else values.numpy()
    assert value_probs.shape == targets.shape
    mse = np.mean((value_probs - targets) ** 2)
    print(f"Test batch MSE: {mse:.6f}")
    print(f"Test batch mean abs error: {np.mean(np.abs(value_probs - targets)):.6f}")
    print(f"Test batch mean target: {np.mean(targets):.4f}, mean prediction: {np.mean(value_probs):.4f}") 