#!/usr/bin/env python3
"""
Minimal hyperparameter tuning test.
- 100 positions total
- 1 epoch per experiment  
- 2 experiment arms
- Fast failure detection
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List
import torch
import numpy as np
from datetime import datetime
import multiprocessing

# Fix multiprocessing on macOS
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

from hex_ai.models import TwoHeadedResNet
from hex_ai.training_utils import (
    run_hyperparameter_tuning,
    discover_processed_files,
    estimate_dataset_size
)

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    print(f"Using CUDA GPU: {device_name}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS GPU")
else:
    device = torch.device("cpu")
    print("Using CPU (no GPU detected)")

# Minimal test configuration
NUM_EPOCHS = 1
BATCH_SIZE = 32
TARGET_EXAMPLES = 100  # Very small for fast testing

# Define minimal experiments
experiments = [
    {
        'experiment_name': 'test_baseline',
        'hyperparameters': {
            'learning_rate': 0.001,
            'batch_size': BATCH_SIZE,
            'dropout_prob': 0.1,
            'weight_decay': 1e-4,
            'policy_weight': 0.14,
            'value_weight': 0.86
        }
    },
    {
        'experiment_name': 'test_no_dropout',
        'hyperparameters': {
            'learning_rate': 0.001,
            'batch_size': BATCH_SIZE,
            'dropout_prob': 0.0,
            'weight_decay': 1e-4,
            'policy_weight': 0.14,
            'value_weight': 0.86
        }
    }
]

# Create results directory
results_dir = Path("checkpoints/minimal_test")
results_dir.mkdir(parents=True, exist_ok=True)

# Save test configuration
config = {
    'num_epochs': NUM_EPOCHS,
    'batch_size': BATCH_SIZE,
    'target_examples': TARGET_EXAMPLES,
    'device': str(device),
    'num_experiments': len(experiments),
    'timestamp': datetime.now().isoformat()
}
with open(results_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"\n{'='*60}")
print(f"MINIMAL HYPERPARAMETER TUNING TEST")
print(f"Device: {device}")
print(f"Epochs per experiment: {NUM_EPOCHS}")
print(f"Target examples: {TARGET_EXAMPLES}")
print(f"Number of experiments: {len(experiments)}")
print(f"Results directory: {results_dir}")
print(f"{'='*60}")

# Discover and analyze data
print("\nDiscovering processed data files...")
data_files = discover_processed_files("data/processed")
total_examples = estimate_dataset_size(data_files, max_files=5)  # Sample 5 files for speed
print(f"Found {len(data_files)} data files with approximately {total_examples:,} training examples")

print(f"Using {TARGET_EXAMPLES} examples for minimal testing")
print("(This allows for fast failure detection)")

# Run minimal hyperparameter tuning
start_time = time.time()
overall_results = run_hyperparameter_tuning(
    experiments=experiments,
    data_dir="data/processed",
    results_dir=str(results_dir),
    train_ratio=0.8,
    num_epochs=NUM_EPOCHS,
    early_stopping_patience=None,  # No early stopping for 1 epoch
    random_seed=42,
    max_examples_per_split=TARGET_EXAMPLES
)

total_time = time.time() - start_time

print(f"\n{'='*60}")
print("MINIMAL TEST COMPLETE")
print(f"{'='*60}")
print(f"Total time: {total_time:.1f}s")
print(f"Successful experiments: {overall_results['successful_experiments']}/{overall_results['num_experiments']}")

# Find best experiment
if overall_results['experiments']:
    best_exp = min(overall_results['experiments'], key=lambda x: x['best_val_loss'])
    print(f"\nBest experiment: {best_exp['experiment_name']}")
    print(f"Best validation loss: {best_exp['best_val_loss']:.6f}")
    print(f"Hyperparameters: {best_exp['hyperparameters']}")
else:
    print("\nNo successful experiments!")

print(f"\nAll results saved to: {results_dir}")
print("Check individual experiment directories for detailed results and checkpoints.") 