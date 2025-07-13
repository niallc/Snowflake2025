#!/usr/bin/env python3
"""
Scaled-up hyperparameter tuning for Hex AI.
- 2000+ positions per experiment
- 5 epochs per experiment  
- 6 experiment arms with different hyperparameters
- Comprehensive loss tracking
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

# Scaled-up configuration
NUM_EPOCHS = 5
BATCH_SIZE = 64
TARGET_EXAMPLES = 2500  # More realistic dataset size

# Define comprehensive experiments
experiments = [
    {
        'experiment_name': 'baseline',
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
        'experiment_name': 'high_lr',
        'hyperparameters': {
            'learning_rate': 0.003,
            'batch_size': BATCH_SIZE,
            'dropout_prob': 0.1,
            'weight_decay': 1e-4,
            'policy_weight': 0.14,
            'value_weight': 0.86
        }
    },
    {
        'experiment_name': 'low_lr',
        'hyperparameters': {
            'learning_rate': 0.0003,
            'batch_size': BATCH_SIZE,
            'dropout_prob': 0.1,
            'weight_decay': 1e-4,
            'policy_weight': 0.14,
            'value_weight': 0.86
        }
    },
    {
        'experiment_name': 'no_dropout',
        'hyperparameters': {
            'learning_rate': 0.001,
            'batch_size': BATCH_SIZE,
            'dropout_prob': 0.0,
            'weight_decay': 1e-4,
            'policy_weight': 0.14,
            'value_weight': 0.86
        }
    },
    {
        'experiment_name': 'high_weight_decay',
        'hyperparameters': {
            'learning_rate': 0.001,
            'batch_size': BATCH_SIZE,
            'dropout_prob': 0.1,
            'weight_decay': 1e-3,
            'policy_weight': 0.14,
            'value_weight': 0.86
        }
    },
    {
        'experiment_name': 'balanced_weights',
        'hyperparameters': {
            'learning_rate': 0.001,
            'batch_size': BATCH_SIZE,
            'dropout_prob': 0.1,
            'weight_decay': 1e-4,
            'policy_weight': 0.5,
            'value_weight': 0.5
        }
    }
]

# Create results directory
results_dir = Path("checkpoints/scaled_tuning")
results_dir.mkdir(parents=True, exist_ok=True)

# Save configuration
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
print(f"SCALED-UP HYPERPARAMETER TUNING")
print(f"Device: {device}")
print(f"Epochs per experiment: {NUM_EPOCHS}")
print(f"Target examples: {TARGET_EXAMPLES}")
print(f"Number of experiments: {len(experiments)}")
print(f"Results directory: {results_dir}")
print(f"{'='*60}")

# Discover and analyze data
print("\nDiscovering processed data files...")
data_files = discover_processed_files("data/processed")
total_examples = estimate_dataset_size(data_files, max_files=10)  # Sample more files
print(f"Found {len(data_files)} data files with approximately {total_examples:,} training examples")

print(f"Using {TARGET_EXAMPLES} examples for training")
print("(This provides enough data for meaningful results)")

# Run scaled-up hyperparameter tuning
start_time = time.time()
overall_results = run_hyperparameter_tuning(
    experiments=experiments,
    data_dir="data/processed",
    results_dir=str(results_dir),
    train_ratio=0.8,
    num_epochs=NUM_EPOCHS,
    early_stopping_patience=3,  # Early stopping after 3 epochs without improvement
    random_seed=42,
    max_examples_per_split=TARGET_EXAMPLES
)

total_time = time.time() - start_time

print(f"\n{'='*60}")
print("SCALED-UP TUNING COMPLETE")
print(f"{'='*60}")
print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
print(f"Successful experiments: {overall_results['successful_experiments']}/{overall_results['num_experiments']}")

# Find best experiment
if overall_results['experiments']:
    best_exp = min(overall_results['experiments'], key=lambda x: x['best_val_loss'])
    print(f"\nBest experiment: {best_exp['experiment_name']}")
    print(f"Best validation loss: {best_exp['best_val_loss']:.6f}")
    print(f"Hyperparameters: {best_exp['hyperparameters']}")
    
    # Show all results sorted by validation loss
    print(f"\nAll experiments ranked by validation loss:")
    sorted_experiments = sorted(overall_results['experiments'], key=lambda x: x['best_val_loss'])
    for i, exp in enumerate(sorted_experiments):
        print(f"{i+1}. {exp['experiment_name']}: {exp['best_val_loss']:.6f}")
else:
    print("\nNo successful experiments!")

print(f"\nAll results saved to: {results_dir}")
print("Check individual experiment directories for detailed results and checkpoints.") 