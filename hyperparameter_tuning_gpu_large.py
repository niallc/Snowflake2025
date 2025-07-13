#!/usr/bin/env python3
"""
GPU-based hyperparameter tuning for Hex AI - Large Scale Version.
Uses processed .pkl.gz files with GPU acceleration and mixed precision.
Updated to work with the new data format and training utilities.
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

import logging
logging.basicConfig(level=logging.DEBUG)

# Fix multiprocessing on macOS
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

from hex_ai.models import TwoHeadedResNet
from hex_ai.training_utils import (
    run_hyperparameter_tuning,
    discover_processed_files,
    estimate_dataset_size,
    create_experiment_config
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

# Quick hyperparameter exploration config
NUM_EPOCHS = 10
BATCH_SIZE = 64  # Larger batch size for GPU efficiency
TARGET_EXAMPLES = 2000  # Use ~2000 positions for quick exploration

# Define experiments
experiments = [
    {
        'experiment_name': 'gpu_baseline',
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
        'experiment_name': 'gpu_no_dropout',
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
        'experiment_name': 'gpu_high_dropout',
        'hyperparameters': {
            'learning_rate': 0.001,
            'batch_size': BATCH_SIZE,
            'dropout_prob': 0.2,
            'weight_decay': 1e-4,
            'policy_weight': 0.14,
            'value_weight': 0.86
        }
    },
    {
        'experiment_name': 'gpu_high_weight_decay',
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
        'experiment_name': 'gpu_lower_lr',
        'hyperparameters': {
            'learning_rate': 0.0005,
            'batch_size': BATCH_SIZE,
            'dropout_prob': 0.1,
            'weight_decay': 1e-4,
            'policy_weight': 0.14,
            'value_weight': 0.86
        }
    },
    {
        'experiment_name': 'gpu_higher_lr',
        'hyperparameters': {
            'learning_rate': 0.002,
            'batch_size': BATCH_SIZE,
            'dropout_prob': 0.1,
            'weight_decay': 1e-4,
            'policy_weight': 0.14,
            'value_weight': 0.86
        }
    }
]

# Create results directory
results_dir = Path("checkpoints/gpu_large_tuning")
results_dir.mkdir(parents=True, exist_ok=True)

# Save experiment configuration
config = {
    'num_epochs': NUM_EPOCHS,
    'batch_size': BATCH_SIZE,
    'device': str(device),
    'num_experiments': len(experiments),
    'timestamp': datetime.now().isoformat()
}
with open(results_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"\n{'='*60}")
print(f"Starting GPU Hyperparameter Tuning")
print(f"Device: {device}")
print(f"Epochs per experiment: {NUM_EPOCHS}")
print(f"Number of experiments: {len(experiments)}")
print(f"Results directory: {results_dir}")
print(f"{'='*60}")

# Discover and analyze data
print("\nDiscovering processed data files...")
data_files = discover_processed_files("data/processed")
# Use sampling for faster estimation (check first 10 files)
total_examples = estimate_dataset_size(data_files, max_files=10)
print(f"Found {len(data_files)} data files with approximately {total_examples:,} training examples")

# For quick exploration, use a subset of data
print(f"Using ~{TARGET_EXAMPLES:,} examples for quick hyperparameter exploration")
print("(This allows for fast iteration to find promising hyperparameters)")

# Run hyperparameter tuning with limited data for quick exploration
overall_results = run_hyperparameter_tuning(
    experiments=experiments,
    data_dir="data/processed",
    results_dir=str(results_dir),
    train_ratio=0.8,
    num_epochs=NUM_EPOCHS,
    early_stopping_patience=5,
    random_seed=42,
    max_examples_per_split=TARGET_EXAMPLES  # Limit data for quick exploration
)

print(f"\n{'='*60}")
print("GPU HYPERPARAMETER TUNING COMPLETE")
print(f"{'='*60}")
print(f"Total training time: {overall_results['total_training_time']:.1f}s")
print(f"Successful experiments: {overall_results['successful_experiments']}/{overall_results['num_experiments']}")

# Find best experiment
if overall_results['experiments']:
    best_exp = min(overall_results['experiments'], key=lambda x: x['best_val_loss'])
    print(f"\nBest experiment: {best_exp['experiment_name']}")
    print(f"Best validation loss: {best_exp['best_val_loss']:.6f}")
    print(f"Hyperparameters: {best_exp['hyperparameters']}")

print(f"\nAll results saved to: {results_dir}")
print("Check individual experiment directories for detailed results and checkpoints.") 