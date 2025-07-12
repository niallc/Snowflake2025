#!/usr/bin/env python3
"""
GPU-based hyperparameter tuning for Hex AI - Large Scale Version.
Uses processed .pkl.gz shards with GPU acceleration and mixed precision.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List
import torch
import numpy as np
from datetime import datetime

from hex_ai.models import TwoHeadedResNet
from hex_ai.training import Trainer
from hex_ai.data_processing import create_processed_dataloader

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

# Large test config
DATASET_SIZE = 10000  # 10k games
NUM_EPOCHS = 10
BATCH_SIZE = 64  # Larger batch size for GPU efficiency

# Find available processed shard files
processed_dir = Path("data/processed")
shard_files = list(processed_dir.glob("*.pkl.gz")) + list(processed_dir.glob("*.pkl"))
if not shard_files:
    raise FileNotFoundError("No processed shard files found in data/processed/")
print(f"Found {len(shard_files)} processed shard files")

# Take enough shards to get ~DATASET_SIZE games
num_shards_needed = max(1, DATASET_SIZE // 1000)
np.random.shuffle(shard_files)
shard_files = shard_files[:num_shards_needed]
print(f"Using {len(shard_files)} shards for {DATASET_SIZE} games")

# Split into train/validation
np.random.shuffle(shard_files)
split_idx = int(0.8 * len(shard_files))
train_files = [Path(f) for f in shard_files[:split_idx]]
val_files = [Path(f) for f in shard_files[split_idx:]]
print(f"Dataset split: {len(train_files)} train shards, {len(val_files)} validation shards")

# Create dataloaders
train_loader = create_processed_dataloader(
    train_files,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)
val_loader = create_processed_dataloader(
    val_files,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
) if val_files else None

# Define experiments
experiments = [
    {
        'name': 'gpu_baseline',
        'hyperparams': {
            'learning_rate': 0.001,
            'batch_size': BATCH_SIZE,
            'dropout_prob': 0.1,
            'weight_decay': 1e-4,
            'policy_weight': 0.14,
            'value_weight': 0.86
        }
    },
    {
        'name': 'gpu_no_dropout',
        'hyperparams': {
            'learning_rate': 0.001,
            'batch_size': BATCH_SIZE,
            'dropout_prob': 0.0,
            'weight_decay': 1e-4,
            'policy_weight': 0.14,
            'value_weight': 0.86
        }
    },
    {
        'name': 'gpu_high_dropout',
        'hyperparams': {
            'learning_rate': 0.001,
            'batch_size': BATCH_SIZE,
            'dropout_prob': 0.2,
            'weight_decay': 1e-4,
            'policy_weight': 0.14,
            'value_weight': 0.86
        }
    },
    {
        'name': 'gpu_high_weight_decay',
        'hyperparams': {
            'learning_rate': 0.001,
            'batch_size': BATCH_SIZE,
            'dropout_prob': 0.1,
            'weight_decay': 1e-3,
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
    'dataset_size': DATASET_SIZE,
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
print(f"Dataset: {DATASET_SIZE} games")
print(f"Epochs per experiment: {NUM_EPOCHS}")
print(f"Number of experiments: {len(experiments)}")
print(f"Results directory: {results_dir}")
print(f"{'='*60}")

all_results = []
total_start_time = time.time()

for i, exp in enumerate(experiments):
    print(f"\n{'='*60}")
    print(f"Experiment {i+1}/{len(experiments)}: {exp['name']}")
    print(f"Hyperparameters: {exp['hyperparams']}")
    print(f"{'='*60}")
    
    exp_start_time = time.time()
    
    # Create experiment directory
    exp_dir = results_dir / exp['name']
    exp_dir.mkdir(exist_ok=True)
    
    # Save experiment config
    with open(exp_dir / "config.json", "w") as f:
        json.dump(exp['hyperparams'], f, indent=2)
    
    # Create model
    model = TwoHeadedResNet(dropout_prob=exp['hyperparams']['dropout_prob'])
    model = model.to(device)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=exp['hyperparams']['learning_rate'],
        device=device,
        enable_system_analysis=True,
        policy_weight=exp['hyperparams']['policy_weight'],
        value_weight=exp['hyperparams']['value_weight'],
        weight_decay=exp['hyperparams']['weight_decay']
    )
    
    # Train
    results = trainer.train(
        num_epochs=NUM_EPOCHS,
        save_dir=str(exp_dir),
        early_stopping=None
    )
    
    exp_time = time.time() - exp_start_time
    
    # Extract metrics
    best_val_loss = min(results['val_losses']) if results['val_losses'] else float('inf')
    best_train_loss = min(results['train_losses']) if results['train_losses'] else float('inf')
    final_val_loss = results['val_losses'][-1] if results['val_losses'] else float('inf')
    final_train_loss = results['train_losses'][-1] if results['train_losses'] else float('inf')
    
    experiment_results = {
        'experiment_name': exp['name'],
        'hyperparameters': exp['hyperparams'],
        'best_val_loss': best_val_loss,
        'best_train_loss': best_train_loss,
        'final_val_loss': final_val_loss,
        'final_train_loss': final_train_loss,
        'training_time': exp_time,
        'epochs_trained': len(results['train_losses']),
        'early_stopped': results['early_stopped'],
        'all_metrics': results
    }
    
    all_results.append(experiment_results)
    
    print(f"Results for {exp['name']}:")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Final val loss: {final_val_loss:.6f}")
    print(f"  Training time: {exp_time:.1f}s")
    print(f"  Epochs trained: {len(results['train_losses'])}")
    
    # Save experiment results
    with open(exp_dir / "experiment_results.json", "w") as f:
        json.dump(experiment_results, f, indent=2, default=str)

# Save overall results
total_time = time.time() - total_start_time
overall_results = {
    'total_training_time': total_time,
    'num_experiments': len(experiments),
    'device': str(device),
    'experiments': all_results
}

with open(results_dir / "overall_results.json", "w") as f:
    json.dump(overall_results, f, indent=2, default=str)

# Print summary
print(f"\n{'='*60}")
print("GPU HYPERPARAMETER TUNING COMPLETE")
print(f"{'='*60}")
print(f"Total training time: {total_time:.1f}s")
print(f"Average time per experiment: {total_time/len(experiments):.1f}s")

# Find best experiment
best_exp = min(all_results, key=lambda x: x['best_val_loss'])
print(f"\nBest experiment: {best_exp['experiment_name']}")
print(f"Best validation loss: {best_exp['best_val_loss']:.6f}")
print(f"Hyperparameters: {best_exp['hyperparameters']}")

print(f"\nAll results saved to: {results_dir}")
print("Check individual experiment directories for detailed results and checkpoints.") 