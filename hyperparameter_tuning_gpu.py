#!/usr/bin/env python3
"""
GPU-based hyperparameter tuning for Hex AI, using processed .pkl.gz shards.
Tests GPU training on a small dataset (2000 games, 2 epochs).
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

# Small test config
DATASET_SIZE = 2000
NUM_EPOCHS = 2
BATCH_SIZE = 32

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

# Model and experiment config
experiment_name = "gpu_test_small"
results_dir = Path("checkpoints/gpu_test")
results_dir.mkdir(parents=True, exist_ok=True)
exp_dir = results_dir / experiment_name
exp_dir.mkdir(exist_ok=True)

hyperparams = {
    'learning_rate': 0.001,
    'batch_size': BATCH_SIZE,
    'dropout_prob': 0.1,
    'weight_decay': 1e-4,
    'policy_weight': 0.14,
    'value_weight': 0.86
}

# Save experiment config
with open(exp_dir / "config.json", "w") as f:
    json.dump(hyperparams, f, indent=2)

# Create model
model = TwoHeadedResNet(dropout_prob=hyperparams['dropout_prob'])
model = model.to(device)

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=hyperparams['learning_rate'],
    device=device,
    enable_system_analysis=True,
    policy_weight=hyperparams['policy_weight'],
    value_weight=hyperparams['value_weight'],
    weight_decay=hyperparams['weight_decay']
)

# Train
print(f"\n{'='*60}")
print(f"Running GPU training test: {experiment_name}")
print(f"Device: {device}")
print(f"Hyperparameters: {hyperparams}")
print(f"{'='*60}")
start_time = time.time()
results = trainer.train(
    num_epochs=NUM_EPOCHS,
    save_dir=str(exp_dir),
    early_stopping=None
)
training_time = time.time() - start_time

print(f"Results keys: {list(results.keys())}")

# Extract metrics
val_losses = results.get('val_losses', [])
train_losses = results.get('train_losses', [])
best_val_loss = min(val_losses) if val_losses else float('inf')
best_train_loss = min(train_losses) if train_losses else float('inf')
final_val_loss = val_losses[-1] if val_losses else float('inf')
final_train_loss = train_losses[-1] if train_losses else float('inf')

experiment_results = {
    'experiment_name': experiment_name,
    'hyperparameters': hyperparams,
    'best_val_loss': best_val_loss,
    'best_train_loss': best_train_loss,
    'final_val_loss': final_val_loss,
    'final_train_loss': final_train_loss,
    'training_time': training_time,
    'epochs_trained': len(results['train_losses']),
    'all_metrics': results
}

print(f"Results for {experiment_name}:")
print(f"  Best val loss: {best_val_loss:.6f}")
print(f"  Final val loss: {final_val_loss:.6f}")
print(f"  Training time: {training_time:.1f}s")
print(f"  Epochs trained: {len(results['train_losses'])}")

# Save experiment results
with open(exp_dir / "experiment_results.json", "w") as f:
    json.dump(experiment_results, f, indent=2, default=str)

print("\nTest completed! Check results in checkpoints/gpu_test/") 