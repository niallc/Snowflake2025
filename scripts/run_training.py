#!/usr/bin/env python3
"""
Modern training entry point for Hex AI.
Uses StreamingProcessedDataset, Trainer, and robust device/mixed precision logic.
Suitable for both quick tests and full-scale runs.
"""

import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
import torch
import json

from hex_ai.training_utils import get_device
from hex_ai.training_utils import discover_processed_files, create_train_val_split, StreamingProcessedDataset
from hex_ai.models import TwoHeadedResNet
from hex_ai.training import Trainer

# Argument parsing for flexibility
parser = argparse.ArgumentParser(description="Train Hex AI model with modern pipeline.")
parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs (default: 2)')
parser.add_argument('--batch-size', type=int, default=256, help='Batch size (default: 256)')
parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate (default: 0.01)')
parser.add_argument('--dropout', type=float, default=0.05, help='Dropout probability (default: 0.05)')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay (default: 5e-4)')
parser.add_argument('--policy-weight', type=float, default=0.2, help='Policy loss weight (default: 0.2)')
parser.add_argument('--value-weight', type=float, default=0.8, help='Value loss weight (default: 0.8)')
parser.add_argument('--train-ratio', type=float, default=0.8, help='Train/val split ratio (default: 0.8)')
parser.add_argument('--max-samples', type=int, default=1024, help='Max total samples for quick test (default: 1024)')
parser.add_argument('--early-stopping', type=int, default=3, help='Early stopping patience (default: 3)')
parser.add_argument('--data-dir', type=str, default='data/processed', help='Directory with processed .pkl.gz files')
parser.add_argument('--results-dir', type=str, default='checkpoints', help='Directory to save checkpoints/results')
parser.add_argument('--experiment-name', type=str, default=None, help='Optional experiment name (default: timestamped)')
parser.add_argument('--verbose', type=int, default=2, help='Verbosity: 0=critical, 1=warning, 2=info, 3=debug')
args = parser.parse_args()

# Set up logging
if args.verbose == 0:
    logging.basicConfig(level=logging.CRITICAL)
elif args.verbose == 1:
    logging.basicConfig(level=logging.WARNING)
elif args.verbose == 2:
    logging.basicConfig(level=logging.INFO)
elif args.verbose >= 3:
    logging.basicConfig(level=logging.DEBUG)

# Device and experiment setup
start_time = time.time()
device = get_device()
print(f"Using device: {device}")

# Discover data files
data_files = discover_processed_files(args.data_dir)
if args.max_samples and args.max_samples < len(data_files):
    data_files = data_files[:args.max_samples]  # crude cap for quick test
print(f"Found {len(data_files)} data files.")

# Train/val split
train_files, val_files = create_train_val_split(data_files, train_ratio=args.train_ratio, random_seed=42)
print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

# Create datasets
train_dataset = StreamingProcessedDataset(train_files, chunk_size=args.batch_size)
val_dataset = StreamingProcessedDataset(val_files, chunk_size=args.batch_size)

# Model
model = TwoHeadedResNet(dropout_prob=args.dropout)
model = model.to(device)

# Results directory
experiment_name = args.experiment_name or f"hex_ai_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
results_dir = Path(args.results_dir) / experiment_name
results_dir.mkdir(parents=True, exist_ok=True)

# Trainer
trainer = Trainer(
    model=model,
    train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),
    val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False),
    learning_rate=args.learning_rate,
    device=device,
    enable_system_analysis=True,
    enable_csv_logging=True,
    experiment_name=experiment_name,
    policy_weight=args.policy_weight,
    value_weight=args.value_weight,
    weight_decay=args.weight_decay
)

# Print mixed precision status
print(f"Mixed precision enabled: {trainer.mixed_precision.use_mixed_precision}")

# Save config
config = vars(args)
config['device'] = str(device)
config['mixed_precision'] = trainer.mixed_precision.use_mixed_precision
config['timestamp'] = datetime.now().isoformat()
with open(results_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2)

# Training
print(f"\n{'='*60}\nStarting training for {args.epochs} epochs\n{'='*60}")
train_start = time.time()
training_results = trainer.train(
    num_epochs=args.epochs,
    save_dir=str(results_dir),
    early_stopping=args.early_stopping
)
train_time = time.time() - train_start

# Print summary
print(f"\n{'='*60}\nTraining complete\n{'='*60}")
print(f"Total training time: {train_time:.1f} seconds ({train_time/60:.1f} minutes)")
print(f"Best validation loss: {min(training_results['val_losses']):.6f}")
print(f"Best training loss: {min(training_results['train_losses']):.6f}")
print(f"Epochs trained: {training_results['epochs_trained']}")
print(f"Checkpoints and logs saved to: {results_dir}")
print(f"Mixed precision was {'ENABLED' if trainer.mixed_precision.use_mixed_precision else 'DISABLED'} for this run.")

# Save results
with open(results_dir / "training_results.json", "w") as f:
    json.dump(training_results, f, indent=2, default=str)

print(f"\n{'='*60}\nRun complete\n{'='*60}") 