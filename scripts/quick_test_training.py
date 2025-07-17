#!/usr/bin/env python3
"""
Quick test training with improved hyperparameters.
"""

import argparse
import logging
from pathlib import Path
import torch

from hex_ai.training_utils import get_device
from hex_ai.data_pipeline import discover_processed_files, create_train_val_split, StreamingProcessedDataset
from hex_ai.models import TwoHeadedResNet
from hex_ai.training import Trainer
from hex_ai.training import EarlyStopping
from hex_ai.training_logger import TrainingLogger

def main():
    parser = argparse.ArgumentParser(description="Quick test training with improved hyperparameters")
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (smaller for stability)')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate (reduced)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability (increased)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--policy-weight', type=float, default=0.3, help='Policy loss weight (increased)')
    parser.add_argument('--value-weight', type=float, default=0.7, help='Value loss weight (reduced)')
    parser.add_argument('--max-samples', type=int, default=100000, help='Max samples for quick test')
    parser.add_argument('--data-dir', type=str, default='data/processed', help='Data directory')
    parser.add_argument('--results-dir', type=str, default='checkpoints', help='Results directory')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Device setup
    device = get_device()
    print(f"Using device: {device}")
    
    # Discover data files
    data_files = discover_processed_files(args.data_dir)
    if args.max_samples and args.max_samples < len(data_files):
        data_files = data_files[:args.max_samples]
    print(f"Found {len(data_files)} data files.")
    
    # Train/val split
    train_files, val_files = create_train_val_split(data_files, train_ratio=0.8, random_seed=42)
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
    
    # Create datasets
    train_dataset = StreamingProcessedDataset(train_files, chunk_size=args.batch_size)
    val_dataset = StreamingProcessedDataset(val_files, chunk_size=args.batch_size)
    
    # Model
    model = TwoHeadedResNet(dropout_prob=args.dropout)
    model = model.to(device)
    
    # Results directory
    from datetime import datetime
    experiment_name = f"quick_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = Path(args.results_dir) / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Trainer with improved settings
    trainer = Trainer(
        model=model,
        train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False),
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
    
    print(f"Mixed precision enabled: {trainer.mixed_precision.use_mixed_precision}")
    
    # Early stopping
    early_stopping = EarlyStopping(patience=2)
    
    # Training
    print(f"\n{'='*60}\nStarting quick test training for {args.epochs} epochs\n{'='*60}")
    training_results = trainer.train(
        num_epochs=args.epochs,
        save_dir=str(results_dir),
        early_stopping=early_stopping
    )
    
    # Print summary
    print(f"\n{'='*60}\nQuick test complete\n{'='*60}")
    print(f"Best validation loss: {min(training_results['val_losses']):.6f}")
    print(f"Best training loss: {min(training_results['train_losses']):.6f}")
    print(f"Final policy loss: {training_results['train_policy_losses'][-1]:.4f}")
    print(f"Final value loss: {training_results['train_value_losses'][-1]:.4f}")
    print(f"Results saved to: {results_dir}")
    
    return results_dir

if __name__ == "__main__":
    main() 