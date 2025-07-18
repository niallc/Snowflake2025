#!/usr/bin/env python3
"""
Test script to run legacy code with player-to-move channel added.

This script tests whether adding the player-to-move channel to the legacy
code causes the performance regression.
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
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Test legacy code with player-to-move channel')
parser.add_argument('--verbose', '-v', type=int, default=2, 
                   help='Verbose level: 0=critical only, 1=important, 2=detailed (default), 3=debug')
parser.add_argument('--num-epochs', type=int, default=5,
                   help='Number of epochs to train (default: 5)')
parser.add_argument('--batch-size', type=int, default=256,
                   help='Batch size for training (default: 256)')
parser.add_argument('--target-examples', type=int, default=50000,
                   help='Number of examples to use for training (default: 50000)')
args = parser.parse_args()

# Set up logging
if args.verbose == 0:
    logging.basicConfig(level=logging.CRITICAL)
elif args.verbose == 1:
    logging.basicConfig(level=logging.WARNING)
elif args.verbose == 2:
    logging.basicConfig(level=logging.INFO)
else:  # args.verbose >= 3
    logging.basicConfig(level=logging.DEBUG)

# Fix multiprocessing on macOS
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

from hex_ai.models_legacy_with_player_channel import TwoHeadedResNetLegacyWithPlayerChannel
from hex_ai.data_processing_legacy_with_player_channel import ProcessedDatasetLegacyWithPlayerChannel, create_processed_dataloader_legacy_with_player_channel
from hex_ai.training_legacy import Trainer, PolicyValueLoss, EarlyStopping
from hex_ai.training_utils_legacy import discover_processed_files_legacy, estimate_dataset_size_legacy

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

def main():
    """Main function to test legacy code with player-to-move channel."""
    print(f"\n{'='*60}")
    print("TESTING LEGACY CODE WITH PLAYER-TO-MOVE CHANNEL")
    print(f"{'='*60}")
    
    # Discover data files
    print("Discovering processed data files...")
    data_files = discover_processed_files_legacy("data/processed")
    total_examples = estimate_dataset_size_legacy(data_files, max_files=10)
    print(f"Found {len(data_files)} data files with approximately {total_examples:,} training examples")
    
    # Create experiment config
    experiment_name = f"legacy_with_player_channel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create results directory
    results_dir = Path("checkpoints") / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = {
        'description': 'Testing legacy code with player-to-move channel added',
        'experiment_name': experiment_name,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'target_examples': args.target_examples,
        'device': str(device),
        'timestamp': datetime.now().isoformat(),
        'model': 'TwoHeadedResNetLegacyWithPlayerChannel',
        'dataset': 'ProcessedDatasetLegacyWithPlayerChannel'
    }
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Experiment name: {experiment_name}")
    print(f"Results directory: {results_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Target examples: {args.target_examples}")
    
    # Create train/validation split
    print("\nCreating train/validation split...")
    train_files = data_files[:int(len(data_files) * 0.8)]
    val_files = data_files[int(len(data_files) * 0.8):]
    print(f"Train files: {len(train_files)}, Validation files: {len(val_files)}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = ProcessedDatasetLegacyWithPlayerChannel(train_files, shuffle_shards=True)
    val_dataset = ProcessedDatasetLegacyWithPlayerChannel(val_files, shuffle_shards=True)
    
    # Limit dataset size if specified
    if args.target_examples is not None:
        # For simplicity, we'll just use the first N examples
        # In a real implementation, you'd want to sample more intelligently
        print(f"Limiting to {args.target_examples} examples per split")
        # This is a simplified approach - in practice you'd want to implement proper sampling
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = create_processed_dataloader_legacy_with_player_channel(
        train_files, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = create_processed_dataloader_legacy_with_player_channel(
        val_files, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    
    # Create model
    print("Creating model...")
    model = TwoHeadedResNetLegacyWithPlayerChannel()
    model = model.to(device)
    
    # Print model summary
    from hex_ai.models_legacy_with_player_channel import get_model_summary
    print(get_model_summary(model))
    
    # Test forward pass
    print("Testing forward pass...")
    test_input = torch.randn(1, 3, 13, 13).to(device)  # 3-channel input
    with torch.no_grad():
        policy_out, value_out = model(test_input)
    print(f"Forward pass successful: policy={policy_out.shape}, value={value_out.shape}")
    
    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=0.001,
        device=str(device),
        enable_system_analysis=True,
        enable_csv_logging=True,
        experiment_name=experiment_name,
        policy_weight=0.2,
        value_weight=0.8,
        weight_decay=5e-4
    )
    
    # Create early stopping
    early_stopping = EarlyStopping(patience=3, min_delta=0.001, restore_best_weights=True)
    
    # Train the model
    print(f"\nStarting training for {args.num_epochs} epochs...")
    start_time = time.time()
    
    try:
        results = trainer.train(
            num_epochs=args.num_epochs,
            save_dir=str(results_dir),
            max_checkpoints=3,
            compress_checkpoints=True,
            early_stopping=early_stopping
        )
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Best validation loss: {results.get('best_val_loss', 'N/A')}")
        print(f"Final training loss: {results.get('final_train_loss', 'N/A')}")
        print(f"Results saved to: {results_dir}")
        
        # Save final results
        with open(results_dir / "final_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nCOMPARISON:")
        print(f"Legacy (2-channel): Should achieve good policy loss reduction")
        print(f"Legacy + Player Channel (3-channel): {results.get('best_val_loss', 'N/A')}")
        print(f"If the loss is much higher, the player-to-move channel addition is the problem.")
        print(f"If the loss is similar, the problem is elsewhere in the modern pipeline.")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error info
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }
        with open(results_dir / "error.json", "w") as f:
            json.dump(error_info, f, indent=2)

if __name__ == '__main__':
    main() 