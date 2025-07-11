#!/usr/bin/env python3
"""
Exploratory hyperparameter tuning for Hex AI.

This script runs a small hyperparameter search on 2,000 games to identify
the biggest gains before scaling up to larger datasets.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import numpy as np
from datetime import datetime

from hex_ai.models import TwoHeadedResNet
from hex_ai.training import Trainer
from hex_ai.data_processing import create_processed_dataloader

from hex_ai.config import BOARD_SIZE, POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE


def load_small_dataset(num_games: int = 2000) -> Tuple[List[str], List[str]]:
    """Load a small dataset for quick experimentation."""
    print(f"Loading {num_games} games for hyperparameter tuning...")
    
    # Find available processed shard files
    processed_dir = Path("data/processed")
    shard_files = list(processed_dir.glob("*.pkl.gz")) + list(processed_dir.glob("*.pkl"))
    
    if not shard_files:
        raise FileNotFoundError("No processed shard files found in data/processed/ directory")
    
    print(f"Found {len(shard_files)} processed shard files")
    
    # Estimate total games from shards
    total_estimated_games = len(shard_files) * 1000  # Each shard has up to 1000 games
    print(f"Estimated total games available: {total_estimated_games}")
    
    # Take a subset for quick experimentation
    if len(shard_files) > num_games // 1000:
        # Take enough shards to get approximately num_games
        num_shards_needed = max(1, num_games // 1000)
        np.random.shuffle(shard_files)
        shard_files = shard_files[:num_shards_needed]
        print(f"Using {len(shard_files)} shards for {num_games} games")
    
    # Split into train/validation
    np.random.shuffle(shard_files)
    split_idx = int(0.8 * len(shard_files))
    train_files = [str(f) for f in shard_files[:split_idx]]
    val_files = [str(f) for f in shard_files[split_idx:]]
    
    print(f"Dataset split: {len(train_files)} train shards, {len(val_files)} validation shards")
    return train_files, val_files


def run_hyperparameter_experiment(
    train_files: List[str],
    val_files: List[str],
    experiment_name: str,
    hyperparams: Dict,
    num_epochs: int = 20
) -> Dict:
    """Run a single hyperparameter experiment."""
    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_name}")
    print(f"Hyperparameters: {hyperparams}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Create model
    model = TwoHeadedResNet()
    
    # Ensure checkpoint directory exists
    save_dir = Path(f"checkpoints/exploratory/{experiment_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders from processed shards
    train_loader = create_processed_dataloader(
        [Path(f) for f in train_files],
        batch_size=hyperparams.get('batch_size', 32),
        shuffle=True,
        num_workers=0
    )
    val_loader = create_processed_dataloader(
        [Path(f) for f in val_files],
        batch_size=hyperparams.get('batch_size', 32),
        shuffle=False,
        num_workers=0
    ) if val_files else None
    
    # Construct Trainer directly
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=hyperparams.get('learning_rate', 0.001),
        device=None,
        enable_system_analysis=True
    )
    
    # Train model
    results = trainer.train(
        num_epochs=num_epochs,
        save_dir=str(save_dir),
        early_stopping=None
    )
    
    training_time = time.time() - start_time
    
    # Extract key metrics
    best_val_loss = min(results['val_losses']) if results['val_losses'] else float('inf')
    best_train_loss = min(results['train_losses']) if results['train_losses'] else float('inf')
    final_val_loss = results['val_losses'][-1] if results['val_losses'] else float('inf')
    final_train_loss = results['train_losses'][-1] if results['train_losses'] else float('inf')
    
    experiment_results = {
        'experiment_name': experiment_name,
        'hyperparameters': hyperparams,
        'best_val_loss': best_val_loss,
        'best_train_loss': best_train_loss,
        'final_val_loss': final_val_loss,
        'final_train_loss': final_train_loss,
        'training_time': training_time,
        'epochs_trained': len(results['train_losses']),
        'early_stopped': len(results['train_losses']) < num_epochs,
        'all_metrics': results
    }
    
    print(f"Results for {experiment_name}:")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Final val loss: {final_val_loss:.6f}")
    print(f"  Training time: {training_time:.1f}s")
    print(f"  Epochs trained: {len(results['train_losses'])}")
    
    return experiment_results


def run_exploratory_search():
    """Run exploratory hyperparameter search."""
    print("Starting exploratory hyperparameter tuning on 2,000 games...")
    
    # Load small dataset
    train_files, val_files = load_small_dataset(2000)
    
    # Define hyperparameter combinations to test
    experiments = [
        {
            'name': 'baseline',
            'hyperparams': {
                'learning_rate': 0.001,
                'batch_size': 64
            }
        },
        {
            'name': 'higher_lr',
            'hyperparams': {
                'learning_rate': 0.003,
                'batch_size': 64
            }
        },
        {
            'name': 'lower_lr',
            'hyperparams': {
                'learning_rate': 0.0003,
                'batch_size': 64
            }
        },
        {
            'name': 'larger_batch',
            'hyperparams': {
                'learning_rate': 0.001,
                'batch_size': 128
            }
        },
        {
            'name': 'smaller_batch',
            'hyperparams': {
                'learning_rate': 0.001,
                'batch_size': 32
            }
        },
        {
            'name': 'high_lr_large_batch',
            'hyperparams': {
                'learning_rate': 0.003,
                'batch_size': 128
            }
        },
        {
            'name': 'low_lr_small_batch',
            'hyperparams': {
                'learning_rate': 0.0003,
                'batch_size': 32
            }
        }
    ]
    
    # Create results directory
    results_dir = Path("exploratory_results")
    results_dir.mkdir(exist_ok=True)
    
    all_results = []
    
    # Run experiments
    for experiment in experiments:
        try:
            results = run_hyperparameter_experiment(
                train_files=train_files,
                val_files=val_files,
                experiment_name=experiment['name'],
                hyperparams=experiment['hyperparams'],
                num_epochs=20
            )
            all_results.append(results)
            
            # Save intermediate results
            with open(results_dir / "intermediate_results.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Error in experiment {experiment['name']}: {e}")
            continue
    
    # Analyze results
    print(f"\n{'='*60}")
    print("EXPLORATORY SEARCH RESULTS")
    print(f"{'='*60}")
    
    # Sort by best validation loss
    all_results.sort(key=lambda x: x['best_val_loss'])
    
    print("\nRanked by best validation loss:")
    for i, result in enumerate(all_results, 1):
        print(f"{i}. {result['experiment_name']}: {result['best_val_loss']:.6f}")
        print(f"   Hyperparams: {result['hyperparameters']}")
        print(f"   Training time: {result['training_time']:.1f}s")
        print()
    
    # Find best configuration
    best_result = all_results[0]
    print(f"BEST CONFIGURATION:")
    print(f"  Experiment: {best_result['experiment_name']}")
    print(f"  Hyperparameters: {best_result['hyperparameters']}")
    print(f"  Best val loss: {best_result['best_val_loss']:.6f}")
    print(f"  Training time: {best_result['training_time']:.1f}s")
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"exploratory_results_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump({
            'timestamp': timestamp,
            'dataset_size': len(train_files) + len(val_files),
            'experiments': all_results,
            'best_configuration': best_result
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    return best_result


if __name__ == "__main__":
    run_exploratory_search() 