#!/usr/bin/env python3
"""
Enhanced hyperparameter tuning for Hex AI - Second Iteration.

This script runs hyperparameter search with improved features:
- Dropout testing
- Weight decay testing  
- Loss weight balancing
- CSV logging
- Smart checkpoint retention
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


class EnhancedHyperparameterTuner:
    """Enhanced hyperparameter tuning with dropout, weight decay, and loss weights."""
    
    def __init__(self, dataset_size: int = 10000, num_epochs: int = 20):
        self.dataset_size = dataset_size
        self.num_epochs = num_epochs
        self.results_dir = Path("hyperparameter_results_v2")
        self.results_dir.mkdir(exist_ok=True)
        
        # Create timestamped experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.results_dir / f"experiment_{timestamp}"
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Save experiment configuration
        self.config = {
            'dataset_size': dataset_size,
            'num_epochs': num_epochs,
            'timestamp': timestamp,
            'description': 'Enhanced hyperparameter tuning with dropout, weight decay, and loss weights',
            'improvements': [
                'Dropout testing (0.0, 0.1, 0.2)',
                'Weight decay testing (0, 1e-4, 1e-3)',
                'Loss weight balancing (value_weight: 0.86, policy_weight: 0.14)',
                'CSV logging for all metrics',
                'Smart checkpoint retention'
            ]
        }
        with open(self.experiment_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)
    
    def load_dataset(self) -> Tuple[List[str], List[str]]:
        """Load dataset for hyperparameter tuning."""
        print(f"Loading {self.dataset_size} games for hyperparameter tuning...")
        
        # Find available processed shard files
        processed_dir = Path("data/processed")
        shard_files = list(processed_dir.glob("*.pkl.gz")) + list(processed_dir.glob("*.pkl"))
        
        if not shard_files:
            raise FileNotFoundError("No processed shard files found in data/processed/ directory")
        
        print(f"Found {len(shard_files)} processed shard files")
        
        # Take enough shards to get approximately dataset_size games
        num_shards_needed = max(1, self.dataset_size // 1000)
        np.random.shuffle(shard_files)
        shard_files = shard_files[:num_shards_needed]
        print(f"Using {len(shard_files)} shards for {self.dataset_size} games")
        
        # Split into train/validation
        np.random.shuffle(shard_files)
        split_idx = int(0.8 * len(shard_files))
        train_files = [str(f) for f in shard_files[:split_idx]]
        val_files = [str(f) for f in shard_files[split_idx:]]
        
        print(f"Dataset split: {len(train_files)} train shards, {len(val_files)} validation shards")
        return train_files, val_files
    
    def run_experiment(self, experiment_name: str, hyperparams: Dict) -> Dict:
        """Run a single hyperparameter experiment."""
        print(f"\n{'='*60}")
        print(f"Running experiment: {experiment_name}")
        print(f"Hyperparameters: {hyperparams}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Create model with dropout
        dropout_prob = hyperparams.get('dropout_prob', 0.1)
        model = TwoHeadedResNet(dropout_prob=dropout_prob)
        
        # Create experiment-specific checkpoint directory
        save_dir = self.experiment_dir / experiment_name
        save_dir.mkdir(exist_ok=True)
        
        # Load dataset
        train_files, val_files = self.load_dataset()
        
        # Create dataloaders
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
        
        # Create trainer with enhanced features
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=hyperparams.get('learning_rate', 0.001),
            device=None,
            enable_system_analysis=True,
            # Enhanced loss weights
            policy_weight=hyperparams.get('policy_weight', 0.14),
            value_weight=hyperparams.get('value_weight', 0.86),
            # Enhanced optimizer settings
            weight_decay=hyperparams.get('weight_decay', 1e-4)
        )
        
        # Train model
        results = trainer.train(
            num_epochs=self.num_epochs,
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
            'early_stopped': len(results['train_losses']) < self.num_epochs,
            'all_metrics': results
        }
        
        print(f"Results for {experiment_name}:")
        print(f"  Best val loss: {best_val_loss:.6f}")
        print(f"  Final val loss: {final_val_loss:.6f}")
        print(f"  Training time: {training_time:.1f}s")
        print(f"  Epochs trained: {len(results['train_losses'])}")
        
        # Save experiment results
        results_file = save_dir / "experiment_results.json"
        with open(results_file, "w") as f:
            json.dump(experiment_results, f, indent=2, default=str)
        
        return experiment_results
    
    def run_enhanced_search(self):
        """Run enhanced hyperparameter search with new features."""
        print("Starting enhanced hyperparameter tuning...")
        print(f"Dataset size: {self.dataset_size} games")
        print(f"Epochs per experiment: {self.num_epochs}")
        print(f"Results directory: {self.experiment_dir}")
        
        # Define enhanced hyperparameter combinations
        experiments = [
            # Baseline with improved loss weights
            {
                'name': 'baseline_v2',
                'hyperparams': {
                    'learning_rate': 0.001,
                    'batch_size': 64,
                    'dropout_prob': 0.1,
                    'weight_decay': 1e-4,
                    'policy_weight': 0.14,
                    'value_weight': 0.86
                }
            },
            # Dropout variations
            {
                'name': 'no_dropout',
                'hyperparams': {
                    'learning_rate': 0.001,
                    'batch_size': 64,
                    'dropout_prob': 0.0,
                    'weight_decay': 1e-4,
                    'policy_weight': 0.14,
                    'value_weight': 0.86
                }
            },
            {
                'name': 'high_dropout',
                'hyperparams': {
                    'learning_rate': 0.001,
                    'batch_size': 64,
                    'dropout_prob': 0.2,
                    'weight_decay': 1e-4,
                    'policy_weight': 0.14,
                    'value_weight': 0.86
                }
            },
            # Weight decay variations
            {
                'name': 'no_weight_decay',
                'hyperparams': {
                    'learning_rate': 0.001,
                    'batch_size': 64,
                    'dropout_prob': 0.1,
                    'weight_decay': 0.0,
                    'policy_weight': 0.14,
                    'value_weight': 0.86
                }
            },
            {
                'name': 'high_weight_decay',
                'hyperparams': {
                    'learning_rate': 0.001,
                    'batch_size': 64,
                    'dropout_prob': 0.1,
                    'weight_decay': 1e-3,
                    'policy_weight': 0.14,
                    'value_weight': 0.86
                }
            },
            # Learning rate variations with improved settings
            {
                'name': 'higher_lr_v2',
                'hyperparams': {
                    'learning_rate': 0.003,
                    'batch_size': 64,
                    'dropout_prob': 0.1,
                    'weight_decay': 1e-4,
                    'policy_weight': 0.14,
                    'value_weight': 0.86
                }
            },
            {
                'name': 'lower_lr_v2',
                'hyperparams': {
                    'learning_rate': 0.0003,
                    'batch_size': 64,
                    'dropout_prob': 0.1,
                    'weight_decay': 1e-4,
                    'policy_weight': 0.14,
                    'value_weight': 0.86
                }
            },
            # Batch size variations with improved settings
            {
                'name': 'larger_batch_v2',
                'hyperparams': {
                    'learning_rate': 0.001,
                    'batch_size': 128,
                    'dropout_prob': 0.1,
                    'weight_decay': 1e-4,
                    'policy_weight': 0.14,
                    'value_weight': 0.86
                }
            },
            {
                'name': 'smaller_batch_v2',
                'hyperparams': {
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'dropout_prob': 0.1,
                    'weight_decay': 1e-4,
                    'policy_weight': 0.14,
                    'value_weight': 0.86
                }
            }
        ]
        
        all_results = []
        
        # Run experiments
        for experiment in experiments:
            try:
                results = self.run_experiment(
                    experiment_name=experiment['name'],
                    hyperparams=experiment['hyperparams']
                )
                all_results.append(results)
                
                # Save intermediate results
                with open(self.experiment_dir / "intermediate_results.json", "w") as f:
                    json.dump(all_results, f, indent=2, default=str)
                    
            except Exception as e:
                print(f"Error in experiment {experiment['name']}: {e}")
                continue
        
        # Analyze results
        self.analyze_results(all_results)
        
        return all_results
    
    def analyze_results(self, results: List[Dict]):
        """Analyze and summarize results."""
        print(f"\n{'='*80}")
        print("ENHANCED HYPERPARAMETER TUNING RESULTS SUMMARY")
        print(f"{'='*80}")
        
        # Sort by best validation loss
        results.sort(key=lambda x: x['best_val_loss'])
        
        print(f"\nRanked by best validation loss:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['experiment_name']}: {result['best_val_loss']:.6f}")
            print(f"   Hyperparams: {result['hyperparameters']}")
            print(f"   Final val loss: {result['final_val_loss']:.6f}")
            print(f"   Training time: {result['training_time']:.1f}s")
            print(f"   Epochs trained: {result['epochs_trained']}")
            print()
        
        # Find best configuration
        if results:
            best_result = results[0]
            print(f"BEST CONFIGURATION:")
            print(f"  Experiment: {best_result['experiment_name']}")
            print(f"  Hyperparameters: {best_result['hyperparameters']}")
            print(f"  Best val loss: {best_result['best_val_loss']:.6f}")
            print(f"  Final val loss: {best_result['final_val_loss']:.6f}")
        
        # Save final results
        final_results = {
            'experiment_config': self.config,
            'results': results,
            'best_configuration': results[0] if results else None,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        results_file = self.experiment_dir / "final_results.json"
        with open(results_file, "w") as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # Create markdown summary
        self.create_markdown_summary(results)
        
        print(f"\nResults saved to: {self.experiment_dir}")
        print(f"Final results: {results_file}")
    
    def create_markdown_summary(self, results: List[Dict]):
        """Create a markdown summary of results."""
        summary_file = self.experiment_dir / "results_summary.md"
        
        with open(summary_file, "w") as f:
            f.write("# Enhanced Hyperparameter Tuning Results Summary\n\n")
            f.write(f"**Dataset Size:** {self.dataset_size} games\n")
            f.write(f"**Epochs per Experiment:** {self.num_epochs}\n")
            f.write(f"**Timestamp:** {self.config['timestamp']}\n\n")
            
            f.write("## Improvements Tested\n\n")
            for improvement in self.config['improvements']:
                f.write(f"- {improvement}\n")
            f.write("\n")
            
            f.write("## Results Table\n\n")
            f.write("| Experiment | LR | Batch | Dropout | Weight Decay | Best Val Loss | Final Val Loss |\n")
            f.write("|------------|----|-------|---------|--------------|---------------|---------------|\n")
            
            for result in results:
                name = result['experiment_name']
                lr = result['hyperparameters']['learning_rate']
                bs = result['hyperparameters']['batch_size']
                dropout = result['hyperparameters']['dropout_prob']
                wd = result['hyperparameters']['weight_decay']
                best_val = result['best_val_loss']
                final_val = result['final_val_loss']
                
                f.write(f"| {name} | {lr} | {bs} | {dropout} | {wd} | {best_val:.6f} | {final_val:.6f} |\n")
            
            if results:
                best = results[0]
                f.write(f"\n## Best Configuration\n\n")
                f.write(f"- **Experiment:** {best['experiment_name']}\n")
                f.write(f"- **Learning Rate:** {best['hyperparameters']['learning_rate']}\n")
                f.write(f"- **Batch Size:** {best['hyperparameters']['batch_size']}\n")
                f.write(f"- **Dropout:** {best['hyperparameters']['dropout_prob']}\n")
                f.write(f"- **Weight Decay:** {best['hyperparameters']['weight_decay']}\n")
                f.write(f"- **Best Validation Loss:** {best['best_val_loss']:.6f}\n")
                f.write(f"- **Final Validation Loss:** {best['final_val_loss']:.6f}\n")


def main():
    """Run enhanced hyperparameter tuning."""
    tuner = EnhancedHyperparameterTuner(dataset_size=10000, num_epochs=20)
    tuner.run_enhanced_search()


if __name__ == "__main__":
    main()
