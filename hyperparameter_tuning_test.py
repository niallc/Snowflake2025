#!/usr/bin/env python3
"""
Minimal hyperparameter tuning test.
Tests just 2 experiments with 1 shard each and 2 epochs.
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


class MinimalHyperparameterTuner:
    """Minimal hyperparameter tuning for testing."""
    
    def __init__(self, dataset_size: int = 1000, num_epochs: int = 2):
        self.dataset_size = dataset_size
        self.num_epochs = num_epochs
        self.results_dir = Path("hyperparameter_test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Create timestamped experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.results_dir / f"test_experiment_{timestamp}"
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Save experiment configuration
        self.config = {
            'dataset_size': dataset_size,
            'num_epochs': num_epochs,
            'timestamp': timestamp,
            'description': 'Minimal hyperparameter test with 2 experiments',
            'test_mode': True
        }
        with open(self.experiment_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)
    
    def load_dataset(self) -> Tuple[List[str], List[str]]:
        """Load minimal dataset for testing."""
        print(f"Loading {self.dataset_size} games for testing...")
        
        # Find available processed shard files
        processed_dir = Path("data/processed")
        shard_files = list(processed_dir.glob("*.pkl.gz")) + list(processed_dir.glob("*.pkl"))
        
        if not shard_files:
            raise FileNotFoundError("No processed shard files found in data/processed/ directory")
        
        print(f"Found {len(shard_files)} processed shard files")
        
        # Use just 1 shard for testing
        shard_files = shard_files[:1]
        print(f"Using {len(shard_files)} shard for testing")
        
        # Split into train/validation (same shard for both in test)
        train_files = [str(f) for f in shard_files]
        val_files = [str(f) for f in shard_files]
        
        print(f"Dataset split: {len(train_files)} train shard, {len(val_files)} validation shard")
        return train_files, val_files
    
    def run_experiment(self, experiment_name: str, hyperparams: Dict) -> Dict:
        """Run a single hyperparameter experiment."""
        print(f"\n{'='*50}")
        print(f"Running test experiment: {experiment_name}")
        print(f"Hyperparameters: {hyperparams}")
        print(f"{'='*50}")
        
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
            enable_system_analysis=False,  # Disable for test
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
    
    def run_minimal_test(self):
        """Run minimal hyperparameter test."""
        print("Starting minimal hyperparameter test...")
        print(f"Dataset size: {self.dataset_size} games")
        print(f"Epochs per experiment: {self.num_epochs}")
        print(f"Results directory: {self.experiment_dir}")
        
        # Define just 2 test experiments
        experiments = [
            {
                'name': 'test_baseline',
                'hyperparams': {
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'dropout_prob': 0.1,
                    'weight_decay': 1e-4,
                    'policy_weight': 0.14,
                    'value_weight': 0.86
                }
            },
            {
                'name': 'test_no_dropout',
                'hyperparams': {
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'dropout_prob': 0.0,
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
                import traceback
                traceback.print_exc()
                continue
        
        # Analyze results
        self.analyze_results(all_results)
        
        return all_results
    
    def analyze_results(self, results: List[Dict]):
        """Analyze and summarize results."""
        print(f"\n{'='*60}")
        print("MINIMAL TEST RESULTS SUMMARY")
        print(f"{'='*60}")
        
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
            print(f"BEST TEST CONFIGURATION:")
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
        
        print(f"\nTest results saved to: {self.experiment_dir}")
        print(f"Final results: {results_file}")
        print("\n✅ If this test passes, you're ready for full hyperparameter tuning!")


def main():
    """Run minimal hyperparameter test."""
    tuner = MinimalHyperparameterTuner(dataset_size=1000, num_epochs=2)
    tuner.run_minimal_test()


if __name__ == "__main__":
    main()
