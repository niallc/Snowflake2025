#!/usr/bin/env python3
"""
Analyze hyperparameter tuning results from exploratory runs.

This script examines the checkpoints and training history to extract
hyperparameter information and loss values for comparison.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

def extract_hyperparameters_from_checkpoint(checkpoint_path: Path) -> Optional[Dict]:
    """Extract hyperparameters from a checkpoint file."""
    try:
        # Try with weights_only=False for older checkpoints
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Try to extract hyperparameters from the checkpoint
        hyperparams = {}
        
        # Check if optimizer state contains learning rate info
        if 'optimizer_state_dict' in checkpoint:
            optimizer_state = checkpoint['optimizer_state_dict']
            if 'param_groups' in optimizer_state:
                for group in optimizer_state['param_groups']:
                    if 'lr' in group:
                        hyperparams['learning_rate'] = group['lr']
        
        # Check for batch size info (might be in training history)
        if 'train_metrics' in checkpoint:
            # Could infer batch size from training metrics
            pass
        
        return hyperparams if hyperparams else None
        
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None

def analyze_exploratory_results():
    """Analyze results from exploratory hyperparameter tuning."""
    print("Analyzing exploratory hyperparameter tuning results...")
    
    exploratory_dir = Path("checkpoints/exploratory")
    if not exploratory_dir.exists():
        print("No exploratory results found!")
        return
    
    # Define the expected hyperparameters for each experiment
    expected_hyperparams = {
        'baseline': {'learning_rate': 0.001, 'batch_size': 64},
        'higher_lr': {'learning_rate': 0.003, 'batch_size': 64},
        'lower_lr': {'learning_rate': 0.0003, 'batch_size': 64},
        'larger_batch': {'learning_rate': 0.001, 'batch_size': 128},
        'smaller_batch': {'learning_rate': 0.001, 'batch_size': 32},
        'high_lr_large_batch': {'learning_rate': 0.003, 'batch_size': 128},
        'low_lr_small_batch': {'learning_rate': 0.0003, 'batch_size': 32}
    }
    
    results = []
    
    # Analyze each experiment directory
    for exp_name in expected_hyperparams.keys():
        exp_dir = exploratory_dir / exp_name
        if not exp_dir.exists():
            print(f"Warning: Experiment directory {exp_name} not found")
            continue
        
        print(f"\nAnalyzing experiment: {exp_name}")
        print(f"Expected hyperparameters: {expected_hyperparams[exp_name]}")
        
        # Find best model checkpoint
        best_model_path = exp_dir / "best_model.pt"
        if not best_model_path.exists():
            print(f"  No best model found for {exp_name}")
            continue
        
        # Extract hyperparameters from checkpoint
        checkpoint_hyperparams = extract_hyperparameters_from_checkpoint(best_model_path)
        if checkpoint_hyperparams:
            print(f"  Extracted hyperparameters: {checkpoint_hyperparams}")
        
        # Load the best model to get loss information
        try:
            checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
            
            # Extract loss information
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            train_metrics = checkpoint.get('train_metrics', {})
            val_metrics = checkpoint.get('val_metrics', {})
            
            # Get final losses
            final_train_loss = train_metrics.get('total_loss', float('inf')) if train_metrics else float('inf')
            final_val_loss = val_metrics.get('total_loss', float('inf')) if val_metrics else float('inf')
            
            # Count checkpoints to estimate training progress
            checkpoint_files = list(exp_dir.glob("checkpoint_epoch_*.pt"))
            epochs_trained = len(checkpoint_files)
            
            result = {
                'experiment_name': exp_name,
                'expected_hyperparams': expected_hyperparams[exp_name],
                'extracted_hyperparams': checkpoint_hyperparams,
                'best_val_loss': best_val_loss,
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss,
                'epochs_trained': epochs_trained,
                'checkpoint_files': len(checkpoint_files)
            }
            
            results.append(result)
            
            print(f"  Best val loss: {best_val_loss:.6f}")
            print(f"  Final train loss: {final_train_loss:.6f}")
            print(f"  Final val loss: {final_val_loss:.6f}")
            print(f"  Epochs trained: {epochs_trained}")
            
        except Exception as e:
            print(f"  Error analyzing {exp_name}: {e}")
    
    # Analyze main training history
    main_history_path = Path("checkpoints/training_history.json")
    if main_history_path.exists():
        print(f"\nAnalyzing main training history...")
        try:
            with open(main_history_path, 'r') as f:
                history = json.load(f)
            
            # Extract final losses
            if history:
                final_epoch = history[-1]
                final_train_loss = final_epoch['train']['total_loss']
                final_val_loss = final_epoch['val']['total_loss']
                
                # Find best validation loss
                val_losses = [epoch['val']['total_loss'] for epoch in history if 'val' in epoch]
                best_val_loss = min(val_losses) if val_losses else float('inf')
                
                main_result = {
                    'experiment_name': 'main_training',
                    'expected_hyperparams': {'learning_rate': 0.001, 'batch_size': 32},  # Default from config
                    'extracted_hyperparams': None,
                    'best_val_loss': best_val_loss,
                    'final_train_loss': final_train_loss,
                    'final_val_loss': final_val_loss,
                    'epochs_trained': len(history),
                    'checkpoint_files': 0
                }
                
                results.append(main_result)
                
                print(f"  Best val loss: {best_val_loss:.6f}")
                print(f"  Final train loss: {final_train_loss:.6f}")
                print(f"  Final val loss: {final_val_loss:.6f}")
                print(f"  Epochs trained: {len(history)}")
                
        except Exception as e:
            print(f"  Error analyzing main history: {e}")
    
    # Sort results by best validation loss
    results.sort(key=lambda x: x['best_val_loss'])
    
    # Print summary
    print(f"\n{'='*80}")
    print("HYPERPARAMETER TUNING RESULTS SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nRanked by best validation loss:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['experiment_name']}: {result['best_val_loss']:.6f}")
        print(f"   Hyperparams: {result['expected_hyperparams']}")
        print(f"   Final val loss: {result['final_val_loss']:.6f}")
        print(f"   Epochs trained: {result['epochs_trained']}")
        print()
    
    # Find best configuration
    if results:
        best_result = results[0]
        print(f"BEST CONFIGURATION:")
        print(f"  Experiment: {best_result['experiment_name']}")
        print(f"  Hyperparameters: {best_result['expected_hyperparams']}")
        print(f"  Best val loss: {best_result['best_val_loss']:.6f}")
        print(f"  Final val loss: {best_result['final_val_loss']:.6f}")
    
    # Save results
    results_file = Path("exploratory_results/analysis_results.json")
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'analysis_timestamp': str(Path().cwd()),
            'results': results,
            'best_configuration': results[0] if results else None
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    analyze_exploratory_results() 