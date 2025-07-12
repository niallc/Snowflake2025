#!/usr/bin/env python3
"""
Monitor enhanced hyperparameter tuning progress.
"""

import json
import torch
from pathlib import Path
import time
from datetime import datetime

def extract_checkpoint_info(checkpoint_path: Path) -> dict:
    """Extract information from a checkpoint file."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract basic info
        epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        train_metrics = checkpoint.get('train_metrics', {})
        val_metrics = checkpoint.get('val_metrics', {})
        
        return {
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'train_loss': train_metrics.get('total_loss', float('inf')),
            'val_loss': val_metrics.get('total_loss', float('inf')),
            'policy_loss': train_metrics.get('policy_loss', float('inf')),
            'value_loss': train_metrics.get('value_loss', float('inf'))
        }
    except Exception as e:
        return {'error': str(e)}

def get_experiment_results(exp_dir: Path) -> dict:
    """Get results for a single experiment."""
    best_model_path = exp_dir / "best_model.pt"
    checkpoint_files = list(exp_dir.glob("checkpoint_epoch_*.pt"))
    
    if not checkpoint_files and not best_model_path.exists():
        return {'status': 'no_checkpoints'}
    
    # Get best model info
    best_info = {}
    if best_model_path.exists():
        best_info = extract_checkpoint_info(best_model_path)
    
    # Get latest checkpoint info
    latest_info = {}
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
        latest_info = extract_checkpoint_info(latest_checkpoint)
    
    # Parse experiment name for hyperparameters
    exp_name = exp_dir.name
    hyperparams = {}
    
    # Enhanced hyperparameter parsing
    if "baseline_v2" in exp_name:
        hyperparams = {"learning_rate": 0.001, "batch_size": 64, "dropout_prob": 0.1, "weight_decay": 1e-4}
    elif "no_dropout" in exp_name:
        hyperparams = {"learning_rate": 0.001, "batch_size": 64, "dropout_prob": 0.0, "weight_decay": 1e-4}
    elif "high_dropout" in exp_name:
        hyperparams = {"learning_rate": 0.001, "batch_size": 64, "dropout_prob": 0.2, "weight_decay": 1e-4}
    elif "no_weight_decay" in exp_name:
        hyperparams = {"learning_rate": 0.001, "batch_size": 64, "dropout_prob": 0.1, "weight_decay": 0.0}
    elif "high_weight_decay" in exp_name:
        hyperparams = {"learning_rate": 0.001, "batch_size": 64, "dropout_prob": 0.1, "weight_decay": 1e-3}
    elif "higher_lr_v2" in exp_name:
        hyperparams = {"learning_rate": 0.003, "batch_size": 64, "dropout_prob": 0.1, "weight_decay": 1e-4}
    elif "lower_lr_v2" in exp_name:
        hyperparams = {"learning_rate": 0.0003, "batch_size": 64, "dropout_prob": 0.1, "weight_decay": 1e-4}
    elif "larger_batch_v2" in exp_name:
        hyperparams = {"learning_rate": 0.001, "batch_size": 128, "dropout_prob": 0.1, "weight_decay": 1e-4}
    elif "smaller_batch_v2" in exp_name:
        hyperparams = {"learning_rate": 0.001, "batch_size": 32, "dropout_prob": 0.1, "weight_decay": 1e-4}
    
    return {
        'name': exp_name,
        'hyperparams': hyperparams,
        'best_model': best_info,
        'latest_checkpoint': latest_info,
        'checkpoint_count': len(checkpoint_files),
        'status': 'completed' if len(checkpoint_files) >= 20 else 'in_progress'
    }

def find_currently_running_experiment(results_dir: Path) -> str:
    """Find which experiment is currently running."""
    experiment_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name != "__pycache__"]
    
    if not experiment_dirs:
        return "None"
    
    # Check for recently modified checkpoint files
    current_time = time.time()
    most_recent = None
    most_recent_time = 0
    
    for exp_dir in experiment_dirs:
        checkpoint_files = list(exp_dir.glob("checkpoint_epoch_*.pt"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            mtime = latest_checkpoint.stat().st_mtime
            if mtime > most_recent_time:
                most_recent_time = mtime
                most_recent = exp_dir.name
    
    # If most recent was within last 5 minutes, it's probably still running
    if most_recent and (current_time - most_recent_time) < 300:
        return most_recent
    
    return "None"

def monitor_progress():
    """Monitor enhanced hyperparameter tuning progress."""
    results_dir = Path("hyperparameter_results_v2")
    
    if not results_dir.exists():
        print("No hyperparameter_results_v2 directory found.")
        return
    
    # Find the latest experiment
    experiments = list(results_dir.glob("experiment_*"))
    if not experiments:
        print("No experiments found.")
        return
    
    latest_experiment = max(experiments, key=lambda x: x.stat().st_mtime)
    print(f"Monitoring: {latest_experiment.name}")
    
    # Check config
    config_file = latest_experiment / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        print(f"Dataset size: {config['dataset_size']} games")
        print(f"Epochs per experiment: {config['num_epochs']}")
        print(f"Description: {config['description']}")
    
    # Find currently running experiment
    currently_running = find_currently_running_experiment(latest_experiment)
    print(f"Currently running: {currently_running}")
    
    # Get results for all experiments
    experiment_dirs = [d for d in latest_experiment.iterdir() if d.is_dir() and d.name != "__pycache__"]
    results = []
    
    for exp_dir in experiment_dirs:
        result = get_experiment_results(exp_dir)
        results.append(result)
    
    # Sort by best validation loss
    completed_results = [r for r in results if r['status'] == 'completed' and 'best_model' in r and 'best_val_loss' in r['best_model']]
    completed_results.sort(key=lambda x: x['best_model']['best_val_loss'])
    
    print(f"\nCompleted experiments ({len(completed_results)}):")
    print("=" * 100)
    print(f"{'Rank':<4} {'Experiment':<25} {'Best Loss':<12} {'Dropout':<8} {'Weight Decay':<12} {'LR':<8} {'Batch':<8}")
    print("-" * 100)
    
    for i, result in enumerate(completed_results, 1):
        name = result['name']
        best_loss = result['best_model']['best_val_loss']
        dropout = result['hyperparams'].get('dropout_prob', 'N/A')
        weight_decay = result['hyperparams'].get('weight_decay', 'N/A')
        lr = result['hyperparams']['learning_rate']
        batch_size = result['hyperparams']['batch_size']
        
        print(f"{i:<4} {name:<25} {best_loss:<12.6f} {dropout:<8} {weight_decay:<12} {lr:<8} {batch_size:<8}")
    
    # Show in-progress experiments
    in_progress = [r for r in results if r['status'] == 'in_progress']
    if in_progress:
        print(f"\nIn progress experiments ({len(in_progress)}):")
        print("=" * 100)
        for result in in_progress:
            name = result['name']
            checkpoint_count = result['checkpoint_count']
            latest_epoch = result['latest_checkpoint'].get('epoch', 0) if 'latest_checkpoint' in result else 0
            current_loss = result['latest_checkpoint'].get('val_loss', 'N/A') if 'latest_checkpoint' in result else 'N/A'
            
            print(f"  {name}: {checkpoint_count} checkpoints, epoch {latest_epoch}, loss {current_loss}")
    
    # Show summary statistics
    if completed_results:
        best_result = completed_results[0]
        print(f"\nBest configuration so far:")
        print(f"  Experiment: {best_result['name']}")
        print(f"  Best validation loss: {best_result['best_model']['best_val_loss']:.6f}")
        print(f"  Hyperparameters: {best_result['hyperparams']}")

if __name__ == "__main__":
    while True:
        print("\n" + "="*100)
        print(f"Enhanced Progress Update: {datetime.now().strftime('%H:%M:%S')}")
        print("="*100)
        monitor_progress()
        print("\nPress Ctrl+C to stop monitoring...")
        time.sleep(30)  # Update every 30 seconds
