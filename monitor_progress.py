#!/usr/bin/env python3
"""
Monitor hyperparameter tuning progress with detailed results extraction.
Updated for large-scale experiment (50k samples, 10 epochs).
"""

import json
import torch
from pathlib import Path
import time
from datetime import datetime
import csv

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
    results_file = exp_dir / "experiment_results.json"
    
    if not checkpoint_files and not best_model_path.exists():
        return {'status': 'no_checkpoints'}
    
    # Get experiment config
    config_file = exp_dir / "config.json"
    hyperparams = {}
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
            hyperparams = config.get('hyperparameters', {})
    
    # Get results from JSON file if available
    experiment_results = {}
    if results_file.exists():
        with open(results_file) as f:
            experiment_results = json.load(f)
    
    # Get best model info
    best_info = {}
    if best_model_path.exists():
        best_info = extract_checkpoint_info(best_model_path)
    
    # Get latest checkpoint info
    latest_info = {}
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
        latest_info = extract_checkpoint_info(latest_checkpoint)
    
    # Determine status based on expected epochs
    expected_epochs = 10  # For the large-scale experiment
    current_epochs = len(checkpoint_files)
    status = 'completed' if current_epochs >= expected_epochs else 'in_progress'
    
    return {
        'name': exp_dir.name,
        'hyperparams': hyperparams,
        'best_model': best_info,
        'latest_checkpoint': latest_info,
        'checkpoint_count': current_epochs,
        'expected_epochs': expected_epochs,
        'status': status,
        'experiment_results': experiment_results
    }

def find_currently_running_experiment(results_dir: Path) -> str:
    """Find which experiment is currently running by checking file modification times."""
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

def calculate_overall_progress(results_dir: Path) -> dict:
    """Calculate overall progress across all experiments."""
    experiment_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name != "__pycache__"]
    
    total_experiments = len(experiment_dirs)
    completed_experiments = 0
    total_epochs_completed = 0
    total_epochs_expected = total_experiments * 10  # 10 epochs per experiment
    
    for exp_dir in experiment_dirs:
        checkpoint_files = list(exp_dir.glob("checkpoint_epoch_*.pt"))
        epochs_completed = len(checkpoint_files)
        total_epochs_completed += epochs_completed
        
        if epochs_completed >= 10:  # Expected epochs
            completed_experiments += 1
    
    return {
        'total_experiments': total_experiments,
        'completed_experiments': completed_experiments,
        'total_epochs_completed': total_epochs_completed,
        'total_epochs_expected': total_epochs_expected,
        'progress_percentage': (total_epochs_completed / total_epochs_expected * 100) if total_epochs_expected > 0 else 0
    }

def monitor_progress():
    """Monitor hyperparameter tuning progress with detailed results."""
    # Check for large-scale experiment first
    results_dir = Path("checkpoints/scaled_tuning_50k")
    
    if not results_dir.exists():
        # Fall back to original experiment
        results_dir = Path("checkpoints/scaled_tuning")
        if not results_dir.exists():
            print("No experiment results directory found.")
            return
    
    print(f"Monitoring: {results_dir}")
    
    # Check config
    config_file = results_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        print(f"Target examples: {config.get('target_examples', 'N/A')}")
        print(f"Epochs per experiment: {config.get('num_epochs', 'N/A')}")
        print(f"Number of experiments: {config.get('num_experiments', 'N/A')}")
    
    # Calculate overall progress
    progress = calculate_overall_progress(results_dir)
    print(f"\nOverall Progress:")
    print(f"  Experiments completed: {progress['completed_experiments']}/{progress['total_experiments']}")
    print(f"  Epochs completed: {progress['total_epochs_completed']}/{progress['total_epochs_expected']}")
    print(f"  Progress: {progress['progress_percentage']:.1f}%")
    
    # Find currently running experiment
    currently_running = find_currently_running_experiment(results_dir)
    print(f"Currently running: {currently_running}")
    
    # Get results for all experiments
    experiment_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name != "__pycache__"]
    results = []
    
    for exp_dir in experiment_dirs:
        result = get_experiment_results(exp_dir)
        results.append(result)
    
    # Sort by best validation loss
    completed_results = [r for r in results if r['status'] == 'completed' and 'best_model' in r and 'best_val_loss' in r['best_model']]
    completed_results.sort(key=lambda x: x['best_model']['best_val_loss'])
    
    print(f"\nCompleted experiments ({len(completed_results)}):")
    print("=" * 100)
    print(f"{'Rank':<4} {'Experiment':<20} {'Best Loss':<12} {'Best Epoch':<12} {'LR':<8} {'Batch':<8} {'Dropout':<8}")
    print("-" * 100)
    
    for i, result in enumerate(completed_results, 1):
        name = result['name']
        best_loss = result['best_model']['best_val_loss']
        best_epoch = result['best_model']['epoch']
        lr = result['hyperparams'].get('learning_rate', 'N/A')
        batch_size = result['hyperparams'].get('batch_size', 'N/A')
        dropout = result['hyperparams'].get('dropout_prob', 'N/A')
        
        print(f"{i:<4} {name:<20} {best_loss:<12.6f} {best_epoch:<12} {lr:<8} {batch_size:<8} {dropout:<8}")
    
    # Show in-progress experiments
    in_progress = [r for r in results if r['status'] == 'in_progress']
    if in_progress:
        print(f"\nIn progress experiments ({len(in_progress)}):")
        print("=" * 100)
        for result in in_progress:
            name = result['name']
            checkpoint_count = result['checkpoint_count']
            expected_epochs = result['expected_epochs']
            latest_epoch = result['latest_checkpoint'].get('epoch', 0) if 'latest_checkpoint' in result else 0
            current_loss = result['latest_checkpoint'].get('val_loss', 'N/A') if 'latest_checkpoint' in result else 'N/A'
            progress_pct = (checkpoint_count / expected_epochs * 100) if expected_epochs > 0 else 0
            
            print(f"  {name}: {checkpoint_count}/{expected_epochs} epochs ({progress_pct:.1f}%), "
                  f"current loss: {current_loss}")
    
    # Show summary statistics
    if completed_results:
        best_result = completed_results[0]
        print(f"\nBest configuration so far:")
        print(f"  Experiment: {best_result['name']}")
        print(f"  Best validation loss: {best_result['best_model']['best_val_loss']:.6f}")
        print(f"  Best epoch: {best_result['best_model']['epoch']}")
        print(f"  Hyperparameters: {best_result['hyperparams']}")
    
    # Check for CSV files
    csv_files = list(results_dir.glob("**/training_metrics.csv"))
    if csv_files:
        print(f"\nCSV logging files found: {len(csv_files)}")
        for csv_file in csv_files:
            print(f"  {csv_file.relative_to(results_dir)}")

if __name__ == "__main__":
    while True:
        print("\n" + "="*100)
        print(f"Progress Update: {datetime.now().strftime('%H:%M:%S')}")
        print("="*100)
        monitor_progress()
        print("\nPress Ctrl+C to stop monitoring...")
        time.sleep(30)  # Update every 30 seconds 