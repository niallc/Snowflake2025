#!/usr/bin/env python3
"""
Monitor hyperparameter tuning progress.
"""

import json
from pathlib import Path
import time

def monitor_progress():
    """Monitor hyperparameter tuning progress."""
    results_dir = Path("hyperparameter_results")
    
    if not results_dir.exists():
        print("No hyperparameter_results directory found.")
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
    
    # Count checkpoints
    checkpoint_files = list(latest_experiment.rglob("*.pt"))
    print(f"Total checkpoints: {len(checkpoint_files)}")
    
    # List experiment directories
    experiment_dirs = [d for d in latest_experiment.iterdir() if d.is_dir() and d.name != "__pycache__"]
    print(f"Experiments completed: {len(experiment_dirs)}")
    
    for exp_dir in experiment_dirs:
        checkpoints = list(exp_dir.glob("*.pt"))
        print(f"  {exp_dir.name}: {len(checkpoints)} checkpoints")
    
    # Check for intermediate results
    intermediate_file = latest_experiment / "intermediate_results.json"
    if intermediate_file.exists():
        with open(intermediate_file) as f:
            results = json.load(f)
        print(f"Experiments with results: {len(results)}")
        
        if results:
            print("\nCurrent rankings:")
            sorted_results = sorted(results, key=lambda x: x['best_val_loss'])
            for i, result in enumerate(sorted_results, 1):
                print(f"  {i}. {result['experiment_name']}: {result['best_val_loss']:.6f}")

if __name__ == "__main__":
    while True:
        print("\n" + "="*50)
        print(f"Progress Update: {time.strftime('%H:%M:%S')}")
        print("="*50)
        monitor_progress()
        print("\nPress Ctrl+C to stop monitoring...")
        time.sleep(30)  # Update every 30 seconds 