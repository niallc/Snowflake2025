#!/usr/bin/env python3
"""
GPU-based hyperparameter tuning for Hex AI - Large Scale Version.
Uses processed .pkl.gz files with GPU acceleration and mixed precision.
Updated to work with the new data format and training utilities.
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
parser = argparse.ArgumentParser(description='Large-scale hyperparameter tuning for Hex AI')
parser.add_argument('--verbose', '-v', type=int, default=2, 
                   help='Verbose level: 0=critical only, 1=important, 2=detailed (default), 3=debug, 4=very debug')
parser.add_argument('--auto-analyze', action='store_true',
                   help='Automatically run analysis when tuning completes')
args = parser.parse_args()

# Set up logging based on verbose level
if args.verbose == 0:
    logging.basicConfig(level=logging.CRITICAL)
elif args.verbose == 1:
    logging.basicConfig(level=logging.WARNING)
elif args.verbose == 2:
    logging.basicConfig(level=logging.INFO)
elif args.verbose == 3:
    logging.basicConfig(level=logging.DEBUG)
else:  # args.verbose >= 4
    logging.basicConfig(level=logging.DEBUG)

# Update the global verbose level
from hex_ai.config import VERBOSE_LEVEL
import hex_ai.config
hex_ai.config.VERBOSE_LEVEL = args.verbose

# Fix multiprocessing on macOS
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

from hex_ai.models import TwoHeadedResNet
from hex_ai.training_utils import (
    run_hyperparameter_tuning,
    discover_processed_files,
    estimate_dataset_size,
    create_experiment_config
)

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

# Large-scale hyperparameter tuning config - focusing on balanced loss variants
NUM_EPOCHS = 10
BATCH_SIZE = 64  # Larger batch size for GPU efficiency
TARGET_EXAMPLES = 500000  # 500k positions for comprehensive training

# Experiment naming
from datetime import datetime
EXPERIMENT_NAME = f"hex_ai_hyperparam_tuning_v3_500k_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Define experiments - focusing on balanced loss variants based on previous results
experiments = [
    {
        'experiment_name': 'balanced_weights',
        'hyperparameters': {
            'learning_rate': 0.001,
            'batch_size': BATCH_SIZE,
            'dropout_prob': 0.1,
            'weight_decay': 1e-4,
            'policy_weight': 0.5,
            'value_weight': 0.5
        }
    },
    {
        'experiment_name': 'balanced_high_weight_decay',
        'hyperparameters': {
            'learning_rate': 0.001,
            'batch_size': BATCH_SIZE,
            'dropout_prob': 0.1,
            'weight_decay': 1e-3,
            'policy_weight': 0.5,
            'value_weight': 0.5
        }
    },
    {
        'experiment_name': 'policy_heavy',
        'hyperparameters': {
            'learning_rate': 0.001,
            'batch_size': BATCH_SIZE,
            'dropout_prob': 0.1,
            'weight_decay': 1e-4,
            'policy_weight': 0.7,
            'value_weight': 0.3
        }
    },
    {
        'experiment_name': 'policy_intermediate',
        'hyperparameters': {
            'learning_rate': 0.001,
            'batch_size': BATCH_SIZE,
            'dropout_prob': 0.1,
            'weight_decay': 1e-4,
            'policy_weight': 0.33,
            'value_weight': 0.67
        }
    },
    {
        'experiment_name': 'balanced_no_dropout',
        'hyperparameters': {
            'learning_rate': 0.001,
            'batch_size': BATCH_SIZE,
            'dropout_prob': 0.0,
            'weight_decay': 1e-4,
            'policy_weight': 0.5,
            'value_weight': 0.5
        }
    },
    {
        'experiment_name': 'balanced_high_lr',
        'hyperparameters': {
            'learning_rate': 0.003,
            'batch_size': BATCH_SIZE,
            'dropout_prob': 0.1,
            'weight_decay': 1e-4,
            'policy_weight': 0.5,
            'value_weight': 0.5
        }
    }
]

# Create results directory
results_dir = Path("checkpoints") / EXPERIMENT_NAME
results_dir.mkdir(parents=True, exist_ok=True)

# Save configuration
config = {
    'experiment_name': EXPERIMENT_NAME,
    'description': 'Large-scale hyperparameter tuning with 500k samples per experiment, focusing on balanced loss variants',
    'num_epochs': NUM_EPOCHS,
    'batch_size': BATCH_SIZE,
    'target_examples': TARGET_EXAMPLES,
    'device': str(device),
    'num_experiments': len(experiments),
    'timestamp': datetime.now().isoformat(),
    'parameters': {
        'early_stopping_patience': 3,
        'train_ratio': 0.8,
        'random_seed': 42
    }
}
with open(results_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"\n{'='*60}")
print(f"LARGE-SCALE HYPERPARAMETER TUNING (500K)")
print(f"Experiment Name: {EXPERIMENT_NAME}")
print(f"Device: {device}")
print(f"Epochs per experiment: {NUM_EPOCHS}")
print(f"Target examples: {TARGET_EXAMPLES}")
print(f"Number of experiments: {len(experiments)}")
print(f"Results directory: {results_dir}")
print(f"Verbose level: {args.verbose} (0=critical, 1=important, 2=detailed, 3=debug, 4=very debug)")
print(f"{'='*60}")

# Discover and analyze data
print("\nDiscovering processed data files...")
data_files = discover_processed_files("data/processed")
total_examples = estimate_dataset_size(data_files, max_files=10)  # Sample more files for better estimate
print(f"Found {len(data_files)} data files with approximately {total_examples:,} training examples")

print(f"Using {TARGET_EXAMPLES} examples for training")
print("(This provides comprehensive training for meaningful results)")

# Run large-scale hyperparameter tuning
start_time = time.time()
overall_results = run_hyperparameter_tuning(
    experiments=experiments,
    data_dir="data/processed",
    results_dir=str(results_dir),
    train_ratio=0.8,
    num_epochs=NUM_EPOCHS,
    early_stopping_patience=3,  # Early stopping after 3 epochs without improvement
    random_seed=42,
    max_examples_per_split=TARGET_EXAMPLES,
    experiment_name=EXPERIMENT_NAME  # Pass experiment name through
)

total_time = time.time() - start_time

print(f"\n{'='*60}")
print("LARGE-SCALE TUNING (500K) COMPLETE")
print(f"{'='*60}")
print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
print(f"Successful experiments: {overall_results['successful_experiments']}/{overall_results['num_experiments']}")

# Find best experiment
if overall_results['experiments']:
    best_exp = min(overall_results['experiments'], key=lambda x: x['best_val_loss'])
    print(f"\nBest experiment: {best_exp['experiment_name']}")
    print(f"Best validation loss: {best_exp['best_val_loss']:.6f}")
    print(f"Hyperparameters: {best_exp['hyperparameters']}")
    
    # Show all results sorted by validation loss
    print(f"\nAll experiments ranked by validation loss:")
    sorted_experiments = sorted(overall_results['experiments'], key=lambda x: x['best_val_loss'])
    for i, exp in enumerate(sorted_experiments):
        print(f"{i+1}. {exp['experiment_name']}: {exp['best_val_loss']:.6f}")
else:
    print("\nNo successful experiments!")

print(f"\nAll results saved to: {results_dir}")
print("Run 'python analyze_tuning_results.py' to analyze the results.")
print("\nPROGRESS MONITORING:")
print(f"- Check progress: tail -f {results_dir}/overall_results.json")
print(f"- Monitor system: htop or Activity Monitor")
print(f"- Check GPU usage: nvidia-smi (if CUDA) or system_profiler SPDisplaysDataType (if MPS)")

# Auto-analysis if requested
if args.auto_analyze:
    print(f"\n{'='*60}")
    print("AUTO-ANALYSIS ENABLED")
    print(f"{'='*60}")
    print("Running analysis automatically...")
    
    try:
        import subprocess
        import sys
        
        # Run the analysis script
        analysis_cmd = [sys.executable, "analyze_tuning_results.py", str(results_dir)]
        print(f"Running: {' '.join(analysis_cmd)}")
        
        result = subprocess.run(analysis_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Analysis completed successfully!")
            print("Generated plots and summary reports.")
        else:
            print("❌ Analysis failed with errors:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Failed to run auto-analysis: {e}")
        print("You can run analysis manually with:")
        print(f"python analyze_tuning_results.py {results_dir}") 