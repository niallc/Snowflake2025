#!/usr/bin/env python3
"""
Parallel hyperparameter tuning for Hex AI.
Runs multiple experiments simultaneously to better utilize system resources.
"""

import os
import json
import time
import subprocess
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional
import torch
import numpy as np
from datetime import datetime
import logging
import argparse
from hex_ai.utils import get_device

# Parse command line arguments
parser = argparse.ArgumentParser(description='Parallel hyperparameter tuning for Hex AI')
parser.add_argument('--verbose', '-v', type=int, default=2, 
                   help='Verbose level: 0=critical only, 1=important, 2=detailed (default), 3=debug, 4=very debug')
parser.add_argument('--max-parallel', type=int, default=3, 
                   help='Maximum number of parallel experiments (default: 3)')
parser.add_argument('--auto-analyze', action='store_true',
                   help='Automatically run analysis when tuning completes')
args = parser.parse_args()

# Set up logging
if args.verbose == 0:
    logging.basicConfig(level=logging.CRITICAL)
elif args.verbose == 1:
    logging.basicConfig(level=logging.WARNING)
elif args.verbose == 2:
    logging.basicConfig(level=logging.INFO)
elif args.verbose == 3:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.DEBUG)

# Fix multiprocessing on macOS
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

from hex_ai.models import TwoHeadedResNet
from hex_ai.training_utils import (
    discover_processed_files,
    estimate_dataset_size,
    create_experiment_config,
    create_train_val_split,
    StreamingProcessedDataset
)
from scripts.run_hyperparameter_experiment import run_hyperparameter_experiment

device = get_device()
print(f"Using device: {device}")

# Large-scale hyperparameter tuning config
NUM_EPOCHS = 10
BATCH_SIZE = 64
TARGET_EXAMPLES = 500000

# Experiment naming
EXPERIMENT_NAME = f"hex_ai_parallel_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Define experiments - split into groups for parallel execution
experiments = [
    # Group 1: Balanced variants
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
    # Group 2: Policy-focused variants
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
    # Group 3: Learning rate variants
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

def run_single_experiment(experiment_config: Dict, results_dir: str, experiment_name: str) -> Dict:
    """
    Run a single experiment in a separate process.
    
    Args:
        experiment_config: Experiment configuration
        results_dir: Base results directory
        experiment_name: Overall experiment name
        
    Returns:
        Dictionary with experiment results
    """
    exp_name = experiment_config['experiment_name']
    exp_results_dir = Path(results_dir) / exp_name
    
    # Create experiment-specific script
    script_content = f'''#!/usr/bin/env python3
"""
Single experiment runner for parallel tuning.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hex_ai.training_utils import run_hyperparameter_experiment, StreamingProcessedDataset
from hex_ai.models import TwoHeadedResNet
from hex_ai.training import Trainer, EarlyStopping
import torch
import json
from pathlib import Path

# Load data
data_files = {discover_processed_files("data/processed")}
train_files, val_files = create_train_val_split(data_files, train_ratio=0.8, random_seed=42)

# Create datasets
train_dataset = StreamingProcessedDataset(train_files, chunk_size=100000)
val_dataset = StreamingProcessedDataset(val_files, chunk_size=100000)

# Run experiment
results = run_hyperparameter_experiment(
    experiment_config={experiment_config},
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    results_dir=Path("{exp_results_dir}"),
    num_epochs={NUM_EPOCHS},
    early_stopping_patience=3,
    experiment_name="{experiment_name}"
)

print(f"Experiment {{exp_name}} completed with best val loss: {{results['best_val_loss']:.6f}}")
'''
    
    # Write temporary script
    script_path = Path(f"temp_experiment_{exp_name}.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    try:
        # Run the experiment
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per experiment
        )
        
        if result.returncode == 0:
            # Load results
            results_file = exp_results_dir / "experiment_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    return json.load(f)
            else:
                return {'error': 'Results file not found', 'experiment_name': exp_name}
        else:
            return {'error': result.stderr, 'experiment_name': exp_name}
            
    except subprocess.TimeoutExpired:
        return {'error': 'Experiment timed out', 'experiment_name': exp_name}
    except Exception as e:
        return {'error': str(e), 'experiment_name': exp_name}
    finally:
        # Clean up temporary script
        if script_path.exists():
            script_path.unlink()

def run_parallel_tuning(experiments: List[Dict], max_parallel: int = 3) -> Dict:
    """
    Run hyperparameter tuning experiments in parallel.
    
    Args:
        experiments: List of experiment configurations
        max_parallel: Maximum number of parallel processes
        
    Returns:
        Dictionary with overall results
    """
    # Create results directory
    results_dir = Path("checkpoints") / EXPERIMENT_NAME
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = {
        'experiment_name': EXPERIMENT_NAME,
        'description': f'Parallel hyperparameter tuning with {max_parallel} parallel processes',
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'target_examples': TARGET_EXAMPLES,
        'device': str(device),
        'num_experiments': len(experiments),
        'max_parallel': max_parallel,
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
    print(f"PARALLEL HYPERPARAMETER TUNING")
    print(f"{'='*60}")
    print(f"Experiment Name: {EXPERIMENT_NAME}")
    print(f"Device: {device}")
    print(f"Max parallel processes: {max_parallel}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Results directory: {results_dir}")
    print(f"{'='*60}")
    
    # Discover data
    print("\nDiscovering processed data files...")
    data_files = discover_processed_files("data/processed")
    total_examples = estimate_dataset_size(data_files, max_files=10)
    print(f"Found {len(data_files)} data files with approximately {total_examples:,} training examples")
    
    # Run experiments in parallel
    start_time = time.time()
    all_results = []
    
    with multiprocessing.Pool(processes=max_parallel) as pool:
        # Submit all experiments
        futures = []
        for exp_config in experiments:
            future = pool.apply_async(
                run_single_experiment,
                args=(exp_config, str(results_dir), EXPERIMENT_NAME)
            )
            futures.append(future)
        
        # Collect results as they complete
        for i, future in enumerate(futures):
            try:
                result = future.get(timeout=7200)  # 2 hour timeout
                all_results.append(result)
                print(f"Completed experiment {i+1}/{len(experiments)}: {result.get('experiment_name', 'unknown')}")
                
                if 'error' in result:
                    print(f"  Error: {result['error']}")
                else:
                    print(f"  Best val loss: {result.get('best_val_loss', 'N/A')}")
                    
            except Exception as e:
                print(f"Experiment {i+1} failed: {e}")
                all_results.append({'error': str(e), 'experiment_name': experiments[i]['experiment_name']})
    
    total_time = time.time() - start_time
    
    # Create overall results
    successful_experiments = [r for r in all_results if 'error' not in r]
    
    overall_results = {
        'experiment_name': EXPERIMENT_NAME,
        'total_time': total_time,
        'num_experiments': len(experiments),
        'successful_experiments': len(successful_experiments),
        'failed_experiments': len(all_results) - len(successful_experiments),
        'experiments': successful_experiments,
        'failed': [r for r in all_results if 'error' in r]
    }
    
    # Save overall results
    with open(results_dir / "overall_results.json", "w") as f:
        json.dump(overall_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("PARALLEL TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Successful experiments: {len(successful_experiments)}/{len(experiments)}")
    
    if successful_experiments:
        best_exp = min(successful_experiments, key=lambda x: x['best_val_loss'])
        print(f"\nBest experiment: {best_exp['experiment_name']}")
        print(f"Best validation loss: {best_exp['best_val_loss']:.6f}")
    
    print(f"\nAll results saved to: {results_dir}")
    return overall_results

if __name__ == "__main__":
    # Run parallel tuning
    overall_results = run_parallel_tuning(experiments, args.max_parallel)
    
    # Auto-analysis if requested
    if args.auto_analyze:
        print(f"\n{'='*60}")
        print("AUTO-ANALYSIS ENABLED")
        print(f"{'='*60}")
        print("Running analysis automatically...")
        
        try:
            import subprocess
            import sys
            
            results_dir = f"checkpoints/{EXPERIMENT_NAME}"
            analysis_cmd = [sys.executable, "analyze_tuning_results.py", results_dir]
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
            print(f"python analyze_tuning_results.py checkpoints/{EXPERIMENT_NAME}") 