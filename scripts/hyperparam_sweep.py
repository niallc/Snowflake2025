"""
In-process hyperparameter sweep for Hex AI, using run_hyperparameter_tuning_current_data.
No subprocesses are launched; all experiments are run in-process for easier debugging and unified code path.
"""

# =============================================================
#  Imports and Logging Setup
# =============================================================

import itertools
import logging
import time
from pathlib import Path
from datetime import datetime
import traceback
import sys
import os
import hashlib
import json
import math
import argparse

from hex_ai.training_orchestration import run_hyperparameter_tuning_current_data
from hex_ai.file_utils import GracefulShutdown
from hex_ai.system_utils import check_virtual_env
from hex_ai.error_handling import GracefulShutdownRequested
check_virtual_env("hex_ai_env")

###### Logging setup ######
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / ('hex_ai_training_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.log')

file_handler = logging.FileHandler(log_file, mode='a')
formatter = logging.Formatter('%(asctime)s %(levelname)s:%(name)s: %(message)s')
file_handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.addHandler(file_handler)
root_logger.setLevel(logging.INFO)  # Or whatever level you want

# Optionally, also add a StreamHandler for terminal output:
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)
###### End of logging setup ######

# =============================================================
#  Sweep Configuration and Parameter Processing
# =============================================================

# Define your sweep grid here (edit as needed)
SWEEP = {
    "batch_size": [256],
    "max_grad_norm": [20],
    "weight_decay": [1e-4],
    "value_learning_rate_factor": [1],  # Value head learns slower if this is < 1
    "value_weight_decay_factor": [1],  # Value head gets more regularization if this is > 1
    "policy_weight": [0.2],
    
    # Likely resolved:
    "dropout_prob": [0],
    "learning_rate": [0.001],
    # Add more as needed
}

# Short labels for parameters
SHORT_LABELS = {
    "learning_rate": "lr",
    "batch_size": "bs",
    "max_grad_norm": "mgn",
    "dropout_prob": "do",
    "weight_decay": "wd",
    "value_learning_rate_factor": "vlrf",
    "value_weight_decay_factor": "vwdf",
    "policy_weight": "pw",
    "value_weight": "vw",
    # Add more as needed
}

# Determine which parameters vary in this sweep
VARYING_PARAMS = [k for k, v in SWEEP.items() if len(v) > 1]

# Configuration
MAX_SAMPLES = 35_000_000  # Training samples (will be 4x larger with augmentation)
MAX_VALIDATION_SAMPLES = 925_000  # Validation samples (no augmentation)
MINI_EPOCH_BATCHES = math.floor(500000 * 2 /256) # The total samples per epoch is batch_size (see sweep) * mini_epoch_batches
AUGMENTATION_CONFIG = {'enable_augmentation': True}
EPOCHS = 2

# Build all parameter combinations
def all_param_combinations(sweep_dict):
    keys = list(sweep_dict.keys())
    for values in itertools.product(*[sweep_dict[k] for k in keys]):
        yield dict(zip(keys, values))

# =============================================================
#  Helper Functions
# =============================================================

def make_experiment_name(config, idx, tag = ""):
    # Only include varying params, use short labels
    parts = []
    for k in VARYING_PARAMS:
        short = SHORT_LABELS.get(k, k)
        val = config[k]
        # Remove trailing zeros for floats, e.g. 0.10 -> 0.1
        if isinstance(val, float):
            val = ('%.4f' % val).rstrip('0').rstrip('.')
        parts.append(f"{short}{val}")
    # Optionally, add a short hash for extra uniqueness
    config_str = json.dumps({k: config[k] for k in VARYING_PARAMS}, sort_keys=True)
    short_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]
    # Timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{tag}_exp{idx}_{'_'.join(parts)}_{short_hash}_{timestamp}"

def print_sweep_summary(results, results_dir, interrupted=False):
    print("\n" + "="*60)
    if not interrupted:
        print("SWEEP COMPLETE")
    else:
        print("SWEEP INTERRUPTED")
    print("="*60)
    if results is not None:
        print(f"Successful experiments: {results.get('successful_experiments', 'N/A')}/{results.get('num_experiments', 'N/A')}")
        # Find best experiment
        if results.get('experiments'):
            best_exp = min(results['experiments'], key=lambda x: x['best_val_loss'])
            print(f"\nBest experiment: {best_exp['experiment_name']}")
            print(f"Best validation loss: {best_exp['best_val_loss']:.6f}")
            print(f"Hyperparameters: {best_exp['hyperparameters']}")
            # Show all results sorted by validation loss
            print(f"\nAll experiments ranked by validation loss:")
            sorted_experiments = sorted(results['experiments'], key=lambda x: x['best_val_loss'])
            for i, exp in enumerate(sorted_experiments):
                print(f"{i+1}. {exp['experiment_name']}: {exp['best_val_loss']:.6f}")
        else:
            print("\nNo successful experiments!")
    print(f"\nAll results saved to: {results_dir}")
    print("Run 'python analyze_tuning_results.py' to analyze the results.")

# =============================================================
#  Main Execution and Experiment Loop
# =============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed/shuffled", help="Directory containing processed data files")
    parser.add_argument("--results_dir", type=str, default="checkpoints/hyperparameter_tuning", help="Directory to save experiment results")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs to train")
    parser.add_argument("--max_samples", type=int, default=MAX_SAMPLES, help="Max training samples (unaugmented)")
    parser.add_argument("--max_validation_samples", type=int, default=MAX_VALIDATION_SAMPLES, help="Max validation samples (unaugmented)")
    parser.add_argument("--mini_epoch_batches", type=int, default=MINI_EPOCH_BATCHES, help="Mini-epoch batches per epoch")
    parser.add_argument("--no_augmentation", action="store_true", help="Disable data augmentation")
    args = parser.parse_args()

    shutdown_handler = GracefulShutdown()

    all_configs = list(all_param_combinations(SWEEP))
    experiments = []
    for i, config in enumerate(all_configs):
        if shutdown_handler.shutdown_requested:
            break
        # Compute value_weight so that policy_weight + value_weight = 1
        config = dict(config)  # Make a copy to avoid mutating the sweep dict
        if "policy_weight" in config:
            config["value_weight"] = 1.0 - config["policy_weight"]
        exp_name = make_experiment_name(config, i, tag = "loss_weight_sweep")
        experiments.append({
            'experiment_name': exp_name,
            'hyperparameters': config
        })

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    results = None
    interrupted = False
    try:
        if shutdown_handler.shutdown_requested:
            interrupted = True
            print_sweep_summary(results, results_dir, interrupted=True)
            sys.exit(0)
        # Run hyperparameter tuning with fail_fast enabled
        results = run_hyperparameter_tuning_current_data(
            experiments=experiments,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            train_ratio=0.8,
            num_epochs=args.epochs,
            early_stopping_patience=None,  # Disable early stopping for now
            mini_epoch_batches=args.mini_epoch_batches,
            random_seed=42,
            max_examples_unaugmented=args.max_samples,
            max_validation_examples=args.max_validation_samples,
            enable_augmentation=not args.no_augmentation,
            fail_fast=True,
            shutdown_handler=shutdown_handler
        )
        total_time = time.time() - start_time
        print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print_sweep_summary(results, results_dir, interrupted=False)
    except GracefulShutdownRequested:
        print("\n" + "="*60)
        print("GRACEFUL SHUTDOWN REQUESTED")
        print("="*60)
        interrupted = True
        # Enhanced summary output
        if results is not None and isinstance(results, dict):
            completed = results.get('experiments', [])
            completed_names = [exp.get('experiment_name', 'UNKNOWN') for exp in completed]
            num_completed = len(completed_names)
            num_total = results.get('num_experiments', len(experiments))
            print(f"Experiments completed: {num_completed}/{num_total}")
            for name in completed_names:
                print(f"  - {name}")
            # Incomplete experiments
            all_names = [exp['experiment_name'] for exp in experiments]
            incomplete_names = [name for name in all_names if name not in completed_names]
            num_incomplete = len(incomplete_names)
            if num_incomplete > 0:
                print(f"Experiments not completed: {num_incomplete}/{num_total}")
                for name in incomplete_names:
                    print(f"  - {name}")
            # Best validation loss so far
            if completed:
                best_exp = min(
                    (exp for exp in completed if 'best_val_loss' in exp),
                    key=lambda x: x['best_val_loss'],
                    default=None
                )
                if best_exp is not None:
                    print(f"\nBest validation loss so far: {best_exp['best_val_loss']:.6f} ({best_exp['experiment_name']})")
        else:
            print("No completed experiments to summarize.")
        print(f"\nAll results saved to: {results_dir}")
        print(f"Logs saved to: {log_file}")
        print_sweep_summary(results, results_dir, interrupted=True)
        sys.exit(0)
    except Exception as e:
        print("\n" + "="*60)
        print("SWEEP FAILED - FAIL FAST MODE")
        print("="*60)
        print(f"Exception: {e}")
        print("Traceback:")
        traceback.print_exc()
        # Try to print more context if available
        if hasattr(e, 'args') and e.args and isinstance(e.args[0], dict):
            err_info = e.args[0]
            print("\nAdditional error context:")
            for k, v in err_info.items():
                print(f"  {k}: {v}")
        print("\nSweep stopped due to a critical error. Please check the logs above and in the logs/ directory for more details.")
        print("If the error was due to a missing or corrupt file, check the data directory and filenames listed above.")
        print_sweep_summary(results, results_dir, interrupted=True)