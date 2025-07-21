"""
In-process hyperparameter sweep for Hex AI, using run_hyperparameter_tuning_current_data.
No subprocesses are launched; all experiments are run in-process for easier debugging and unified code path.
"""

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

# Safety check: ensure running inside the correct virtual environment
expected_env = "hex_ai_env"
venv_path = os.environ.get("VIRTUAL_ENV", "")
if not venv_path or expected_env not in venv_path:
    sys.stderr.write(
        f"\nERROR: This script must be run inside the '{expected_env}' virtual environment.\n"
        f"Please activate it first by running:\n\n"
        f"    source {expected_env}/bin/activate\n\n"
        f"Then re-run this script.\n"
    )
    sys.exit(1)

from hex_ai.training_utils_legacy import run_hyperparameter_tuning_current_data
from hex_ai.file_utils import GracefulShutdown

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

# Define your sweep grid here (edit as needed)
SWEEP = {
    "learning_rate": [0.001],
    "batch_size": [256],
    "max_grad_norm": [20],
    "dropout_prob": [0],
    "weight_decay": [1e-4],
    "value_learning_rate_factor": [0.1],  # Value head learns slower
    "value_weight_decay_factor": [1],  # Value head gets more regularization
    "policy_weight": [0.01, 0.7],
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

# Configuration
MAX_SAMPLES = 3_200_000  # Training samples (will be 4x larger with augmentation)
MAX_VALIDATION_SAMPLES = 400_000  # Validation samples (no augmentation)
AUGMENTATION_CONFIG = {'enable_augmentation': True}
EPOCHS = 2

print(f"Running hyperparameter sweep with shuffled data:")
print(f"  Data directory: data/processed/shuffled")
print(f"  Training samples: {MAX_SAMPLES:,} (effective: {MAX_SAMPLES * 4:,} with augmentation)")
print(f"  Validation samples: {MAX_VALIDATION_SAMPLES:,}")
print(f"  Data augmentation: {'Enabled' if AUGMENTATION_CONFIG['enable_augmentation'] else 'Disabled'}")
print(f"  Max Epochs: {EPOCHS}")

# Build all parameter combinations
def all_param_combinations(sweep_dict):
    keys = list(sweep_dict.keys())
    for values in itertools.product(*[sweep_dict[k] for k in keys]):
        yield dict(zip(keys, values))

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

if __name__ == "__main__":
    print("WARNING: This sweep runs all experiments in-process using run_hyperparameter_tuning_current_data. No subprocesses will be launched.")
    print(f"Data augmentation: {'ENABLED' if AUGMENTATION_CONFIG['enable_augmentation'] else 'DISABLED'}")

    shutdown_handler = GracefulShutdown()

    all_configs = list(all_param_combinations(SWEEP))
    print(f"Total runs to launch: {len(all_configs)}")
    experiments = []
    for i, config in enumerate(all_configs):
        if shutdown_handler.shutdown_requested:
            print("\nGraceful shutdown requested. Exiting sweep early.")
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

    results_dir = Path("checkpoints/hyperparameter_tuning")
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
            data_dir="data/processed/shuffled",
            results_dir="checkpoints/hyperparameter_tuning",
            train_ratio=0.8,
            num_epochs=EPOCHS,
            early_stopping_patience=None,  # Disable early stopping for now
            random_seed=42,
            max_examples_per_split=MAX_SAMPLES,
            max_validation_examples=MAX_VALIDATION_SAMPLES,
            enable_augmentation=AUGMENTATION_CONFIG['enable_augmentation'],
            fail_fast=True
        )
        total_time = time.time() - start_time
        print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print_sweep_summary(results, results_dir, interrupted=False)
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