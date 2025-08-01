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
import argparse

from hex_ai.training_orchestration import run_hyperparameter_tuning_current_data
from hex_ai.file_utils import GracefulShutdown
from hex_ai.error_handling import GracefulShutdownRequested
# Environment validation is now handled automatically in hex_ai/__init__.py

# Create timestamp for the entire run (without minutes/seconds)
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H")

###### Logging setup ######
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / (f'hex_ai_training_{RUN_TIMESTAMP}.log')

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

def calculate_mini_epoch_samples(
    max_samples: int,
    min_mini_epochs: int = 10,
    max_mini_epochs: int = 300,
    target_samples_per_mini_epoch: int = 250000
) -> int:
    """Calculate mini-epoch samples based on target samples and min/max constraints."""
    target_mini_epochs = max_samples // target_samples_per_mini_epoch
    num_mini_epochs = max(min_mini_epochs, min(max_mini_epochs, target_mini_epochs))
    return max_samples // num_mini_epochs

# Define your sweep grid here (edit as needed)
SWEEP = {
    "batch_size": [256],
    "max_grad_norm": [2.0, 1.0],
    "weight_decay": [1e-4],
    "value_learning_rate_factor": [0.2, 1],  # Value head learns slower if this is < 1
    "value_weight_decay_factor": [1],  # Value head gets more regularization if this is > 1
    "policy_weight": [0.2],
    
    # Likely resolved:
    "dropout_prob": [0],
    "learning_rate": [0.0003, 0.00003],
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
MAX_SAMPLES = 1_000_000  # Training samples (will be 4x larger with augmentation)
MAX_VALIDATION_SAMPLES = 95_000  # Validation samples (no augmentation)

# Mini-epoch configuration - more intuitive parameters
MIN_MINI_EPOCHS_PER_EPOCH = 5      # At least 5 mini-epochs per epoch
MAX_MINI_EPOCHS_PER_EPOCH = 200     # No more than 200 mini-epochs per epoch
TARGET_SAMPLES_PER_MINI_EPOCH = 250000  # Target ~250k unaugmented samples per mini-epoch

AUGMENTATION_CONFIG = {'enable_augmentation': True}
EPOCHS = 1  # training now resets the epoch count this this is the further number of epochs to train

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
            # Filter out None results, failed experiments, and experiments without best_val_loss
            valid_experiments = [exp for exp in results['experiments'] 
                               if exp is not None and exp.get('status') != 'failed' and 'best_val_loss' in exp]
            
            if valid_experiments:
                best_exp = min(valid_experiments, key=lambda x: x['best_val_loss'])
                print(f"\nBest experiment: {best_exp['experiment_name']}")
                print(f"Best validation loss: {best_exp['best_val_loss']:.6f}")
                print(f"Hyperparameters: {best_exp['hyperparameters']}")
                # Show all results sorted by validation loss
                print(f"\nAll experiments ranked by validation loss:")
                sorted_experiments = sorted(valid_experiments, key=lambda x: x['best_val_loss'])
                for i, exp in enumerate(sorted_experiments):
                    print(f"{i+1}. {exp['experiment_name']}: {exp['best_val_loss']:.6f}")
            else:
                print("\nNo experiments with valid validation loss!")
            
            # Show failed experiments if any
            failed_experiments = [exp for exp in results['experiments'] 
                                if exp is not None and exp.get('status') == 'failed']
            if failed_experiments:
                print(f"\nFailed experiments ({len(failed_experiments)}):")
                for exp in failed_experiments:
                    print(f"  - {exp['experiment_name']}: {exp.get('error', 'Unknown error')}")
        else:
            print("\nNo successful experiments!")
    print(f"\nAll results saved to: {results_dir}")
    print("Run 'python analyze_tuning_results.py' to analyze the results.")

# =============================================================
#  Main Execution and Experiment Loop
# =============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run hyperparameter tuning with multiple data sources",
        epilog="""
Examples:
  # Single data directory (backward compatible)
  python scripts/hyperparam_sweep.py --data_dirs data/processed/shuffled

  # Multiple data directories
  python scripts/hyperparam_sweep.py --data_dirs data/processed/shuffled data/processed/jul_29_shuffled

  # Skip first 200 files from second directory
  python scripts/hyperparam_sweep.py --data_dirs data/processed/shuffled data/processed/jul_29_shuffled --skip_files 0 200

  # Resume training from a specific checkpoint file
  python scripts/hyperparam_sweep.py --data_dirs data/processed/shuffled --resume_from checkpoints/hyperparameter_tuning/experiment_name/epoch2_mini36.pt.gz
        """
    )
    
    # Data source arguments
    parser.add_argument("--data_dirs", type=str, nargs='+', required=True,
                       help="One or more directories containing processed data files")
    
    parser.add_argument(
        '--skip_files',
        type=int,
        nargs='+',
        help='Number of files to skip from the beginning of each data directory (one value per directory, e.g., --skip_files 0 200 to skip 0 from first dir, 200 from second)'
    )
    
    # Resume training arguments
    parser.add_argument(
        '--resume_from', 
        type=str,
        help='Resume training from a specific checkpoint file (e.g., checkpoints/hyperparameter_tuning/experiment_name/epoch2_mini36.pt.gz)'
    )
    
    # Training arguments
    parser.add_argument("--results_dir", type=str, default="checkpoints/hyperparameter_tuning", 
                       help="Directory to save experiment results")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs to train")
    parser.add_argument("--max_samples", type=int, default=MAX_SAMPLES, 
                       help="Max training samples (unaugmented)")
    parser.add_argument("--max_validation_samples", type=int, default=MAX_VALIDATION_SAMPLES, 
                       help="Max validation samples (unaugmented)")
    parser.add_argument("--min_mini_epochs_per_epoch", type=int, default=MIN_MINI_EPOCHS_PER_EPOCH,
                       help="Minimum mini-epochs per epoch")
    parser.add_argument("--max_mini_epochs_per_epoch", type=int, default=MAX_MINI_EPOCHS_PER_EPOCH,
                       help="Maximum mini-epochs per epoch")
    parser.add_argument("--target_samples_per_mini_epoch", type=int, default=TARGET_SAMPLES_PER_MINI_EPOCH,
                       help="Target unaugmented samples per mini-epoch")
    parser.add_argument("--no_augmentation", action="store_true", help="Disable data augmentation")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducible results")
    
    args = parser.parse_args()

    # Validate resume arguments
    if args.resume_from and not Path(args.resume_from).exists():
        print(f"ERROR: Resume path does not exist: {args.resume_from}")
        sys.exit(1)

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
    
    print(f"Starting hyperparameter sweep with random seed: {args.random_seed}")

    start_time = time.time()
    results = None
    interrupted = False
    try:
        if shutdown_handler.shutdown_requested:
            interrupted = True
            print_sweep_summary(results, results_dir, interrupted=True)
            sys.exit(0)
        
        # Calculate mini-epoch samples
        mini_epoch_samples = calculate_mini_epoch_samples(
            max_samples=args.max_samples,
            min_mini_epochs=args.min_mini_epochs_per_epoch,
            max_mini_epochs=args.max_mini_epochs_per_epoch,
            target_samples_per_mini_epoch=args.target_samples_per_mini_epoch
        )
        
        print(f"Mini-epochs: {args.max_samples // args.target_samples_per_mini_epoch} target, {args.min_mini_epochs_per_epoch}-{args.max_mini_epochs_per_epoch} range, {mini_epoch_samples:,} samples each")
        
        # Run hyperparameter tuning
        results = run_hyperparameter_tuning_current_data(
            experiments=experiments,
            data_dirs=args.data_dirs,
            results_dir=args.results_dir,
            train_ratio=0.8,
            num_epochs=args.epochs,
            early_stopping_patience=None,
            random_seed=args.random_seed,
            max_examples_unaugmented=args.max_samples,
            max_validation_examples=args.max_validation_samples,
            experiment_name=None,
            enable_augmentation=not args.no_augmentation,
            mini_epoch_samples=mini_epoch_samples,
            resume_from=args.resume_from,
            skip_files=args.skip_files,
            shutdown_handler=shutdown_handler,
            run_timestamp=RUN_TIMESTAMP
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