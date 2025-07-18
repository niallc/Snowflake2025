#!/usr/bin/env python3
"""
In-process hyperparameter sweep for Hex AI, using run_hyperparameter_tuning_current_data.
No subprocesses are launched; all experiments are run in-process for easier debugging and unified code path.
"""

import itertools
import logging
import time
import json
import csv
from pathlib import Path
from datetime import datetime
import os
import sys

# Removed for now as I'm concerned about brittleness, using PYTHONPATH=. instead.
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hex_ai.training_utils_legacy import run_hyperparameter_tuning_current_data

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
    "weight_decay": [2e-4],
    "value_learning_rate_factor": [0.1, 0.5],  # Value head learns slower
    "value_weight_decay_factor": [2.0, 5.0],  # Value head gets more regularization
    # Add more as needed
}

DATA_DIR = "data/processed"
RESULTS_DIR = "checkpoints/sweep"
EPOCHS = 3
MAX_SAMPLES = 200_000

# Build all parameter combinations
def all_param_combinations(sweep_dict):
    keys = list(sweep_dict.keys())
    for values in itertools.product(*[sweep_dict[k] for k in keys]):
        yield dict(zip(keys, values))

if __name__ == "__main__":
    print("WARNING: This sweep runs all experiments in-process using run_hyperparameter_tuning_current_data. No subprocesses will be launched.")

    all_configs = list(all_param_combinations(SWEEP))
    print(f"Total runs to launch: {len(all_configs)}")
    experiments = []
    for i, config in enumerate(all_configs):
        exp_name = f"sweep_run_{i}_" + "_".join(f"{k}{v}" for k, v in config.items()) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        experiments.append({
            'experiment_name': exp_name,
            'hyperparameters': config
        })

    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    overall_results = run_hyperparameter_tuning_current_data(
        experiments=experiments,
        data_dir=DATA_DIR,
        results_dir=str(results_dir),
        train_ratio=0.8,
        num_epochs=EPOCHS,
        early_stopping_patience=3,
        random_seed=42,
        max_examples_per_split=MAX_SAMPLES,
        experiment_name="sweep_run"
    )
    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Successful experiments: {overall_results.get('successful_experiments', 'N/A')}/{overall_results.get('num_experiments', 'N/A')}")

    # Find best experiment
    if overall_results.get('experiments'):
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