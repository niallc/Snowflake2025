#!/usr/bin/env python3
import subprocess
import itertools
import time
import json
import csv
from pathlib import Path
from datetime import datetime
import os

# Define your sweep grid here (edit as needed)
SWEEP = {
    "learning-rate": [0.001, 0.01],
    "batch-size": [128, 256],
    "max-grad-norm": [100, 8, 20],
    "dropout": [0.0005, 0],
    "weight-decay": [1e-4, 1e-3],
    # Add more as needed
}

DATA_DIR = "data/processed"
RESULTS_DIR = "checkpoints/sweep"
EPOCHS = 10
MAX_SAMPLES = 200_000 # Is an underscore ignored in a numberic literal in python? Answer: yes.

def all_param_combinations(sweep_dict):
    keys = list(sweep_dict.keys())
    for values in itertools.product(*[sweep_dict[k] for k in keys]):
        yield dict(zip(keys, values))

def run_one(config, run_idx):
    # Build command
    # Include the data and time (up to minutes) in the experiment name
    exp_name = f"sweep_run_{run_idx}_" + "_".join(f"{k}{v}" for k, v in config.items()) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(RESULTS_DIR) / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "-m", "scripts.run_training",
        "--epochs", str(EPOCHS),
        "--max-samples", str(MAX_SAMPLES),
        "--data-dir", DATA_DIR,
        "--results-dir", str(RESULTS_DIR),
        "--experiment-name", exp_name,
        "--verbose", "2",
    ]
    for k, v in config.items():
        cmd += [f"--{k}", str(v)]
    print(f"\n=== Starting run {run_idx}: {exp_name} ===")
    print("Command:", " ".join(cmd))
    start = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    proc = subprocess.run(cmd, capture_output=False, text=True, env=env)
    elapsed = time.time() - start
    print(f"Run {run_idx} finished in {elapsed/60:.1f} min")
    if proc.returncode != 0:
        print(f"Run {run_idx} failed! Output:\n{proc.stdout}\n{proc.stderr}")
        return None
    # Print last lines of stdout for quick glance
    print(proc.stdout.splitlines()[-15:])
    # Parse results
    summary = summarize_run(results_dir)
    print(f"Summary for {exp_name}: {summary}")
    return summary

def summarize_run(results_dir):
    # Try to read training_results.json
    try:
        with open(Path(results_dir) / "training_results.json") as f:
            results = json.load(f)
        best_val = results.get("best_val_loss")
        best_train = min(results.get("train_losses", []), default=None)
        epochs = results.get("epochs_trained")
        total_time = results.get("total_training_time")
        return {
            "best_val_loss": best_val,
            "best_train_loss": best_train,
            "epochs_trained": epochs,
            "total_time_min": total_time / 60 if total_time else None,
        }
    except Exception as e:
        print(f"Could not summarize run in {results_dir}: {e}")
        return {}

# To run this script from the command line in the parent direcory (project root),
# being mindful of imports, run:
# python -m scripts.hyperparam_sweep
def main():
    all_configs = list(all_param_combinations(SWEEP))
    print(f"Total runs to launch: {len(all_configs)}")
    all_summaries = []
    for i, config in enumerate(all_configs):
        summary = run_one(config, i)
        all_summaries.append({"config": config, "summary": summary})
        # Optionally: write intermediate results
        with open(Path(RESULTS_DIR) / "sweep_summary.json", "w") as f:
            json.dump(all_summaries, f, indent=2)
    print("\n=== Sweep complete! ===")
    for entry in all_summaries:
        print(entry)

if __name__ == "__main__":
    main()