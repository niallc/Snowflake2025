#!/usr/bin/env python3
"""
Analyze hyperparameter tuning results.
- Plot training curves for all experiments
- Sanity check loss progression
- Compare experiment performance
- Generate summary report
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Standardized loss weights for fair comparison across experiments
# These weights are used to recalculate total loss from individual policy/value losses
# to ensure fair comparison regardless of what weights were used during training.
# 
# IMPORTANT: Different experiments may train with different policy/value weight combinations
# (e.g., balanced training uses 0.5/0.5, baseline uses 0.14/0.86). For analysis,
# we always use these fixed standardized weights to make loss comparisons fair.
# 
# Without standardization, experiments with higher value loss weights would appear
# to have higher total loss even if they achieved better individual policy and value losses.
POLICY_LOSS_WEIGHT = 0.14
VALUE_LOSS_WEIGHT = 0.86

def load_experiment_results(results_dir: str) -> Dict:
    """Load all experiment results from the results directory."""
    results_dir = Path(results_dir)
    experiments = {}
    
    # Load overall results
    overall_file = results_dir / "overall_results.json"
    if overall_file.exists():
        with open(overall_file, 'r') as f:
            overall_results = json.load(f)
    else:
        print(f"Warning: {overall_file} not found")
        overall_results = {}
    
    # Load individual experiment results
    for exp_dir in results_dir.iterdir():
        if exp_dir.is_dir() and exp_dir.name != "__pycache__":
            exp_name = exp_dir.name
            results_file = exp_dir / "experiment_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    experiments[exp_name] = json.load(f)
    
    return {
        'overall': overall_results,
        'experiments': experiments
    }

def extract_losses(exp_data, exp_dir=None):
    """
    Helper to get train/val losses from all_metrics if present.
    Falls back to CSV files if individual losses aren't in JSON.
    """
    if 'all_metrics' in exp_data:
        metrics = exp_data['all_metrics']
        train_losses = metrics.get('train_losses', [])
        val_losses = metrics.get('val_losses', [])
        # Also extract policy and value losses if available
        train_policy = metrics.get('train_policy_losses', metrics.get('policy_losses', []))
        train_value = metrics.get('train_value_losses', metrics.get('value_losses', []))
        val_policy = metrics.get('val_policy_losses', metrics.get('policy_losses', []))
        val_value = metrics.get('val_value_losses', metrics.get('value_losses', []))
        
        # If individual losses aren't in JSON, try to get them from CSV
        if not train_policy and exp_dir:
            csv_file = exp_dir / "training_metrics.csv"
            if csv_file.exists():
                try:
                    import pandas as pd
                    # CSV files don't have headers, so read without header
                    df = pd.read_csv(csv_file, header=None)
                    # Based on training_logger.py headers, the columns are:
                    # 3: epoch, 13: policy_loss, 14: value_loss, 15: total_loss
                    # 16: val_policy_loss, 17: val_value_loss, 18: val_total_loss
                    if len(df) > 0:
                        # Filter out summary rows (epoch column 3 should be numeric)
                        df = df[pd.to_numeric(df[3], errors='coerce').notna()]
                        if len(df) > 0:
                            train_policy = df[13].tolist()  # policy_loss
                            train_value = df[14].tolist()   # value_loss
                            val_policy = df[16].tolist()    # val_policy_loss
                            val_value = df[17].tolist()     # val_value_loss
                            print(f"  Extracted individual losses from CSV for {exp_data['experiment_name']}")
                except Exception as e:
                    print(f"  Warning: Could not read CSV for {exp_data['experiment_name']}: {e}")
        
        return train_losses, val_losses, train_policy, train_value, val_policy, val_value
    else:
        return (exp_data.get('train_losses', []), exp_data.get('val_losses', []),
                exp_data.get('policy_losses', []), exp_data.get('value_losses', []),
                exp_data.get('policy_losses', []), exp_data.get('value_losses', []))

def plot_training_curves(experiments: dict, save_path: str = None, standardized: bool = False, results_dir: str = None):
    """
    Plot training and validation loss curves for all experiments.
    
    Args:
        experiments: Dictionary of experiment results
        save_path: Path to save the plot
        standardized: If True, recalculate total loss using standardized weights
                     (POLICY_LOSS_WEIGHT=0.14, VALUE_LOSS_WEIGHT=0.86) for fair comparison.
                     This ensures experiments trained with different weight combinations
                     can be compared fairly.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(experiments)))
    for i, (exp_name, exp_data) in enumerate(experiments.items()):
        # Get experiment directory for CSV fallback
        exp_dir = Path(results_dir) / exp_name if results_dir else None
        train_losses, val_losses, train_policy, train_value, val_policy, val_value = extract_losses(exp_data, exp_dir)
        if standardized and train_policy and train_value and val_policy and val_value:
            # Recalculate total loss using standardized weights for fair comparison
            # This ensures experiments trained with different policy/value weight combinations
            # can be compared fairly (e.g., balanced training 0.5/0.5 vs baseline 0.14/0.86)
            train_losses = [POLICY_LOSS_WEIGHT * p + VALUE_LOSS_WEIGHT * v for p, v in zip(train_policy, train_value)]
            val_losses = [POLICY_LOSS_WEIGHT * p + VALUE_LOSS_WEIGHT * v for p, v in zip(val_policy, val_value)]
        if train_losses and val_losses:
            epochs = range(1, len(train_losses) + 1)
            ax1.plot(epochs, train_losses, label=exp_name, color=colors[i], marker='o', linewidth=2)
            ax2.plot(epochs, val_losses, label=exp_name, color=colors[i], marker='s', linewidth=2)
    ax1.set_title('Training Loss' + (' (Standardized)' if standardized else ''))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.set_title('Validation Loss' + (' (Standardized)' if standardized else ''))
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    plt.close(fig)

def debug_standardized_loss_calculation(experiments: Dict):
    """
    Debug function to verify standardized loss calculation and show differences.
    This helps verify that the standardized loss calculation is working correctly
    and shows the impact of using standardized weights vs training weights.
    """
    print("\n" + "="*80)
    print("STANDARDIZED LOSS CALCULATION DEBUG")
    print("="*80)
    
    for exp_name, exp_data in experiments.items():
        # Get experiment directory for CSV fallback
        exp_dir = Path(results_dir) / exp_name if results_dir else None
        train_losses, val_losses, train_policy, train_value, val_policy, val_value = extract_losses(exp_data, exp_dir)
        
        if train_policy and train_value and val_policy and val_value:
            # Get training hyperparameters to see what weights were used during training
            hyperparams = exp_data.get('hyperparameters', {})
            train_policy_weight = hyperparams.get('policy_weight', 'unknown')
            train_value_weight = hyperparams.get('value_weight', 'unknown')
            
            # Calculate standardized losses for comparison
            if len(train_policy) > 0 and len(train_value) > 0:
                # Raw total loss (what was actually used during training)
                raw_train_final = train_losses[-1] if train_losses else 'N/A'
                raw_val_final = val_losses[-1] if val_losses else 'N/A'
                
                # Standardized total loss (what we use for fair comparison)
                std_train_final = POLICY_LOSS_WEIGHT * train_policy[-1] + VALUE_LOSS_WEIGHT * train_value[-1]
                std_val_final = POLICY_LOSS_WEIGHT * val_policy[-1] + VALUE_LOSS_WEIGHT * val_value[-1]
                
                print(f"\n{exp_name}:")
                print(f"  Training weights used: policy={train_policy_weight}, value={train_value_weight}")
                print(f"  Standardized weights: policy={POLICY_LOSS_WEIGHT}, value={VALUE_LOSS_WEIGHT}")
                print(f"  Raw final train loss: {raw_train_final}")
                print(f"  Raw final val loss: {raw_val_final}")
                print(f"  Standardized final train loss: {std_train_final:.6f}")
                print(f"  Standardized final val loss: {std_val_final:.6f}")
                
                if isinstance(raw_train_final, (int, float)):
                    diff = std_train_final - raw_train_final
                    print(f"  Difference (std - raw): {diff:.6f}")
                    if abs(diff) > 0.1:
                        print(f"  ⚠️  Large difference detected - standardized calculation working correctly")

def sanity_check_losses(experiments: Dict) -> Dict:
    """Sanity check loss progression for each experiment."""
    sanity_results = {}
    
    for exp_name, exp_data in experiments.items():
        issues = []
        train_losses, val_losses = extract_losses(exp_data)[:2] # Only check raw losses for now
        if train_losses and val_losses:
            # Check if losses are decreasing (generally expected)
            if len(train_losses) > 1:
                if train_losses[-1] > train_losses[0]:
                    issues.append("Training loss increased over time")
                
                # Check for NaN or inf values
                if any(np.isnan(train_losses)) or any(np.isinf(train_losses)):
                    issues.append("Training loss contains NaN or inf values")
                
                if any(np.isnan(val_losses)) or any(np.isinf(val_losses)):
                    issues.append("Validation loss contains NaN or inf values")
                
                # Check for reasonable loss values
                if max(train_losses) > 10:
                    issues.append("Training loss seems unusually high")
                
                if max(val_losses) > 10:
                    issues.append("Validation loss seems unusually high")
                
                # Check for overfitting (val loss increasing while train loss decreasing)
                if len(train_losses) > 2 and len(val_losses) > 2:
                    train_trend = train_losses[-1] - train_losses[-2]
                    val_trend = val_losses[-1] - val_losses[-2]
                    
                    if train_trend < 0 and val_trend > 0.1:  # Significant increase
                        issues.append("Possible overfitting detected")
            
            sanity_results[exp_name] = {
                'issues': issues,
                'final_train_loss': train_losses[-1] if train_losses else None,
                'final_val_loss': val_losses[-1] if val_losses else None,
                'best_val_loss': min(val_losses) if val_losses else None,
                'loss_decrease': train_losses[0] - train_losses[-1] if len(train_losses) > 1 else 0
            }
    
    return sanity_results

def create_summary_report(results_dir: str, save_path: str = None):
    """Create a comprehensive summary report."""
    results = load_experiment_results(results_dir)
    
    if not results['experiments']:
        print("No experiment results found!")
        return
    
    # Debug standardized loss calculation
    debug_standardized_loss_calculation(results['experiments'])
    
    # Sanity check
    sanity_results = sanity_check_losses(results['experiments'])
    
    # Create summary table
    summary_data = []
    for exp_name, exp_data in results['experiments'].items():
        sanity = sanity_results.get(exp_name, {})
        
        # Get early stopping info
        early_stopped = exp_data.get('early_stopped', False)
        epochs_trained = exp_data.get('epochs_trained', 'N/A')
        
        summary_data.append({
            'Experiment': exp_name,
            'Final Train Loss': sanity.get('final_train_loss', 'N/A'),
            'Final Val Loss': sanity.get('final_val_loss', 'N/A'),
            'Best Val Loss': sanity.get('best_val_loss', 'N/A'),
            'Epochs Trained': epochs_trained,
            'Early Stopped': 'Yes' if early_stopped else 'No',
            'Loss Decrease': f"{sanity.get('loss_decrease', 0):.4f}",
            'Issues': '; '.join(sanity.get('issues', [])) if sanity.get('issues') else 'None'
        })
    
    # Sort by best validation loss
    summary_data.sort(key=lambda x: x['Best Val Loss'] if x['Best Val Loss'] != 'N/A' else float('inf'))
    
    # Create DataFrame
    df = pd.DataFrame(summary_data)
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING SUMMARY REPORT")
    print("="*80)
    
    print(f"\nResults Directory: {results_dir}")
    print(f"Number of Experiments: {len(results['experiments'])}")
    
    if results['overall']:
        print(f"Successful Experiments: {results['overall'].get('successful_experiments', 'N/A')}")
        total_time = results['overall'].get('total_time', 'N/A')
        if isinstance(total_time, (int, float)):
            print(f"Total Time: {total_time:.1f}s")
        else:
            print(f"Total Time: {total_time}s")
    
    print("\n" + "="*80)
    print("EXPERIMENT RANKINGS (by Best Validation Loss)")
    print("="*80)
    print(df.to_string(index=False))
    
    print("\n" + "="*80)
    print("SANITY CHECK RESULTS")
    print("="*80)
    
    total_issues = 0
    for exp_name, sanity in sanity_results.items():
        issues = sanity.get('issues', [])
        total_issues += len(issues)
        
        if issues:
            print(f"\n{exp_name}:")
            for issue in issues:
                print(f"  ⚠️  {issue}")
        else:
            print(f"\n{exp_name}: ✅ No issues detected")
    
    print(f"\nTotal issues found: {total_issues}")
    
    # Plot training curves (raw and standardized)
    plot_save_path = Path(results_dir) / ("training_curves_large.png" if "50k" in str(results_dir) else "training_curves_small.png")
    plot_training_curves(results['experiments'], str(plot_save_path), standardized=False, results_dir=results_dir)
    plot_save_path_std = Path(results_dir) / ("training_curves_large_standardized.png" if "50k" in str(results_dir) else "training_curves_small_standardized.png")
    plot_training_curves(results['experiments'], str(plot_save_path_std), standardized=True, results_dir=results_dir)
    
    # Save summary to file
    if save_path:
        summary_file = Path(save_path).parent / "summary_report.txt"
        with open(summary_file, 'w') as f:
            f.write("HYPERPARAMETER TUNING SUMMARY REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Results Directory: {results_dir}\n")
            f.write(f"Number of Experiments: {len(results['experiments'])}\n\n")
            f.write("EXPERIMENT RANKINGS:\n")
            f.write(df.to_string(index=False))
            f.write("\n\nSANITY CHECK RESULTS:\n")
            for exp_name, sanity in sanity_results.items():
                issues = sanity.get('issues', [])
                if issues:
                    f.write(f"\n{exp_name}:\n")
                    for issue in issues:
                        f.write(f"  ⚠️  {issue}\n")
                else:
                    f.write(f"\n{exp_name}: ✅ No issues detected\n")
        
        print(f"Summary report saved to: {summary_file}")

if __name__ == "__main__":
    import sys
    
    # Check for experiment name argument
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
        # Try to find the experiment directory
        possible_dirs = [
            f"checkpoints/{experiment_name}",
            f"checkpoints/hex_ai_hyperparam_tuning_{experiment_name}",
            f"checkpoints/{experiment_name}_samples"
        ]
        
        results_dir = None
        for dir_path in possible_dirs:
            if Path(dir_path).exists():
                results_dir = dir_path
                break
        
        if results_dir is None:
            print(f"Error: Could not find results directory for experiment '{experiment_name}'")
            print("Available directories:")
            for path in Path("checkpoints").glob("*"):
                if path.is_dir():
                    print(f"  - {path}")
            sys.exit(1)
    else:
        print("No experiment name provided")
        print("Available experiments:")
        for path in Path("checkpoints").glob("*"):
            if path.is_dir():
                print(f"  - {path}")
        sys.exit(1)
    
    print(f"Analyzing experiment: {results_dir}")
    create_summary_report(results_dir) 