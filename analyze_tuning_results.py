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

def extract_losses(exp_data):
    # Helper to get train/val losses from all_metrics if present
    if 'all_metrics' in exp_data:
        metrics = exp_data['all_metrics']
        train_losses = metrics.get('train_losses', [])
        val_losses = metrics.get('val_losses', [])
        return train_losses, val_losses
    else:
        return exp_data.get('train_losses', []), exp_data.get('val_losses', [])

def plot_training_curves(experiments: Dict, save_path: str = None):
    """Plot training and validation loss curves for all experiments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(experiments)))
    
    for i, (exp_name, exp_data) in enumerate(experiments.items()):
        train_losses, val_losses = extract_losses(exp_data)
        if train_losses and val_losses:
            epochs = range(1, len(train_losses) + 1)
            
            # Training loss
            ax1.plot(epochs, train_losses, 
                    label=exp_name, color=colors[i], marker='o', linewidth=2)
            
            # Validation loss
            ax2.plot(epochs, val_losses, 
                    label=exp_name, color=colors[i], marker='s', linewidth=2)
    
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()

def sanity_check_losses(experiments: Dict) -> Dict:
    """Sanity check loss progression for each experiment."""
    sanity_results = {}
    
    for exp_name, exp_data in experiments.items():
        issues = []
        train_losses, val_losses = extract_losses(exp_data)
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
    
    # Sanity check
    sanity_results = sanity_check_losses(results['experiments'])
    
    # Create summary table
    summary_data = []
    for exp_name, exp_data in results['experiments'].items():
        sanity = sanity_results.get(exp_name, {})
        
        summary_data.append({
            'Experiment': exp_name,
            'Final Train Loss': sanity.get('final_train_loss', 'N/A'),
            'Final Val Loss': sanity.get('final_val_loss', 'N/A'),
            'Best Val Loss': sanity.get('best_val_loss', 'N/A'),
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
    
    # Plot training curves
    print("\n" + "="*80)
    print("GENERATING TRAINING CURVES")
    print("="*80)
    
    plot_save_path = Path(results_dir) / "training_curves.png" if save_path is None else save_path
    plot_training_curves(results['experiments'], str(plot_save_path))
    
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
    # Analyze the scaled tuning results
    results_dir = "checkpoints/scaled_tuning"
    
    if not Path(results_dir).exists():
        print(f"Results directory {results_dir} not found!")
        print("Available results directories:")
        for path in Path("checkpoints").glob("*"):
            if path.is_dir():
                print(f"  - {path}")
    else:
        create_summary_report(results_dir) 