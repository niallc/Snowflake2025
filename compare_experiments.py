#!/usr/bin/env python3
"""
Compare small-scale vs large-scale hyperparameter tuning experiments.
"""

import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

POLICY_LOSS_WEIGHT = 0.14
VALUE_LOSS_WEIGHT = 0.86

def get_standardized_loss(row):
    return POLICY_LOSS_WEIGHT * row['policy_loss'] + VALUE_LOSS_WEIGHT * row['value_loss']

def load_experiment_results(results_dir: str):
    """Load experiment results from a directory."""
    results_file = Path(results_dir) / "experiment_summary.csv"
    if results_file.exists():
        return pd.read_csv(results_file)
    return None

def compare_experiments():
    """Compare small-scale vs large-scale experiments."""
    
    # Load results
    small_scale = load_experiment_results("checkpoints/scaled_tuning")
    large_scale = load_experiment_results("checkpoints/scaled_tuning_50k")
    
    if small_scale is None or large_scale is None:
        print("Error: Could not load experiment results")
        return
    
    print("="*80)
    print("COMPARISON: SMALL-SCALE vs LARGE-SCALE EXPERIMENTS")
    print("="*80)
    
    print(f"\nSmall-scale experiment (2,500 samples, 5 epochs):")
    print(f"  Best validation loss: {small_scale['best_val_loss'].min():.6f}")
    print(f"  Average validation loss: {small_scale['best_val_loss'].mean():.6f}")
    print(f"  Experiments with early stopping: {(small_scale['early_stopped'] == True).sum()}/{len(small_scale)}")
    
    print(f"\nLarge-scale experiment (50,000 samples, 10 epochs):")
    print(f"  Best validation loss: {large_scale['best_val_loss'].min():.6f}")
    print(f"  Average validation loss: {large_scale['best_val_loss'].mean():.6f}")
    print(f"  Experiments with early stopping: {(large_scale['early_stopped'] == True).sum()}/{len(large_scale)}")
    
    # Add standardized loss columns
    for df in [small_scale, large_scale]:
        if 'policy_loss' in df.columns and 'value_loss' in df.columns:
            df['standardized_loss'] = df.apply(get_standardized_loss, axis=1)
        else:
            df['standardized_loss'] = df['best_val_loss']  # fallback
    
    # Compare rankings
    print(f"\nRankings comparison (by standardized loss):")
    print(f"{'Experiment':<20} {'Small-scale':<16} {'Large-scale':<16} {'Difference':<12}")
    print("-" * 70)
    
    for exp_name in small_scale['experiment_name']:
        small_loss = small_scale[small_scale['experiment_name'] == exp_name]['standardized_loss'].iloc[0]
        large_loss = large_scale[large_scale['experiment_name'] == exp_name]['standardized_loss'].iloc[0]
        diff = large_loss - small_loss
        
        print(f"{exp_name:<20} {small_loss:<16.6f} {large_loss:<16.6f} {diff:<12.6f}")
    
    # Early stopping analysis
    print(f"\nEarly stopping analysis:")
    print(f"Small-scale: {(small_scale['early_stopped'] == True).sum()}/{len(small_scale)} experiments stopped early")
    print(f"Large-scale: {(large_scale['early_stopped'] == True).sum()}/{len(large_scale)} experiments stopped early")
    
    # Best configurations
    small_best = small_scale.loc[small_scale['best_val_loss'].idxmin()]
    large_best = large_scale.loc[large_scale['best_val_loss'].idxmin()]
    
    print(f"\nBest configurations:")
    print(f"Small-scale: {small_best['experiment_name']} (loss: {small_best['best_val_loss']:.6f})")
    print(f"Large-scale: {large_best['experiment_name']} (loss: {large_best['best_val_loss']:.6f})")
    
    # Create comparison plot (standardized loss)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    experiments = small_scale['experiment_name'].tolist()
    small_losses = small_scale['standardized_loss'].tolist()
    large_losses = large_scale['standardized_loss'].tolist()
    
    x = np.arange(len(experiments))
    width = 0.35
    
    ax1.bar(x - width/2, small_losses, width, label='Small-scale (2.5k samples)', alpha=0.8)
    ax1.bar(x + width/2, large_losses, width, label='Large-scale (50k samples)', alpha=0.8)
    
    ax1.set_xlabel('Experiment')
    ax1.set_ylabel('Standardized Loss')
    ax1.set_title('Standardized Loss Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(experiments, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Early stopping comparison
    small_early = (small_scale['early_stopped'] == True).sum()
    large_early = (large_scale['early_stopped'] == True).sum()
    
    ax2.bar(['Small-scale', 'Large-scale'], [small_early, large_early], 
            color=['lightblue', 'lightcoral'], alpha=0.8)
    ax2.set_ylabel('Number of Experiments with Early Stopping')
    ax2.set_title('Early Stopping Comparison')
    ax2.set_ylim(0, len(small_scale))
    
    # Add value labels on bars
    for i, v in enumerate([small_early, large_early]):
        ax2.text(i, v + 0.1, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('experiment_comparison_standardized.png', dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: experiment_comparison_standardized.png")
    plt.close(fig)

if __name__ == "__main__":
    compare_experiments() 