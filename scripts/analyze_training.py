#!/usr/bin/env python3
"""
Training analysis script to diagnose performance issues.
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_training_results(results_path: str):
    """Analyze training results and identify potential issues."""
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print("=== Training Analysis ===")
    
    # Extract losses
    train_losses = results.get('train_losses', [])
    val_losses = results.get('val_losses', [])
    train_policy_losses = results.get('train_policy_losses', [])
    train_value_losses = results.get('train_value_losses', [])
    val_policy_losses = results.get('val_policy_losses', [])
    val_value_losses = results.get('val_value_losses', [])
    
    print(f"Epochs trained: {len(train_losses)}")
    print(f"Best validation loss: {min(val_losses) if val_losses else 'N/A'}")
    print(f"Final training loss: {train_losses[-1] if train_losses else 'N/A'}")
    
    # Check for overfitting
    if val_losses and len(val_losses) > 1:
        val_improvement = val_losses[0] - min(val_losses)
        print(f"Validation improvement: {val_improvement:.4f}")
        
        if val_losses[-1] > val_losses[-2]:
            print("⚠️  WARNING: Validation loss increased in final epoch (potential overfitting)")
    
    # Check learning rate behavior
    if 'learning_rates' in results:
        lrs = results['learning_rates']
        print(f"Learning rate range: {min(lrs):.6f} to {max(lrs):.6f}")
        if lrs[-1] == lrs[0]:
            print("⚠️  WARNING: Learning rate never decreased")
    
    # Analyze policy vs value loss
    if train_policy_losses and train_value_losses:
        print(f"\nPolicy Loss Analysis:")
        print(f"  Initial: {train_policy_losses[0]:.4f}")
        print(f"  Final: {train_policy_losses[-1]:.4f}")
        print(f"  Improvement: {train_policy_losses[0] - train_policy_losses[-1]:.4f}")
        
        print(f"\nValue Loss Analysis:")
        print(f"  Initial: {train_value_losses[0]:.4f}")
        print(f"  Final: {train_value_losses[-1]:.4f}")
        print(f"  Improvement: {train_value_losses[0] - train_value_losses[-1]:.4f}")
        
        # Check if value loss is stuck
        if len(train_value_losses) > 3:
            recent_value_losses = train_value_losses[-3:]
            if max(recent_value_losses) - min(recent_value_losses) < 0.01:
                print("⚠️  WARNING: Value loss appears stuck (minimal variation in last 3 epochs)")
    
    # Check for gradient issues
    if 'gradient_norms' in results:
        grad_norms = results['gradient_norms']
        print(f"\nGradient Analysis:")
        print(f"  Average gradient norm: {np.mean(grad_norms):.4f}")
        print(f"  Max gradient norm: {np.max(grad_norms):.4f}")
        print(f"  Min gradient norm: {np.min(grad_norms):.4f}")
        
        if np.max(grad_norms) > 10:
            print("⚠️  WARNING: Very large gradient norms detected")
        if np.min(grad_norms) < 0.001:
            print("⚠️  WARNING: Very small gradient norms detected (vanishing gradients)")
    
    # Check training time
    if 'epoch_times' in results:
        epoch_times = results['epoch_times']
        print(f"\nTiming Analysis:")
        print(f"  Average epoch time: {np.mean(epoch_times):.1f}s")
        print(f"  Total training time: {sum(epoch_times):.1f}s")
    
    # Recommendations
    print(f"\n=== Recommendations ===")
    
    if train_losses and train_losses[-1] > 4.0:
        print("🔧 Consider reducing learning rate (loss seems high)")
    
    if val_losses and len(val_losses) > 3:
        if val_losses[-1] > val_losses[-3]:
            print("🔧 Consider early stopping or reducing model complexity")
    
    if train_policy_losses and train_policy_losses[-1] > 4.5:
        print("🔧 Policy loss is high - check data quality or model architecture")
    
    if train_value_losses and train_value_losses[-1] > 0.3:
        print("🔧 Value loss is high - check value target distribution")
    
    return results

def plot_training_curves(results_path: str, save_plot: bool = False):
    """Plot training curves for visual analysis."""
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Training losses
    if 'train_losses' in results and 'val_losses' in results:
        axes[0, 0].plot(results['train_losses'], label='Train')
        axes[0, 0].plot(results['val_losses'], label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Policy losses
    if 'train_policy_losses' in results and 'val_policy_losses' in results:
        axes[0, 1].plot(results['train_policy_losses'], label='Train')
        axes[0, 1].plot(results['val_policy_losses'], label='Validation')
        axes[0, 1].set_title('Policy Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Value losses
    if 'train_value_losses' in results and 'val_value_losses' in results:
        axes[1, 0].plot(results['train_value_losses'], label='Train')
        axes[1, 0].plot(results['val_value_losses'], label='Validation')
        axes[1, 0].set_title('Value Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Learning rate
    if 'learning_rates' in results:
        axes[1, 1].plot(results['learning_rates'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_plot:
        plot_path = Path(results_path).parent / "training_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze training results")
    parser.add_argument('results_path', help='Path to training_results.json')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--save-plot', action='store_true', help='Save plot to file')
    
    args = parser.parse_args()
    
    results = analyze_training_results(args.results_path)
    
    if args.plot or args.save_plot:
        plot_training_curves(args.results_path, save_plot=args.save_plot) 