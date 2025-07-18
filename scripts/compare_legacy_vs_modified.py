#!/usr/bin/env python3
"""
Compare performance between legacy and modified versions.

This script helps analyze the results from different training runs
to identify which changes cause performance regressions.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import matplotlib.pyplot as plt
import numpy as np

def load_training_metrics(checkpoint_dir: Path) -> Optional[pd.DataFrame]:
    """Load training metrics from a checkpoint directory."""
    metrics_file = checkpoint_dir / "training_metrics.csv"
    if metrics_file.exists():
        return pd.read_csv(metrics_file)
    return None

def load_config(checkpoint_dir: Path) -> Optional[Dict]:
    """Load configuration from a checkpoint directory."""
    config_file = checkpoint_dir / "config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            return json.load(f)
    return None

def analyze_training_run(checkpoint_dir: Path) -> Dict:
    """Analyze a single training run."""
    metrics_df = load_training_metrics(checkpoint_dir)
    config = load_config(checkpoint_dir)
    
    if metrics_df is None:
        return {
            'checkpoint_dir': str(checkpoint_dir),
            'status': 'no_metrics',
            'error': 'No training_metrics.csv found'
        }
    
    # Calculate key metrics
    final_epoch = metrics_df.iloc[-1]
    best_epoch = metrics_df.loc[metrics_df['val_loss'].idxmin()]
    
    return {
        'checkpoint_dir': str(checkpoint_dir),
        'status': 'success',
        'config': config,
        'final_epoch': final_epoch.to_dict(),
        'best_epoch': best_epoch.to_dict(),
        'total_epochs': len(metrics_df),
        'final_policy_loss': final_epoch.get('policy_loss', None),
        'final_value_loss': final_epoch.get('value_loss', None),
        'final_total_loss': final_epoch.get('total_loss', None),
        'best_policy_loss': best_epoch.get('policy_loss', None),
        'best_value_loss': best_epoch.get('value_loss', None),
        'best_total_loss': best_epoch.get('total_loss', None),
        'policy_loss_improvement': metrics_df['policy_loss'].iloc[0] - metrics_df['policy_loss'].iloc[-1] if len(metrics_df) > 1 else 0,
        'value_loss_improvement': metrics_df['value_loss'].iloc[0] - metrics_df['value_loss'].iloc[-1] if len(metrics_df) > 1 else 0
    }

def compare_runs(legacy_dir: Path, modified_dir: Path, output_dir: Optional[Path] = None) -> Dict:
    """Compare two training runs."""
    print(f"Comparing runs:")
    print(f"  Legacy: {legacy_dir}")
    print(f"  Modified: {modified_dir}")
    
    legacy_analysis = analyze_training_run(legacy_dir)
    modified_analysis = analyze_training_run(modified_dir)
    
    if legacy_analysis['status'] != 'success':
        print(f"Error analyzing legacy run: {legacy_analysis['error']}")
        return {'error': 'Legacy analysis failed'}
    
    if modified_analysis['status'] != 'success':
        print(f"Error analyzing modified run: {modified_analysis['error']}")
        return {'error': 'Modified analysis failed'}
    
    # Calculate differences
    comparison = {
        'legacy': legacy_analysis,
        'modified': modified_analysis,
        'differences': {
            'final_policy_loss_diff': modified_analysis['final_policy_loss'] - legacy_analysis['final_policy_loss'],
            'final_value_loss_diff': modified_analysis['final_value_loss'] - legacy_analysis['final_value_loss'],
            'final_total_loss_diff': modified_analysis['final_total_loss'] - legacy_analysis['final_total_loss'],
            'best_policy_loss_diff': modified_analysis['best_policy_loss'] - legacy_analysis['best_policy_loss'],
            'best_value_loss_diff': modified_analysis['best_value_loss'] - legacy_analysis['best_value_loss'],
            'best_total_loss_diff': modified_analysis['best_total_loss'] - legacy_analysis['best_total_loss'],
            'policy_loss_improvement_diff': modified_analysis['policy_loss_improvement'] - legacy_analysis['policy_loss_improvement'],
            'value_loss_improvement_diff': modified_analysis['value_loss_improvement'] - legacy_analysis['value_loss_improvement']
        }
    }
    
    # Print comparison
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    print(f"\nFinal Policy Loss:")
    print(f"  Legacy: {legacy_analysis['final_policy_loss']:.6f}")
    print(f"  Modified: {modified_analysis['final_policy_loss']:.6f}")
    print(f"  Difference: {comparison['differences']['final_policy_loss_diff']:+.6f}")
    
    print(f"\nFinal Value Loss:")
    print(f"  Legacy: {legacy_analysis['final_value_loss']:.6f}")
    print(f"  Modified: {modified_analysis['final_value_loss']:.6f}")
    print(f"  Difference: {comparison['differences']['final_value_loss_diff']:+.6f}")
    
    print(f"\nFinal Total Loss:")
    print(f"  Legacy: {legacy_analysis['final_total_loss']:.6f}")
    print(f"  Modified: {modified_analysis['final_total_loss']:.6f}")
    print(f"  Difference: {comparison['differences']['final_total_loss_diff']:+.6f}")
    
    print(f"\nPolicy Loss Improvement:")
    print(f"  Legacy: {legacy_analysis['policy_loss_improvement']:.6f}")
    print(f"  Modified: {modified_analysis['policy_loss_improvement']:.6f}")
    print(f"  Difference: {comparison['differences']['policy_loss_improvement_diff']:+.6f}")
    
    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    
    policy_diff = comparison['differences']['final_policy_loss_diff']
    if policy_diff > 0.5:
        print(f"❌ SIGNIFICANT REGRESSION: Policy loss is {policy_diff:.3f} higher")
        print(f"   This suggests the modification significantly hurts policy learning")
    elif policy_diff > 0.1:
        print(f"⚠️  MODERATE REGRESSION: Policy loss is {policy_diff:.3f} higher")
        print(f"   This suggests the modification may hurt policy learning")
    elif policy_diff < -0.1:
        print(f"✅ IMPROVEMENT: Policy loss is {abs(policy_diff):.3f} lower")
        print(f"   This suggests the modification helps policy learning")
    else:
        print(f"✅ NO SIGNIFICANT CHANGE: Policy loss difference is {abs(policy_diff):.3f}")
        print(f"   This suggests the modification doesn't affect policy learning")
    
    # Save comparison
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        comparison_file = output_dir / "comparison_results.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to: {comparison_file}")
    
    return comparison

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Compare legacy vs modified training runs')
    parser.add_argument('--legacy-dir', type=str, required=True,
                       help='Path to legacy training results directory')
    parser.add_argument('--modified-dir', type=str, required=True,
                       help='Path to modified training results directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    legacy_dir = Path(args.legacy_dir)
    modified_dir = Path(args.modified_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    if not legacy_dir.exists():
        print(f"Error: Legacy directory {legacy_dir} does not exist")
        return
    
    if not modified_dir.exists():
        print(f"Error: Modified directory {modified_dir} does not exist")
        return
    
    comparison = compare_runs(legacy_dir, modified_dir, output_dir)
    
    if 'error' in comparison:
        print(f"Comparison failed: {comparison['error']}")

if __name__ == '__main__':
    main() 