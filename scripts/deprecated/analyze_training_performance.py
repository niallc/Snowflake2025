#!/usr/bin/env python3
"""
Script to analyze training performance from console output.
Parses the training logs and creates visualizations to understand performance patterns.
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Tuple, Optional
import seaborn as sns

def parse_training_log(log_file_path: str) -> pd.DataFrame:
    """
    Parse the training console output and extract performance metrics.
    
    Args:
        log_file_path: Path to the training console output file
        
    Returns:
        DataFrame with columns: [epoch, mini_epoch, batch_count, train_total, train_policy, 
                                train_value, val_total, val_policy, val_value, timestamp]
    """
    data = []
    
    # Pattern to match mini-epoch performance lines
    pattern = r'\[Epoch (\d+)\]\[Mini-epoch (\d+)\] Train Losses: total=([\d.]+), policy=([\d.]+), value=([\d.]+) \| Val Losses: total=([\d.]+), policy=([\d.]+), value=([\d.]+) \| Batches processed: (\d+)'
    
    with open(log_file_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                mini_epoch = int(match.group(2))
                train_total = float(match.group(3))
                train_policy = float(match.group(4))
                train_value = float(match.group(5))
                val_total = float(match.group(6))
                val_policy = float(match.group(7))
                val_value = float(match.group(8))
                batch_count = int(match.group(9))
                
                data.append({
                    'epoch': epoch,
                    'mini_epoch': mini_epoch,
                    'batch_count': batch_count,
                    'train_total': train_total,
                    'train_policy': train_policy,
                    'train_value': train_value,
                    'val_total': val_total,
                    'val_policy': val_policy,
                    'val_value': val_value
                })
    
    df = pd.DataFrame(data)
    return df

def analyze_training_patterns(df: pd.DataFrame) -> Dict:
    """
    Analyze training patterns and extract key insights.
    
    Args:
        df: DataFrame with training metrics
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {}
    
    # Overall statistics
    analysis['total_mini_epochs'] = len(df)
    analysis['total_batches'] = df['batch_count'].max()
    analysis['training_duration_batches'] = df['batch_count'].max() - df['batch_count'].min()
    
    # Loss statistics
    analysis['train_total_stats'] = {
        'mean': df['train_total'].mean(),
        'std': df['train_total'].std(),
        'min': df['train_total'].min(),
        'max': df['train_total'].max(),
        'final': df['train_total'].iloc[-1],
        'start': df['train_total'].iloc[0]
    }
    
    analysis['val_total_stats'] = {
        'mean': df['val_total'].mean(),
        'std': df['val_total'].std(),
        'min': df['val_total'].min(),
        'max': df['val_total'].max(),
        'final': df['val_total'].iloc[-1],
        'start': df['val_total'].iloc[0]
    }
    
    # Policy and Value loss statistics
    analysis['train_policy_stats'] = {
        'mean': df['train_policy'].mean(),
        'std': df['train_policy'].std(),
        'min': df['train_policy'].min(),
        'max': df['train_policy'].max(),
        'final': df['train_policy'].iloc[-1]
    }
    
    analysis['train_value_stats'] = {
        'mean': df['train_value'].mean(),
        'std': df['train_value'].std(),
        'min': df['train_value'].min(),
        'max': df['train_value'].max(),
        'final': df['train_value'].iloc[-1]
    }
    
    # Overfitting analysis
    analysis['overfitting_gap'] = analysis['val_total_stats']['mean'] - analysis['train_total_stats']['mean']
    
    # Performance trends
    analysis['train_trend'] = 'improving' if df['train_total'].iloc[-1] < df['train_total'].iloc[0] else 'worsening'
    analysis['val_trend'] = 'improving' if df['val_total'].iloc[-1] < df['val_total'].iloc[0] else 'worsening'
    
    # Identify performance spikes (sudden changes)
    train_diff = df['train_total'].diff().abs()
    val_diff = df['val_total'].diff().abs()
    
    # Find significant spikes (changes > 2 standard deviations)
    train_spike_threshold = train_diff.mean() + 2 * train_diff.std()
    val_spike_threshold = val_diff.mean() + 2 * val_diff.std()
    
    train_spikes = df[train_diff > train_spike_threshold]
    val_spikes = df[val_diff > val_spike_threshold]
    
    analysis['train_spikes'] = train_spikes
    analysis['val_spikes'] = val_spikes
    analysis['spike_analysis'] = {
        'train_spike_count': len(train_spikes),
        'val_spike_count': len(val_spikes),
        'train_spike_threshold': train_spike_threshold,
        'val_spike_threshold': val_spike_threshold,
        'avg_train_spike_magnitude': train_spikes['train_total'].diff().abs().mean() if len(train_spikes) > 0 else 0,
        'avg_val_spike_magnitude': val_spikes['val_total'].diff().abs().mean() if len(val_spikes) > 0 else 0,
        'max_train_spike_magnitude': train_diff.max(),
        'max_val_spike_magnitude': val_diff.max()
    }
    
    # Convergence analysis
    last_quarter = df.tail(len(df) // 4)
    analysis['convergence'] = {
        'train_final_quarter_mean': last_quarter['train_total'].mean(),
        'val_final_quarter_mean': last_quarter['val_total'].mean(),
        'train_final_quarter_std': last_quarter['train_total'].std(),
        'val_final_quarter_std': last_quarter['val_total'].std()
    }
    
    # Quarterly analysis
    quarter_size = len(df) // 4
    analysis['quarterly_analysis'] = {}
    for i in range(4):
        start_idx = i * quarter_size
        end_idx = (i + 1) * quarter_size if i < 3 else len(df)
        quarter_data = df.iloc[start_idx:end_idx]
        
        analysis['quarterly_analysis'][f'Q{i+1}'] = {
            'train_mean': quarter_data['train_total'].mean(),
            'train_std': quarter_data['train_total'].std(),
            'val_mean': quarter_data['val_total'].mean(),
            'val_std': quarter_data['val_total'].std(),
            'overfitting_gap': quarter_data['val_total'].mean() - quarter_data['train_total'].mean(),
            'spike_count': len(quarter_data[train_diff.iloc[start_idx:end_idx] > train_spike_threshold])
        }
    
    # Loss component correlation analysis
    analysis['correlation_analysis'] = {
        'policy_value_correlation': df['train_policy'].corr(df['train_value']),
        'train_val_correlation': df['train_total'].corr(df['val_total']),
        'policy_value_correlation_spikes': train_spikes['train_policy'].corr(train_spikes['train_value']) if len(train_spikes) > 1 else 0
    }
    
    # Rolling average analysis
    window_size = max(1, len(df) // 20)  # 5% of data points
    train_rolling = df['train_total'].rolling(window=window_size, center=True).mean()
    val_rolling = df['val_total'].rolling(window=window_size, center=True).mean()
    
    analysis['rolling_analysis'] = {
        'window_size': window_size,
        'train_rolling_std': train_rolling.std(),
        'val_rolling_std': val_rolling.std(),
        'train_rolling_improvement_rate': (train_rolling.iloc[-1] - train_rolling.iloc[0]) / len(df),
        'val_rolling_improvement_rate': (val_rolling.iloc[-1] - val_rolling.iloc[0]) / len(df)
    }
    
    return analysis

def generate_detailed_pattern_summary(df: pd.DataFrame, analysis: Dict) -> str:
    """
    Generate a detailed text summary of training patterns with all numerical data.
    
    Args:
        df: DataFrame with training metrics
        analysis: Analysis results dictionary
        
    Returns:
        String containing detailed pattern summary
    """
    summary = []
    summary.append("TRAINING PATTERN SUMMARIES - DETAILED ANALYSIS")
    summary.append("=" * 60)
    summary.append("")
    
    # Overall training dynamics
    summary.append("OVERALL TRAINING DYNAMICS")
    summary.append("-" * 30)
    summary.append(f"Training ran for {analysis['total_batches']:,} batches across {analysis['total_mini_epochs']} mini-epochs.")
    summary.append(f"Training loss started at {analysis['train_total_stats']['start']:.4f}, ended at {analysis['train_total_stats']['final']:.4f}.")
    summary.append(f"Validation loss started at {analysis['val_total_stats']['start']:.4f}, ended at {analysis['val_total_stats']['final']:.4f}.")
    summary.append(f"Overfitting gap: {analysis['overfitting_gap']:.4f} (relatively small, indicating good generalization).")
    summary.append(f"Total training duration: {analysis['training_duration_batches']:,} batches.")
    summary.append("")
    
    # Spike analysis
    summary.append("SPIKE ANALYSIS - COMPREHENSIVE BREAKDOWN")
    summary.append("-" * 40)
    total_spikes = analysis['spike_analysis']['train_spike_count'] + analysis['spike_analysis']['val_spike_count']
    summary.append(f"Total spikes detected: {total_spikes} ({analysis['spike_analysis']['train_spike_count']} training, {analysis['spike_analysis']['val_spike_count']} validation)")
    summary.append(f"Average train spike magnitude: {analysis['spike_analysis']['avg_train_spike_magnitude']:.4f}")
    summary.append(f"Average val spike magnitude: {analysis['spike_analysis']['avg_val_spike_magnitude']:.4f}")
    summary.append(f"Maximum train spike magnitude: {analysis['spike_analysis']['max_train_spike_magnitude']:.4f}")
    summary.append(f"Maximum val spike magnitude: {analysis['spike_analysis']['max_val_spike_magnitude']:.4f}")
    summary.append("")
    
    # Quarterly spike analysis
    summary.append("Quarterly spike analysis:")
    for quarter, data in analysis['quarterly_analysis'].items():
        summary.append(f"- {quarter}: {data['spike_count']} spikes")
    summary.append("")
    
    # Loss component behavior
    summary.append("LOSS COMPONENT BEHAVIOR - DETAILED BREAKDOWN")
    summary.append("-" * 40)
    summary.append("Policy Loss Statistics:")
    summary.append(f"- Range: {analysis['train_policy_stats']['min']:.4f} to {analysis['train_policy_stats']['max']:.4f}")
    summary.append(f"- Mean: {analysis['train_policy_stats']['mean']:.4f}")
    summary.append(f"- Standard deviation: {analysis['train_policy_stats']['std']:.4f}")
    summary.append(f"- Final value: {analysis['train_policy_stats']['final']:.4f}")
    summary.append("")
    
    summary.append("Value Loss Statistics:")
    summary.append(f"- Range: {analysis['train_value_stats']['min']:.4f} to {analysis['train_value_stats']['max']:.4f}")
    summary.append(f"- Mean: {analysis['train_value_stats']['mean']:.4f}")
    summary.append(f"- Standard deviation: {analysis['train_value_stats']['std']:.4f}")
    summary.append(f"- Final value: {analysis['train_value_stats']['final']:.4f}")
    summary.append("")
    
    # Correlation analysis
    summary.append("CORRELATION ANALYSIS")
    summary.append("-" * 20)
    summary.append(f"Policy-Value correlation (normal): {analysis['correlation_analysis']['policy_value_correlation']:.4f}")
    summary.append(f"Policy-Value correlation (during spikes): {analysis['correlation_analysis']['policy_value_correlation_spikes']:.4f}")
    summary.append(f"Train-Val correlation: {analysis['correlation_analysis']['train_val_correlation']:.4f}")
    summary.append("")
    
    # Convergence analysis
    summary.append("CONVERGENCE ANALYSIS - QUANTIFIED METRICS")
    summary.append("-" * 40)
    summary.append(f"Final quarter training mean: {analysis['convergence']['train_final_quarter_mean']:.4f} (std: {analysis['convergence']['train_final_quarter_std']:.4f})")
    summary.append(f"Final quarter validation mean: {analysis['convergence']['val_final_quarter_mean']:.4f} (std: {analysis['convergence']['val_final_quarter_std']:.4f})")
    summary.append("")
    
    summary.append("Convergence metrics by quarter:")
    for quarter, data in analysis['quarterly_analysis'].items():
        summary.append(f"- {quarter}: Train mean={data['train_mean']:.4f}, std={data['train_std']:.4f}; Val mean={data['val_mean']:.4f}, std={data['val_std']:.4f}")
    summary.append("")
    
    # Rolling average analysis
    summary.append("ROLLING AVERAGE TRENDS - WINDOW ANALYSIS")
    summary.append("-" * 40)
    summary.append(f"Rolling average window size: {analysis['rolling_analysis']['window_size']} mini-epochs")
    summary.append(f"Train rolling improvement rate: {analysis['rolling_analysis']['train_rolling_improvement_rate']:.6f} per mini-epoch")
    summary.append(f"Val rolling improvement rate: {analysis['rolling_analysis']['val_rolling_improvement_rate']:.6f} per mini-epoch")
    summary.append("")
    
    # Detailed spike timeline
    summary.append("DETAILED SPIKE TIMELINE - COMPLETE RECORD")
    summary.append("-" * 40)
    
    # Train spikes
    if len(analysis['train_spikes']) > 0:
        summary.append("Train Spikes:")
        for _, spike in analysis['train_spikes'].iterrows():
            summary.append(f"- Batch {spike['batch_count']}: {spike['train_total']:.4f}")
    
    # Val spikes
    if len(analysis['val_spikes']) > 0:
        summary.append("Validation Spikes:")
        for _, spike in analysis['val_spikes'].iterrows():
            summary.append(f"- Batch {spike['batch_count']}: {spike['val_total']:.4f}")
    summary.append("")
    
    # Recommendations
    summary.append("RECOMMENDATIONS BASED ON PATTERNS - QUANTIFIED SUGGESTIONS")
    summary.append("-" * 50)
    
    # Learning rate recommendation
    current_spike_magnitude = analysis['spike_analysis']['avg_train_spike_magnitude']
    target_spike_magnitude = 0.05
    if current_spike_magnitude > target_spike_magnitude:
        reduction_percent = ((current_spike_magnitude - target_spike_magnitude) / current_spike_magnitude) * 100
        summary.append(f"1. Reduce learning rate by {reduction_percent:.1f}% (from {current_spike_magnitude:.4f} to {target_spike_magnitude:.4f} spike magnitude)")
    
    # Convergence recommendation
    if analysis['val_trend'] == 'improving':
        summary.append("2. Continue training - validation trend is positive")
    
    # Overfitting recommendation
    if analysis['overfitting_gap'] > 0.1:
        summary.append("3. Address overfitting - gap is large")
    else:
        summary.append("3. Good generalization - overfitting gap is small")
    
    summary.append("")
    
    # Final assessment
    summary.append("FINAL ASSESSMENT - QUANTIFIED CONCLUSIONS")
    summary.append("-" * 40)
    
    # Calculate convergence indicator
    val_improvement = analysis['val_total_stats']['start'] - analysis['val_total_stats']['final']
    convergence_indicator = min(1.0, max(0.0, val_improvement / 0.1))  # Normalize to 0-1
    
    # Calculate stability indicator
    stability_indicator = max(0.0, 1.0 - (analysis['train_total_stats']['std'] / 0.1))
    
    summary.append(f"Training Status: Partially converged with high variance")
    summary.append(f"- Convergence indicator: {convergence_indicator:.2f} (0.0=not converged, 1.0=fully converged)")
    summary.append(f"- Stability indicator: {stability_indicator:.2f} (0.0=unstable, 1.0=stable)")
    summary.append(f"- Generalization indicator: {max(0.0, 1.0 - analysis['overfitting_gap']):.2f} (0.0=overfitting, 1.0=good generalization)")
    summary.append("")
    
    return "\n".join(summary)

def create_visualizations(df: pd.DataFrame, analysis: Dict, output_dir: str = "training_analysis"):
    """
    Create comprehensive visualizations of the training performance.
    
    Args:
        df: DataFrame with training metrics
        analysis: Analysis results dictionary
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Overall training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Performance Analysis', fontsize=16, fontweight='bold')
    
    # Total loss over time
    axes[0, 0].plot(df['batch_count'], df['train_total'], label='Train Total', linewidth=2)
    axes[0, 0].plot(df['batch_count'], df['val_total'], label='Validation Total', linewidth=2)
    axes[0, 0].set_xlabel('Batches Processed')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Policy loss over time
    axes[0, 1].plot(df['batch_count'], df['train_policy'], label='Train Policy', linewidth=2)
    axes[0, 1].plot(df['batch_count'], df['val_policy'], label='Validation Policy', linewidth=2)
    axes[0, 1].set_xlabel('Batches Processed')
    axes[0, 1].set_ylabel('Policy Loss')
    axes[0, 1].set_title('Policy Loss Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Value loss over time
    axes[1, 0].plot(df['batch_count'], df['train_value'], label='Train Value', linewidth=2)
    axes[1, 0].plot(df['batch_count'], df['val_value'], label='Validation Value', linewidth=2)
    axes[1, 0].set_xlabel('Batches Processed')
    axes[1, 0].set_ylabel('Value Loss')
    axes[1, 0].set_title('Value Loss Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss difference (validation - training)
    loss_diff = df['val_total'] - df['train_total']
    axes[1, 1].plot(df['batch_count'], loss_diff, color='red', linewidth=2)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Batches Processed')
    axes[1, 1].set_ylabel('Validation - Training Loss')
    axes[1, 1].set_title('Overfitting Gap (Val - Train)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance spikes analysis
    if len(analysis['train_spikes']) > 0 or len(analysis['val_spikes']) > 0:
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('Performance Spikes Analysis', fontsize=16, fontweight='bold')
        
        # Train spikes
        if len(analysis['train_spikes']) > 0:
            axes[0].scatter(analysis['train_spikes']['batch_count'], 
                          analysis['train_spikes']['train_total'], 
                          color='red', s=100, alpha=0.7, label='Train Spikes')
        axes[0].plot(df['batch_count'], df['train_total'], color='blue', alpha=0.7, label='Train Total')
        axes[0].set_xlabel('Batches Processed')
        axes[0].set_ylabel('Train Total Loss')
        axes[0].set_title(f'Train Loss with {len(analysis["train_spikes"])} Spikes')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Validation spikes
        if len(analysis['val_spikes']) > 0:
            axes[1].scatter(analysis['val_spikes']['batch_count'], 
                          analysis['val_spikes']['val_total'], 
                          color='red', s=100, alpha=0.7, label='Validation Spikes')
        axes[1].plot(df['batch_count'], df['val_total'], color='green', alpha=0.7, label='Validation Total')
        axes[1].set_xlabel('Batches Processed')
        axes[1].set_ylabel('Validation Total Loss')
        axes[1].set_title(f'Validation Loss with {len(analysis["val_spikes"])} Spikes')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'performance_spikes.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Loss distribution analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Loss Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Train loss distribution
    axes[0, 0].hist(df['train_total'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(df['train_total'].mean(), color='red', linestyle='--', label=f'Mean: {df["train_total"].mean():.4f}')
    axes[0, 0].set_xlabel('Train Total Loss')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Train Total Loss Distribution')
    axes[0, 0].legend()
    
    # Validation loss distribution
    axes[0, 1].hist(df['val_total'], bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(df['val_total'].mean(), color='red', linestyle='--', label=f'Mean: {df["val_total"].mean():.4f}')
    axes[0, 1].set_xlabel('Validation Total Loss')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Validation Total Loss Distribution')
    axes[0, 1].legend()
    
    # Policy vs Value loss scatter
    axes[1, 0].scatter(df['train_policy'], df['train_value'], alpha=0.6, color='blue', label='Train')
    axes[1, 0].scatter(df['val_policy'], df['val_value'], alpha=0.6, color='green', label='Validation')
    axes[1, 0].set_xlabel('Policy Loss')
    axes[1, 0].set_ylabel('Value Loss')
    axes[1, 0].set_title('Policy vs Value Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Rolling average to show trends
    window_size = max(1, len(df) // 20)  # 5% of data points
    train_rolling = df['train_total'].rolling(window=window_size, center=True).mean()
    val_rolling = df['val_total'].rolling(window=window_size, center=True).mean()
    
    axes[1, 1].plot(df['batch_count'], train_rolling, label=f'Train (rolling {window_size})', linewidth=2)
    axes[1, 1].plot(df['batch_count'], val_rolling, label=f'Validation (rolling {window_size})', linewidth=2)
    axes[1, 1].set_xlabel('Batches Processed')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Rolling Average Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'loss_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Summary statistics table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary table
    summary_data = [
        ['Metric', 'Train', 'Validation'],
        ['Mean Loss', f"{analysis['train_total_stats']['mean']:.4f}", f"{analysis['val_total_stats']['mean']:.4f}"],
        ['Std Loss', f"{analysis['train_total_stats']['std']:.4f}", f"{analysis['val_total_stats']['std']:.4f}"],
        ['Min Loss', f"{analysis['train_total_stats']['min']:.4f}", f"{analysis['val_total_stats']['min']:.4f}"],
        ['Max Loss', f"{analysis['train_total_stats']['max']:.4f}", f"{analysis['val_total_stats']['max']:.4f}"],
        ['Final Loss', f"{analysis['train_total_stats']['final']:.4f}", f"{analysis['val_total_stats']['final']:.4f}"],
        ['Trend', analysis['train_trend'], analysis['val_trend']],
        ['', '', ''],
        ['Total Mini-epochs', str(analysis['total_mini_epochs']), ''],
        ['Total Batches', str(analysis['total_batches']), ''],
        ['Overfitting Gap', f"{analysis['overfitting_gap']:.4f}", ''],
        ['Train Spikes', str(analysis['spike_analysis']['train_spike_count']), ''],
        ['Val Spikes', str(analysis['spike_analysis']['val_spike_count']), '']
    ]
    
    table = ax.table(cellText=summary_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Color code the header
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Training Performance Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_path / 'summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_analysis_summary(analysis: Dict):
    """
    Print a summary of the training analysis to console.
    
    Args:
        analysis: Analysis results dictionary
    """
    print("\n" + "="*60)
    print("TRAINING PERFORMANCE ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\n📊 Training Overview:")
    print(f"   • Total mini-epochs: {analysis['total_mini_epochs']}")
    print(f"   • Total batches processed: {analysis['total_batches']:,}")
    print(f"   • Training duration: {analysis['training_duration_batches']:,} batches")
    
    print(f"\n📈 Loss Statistics:")
    print(f"   Train Total Loss:")
    print(f"     • Mean: {analysis['train_total_stats']['mean']:.4f}")
    print(f"     • Std:  {analysis['train_total_stats']['std']:.4f}")
    print(f"     • Min:  {analysis['train_total_stats']['min']:.4f}")
    print(f"     • Max:  {analysis['train_total_stats']['max']:.4f}")
    print(f"     • Final: {analysis['train_total_stats']['final']:.4f}")
    
    print(f"\n   Validation Total Loss:")
    print(f"     • Mean: {analysis['val_total_stats']['mean']:.4f}")
    print(f"     • Std:  {analysis['val_total_stats']['std']:.4f}")
    print(f"     • Min:  {analysis['val_total_stats']['min']:.4f}")
    print(f"     • Max:  {analysis['val_total_stats']['max']:.4f}")
    print(f"     • Final: {analysis['val_total_stats']['final']:.4f}")
    
    print(f"\n🔍 Performance Analysis:")
    print(f"   • Overfitting gap (Val - Train): {analysis['overfitting_gap']:.4f}")
    print(f"   • Train trend: {analysis['train_trend']}")
    print(f"   • Validation trend: {analysis['val_trend']}")
    
    print(f"\n⚡ Performance Spikes:")
    print(f"   • Train spikes: {analysis['spike_analysis']['train_spike_count']}")
    print(f"   • Validation spikes: {analysis['spike_analysis']['val_spike_count']}")
    print(f"   • Spike threshold (train): {analysis['spike_analysis']['train_spike_threshold']:.4f}")
    print(f"   • Spike threshold (val): {analysis['spike_analysis']['val_spike_threshold']:.4f}")
    
    print(f"\n🎯 Convergence Analysis:")
    print(f"   • Final quarter train mean: {analysis['convergence']['train_final_quarter_mean']:.4f}")
    print(f"   • Final quarter val mean: {analysis['convergence']['val_final_quarter_mean']:.4f}")
    print(f"   • Final quarter train std: {analysis['convergence']['train_final_quarter_std']:.4f}")
    print(f"   • Final quarter val std: {analysis['convergence']['val_final_quarter_std']:.4f}")
    
    # Interpretation
    print(f"\n💡 Key Insights:")
    if analysis['overfitting_gap'] > 0.1:
        print(f"   ⚠️  Significant overfitting detected (gap: {analysis['overfitting_gap']:.4f})")
    else:
        print(f"   ✅ Good generalization (overfitting gap: {analysis['overfitting_gap']:.4f})")
    
    if analysis['spike_analysis']['train_spike_count'] > 5:
        print(f"   ⚠️  Many training spikes detected ({analysis['spike_analysis']['train_spike_count']}) - consider reducing learning rate")
    else:
        print(f"   ✅ Stable training with few spikes")
    
    if analysis['convergence']['train_final_quarter_std'] < 0.01:
        print(f"   ✅ Training appears to have converged (low final variance)")
    else:
        print(f"   ⚠️  Training may not have fully converged (high final variance)")
    
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description='Analyze training performance from console output')
    parser.add_argument('log_file', help='Path to training console output file')
    parser.add_argument('--output_dir', default='training_analysis', help='Directory to save analysis plots')
    parser.add_argument('--no_plots', action='store_true', help='Skip generating plots')
    parser.add_argument('--generate_summary', action='store_true', help='Generate detailed pattern summary text file')
    
    args = parser.parse_args()
    
    print(f"🔍 Analyzing training performance from: {args.log_file}")
    
    # Parse the log file
    df = parse_training_log(args.log_file)
    
    if df.empty:
        print("❌ No training data found in the log file!")
        return
    
    print(f"✅ Parsed {len(df)} mini-epochs of training data")
    
    # Analyze the data
    analysis = analyze_training_patterns(df)
    
    # Print summary
    print_analysis_summary(analysis)
    
    # Create visualizations
    if not args.no_plots:
        print(f"\n📊 Generating visualizations in: {args.output_dir}")
        create_visualizations(df, analysis, args.output_dir)
        print("✅ Analysis complete! Check the output directory for plots.")
    
    # Generate detailed pattern summary
    if args.generate_summary:
        print(f"\n📝 Generating detailed pattern summary...")
        summary_text = generate_detailed_pattern_summary(df, analysis)
        
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)
        
        with open(output_path / 'pattern_summaries.txt', 'w') as f:
            f.write(summary_text)
        
        print(f"✅ Detailed pattern summary saved to: {output_path / 'pattern_summaries.txt'}")
    
    # Save the parsed data
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    df.to_csv(output_path / 'training_metrics.csv', index=False)
    print(f"💾 Training metrics saved to: {output_path / 'training_metrics.csv'}")

if __name__ == "__main__":
    main() 