#!/usr/bin/env python3
"""
Script to investigate specific spike patterns in training data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def load_training_data(csv_path: str) -> pd.DataFrame:
    """Load the parsed training data."""
    return pd.read_csv(csv_path)

def identify_spikes(df: pd.DataFrame, threshold_multiplier: float = 2.0) -> pd.DataFrame:
    """Identify spikes in the training data."""
    # Calculate differences
    train_diff = df['train_total'].diff().abs()
    val_diff = df['val_total'].diff().abs()
    
    # Calculate thresholds
    train_threshold = train_diff.mean() + threshold_multiplier * train_diff.std()
    val_threshold = val_diff.mean() + threshold_multiplier * val_diff.std()
    
    # Find spikes
    train_spikes = df[train_diff > train_threshold].copy()
    val_spikes = df[val_diff > val_threshold].copy()
    
    # Add spike information
    train_spikes['spike_type'] = 'train'
    train_spikes['spike_magnitude'] = train_diff[train_diff > train_threshold]
    val_spikes['spike_type'] = 'validation'
    val_spikes['spike_magnitude'] = val_diff[val_diff > val_threshold]
    
    # Combine and sort
    all_spikes = pd.concat([train_spikes, val_spikes]).sort_values('batch_count')
    
    return all_spikes

def analyze_spike_patterns(spikes: pd.DataFrame) -> dict:
    """Analyze patterns in the spikes."""
    analysis = {}
    
    # Basic statistics
    analysis['total_spikes'] = len(spikes)
    analysis['train_spikes'] = len(spikes[spikes['spike_type'] == 'train'])
    analysis['val_spikes'] = len(spikes[spikes['spike_type'] == 'validation'])
    
    # Timing analysis
    analysis['spikes_per_quarter'] = []
    total_batches = spikes['batch_count'].max()
    quarter_size = total_batches // 4
    
    for i in range(4):
        start_batch = i * quarter_size
        end_batch = (i + 1) * quarter_size if i < 3 else total_batches
        quarter_spikes = len(spikes[(spikes['batch_count'] >= start_batch) & 
                                  (spikes['batch_count'] < end_batch)])
        analysis['spikes_per_quarter'].append(quarter_spikes)
    
    # Magnitude analysis
    analysis['avg_spike_magnitude'] = spikes['spike_magnitude'].mean()
    analysis['max_spike_magnitude'] = spikes['spike_magnitude'].max()
    
    # Loss component analysis for spikes
    train_spikes = spikes[spikes['spike_type'] == 'train']
    if len(train_spikes) > 0:
        analysis['train_spike_policy_change'] = train_spikes['train_policy'].diff().abs().mean()
        analysis['train_spike_value_change'] = train_spikes['train_value'].diff().abs().mean()
    
    return analysis

def generate_spike_analysis_summary(spikes: pd.DataFrame, analysis: dict, df: pd.DataFrame) -> str:
    """
    Generate detailed spike analysis summary for appending to pattern summaries.
    
    Args:
        spikes: DataFrame with spike information
        analysis: Analysis results dictionary
        df: Original training data DataFrame
        
    Returns:
        String containing detailed spike analysis
    """
    summary = []
    summary.append("DETAILED SPIKE ANALYSIS - APPENDED INVESTIGATION")
    summary.append("=" * 60)
    summary.append("")
    
    # Spike overview
    summary.append("SPIKE OVERVIEW - COMPREHENSIVE BREAKDOWN")
    summary.append("-" * 40)
    summary.append(f"Total spikes detected: {analysis['total_spikes']}")
    summary.append(f"Training spikes: {analysis['train_spikes']}")
    summary.append(f"Validation spikes: {analysis['val_spikes']}")
    summary.append(f"Average spike magnitude: {analysis['avg_spike_magnitude']:.4f}")
    summary.append(f"Maximum spike magnitude: {analysis['max_spike_magnitude']:.4f}")
    summary.append("")
    
    # Timing analysis
    summary.append("TIMING ANALYSIS - QUARTERLY BREAKDOWN")
    summary.append("-" * 40)
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    for i, (quarter, count) in enumerate(zip(quarters, analysis['spikes_per_quarter'])):
        summary.append(f"{quarter}: {count} spikes")
    summary.append("")
    
    # Magnitude analysis
    summary.append("MAGNITUDE ANALYSIS - DETAILED STATISTICS")
    summary.append("-" * 40)
    summary.append(f"Average spike magnitude: {analysis['avg_spike_magnitude']:.4f}")
    summary.append(f"Maximum spike magnitude: {analysis['max_spike_magnitude']:.4f}")
    summary.append(f"Spike magnitude standard deviation: {spikes['spike_magnitude'].std():.4f}")
    summary.append("")
    
    # Loss component changes during spikes
    if 'train_spike_policy_change' in analysis:
        summary.append("LOSS COMPONENT CHANGES DURING TRAIN SPIKES")
        summary.append("-" * 40)
        summary.append(f"Average policy loss change: {analysis['train_spike_policy_change']:.4f}")
        summary.append(f"Average value loss change: {analysis['train_spike_value_change']:.4f}")
        summary.append(f"Policy/Value change ratio: {analysis['train_spike_policy_change'] / analysis['train_spike_value_change']:.2f}")
        summary.append("")
    
    # Detailed spike timeline with magnitudes
    summary.append("DETAILED SPIKE TIMELINE WITH MAGNITUDES")
    summary.append("-" * 40)
    
    train_spikes = spikes[spikes['spike_type'] == 'train']
    if len(train_spikes) > 0:
        summary.append("Train Spikes (with magnitudes):")
        for _, spike in train_spikes.iterrows():
            summary.append(f"- Batch {spike['batch_count']}: {spike['train_total']:.4f} (magnitude: {spike['spike_magnitude']:.4f})")
        summary.append("")
    
    val_spikes = spikes[spikes['spike_type'] == 'validation']
    if len(val_spikes) > 0:
        summary.append("Validation Spikes (with magnitudes):")
        for _, spike in val_spikes.iterrows():
            summary.append(f"- Batch {spike['batch_count']}: {spike['val_total']:.4f} (magnitude: {spike['spike_magnitude']:.4f})")
        summary.append("")
    
    # Spike clustering analysis
    summary.append("SPIKE CLUSTERING ANALYSIS")
    summary.append("-" * 30)
    
    # Find clusters of spikes (spikes within 10 batches of each other)
    spike_batches = sorted(spikes['batch_count'].tolist())
    clusters = []
    current_cluster = [spike_batches[0]]
    
    for batch in spike_batches[1:]:
        if batch - current_cluster[-1] <= 10:
            current_cluster.append(batch)
        else:
            if len(current_cluster) > 1:
                clusters.append(current_cluster)
            current_cluster = [batch]
    
    if len(current_cluster) > 1:
        clusters.append(current_cluster)
    
    summary.append(f"Spike clusters detected: {len(clusters)}")
    for i, cluster in enumerate(clusters):
        summary.append(f"Cluster {i+1}: {len(cluster)} spikes at batches {cluster}")
    summary.append("")
    
    # Spike frequency analysis
    summary.append("SPIKE FREQUENCY ANALYSIS")
    summary.append("-" * 30)
    
    # Calculate spike frequency over time windows
    window_size = max(1, len(df) // 20)
    spike_counts = []
    batch_ranges = []
    
    for i in range(0, len(df), window_size):
        end_idx = min(i + window_size, len(df))
        batch_start = df.iloc[i]['batch_count']
        batch_end = df.iloc[end_idx-1]['batch_count']
        
        window_spikes = len(spikes[(spikes['batch_count'] >= batch_start) & 
                                 (spikes['batch_count'] <= batch_end)])
        spike_counts.append(window_spikes)
        batch_ranges.append(batch_start)
    
    summary.append(f"Spike frequency analysis (window size: {window_size} mini-epochs):")
    summary.append(f"Average spikes per window: {np.mean(spike_counts):.2f}")
    summary.append(f"Maximum spikes in a window: {np.max(spike_counts)}")
    summary.append(f"Minimum spikes in a window: {np.min(spike_counts)}")
    summary.append("")
    
    # Spike impact analysis
    summary.append("SPIKE IMPACT ANALYSIS")
    summary.append("-" * 25)
    
    # Analyze how spikes affect subsequent performance
    spike_impact_periods = []
    for _, spike in spikes.iterrows():
        spike_batch = spike['batch_count']
        # Look at performance in the 5 mini-epochs after the spike
        future_data = df[df['batch_count'] > spike_batch].head(5)
        if len(future_data) > 0:
            if spike['spike_type'] == 'train':
                # For train spikes, check if validation loss increases
                if len(future_data) > 1:
                    val_change = future_data['val_total'].iloc[-1] - future_data['val_total'].iloc[0]
                    spike_impact_periods.append({
                        'spike_batch': spike_batch,
                        'spike_type': 'train',
                        'val_change': val_change,
                        'magnitude': spike['spike_magnitude']
                    })
    
    if spike_impact_periods:
        summary.append("Train spike impact on validation performance:")
        val_increases = [p for p in spike_impact_periods if p['val_change'] > 0]
        summary.append(f"Spikes followed by validation loss increase: {len(val_increases)}/{len(spike_impact_periods)} ({len(val_increases)/len(spike_impact_periods)*100:.1f}%)")
        if val_increases:
            avg_increase = np.mean([p['val_change'] for p in val_increases])
            summary.append(f"Average validation loss increase: {avg_increase:.4f}")
        summary.append("")
    
    # Recommendations based on spike analysis
    summary.append("SPIKE-BASED RECOMMENDATIONS")
    summary.append("-" * 30)
    
    if analysis['avg_spike_magnitude'] > 0.1:
        summary.append("⚠️  Large average spike magnitude - learning rate likely too high")
        summary.append(f"   Current: {analysis['avg_spike_magnitude']:.4f}, Target: <0.05")
        reduction_needed = ((analysis['avg_spike_magnitude'] - 0.05) / analysis['avg_spike_magnitude']) * 100
        summary.append(f"   Suggested learning rate reduction: {reduction_needed:.1f}%")
    
    if analysis['spikes_per_quarter'][0] > analysis['spikes_per_quarter'][-1]:
        summary.append("✅ Spike frequency decreased over time - training stabilizing")
    else:
        summary.append("⚠️  Spike frequency increased over time - concerning trend")
    
    if len(clusters) > 0:
        summary.append("⚠️  Spike clustering detected - possible data shard effects")
        summary.append("   Consider investigating data shard transitions")
    
    summary.append("")
    
    return "\n".join(summary)

def plot_spike_analysis(df: pd.DataFrame, spikes: pd.DataFrame, output_dir: str):
    """Create detailed spike analysis plots."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 1. Spike timeline
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('Detailed Spike Analysis', fontsize=16, fontweight='bold')
    
    # Training spikes
    train_spikes = spikes[spikes['spike_type'] == 'train']
    axes[0].plot(df['batch_count'], df['train_total'], color='blue', alpha=0.7, label='Train Total')
    if len(train_spikes) > 0:
        axes[0].scatter(train_spikes['batch_count'], train_spikes['train_total'], 
                       color='red', s=100, alpha=0.8, label=f'Train Spikes ({len(train_spikes)})')
    axes[0].set_xlabel('Batches Processed')
    axes[0].set_ylabel('Train Total Loss')
    axes[0].set_title('Training Loss with Spike Annotations')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Validation spikes
    val_spikes = spikes[spikes['spike_type'] == 'validation']
    axes[1].plot(df['batch_count'], df['val_total'], color='green', alpha=0.7, label='Validation Total')
    if len(val_spikes) > 0:
        axes[1].scatter(val_spikes['batch_count'], val_spikes['val_total'], 
                       color='red', s=100, alpha=0.8, label=f'Validation Spikes ({len(val_spikes)})')
    axes[1].set_xlabel('Batches Processed')
    axes[1].set_ylabel('Validation Total Loss')
    axes[1].set_title('Validation Loss with Spike Annotations')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'detailed_spikes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Spike magnitude analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Spike Magnitude Analysis', fontsize=16, fontweight='bold')
    
    # Spike magnitude distribution
    axes[0, 0].hist(spikes['spike_magnitude'], bins=20, alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].axvline(spikes['spike_magnitude'].mean(), color='blue', linestyle='--', 
                       label=f'Mean: {spikes["spike_magnitude"].mean():.4f}')
    axes[0, 0].set_xlabel('Spike Magnitude')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Spike Magnitude Distribution')
    axes[0, 0].legend()
    
    # Spike timing distribution
    quarter_labels = ['Q1', 'Q2', 'Q3', 'Q4']
    analysis = analyze_spike_patterns(spikes)
    axes[0, 1].bar(quarter_labels, analysis['spikes_per_quarter'], color=['red', 'orange', 'yellow', 'green'])
    axes[0, 1].set_xlabel('Training Quarter')
    axes[0, 1].set_ylabel('Number of Spikes')
    axes[0, 1].set_title('Spikes by Training Quarter')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss component changes during spikes
    if len(train_spikes) > 0:
        policy_changes = train_spikes['train_policy'].diff().abs()
        value_changes = train_spikes['train_value'].diff().abs()
        
        axes[1, 0].scatter(policy_changes, value_changes, alpha=0.7, color='blue')
        axes[1, 0].set_xlabel('Policy Loss Change')
        axes[1, 0].set_ylabel('Value Loss Change')
        axes[1, 0].set_title('Loss Component Changes During Train Spikes')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Spike frequency over time
    window_size = max(1, len(df) // 20)
    spike_counts = []
    batch_ranges = []
    
    for i in range(0, len(df), window_size):
        end_idx = min(i + window_size, len(df))
        batch_start = df.iloc[i]['batch_count']
        batch_end = df.iloc[end_idx-1]['batch_count']
        
        window_spikes = len(spikes[(spikes['batch_count'] >= batch_start) & 
                                 (spikes['batch_count'] <= batch_end)])
        spike_counts.append(window_spikes)
        batch_ranges.append(batch_start)
    
    axes[1, 1].plot(batch_ranges, spike_counts, marker='o', linewidth=2)
    axes[1, 1].set_xlabel('Batch Count')
    axes[1, 1].set_ylabel('Spikes per Window')
    axes[1, 1].set_title(f'Spike Frequency (Window Size: {window_size})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'spike_magnitude_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_spike_summary(spikes: pd.DataFrame, analysis: dict):
    """Print a summary of spike analysis."""
    print("\n" + "="*60)
    print("DETAILED SPIKE ANALYSIS")
    print("="*60)
    
    print(f"\n📊 Spike Overview:")
    print(f"   • Total spikes: {analysis['total_spikes']}")
    print(f"   • Training spikes: {analysis['train_spikes']}")
    print(f"   • Validation spikes: {analysis['val_spikes']}")
    
    print(f"\n⏰ Timing Analysis:")
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    for i, (quarter, count) in enumerate(zip(quarters, analysis['spikes_per_quarter'])):
        print(f"   • {quarter}: {count} spikes")
    
    print(f"\n📈 Magnitude Analysis:")
    print(f"   • Average spike magnitude: {analysis['avg_spike_magnitude']:.4f}")
    print(f"   • Maximum spike magnitude: {analysis['max_spike_magnitude']:.4f}")
    
    if 'train_spike_policy_change' in analysis:
        print(f"\n🔍 Loss Component Changes (Train Spikes):")
        print(f"   • Average policy loss change: {analysis['train_spike_policy_change']:.4f}")
        print(f"   • Average value loss change: {analysis['train_spike_value_change']:.4f}")
    
    print(f"\n💡 Key Insights:")
    if analysis['spikes_per_quarter'][0] > analysis['spikes_per_quarter'][-1]:
        print(f"   ✅ Spike frequency decreased over time (good sign)")
    else:
        print(f"   ⚠️  Spike frequency increased over time (concerning)")
    
    if analysis['avg_spike_magnitude'] > 0.1:
        print(f"   ⚠️  Large average spike magnitude ({analysis['avg_spike_magnitude']:.4f}) - learning rate likely too high")
    else:
        print(f"   ✅ Reasonable spike magnitudes")
    
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description='Investigate spike patterns in training data')
    parser.add_argument('csv_file', help='Path to training metrics CSV file')
    parser.add_argument('--output_dir', default='spike_analysis', help='Directory to save spike analysis plots')
    parser.add_argument('--threshold', type=float, default=2.0, help='Standard deviation multiplier for spike detection')
    parser.add_argument('--append_to_summary', help='Path to pattern summaries file to append spike analysis')
    
    args = parser.parse_args()
    
    print(f"🔍 Investigating spikes in: {args.csv_file}")
    
    # Load data
    df = load_training_data(args.csv_file)
    print(f"✅ Loaded {len(df)} data points")
    
    # Identify spikes
    spikes = identify_spikes(df, args.threshold)
    print(f"✅ Identified {len(spikes)} spikes")
    
    # Analyze patterns
    analysis = analyze_spike_patterns(spikes)
    
    # Print summary
    print_spike_summary(spikes, analysis)
    
    # Create plots
    print(f"\n📊 Generating spike analysis plots in: {args.output_dir}")
    plot_spike_analysis(df, spikes, args.output_dir)
    print("✅ Spike analysis complete!")
    
    # Append to pattern summary if requested
    if args.append_to_summary:
        print(f"\n📝 Appending spike analysis to: {args.append_to_summary}")
        spike_summary = generate_spike_analysis_summary(spikes, analysis, df)
        
        # Append to existing file
        with open(args.append_to_summary, 'a') as f:
            f.write("\n\n")
            f.write(spike_summary)
        
        print(f"✅ Spike analysis appended to pattern summaries")
    
    # Save spike data
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    spikes.to_csv(output_path / 'spikes_detailed.csv', index=False)
    print(f"💾 Detailed spike data saved to: {output_path / 'spikes_detailed.csv'}")

if __name__ == "__main__":
    main() 