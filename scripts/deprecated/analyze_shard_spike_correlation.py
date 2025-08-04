#!/usr/bin/env python3
"""
Script to analyze correlation between data shard transitions and performance spikes.
This helps determine if the sawtooth pattern is caused by data shard effects.
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Tuple, Optional
import seaborn as sns

def parse_shard_logs(log_file_path: str) -> pd.DataFrame:
    """
    Parse shard transition logs from training console output.
    
    Args:
        log_file_path: Path to the training console output file
        
    Returns:
        DataFrame with shard transition information
    """
    data = []
    
    # Pattern to match shard start logs
    pattern = r'\[SHARD_START\] Processing shard (\d+)/(\d+): ([^\s]+)'
    
    with open(log_file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            match = re.search(pattern, line)
            if match:
                shard_num = int(match.group(1))
                total_shards = int(match.group(2))
                file_name = match.group(3)
                
                # Try to extract batch count if present
                batch_match = re.search(r'\(batch (\d+)\)', line)
                batch_count = int(batch_match.group(1)) if batch_match else None
                
                data.append({
                    'line_number': line_num,
                    'shard_num': shard_num,
                    'total_shards': total_shards,
                    'file_name': file_name,
                    'batch_count': batch_count
                })
    
    df = pd.DataFrame(data)
    return df

def parse_training_metrics(csv_file_path: str) -> pd.DataFrame:
    """
    Load training metrics from CSV file.
    
    Args:
        csv_file_path: Path to training metrics CSV
        
    Returns:
        DataFrame with training metrics
    """
    return pd.read_csv(csv_file_path)

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

def analyze_shard_spike_correlation(shard_df: pd.DataFrame, metrics_df: pd.DataFrame, spikes_df: pd.DataFrame) -> Dict:
    """
    Analyze correlation between shard transitions and performance spikes.
    
    Args:
        shard_df: DataFrame with shard transition information
        metrics_df: DataFrame with training metrics
        spikes_df: DataFrame with identified spikes
        
    Returns:
        Dictionary with correlation analysis results
    """
    analysis = {}
    
    # Basic statistics
    analysis['total_shards'] = len(shard_df)
    analysis['total_spikes'] = len(spikes_df)
    analysis['train_spikes'] = len(spikes_df[spikes_df['spike_type'] == 'train'])
    analysis['val_spikes'] = len(spikes_df[spikes_df['spike_type'] == 'validation'])
    
    # Find spikes that occur near shard transitions
    shard_transition_batches = shard_df['batch_count'].dropna().tolist()
    
    # Define "near" as within 10 batches of a shard transition
    near_threshold = 10
    spikes_near_shard_transitions = []
    
    for _, spike in spikes_df.iterrows():
        spike_batch = spike['batch_count']
        for shard_batch in shard_transition_batches:
            if abs(spike_batch - shard_batch) <= near_threshold:
                spikes_near_shard_transitions.append({
                    'spike_batch': spike_batch,
                    'shard_batch': shard_batch,
                    'distance': abs(spike_batch - shard_batch),
                    'spike_type': spike['spike_type'],
                    'spike_magnitude': spike['spike_magnitude']
                })
                break
    
    analysis['spikes_near_shard_transitions'] = len(spikes_near_shard_transitions)
    analysis['spikes_near_shard_transitions_ratio'] = len(spikes_near_shard_transitions) / len(spikes_df) if len(spikes_df) > 0 else 0
    
    # Analyze shard-specific patterns
    shard_spike_counts = {}
    for _, spike in spikes_df.iterrows():
        spike_batch = spike['batch_count']
        # Find which shard this spike occurred in
        for _, shard in shard_df.iterrows():
            if shard['batch_count'] is not None:
                # This is a simplified approach - in reality we'd need to track shard boundaries more precisely
                if spike_batch >= shard['batch_count']:
                    shard_name = shard['file_name']
                    if shard_name not in shard_spike_counts:
                        shard_spike_counts[shard_name] = {'train': 0, 'val': 0, 'total': 0}
                    shard_spike_counts[shard_name][spike['spike_type']] += 1
                    shard_spike_counts[shard_name]['total'] += 1
                    break
    
    analysis['shard_spike_counts'] = shard_spike_counts
    
    # Calculate correlation statistics
    if len(spikes_near_shard_transitions) > 0:
        distances = [s['distance'] for s in spikes_near_shard_transitions]
        analysis['avg_distance_to_shard_transition'] = np.mean(distances)
        analysis['min_distance_to_shard_transition'] = np.min(distances)
        analysis['max_distance_to_shard_transition'] = np.max(distances)
    else:
        analysis['avg_distance_to_shard_transition'] = None
        analysis['min_distance_to_shard_transition'] = None
        analysis['max_distance_to_shard_transition'] = None
    
    return analysis

def create_shard_spike_visualizations(shard_df: pd.DataFrame, metrics_df: pd.DataFrame, spikes_df: pd.DataFrame, analysis: Dict, output_dir: str):
    """
    Create visualizations showing shard-spike correlations.
    
    Args:
        shard_df: DataFrame with shard transition information
        metrics_df: DataFrame with training metrics
        spikes_df: DataFrame with identified spikes
        analysis: Analysis results dictionary
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Training curves with shard transitions and spikes
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    fig.suptitle('Shard Transitions and Performance Spikes', fontsize=16, fontweight='bold')
    
    # Training loss with shard transitions and spikes
    axes[0].plot(metrics_df['batch_count'], metrics_df['train_total'], color='blue', alpha=0.7, label='Train Total Loss')
    
    # Mark shard transitions
    shard_batches = shard_df['batch_count'].dropna()
    for batch in shard_batches:
        axes[0].axvline(x=batch, color='green', alpha=0.3, linestyle='--', linewidth=1)
    
    # Mark spikes
    train_spikes = spikes_df[spikes_df['spike_type'] == 'train']
    if len(train_spikes) > 0:
        axes[0].scatter(train_spikes['batch_count'], train_spikes['train_total'], 
                       color='red', s=100, alpha=0.8, label=f'Train Spikes ({len(train_spikes)})')
    
    axes[0].set_xlabel('Batches Processed')
    axes[0].set_ylabel('Train Total Loss')
    axes[0].set_title('Training Loss with Shard Transitions (green lines) and Spikes (red dots)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Validation loss with shard transitions and spikes
    axes[1].plot(metrics_df['batch_count'], metrics_df['val_total'], color='green', alpha=0.7, label='Validation Total Loss')
    
    # Mark shard transitions
    for batch in shard_batches:
        axes[1].axvline(x=batch, color='green', alpha=0.3, linestyle='--', linewidth=1)
    
    # Mark spikes
    val_spikes = spikes_df[spikes_df['spike_type'] == 'validation']
    if len(val_spikes) > 0:
        axes[1].scatter(val_spikes['batch_count'], val_spikes['val_total'], 
                       color='red', s=100, alpha=0.8, label=f'Validation Spikes ({len(val_spikes)})')
    
    axes[1].set_xlabel('Batches Processed')
    axes[1].set_ylabel('Validation Total Loss')
    axes[1].set_title('Validation Loss with Shard Transitions (green lines) and Spikes (red dots)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'shard_spike_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Shard spike analysis
    if analysis['shard_spike_counts']:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Shard-Specific Spike Analysis', fontsize=16, fontweight='bold')
        
        # Shard spike counts
        shard_names = list(analysis['shard_spike_counts'].keys())
        train_counts = [analysis['shard_spike_counts'][name]['train'] for name in shard_names]
        val_counts = [analysis['shard_spike_counts'][name]['val'] for name in shard_names]
        
        x = np.arange(len(shard_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, train_counts, width, label='Train Spikes', alpha=0.7)
        axes[0, 0].bar(x + width/2, val_counts, width, label='Validation Spikes', alpha=0.7)
        axes[0, 0].set_xlabel('Shard')
        axes[0, 0].set_ylabel('Number of Spikes')
        axes[0, 0].set_title('Spikes per Shard')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([name[:20] + '...' if len(name) > 20 else name for name in shard_names], rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Total spikes per shard
        total_counts = [analysis['shard_spike_counts'][name]['total'] for name in shard_names]
        axes[0, 1].bar(range(len(shard_names)), total_counts, alpha=0.7, color='purple')
        axes[0, 1].set_xlabel('Shard')
        axes[0, 1].set_ylabel('Total Spikes')
        axes[0, 1].set_title('Total Spikes per Shard')
        axes[0, 1].set_xticks(range(len(shard_names)))
        axes[0, 1].set_xticklabels([name[:20] + '...' if len(name) > 20 else name for name in shard_names], rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Spike distance to shard transitions
        if analysis['avg_distance_to_shard_transition'] is not None:
            axes[1, 0].hist([s['distance'] for s in analysis.get('spikes_near_shard_transitions', [])], 
                           bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].axvline(analysis['avg_distance_to_shard_transition'], color='red', linestyle='--', 
                              label=f'Mean: {analysis["avg_distance_to_shard_transition"]:.1f}')
            axes[1, 0].set_xlabel('Distance to Shard Transition (batches)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Distance of Spikes to Shard Transitions')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Correlation summary
        axes[1, 1].axis('off')
        summary_text = f"""
Shard-Spike Correlation Summary:

Total Shards: {analysis['total_shards']}
Total Spikes: {analysis['total_spikes']}
Train Spikes: {analysis['train_spikes']}
Val Spikes: {analysis['val_spikes']}

Spikes Near Shard Transitions:
- Count: {analysis['spikes_near_shard_transitions']}
- Ratio: {analysis['spikes_near_shard_transitions_ratio']:.2%}

Distance Analysis:
- Avg Distance: {analysis['avg_distance_to_shard_transition']:.1f} batches
- Min Distance: {analysis['min_distance_to_shard_transition']} batches
- Max Distance: {analysis['max_distance_to_shard_transition']} batches
        """
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(output_path / 'shard_spike_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def print_correlation_summary(analysis: Dict):
    """
    Print a summary of the shard-spike correlation analysis.
    
    Args:
        analysis: Analysis results dictionary
    """
    print("\n" + "="*60)
    print("SHARD-SPIKE CORRELATION ANALYSIS")
    print("="*60)
    
    print(f"\n📊 Overview:")
    print(f"   • Total shards processed: {analysis['total_shards']}")
    print(f"   • Total spikes detected: {analysis['total_spikes']}")
    print(f"   • Train spikes: {analysis['train_spikes']}")
    print(f"   • Validation spikes: {analysis['val_spikes']}")
    
    print(f"\n🔗 Correlation Analysis:")
    print(f"   • Spikes near shard transitions: {analysis['spikes_near_shard_transitions']}")
    print(f"   • Correlation ratio: {analysis['spikes_near_shard_transitions_ratio']:.2%}")
    
    if analysis['avg_distance_to_shard_transition'] is not None:
        print(f"   • Average distance to shard transition: {analysis['avg_distance_to_shard_transition']:.1f} batches")
        print(f"   • Min distance: {analysis['min_distance_to_shard_transition']} batches")
        print(f"   • Max distance: {analysis['max_distance_to_shard_transition']} batches")
    
    print(f"\n💡 Interpretation:")
    if analysis['spikes_near_shard_transitions_ratio'] > 0.5:
        print(f"   ✅ Strong correlation between shard transitions and spikes")
        print(f"   ✅ This supports the data shard effects hypothesis")
    elif analysis['spikes_near_shard_transitions_ratio'] > 0.2:
        print(f"   ⚠️  Moderate correlation between shard transitions and spikes")
        print(f"   ⚠️  Data shard effects may be partially responsible")
    else:
        print(f"   ❌ Weak correlation between shard transitions and spikes")
        print(f"   ❌ Data shard effects are unlikely to be the primary cause")
    
    print(f"\n📈 Shard-Specific Patterns:")
    if analysis['shard_spike_counts']:
        high_spike_shards = [(name, counts['total']) for name, counts in analysis['shard_spike_counts'].items() if counts['total'] > 0]
        if high_spike_shards:
            print(f"   • Shards with spikes: {len(high_spike_shards)}")
            print(f"   • Highest spike count: {max(high_spike_shards, key=lambda x: x[1])}")
        else:
            print(f"   • No spikes detected in any shard")
    
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description='Analyze correlation between shard transitions and performance spikes')
    parser.add_argument('log_file', help='Path to training console output file')
    parser.add_argument('metrics_csv', help='Path to training metrics CSV file')
    parser.add_argument('--output_dir', default='shard_spike_analysis', help='Directory to save analysis plots')
    parser.add_argument('--threshold', type=float, default=2.0, help='Standard deviation multiplier for spike detection')
    
    args = parser.parse_args()
    
    print(f"🔍 Analyzing shard-spike correlation from: {args.log_file}")
    
    # Parse shard logs
    shard_df = parse_shard_logs(args.log_file)
    if shard_df.empty:
        print("❌ No shard transition logs found! Make sure to run training with shard logging enabled.")
        return
    
    print(f"✅ Parsed {len(shard_df)} shard transitions")
    
    # Load training metrics
    metrics_df = parse_training_metrics(args.metrics_csv)
    print(f"✅ Loaded {len(metrics_df)} training metrics")
    
    # Identify spikes
    spikes_df = identify_spikes(metrics_df, args.threshold)
    print(f"✅ Identified {len(spikes_df)} spikes")
    
    # Analyze correlation
    analysis = analyze_shard_spike_correlation(shard_df, metrics_df, spikes_df)
    
    # Print summary
    print_correlation_summary(analysis)
    
    # Create visualizations
    print(f"\n📊 Generating shard-spike correlation plots in: {args.output_dir}")
    create_shard_spike_visualizations(shard_df, metrics_df, spikes_df, analysis, args.output_dir)
    print("✅ Shard-spike correlation analysis complete!")
    
    # Save analysis data
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save shard transitions
    shard_df.to_csv(output_path / 'shard_transitions.csv', index=False)
    
    # Save correlation analysis
    import json
    with open(output_path / 'correlation_analysis.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        analysis_json = {}
        for key, value in analysis.items():
            if isinstance(value, np.integer):
                analysis_json[key] = int(value)
            elif isinstance(value, np.floating):
                analysis_json[key] = float(value)
            elif isinstance(value, np.ndarray):
                analysis_json[key] = value.tolist()
            else:
                analysis_json[key] = value
        json.dump(analysis_json, f, indent=2)
    
    print(f"💾 Analysis data saved to: {output_path}")

if __name__ == "__main__":
    main() 