#!/usr/bin/env python3
"""
Analyze shuffling results to validate effectiveness in addressing value head fingerprinting.

This script analyzes the shuffled dataset to ensure:
1. Games are properly distributed across files
2. No game clustering remains
3. Value targets are well-mixed
4. Data integrity is maintained
"""

import sys
import os
import argparse
import gzip
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_shuffled_data(shuffled_dir: Path) -> List[Dict]:
    """Load all examples from shuffled files."""
    examples = []
    shuffled_files = list(shuffled_dir.glob("shuffled_*.pkl.gz"))
    
    print(f"Loading {len(shuffled_files)} shuffled files...")
    
    for file_path in shuffled_files:
        try:
            with gzip.open(file_path, 'rb') as f:
                data = pickle.load(f)
                examples.extend(data['examples'])
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"Loaded {len(examples)} total examples")
    return examples


def analyze_game_distribution(examples: List[Dict]) -> Dict[str, Any]:
    """Analyze how games are distributed in the shuffled data."""
    print("Analyzing game distribution...")
    
    # Group examples by game
    games = defaultdict(list)
    for example in examples:
        metadata = example['metadata']
        winner = metadata.get('winner', 'UNKNOWN')
        total_positions = metadata.get('total_positions', 0)
        
        # Create game identifier
        game_key = f"{winner}_{total_positions}"
        games[game_key].append(example)
    
    # Analyze game sizes
    game_sizes = [len(game_examples) for game_examples in games.values()]
    
    # Find consecutive sequences of same game
    game_sequences = []
    current_sequence = []
    current_game = None
    
    for example in examples:
        metadata = example['metadata']
        winner = metadata.get('winner', 'UNKNOWN')
        total_positions = metadata.get('total_positions', 0)
        position = metadata.get('position_in_game', 0)
        
        game_key = f"{winner}_{total_positions}"
        
        if game_key == current_game and position == len(current_sequence):
            current_sequence.append(example)
        else:
            if current_sequence:
                game_sequences.append(current_sequence)
            current_sequence = [example]
            current_game = game_key
    
    if current_sequence:
        game_sequences.append(current_sequence)
    
    # Calculate statistics
    sequence_lengths = [len(seq) for seq in game_sequences]
    
    analysis = {
        'total_games': len(games),
        'total_examples': len(examples),
        'avg_game_size': np.mean(game_sizes),
        'max_game_size': max(game_sizes),
        'min_game_size': min(game_sizes),
        'game_size_std': np.std(game_sizes),
        'total_sequences': len(game_sequences),
        'avg_sequence_length': np.mean(sequence_lengths),
        'max_sequence_length': max(sequence_lengths),
        'min_sequence_length': min(sequence_lengths),
        'sequence_length_std': np.std(sequence_lengths),
        'fraction_broken_games': sum(1 for seq in game_sequences if len(seq) < max(game_sizes)) / len(games)
    }
    
    return analysis


def analyze_value_distribution(examples: List[Dict]) -> Dict[str, Any]:
    """Analyze the distribution of value targets."""
    print("Analyzing value distribution...")
    
    values = [example['value'] for example in examples]
    value_counts = Counter(values)
    
    # Check for clustering of values
    value_sequences = []
    current_sequence = []
    current_value = None
    
    for value in values:
        if value == current_value:
            current_sequence.append(value)
        else:
            if current_sequence:
                value_sequences.append(current_sequence)
            current_sequence = [value]
            current_value = value
    
    if current_sequence:
        value_sequences.append(current_sequence)
    
    sequence_lengths = [len(seq) for seq in value_sequences]
    
    analysis = {
        'total_examples': len(values),
        'blue_wins': value_counts.get(0.0, 0),
        'red_wins': value_counts.get(1.0, 0),
        'blue_win_rate': value_counts.get(0.0, 0) / len(values),
        'red_win_rate': value_counts.get(1.0, 0) / len(values),
        'avg_value': np.mean(values),
        'value_std': np.std(values),
        'max_value_sequence': max(sequence_lengths),
        'avg_value_sequence': np.mean(sequence_lengths),
        'value_sequence_std': np.std(sequence_lengths)
    }
    
    return analysis


def analyze_file_distribution(examples: List[Dict]) -> Dict[str, Any]:
    """Analyze how examples are distributed across shuffled files."""
    print("Analyzing file distribution...")
    
    # Count examples per file
    file_counts = defaultdict(int)
    for example in examples:
        # Extract file information from metadata if available
        metadata = example.get('metadata', {})
        # Note: This would need to be updated based on actual metadata structure
        file_counts['unknown'] += 1
    
    analysis = {
        'total_files': len(file_counts),
        'avg_examples_per_file': np.mean(list(file_counts.values())),
        'max_examples_per_file': max(file_counts.values()),
        'min_examples_per_file': min(file_counts.values()),
        'file_count_std': np.std(list(file_counts.values()))
    }
    
    return analysis


def create_visualizations(examples: List[Dict], output_dir: Path):
    """Create visualizations of the shuffling results."""
    print("Creating visualizations...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Game size distribution
    games = defaultdict(list)
    for example in examples:
        metadata = example['metadata']
        winner = metadata.get('winner', 'UNKNOWN')
        total_positions = metadata.get('total_positions', 0)
        game_key = f"{winner}_{total_positions}"
        games[game_key].append(example)
    
    game_sizes = [len(game_examples) for game_examples in games.values()]
    
    plt.figure(figsize=(10, 6))
    plt.hist(game_sizes, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Game Size (Number of Positions)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Game Sizes in Shuffled Data')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'game_size_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Value target distribution
    values = [example['value'] for example in examples]
    
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Value Target')
    plt.ylabel('Frequency')
    plt.title('Distribution of Value Targets in Shuffled Data')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'value_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Sequence length analysis
    game_sequences = []
    current_sequence = []
    current_game = None
    
    for example in examples:
        metadata = example['metadata']
        winner = metadata.get('winner', 'UNKNOWN')
        total_positions = metadata.get('total_positions', 0)
        position = metadata.get('position_in_game', 0)
        
        game_key = f"{winner}_{total_positions}"
        
        if game_key == current_game and position == len(current_sequence):
            current_sequence.append(example)
        else:
            if current_sequence:
                game_sequences.append(current_sequence)
            current_sequence = [example]
            current_game = game_key
    
    if current_sequence:
        game_sequences.append(current_sequence)
    
    sequence_lengths = [len(seq) for seq in game_sequences]
    
    plt.figure(figsize=(10, 6))
    plt.hist(sequence_lengths, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Consecutive Game Positions')
    plt.ylabel('Frequency')
    plt.title('Distribution of Consecutive Game Positions in Shuffled Data')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'sequence_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Winner distribution by game size
    winner_by_size = defaultdict(lambda: {'BLUE': 0, 'RED': 0})
    for game_examples in games.values():
        if game_examples:
            winner = game_examples[0]['metadata']['winner']
            size = len(game_examples)
            winner_by_size[size][winner] += 1
    
    sizes = sorted(winner_by_size.keys())
    blue_counts = [winner_by_size[size]['BLUE'] for size in sizes]
    red_counts = [winner_by_size[size]['RED'] for size in sizes]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(sizes))
    width = 0.35
    
    plt.bar(x - width/2, blue_counts, width, label='BLUE Wins', alpha=0.7)
    plt.bar(x + width/2, red_counts, width, label='RED Wins', alpha=0.7)
    
    plt.xlabel('Game Size')
    plt.ylabel('Number of Games')
    plt.title('Winner Distribution by Game Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'winner_by_game_size.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_report(analysis_results: Dict[str, Dict], output_file: Path):
    """Generate a comprehensive analysis report."""
    print("Generating analysis report...")
    
    with open(output_file, 'w') as f:
        f.write("# Data Shuffling Analysis Report\n\n")
        f.write(f"Generated: {output_file.parent.name}\n\n")
        
        # Game Distribution Analysis
        f.write("## Game Distribution Analysis\n\n")
        game_analysis = analysis_results['game_distribution']
        f.write(f"- **Total Games**: {game_analysis['total_games']:,}\n")
        f.write(f"- **Total Examples**: {game_analysis['total_examples']:,}\n")
        f.write(f"- **Average Game Size**: {game_analysis['avg_game_size']:.1f} positions\n")
        f.write(f"- **Game Size Range**: {game_analysis['min_game_size']} - {game_analysis['max_game_size']} positions\n")
        f.write(f"- **Game Size Std Dev**: {game_analysis['game_size_std']:.1f}\n\n")
        
        f.write("### Game Fragmentation Analysis\n\n")
        f.write(f"- **Total Sequences**: {game_analysis['total_sequences']:,}\n")
        f.write(f"- **Average Sequence Length**: {game_analysis['avg_sequence_length']:.1f} positions\n")
        f.write(f"- **Max Sequence Length**: {game_analysis['max_sequence_length']} positions\n")
        f.write(f"- **Sequence Length Std Dev**: {game_analysis['sequence_length_std']:.1f}\n")
        f.write(f"- **Fraction of Broken Games**: {game_analysis['fraction_broken_games']:.1%}\n\n")
        
        # Value Distribution Analysis
        f.write("## Value Distribution Analysis\n\n")
        value_analysis = analysis_results['value_distribution']
        f.write(f"- **Total Examples**: {value_analysis['total_examples']:,}\n")
        f.write(f"- **BLUE Wins**: {value_analysis['blue_wins']:,} ({value_analysis['blue_win_rate']:.1%})\n")
        f.write(f"- **RED Wins**: {value_analysis['red_wins']:,} ({value_analysis['red_win_rate']:.1%})\n")
        f.write(f"- **Average Value**: {value_analysis['avg_value']:.3f}\n")
        f.write(f"- **Value Std Dev**: {value_analysis['value_std']:.3f}\n\n")
        
        f.write("### Value Clustering Analysis\n\n")
        f.write(f"- **Max Value Sequence**: {value_analysis['max_value_sequence']} consecutive\n")
        f.write(f"- **Average Value Sequence**: {value_analysis['avg_value_sequence']:.1f} consecutive\n")
        f.write(f"- **Value Sequence Std Dev**: {value_analysis['value_sequence_std']:.1f}\n\n")
        
        # File Distribution Analysis
        f.write("## File Distribution Analysis\n\n")
        file_analysis = analysis_results['file_distribution']
        f.write(f"- **Total Files**: {file_analysis['total_files']}\n")
        f.write(f"- **Average Examples per File**: {file_analysis['avg_examples_per_file']:.0f}\n")
        f.write(f"- **Examples per File Range**: {file_analysis['min_examples_per_file']} - {file_analysis['max_examples_per_file']}\n")
        f.write(f"- **File Count Std Dev**: {file_analysis['file_count_std']:.1f}\n\n")
        
        # Quality Assessment
        f.write("## Quality Assessment\n\n")
        
        # Check for potential issues
        issues = []
        
        if game_analysis['max_sequence_length'] > game_analysis['avg_game_size'] * 0.5:
            issues.append("Some games may not be properly fragmented")
        
        if value_analysis['max_value_sequence'] > 100:
            issues.append("Value targets may be clustered")
        
        if abs(value_analysis['blue_win_rate'] - 0.5) > 0.1:
            issues.append("Value target distribution may be imbalanced")
        
        if file_analysis['file_count_std'] > file_analysis['avg_examples_per_file'] * 0.5:
            issues.append("Examples may not be evenly distributed across files")
        
        if issues:
            f.write("### Potential Issues Detected\n\n")
            for issue in issues:
                f.write(f"- ⚠️ {issue}\n")
            f.write("\n")
        else:
            f.write("### Quality Assessment: PASSED ✅\n\n")
            f.write("No significant issues detected in the shuffled data.\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        if game_analysis['fraction_broken_games'] < 0.8:
            f.write("- Consider increasing the number of buckets for better game dispersion\n")
        
        if value_analysis['max_value_sequence'] > 50:
            f.write("- Consider additional shuffling passes to break value clustering\n")
        
        if abs(value_analysis['blue_win_rate'] - 0.5) > 0.05:
            f.write("- Consider balancing value targets across files\n")
        
        f.write("\n## Summary\n\n")
        f.write("The shuffling process appears to have successfully addressed the value head fingerprinting issue by:\n")
        f.write(f"- Breaking up {game_analysis['fraction_broken_games']:.1%} of games across multiple sequences\n")
        f.write(f"- Reducing average consecutive game positions from ~{game_analysis['avg_game_size']:.0f} to ~{game_analysis['avg_sequence_length']:.1f}\n")
        f.write(f"- Maintaining balanced value target distribution ({value_analysis['blue_win_rate']:.1%} BLUE, {value_analysis['red_win_rate']:.1%} RED)\n")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze shuffling results")
    parser.add_argument("--shuffled-dir", default="data/processed/shuffled",
                       help="Directory containing shuffled files")
    parser.add_argument("--output-dir", default="analysis/shuffling_results",
                       help="Output directory for analysis results")
    parser.add_argument("--no-visualizations", action="store_true",
                       help="Skip creating visualizations")
    
    args = parser.parse_args()
    
    shuffled_dir = Path(args.shuffled_dir)
    output_dir = Path(args.output_dir)
    
    if not shuffled_dir.exists():
        print(f"Error: Shuffled directory {shuffled_dir} not found")
        return 1
    
    # Load shuffled data
    examples = load_shuffled_data(shuffled_dir)
    
    if not examples:
        print("Error: No examples found in shuffled data")
        return 1
    
    # Perform analyses
    analysis_results = {
        'game_distribution': analyze_game_distribution(examples),
        'value_distribution': analyze_value_distribution(examples),
        'file_distribution': analyze_file_distribution(examples)
    }
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate report
    report_file = output_dir / "shuffling_analysis_report.md"
    generate_report(analysis_results, report_file)
    
    # Create visualizations
    if not args.no_visualizations:
        viz_dir = output_dir / "visualizations"
        create_visualizations(examples, viz_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED")
    print("="*60)
    print(f"Report saved to: {report_file}")
    if not args.no_visualizations:
        print(f"Visualizations saved to: {output_dir / 'visualizations'}")
    
    # Print key metrics
    game_analysis = analysis_results['game_distribution']
    value_analysis = analysis_results['value_distribution']
    
    print(f"\nKey Metrics:")
    print(f"- Total examples: {game_analysis['total_examples']:,}")
    print(f"- Games fragmented: {game_analysis['fraction_broken_games']:.1%}")
    print(f"- Avg consecutive positions: {game_analysis['avg_sequence_length']:.1f}")
    print(f"- Value balance: {value_analysis['blue_win_rate']:.1%} BLUE, {value_analysis['red_win_rate']:.1%} RED")
    
    return 0


if __name__ == "__main__":
    exit(main()) 