#!/usr/bin/env python3
"""
Performance analysis script for MCTS optimization.

This script parses PERF JSON output from MCTS runs and provides
quick insights for identifying bottlenecks and optimization targets.
"""

import json
import sys
import argparse
from typing import Dict, Any, Optional


def analyze_perf_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Analyze a single PERF JSON line.
    
    Args:
        line: Log line containing PERF JSON data
        
    Returns:
        Analysis results or None if not a PERF line
    """
    if not line.startswith('INFO:hex_ai.utils.perf:PERF:'):
        return None
    
    try:
        # Extract JSON part
        json_str = line.split('PERF: ')[1]
        data = json.loads(json_str)
        
        # Calculate key metrics
        timings = data.get('timings_s', {})
        counters = data.get('counters', {})
        samples = data.get('samples', {})
        
        total_time = sum(timings.values())
        if total_time == 0:
            return None
        
        # Basic metrics
        sims_per_sec = counters.get('mcts.sim', 0) / total_time
        batches = counters.get('nn.batch', 0)
        
        # Batch utilization
        avg_batch_size = 0
        if 'nn.batch_size' in samples:
            count, total = samples['nn.batch_size']
            avg_batch_size = total / count if count > 0 else 0
        
        # Time distribution
        time_distribution = {}
        for phase, time in timings.items():
            time_distribution[phase] = (time / total_time * 100)
        
        return {
            'sims_per_sec': sims_per_sec,
            'total_time': total_time,
            'total_sims': counters.get('mcts.sim', 0),
            'total_batches': batches,
            'avg_batch_size': avg_batch_size,
            'time_distribution': time_distribution,
            'device': data.get('meta', {}).get('device', 'unknown'),
            'dtype': data.get('meta', {}).get('dtype', 'unknown')
        }
        
    except (json.JSONDecodeError, KeyError, ZeroDivisionError) as e:
        print(f"Error parsing PERF line: {e}", file=sys.stderr)
        return None


def print_analysis(analysis: Dict[str, Any], verbose: bool = False):
    """
    Print formatted analysis results.
    
    Args:
        analysis: Analysis results from analyze_perf_line
        verbose: Whether to print detailed information
    """
    print(f"=== MCTS Performance Analysis ===")
    print(f"Device: {analysis['device']}, Dtype: {analysis['dtype']}")
    print(f"Total time: {analysis['total_time']:.3f}s")
    print(f"Simulations: {analysis['total_sims']}")
    print(f"Simulations/sec: {analysis['sims_per_sec']:.1f}")
    print(f"Batches: {analysis['total_batches']}")
    print(f"Avg batch size: {analysis['avg_batch_size']:.1f}")
    
    print(f"\nTime distribution:")
    sorted_phases = sorted(analysis['time_distribution'].items(), 
                          key=lambda x: x[1], reverse=True)
    for phase, pct in sorted_phases:
        print(f"  {phase}: {pct:.1f}%")
    
    # Identify bottlenecks
    print(f"\nBottleneck analysis:")
    expansion_pct = analysis['time_distribution'].get('mcts.expand', 0)
    selection_pct = analysis['time_distribution'].get('mcts.select', 0)
    inference_pct = analysis['time_distribution'].get('nn.infer', 0)
    
    if expansion_pct > 50:
        print(f"  ⚠️  EXPANSION DOMINATES ({expansion_pct:.1f}%) - Apply/undo pattern needed")
    elif selection_pct > 30:
        print(f"  ⚠️  SELECTION DOMINATES ({selection_pct:.1f}%) - Vectorize UCT calculations")
    elif inference_pct > 40:
        print(f"  ⚠️  INFERENCE DOMINATES ({inference_pct:.1f}%) - Optimize batch utilization")
    else:
        print(f"  ✅ Balanced performance - no clear bottleneck")
    
    # Batch utilization analysis
    if analysis['avg_batch_size'] < 32:  # Assuming target of 64
        print(f"  ⚠️  LOW BATCH UTILIZATION ({analysis['avg_batch_size']:.1f}) - Tune batch collection")
    else:
        print(f"  ✅ Good batch utilization ({analysis['avg_batch_size']:.1f})")
    
    if verbose:
        print(f"\nDetailed metrics:")
        for key, value in analysis.items():
            if key != 'time_distribution':
                print(f"  {key}: {value}")


def analyze_file(filename: str, verbose: bool = False):
    """
    Analyze PERF output from a file.
    
    Args:
        filename: File containing PERF log lines
        verbose: Whether to print detailed information
    """
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            analysis = analyze_perf_line(line.strip())
            if analysis:
                print(f"\n--- Analysis from line {line_num} ---")
                print_analysis(analysis, verbose)


def analyze_stdin(verbose: bool = False):
    """
    Analyze PERF output from stdin.
    
    Args:
        verbose: Whether to print detailed information
    """
    line_count = 0
    for line in sys.stdin:
        line_count += 1
        analysis = analyze_perf_line(line.strip())
        if analysis:
            print(f"\n--- Analysis from line {line_count} ---")
            print_analysis(analysis, verbose)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze MCTS performance from PERF JSON output"
    )
    parser.add_argument(
        '--file', '-f',
        help='File containing PERF log lines (default: read from stdin)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed analysis information'
    )
    
    args = parser.parse_args()
    
    if args.file:
        analyze_file(args.file, args.verbose)
    else:
        analyze_stdin(args.verbose)


if __name__ == "__main__":
    main()
