#!/usr/bin/env python3
"""
Compare random vs deterministic tournament methods.

This script runs both tournament methods on the same strategies and compares
the results to validate that the deterministic approach produces similar
outcomes to the random approach.
"""

import argparse
import os
import sys
import subprocess
import tempfile
import json
from datetime import datetime
from typing import Dict, List, Any

from hex_ai.inference.tournament import TournamentResult


def run_tournament_command(cmd: List[str], description: str) -> Dict[str, Any]:
    """Run a tournament command and capture results."""
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    
    if result.returncode != 0:
        print(f"ERROR: Command failed with return code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return None
    
    # Parse the output to extract results
    output_lines = result.stdout.split('\n')
    
    # Look for tournament results
    results = {}
    in_results = False
    for line in output_lines:
        if "Tournament Results" in line or "Strategy Tournament Complete" in line:
            in_results = True
            continue
        elif in_results and line.strip() and not line.startswith('Results saved to:'):
            # Parse result lines
            if ':' in line and ('wins' in line or 'losses' in line or 'draws' in line):
                parts = line.split(':')
                if len(parts) >= 2:
                    strategy = parts[0].strip()
                    stats = parts[1].strip()
                    results[strategy] = stats
    
    return {
        'returncode': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'results': results
    }


def compare_results(random_results: Dict[str, Any], deterministic_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare results from random and deterministic tournaments."""
    comparison = {
        'random_results': random_results.get('results', {}),
        'deterministic_results': deterministic_results.get('results', {}),
        'strategies': set(),
        'analysis': {}
    }
    
    # Collect all strategies
    if random_results:
        comparison['strategies'].update(random_results.get('results', {}).keys())
    if deterministic_results:
        comparison['strategies'].update(deterministic_results.get('results', {}).keys())
    
    # Analyze each strategy
    for strategy in comparison['strategies']:
        random_stats = random_results.get('results', {}).get(strategy, 'N/A')
        deterministic_stats = deterministic_results.get('results', {}).get(strategy, 'N/A')
        
        comparison['analysis'][strategy] = {
            'random': random_stats,
            'deterministic': deterministic_stats,
            'consistent': random_stats == deterministic_stats if random_stats != 'N/A' and deterministic_stats != 'N/A' else False
        }
    
    return comparison


def print_comparison(comparison: Dict[str, Any]):
    """Print a formatted comparison of results."""
    print(f"\n{'='*80}")
    print("TOURNAMENT METHOD COMPARISON")
    print(f"{'='*80}")
    
    print(f"\nStrategies tested: {', '.join(sorted(comparison['strategies']))}")
    
    print(f"\n{'Strategy':<20} {'Random':<30} {'Deterministic':<30} {'Consistent'}")
    print("-" * 90)
    
    for strategy in sorted(comparison['strategies']):
        analysis = comparison['analysis'][strategy]
        consistent = "✓" if analysis['consistent'] else "✗"
        print(f"{strategy:<20} {analysis['random']:<30} {analysis['deterministic']:<30} {consistent}")
    
    # Summary statistics
    total_strategies = len(comparison['strategies'])
    consistent_strategies = sum(1 for a in comparison['analysis'].values() if a['consistent'])
    
    print(f"\nSummary:")
    print(f"  Total strategies: {total_strategies}")
    print(f"  Consistent results: {consistent_strategies}")
    print(f"  Consistency rate: {consistent_strategies/total_strategies*100:.1f}%")


def save_comparison_report(comparison: Dict[str, Any], output_file: str):
    """Save comparison results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison report saved to: {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare random vs deterministic tournament methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare policy vs MCTS with 50 games each
  %(prog)s --strategies=policy,mcts_100 --num-games=50 --num-openings=50
  
  # Compare multiple strategies
  %(prog)s --strategies=policy,mcts_122,fixed_tree_13_8 --num-games=100 --num-openings=100
        """
    )
    
    parser.add_argument('--strategies', type=str, required=True,
                       help='Comma-separated list of strategies to compare')
    parser.add_argument('--model', type=str, default='current_best',
                       help='Model to use (default: current_best)')
    parser.add_argument('--num-games', type=int, default=50,
                       help='Number of games for random tournament (default: 50)')
    parser.add_argument('--num-openings', type=int, default=50,
                       help='Number of openings for deterministic tournament (default: 50)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for random tournament (default: 1.0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output', type=str,
                       help='Output file for comparison report (JSON format)')
    parser.add_argument('--skip-random', action='store_true',
                       help='Skip random tournament (only run deterministic)')
    parser.add_argument('--skip-deterministic', action='store_true',
                       help='Skip deterministic tournament (only run random)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set environment
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    
    # Build commands
    strategies_arg = args.strategies
    
    # Random tournament command
    random_cmd = [
        'python', 'scripts/run_strategy_tournament.py',
        '--model', args.model,
        '--strategies', strategies_arg,
        '--num-games', str(args.num_games),
        '--temperature', str(args.temperature),
        '--seed', str(args.seed),
        '--verbose', '0'  # Reduce output
    ]
    
    # Deterministic tournament command
    deterministic_cmd = [
        'python', 'scripts/run_deterministic_tournament.py',
        '--model', args.model,
        '--strategies', strategies_arg,
        '--num-openings', str(args.num_openings),
        '--seed', str(args.seed),
        '--verbose', '0'  # Reduce output
    ]
    
    # Run tournaments
    random_results = None
    deterministic_results = None
    
    if not args.skip_random:
        random_results = run_tournament_command(random_cmd, "RANDOM TOURNAMENT")
        if random_results is None:
            print("ERROR: Random tournament failed")
            sys.exit(1)
    
    if not args.skip_deterministic:
        deterministic_results = run_tournament_command(deterministic_cmd, "DETERMINISTIC TOURNAMENT")
        if deterministic_results is None:
            print("ERROR: Deterministic tournament failed")
            sys.exit(1)
    
    # Compare results
    if random_results and deterministic_results:
        comparison = compare_results(random_results, deterministic_results)
        print_comparison(comparison)
        
        # Save report if requested
        if args.output:
            save_comparison_report(comparison, args.output)
    
    elif random_results:
        print("\nRandom tournament completed successfully")
        print("Results:", random_results.get('results', {}))
    
    elif deterministic_results:
        print("\nDeterministic tournament completed successfully")
        print("Results:", deterministic_results.get('results', {}))
    
    print(f"\nComparison complete!")


if __name__ == "__main__":
    main()
