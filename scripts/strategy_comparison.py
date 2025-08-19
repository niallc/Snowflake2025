#!/usr/bin/env python3
"""
Systematic comparison of different inference strategies using tournaments.

This script runs multiple tournaments with different move selection strategies
(policy, MCTS, fixed tree search) and their configurations to provide
comprehensive performance analysis.
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hex_ai.inference.tournament import TournamentConfig, TournamentPlayConfig, run_round_robin_tournament
from hex_ai.inference.move_selection import list_available_strategies


def create_strategy_configs():
    """Create different strategy configurations to compare."""
    return [
        # Policy-based selection (baseline)
        {
            "name": "policy_baseline",
            "strategy": "policy",
            "config": {},
            "description": "Direct policy sampling (fastest)"
        },
        
        # MCTS with different simulation counts
        {
            "name": "mcts_50_sims",
            "strategy": "mcts",
            "config": {"mcts_sims": 50, "mcts_c_puct": 1.5},
            "description": "MCTS with 50 simulations"
        },
        {
            "name": "mcts_100_sims", 
            "strategy": "mcts",
            "config": {"mcts_sims": 100, "mcts_c_puct": 1.5},
            "description": "MCTS with 100 simulations"
        },
        {
            "name": "mcts_200_sims",
            "strategy": "mcts", 
            "config": {"mcts_sims": 200, "mcts_c_puct": 1.5},
            "description": "MCTS with 200 simulations"
        },
        
        # Fixed tree search with different widths
        {
            "name": "fixed_tree_narrow",
            "strategy": "fixed_tree",
            "config": {"search_widths": [10, 5]},
            "description": "Fixed tree search (narrow)"
        },
        {
            "name": "fixed_tree_medium",
            "strategy": "fixed_tree", 
            "config": {"search_widths": [20, 10, 5]},
            "description": "Fixed tree search (medium)"
        },
        {
            "name": "fixed_tree_wide",
            "strategy": "fixed_tree",
            "config": {"search_widths": [30, 15, 8]},
            "description": "Fixed tree search (wide)"
        },
    ]


def run_strategy_comparison(checkpoint_paths, num_games=20, temperature=1.0, 
                          pie_rule=False, verbose=1):
    """Run tournaments comparing different strategies."""
    
    strategy_configs = create_strategy_configs()
    results = {}
    
    print(f"Comparing {len(strategy_configs)} strategies")
    print(f"Models: {[os.path.basename(p) for p in checkpoint_paths]}")
    print(f"Games per pair: {num_games}")
    print(f"Temperature: {temperature}")
    print(f"Pie rule: {pie_rule}")
    print()
    
    # Create base tournament config
    base_config = TournamentConfig(
        checkpoint_paths=checkpoint_paths,
        num_games=num_games
    )
    
    # Create timestamp for this comparison
    timestamp = datetime.now().strftime('%y%m%d_%H%M')
    
    for i, strategy_info in enumerate(strategy_configs):
        print(f"Strategy {i+1}/{len(strategy_configs)}: {strategy_info['name']}")
        print(f"  Description: {strategy_info['description']}")
        
        # Create play config for this strategy
        play_config = TournamentPlayConfig(
            strategy=strategy_info['strategy'],
            strategy_config=strategy_info['config'],
            temperature=temperature,
            pie_rule=pie_rule,
            random_seed=42  # Fixed seed for reproducibility
        )
        
        # Create output files
        log_dir = f"data/strategy_comparison_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"{strategy_info['name']}.trmph")
        csv_file = os.path.join(log_dir, f"{strategy_info['name']}.csv")
        
        try:
            # Run tournament
            result = run_round_robin_tournament(
                base_config, verbose=verbose, 
                log_file=log_file, csv_file=csv_file, 
                play_config=play_config
            )
            
            # Store results
            results[strategy_info['name']] = {
                'result': result,
                'config': strategy_info,
                'log_file': log_file,
                'csv_file': csv_file
            }
            
            print(f"  ✓ Completed: {result.total_games} games")
            print(f"  Win rates: {result.win_rates()}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[strategy_info['name']] = {
                'error': str(e),
                'config': strategy_info
            }
        
        print()
    
    return results


def print_comparison_summary(results):
    """Print a summary comparing all strategies."""
    print("=" * 60)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 60)
    
    successful_results = {k: v for k, v in results.items() if 'result' in v}
    
    if not successful_results:
        print("No successful tournaments to compare.")
        return
    
    # Print win rates comparison
    print("\nWin Rates:")
    print("-" * 40)
    
    # Get all participants from the first successful result
    first_result = next(iter(successful_results.values()))['result']
    participants = first_result.participants
    
    # Print header
    strategy_names = list(successful_results.keys())
    header = f"{'Model':<30}"
    for strategy in strategy_names:
        header += f"{strategy:<15}"
    print(header)
    
    # Print win rates for each model
    for participant in participants:
        row = f"{os.path.basename(participant):<30}"
        for strategy_name in strategy_names:
            result = successful_results[strategy_name]['result']
            win_rate = result.win_rates().get(participant, 0.0)
            row += f"{win_rate*100:6.1f}%{'':<9}"
        print(row)
    
    # Print ELO ratings comparison
    print("\nElo Ratings:")
    print("-" * 40)
    
    for strategy_name, data in successful_results.items():
        result = data['result']
        elos = result.elo_ratings()
        print(f"\n{strategy_name}:")
        for name, elo in sorted(elos.items(), key=lambda x: -x[1]):
            print(f"  {os.path.basename(name)}: {elo:.1f}")
    
    # Print configuration details
    print("\nStrategy Configurations:")
    print("-" * 40)
    
    for strategy_name, data in results.items():
        config = data['config']
        print(f"\n{strategy_name}:")
        print(f"  Strategy: {config['strategy']}")
        print(f"  Config: {config['config']}")
        print(f"  Description: {config['description']}")
        if 'error' in data:
            print(f"  Status: FAILED - {data['error']}")
        else:
            print(f"  Status: SUCCESS - {data['result'].total_games} games")


def main():
    parser = argparse.ArgumentParser(
        description='Compare different inference strategies using tournaments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare strategies with default models
  %(prog)s --num-games=20
  
  # Compare strategies with specific models
  %(prog)s --checkpoints="epoch1_mini1.pt.gz,epoch2_mini16.pt.gz" --num-games=50
  
  # Quick comparison with fewer games
  %(prog)s --num-games=10 --temperature=1.5
        """
    )
    
    parser.add_argument('--checkpoints', type=str,
                       help='Comma-separated list of checkpoint filenames')
    parser.add_argument('--checkpoint-dirs', type=str,
                       help='Comma-separated list of checkpoint directories')
    parser.add_argument('--num-games', type=int, default=20,
                       help='Number of games per pair (default: 20)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for move selection (default: 1.0)')
    parser.add_argument('--pie-rule', action='store_true',
                       help='Enable pie rule (disabled by default)')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (default: 1)')
    
    args = parser.parse_args()
    
    # Determine checkpoint paths
    if args.checkpoints:
        checkpoint_names = [name.strip() for name in args.checkpoints.split(',')]
    else:
        # Use default checkpoints
        checkpoint_names = ["epoch1_mini1.pt.gz", "epoch2_mini16.pt.gz"]
    
    # Build checkpoint paths
    if args.checkpoint_dirs:
        checkpoint_dirs = [dir_name.strip() for dir_name in args.checkpoint_dirs.split(',')]
    else:
        # Use default directory
        checkpoint_dirs = ["checkpoints/current_best"]
    
    # Construct full paths
    if len(checkpoint_dirs) == 1:
        base_dir = checkpoint_dirs[0]
        if not base_dir.startswith('checkpoints/'):
            base_dir = os.path.join("checkpoints/hyperparameter_tuning", base_dir)
        checkpoint_paths = [os.path.join(base_dir, fname) for fname in checkpoint_names]
    else:
        checkpoint_paths = []
        for dir_name, fname in zip(checkpoint_dirs, checkpoint_names):
            if not dir_name.startswith('checkpoints/'):
                dir_name = os.path.join("checkpoints/hyperparameter_tuning", dir_name)
            checkpoint_paths.append(os.path.join(dir_name, fname))
    
    # Check that checkpoints exist
    missing_paths = [p for p in checkpoint_paths if not os.path.isfile(p)]
    if missing_paths:
        print("ERROR: The following checkpoint files do not exist:")
        for p in missing_paths:
            print(f"  {p}")
        sys.exit(1)
    
    # Run the comparison
    results = run_strategy_comparison(
        checkpoint_paths=checkpoint_paths,
        num_games=args.num_games,
        temperature=args.temperature,
        pie_rule=args.pie_rule,
        verbose=args.verbose
    )
    
    # Print summary
    print_comparison_summary(results)


if __name__ == "__main__":
    main()
