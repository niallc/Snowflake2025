#!/usr/bin/env python3
"""
Large-scale self-play generation script with optimized performance.
"""

import argparse
import numpy as np
import os
import random
import sys
import time
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hex_ai.inference.model_config import get_model_path
from hex_ai.selfplay.selfplay_engine import SelfPlayEngine
from hex_ai.utils.opening_strategies import create_pie_rule_strategy, RandomOpeningStrategy



def main():
    parser = argparse.ArgumentParser(description="Generate large-scale self-play games")
    parser.add_argument('--num_games', type=int, default=1000, help='Number of games to generate')
    parser.add_argument('--model_path', type=str, 
                       default=get_model_path("current_best"),
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='data/sf25/aug02', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for inference')
    parser.add_argument('--cache_size', type=int, default=60000, help='Cache size for model inference')
    parser.add_argument('--mcts_sims', type=int, default=500, 
                       help='Number of MCTS simulations per move')
    parser.add_argument('--temperature', type=float, default=0.5, help='Starting temperature for move sampling')
    parser.add_argument('--temperature_end', type=float, default=0.01, help='Final temperature for move sampling (for decay)')
    parser.add_argument('--opening_strategy', type=str, default='pie_rule', 
                       choices=['pie_rule', 'random', 'none'],
                       help='Opening strategy: pie_rule (default), random, or none')
    parser.add_argument('--bad_move_frequency', type=float, default=0.1,
                       help='Frequency of bad moves in pie rule openings (0.0-1.0)')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level (0=quiet, 1=normal, 2=detailed)')
    parser.add_argument('--streaming_save', action='store_true', 
                       help='Save games incrementally to avoid data loss')
    parser.add_argument('--no_batched_inference', action='store_true',
                       help='Disable batched inference (use individual calls)')
    parser.add_argument('--progress_interval', type=int, default=10, 
                       help='How often to print progress updates')
    
    args = parser.parse_args()
    
    # Don't set global seeds - let each game use different randomness
    # This ensures games are diverse while maintaining reproducibility within each game
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"=== Large-Scale Self-Play Generation ===")
    print(f"Model: {args.model_path}")
    print(f"Games: {args.num_games}")
    print(f"Batch size: {args.batch_size}")
    print(f"Search method: MCTS ({args.mcts_sims} simulations)")
    print(f"Temperature: {args.temperature} -> {args.temperature_end}")
    print(f"Opening strategy: {args.opening_strategy}")
    if args.opening_strategy == 'pie_rule':
        print(f"Bad move frequency: {args.bad_move_frequency}")
    print(f"Batched inference: {not args.no_batched_inference}")
    print(f"Output directory: {args.output_dir}")
    print(f"Timestamp: {timestamp}")
    

    
    # Note about execution configuration
    if not args.no_batched_inference:
        print(f"\nNOTE: Using single-threaded execution with batched inference.")
        print("This is the recommended configuration for optimal performance.")
    else:
        print(f"\nNOTE: Using individual inference calls.")
        print("This configuration may provide better performance for non-batched inference.")
    
    print()
    
    # Create opening strategy
    opening_strategy = None
    if args.opening_strategy != 'none':
        if args.opening_strategy == 'pie_rule':
            opening_strategy = create_pie_rule_strategy(
                board_size=13,
                bad_move_frequency=args.bad_move_frequency
            )
        elif args.opening_strategy == 'random':
            # Create random strategy with some common opening moves
            common_moves = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12)]
            opening_strategy = RandomOpeningStrategy(common_moves, board_size=13, empty_board_prob=0.1)
    
    # Initialize self-play engine
    engine = SelfPlayEngine(
        model_path=args.model_path,
        batch_size=args.batch_size,
        cache_size=args.cache_size,
        temperature=args.temperature,
        temperature_end=args.temperature_end,
        verbose=args.verbose,
        streaming_save=args.streaming_save,
        use_batched_inference=not args.no_batched_inference,
        output_dir=args.output_dir,
        mcts_sims=args.mcts_sims
    )
    
    start_time = time.time()
    
    try:
        # Generate games
        if args.streaming_save:
            games = engine.generate_games_streaming(
                num_games=args.num_games,
                board_size=13,
                progress_interval=args.progress_interval,
                opening_strategy=opening_strategy
            )
        else:
            games = engine.generate_games_with_monitoring(
                num_games=args.num_games,
                board_size=13,
                progress_interval=args.progress_interval,
                opening_strategy=opening_strategy
            )
        
        # Save games
        if games:
            # Save as TRMPH text file
            base_filename = f"{args.output_dir}/selfplay_{timestamp}"
            trmph_file = engine.save_games_simple(games, base_filename)
            print(f"\nSaved games:")
            print(f"  TRMPH file: {trmph_file}")
        
        # Print final statistics
        total_time = time.time() - start_time
        print(f"\n=== Generation Complete ===")
        print(f"Total time: {total_time:.1f}s")
        print(f"Games per second: {len(games) / total_time:.1f}")
        
        # Winner distribution
        if games:
            winners = [game.get('winner', 'unknown') for game in games]
            red_wins = winners.count('r')
            blue_wins = winners.count('b')
            print(f"Winner distribution: Red {red_wins}, Blue {blue_wins}")
            print(f"Red win rate: {red_wins / len(games):.1%}")
        
        # Performance statistics
        stats = engine.get_performance_stats()
        if 'model' in stats:
            model_stats = stats['model']
            print(f"\n=== Model Performance ===")
            print(f"Total inferences: {model_stats.get('total_inferences', 0)}")
            print(f"Cache hit rate: {model_stats.get('cache', {}).get('hit_rate', 0):.1%}")
            print(f"Average batch size: {model_stats.get('avg_batch_size', 0):.1f}")
            print(f"Throughput: {model_stats.get('throughput', 0):.1f} boards/s")
        
    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user.")
        if args.streaming_save:
            print("Games saved incrementally - no data loss.")
    except Exception as e:
        print(f"\nError during generation: {e}")
        raise
    finally:
        # Clean shutdown
        engine.shutdown()


if __name__ == "__main__":
    main()