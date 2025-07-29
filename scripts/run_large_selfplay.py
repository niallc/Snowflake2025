#!/usr/bin/env python3
"""
Large-scale self-play generation script with optimized performance.
"""

import argparse
import sys
import os
import time
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hex_ai.selfplay.selfplay_engine import SelfPlayEngine


def main():
    parser = argparse.ArgumentParser(description="Generate large-scale self-play games")
    parser.add_argument('--num_games', type=int, default=1000, help='Number of games to generate')
    parser.add_argument('--model_path', type=str, 
                       default="checkpoints/hyperparameter_tuning/loss_weight_sweep_exp0_bs256_98f719_20250724_233408/epoch2_mini16.pt.gz",
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='data/sf25/jul29', help='Output directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--cache_size', type=int, default=10000, help='Cache size for model inference')
    parser.add_argument('--search_widths', type=int, nargs='+', default=[3, 2], 
                       help='Search widths for minimax (e.g., 3 2 for width 3 at depth 1, width 2 at depth 2)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for move sampling')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level (0=quiet, 1=normal, 2=detailed)')
    parser.add_argument('--streaming_save', action='store_true', 
                       help='Save games incrementally to avoid data loss')
    parser.add_argument('--no_batched_inference', action='store_true',
                       help='Disable batched inference (use individual calls)')
    parser.add_argument('--progress_interval', type=int, default=10, 
                       help='How often to print progress updates')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"=== Large-Scale Self-Play Generation ===")
    print(f"Model: {args.model_path}")
    print(f"Games: {args.num_games}")
    print(f"Workers: {args.num_workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Search widths: {args.search_widths}")
    print(f"Temperature: {args.temperature}")
    print(f"Batched inference: {not args.no_batched_inference}")
    print(f"Output directory: {args.output_dir}")
    print(f"Timestamp: {timestamp}")
    print()
    
    # Initialize self-play engine
    engine = SelfPlayEngine(
        model_path=args.model_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        cache_size=args.cache_size,
        search_widths=args.search_widths,
        temperature=args.temperature,
        verbose=args.verbose,
        streaming_save=args.streaming_save,
        use_batched_inference=not args.no_batched_inference
    )
    
    start_time = time.time()
    
    try:
        # Generate games
        if args.streaming_save:
            games = engine.generate_games_streaming(
                num_games=args.num_games,
                progress_interval=args.progress_interval
            )
        else:
            games = engine.generate_games_with_monitoring(
                num_games=args.num_games,
                progress_interval=args.progress_interval
            )
        
        # Save games
        if games:
            # Save with detailed move data
            base_filename = f"{args.output_dir}/selfplay_{timestamp}"
            compressed_file, csv_dir = engine.save_games_with_details(games, base_filename)
            print(f"\nSaved detailed games:")
            print(f"  Compressed: {compressed_file}")
            print(f"  CSV details: {csv_dir}")
        
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