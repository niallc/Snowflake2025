#!/usr/bin/env python3
"""
Script to run large-scale self-play using the enhanced inference system.
"""

import sys
import os
import argparse
import time
from datetime import datetime

# Add the parent directory to the path so we can import hex_ai modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hex_ai.selfplay.selfplay_engine import SelfPlayEngine


def main():
    parser = argparse.ArgumentParser(description='Run large-scale self-play')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the model checkpoint')
    parser.add_argument('--num-games', type=int, default=1000,
                       help='Number of games to generate (default: 1000)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Maximum batch size for inference (default: 100)')
    parser.add_argument('--cache-size', type=int, default=10000,
                       help='Size of inference cache (default: 10000)')
    parser.add_argument('--search-widths', type=int, nargs='+', default=[3, 2],
                       help='Search widths for minimax search (default: 3 2)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for move selection (default: 1.0)')
    parser.add_argument('--output-dir', type=str, default='data/sf25/jul29/',
                       help='Output directory for generated games (default: selfplay_data)')
    parser.add_argument('--progress-interval', type=int, default=50,
                       help='Progress update interval (default: 50)')
    parser.add_argument('--disable-caching', action='store_true',
                       help='Disable inference caching')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2],
                       help='Verbosity level: 0=quiet, 1=normal, 2=detailed (default: 1)')
    parser.add_argument('--cautious-mode', action='store_true',
                       help='Run with cautious settings (5 games, verbose=2, detailed moves)')
    parser.add_argument('--production-mode', action='store_true',
                       help='Run with production settings (minimal output, essential data only, streaming save)')
    parser.add_argument('--streaming-save', action='store_true',
                       help='Save games incrementally to avoid data loss on interruption')
    parser.add_argument('--save-essential-only', action='store_true',
                       help='Save only essential game data (no detailed move-by-move info)')
    
    args = parser.parse_args()
    
    # Handle mode overrides
    if args.cautious_mode:
        args.num_games = 5
        args.verbose = 2
        print("Running in cautious mode: 5 games, verbose=2, detailed CSV output")
    elif args.production_mode:
        args.num_games = 5  # Keep small for testing
        args.verbose = 1  # Show progress dots but suppress detailed logs
        args.save_essential_only = True  # Only essential data
        args.streaming_save = True  # Stream to avoid data loss
        print("Running in production mode: 5 games, progress dots, essential data only, streaming save")
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found: {args.model_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp for output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.join(args.output_dir, f"selfplay_games_{timestamp}")
    
    if args.verbose >= 1:
        print("Large-Scale Self-Play Generation")
        print("=" * 50)
        print(f"Model: {args.model_path}")
        print(f"Games to generate: {args.num_games}")
        print(f"Workers: {args.num_workers}")
        print(f"Batch size: {args.batch_size}")
        print(f"Cache size: {args.cache_size}")
        print(f"Search widths: {args.search_widths}")
        print(f"Temperature: {args.temperature}")
        print(f"Verbose: {args.verbose}")
        print(f"Output base: {base_filename}")
        print(f"Caching: {'enabled' if not args.disable_caching else 'disabled'}")
        print()
    
    # Initialize self-play engine
    engine = SelfPlayEngine(
        model_path=args.model_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        cache_size=args.cache_size,
        search_widths=args.search_widths,
        temperature=args.temperature,
        enable_caching=not args.disable_caching,
        verbose=args.verbose,
        save_essential_only=args.save_essential_only,
        streaming_save=args.streaming_save
    )
    
    # Generate games
    start_time = time.time()
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
        if args.save_detailed_csv:
            compressed_file, csv_dir = engine.save_games_with_details(games, base_filename)
        else:
            compressed_file = f"{base_filename}.pkl.gz"
            engine.save_games_to_file(games, compressed_file)
            csv_dir = None
        
        # Print final statistics
        total_time = time.time() - start_time
        total_moves = sum(len(game['moves']) for game in games)
        
        if args.verbose >= 1:
            print(f"\n=== Generation Complete ===")
            print(f"Games generated: {len(games)}")
            print(f"Total time: {total_time:.1f}s")
            if len(games) > 0:
                print(f"Games per second: {len(games) / total_time:.1f}")
                print(f"Total moves: {total_moves}")
                print(f"Average moves per game: {total_moves / len(games):.1f}")
                
                # Policy vs Minimax agreement
                agreement_rates = [game.get('policy_minimax_agreement_rate', 0) for game in games]
                avg_agreement = sum(agreement_rates) / len(agreement_rates)
                print(f"Average policy-minimax agreement: {avg_agreement:.1%}")
            else:
                print("No games generated successfully.")
            print(f"Compressed data: {compressed_file}")
            if csv_dir:
                print(f"Detailed CSV data: {csv_dir}")
    
    # Print model performance summary
    engine.shutdown()


if __name__ == "__main__":
    main()