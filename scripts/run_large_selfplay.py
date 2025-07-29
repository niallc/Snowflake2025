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
    parser.add_argument('--output-dir', type=str, default='selfplay_data',
                       help='Output directory for generated games (default: selfplay_data)')
    parser.add_argument('--progress-interval', type=int, default=50,
                       help='Progress update interval (default: 50)')
    parser.add_argument('--disable-caching', action='store_true',
                       help='Disable inference caching')
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found: {args.model_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp for output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"selfplay_games_{timestamp}.pkl.gz")
    
    print("Large-Scale Self-Play Generation")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Games to generate: {args.num_games}")
    print(f"Workers: {args.num_workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Cache size: {args.cache_size}")
    print(f"Search widths: {args.search_widths}")
    print(f"Temperature: {args.temperature}")
    print(f"Output file: {output_file}")
    print(f"Caching: {'enabled' if not args.disable_caching else 'disabled'}")
    print()
    
    # Initialize self-play engine
    print("Initializing self-play engine...")
    engine = SelfPlayEngine(
        model_path=args.model_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        cache_size=args.cache_size,
        search_widths=args.search_widths,
        temperature=args.temperature,
        enable_caching=not args.disable_caching
    )
    
    # Generate games
    print(f"Starting generation of {args.num_games} games...")
    start_time = time.time()
    
    try:
        games = engine.generate_games_with_monitoring(
            num_games=args.num_games,
            progress_interval=args.progress_interval
        )
        
        # Save games
        print(f"\nSaving {len(games)} games to {output_file}...")
        engine.save_games_to_file(games, output_file)
        
        # Print final statistics
        total_time = time.time() - start_time
        total_moves = sum(len(game['moves']) for game in games)
        
        print(f"\n=== Generation Complete ===")
        print(f"Games generated: {len(games)}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Games per second: {len(games) / total_time:.1f}")
        print(f"Total moves: {total_moves}")
        print(f"Average moves per game: {total_moves / len(games):.1f}")
        print(f"Output file: {output_file}")
        
        # Print model performance summary
        engine.shutdown()
        
    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()