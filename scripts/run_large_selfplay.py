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

# Environment validation is now handled automatically in hex_ai/__init__.py

from hex_ai.config import DEFAULT_GUMBEL_SIM_THRESHOLD, DEFAULT_C_PUCT, DEFAULT_MCTS_SIMS, DEFAULT_CACHE_SIZE, BOARD_SIZE, DEFAULT_TEMPERATURE_START, DEFAULT_TEMPERATURE_END
from hex_ai.inference.model_config import get_model_path
from hex_ai.selfplay.selfplay_engine import SelfPlayEngine
from hex_ai.system_utils import get_git_commit_info
from hex_ai.utils.opening_strategies import create_pie_rule_strategy, RandomOpeningStrategy
from hex_ai.utils.tournament_logging import get_command_line
from hex_ai.utils.gumbel_utils import generate_gumbel_summary_from_params
from hex_ai.utils.script_logging import ScriptConfig, print_script_configuration, print_script_results



def main():
    parser = argparse.ArgumentParser(description="Generate large-scale self-play games")
    parser.add_argument('--num_games', type=int, default=1000, help='Number of games to generate')
    parser.add_argument('--model_path', type=str, 
                       default=get_model_path("best"),
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='data/sf25/aug02', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for inference')
    parser.add_argument('--cache_size', type=int, default=DEFAULT_CACHE_SIZE, help=f'Cache size for model inference (default: {DEFAULT_CACHE_SIZE})')
    parser.add_argument('--mcts_sims', type=int, default=DEFAULT_MCTS_SIMS, 
                       help=f'Number of MCTS simulations per move (default: {DEFAULT_MCTS_SIMS})')
    parser.add_argument('--c-puct', type=float, default=DEFAULT_C_PUCT, 
                       help=f'PUCT exploration constant for MCTS (default: {DEFAULT_C_PUCT})')
    parser.add_argument('--disable-gumbel', action='store_true',
                       help='Disable Gumbel-AlphaZero root selection for MCTS (enabled by default)')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE_START, help=f'Starting temperature for move sampling (default: {DEFAULT_TEMPERATURE_START})')
    parser.add_argument('--temperature_end', type=float, default=DEFAULT_TEMPERATURE_END, help=f'Final temperature for move sampling (for decay) (default: {DEFAULT_TEMPERATURE_END})')
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
    parser.add_argument('--progress_interval', type=int, default=20, 
                       help='How often to print progress updates')

    # Lightweight MCTS timing profiler (GPU vs CPU breakdown)
    parser.add_argument('--mcts-profile', action='store_true',
                       help='Print lightweight MCTS timing breakdown every N calls (GPU vs CPU time).')
    parser.add_argument('--mcts-profile-every', type=int, default=10,
                       help='Print MCTS profile once every N MCTS move selections (default: 10).')
    parser.add_argument('--mcts-profile-max-calls', type=int, default=50,
                       help='Maximum number of MCTS move selections to profile (default: 50).')
    
    args = parser.parse_args()
    
    # Get command line early - crash if not available
    try:
        command_line = get_command_line()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Don't set global seeds - let each game use different randomness
    # This ensures games are diverse while maintaining reproducibility within each game
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Print configuration using unified logging
    script_config = ScriptConfig(
        script_type="selfplay",
        models=[args.model_path],
        strategies=[f"mcts_{args.mcts_sims}"],
        num_games=args.num_games,
        strategy_config={"mcts_sims": args.mcts_sims, "c_puct": args.c_puct},
        temperatures=args.temperature,
        pie_rule=False,  # Not applicable to selfplay
        opening_strategy=args.opening_strategy,
        batch_size=args.batch_size,
        cache_size=args.cache_size,
        mcts_sims=args.mcts_sims,
        c_puct=args.c_puct,
        enable_gumbel=not args.disable_gumbel,
        gumbel_sim_threshold=DEFAULT_GUMBEL_SIM_THRESHOLD,
        temperature_end=args.temperature_end,
        no_batched_inference=args.no_batched_inference,
        output_dir=args.output_dir
    )
    
    print_script_configuration(script_config)
    
    # Print additional selfplay specific info
    if args.opening_strategy == 'pie_rule':
        print(f"  Bad move frequency: {args.bad_move_frequency}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Timestamp: {timestamp}")
    print()
    
    # # Note about execution configuration
    # if not args.no_batched_inference:
    #     print(f"\nNOTE: Using single-threaded execution with batched inference.")
    #     print("This is the recommended configuration for optimal performance.")
    # else:
    #     print(f"\nNOTE: Using individual inference calls.")
    #     print("This configuration may provide better performance for non-batched inference.")
        
    # Create opening strategy
    opening_strategy = None
    if args.opening_strategy != 'none':
        if args.opening_strategy == 'pie_rule':
            opening_strategy = create_pie_rule_strategy(
                board_size=BOARD_SIZE,
                bad_move_frequency=args.bad_move_frequency
            )
        elif args.opening_strategy == 'random':
            # Create random strategy with some common opening moves
            common_moves = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12)]
            opening_strategy = RandomOpeningStrategy(common_moves, board_size=BOARD_SIZE, empty_board_prob=0.1)
    
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
        mcts_sims=args.mcts_sims,
        c_puct=args.c_puct,
        enable_gumbel=not args.disable_gumbel,
        command_line=command_line,
        mcts_profile=args.mcts_profile,
        mcts_profile_every=args.mcts_profile_every,
        mcts_profile_max_calls=args.mcts_profile_max_calls,
    )
    
    start_time = time.time()
    
    try:
        # Generate games
        if args.streaming_save:
            games = engine.generate_games_streaming(
                num_games=args.num_games,
                board_size=BOARD_SIZE,
                progress_interval=args.progress_interval,
                opening_strategy=opening_strategy
            )
        else:
            games = engine.generate_games_with_monitoring(
                num_games=args.num_games,
                board_size=BOARD_SIZE,
                progress_interval=args.progress_interval,
                opening_strategy=opening_strategy
            )
        
        # Save games and prepare results
        trmph_file = None
        if games:
            # Save as TRMPH text file
            base_filename = f"{args.output_dir}/selfplay_{timestamp}"
            trmph_file = engine.save_games_simple(games, base_filename)
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Print results using unified analyzer
        output_files = {}
        if trmph_file:
            output_files["trmph"] = trmph_file
        
        print_script_results("selfplay", games, script_config, output_files, total_time)
        
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