#!/usr/bin/env python3
"""
Large-scale self-play generation script with optimized performance.
"""

import argparse
import numpy as np
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

# Environment validation is now handled automatically in hex_ai/__init__.py

from hex_ai.config import DEFAULT_GUMBEL_SIM_THRESHOLD, DEFAULT_C_PUCT, DEFAULT_MCTS_SIMS, DEFAULT_CACHE_SIZE, BOARD_SIZE, DEFAULT_TEMPERATURE_START, DEFAULT_TEMPERATURE_END
from hex_ai.inference.model_config import get_model_path
from hex_ai.selfplay.selfplay_engine import (
    DEFAULT_SELFPLAY_CONFIDENCE_TERMINATION_THRESHOLD,
    SelfPlayEngine,
)
from hex_ai.system_utils import get_git_commit_info
from hex_ai.utils.opening_strategies import create_pie_rule_strategy, RandomOpeningStrategy
from hex_ai.utils.tournament_logging import get_command_line
from hex_ai.utils.gumbel_utils import generate_gumbel_summary_from_params
from hex_ai.utils.run_state_store import (
    JsonRunStateStore,
    RunStateMismatchError,
    compute_config_fingerprint,
    utc_now_iso,
)
from hex_ai.utils.script_logging import ScriptConfig, print_script_configuration, print_script_results


DEFAULT_RESTART_STATE_FILENAME = "selfplay_restart_state.json"
CHUNKED_RUN_TYPE = "selfplay_chunked"


def parse_args() -> argparse.Namespace:
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
    parser.add_argument(
        '--restart-every-games',
        type=int,
        default=0,
        help=(
            'If >0, run in chunked subprocess mode and restart the Python process '
            'after every N generated games.'
        ),
    )
    parser.add_argument(
        '--state-file',
        type=str,
        help=(
            'Path to JSON run state file for chunked restart mode '
            '(default: config-scoped file under <output_dir>).'
        ),
    )
    parser.add_argument(
        '--reset-run-state',
        action='store_true',
        help='Delete any existing chunked run state and start from game 0.',
    )

    # Lightweight MCTS timing profiler (GPU vs CPU breakdown)
    parser.add_argument('--mcts-profile', action='store_true',
                       help='Print lightweight MCTS timing breakdown every N calls (GPU vs CPU time).')
    parser.add_argument('--mcts-profile-every', type=int, default=10,
                       help='Print MCTS profile once every N MCTS move selections (default: 10).')
    parser.add_argument('--mcts-profile-max-calls', type=int, default=50,
                       help='Maximum number of MCTS move selections to profile (default: 50).')
    parser.add_argument('--internal-chunk-run', action='store_true', help=argparse.SUPPRESS)
    return parser.parse_args()


def _resolve_state_file(args: argparse.Namespace) -> str:
    if args.state_file:
        return args.state_file
    return os.path.join(args.output_dir, DEFAULT_RESTART_STATE_FILENAME)


def _build_scoped_state_file(output_dir: str, config_fingerprint: str) -> str:
    fingerprint_prefix = config_fingerprint[:12]
    return os.path.join(
        output_dir,
        f"{DEFAULT_RESTART_STATE_FILENAME.removesuffix('.json')}_{fingerprint_prefix}.json",
    )


def _resolve_chunked_state_file(args: argparse.Namespace, config_fingerprint: str) -> str:
    """
    Resolve run-state path for chunked mode.

    Behavior:
    - Explicit --state-file remains authoritative.
    - Default mode uses a config-scoped state file to support parallel runs.
    - Legacy default file is auto-resumed if compatible to preserve existing runs.
    """
    if args.state_file:
        return args.state_file

    legacy_default_file = _resolve_state_file(args)
    scoped_file = _build_scoped_state_file(args.output_dir, config_fingerprint)

    # Prefer scoped file for new runs.
    if os.path.exists(scoped_file):
        return scoped_file

    # Backward compatibility: if a legacy default file exists and matches this config,
    # continue using it so existing runs can resume naturally.
    if os.path.exists(legacy_default_file):
        legacy_store = JsonRunStateStore(legacy_default_file)
        legacy_state = legacy_store.load()
        if legacy_state is not None:
            try:
                legacy_store.assert_compatible(
                    legacy_state,
                    run_type=CHUNKED_RUN_TYPE,
                    config_fingerprint=config_fingerprint,
                )
                print(f"Resuming compatible legacy run state: {legacy_default_file}")
                return legacy_default_file
            except RunStateMismatchError:
                print(
                    "Detected incompatible legacy run state; "
                    f"starting fresh config-scoped state: {scoped_file}"
                )

    return scoped_file


def _build_chunked_config_snapshot(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "num_games_total": args.num_games,
        "model_path": args.model_path,
        "output_dir": args.output_dir,
        "batch_size": args.batch_size,
        "cache_size": args.cache_size,
        "mcts_sims": args.mcts_sims,
        "c_puct": args.c_puct,
        "disable_gumbel": args.disable_gumbel,
        "temperature": args.temperature,
        "temperature_end": args.temperature_end,
        "opening_strategy": args.opening_strategy,
        "bad_move_frequency": args.bad_move_frequency,
        "verbose": args.verbose,
        "streaming_save": args.streaming_save,
        "no_batched_inference": args.no_batched_inference,
        "progress_interval": args.progress_interval,
        "mcts_profile": args.mcts_profile,
        "mcts_profile_every": args.mcts_profile_every,
        "mcts_profile_max_calls": args.mcts_profile_max_calls,
        "restart_every_games": args.restart_every_games,
    }


def _build_chunk_command(args: argparse.Namespace, chunk_games: int) -> List[str]:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--num_games",
        str(chunk_games),
        "--model_path",
        args.model_path,
        "--output_dir",
        args.output_dir,
        "--batch_size",
        str(args.batch_size),
        "--cache_size",
        str(args.cache_size),
        "--mcts_sims",
        str(args.mcts_sims),
        "--c-puct",
        str(args.c_puct),
        "--temperature",
        str(args.temperature),
        "--temperature_end",
        str(args.temperature_end),
        "--opening_strategy",
        args.opening_strategy,
        "--bad_move_frequency",
        str(args.bad_move_frequency),
        "--verbose",
        str(args.verbose),
        "--progress_interval",
        str(args.progress_interval),
        "--mcts-profile-every",
        str(args.mcts_profile_every),
        "--mcts-profile-max-calls",
        str(args.mcts_profile_max_calls),
        "--internal-chunk-run",
    ]
    if args.disable_gumbel:
        cmd.append("--disable-gumbel")
    if args.streaming_save:
        cmd.append("--streaming_save")
    if args.no_batched_inference:
        cmd.append("--no_batched_inference")
    if args.mcts_profile:
        cmd.append("--mcts-profile")
    return cmd


def _run_chunked_selfplay(args: argparse.Namespace) -> int:
    if args.restart_every_games <= 0:
        raise ValueError("--restart-every-games must be > 0 when chunked mode is enabled")

    os.makedirs(args.output_dir, exist_ok=True)
    config_snapshot = _build_chunked_config_snapshot(args)
    config_fingerprint = compute_config_fingerprint(config_snapshot)
    state_file = _resolve_chunked_state_file(args, config_fingerprint)
    state_store = JsonRunStateStore(state_file)

    if args.reset_run_state:
        state_store.delete()
        print(f"Reset run state: {state_file}")

    state = state_store.load()
    if state is None:
        state = state_store.create(
            run_type=CHUNKED_RUN_TYPE,
            config_snapshot=config_snapshot,
            progress={
                "total_games": args.num_games,
                "games_completed": 0,
                "chunks_completed": 0,
                "restart_every_games": args.restart_every_games,
                "current_chunk": None,
                "current_chunk_games": None,
                "current_chunk_started_at": None,
                "last_chunk_completed_at": None,
                "last_chunk_games": 0,
                "last_error": None,
                "chunk_history": [],
            },
            metadata={"output_dir": args.output_dir, "model_path": args.model_path},
        )
        print(f"Initialized new run state: {state_file}")
    else:
        state_store.assert_compatible(
            state,
            run_type=CHUNKED_RUN_TYPE,
            config_fingerprint=config_fingerprint,
        )
        print(f"Resuming existing run state: {state_file}")

    progress = state.get("progress")
    if not isinstance(progress, dict):
        raise RuntimeError(f"Invalid run state format in {state_file}: 'progress' must be a dict")

    total_games = int(progress.get("total_games", 0))
    games_completed = int(progress.get("games_completed", 0))
    chunks_completed = int(progress.get("chunks_completed", 0))

    if total_games <= 0:
        raise RuntimeError(f"Invalid total_games in {state_file}: {total_games}")
    if games_completed < 0 or games_completed > total_games:
        raise RuntimeError(
            f"Invalid games_completed in {state_file}: {games_completed} (total_games={total_games})"
        )

    print(
        f"Chunked restart mode: total_games={total_games}, restart_every={args.restart_every_games}, "
        f"completed={games_completed}, completed_chunks={chunks_completed}"
    )

    if games_completed >= total_games:
        state["status"] = "completed"
        state["completed_at"] = state.get("completed_at") or utc_now_iso()
        state_store.save(state)
        print("Run already complete according to saved state.")
        return 0

    try:
        while games_completed < total_games:
            remaining_games = total_games - games_completed
            chunk_games = min(args.restart_every_games, remaining_games)
            chunk_index = chunks_completed + 1
            chunk_started_at = utc_now_iso()

            progress["current_chunk"] = chunk_index
            progress["current_chunk_games"] = chunk_games
            progress["current_chunk_started_at"] = chunk_started_at
            state["status"] = "running"
            state["completed_at"] = None
            state = state_store.save(state)

            print(
                f"\n[Chunk {chunk_index}] Starting child process for {chunk_games} games "
                f"({games_completed}/{total_games} completed so far)"
            )
            child_cmd = _build_chunk_command(args, chunk_games)
            child_result = subprocess.run(child_cmd, check=False)

            if child_result.returncode != 0:
                progress["current_chunk"] = None
                progress["current_chunk_games"] = None
                progress["current_chunk_started_at"] = None
                progress["last_error"] = (
                    f"Chunk {chunk_index} failed with exit code {child_result.returncode}"
                )
                state["status"] = "failed"
                state = state_store.save(state)
                print(
                    f"Chunk {chunk_index} failed (exit {child_result.returncode}). "
                    f"Saved progress to {state_file}."
                )
                return child_result.returncode

            chunk_completed_at = utc_now_iso()
            games_completed += chunk_games
            chunks_completed += 1
            progress["games_completed"] = games_completed
            progress["chunks_completed"] = chunks_completed
            progress["last_chunk_completed_at"] = chunk_completed_at
            progress["last_chunk_games"] = chunk_games
            progress["current_chunk"] = None
            progress["current_chunk_games"] = None
            progress["current_chunk_started_at"] = None
            progress["last_error"] = None

            chunk_history = progress.get("chunk_history")
            if not isinstance(chunk_history, list):
                raise RuntimeError(
                    f"Invalid run state format in {state_file}: 'chunk_history' must be a list"
                )
            chunk_history.append(
                {
                    "chunk_index": chunk_index,
                    "games_requested": chunk_games,
                    "started_at": chunk_started_at,
                    "completed_at": chunk_completed_at,
                    "exit_code": 0,
                }
            )

            if games_completed >= total_games:
                state["status"] = "completed"
                state["completed_at"] = chunk_completed_at
            else:
                state["status"] = "running"
                state["completed_at"] = None

            state = state_store.save(state)
            print(
                f"[Chunk {chunk_index}] Completed successfully. "
                f"Progress: {games_completed}/{total_games} games."
            )

        print(f"\nChunked self-play run completed. State saved to {state_file}.")
        return 0
    except KeyboardInterrupt:
        progress["current_chunk"] = None
        progress["current_chunk_games"] = None
        progress["current_chunk_started_at"] = None
        progress["last_error"] = "Interrupted by user"
        state["status"] = "interrupted"
        state["completed_at"] = None
        state_store.save(state)
        print(f"\nInterrupted by user. Progress saved to {state_file}.")
        return 1


def _run_single_process(args: argparse.Namespace) -> None:
    
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
        confidence_termination_threshold=DEFAULT_SELFPLAY_CONFIDENCE_TERMINATION_THRESHOLD,
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
        confidence_termination_threshold=DEFAULT_SELFPLAY_CONFIDENCE_TERMINATION_THRESHOLD,
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


def main():
    args = parse_args()

    if args.num_games <= 0:
        raise ValueError("--num_games must be > 0")
    if args.restart_every_games < 0:
        raise ValueError("--restart-every-games cannot be negative")

    try:
        if args.restart_every_games > 0 and not args.internal_chunk_run:
            exit_code = _run_chunked_selfplay(args)
            sys.exit(exit_code)
        _run_single_process(args)
    except RunStateMismatchError as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
