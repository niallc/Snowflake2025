#!/usr/bin/env python3
"""
Large-scale self-play generation script with optimized performance.
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# Environment validation is now handled automatically in hex_ai/__init__.py

from hex_ai.config import DEFAULT_GUMBEL_SIM_THRESHOLD, DEFAULT_C_PUCT, DEFAULT_MCTS_SIMS, DEFAULT_CACHE_SIZE, BOARD_SIZE, DEFAULT_TEMPERATURE_START, DEFAULT_TEMPERATURE_END
from hex_ai.inference.model_config import get_model_path
from hex_ai.move_provenance import (
    MOVE_CODE_VISIT_COUNT,
    make_move_provenance_record,
    sidecar_path_for_trmph,
)
from hex_ai.selfplay.selfplay_engine import (
    DEFAULT_SELFPLAY_CONFIDENCE_TERMINATION_THRESHOLD,
    SelfPlayEngine,
)
from hex_ai.selfplay.generation_summary import SelfPlayGenerationSummary
from hex_ai.utils.opening_strategies import (
    PIE_RULE_VALUE_BALANCED_CYCLE_LENGTH,
    PIE_RULE_VALUE_BALANCED_MIN_MOVE_PROBABILITY,
    PIE_RULE_VALUE_BALANCED_WEIGHT_EXPONENT,
    RandomOpeningStrategy,
    create_pie_rule_strategy,
)
from hex_ai.utils.tournament_logging import get_command_line
from hex_ai.utils.format_conversion import count_trmph_moves
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
    parser.add_argument(
        '--board-size',
        type=int,
        default=BOARD_SIZE,
        help=(
            f'Board size for self-play generation '
            f'(currently only {BOARD_SIZE} is supported).'
        ),
    )
    parser.add_argument('--model_path', type=str, 
                       default=get_model_path("best"),
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='data/sf25/aug02', help='Output directory')
    parser.add_argument('--cache_size', type=int, default=DEFAULT_CACHE_SIZE, help=f'Cache size for model inference (default: {DEFAULT_CACHE_SIZE})')
    parser.add_argument('--mcts_sims', type=int, default=DEFAULT_MCTS_SIMS, 
                       help=f'Number of MCTS simulations per move (default: {DEFAULT_MCTS_SIMS})')
    parser.add_argument('--c-puct', type=float, default=DEFAULT_C_PUCT, 
                       help=f'PUCT exploration constant for MCTS (default: {DEFAULT_C_PUCT})')
    parser.add_argument('--disable-gumbel', action='store_true',
                       help='Disable Gumbel-AlphaZero root selection for MCTS (enabled by default)')
    parser.add_argument(
        '--temperature',
        type=float,
        default=DEFAULT_TEMPERATURE_START,
        help=(
            f'Starting temperature for non-Gumbel visit-count move sampling '
            f'(default: {DEFAULT_TEMPERATURE_START})'
        ),
    )
    parser.add_argument(
        '--temperature_end',
        type=float,
        default=DEFAULT_TEMPERATURE_END,
        help=(
            f'Final temperature for non-Gumbel visit-count move sampling decay '
            f'(default: {DEFAULT_TEMPERATURE_END})'
        ),
    )
    parser.add_argument(
        '--confidence-termination-threshold',
        type=float,
        default=DEFAULT_SELFPLAY_CONFIDENCE_TERMINATION_THRESHOLD,
        help=(
            "Early-termination confidence threshold for self-play MCTS. "
            "Must be in [0, 1]. Higher values are more conservative; "
            "1.0 is the most conservative setting."
        ),
    )
    parser.add_argument('--opening_strategy', type=str, default='pie_rule', 
                       choices=['pie_rule', 'pie_rule_legacy', 'random', 'none'],
                       help='Opening strategy: pie_rule (value-balanced, default), pie_rule_legacy, random, or none')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level (0=quiet, 1=normal, 2=detailed, 3+=debug)')
    parser.add_argument('--streaming_save', action='store_true', 
                       help='Save games incrementally to avoid data loss')
    parser.add_argument(
        '--streaming-pair-integrity',
        type=str,
        choices=['check', 'repair-tail', 'off'],
        default='check',
        help=(
            "Streaming TRMPH/provenance integrity handling: "
            "'check' (fail on mismatch), "
            "'repair-tail' (repair exactly one missing terminal sidecar line), "
            "or 'off'."
        ),
    )
    parser.add_argument(
        '--write-provenance',
        dest='write_provenance',
        action='store_true',
        default=True,
        help='Write move-provenance sidecar (.provenance.jsonl) alongside TRMPH output (default: enabled)',
    )
    parser.add_argument(
        '--no-write-provenance',
        dest='write_provenance',
        action='store_false',
        help='Disable move-provenance sidecar writing',
    )
    parser.add_argument('--progress_interval', type=int, default=20, 
                       help='How often to print progress updates (must be > 0)')
    parser.add_argument(
        '--restart-every-games',
        type=int,
        default=20000,
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
                       help='Print MCTS profile once every N MCTS move selections (must be > 0).')
    parser.add_argument('--mcts-profile-max-calls', type=int, default=50,
                       help='Maximum number of MCTS move selections to profile (must be >= 0).')
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
        "board_size": args.board_size,
        "model_path": args.model_path,
        "output_dir": args.output_dir,
        "cache_size": args.cache_size,
        "mcts_sims": args.mcts_sims,
        "c_puct": args.c_puct,
        "disable_gumbel": args.disable_gumbel,
        "temperature": args.temperature,
        "temperature_end": args.temperature_end,
        "confidence_termination_threshold": args.confidence_termination_threshold,
        "opening_strategy": args.opening_strategy,
        "verbose": args.verbose,
        "streaming_save": args.streaming_save,
        "streaming_pair_integrity": args.streaming_pair_integrity,
        "write_provenance": args.write_provenance,
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
        "--board-size",
        str(args.board_size),
        "--model_path",
        args.model_path,
        "--output_dir",
        args.output_dir,
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
        "--confidence-termination-threshold",
        str(args.confidence_termination_threshold),
        "--opening_strategy",
        args.opening_strategy,
        "--verbose",
        str(args.verbose),
        "--streaming-pair-integrity",
        args.streaming_pair_integrity,
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
    if args.write_provenance:
        cmd.append("--write-provenance")
    else:
        cmd.append("--no-write-provenance")
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
            child_process = subprocess.Popen(child_cmd)
            try:
                child_returncode = child_process.wait()
            except KeyboardInterrupt:
                print("\nInterrupt received. Requesting graceful shutdown of active chunk...")
                if child_process.poll() is None:
                    child_process.send_signal(signal.SIGINT)
                    try:
                        child_process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        print("Active chunk did not exit after SIGINT; terminating child process.")
                        child_process.terminate()
                        try:
                            child_process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            print("Active chunk still running after terminate; killing child process.")
                            child_process.kill()
                            child_process.wait()
                raise
            child_result = subprocess.CompletedProcess(
                args=child_cmd,
                returncode=child_returncode,
            )

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
    def _count_trmph_game_lines(file_path: str) -> int:
        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("#13,"):
                    count += 1
        return count

    def _count_nonempty_lines(file_path: str) -> int:
        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def _get_last_trmph_game_line(file_path: str) -> Optional[str]:
        last_line: Optional[str] = None
        with open(file_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if line.startswith("#13,"):
                    last_line = line
        return last_line

    def _reconcile_streaming_pair(
        streaming_file: str,
        provenance_file: str,
        *,
        mode: str,
    ) -> None:
        if mode == "off":
            return
        if mode not in {"check", "repair-tail"}:
            raise ValueError(
                f"Unsupported streaming pair integrity mode: {mode!r}"
            )

        if not os.path.exists(streaming_file):
            raise RuntimeError(
                f"Streaming TRMPH file is missing: {streaming_file}"
            )
        if not os.path.exists(provenance_file):
            raise RuntimeError(
                f"Streaming provenance sidecar is missing: {provenance_file}"
            )

        trmph_games = _count_trmph_game_lines(streaming_file)
        provenance_records = _count_nonempty_lines(provenance_file)
        if trmph_games == provenance_records:
            return

        mismatch_message = (
            "Streaming TRMPH/provenance mismatch: "
            f"{trmph_games} game lines vs {provenance_records} provenance lines "
            f"(TRMPH: {streaming_file}, sidecar: {provenance_file})."
        )
        if mode == "check":
            raise RuntimeError(mismatch_message)

        # repair-tail mode: only permit exactly one missing terminal sidecar line.
        if trmph_games != provenance_records + 1:
            raise RuntimeError(
                mismatch_message
                + " repair-tail only supports exactly one missing terminal sidecar line."
            )

        last_game_line = _get_last_trmph_game_line(streaming_file)
        if last_game_line is None:
            raise RuntimeError(
                "Cannot repair streaming sidecar: no TRMPH game lines found."
            )
        parts = last_game_line.split()
        if len(parts) != 2:
            raise RuntimeError(
                "Cannot repair streaming sidecar: trailing TRMPH game line is malformed."
            )
        trmph_text = parts[0]
        move_count = count_trmph_moves(trmph_text)
        move_codes = MOVE_CODE_VISIT_COUNT * move_count
        recovery_record = make_move_provenance_record(
            game_index=provenance_records,
            move_codes=move_codes,
        )

        backup_dir = os.path.join(
            os.path.dirname(streaming_file),
            ".stream_integrity_backups",
        )
        os.makedirs(backup_dir, exist_ok=True)
        backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trmph_backup = os.path.join(
            backup_dir,
            f"{os.path.basename(streaming_file)}.{backup_timestamp}.bak",
        )
        provenance_backup = os.path.join(
            backup_dir,
            f"{os.path.basename(provenance_file)}.{backup_timestamp}.bak",
        )
        shutil.copy2(streaming_file, trmph_backup)
        shutil.copy2(provenance_file, provenance_backup)

        with open(provenance_file, "a", encoding="utf-8") as f:
            f.write(recovery_record.to_json_line())
            f.write("\n")

        repaired_trmph_games = _count_trmph_game_lines(streaming_file)
        repaired_provenance_records = _count_nonempty_lines(provenance_file)
        if repaired_trmph_games != repaired_provenance_records:
            raise RuntimeError(
                "repair-tail wrote fallback provenance but counts still mismatch: "
                f"{repaired_trmph_games} vs {repaired_provenance_records}."
            )

        print(
            "WARNING: repaired one missing terminal streaming provenance line "
            f"using all-trainable fallback codes. Backups: {trmph_backup}, "
            f"{provenance_backup}"
        )
    
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
    
    # Generate timestamp-based base filename (save path may append to existing files).
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Print configuration using unified logging
    script_config = ScriptConfig(
        script_type="selfplay",
        models=[args.model_path],
        strategies=[f"mcts_{args.mcts_sims}"],
        num_games=args.num_games,
        strategy_config={
            "mcts_sims": args.mcts_sims,
            "c_puct": args.c_puct,
            "board_size": args.board_size,
        },
        temperatures=args.temperature,
        pie_rule=False,  # Not applicable to selfplay
        opening_strategy=args.opening_strategy,
        cache_size=args.cache_size,
        mcts_sims=args.mcts_sims,
        c_puct=args.c_puct,
        enable_gumbel=not args.disable_gumbel,
        gumbel_sim_threshold=DEFAULT_GUMBEL_SIM_THRESHOLD,
        confidence_termination_threshold=args.confidence_termination_threshold,
        temperature_end=args.temperature_end,
        output_dir=args.output_dir
    )
    
    print_script_configuration(script_config)
    
    # Print additional selfplay specific info
    if args.opening_strategy == 'pie_rule':
        print("  Pie-rule mode: value_balanced")
        print(f"  Pie-rule weight exponent: {PIE_RULE_VALUE_BALANCED_WEIGHT_EXPONENT}")
        print(f"  Pie-rule min move probability: {PIE_RULE_VALUE_BALANCED_MIN_MOVE_PROBABILITY:.4%}")
        print(f"  Pie-rule opening cycle length: {PIE_RULE_VALUE_BALANCED_CYCLE_LENGTH}")
    elif args.opening_strategy == 'pie_rule_legacy':
        print("  Pie-rule mode: legacy")
    print(f"  Board size: {args.board_size}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Timestamp: {timestamp}")
    print()
    
    # Create opening strategy
    opening_strategy = None
    if args.opening_strategy != 'none':
        if args.opening_strategy == 'pie_rule':
            opening_strategy = create_pie_rule_strategy(
                board_size=args.board_size,
                strategy_mode="value_balanced",
            )
        elif args.opening_strategy == 'pie_rule_legacy':
            opening_strategy = create_pie_rule_strategy(
                board_size=args.board_size,
                strategy_mode="legacy",
            )
        elif args.opening_strategy == 'random':
            # Create random strategy with some common opening moves
            common_moves = [(idx, idx) for idx in range(args.board_size)]
            opening_strategy = RandomOpeningStrategy(
                common_moves,
                board_size=args.board_size,
                empty_board_prob=0.1,
            )
    
    # Initialize self-play engine
    engine = SelfPlayEngine(
        model_path=args.model_path,
        cache_size=args.cache_size,
        temperature=args.temperature,
        temperature_end=args.temperature_end,
        verbose=args.verbose,
        streaming_save=args.streaming_save,
        write_provenance=args.write_provenance,
        output_dir=args.output_dir,
        mcts_sims=args.mcts_sims,
        c_puct=args.c_puct,
        enable_gumbel=not args.disable_gumbel,
        confidence_termination_threshold=args.confidence_termination_threshold,
        command_line=command_line,
        mcts_profile=args.mcts_profile,
        mcts_profile_every=args.mcts_profile_every,
        mcts_profile_max_calls=args.mcts_profile_max_calls,
        board_size=args.board_size,
    )
    
    start_time = time.time()
    
    generation_error: Optional[Exception] = None
    integrity_error: Optional[Exception] = None
    interrupted = False
    games: Optional[List[Dict[str, Any]]] = None
    summary: Optional[SelfPlayGenerationSummary] = None
    try:
        # Generate games
        if args.streaming_save:
            summary = engine.generate_games_streaming(
                num_games=args.num_games,
                board_size=args.board_size,
                progress_interval=args.progress_interval,
                opening_strategy=opening_strategy
            )
        else:
            games, summary = engine.generate_games_with_monitoring(
                num_games=args.num_games,
                board_size=args.board_size,
                progress_interval=args.progress_interval,
                opening_strategy=opening_strategy
            )

        # Save non-streaming games and finalize summary file paths
        if games is not None:
            trmph_file = None
            provenance_file = None
            if games:
                # Save as TRMPH text file
                base_filename = f"{args.output_dir}/selfplay_{timestamp}"
                trmph_file = engine.save_games_simple(games, base_filename)
                if args.write_provenance:
                    provenance_file = str(sidecar_path_for_trmph(trmph_file))
            if summary is None:
                raise RuntimeError("Missing self-play summary for non-streaming generation.")
            summary = summary.with_files(
                trmph_file=trmph_file,
                provenance_file=provenance_file,
            )
        elif summary is None:
            raise RuntimeError("Missing self-play summary after generation.")
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Print results using unified analyzer
        output_files = {}
        if summary.trmph_file:
            output_files["trmph"] = summary.trmph_file
        if summary.provenance_file:
            output_files["provenance"] = summary.provenance_file

        performance_stats = engine.get_performance_stats()
        print_script_results(
            "selfplay",
            summary,
            script_config,
            output_files,
            total_time,
            performance_stats=performance_stats,
        )
        
    except KeyboardInterrupt:
        interrupted = True
        print("\n\nGeneration interrupted by user.")
        if args.streaming_save:
            print("Games saved incrementally - no data loss.")
    except Exception as e:
        print(f"\nError during generation: {e}")
        generation_error = e
    finally:
        if (
            args.streaming_save
            and args.write_provenance
            and args.streaming_pair_integrity != "off"
        ):
            streaming_file = engine.streaming_file
            provenance_file = engine.streaming_provenance_file
            if not streaming_file or not provenance_file:
                integrity_error = RuntimeError(
                    "Streaming integrity check requested but streaming file pair "
                    "is not fully initialized."
                )
                print(f"\nERROR: {integrity_error}")
            else:
                try:
                    _reconcile_streaming_pair(
                        streaming_file,
                        provenance_file,
                        mode=args.streaming_pair_integrity,
                    )
                except Exception as e:
                    integrity_error = e
                    print(f"\nERROR: {integrity_error}")

        # Clean shutdown
        engine.shutdown()

    if generation_error is not None:
        raise generation_error
    if integrity_error is not None:
        raise integrity_error
    # In chunked mode, return non-success so the parent does not count this chunk as complete.
    if interrupted and args.internal_chunk_run:
        raise SystemExit(130)


def main():
    args = parse_args()

    if args.num_games <= 0:
        raise ValueError("--num_games must be > 0")
    if args.board_size <= 0:
        raise ValueError("--board-size must be > 0")
    if args.board_size != BOARD_SIZE:
        raise ValueError(
            f"--board-size currently supports only {BOARD_SIZE}; got {args.board_size}"
        )
    if args.progress_interval <= 0:
        raise ValueError("--progress_interval must be > 0")
    if args.mcts_profile_every <= 0:
        raise ValueError("--mcts-profile-every must be > 0")
    if args.mcts_profile_max_calls < 0:
        raise ValueError("--mcts-profile-max-calls must be >= 0")
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
