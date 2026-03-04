#!/usr/bin/env python3
"""
Run a deterministic strategy tournament using pre-generated opening positions.

This script compares different strategies using the same model and the same
set of opening positions, eliminating randomness as a confounding factor.

The key insight is that by starting from the same opening positions,
we can directly compare how different strategies perform from identical
starting points, making the comparison much more robust.

Different runs can use different opening sets by changing the --seed parameter,
allowing you to gather more data across multiple tournament runs while
maintaining deterministic gameplay within each run.

Examples:

1. Compare strategies using 100 diverse openings:
   python scripts/run_tournament.py \
     --model=best \
     --strategies=policy,mcts_122,fixed_tree_13_8 \
     --num-openings=100

2. Use specific opening file:
   python scripts/run_tournament.py \
     --model=best \
     --strategies=mcts_100,mcts_200 \
     --opening-file=data/deterministic_openings.txt

3. Use custom temperature:
   python scripts/run_tournament.py \
     --model=best \
     --strategies=policy,mcts_122 \
     --num-openings=150 \
     --temperature=0.1

4. Get different opening sets for multiple runs:
   # Each run automatically gets a different seed (from time)
   python scripts/run_tournament.py \
     --model=best \
     --strategies=policy,mcts_122 \
     --num-openings=100
   
   # Or manually specify seeds for reproducible results
   python scripts/run_tournament.py \
     --model=best \
     --strategies=policy,mcts_122 \
     --num-openings=100 \
     --seed=123
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from hex_ai.memory_profiler import start_profiling, stop_profiling
from hex_ai.config import (
    DEFAULT_BATCH_CAP,
    DEFAULT_C_PUCT,
    DEFAULT_MCTS_SIMS,
    DEFAULT_GUMBEL_SIM_THRESHOLD,
    DEFAULT_GUMBEL_C_SCALE,
    DEFAULT_GUMBEL_CANDIDATE_POWER_SCALE,
    DEFAULT_GUMBEL_CANDIDATE_POWER_RATE,
    DEFAULT_GUMBEL_CANDIDATE_POWER_OFFSET,
    DEFAULT_TOURNAMENT_BASE_FRACTION_MCTS_MOVES,
    TOURNAMENT_CONFIDENCE_TERMINATION_THRESHOLD,
)
from hex_ai.inference.model_config import (
    get_all_model_participants_from_generations,
    get_primary_model_paths_from_recent_generations,
)
from hex_ai.utils.gumbel_validation import check_gumbel_configurations
from hex_ai.inference.strategy_config import StrategyConfig
from hex_ai.utils.tournament_logging import get_command_line
from hex_ai.utils.tournament_utils import (
    parse_tournament_parameters,
    parse_model_specifications,
    create_strategy_configs_for_tournament,
    format_strategy_configuration_details,
)
from hex_ai.utils.random_utils import set_deterministic_seeds
from hex_ai.utils.script_logging import ScriptConfig, print_script_configuration, print_script_results
from hex_ai.inference.game_execution import (
    find_trmph_files,
    generate_diverse_openings,
    load_openings_from_file,
    select_random_openings,
)
from hex_ai.inference.checkpoint_discovery import CheckpointDiscovery, CheckpointInfo
from hex_ai.inference.two_stage_tournament import TwoStageTournament
from hex_ai.inference.knockout_tournament import TournamentParticipant

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/tournament.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure logging verbosity based on command line argument
def configure_logging_verbosity(verbose_level: int):
    """Configure logging verbosity based on the --verbose argument."""
    if verbose_level <= 0:
        # Very quiet - only show essential tournament progress
        logging.getLogger('hex_ai.inference.model_wrapper').setLevel(logging.ERROR)
        logging.getLogger('hex_ai.inference.checkpoint_discovery').setLevel(logging.ERROR)
        logging.getLogger('hex_ai.inference.game_execution').setLevel(logging.ERROR)
        logging.getLogger('hex_ai.inference.knockout_tournament').setLevel(logging.ERROR)
    elif verbose_level == 1:
        # Default - show tournament progress but reduce repetitive logs
        logging.getLogger('hex_ai.inference.model_wrapper').setLevel(logging.WARNING)
        logging.getLogger('hex_ai.inference.checkpoint_discovery').setLevel(logging.WARNING)
        logging.getLogger('hex_ai.inference.game_execution').setLevel(logging.WARNING)
        logging.getLogger('hex_ai.inference.knockout_tournament').setLevel(logging.WARNING)
    elif verbose_level >= 2:
        # Verbose - show all logs
        # Keep default INFO level for all loggers
        pass

# Constants
DEFAULT_OPENING_LENGTH = 5
DEFAULT_NUM_OPENINGS = 100
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = None  # Will be set to int(time.time()) if None
DEFAULT_VERBOSE = 1
TRMPH_SOURCE_DIR = "data/sf25/sep28"
DEFAULT_MOST_RECENT_ROOT = "checkpoints/hyperparameter_tuning"
CHECKPOINT_FILE_REGEX = re.compile(r"epoch\d+_mini\d+\.pt\.gz$")
DEFAULT_ROUND_ROBIN_SKIP_RECENT_GENERATIONS = 1

# TODO: Consider adding configuration for:
# Low priority: Timeout handling for long-running strategies
# Low priority: Progress saving/resume functionality for interrupted tournaments


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a deterministic strategy tournament using pre-generated opening positions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare strategies using model registry names
  %(prog)s --models=best,model2 --strategies=policy,mcts,mcts --mcts-sims=100,200 --num-openings=100
  
  # Compare same strategy with different models from registry
  %(prog)s --models=best,previous_best --strategies=mcts,mcts --mcts-sims=100,100 --num-openings=50
  
  # Compare strategies using direct model file specification
  %(prog)s --model-files=epoch13_mini31.pt.gz,epoch13_mini27.pt.gz --model-dirs=checkpoints/dir1,checkpoints/dir2 --strategies=mcts,mcts --mcts-sims=30,30 --num-openings=50
  
  # Use specific opening file with different models
  %(prog)s --models=best --strategies=mcts,mcts --mcts-sims=100,200 --opening-file=data/deterministic_openings.txt
  
  # Compare with custom opening length and temperature
  %(prog)s --models=best --strategies=policy,mcts --mcts-sims=122 --num-openings=200 --opening-length=5 --temperature=0.1
  
  # Compare same strategy with different temperatures
  %(prog)s --models=best,best --strategies=policy,policy --temperatures=0.1,1.0 --num-openings=100
  
  # Use single model for multiple strategies (convenience feature)
  %(prog)s --models=best --strategies=policy,mcts,mcts --mcts-sims=100,200 --num-openings=100
        """
    )
    
    parser.add_argument('--models', type=str,
                       help='Comma-separated list of model registry names (e.g., "best,model2"). If only one model is provided, it will be used for all strategies.')
    parser.add_argument('--model-files', type=str,
                       help='Comma-separated list of model file names (e.g., "epoch13_mini31.pt.gz,epoch13_mini27.pt.gz"). Optional for 2-stage runs: when omitted with knockout + --strategies, defaults to previous generation primaries from MODEL_GENERATIONS.')
    parser.add_argument('--model-dirs', type=str,
                       help='Comma-separated list of model directories (used with --model-files)')
    parser.add_argument('--strategies', type=str,
                       help='Comma-separated list of strategies to compare (e.g., "mcts,policy"). Required for traditional tournaments, optional for knockout-only tournaments.')
    parser.add_argument('--num-openings', type=int, default=DEFAULT_NUM_OPENINGS,
                       help=f'Deprecated alias for --round-robin-games (default: {DEFAULT_NUM_OPENINGS})')
    parser.add_argument('--opening-length', type=int, default=DEFAULT_OPENING_LENGTH,
                       help=f'Number of moves per opening (default: {DEFAULT_OPENING_LENGTH})')
    parser.add_argument('--opening-file', type=str,
                       help='File containing pre-generated openings (overrides num-openings)')
    parser.add_argument('--cache-file', type=str,
                       help='File to cache generated openings for faster subsequent runs')
    parser.add_argument('--trmph-source', type=str, default=TRMPH_SOURCE_DIR,
                       help=f'Directory containing TRMPH files for opening generation (default: {TRMPH_SOURCE_DIR})')
    parser.add_argument('--mcts-sims', type=str,
                       help=f'Comma-separated MCTS simulation counts (default: {DEFAULT_MCTS_SIMS})')
    parser.add_argument(
        '--base-fraction-mcts-moves',
        type=str,
        help=(
            "Comma-separated fraction of moves that run full MCTS for MCTS strategies "
            f"(default: {DEFAULT_TOURNAMENT_BASE_FRACTION_MCTS_MOVES})"
        ),
    )
    parser.add_argument('--batch-sizes', type=str,
                       help=f'Comma-separated batch sizes for MCTS strategies (e.g., "64,128,256", default: {DEFAULT_BATCH_CAP})')
    parser.add_argument('--c-puct', type=str,
                       help=f'Comma-separated PUCT exploration constants for MCTS strategies (e.g., "2.4,2.8,3.6", default: {DEFAULT_C_PUCT})')
    parser.add_argument('--enable-gumbel', type=str,
                       help='Comma-separated boolean values to enable Gumbel AlphaZero root selection for MCTS strategies (e.g., "true,false,true", default: true)')
    parser.add_argument('--gumbel-sim-threshold', type=str,
                       help=f'Comma-separated simulation thresholds for Gumbel AlphaZero root selection (e.g., "200,500,1000", default: {DEFAULT_GUMBEL_SIM_THRESHOLD})')
    parser.add_argument('--gumbel-candidate-power-scale', type=str,
                       help=f'Comma-separated power scales for Gumbel candidate scaling (e.g., "60.0,80.0,100.0", default: {DEFAULT_GUMBEL_CANDIDATE_POWER_SCALE})')
    parser.add_argument('--gumbel-candidate-power-rate', type=str,
                       help=f'Comma-separated power rates for Gumbel candidate scaling (e.g., "0.39,0.45,0.50", default: {DEFAULT_GUMBEL_CANDIDATE_POWER_RATE})')
    parser.add_argument('--gumbel-candidate-power-offset', type=str,
                       help=f'Comma-separated power offsets for Gumbel candidate scaling (e.g., "-4.0,-3.0,-2.0", default: {DEFAULT_GUMBEL_CANDIDATE_POWER_OFFSET})')
    parser.add_argument('--gumbel-c-scale', type=str,
                       help=f'Comma-separated c_scale parameters for Gumbel AlphaZero root selection (e.g., "1000,5000,10000", default: {DEFAULT_GUMBEL_C_SCALE})')
    parser.add_argument('--enable-dead-cell-pruning', type=str,
                       help='Comma-separated boolean values to enable dead-cell hard masking in MCTS strategies (e.g., "true,false,true", default: false)')
    parser.add_argument('--dead-cell-enable-four-run', type=str,
                       help='Comma-separated boolean values to enable dead-cell D1 (4-run) motif (default: true)')
    parser.add_argument('--dead-cell-enable-two-two-split', type=str,
                       help='Comma-separated boolean values to enable dead-cell D2 (2+2 split) motif (default: true)')
    parser.add_argument('--dead-cell-enable-three-plus-one', type=str,
                       help='Comma-separated boolean values to enable dead-cell D3 motif (default: true)')
    parser.add_argument('--dead-cell-enable-a1b2a3-discouraged', type=str,
                       help='Comma-separated boolean values to enable A1B2A3 discouraged motif (default: true)')
    parser.add_argument('--dead-cell-enable-double-dead-pairs', type=str,
                       help='Comma-separated boolean values to enable two-cell dead-pair motif (default: false)')
    parser.add_argument(
        '--dead-cell-debug-log-path',
        type=str,
        help=(
            "Optional JSONL path for root dead-cell pruning debug records. "
            "When set, masked MCTS strategies append state snapshots and triggering rules."
        ),
    )
    parser.add_argument(
        '--dead-cell-debug-max-records-per-move',
        type=int,
        default=200,
        help=(
            "Maximum dead-cell debug records written per root move "
            "(default: 200, <=0 means no cap)."
        ),
    )
    parser.add_argument(
        '--dead-cell-counterfactual-debug-log-path',
        type=str,
        help=(
            "Optional JSONL path for root counterfactual debug records. "
            "When set for masked MCTS, each move also runs an unmasked root probe "
            "and logs cases where unmasked MCTS would choose a masked move."
        ),
    )
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                       help=f'Global temperature for move selection (0.0 = deterministic, default: {DEFAULT_TEMPERATURE})')
    parser.add_argument('--temperatures', type=str,
                       help='Comma-separated temperatures for each strategy (e.g., "0.1,1.0,0.5"). Overrides --temperature.')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                       help=f'Random seed for opening selection (different seeds produce different opening sets) (default: auto-generated from time)')
    parser.add_argument('--verbose', type=int, default=DEFAULT_VERBOSE,
                       help=f'Verbosity level (default: {DEFAULT_VERBOSE})')
    
    # 2-stage tournament arguments
    parser.add_argument('--knockout-dir', type=str,
                       help='Directory containing checkpoints for knockout stage (mutually exclusive with --knockout-from-generations)')
    parser.add_argument('--knockout-from-generations', action='store_true',
                       help='Use all models from MODEL_GENERATIONS for knockout stage (mutually exclusive with --knockout-dir)')
    parser.add_argument('--knockout-config', type=str,
                       help='JSON configuration for knockout stage MCTS strategy (e.g., \'{"mcts_sims": 100, "enable_gumbel_root_selection": true}\')')
    parser.add_argument('--epoch-range', type=str,
                       help='Epoch range for knockout stage. Format: "N" for single epoch, "N,M" for range N to M-1 (e.g., "16" for epoch 16 only, "16,19" for epochs 16,17,18)')
    parser.add_argument('--mini-epoch-range', type=str,
                       help='Mini epoch range for knockout stage. Format: "N" for single mini epoch, "N,M" for range N to M-1 (e.g., "14" for mini epoch 14 only, "14,20" for mini epochs 14-19)')
    parser.add_argument('--games-per-match', type=int, default=50,
                       help='Number of games per knockout match (default: 50)')
    parser.add_argument('--top-k', type=int, default=2,
                       help='Number of winners from knockout stage to advance (default: 2)')
    parser.add_argument('--most-recent', type=int,
                       help='Use the N most recent checkpoints from the latest run under --most-recent-root for knockout stage.')
    parser.add_argument('--most-recent-biased', type=int,
                       help='Use a recency-biased sample of N checkpoints from the latest run under --most-recent-root for knockout stage.')
    parser.add_argument('--most-recent-root', type=str, default=DEFAULT_MOST_RECENT_ROOT,
                       help=f'Root directory for --most-recent/--most-recent-biased (default: {DEFAULT_MOST_RECENT_ROOT})')
    parser.add_argument('--round-robin-games', type=int, default=DEFAULT_NUM_OPENINGS,
                       help='Number of openings per round-robin pair (actual games are doubled via color swap, default: 100)')
    parser.add_argument('--run-desc', type=str,
                       help='Description of this tournament run (e.g., "Testing c_scale = 1.5") - will be included in output headers')

    # Memory profiling / leak triage
    parser.add_argument('--memory-profile', action='store_true',
                       help='Enable RSS/heap memory profiling (writes to temp/memoryProfile/).')
    parser.add_argument('--memory-profile-interval', type=int, default=60,
                       help='Seconds between automatic memory timeline samples (default: 60).')
    parser.add_argument('--memory-profile-dir', type=str, default="temp/memoryProfile",
                       help='Output directory for memory profiling files (default: temp/memoryProfile).')
    parser.add_argument('--mps-empty-cache-per-pair', action='store_true',
                       help='If running on MPS, call torch.mps.empty_cache() after each match/pair (diagnostic only).')

    # Lightweight MCTS timing profiler (GPU vs CPU breakdown)
    parser.add_argument('--mcts-profile', action='store_true',
                       help='Print lightweight MCTS timing breakdown every N calls (GPU vs CPU time).')
    parser.add_argument('--mcts-profile-every', type=int, default=10,
                       help='Print MCTS profile once every N MCTS move selections (default: 10).')
    parser.add_argument('--mcts-profile-max-calls', type=int, default=50,
                       help='Maximum number of MCTS move selections to profile (default: 50).')
    
    return parser.parse_args()


def parse_epoch_range(epoch_range_str: str) -> Tuple[int, int]:
    """
    Parse epoch range string into start and end epoch numbers.
    
    Args:
        epoch_range_str: String like "16" for single epoch or "16,19" for range 16-19 (inclusive)
        
    Returns:
        Tuple of (start_epoch, end_epoch) where end_epoch is exclusive
        Examples:
            "16" -> (16, 17)  # Just epoch 16
            "16,19" -> (16, 20)  # Epochs 16, 17, 18, 19
        
    Raises:
        ValueError: If format is invalid
    """
    if not epoch_range_str:
        raise ValueError("Epoch range string cannot be empty")
    
    parts = epoch_range_str.split(',')
    if len(parts) == 1:
        # Single epoch: "16" -> start=16, end=17
        start_epoch = int(parts[0].strip())
        end_epoch = start_epoch + 1
    elif len(parts) == 2:
        # Range: "16,19" -> start=16, end=20 (includes 16,17,18,19)
        start_epoch = int(parts[0].strip())
        end_epoch = int(parts[1].strip()) + 1
    else:
        raise ValueError(f"Invalid epoch range format: '{epoch_range_str}'. Expected format: 'N' for single epoch (e.g., '16') or 'N,M' for range N to M inclusive (e.g., '16,19' for epochs 16,17,18,19)")
    
    if start_epoch < 1:
        raise ValueError(f"Start epoch must be >= 1, got {start_epoch}")
    if end_epoch <= start_epoch:
        raise ValueError(f"End epoch must be > start epoch, got start={start_epoch}, end={end_epoch}")
    
    return start_epoch, end_epoch


def parse_mini_epoch_range(mini_epoch_range_str: str) -> Tuple[int, int]:
    """
    Parse mini epoch range string into start and end mini epoch numbers.
    
    Args:
        mini_epoch_range_str: String like "14" for single mini epoch or "14,20" for range 14-20 (inclusive)
        
    Returns:
        Tuple of (start_mini_epoch, end_mini_epoch) where end_mini_epoch is exclusive
        Examples:
            "14" -> (14, 15)  # Just mini epoch 14
            "14,20" -> (14, 21)  # Mini epochs 14, 15, 16, 17, 18, 19, 20
        
    Raises:
        ValueError: If format is invalid
    """
    if not mini_epoch_range_str:
        raise ValueError("Mini epoch range string cannot be empty")
    
    parts = mini_epoch_range_str.split(',')
    if len(parts) == 1:
        # Single mini epoch: "14" -> start=14, end=15
        start_mini_epoch = int(parts[0].strip())
        end_mini_epoch = start_mini_epoch + 1
    elif len(parts) == 2:
        # Range: "14,20" -> start=14, end=21 (includes 14,15,16,17,18,19,20)
        start_mini_epoch = int(parts[0].strip())
        end_mini_epoch = int(parts[1].strip()) + 1
    else:
        raise ValueError(f"Invalid mini epoch range format: '{mini_epoch_range_str}'. Expected format: 'N' for single mini epoch (e.g., '14') or 'N,M' for range N to M inclusive (e.g., '14,20' for mini epochs 14-20)")
    
    if start_mini_epoch < 1:
        raise ValueError(f"Start mini epoch must be >= 1, got {start_mini_epoch}")
    if end_mini_epoch <= start_mini_epoch:
        raise ValueError(f"End mini epoch must be > start mini epoch, got start={start_mini_epoch}, end={end_mini_epoch}")
    
    return start_mini_epoch, end_mini_epoch


def _build_recent_biased_checkpoint_offsets_unbounded(count: int) -> List[int]:
    """
    Build recency-biased offsets from the latest checkpoint without history bounds.

    Offset semantics:
      0 -> newest checkpoint
      -1 -> second newest
      -2 -> third newest
      ...

    Examples:
      count=4  -> [0, -1, -2, -4]
      count=8  -> [0, -1, -2, -3, -5, -7, -10, -13]
      count=16 -> [0, -1, -2, -3, -4, -6, -8, -10, -12, -15, -18, -21, -25, -29, -33, -38]
    """
    head_len = min(count, int(round(math.sqrt(count))) + 1)
    distances = list(range(head_len))

    remaining = count - head_len
    step = 2
    while remaining > 0:
        repeats = max(1, int(round(count / (step + 2))))
        take = min(repeats, remaining)
        for _ in range(take):
            distances.append(distances[-1] + step)
        remaining -= take
        step += 1

    return [-distance for distance in distances]


def build_recent_biased_checkpoint_offsets(
    count: int,
    available_count: Optional[int] = None,
) -> List[int]:
    """
    Build recency-biased offsets from the latest checkpoint.

    When available_count is provided, the returned offsets are guaranteed to fit
    the discovered history and will be compressed toward recent checkpoints if
    the default spacing would otherwise spread too far back.
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")

    if available_count is None:
        return _build_recent_biased_checkpoint_offsets_unbounded(count)

    if available_count <= 0:
        raise ValueError(
            f"available_count must be positive when provided, got {available_count}"
        )

    effective_count = min(count, available_count)
    if effective_count <= 0:
        raise ValueError(
            f"effective checkpoint count must be positive, got {effective_count}"
        )

    unbounded_offsets = _build_recent_biased_checkpoint_offsets_unbounded(effective_count)
    max_available_distance = available_count - 1
    max_unbounded_distance = -unbounded_offsets[-1]

    if max_unbounded_distance <= max_available_distance:
        return unbounded_offsets

    # Overflow indicates the default spacing reaches too far into history.
    # Compress the spread toward recent checkpoints while preserving uniqueness.
    compression_ratio = max_unbounded_distance / max_available_distance
    target_span = max(
        effective_count - 1,
        int(round(max_available_distance / compression_ratio)),
    )
    target_span = min(target_span, max_available_distance)
    power = min(4.0, 2.0 + (compression_ratio - 1.0) * 1.5)

    distances = [0]
    for idx in range(1, effective_count):
        t = idx / (effective_count - 1)
        projected = int(round((t ** power) * target_span))
        min_allowed = distances[-1] + 1
        max_allowed = target_span - (effective_count - 1 - idx)
        distance = min(max(projected, min_allowed), max_allowed)
        distances.append(distance)

    return [-distance for distance in distances]


def _discover_latest_checkpoint_directory(root_dir: str) -> Tuple[Path, Path]:
    """Find the latest run dir and its most recent checkpoint-containing directory."""
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(
            f"Most-recent root directory does not exist: {root_dir}. "
            "Please provide a valid directory with hyperparameter runs."
        )
    if not root_path.is_dir():
        raise ValueError(
            f"Most-recent root path is not a directory: {root_dir}. "
            "Please provide a directory path."
        )

    run_dirs = [path for path in root_path.iterdir() if path.is_dir()]
    if not run_dirs:
        raise ValueError(
            f"No run directories found in {root_dir}. "
            "Please ensure the directory contains hyperparameter run subdirectories."
        )

    latest_run_dir: Optional[Path] = None
    latest_checkpoint_dir: Optional[Path] = None
    latest_mtime: Optional[float] = None

    for run_dir in run_dirs:
        checkpoint_files: List[Path] = []
        for file_path in run_dir.rglob("epoch*_mini*.pt.gz"):
            if file_path.is_file() and CHECKPOINT_FILE_REGEX.match(file_path.name):
                checkpoint_files.append(file_path)

        if not checkpoint_files:
            continue

        newest_checkpoint = max(checkpoint_files, key=lambda path: path.stat().st_mtime)
        newest_checkpoint_mtime = newest_checkpoint.stat().st_mtime
        if latest_mtime is None or newest_checkpoint_mtime > latest_mtime:
            latest_mtime = newest_checkpoint_mtime
            latest_run_dir = run_dir
            latest_checkpoint_dir = newest_checkpoint.parent

    if latest_run_dir is None or latest_checkpoint_dir is None:
        raise ValueError(
            f"No checkpoint files matching 'epochN_miniJ.pt.gz' were found under {root_dir}. "
            "Please check that training has produced checkpoints in this tree."
        )

    return latest_run_dir, latest_checkpoint_dir


def _select_checkpoints_from_offsets(
    checkpoints: List[CheckpointInfo],
    offsets: List[int],
) -> List[CheckpointInfo]:
    """Select checkpoints using offsets relative to latest checkpoint (offset 0)."""
    if not offsets:
        raise ValueError("Offset list cannot be empty")

    selected_indices = []
    latest_index = len(checkpoints) - 1

    for offset in offsets:
        if offset > 0:
            raise ValueError(
                f"Offset must be <= 0, got {offset}. "
                "Offsets are relative to the latest checkpoint (0, -1, -2, ...)."
            )

        index = latest_index + offset
        if index < 0 or index >= len(checkpoints):
            raise ValueError(
                f"Offset {offset} is out of range for {len(checkpoints)} available checkpoints. "
                "Use fewer checkpoints or a less aggressive spacing pattern."
            )
        selected_indices.append(index)

    if len(set(selected_indices)) != len(selected_indices):
        raise ValueError(
            f"Offset schedule produced duplicate checkpoint selections: {offsets}. "
            "Please use a schedule with unique offsets."
        )

    selected = [checkpoints[index] for index in selected_indices]
    selected.sort(key=lambda checkpoint: checkpoint.creation_time)
    return selected


def _build_checkpoint_participant(
    checkpoint: CheckpointInfo,
    knockout_config: Dict[str, Any],
) -> TournamentParticipant:
    """Build a knockout participant for one discovered checkpoint."""
    participant_config = knockout_config.copy()
    participant_config["strategy"] = "mcts"
    participant_config["model_path"] = str(checkpoint.file_path)
    if "temperature" not in participant_config:
        participant_config["temperature"] = 1.0

    return TournamentParticipant(
        name=checkpoint.name,
        strategy_config=participant_config,
        metadata={
            "checkpoint_file": str(checkpoint.file_path),
            "epoch": checkpoint.epoch,
            "mini_epoch": checkpoint.mini,
            "checkpoint_number": checkpoint.checkpoint_number,
        },
    )


def _build_most_recent_knockout_participants(
    count: int,
    biased: bool,
    root_dir: str,
    knockout_config: Dict[str, Any],
) -> Tuple[List[TournamentParticipant], Dict[str, Any]]:
    """Build knockout participants from the latest run's checkpoints."""
    if count < 2:
        raise ValueError(
            f"Most-recent checkpoint mode requires at least 2 checkpoints, got {count}."
        )

    latest_run_dir, checkpoint_dir = _discover_latest_checkpoint_directory(root_dir)
    discovery = CheckpointDiscovery(str(checkpoint_dir))
    checkpoints = discovery.discover_checkpoints()
    available_count = len(checkpoints)

    if available_count < 2:
        raise ValueError(
            f"Latest checkpoint directory has only {available_count} checkpoint(s): {checkpoint_dir}. "
            "At least 2 checkpoints are required for knockout."
        )

    selected_count = min(count, available_count)
    offsets = (
        build_recent_biased_checkpoint_offsets(
            selected_count,
            available_count=available_count,
        )
        if biased
        else [-index for index in range(selected_count)]
    )
    selected_checkpoints = _select_checkpoints_from_offsets(checkpoints, offsets)
    participants = [
        _build_checkpoint_participant(checkpoint, knockout_config)
        for checkpoint in selected_checkpoints
    ]

    return participants, {
        "latest_run_dir": str(latest_run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "requested_count": count,
        "available_count": available_count,
        "selected_count": len(selected_checkpoints),
        "offsets": offsets,
        "selected_checkpoints": [checkpoint.name for checkpoint in selected_checkpoints],
        "selection_mode": "biased" if biased else "most_recent",
    }


def _has_knockout_source(args) -> bool:
    """Return True when any knockout source is configured."""
    return bool(
        getattr(args, "knockout_dir", None)
        or getattr(args, "knockout_from_generations", False)
        or getattr(args, "most_recent", None) is not None
        or getattr(args, "most_recent_biased", None) is not None
    )


def _should_auto_default_round_robin_models(args, strategy_names: List[str]) -> bool:
    """
    Decide whether to auto-populate round-robin models from recent generations.

    We only do this for 2-stage runs (knockout source present) when:
      - user provided round-robin strategies
      - user did not explicitly provide any round-robin model source
    """
    if not _has_knockout_source(args):
        return False
    if not strategy_names:
        return False
    if getattr(args, "models", None):
        return False
    if getattr(args, "model_files", None) or getattr(args, "model_dirs", None):
        return False
    return True


def is_knockout_only_tournament(args) -> bool:
    """Check if this is a knockout-only tournament (no round-robin participants)."""
    has_knockout = _has_knockout_source(args)
    has_round_robin_strategy_intent = bool(getattr(args, "strategies", None))
    has_round_robin_model_intent = bool(
        getattr(args, "models", None)
        or getattr(args, "model_files", None)
        or getattr(args, "model_dirs", None)
    )
    return has_knockout and not has_round_robin_strategy_intent and not has_round_robin_model_intent


def print_round_robin_strategy_summary(strategy_configs: List[StrategyConfig]) -> None:
    """Print key per-strategy settings for round-robin tournament participants."""
    if not strategy_configs:
        return

    print("  Round-robin strategy configurations:")
    for i, strategy in enumerate(strategy_configs, start=1):
        details = format_strategy_configuration_details(strategy)
        print(f"    {i}. {strategy.name}: {details}")


def _dedupe_preserve_order(values: List[Any]) -> List[Any]:
    """Return unique values while preserving first-seen order."""
    deduped: List[Any] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _collapse_single_or_list(values: List[Any]) -> Optional[Any]:
    """Return a scalar when values are uniform, otherwise return unique list."""
    if not values:
        return None
    unique_values = _dedupe_preserve_order(values)
    if len(unique_values) == 1:
        return unique_values[0]
    return unique_values


def _single_value_or_none(values: List[Any]) -> Optional[Any]:
    """Return the single value when uniform, otherwise None."""
    if not values:
        return None
    first = values[0]
    if all(value == first for value in values):
        return first
    return None


def _collect_resolved_logging_values(strategy_configs: List[StrategyConfig]) -> Dict[str, Any]:
    """Collect resolved per-strategy values for transparent script logging."""
    if not strategy_configs:
        return {
            "temperatures": None,
            "batch_sizes": None,
            "c_puct": None,
            "mcts_sims": None,
            "base_fraction_mcts_moves": None,
            "enable_gumbel": False,
            "gumbel_sim_threshold": None,
            "gumbel_c_visit": None,
            "gumbel_c_scale": None,
            "gumbel_candidate_power_scale": None,
            "gumbel_candidate_power_rate": None,
            "gumbel_candidate_power_offset": None,
            "gumbel_m_candidates": None,
        }

    temperatures = _collapse_single_or_list([config.temperature for config in strategy_configs])
    mcts_configs = [config.config for config in strategy_configs if config.strategy_type == "mcts"]
    if not mcts_configs:
        return {
            "temperatures": temperatures,
            "batch_sizes": None,
            "c_puct": None,
            "mcts_sims": None,
            "base_fraction_mcts_moves": None,
            "enable_gumbel": False,
            "gumbel_sim_threshold": None,
            "gumbel_c_visit": None,
            "gumbel_c_scale": None,
            "gumbel_candidate_power_scale": None,
            "gumbel_candidate_power_rate": None,
            "gumbel_candidate_power_offset": None,
            "gumbel_m_candidates": None,
        }

    batch_sizes = _dedupe_preserve_order([cfg["batch_size"] for cfg in mcts_configs])
    c_puct = _collapse_single_or_list([cfg["mcts_c_puct"] for cfg in mcts_configs])
    mcts_sims = _single_value_or_none([cfg["mcts_sims"] for cfg in mcts_configs])
    base_fraction_mcts_moves = _single_value_or_none(
        [cfg.get("base_fraction_mcts_moves") for cfg in mcts_configs]
    )
    enable_gumbel = any(bool(cfg["enable_gumbel_root_selection"]) for cfg in mcts_configs)
    gumbel_enabled_configs = [
        cfg for cfg in mcts_configs if bool(cfg["enable_gumbel_root_selection"])
    ]

    return {
        "temperatures": temperatures,
        "batch_sizes": batch_sizes,
        "c_puct": c_puct,
        "mcts_sims": mcts_sims,
        "base_fraction_mcts_moves": base_fraction_mcts_moves,
        "enable_gumbel": enable_gumbel,
        "gumbel_sim_threshold": _single_value_or_none(
            [cfg["gumbel_sim_threshold"] for cfg in gumbel_enabled_configs]
        ),
        "gumbel_c_visit": _single_value_or_none(
            [cfg["gumbel_c_visit"] for cfg in gumbel_enabled_configs]
        ),
        "gumbel_c_scale": _single_value_or_none(
            [cfg["gumbel_c_scale"] for cfg in gumbel_enabled_configs]
        ),
        "gumbel_candidate_power_scale": _single_value_or_none(
            [cfg["gumbel_candidate_power_scale"] for cfg in gumbel_enabled_configs]
        ),
        "gumbel_candidate_power_rate": _single_value_or_none(
            [cfg["gumbel_candidate_power_rate"] for cfg in gumbel_enabled_configs]
        ),
        "gumbel_candidate_power_offset": _single_value_or_none(
            [cfg["gumbel_candidate_power_offset"] for cfg in gumbel_enabled_configs]
        ),
        "gumbel_m_candidates": _single_value_or_none(
            [cfg.get("gumbel_m_candidates") for cfg in gumbel_enabled_configs]
        ),
    }




def run_two_stage_tournament(args, strategy_configs, model_paths, openings, command_line):
    """
    Run a 2-stage tournament: knockout elimination followed by round-robin.
    
    Args:
        args: Parsed command line arguments
        strategy_configs: Strategy configurations for round-robin stage
        model_paths: Model paths for round-robin stage
        openings: Opening positions for the tournament
        command_line: Command line that was used to run the tournament
        
    Returns:
        Tournament result object
    """
    
    # Parse knockout configuration
    # Start with JSON config if provided
    knockout_config = {}
    if args.knockout_config:
        try:
            knockout_config = json.loads(args.knockout_config)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in --knockout-config: {e}")
            sys.exit(1)
    
    # Parse command-line MCTS parameters and merge into knockout_config
    # This allows users to specify --mcts-sims, --enable-gumbel, etc. directly
    parsed_params = parse_tournament_parameters(args, include_defaults=False)
    
    # For knockout-only tournaments, validate that only single values are provided
    # (Multiple values are only meaningful for round-robin stage where different configs are compared)
    is_knockout_only = is_knockout_only_tournament(args)
    if is_knockout_only:
        params_with_multiple_values = []
        if parsed_params.get('mcts_sims') and len(parsed_params['mcts_sims']) > 1:
            params_with_multiple_values.append(f"mcts_sims (got {len(parsed_params['mcts_sims'])} values: {parsed_params['mcts_sims']})")
        if (
            parsed_params.get('base_fraction_mcts_moves')
            and len(parsed_params['base_fraction_mcts_moves']) > 1
        ):
            params_with_multiple_values.append(
                "base_fraction_mcts_moves "
                f"(got {len(parsed_params['base_fraction_mcts_moves'])} values: {parsed_params['base_fraction_mcts_moves']})"
            )
        if parsed_params.get('enable_gumbel') and len(parsed_params['enable_gumbel']) > 1:
            params_with_multiple_values.append(f"enable_gumbel (got {len(parsed_params['enable_gumbel'])} values: {parsed_params['enable_gumbel']})")
        if parsed_params.get('temperatures') and isinstance(parsed_params['temperatures'], list) and len(parsed_params['temperatures']) > 1:
            params_with_multiple_values.append(f"temperatures (got {len(parsed_params['temperatures'])} values: {parsed_params['temperatures']})")
        if parsed_params.get('c_pucts') and len(parsed_params['c_pucts']) > 1:
            params_with_multiple_values.append(f"c_puct (got {len(parsed_params['c_pucts'])} values: {parsed_params['c_pucts']})")
        if parsed_params.get('batch_sizes') and len(parsed_params['batch_sizes']) > 1:
            params_with_multiple_values.append(f"batch_sizes (got {len(parsed_params['batch_sizes'])} values: {parsed_params['batch_sizes']})")
        if parsed_params.get('gumbel_sim_thresholds') and len(parsed_params['gumbel_sim_thresholds']) > 1:
            params_with_multiple_values.append(f"gumbel_sim_threshold (got {len(parsed_params['gumbel_sim_thresholds'])} values: {parsed_params['gumbel_sim_thresholds']})")
        if parsed_params.get('gumbel_c_scales') and len(parsed_params['gumbel_c_scales']) > 1:
            params_with_multiple_values.append(f"gumbel_c_scale (got {len(parsed_params['gumbel_c_scales'])} values: {parsed_params['gumbel_c_scales']})")
        if (
            parsed_params.get('enable_dead_cell_pruning')
            and len(parsed_params['enable_dead_cell_pruning']) > 1
        ):
            params_with_multiple_values.append(
                "enable_dead_cell_pruning "
                f"(got {len(parsed_params['enable_dead_cell_pruning'])} values: {parsed_params['enable_dead_cell_pruning']})"
            )
        if (
            parsed_params.get('dead_cell_enable_four_run')
            and len(parsed_params['dead_cell_enable_four_run']) > 1
        ):
            params_with_multiple_values.append(
                "dead_cell_enable_four_run "
                f"(got {len(parsed_params['dead_cell_enable_four_run'])} values: {parsed_params['dead_cell_enable_four_run']})"
            )
        if (
            parsed_params.get('dead_cell_enable_two_two_split')
            and len(parsed_params['dead_cell_enable_two_two_split']) > 1
        ):
            params_with_multiple_values.append(
                "dead_cell_enable_two_two_split "
                f"(got {len(parsed_params['dead_cell_enable_two_two_split'])} values: {parsed_params['dead_cell_enable_two_two_split']})"
            )
        if (
            parsed_params.get('dead_cell_enable_three_plus_one')
            and len(parsed_params['dead_cell_enable_three_plus_one']) > 1
        ):
            params_with_multiple_values.append(
                "dead_cell_enable_three_plus_one "
                f"(got {len(parsed_params['dead_cell_enable_three_plus_one'])} values: {parsed_params['dead_cell_enable_three_plus_one']})"
            )
        if (
            parsed_params.get('dead_cell_enable_a1b2a3_discouraged')
            and len(parsed_params['dead_cell_enable_a1b2a3_discouraged']) > 1
        ):
            params_with_multiple_values.append(
                "dead_cell_enable_a1b2a3_discouraged "
                "(got "
                f"{len(parsed_params['dead_cell_enable_a1b2a3_discouraged'])} values: "
                f"{parsed_params['dead_cell_enable_a1b2a3_discouraged']})"
            )
        if (
            parsed_params.get('dead_cell_enable_double_dead_pairs')
            and len(parsed_params['dead_cell_enable_double_dead_pairs']) > 1
        ):
            params_with_multiple_values.append(
                "dead_cell_enable_double_dead_pairs "
                f"(got {len(parsed_params['dead_cell_enable_double_dead_pairs'])} values: {parsed_params['dead_cell_enable_double_dead_pairs']})"
            )
        
        if params_with_multiple_values:
            print("ERROR: Knockout-only tournaments use a single config for all participants.")
            print("Multiple values provided for the following parameters:")
            for param in params_with_multiple_values:
                print(f"  - {param}")
            print("\nFor knockout-only tournaments, provide only a single value per parameter.")
            print("Multiple values are only meaningful when you also have round-robin participants.")
            sys.exit(1)
    
    # Extract first value from each parameter list (for knockout, we use single values)
    # Only override if the parameter was actually provided (non-empty list)
    if parsed_params.get('mcts_sims') and len(parsed_params['mcts_sims']) > 0:
        knockout_config['mcts_sims'] = parsed_params['mcts_sims'][0]
    if (
        parsed_params.get('base_fraction_mcts_moves')
        and len(parsed_params['base_fraction_mcts_moves']) > 0
    ):
        knockout_config['base_fraction_mcts_moves'] = parsed_params['base_fraction_mcts_moves'][0]
    if parsed_params.get('enable_gumbel') and len(parsed_params['enable_gumbel']) > 0:
        # enable_gumbel is already a list of booleans from parse_tournament_parameters
        knockout_config['enable_gumbel_root_selection'] = parsed_params['enable_gumbel'][0]
    if parsed_params.get('temperatures') and isinstance(parsed_params['temperatures'], list) and len(parsed_params['temperatures']) > 0:
        knockout_config['temperature'] = parsed_params['temperatures'][0]
    elif command_line and re.search(r"(^|\s)--temperature(?:=|\s|$)", command_line):
        # Respect singular --temperature only when explicitly provided.
        knockout_config['temperature'] = args.temperature
    if parsed_params.get('c_pucts') and len(parsed_params['c_pucts']) > 0:
        knockout_config['c_puct'] = parsed_params['c_pucts'][0]
    if parsed_params.get('batch_sizes') and len(parsed_params['batch_sizes']) > 0:
        knockout_config['batch_size'] = parsed_params['batch_sizes'][0]
    if parsed_params.get('gumbel_sim_thresholds') and len(parsed_params['gumbel_sim_thresholds']) > 0:
        knockout_config['gumbel_sim_threshold'] = parsed_params['gumbel_sim_thresholds'][0]
    if parsed_params.get('gumbel_c_scales') and len(parsed_params['gumbel_c_scales']) > 0:
        knockout_config['gumbel_c_scale'] = parsed_params['gumbel_c_scales'][0]
    if (
        parsed_params.get('enable_dead_cell_pruning')
        and len(parsed_params['enable_dead_cell_pruning']) > 0
    ):
        knockout_config['enable_dead_cell_pruning'] = parsed_params['enable_dead_cell_pruning'][0]
    if (
        parsed_params.get('dead_cell_enable_four_run')
        and len(parsed_params['dead_cell_enable_four_run']) > 0
    ):
        knockout_config['dead_cell_enable_four_run'] = parsed_params['dead_cell_enable_four_run'][0]
    if (
        parsed_params.get('dead_cell_enable_two_two_split')
        and len(parsed_params['dead_cell_enable_two_two_split']) > 0
    ):
        knockout_config['dead_cell_enable_two_two_split'] = parsed_params['dead_cell_enable_two_two_split'][0]
    if (
        parsed_params.get('dead_cell_enable_three_plus_one')
        and len(parsed_params['dead_cell_enable_three_plus_one']) > 0
    ):
        knockout_config['dead_cell_enable_three_plus_one'] = parsed_params['dead_cell_enable_three_plus_one'][0]
    if (
        parsed_params.get('dead_cell_enable_a1b2a3_discouraged')
        and len(parsed_params['dead_cell_enable_a1b2a3_discouraged']) > 0
    ):
        knockout_config['dead_cell_enable_a1b2a3_discouraged'] = (
            parsed_params['dead_cell_enable_a1b2a3_discouraged'][0]
        )
    if (
        parsed_params.get('dead_cell_enable_double_dead_pairs')
        and len(parsed_params['dead_cell_enable_double_dead_pairs']) > 0
    ):
        knockout_config['dead_cell_enable_double_dead_pairs'] = parsed_params['dead_cell_enable_double_dead_pairs'][0]
    
    # Parse epoch range if specified
    epoch_range = None
    if args.epoch_range:
        try:
            epoch_range = parse_epoch_range(args.epoch_range)
            print(f"Using epoch range: {epoch_range[0]}-{epoch_range[1]-1}")
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    
    # Parse mini epoch range if specified
    mini_epoch_range = None
    if args.mini_epoch_range:
        try:
            mini_epoch_range = parse_mini_epoch_range(args.mini_epoch_range)
            print(f"Using mini epoch range: {mini_epoch_range[0]}-{mini_epoch_range[1]-1}")
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    
    # Create round-robin participants from existing strategy configs
    round_robin_participants = []
    for i, (strategy_config, model_path) in enumerate(zip(strategy_configs, model_paths)):
        # Create strategy_config with model_path, strategy type, and temperature included
        participant_strategy_config = strategy_config.config.copy()
        participant_strategy_config["model_path"] = model_path
        participant_strategy_config["strategy"] = strategy_config.strategy_type  # FIX: Add strategy type
        participant_strategy_config["temperature"] = strategy_config.temperature  # FIX: Add temperature
        
        # Use the strategy config's name (which includes model, strategy, and parameters)
        # This ensures the participant name matches what will be used in tournament results
        participant = TournamentParticipant(
            name=strategy_config.name,
            strategy_config=participant_strategy_config,
            metadata={
                "strategy_name": str(strategy_config)
            }
        )
        round_robin_participants.append(participant)
    
    # Handle knockout participants: from MODEL_GENERATIONS, latest-run sampling, or directory discovery.
    knockout_participants = None
    most_recent_selection_info = None
    knockout_dir = args.knockout_dir if not args.knockout_from_generations else None
    if args.knockout_from_generations:
        # Get all participants from MODEL_GENERATIONS
        knockout_participants = get_all_model_participants_from_generations(knockout_config)
        print(f"Loaded {len(knockout_participants)} models from MODEL_GENERATIONS")
    elif args.most_recent is not None or args.most_recent_biased is not None:
        try:
            use_biased_sampling = args.most_recent_biased is not None
            checkpoint_count = (
                args.most_recent_biased
                if use_biased_sampling
                else args.most_recent
            )
            knockout_participants, most_recent_selection_info = _build_most_recent_knockout_participants(
                count=checkpoint_count,
                biased=use_biased_sampling,
                root_dir=args.most_recent_root,
                knockout_config=knockout_config,
            )
            print(
                "Loaded "
                f"{len(knockout_participants)} checkpoints from latest run for knockout "
                f"({most_recent_selection_info['selection_mode']} mode)"
            )
            print(f"  Latest run directory: {most_recent_selection_info['latest_run_dir']}")
            print(f"  Checkpoint directory: {most_recent_selection_info['checkpoint_dir']}")
            print(f"  Offsets from latest checkpoint: {most_recent_selection_info['offsets']}")
            print(f"  Selected checkpoints: {most_recent_selection_info['selected_checkpoints']}")
            knockout_dir = None
        except (FileNotFoundError, ValueError) as error:
            print(f"ERROR: {error}")
            sys.exit(1)
    
    # Validate epoch/mini epoch ranges are not used with knockout-from-generations
    if args.knockout_from_generations and (epoch_range or mini_epoch_range):
        print("WARNING: --epoch-range and --mini-epoch-range are ignored when using --knockout-from-generations")
        epoch_range = None
        mini_epoch_range = None
    
    # Create and run two-stage tournament
    tournament = TwoStageTournament(
        knockout_dir=knockout_dir,
        knockout_participants=knockout_participants,
        knockout_config=knockout_config,
        round_robin_participants=round_robin_participants,
        games_per_match=args.games_per_match,
        top_k=args.top_k,
        round_robin_games=args.round_robin_games,
        epoch_range=epoch_range,
        mini_epoch_range=mini_epoch_range,
        command_line=command_line,
        run_desc=args.run_desc,
        seed=args.seed,
        trmph_source=args.trmph_source,
        mps_empty_cache_per_pair=args.mps_empty_cache_per_pair
    )
    
    print("Running 2-stage tournament...")
    if args.knockout_from_generations:
        print(f"  Knockout participants: {len(knockout_participants)} models from MODEL_GENERATIONS")
    elif most_recent_selection_info:
        print(
            f"  Knockout participants: {len(knockout_participants)} checkpoints from latest run "
            f"({most_recent_selection_info['selection_mode']})"
        )
        print(f"  Latest run directory: {most_recent_selection_info['latest_run_dir']}")
        print(f"  Checkpoint directory: {most_recent_selection_info['checkpoint_dir']}")
        print(f"  Checkpoint offsets: {most_recent_selection_info['offsets']}")
    else:
        print(f"  Knockout directory: {args.knockout_dir}")
    print(f"  Knockout config: {knockout_config}")
    print(f"  Games per match: {args.games_per_match}")
    print(f"  Top K: {args.top_k}")
    print(f"  Round-robin openings per pair: {args.round_robin_games}")
    print(f"  Round-robin games per pair (with color swap): {args.round_robin_games * 2}")
    print(f"  Round-robin participants: {len(round_robin_participants)}")
    print()
    
    # Run the tournament
    results = tournament.run_tournament()
    
    # Convert results to a format compatible with existing tournament result system
    # For now, create a simple result object
    class TwoStageTournamentResult:
        def __init__(self, results):
            self.results = results
            self.participants = []
            
            # Extract participants from results
            if results.get("knockout_results"):
                self.participants.extend([p.name for p in results["knockout_results"]["winners"]])
            if results.get("round_robin_results"):
                self.participants.extend([p["name"] for p in results["round_robin_results"]["participants"]])
        
        def get_summary(self):
            return self.results
    
        def print_results(self):
            print("\n" + "="*60)
            print("2-STAGE TOURNAMENT RESULTS")
            print("="*60)
            
            if self.results.get("knockout_results"):
                ko_results = self.results["knockout_results"]
                print(f"Knockout Stage:")
                print(f"  Total participants: {ko_results['total_participants']}")
                print(f"  Winners: {[p.name for p in ko_results['winners']]}")
                print()
            
            if self.results.get("round_robin_results"):
                rr_results = self.results["round_robin_results"]
                print(f"Round-Robin Stage:")
                print(f"  Total participants: {rr_results['total_participants']}")
                print(f"  Final ranking: {rr_results['ranking']}")
                print()
            
            if self.results.get("final_ranking"):
                print(f"Final Tournament Ranking: {self.results['final_ranking']}")
            
            print("="*60)
    
    return TwoStageTournamentResult(results), tournament.output_dir


def _apply_dead_cell_debug_logging_to_strategies(
    args,
    strategy_configs: List[StrategyConfig],
) -> None:
    """Apply optional dead-cell root-debug logging settings to masked MCTS strategies."""
    if not args.dead_cell_debug_log_path:
        return
    if args.dead_cell_debug_max_records_per_move < 0:
        print(
            "ERROR: --dead-cell-debug-max-records-per-move must be >= 0, "
            f"got {args.dead_cell_debug_max_records_per_move}"
        )
        sys.exit(1)

    configured_count = 0
    for strategy_config in strategy_configs:
        if strategy_config.strategy_type != "mcts":
            continue
        if not strategy_config.config.get("enable_dead_cell_pruning", False):
            continue
        strategy_config.config["dead_cell_debug_log_path"] = args.dead_cell_debug_log_path
        strategy_config.config["dead_cell_debug_strategy_label"] = strategy_config.name
        strategy_config.config["dead_cell_debug_max_records_per_move"] = (
            args.dead_cell_debug_max_records_per_move
        )
        configured_count += 1

    if configured_count == 0:
        print(
            "WARNING: --dead-cell-debug-log-path was set, but no masked MCTS strategies "
            "were found. No dead-cell debug records will be written."
        )
        return

    print(
        "Dead-cell debug logging enabled for "
        f"{configured_count} masked MCTS strateg{'y' if configured_count == 1 else 'ies'}."
    )
    print(f"  JSONL path: {args.dead_cell_debug_log_path}")
    print(
        "  Max records per move: "
        f"{args.dead_cell_debug_max_records_per_move} "
        "(0 means unlimited)"
    )


def _apply_dead_cell_counterfactual_logging_to_strategies(
    args,
    strategy_configs: List[StrategyConfig],
) -> None:
    """Apply optional counterfactual root-choice debug logging to masked MCTS strategies."""
    if not args.dead_cell_counterfactual_debug_log_path:
        return

    configured_count = 0
    for strategy_config in strategy_configs:
        if strategy_config.strategy_type != "mcts":
            continue
        if not strategy_config.config.get("enable_dead_cell_pruning", False):
            continue
        strategy_config.config["dead_cell_counterfactual_debug_log_path"] = (
            args.dead_cell_counterfactual_debug_log_path
        )
        configured_count += 1

    if configured_count == 0:
        print(
            "WARNING: --dead-cell-counterfactual-debug-log-path was set, but no masked MCTS "
            "strategies were found. No counterfactual dead-cell debug records will be written."
        )
        return

    print(
        "Dead-cell counterfactual debug logging enabled for "
        f"{configured_count} masked MCTS strateg{'y' if configured_count == 1 else 'ies'}."
    )
    print(f"  JSONL path: {args.dead_cell_counterfactual_debug_log_path}")


def main():
    args = parse_args()
    
    # Configure logging verbosity based on --verbose argument
    configure_logging_verbosity(args.verbose)

    # Optional: lightweight MCTS timing breakdown (GPU vs CPU).
    if args.mcts_profile:
        from hex_ai.inference.move_selection import configure_mcts_profiling
        configure_mcts_profiling(
            enabled=True,
            every_n_calls=args.mcts_profile_every,
            max_calls=args.mcts_profile_max_calls,
        )
    
    # Get command line early - crash if not available
    try:
        command_line = get_command_line()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Generate seed if none provided, or use provided seed
    if args.seed is None:
        args.seed = int(time.time())
        print(f"Auto-generated seed: {args.seed}")

    # Backward compatibility: --num-openings is an alias for --round-robin-games.
    if args.num_openings != DEFAULT_NUM_OPENINGS:
        if args.round_robin_games == DEFAULT_NUM_OPENINGS:
            args.round_robin_games = args.num_openings
            print(
                f"INFO: Interpreting --num-openings={args.num_openings} as "
                f"--round-robin-games={args.round_robin_games}."
            )
        elif args.round_robin_games != args.num_openings:
            print(
                "ERROR: --num-openings and --round-robin-games disagree. "
                f"Got --num-openings={args.num_openings}, "
                f"--round-robin-games={args.round_robin_games}."
            )
            sys.exit(1)
    
    # Set random seed for reproducible opening selection
    set_deterministic_seeds(args.seed)
    
    # Parse models and strategies using clean separation of concerns
    if args.models and (args.model_files or args.model_dirs):
        print("ERROR: Cannot specify both --models and --model-files/--model-dirs. Use one or the other.")
        sys.exit(1)
    
    # Validate knockout tournament source arguments
    knockout_sources = {
        "--knockout-dir": bool(args.knockout_dir),
        "--knockout-from-generations": bool(args.knockout_from_generations),
        "--most-recent": args.most_recent is not None,
        "--most-recent-biased": args.most_recent_biased is not None,
    }
    enabled_knockout_sources = [name for name, enabled in knockout_sources.items() if enabled]
    if len(enabled_knockout_sources) > 1:
        print("ERROR: Multiple knockout sources specified. Choose exactly one of:")
        print("  --knockout-dir, --knockout-from-generations, --most-recent, --most-recent-biased")
        print(f"  Provided: {', '.join(enabled_knockout_sources)}")
        sys.exit(1)

    if args.most_recent is not None and args.most_recent < 2:
        print(f"ERROR: --most-recent must be >= 2, got {args.most_recent}")
        sys.exit(1)
    if args.most_recent_biased is not None and args.most_recent_biased < 2:
        print(f"ERROR: --most-recent-biased must be >= 2, got {args.most_recent_biased}")
        sys.exit(1)
    if (args.most_recent is not None or args.most_recent_biased is not None) and (
        args.epoch_range or args.mini_epoch_range
    ):
        print("ERROR: --epoch-range and --mini-epoch-range are not supported with --most-recent or --most-recent-biased")
        print("Use --knockout-dir with explicit ranges if you need range filters.")
        sys.exit(1)
    
    # For 2-stage tournaments, models/strategies are optional (only for round-robin stage)
    # Skip this check whenever a knockout source is configured.
    if not enabled_knockout_sources:
        if not args.models and not (args.model_files and args.model_dirs):
            print("ERROR: Must specify either --models (registry) or both --model-files and --model-dirs (direct)")
            sys.exit(1)
    
    # Parse strategy names (e.g., "mcts", "policy")
    # For knockout-only tournaments, strategies are optional
    if args.strategies:
        strategy_names = [name.strip() for name in args.strategies.split(',')]
    else:
        strategy_names = []
    auto_default_round_robin_models = _should_auto_default_round_robin_models(args, strategy_names)
    
    # Parse model specifications (only needed if not knockout-only tournament)
    if is_knockout_only_tournament(args):
        # Knockout-only tournament - no model paths or strategy configs needed
        model_paths = []
        strategy_configs = []
    else:
        if auto_default_round_robin_models:
            default_model_count = len(strategy_names)
            try:
                model_paths = get_primary_model_paths_from_recent_generations(
                    count=default_model_count,
                    skip_most_recent=DEFAULT_ROUND_ROBIN_SKIP_RECENT_GENERATIONS,
                )
            except (ValueError, FileNotFoundError) as error:
                print(f"ERROR: Failed to load default round-robin models from MODEL_GENERATIONS: {error}")
                sys.exit(1)

            if len(strategy_names) != len(model_paths):
                print(
                    "ERROR: Default round-robin model selection produced "
                    f"{len(model_paths)} models, but {len(strategy_names)} strategies were provided."
                )
                print(f"Provide exactly {len(model_paths)} strategies.")
                sys.exit(1)

            print(
                "INFO: Using default round-robin models from MODEL_GENERATIONS "
                f"(skip newest {DEFAULT_ROUND_ROBIN_SKIP_RECENT_GENERATIONS}, "
                f"take next {default_model_count})."
            )
            for model_path in model_paths:
                print(f"  - {model_path}")
        else:
            # Parse model specifications
            model_paths = parse_model_specifications(args, strategy_names)
        
        # Create strategy configurations
        try:
            strategy_configs = create_strategy_configs_for_tournament(
                args,
                strategy_names,
                model_paths,
                num_games=args.round_robin_games,
                board_size=13,
                pie_rule=False,
            )
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        _apply_dead_cell_debug_logging_to_strategies(args, strategy_configs)
        _apply_dead_cell_counterfactual_logging_to_strategies(args, strategy_configs)
        
        # Check for Gumbel algorithm issues and print warnings
        check_gumbel_configurations(strategy_configs)
    
    # Determine how many openings to play per pair.
    # Always use --round-robin-games for the unified tournament system
    # (legacy name retained for CLI compatibility).
    openings_to_play = args.round_robin_games
    
    # Generate or load opening positions (skip for knockout-only tournaments)
    if is_knockout_only_tournament(args):
        # Knockout-only tournament - no openings needed
        all_openings = []
        openings = []
    elif args.opening_file and os.path.exists(args.opening_file):
        print(f"Loading openings from: {args.opening_file}")
        try:
            all_openings = load_openings_from_file(args.opening_file, args.opening_length)
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    else:
        print(f"Generating diverse openings...")
        
        # Find TRMPH files
        trmph_files = find_trmph_files(args.trmph_source)
        if not trmph_files:
            print(f"ERROR: No TRMPH files found in {args.trmph_source}")
            sys.exit(1)
        
        # Generate diverse openings (generate more than needed to allow for random selection)
        # target_generation = max(openings_to_play * 2, 500)  # Generate at least 2x what we need
        # TODO: Figure out whether we need to generate more that we're planning to use for anything.
        target_generation = openings_to_play
        all_openings = generate_diverse_openings(
            trmph_files, 
            opening_length=args.opening_length,
            target_count=target_generation,
            cache_file=args.cache_file
        )
    
    if not is_knockout_only_tournament(args):
        if not all_openings:
            print("ERROR: No opening positions generated")
            sys.exit(1)
        
        # Randomly select the desired number of openings from the available pool
        print(f"Randomly selecting {openings_to_play} openings from pool of {len(all_openings)}...")
        openings = select_random_openings(all_openings, openings_to_play, seed=args.seed)
    
    # Print configuration using unified logging based on resolved strategy settings.
    strategy_names = [str(config) for config in strategy_configs]
    resolved_logging = _collect_resolved_logging_values(strategy_configs)
    
    # Create unified script config (skip for knockout-only tournaments)
    if not is_knockout_only_tournament(args):
        script_config = ScriptConfig(
            script_type="tournament",
            models=model_paths,
            strategies=strategy_names,
            num_games=openings_to_play * 2,  # Each opening is played twice with swapped colors.
            strategy_config={},  # Strategy configs are handled individually
            temperatures=resolved_logging["temperatures"],
            pie_rule=False,  # Deterministic tournaments don't use pie rule
            opening_length=args.opening_length,
            batch_sizes=resolved_logging["batch_sizes"],
            c_puct=resolved_logging["c_puct"],
            mcts_sims=resolved_logging["mcts_sims"],
            base_fraction_mcts_moves=resolved_logging["base_fraction_mcts_moves"],
            enable_gumbel=resolved_logging["enable_gumbel"],
            gumbel_sim_threshold=resolved_logging["gumbel_sim_threshold"],
            gumbel_c_visit=resolved_logging["gumbel_c_visit"],
            gumbel_c_scale=resolved_logging["gumbel_c_scale"],
            gumbel_candidate_power_scale=resolved_logging["gumbel_candidate_power_scale"],
            gumbel_candidate_power_rate=resolved_logging["gumbel_candidate_power_rate"],
            gumbel_candidate_power_offset=resolved_logging["gumbel_candidate_power_offset"],
            gumbel_m_candidates=resolved_logging["gumbel_m_candidates"],
            confidence_termination_threshold=TOURNAMENT_CONFIDENCE_TERMINATION_THRESHOLD,
        )
    else:
        # No script config needed for knockout-only tournaments
        script_config = None
    
    if script_config:
        print_script_configuration(script_config)
    
    # Print additional deterministic tournament specific info
    if not is_knockout_only_tournament(args):
        print(f"  Number of openings: {len(openings)} (randomly selected from pool of {len(all_openings)})")
        print(f"  Games per strategy pair (from openings): {len(openings) * 2}")
        print_round_robin_strategy_summary(strategy_configs)
        print()
    
    # Optional: memory profiling (RSS + tracemalloc heap).
    # This is intended for long-run leak triage. It should not be enabled by default.
    if args.memory_profile:
        start_profiling(output_dir=args.memory_profile_dir, interval_seconds=args.memory_profile_interval)

    try:
        # Always use the unified 2-stage tournament system
        # If knockout_dir is None, it will skip the knockout stage and go straight to round-robin
        result, actual_output_dir = run_two_stage_tournament(args, strategy_configs, model_paths, openings, command_line)
    finally:
        if args.memory_profile:
            stop_profiling()
    
    # Print results using unified analyzer
    # Use the actual output directory from the tournament, not a new timestamp
    output_files = {
        "directory": actual_output_dir
    }
    
    if script_config:
        print_script_results("tournament", result, script_config, output_files)
    else:
        # For knockout-only tournaments, just print the results directly
        result.print_results()


if __name__ == "__main__":
    main()
