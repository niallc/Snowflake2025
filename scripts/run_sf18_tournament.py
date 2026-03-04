#!/usr/bin/env python3
"""
Run a tournament between Snowflake 2025 models and Snowflake 2018 (SF18) AI.

This script allows you to compare your current SF25 models against the older
SF18 AI by running tournaments with pre-generated opening positions.

The SF18 AI runs as a separate webserver that must be started independently.
This script communicates with it via HTTP API calls.

Examples:

1. Compare SF25 models against SF18 with default settings:
   python scripts/run_sf18_tournament.py \
     --model-files=epoch11_mini15.pt.gz,epoch16_mini23.pt.gz \
     --model-dirs=checkpoints/dir1,checkpoints/dir2 \
     --strategies=mcts,mcts \
     --mcts-sims=30,30 \
     --num-openings=100

2. Use specific SF18 difficulty and server URL:
   python scripts/run_sf18_tournament.py \
     --models=best \
     --strategies=mcts,mcts \
     --mcts-sims=100,100 \
     --sf18-difficulty=8 \
     --sf18-server-url=http://localhost:8088 \
     --num-openings=50

3. Use custom opening file:
   python scripts/run_sf18_tournament.py \
     --models=best \
     --strategies=mcts \
     --mcts-sims=30 \
     --opening-file=data/deterministic_openings.txt \
     --sf18-difficulty=9
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from hex_ai.config import (
    BOARD_SIZE,
    DEFAULT_BATCH_CAP,
    DEFAULT_C_PUCT,
    DEFAULT_MCTS_SIMS,
    DEFAULT_GUMBEL_SIM_THRESHOLD,
    DEFAULT_GUMBEL_C_SCALE,
    DEFAULT_GUMBEL_CANDIDATE_POWER_SCALE,
    DEFAULT_GUMBEL_CANDIDATE_POWER_RATE,
    DEFAULT_GUMBEL_CANDIDATE_POWER_OFFSET,
    TOURNAMENT_CONFIDENCE_TERMINATION_THRESHOLD,
    TRMPH_PREFIX,
)
from hex_ai.enums import Player, Winner
from hex_ai.inference.game_engine import apply_move_to_state
from hex_ai.inference.sf18_client import SF18Client, SF18Player
from hex_ai.inference.move_selection import get_strategy, MoveSelectionConfig
from hex_ai.inference.strategy_config import StrategyConfig
from hex_ai.move_provenance import (
    MOVE_CODE_VISIT_COUNT,
    make_move_provenance_record,
    sidecar_path_for_trmph,
)
from hex_ai.utils.format_conversion import rowcol_to_trmph
from hex_ai.utils.tournament_logging import append_trmph_winner_line, write_tournament_trmph_header, find_available_csv_filename, get_command_line
from hex_ai.utils.tournament_utils import (
    parse_model_specifications,
    create_strategy_configs_for_tournament,
    format_strategy_configuration_details,
)
from hex_ai.utils.deterministic_tournament_utils import (
    setup_tournament_output,
    save_opening_positions,
    setup_strategy_pair_files,
    create_play_config_for_pair,
)
from hex_ai.utils.random_utils import set_deterministic_seeds
from hex_ai.inference.model_cache import create_temporary_model_cache
from hex_ai.inference.game_execution import (
    OpeningPosition,
    POLICY_TARGET_CONSTRUCTION_VERSION,
    TOURNAMENT_NON_TRAINABLE_PROVENANCE_CODE,
    _build_policy_target_vector_from_gumbel_final_scores,
    _build_policy_target_vector_from_mcts_result,
    _one_hot_policy_target_vector,
    _resolve_provenance_code_from_selected_move_source,
    _zero_policy_target_vector,
    find_trmph_files,
    generate_diverse_openings,
    load_openings_from_file,
    select_random_openings,
)

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_OPENING_LENGTH = 5
DEFAULT_NUM_OPENINGS = 100
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = None  # Will be set to int(time.time()) if None
DEFAULT_VERBOSE = 1
DEFAULT_SF18_DIFFICULTY = 9
DEFAULT_SF18_SERVER_URL = "http://localhost:8088"
TRMPH_SOURCE_DIR = "data/sf25/sep28"
OUTPUT_DIR_PREFIX = "data/tournament_play/sf18_vs_sf25/sf18_tournament_"


class SF18TournamentResult:
    """Tournament result for SF18 vs SF25 matches."""
    
    def __init__(self, sf25_models: List[str], sf18_difficulty: int):
        self.sf25_models = sf25_models
        self.sf18_difficulty = sf18_difficulty
        self.results = {}  # Dict mapping (sf25_model, sf18_difficulty) -> win/loss counts
        self.game_results = []  # List of individual game results
        
        # Initialize results tracking
        for model in sf25_models:
            self.results[(model, sf18_difficulty)] = {
                'sf25_wins': 0,
                'sf18_wins': 0,
                'total_games': 0,
                'sf25_wins_as_blue': 0,
                'sf25_wins_as_red': 0,
                'sf18_wins_as_blue': 0,
                'sf18_wins_as_red': 0,
                'openings_sf25_won_both': 0,  # Openings where SF25 won both games
                'openings_sf18_won_both': 0,  # Openings where SF18 won both games
                'openings_split': 0  # Openings where each won one game
            }
    
    def record_game(self, sf25_model: str, winner: str, game_data: Dict[str, Any]):
        """Record the result of a single game."""
        key = (sf25_model, self.sf18_difficulty)
        
        if key not in self.results:
            self.results[key] = {
                'sf25_wins': 0,
                'sf18_wins': 0,
                'total_games': 0,
                'sf25_wins_as_blue': 0,
                'sf25_wins_as_red': 0,
                'sf18_wins_as_blue': 0,
                'sf18_wins_as_red': 0,
                'openings_sf25_won_both': 0,
                'openings_sf18_won_both': 0,
                'openings_split': 0
            }
        
        self.results[key]['total_games'] += 1
        
        # Track overall wins
        if winner == 'sf25':
            self.results[key]['sf25_wins'] += 1
        elif winner == 'sf18':
            self.results[key]['sf18_wins'] += 1
        
        # Track color-specific wins
        winner_color = game_data.get('winner', 'unknown')
        if winner == 'sf25':
            if winner_color == 'blue':
                self.results[key]['sf25_wins_as_blue'] += 1
            elif winner_color == 'red':
                self.results[key]['sf25_wins_as_red'] += 1
        elif winner == 'sf18':
            if winner_color == 'blue':
                self.results[key]['sf18_wins_as_blue'] += 1
            elif winner_color == 'red':
                self.results[key]['sf18_wins_as_red'] += 1
        
        self.game_results.append({
            'sf25_model': sf25_model,
            'winner': winner,
            'game_data': game_data
        })
    
    def record_opening_results(self, sf25_model: str, opening_idx: int, 
                              sf25_won_first: bool, sf25_won_second: bool):
        """Record the results for both games of an opening."""
        key = (sf25_model, self.sf18_difficulty)
        
        if key not in self.results:
            self.results[key] = {
                'sf25_wins': 0,
                'sf18_wins': 0,
                'total_games': 0,
                'sf25_wins_as_blue': 0,
                'sf25_wins_as_red': 0,
                'sf18_wins_as_blue': 0,
                'sf18_wins_as_red': 0,
                'openings_sf25_won_both': 0,
                'openings_sf18_won_both': 0,
                'openings_split': 0
            }
        
        # Track opening-level results
        if sf25_won_first and sf25_won_second:
            self.results[key]['openings_sf25_won_both'] += 1
        elif not sf25_won_first and not sf25_won_second:
            self.results[key]['openings_sf18_won_both'] += 1
        else:
            self.results[key]['openings_split'] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get tournament summary."""
        summary = {
            'sf25_models': self.sf25_models,
            'sf18_difficulty': self.sf18_difficulty,
            'model_results': {}
        }
        
        for (model, difficulty), stats in self.results.items():
            if stats['total_games'] > 0:
                win_rate = stats['sf25_wins'] / stats['total_games']
                summary['model_results'][model] = {
                    'sf25_wins': stats['sf25_wins'],
                    'sf18_wins': stats['sf18_wins'],
                    'total_games': stats['total_games'],
                    'sf25_win_rate': win_rate,
                    'sf25_win_percentage': win_rate * 100
                }
        
        return summary
    
    def print_results(self):
        """Print tournament results."""
        print("\n" + "="*60)
        print("SF18 vs SF25 TOURNAMENT RESULTS")
        print("="*60)
        print(f"SF18 Difficulty: {self.sf18_difficulty}")
        print(f"SF25 Models: {len(self.sf25_models)}")
        print()
        
        for (model, difficulty), stats in self.results.items():
            if stats['total_games'] > 0:
                win_rate = stats['sf25_wins'] / stats['total_games']
                print(f"Model: {model}")
                print(f"  SF25 Wins: {stats['sf25_wins']}")
                print(f"  SF18 Wins: {stats['sf18_wins']}")
                print(f"  Total Games: {stats['total_games']}")
                print(f"  SF25 Win Rate: {win_rate*100:.1f}%")
                print()
                
                # Color-specific breakdown
                print(f"  Color-specific wins:")
                print(f"    SF25 as Blue: {stats['sf25_wins_as_blue']}")
                print(f"    SF25 as Red: {stats['sf25_wins_as_red']}")
                print(f"    SF18 as Blue: {stats['sf18_wins_as_blue']}")
                print(f"    SF18 as Red: {stats['sf18_wins_as_red']}")
                print()
                
                # Opening-level results
                total_openings = (stats['openings_sf25_won_both'] + 
                                stats['openings_sf18_won_both'] + 
                                stats['openings_split'])
                if total_openings > 0:
                    print(f"  Opening-level results (out of {total_openings} openings):")
                    print(f"    Openings SF25 won both ways: {stats['openings_sf25_won_both']}")
                    print(f"    Openings SF18 won both ways: {stats['openings_sf18_won_both']}")
                    print(f"    Openings split (1-1): {stats['openings_split']}")
                    print()
        
        print("="*60)


def play_sf18_vs_sf25_game(
    model_cache,
    sf25_strategy: StrategyConfig,
    sf18_player: SF18Player,
    opening: OpeningPosition,
    temperature: float = DEFAULT_TEMPERATURE,
    board_size: int = BOARD_SIZE,
    verbose: int = 0,
    sf25_is_blue: bool = True
) -> Dict[str, Any]:
    """
    Play a single game between an SF25 model and SF18 AI.
    
    Args:
        model_cache: Model cache for SF25 model
        sf25_strategy: Strategy configuration for SF25 model
        sf18_player: SF18 player instance
        opening: Opening position to start from
        temperature: Temperature for SF25 move selection
        board_size: Board size
        verbose: Verbosity level
        sf25_is_blue: Whether SF25 plays as blue (first player)
        
    Returns:
        Dictionary containing game result information
    """
    # Start from opening position
    state = opening.get_state(board_size)
    
    # Get move selection config for SF25
    move_config = MoveSelectionConfig(
        temperature=temperature,
        **sf25_strategy.config
    )
    
    # Get strategy object for SF25
    sf25_strategy_obj = get_strategy(sf25_strategy.strategy_type)
    
    # Track move sequence for TRMPH generation
    move_sequence = list(opening.moves)  # Start with opening moves
    move_provenance_codes: List[str] = []
    policy_target_rows: List[np.ndarray] = []
    for opening_move in opening.moves:
        move_provenance_codes.append(MOVE_CODE_VISIT_COUNT)
        policy_target_rows.append(
            _one_hot_policy_target_vector(opening_move, board_size)
        )
    move_count = len(opening.moves)
    max_moves = board_size * board_size
    
    # Play the game
    while move_count < max_moves:
        if state.game_over:
            break
        
        # Show progress for verbose >= 3
        if verbose >= 3 and move_count % 10 == 0 and move_count > 0:
            print(f"    Move {move_count}...", end="", flush=True)
        
        try:
            if (state.current_player_enum == Player.BLUE and sf25_is_blue) or \
               (state.current_player_enum == Player.RED and not sf25_is_blue):
                # SF25's turn
                model = model_cache.get_simple_model(sf25_strategy.model_path)
                # Only show detailed MCTS output at very high verbosity
                mcts_verbose = max(0, verbose - 2) if verbose >= 4 else 0
                row, col = sf25_strategy_obj.select_move(state, model, move_config, verbose=mcts_verbose)
                move_metadata = sf25_strategy_obj.pop_last_move_metadata()
                if move_metadata is None:
                    provenance_code = TOURNAMENT_NON_TRAINABLE_PROVENANCE_CODE
                    policy_target = _zero_policy_target_vector(board_size)
                else:
                    provenance_code = _resolve_provenance_code_from_selected_move_source(
                        move_metadata.get("selected_move_source")
                    )
                    if provenance_code == "G":
                        mcts_result = move_metadata.get("mcts_result")
                        if mcts_result is None:
                            raise RuntimeError(
                                "Gumbel-root SF25 tournament move missing MCTS result payload."
                            )
                        policy_target = _build_policy_target_vector_from_gumbel_final_scores(
                            mcts_result, board_size=board_size
                        )
                    elif provenance_code in {"V", "T"}:
                        mcts_result = move_metadata.get("mcts_result")
                        if mcts_result is None:
                            raise RuntimeError(
                                "Trainable SF25 tournament move missing MCTS result payload."
                            )
                        policy_target = _build_policy_target_vector_from_mcts_result(
                            mcts_result, board_size=board_size
                        )
                    else:
                        policy_target = _zero_policy_target_vector(board_size)
            else:
                # SF18's turn
                row, col = sf18_player.get_move(state)
                provenance_code = TOURNAMENT_NON_TRAINABLE_PROVENANCE_CODE
                policy_target = _zero_policy_target_vector(board_size)
            
            # Apply the move
            state = apply_move_to_state(state, row, col)
            move_sequence.append((row, col))  # Track the move
            move_provenance_codes.append(provenance_code)
            policy_target_rows.append(policy_target)
            move_count += 1
            
            if verbose >= 4:
                print(f"  Move {move_count}: {rowcol_to_trmph(row, col, board_size)} by {'SF25' if ((state.current_player_enum == Player.RED and sf25_is_blue) or (state.current_player_enum == Player.BLUE and not sf25_is_blue)) else 'SF18'}")
        
        except Exception as e:
            logger.error(f"Error during game play: {e}")
            break
    
    # Add newline after progress indicator if we were showing progress
    if verbose >= 3 and move_count > 10:
        print()  # Newline after progress dots
    
    # Convert move sequence to TRMPH (preserving move order) - do this before error checking
    trmph_moves = ''.join([rowcol_to_trmph(r, c, board_size) for r, c in move_sequence])
    trmph_str = f"{TRMPH_PREFIX}{trmph_moves}"
    
    # Determine winner - this should always be valid in Hex
    if not state.game_over:
        # Game didn't finish - this indicates a serious bug
        print(f"\nERROR: Game did not finish properly!")
        print(f"Game state debug info:")
        print(f"  - Game over: {state.game_over}")
        print(f"  - Winner: {state.winner}")
        print(f"  - Current player: {state.current_player_enum}")
        print(f"  - Move count: {move_count}")
        print(f"  - Max moves: {max_moves}")
        print(f"  - Final TRMPH: {trmph_str}")
        print(f"  - Opening: {opening}")
        raise RuntimeError("Game did not finish - this should never happen in Hex")
    
    if state.winner is None:
        # Winner is None but game is over - this is a serious bug
        print(f"\nERROR: Game is over but winner is None!")
        print(f"Game state debug info:")
        print(f"  - Game over: {state.game_over}")
        print(f"  - Winner: {state.winner}")
        print(f"  - Winner type: {type(state.winner)}")
        print(f"  - Current player: {state.current_player_enum}")
        print(f"  - Move count: {move_count}")
        print(f"  - Final TRMPH: {trmph_str}")
        raise RuntimeError("Game is over but winner is None - this is a bug in the game engine")
    
    winner = state.winner
    if winner == Winner.BLUE:
        winner_str = "blue"
        winner_strategy = "SF25" if sf25_is_blue else "SF18"
    elif winner == Winner.RED:
        winner_str = "red"
        winner_strategy = "SF18" if sf25_is_blue else "SF25"
    else:
        # Unknown winner enum - this is a serious bug
        print(f"\nERROR: Unknown winner enum: {winner} (type: {type(winner)})")
        print(f"Game state debug info:")
        print(f"  - Game over: {state.game_over}")
        print(f"  - Winner: {state.winner}")
        print(f"  - Current player: {state.current_player_enum}")
        print(f"  - Move count: {move_count}")
        print(f"  - Final TRMPH: {trmph_str}")
        raise RuntimeError(f"Unknown winner enum: {winner} - this is a bug in the game engine")
        
    # Show game summary at verbose >= 1
    if verbose >= 1:
        print(f"    Game complete: {winner_strategy} wins in {move_count} moves")

    expected_move_count = len(move_sequence)
    if len(move_provenance_codes) != expected_move_count:
        raise RuntimeError(
            "SF18 tournament move provenance length mismatch: "
            f"expected {expected_move_count}, got {len(move_provenance_codes)}"
        )
    if len(policy_target_rows) != expected_move_count:
        raise RuntimeError(
            "SF18 tournament policy-target row count mismatch: "
            f"expected {expected_move_count}, got {len(policy_target_rows)}"
        )
    if policy_target_rows:
        policy_targets_matrix = np.stack(policy_target_rows, axis=0).astype(
            np.float32, copy=False
        )
    else:
        policy_targets_matrix = np.zeros(
            (0, board_size * board_size), dtype=np.float32
        )
    
    return {
        'winner': winner_str,
        'winner_strategy': winner_strategy,
        'trmph_str': trmph_str,
        'total_moves': move_count,
        'opening': opening,
        'sf25_strategy': sf25_strategy.name,
        'sf18_difficulty': sf18_player.difficulty,
        'move_provenance_codes': ''.join(move_provenance_codes),
        'policy_targets_matrix': policy_targets_matrix,
        'policy_target_source_codes': ''.join(move_provenance_codes),
        'policy_target_version': POLICY_TARGET_CONSTRUCTION_VERSION,
    }


def run_sf18_tournament(
    strategy_configs: List[StrategyConfig],
    sf18_client: SF18Client,
    sf18_difficulty: int,
    openings: List[OpeningPosition],
    temperature: float = DEFAULT_TEMPERATURE,
    verbose: int = DEFAULT_VERBOSE,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
    command_line: str = None,
    run_desc: Optional[str] = None
) -> SF18TournamentResult:
    """
    Run a tournament between SF25 models and SF18 AI.
    
    Args:
        strategy_configs: List of SF25 strategy configurations
        sf18_client: SF18 client for communication
        sf18_difficulty: Difficulty level for SF18
        openings: List of opening positions to use
        temperature: Temperature for SF25 move selection
        verbose: Verbosity level
        seed: Random seed for reproducibility
        output_dir: Output directory for tournament files
        command_line: Command line that was used to run the tournament
        run_desc: Optional description of this tournament run
        
    Returns:
        SF18TournamentResult with tournament results
    """
    # Create tournament result tracker
    sf25_models = [config.name for config in strategy_configs]
    result = SF18TournamentResult(sf25_models, sf18_difficulty)
    
    # Set up tournament output
    if output_dir is None:
        output_dir, openings_file = setup_tournament_output(OUTPUT_DIR_PREFIX)
    else:
        os.makedirs(output_dir, exist_ok=True)
        openings_file = os.path.join(output_dir, "openings.txt")
    
    save_opening_positions(openings, openings_file)
    
    # Create SF18 player
    sf18_player = SF18Player("SF18", sf18_client, sf18_difficulty)
    
    # Play games for each SF25 model
    for strategy_config in strategy_configs:
        logger.info(f"\nPlaying {len(openings)} games: {strategy_config.name} vs SF18 (difficulty {sf18_difficulty})")
        
        # Load model temporarily for this match
        model_cache = create_temporary_model_cache([strategy_config.model_path], verbose=0)
        
        # Set up output files for this match
        trmph_file, csv_file = setup_strategy_pair_files(output_dir, strategy_config, sf18_player)
        
        # Write TRMPH header
        play_config = create_play_config_for_pair(strategy_config, sf18_player, temperature, seed, command_line, run_desc=run_desc)
        pair_model_paths = [strategy_config.model_path, "SF18"]
        pair_strategy_configs = [strategy_config, sf18_player]
        actual_trmph_file = write_tournament_trmph_header(
            trmph_file, pair_model_paths, len(openings), play_config, BOARD_SIZE, 
            strategy_configs=pair_strategy_configs
        )
        provenance_file = str(sidecar_path_for_trmph(actual_trmph_file))
        if os.path.exists(provenance_file):
            raise RuntimeError(
                "Expected fresh provenance sidecar path for SF18 tournament but file already exists: "
                f"{provenance_file}"
            )
        provenance_records_written = 0
        
        # Find available CSV filename
        actual_csv_file = find_available_csv_filename(csv_file)
        
        # Play games
        with open(provenance_file, "w", encoding="utf-8") as provenance_handle:
            for opening_idx, opening in enumerate(openings):
                if verbose >= 1:
                    print(f"  Game {opening_idx + 1}/{len(openings)}: {opening}")
                
                # Game 1: SF25 (Blue) vs SF18 (Red)
                result_1 = play_sf18_vs_sf25_game(
                    model_cache, strategy_config, sf18_player, opening, temperature, 
                    verbose=verbose, sf25_is_blue=True
                )
                
                # Game 2: SF18 (Blue) vs SF25 (Red)
                result_2 = play_sf18_vs_sf25_game(
                    model_cache, strategy_config, sf18_player, opening, temperature, 
                    verbose=verbose, sf25_is_blue=False
                )
                
                # Record results
                winner_1 = "sf25" if result_1['winner_strategy'] == "SF25" else "sf18"
                winner_2 = "sf25" if result_2['winner_strategy'] == "SF25" else "sf18"
                
                result.record_game(strategy_config.name, winner_1, result_1)
                result.record_game(strategy_config.name, winner_2, result_2)
                
                # Record opening-level results
                result.record_opening_results(strategy_config.name, opening_idx, 
                                            winner_1 == "sf25", winner_2 == "sf25")
                
                # Log TRMPH results
                append_trmph_winner_line(result_1['trmph_str'], result_1['winner'][0], actual_trmph_file)
                append_trmph_winner_line(result_2['trmph_str'], result_2['winner'][0], actual_trmph_file)
                provenance_handle.write(
                    make_move_provenance_record(
                        game_index=provenance_records_written,
                        move_codes=result_1["move_provenance_codes"],
                        policy_targets=result_1["policy_targets_matrix"],
                        policy_target_source_codes=result_1.get("policy_target_source_codes"),
                        policy_target_version=result_1.get("policy_target_version"),
                    ).to_json_line()
                )
                provenance_handle.write("\n")
                provenance_records_written += 1
                provenance_handle.write(
                    make_move_provenance_record(
                        game_index=provenance_records_written,
                        move_codes=result_2["move_provenance_codes"],
                        policy_targets=result_2["policy_targets_matrix"],
                        policy_target_source_codes=result_2.get("policy_target_source_codes"),
                        policy_target_version=result_2.get("policy_target_version"),
                    ).to_json_line()
                )
                provenance_handle.write("\n")
                provenance_records_written += 1
            
        # Report results for this model
        print()
        print(f"Results for {strategy_config.name}:")
        model_stats = result.results[(strategy_config.name, sf18_difficulty)]
        if model_stats['total_games'] > 0:
            win_rate = model_stats['sf25_wins'] / model_stats['total_games']
            print(f"  SF25 Wins: {model_stats['sf25_wins']}")
            print(f"  SF18 Wins: {model_stats['sf18_wins']}")
            print(f"  Total Games: {model_stats['total_games']}")
            print(f"  SF25 Win Rate: {win_rate:.3f}")
        print()
    
    return result, output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a tournament between Snowflake 2025 models and Snowflake 2018 AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare SF25 models against SF18 with default settings
  %(prog)s --model-files=epoch11_mini15.pt.gz,epoch16_mini23.pt.gz --model-dirs=checkpoints/dir1,checkpoints/dir2 --strategies=mcts,mcts --mcts-sims=30,30 --num-openings=100
  
  # Use specific SF18 difficulty and server URL
  %(prog)s --models=best --strategies=mcts,mcts --mcts-sims=100,100 --sf18-difficulty=8 --sf18-server-url=http://localhost:8088 --num-openings=50
  
  # Use custom opening file
  %(prog)s --models=best --strategies=mcts --mcts-sims=30 --opening-file=data/deterministic_openings.txt --sf18-difficulty=9
        """
    )
    
    # SF25 model arguments (reused from run_tournament.py)
    parser.add_argument('--models', type=str,
                       help='Comma-separated list of model registry names (e.g., "best,model2")')
    parser.add_argument('--model-files', type=str,
                       help='Comma-separated list of model file names (e.g., "epoch13_mini31.pt.gz,epoch13_mini27.pt.gz")')
    parser.add_argument('--model-dirs', type=str,
                       help='Comma-separated list of model directories (used with --model-files)')
    parser.add_argument('--strategies', type=str, required=True,
                       help='Comma-separated list of strategies to compare (e.g., "mcts,policy")')
    
    # Opening arguments
    parser.add_argument('--num-openings', type=int, default=DEFAULT_NUM_OPENINGS,
                       help=f'Number of opening positions to generate (default: {DEFAULT_NUM_OPENINGS})')
    parser.add_argument('--opening-length', type=int, default=DEFAULT_OPENING_LENGTH,
                       help=f'Number of moves per opening (default: {DEFAULT_OPENING_LENGTH})')
    parser.add_argument('--opening-file', type=str,
                       help='File containing pre-generated openings (overrides num-openings)')
    parser.add_argument('--cache-file', type=str,
                       help='File to cache generated openings for faster subsequent runs')
    parser.add_argument('--trmph-source', type=str, default=TRMPH_SOURCE_DIR,
                       help=f'Directory containing TRMPH files for opening generation (default: {TRMPH_SOURCE_DIR})')
    
    # SF25 strategy arguments
    parser.add_argument('--mcts-sims', type=str,
                       help=f'Comma-separated MCTS simulation counts (default: {DEFAULT_MCTS_SIMS})')
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
                       help='Comma-separated boolean values to enable dead-cell hard masking in MCTS strategies (default: false)')
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
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                       help=f'Global temperature for move selection (0.0 = deterministic, default: {DEFAULT_TEMPERATURE})')
    parser.add_argument('--temperatures', type=str,
                       help='Comma-separated temperatures for each strategy (e.g., "0.1,1.0,0.5"). Overrides --temperature.')
    
    # SF18 arguments
    parser.add_argument('--sf18-difficulty', type=int, default=DEFAULT_SF18_DIFFICULTY,
                       help=f'Difficulty level for SF18 AI (1-10, default: {DEFAULT_SF18_DIFFICULTY})')
    parser.add_argument('--sf18-server-url', type=str, default=DEFAULT_SF18_SERVER_URL,
                       help=f'URL of SF18 server (default: {DEFAULT_SF18_SERVER_URL})')
    parser.add_argument('--sf18-timeout', type=int, default=30,
                       help='Timeout for SF18 server requests in seconds (default: 30)')
    
    # General arguments
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                       help=f'Random seed for opening selection (different seeds produce different opening sets) (default: auto-generated from time)')
    parser.add_argument('--verbose', type=int, default=DEFAULT_VERBOSE,
                       help=f'Verbosity level (default: {DEFAULT_VERBOSE})')
    parser.add_argument('--run-desc', type=str,
                       help='Description of this tournament run (e.g., "Testing c_scale = 1.5") - will be included in output headers')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get command line early - crash if not available
    try:
        command_line = get_command_line()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Generate seed if none provided
    if args.seed is None:
        args.seed = int(time.time())
        print(f"Auto-generated seed: {args.seed}")
    
    # Set random seed for reproducible opening selection
    set_deterministic_seeds(args.seed)
    
    # Validate arguments
    if args.models and (args.model_files or args.model_dirs):
        print("ERROR: Cannot specify both --models and --model-files/--model-dirs. Use one or the other.")
        sys.exit(1)
    
    if not args.models and not (args.model_files and args.model_dirs):
        print("ERROR: Must specify either --models (registry) or both --model-files and --model-dirs (direct)")
        sys.exit(1)
    
    # Parse strategy names
    strategy_names = [name.strip() for name in args.strategies.split(',')]
    
    # Parse model specifications
    model_paths = parse_model_specifications(args, strategy_names)
    
    # Create strategy configurations
    try:
        strategy_configs = create_strategy_configs_for_tournament(
            args,
            strategy_names,
            model_paths,
            num_games=args.num_openings,
            board_size=13,
            pie_rule=False,
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Check SF18 server connectivity
    sf18_client = SF18Client(args.sf18_server_url, args.sf18_timeout)
    if not sf18_client.is_server_running():
        print(f"ERROR: Cannot connect to SF18 server at {args.sf18_server_url}")
        print("Please make sure the SF18 server is running.")
        print("You can start it with:")
        print("python HttpGameServer.py --valueBuilderPath13=... --policyBuilderPath13=... --twoHeadBuilderPath=... --policyNetworkType=twoHeaded --portNum=8088")
        sys.exit(1)
    
    print(f"✓ SF18 server is running at {args.sf18_server_url}")
    
    # Generate or load opening positions
    if args.opening_file and os.path.exists(args.opening_file):
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
        
        # Generate diverse openings
        target_generation = args.num_openings
        all_openings = generate_diverse_openings(
            trmph_files, 
            opening_length=args.opening_length,
            target_count=target_generation,
            cache_file=args.cache_file
        )
    
    if not all_openings:
        print("ERROR: No opening positions generated")
        sys.exit(1)
    
    # Randomly select the desired number of openings
    print(f"Randomly selecting {args.num_openings} openings from pool of {len(all_openings)}...")
    openings = select_random_openings(all_openings, args.num_openings, seed=args.seed)
    
    # Print configuration
    print("\n" + "="*60)
    print("SF18 vs SF25 TOURNAMENT CONFIGURATION")
    print("="*60)
    print(f"SF25 Models: {len(strategy_configs)}")
    for config in strategy_configs:
        print(f"  - {config.name}")
    print("SF25 Strategy configurations:")
    for config in strategy_configs:
        details = format_strategy_configuration_details(config)
        print(f"  - {config.name}: {details}")
    print(f"SF18 Difficulty: {args.sf18_difficulty}")
    print(f"SF18 Server: {args.sf18_server_url}")
    print(f"Number of openings: {len(openings)}")
    print(f"Opening length: {args.opening_length}")
    print(f"Temperature: {args.temperature}")
    print(f"Early termination threshold (MCTS): {TOURNAMENT_CONFIDENCE_TERMINATION_THRESHOLD}")
    print(f"Random seed: {args.seed}")
    print("="*60)
    print()
    
    # Run the tournament
    result, output_dir = run_sf18_tournament(
        strategy_configs=strategy_configs,
        sf18_client=sf18_client,
        sf18_difficulty=args.sf18_difficulty,
        openings=openings,
        temperature=args.temperature,
        verbose=args.verbose,
        seed=args.seed,
        command_line=command_line,
        run_desc=args.run_desc
    )
    
    # Print results
    result.print_results()
    
    # Save results to file in the tournament output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"sf18_tournament_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(result.get_summary(), f, indent=2)
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
