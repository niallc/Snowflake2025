#!/usr/bin/env python3
"""
Run a deterministic strategy tournament using pre-generated opening positions.

This script compares different strategies using the same model and the same
set of opening positions, eliminating randomness as a confounding factor.

The key insight is that by starting from the same opening positions,
we can directly compare how different strategies perform from identical
starting points, making the comparison much more robust.

Examples:

1. Compare strategies using 100 diverse openings:
   PYTHONPATH=. python scripts/run_deterministic_tournament.py \
     --model=current_best \
     --strategies=policy,mcts_122,fixed_tree_13_8 \
     --num-openings=100

2. Use specific opening file:
   PYTHONPATH=. python scripts/run_deterministic_tournament.py \
     --model=current_best \
     --strategies=mcts_100,mcts_200 \
     --opening-file=data/deterministic_openings.txt
"""

import argparse
import os
import sys
import random
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import itertools

from hex_ai.inference.model_config import get_model_path, validate_model_path
from hex_ai.inference.tournament import TournamentResult
from hex_ai.inference.move_selection import list_available_strategies
from hex_ai.inference.game_engine import HexGameState, apply_move_to_state
from hex_ai.inference.move_selection import get_strategy, MoveSelectionConfig
from hex_ai.config import EMPTY_PIECE
from hex_ai.enums import Player, Piece
from hex_ai.utils.format_conversion import rowcol_to_trmph, trmph_move_to_rowcol, strip_trmph_preamble, split_trmph_moves
from hex_ai.utils.tournament_logging import append_trmph_winner_line


class OpeningPosition:
    """Represents an opening position with moves and metadata."""
    
    def __init__(self, moves: List[Tuple[int, int]], source_game: str = "", 
                 opening_length: int = 7):
        self.moves = moves
        self.source_game = source_game
        self.opening_length = opening_length
    
    def get_state(self, board_size: int = 13) -> HexGameState:
        """Create a game state from this opening position."""
        import numpy as np
        
        # Initialize empty board
        board = np.full((board_size, board_size), EMPTY_PIECE, dtype='U1')
        state = HexGameState(board=board, _current_player=Player.BLUE)
        
        # Apply the opening moves
        for row, col in self.moves:
            state = apply_move_to_state(state, row, col)
        
        return state
    
    def get_trmph_string(self, board_size: int = 13) -> str:
        """Get TRMPH representation of the opening moves."""
        trmph_moves = ''.join([rowcol_to_trmph(r, c, board_size) for r, c in self.moves])
        return f"#{board_size},{trmph_moves}"
    
    def __str__(self) -> str:
        return f"Opening({len(self.moves)} moves from {self.source_game})"


class StrategyConfig:
    """Configuration for a single strategy."""
    
    def __init__(self, name: str, strategy_type: str, config: Dict[str, Any]):
        self.name = name
        self.strategy_type = strategy_type
        self.config = config
    
    def get_move_config(self) -> MoveSelectionConfig:
        """Create a MoveSelectionConfig for this strategy."""
        return MoveSelectionConfig(
            temperature=0.0,  # Deterministic
            **self.config
        )
    
    def __str__(self) -> str:
        return f"{self.name}({self.strategy_type})"


def parse_strategy_configs(strategies: List[str], mcts_sims: Optional[List[int]], 
                          search_widths: Optional[List[str]]) -> List[StrategyConfig]:
    """
    Parse strategy configurations from command line arguments.
    
    Args:
        strategies: List of strategy names (e.g., ["policy", "mcts_100", "fixed_tree_13_8"])
        mcts_sims: List of MCTS simulation counts
        search_widths: List of search width strings (e.g., ["13,8", "20,10"])
    
    Returns:
        List of StrategyConfig objects
    """
    configs = []
    
    for strategy_name in strategies:
        # Parse strategy name to determine type and parameters
        if strategy_name == "policy":
            configs.append(StrategyConfig("policy", "policy", {}))
        
        elif strategy_name.startswith("mcts_"):
            # Extract simulation count from name (e.g., "mcts_100" -> 100)
            try:
                sims = int(strategy_name.split("_")[1])
                configs.append(StrategyConfig(
                    strategy_name, "mcts", 
                    {"mcts_sims": sims, "mcts_c_puct": 1.5}
                ))
            except (IndexError, ValueError):
                raise ValueError(f"Invalid MCTS strategy name: {strategy_name}. Expected format: mcts_<sims>")
        
        elif strategy_name.startswith("fixed_tree_"):
            # Extract widths from name (e.g., "fixed_tree_13_8" -> [13, 8])
            try:
                parts = strategy_name.split("_")[2:]
                widths = [int(w) for w in parts]
                configs.append(StrategyConfig(
                    strategy_name, "fixed_tree", 
                    {"search_widths": widths}
                ))
            except (IndexError, ValueError):
                raise ValueError(f"Invalid fixed_tree strategy name: {strategy_name}. Expected format: fixed_tree_<width1>_<width2>_...")
        
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # Override with command line parameters if provided
    if mcts_sims:
        if len(mcts_sims) != len([c for c in configs if c.strategy_type == "mcts"]):
            raise ValueError(f"Number of MCTS simulation counts ({len(mcts_sims)}) must match number of MCTS strategies")
        
        mcts_idx = 0
        for config in configs:
            if config.strategy_type == "mcts":
                config.config["mcts_sims"] = mcts_sims[mcts_idx]
                mcts_idx += 1
    
    if search_widths:
        if len(search_widths) != len([c for c in configs if c.strategy_type == "fixed_tree"]):
            raise ValueError(f"Number of search width sets ({len(search_widths)}) must match number of fixed_tree strategies")
        
        tree_idx = 0
        for config in configs:
            if config.strategy_type == "fixed_tree":
                # Parse widths string (e.g., "13,8" -> [13, 8])
                widths = [int(w.strip()) for w in search_widths[tree_idx].split(",")]
                config.config["search_widths"] = widths
                tree_idx += 1
    
    return configs


def extract_openings_from_trmph_file(file_path: str, opening_length: int = 7, 
                                   max_openings: int = 500) -> List[OpeningPosition]:
    """
    Extract diverse opening positions from a TRMPH file.
    
    Args:
        file_path: Path to TRMPH file
        opening_length: Number of moves to extract for each opening
        max_openings: Maximum number of openings to extract
    
    Returns:
        List of OpeningPosition objects
    """
    openings = []
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if len(openings) >= max_openings:
                break
            
            line = line.strip()
            if not line or not line.startswith('http://www.trmph.com/hex/board#13,'):
                continue
            
            # Parse TRMPH line
            try:
                # Extract moves and winner
                parts = line.split('#13,')
                if len(parts) != 2:
                    continue
                
                moves_winner = parts[1]
                # Find the winner indicator (b or r) at the end
                if moves_winner.endswith(' b'):
                    moves_str = moves_winner[:-2]
                    winner = 'b'
                elif moves_winner.endswith(' r'):
                    moves_str = moves_winner[:-2]
                    winner = 'r'
                else:
                    continue
                
                # Convert TRMPH moves to row,col coordinates
                moves = []
                try:
                    # Use the proper TRMPH parsing functions
                    trmph_moves = split_trmph_moves(moves_str)
                    for trmph_move in trmph_moves:
                        try:
                            row, col = trmph_move_to_rowcol(trmph_move, 13)
                            moves.append((row, col))
                        except ValueError:
                            continue
                except ValueError:
                    continue
                
                # Only use openings with enough moves
                if len(moves) >= opening_length:
                    opening_moves = moves[:opening_length]
                    source_game = f"{os.path.basename(file_path)}:line{line_num}"
                    openings.append(OpeningPosition(opening_moves, source_game, opening_length))
                
            except Exception as e:
                print(f"Warning: Could not parse line {line_num} in {file_path}: {e}")
                continue
    
    return openings


def generate_diverse_openings(trmph_files: List[str], opening_length: int = 7,
                            target_count: int = 500) -> List[OpeningPosition]:
    """
    Generate diverse opening positions from multiple TRMPH files.
    
    Args:
        trmph_files: List of TRMPH file paths
        opening_length: Number of moves per opening
        target_count: Target number of openings to generate
    
    Returns:
        List of diverse OpeningPosition objects
    """
    all_openings = []
    
    # Extract openings from each file
    for file_path in trmph_files:
        if os.path.exists(file_path):
            # Extract more openings per file to ensure we get enough
            max_per_file = max(10, target_count // min(len(trmph_files), 50))
            file_openings = extract_openings_from_trmph_file(
                file_path, opening_length, max_openings=max_per_file
            )
            all_openings.extend(file_openings)
            print(f"Extracted {len(file_openings)} openings from {os.path.basename(file_path)}")
    
    # Shuffle and select diverse subset
    random.shuffle(all_openings)
    
    # Simple diversity: take every Nth opening to spread across different games
    if len(all_openings) > target_count:
        step = len(all_openings) // target_count
        diverse_openings = all_openings[::step][:target_count]
    else:
        diverse_openings = all_openings
    
    print(f"Generated {len(diverse_openings)} diverse openings from {len(trmph_files)} files")
    return diverse_openings


def play_deterministic_game(
    model,
    strategy_a: StrategyConfig,
    strategy_b: StrategyConfig,
    opening: OpeningPosition,
    board_size: int = 13,
    verbose: int = 0
) -> Dict[str, Any]:
    """
    Play a deterministic game from an opening position.
    
    Args:
        model: The model to use for both strategies
        strategy_a: Strategy configuration for player A (Blue)
        strategy_b: Strategy configuration for player B (Red)
        opening: Opening position to start from
        board_size: Board size for the game
        verbose: Verbosity level
    
    Returns:
        Dictionary with game results
    """
    # Start from the opening position
    state = opening.get_state(board_size)
    
    # Create strategy configurations (deterministic)
    config_a = strategy_a.get_move_config()
    config_b = strategy_b.get_move_config()
    
    # Get strategy objects
    strategy_a_obj = get_strategy(strategy_a.strategy_type)
    strategy_b_obj = get_strategy(strategy_b.strategy_type)
    
    # Play the game from the opening position
    move_sequence = list(opening.moves)  # Start with opening moves
    while not state.game_over:
        # Determine which strategy to use based on current player
        current_player = state.current_player_enum
        if current_player == Player.BLUE:
            strategy_obj = strategy_a_obj
            strategy_config = config_a
        else:
            strategy_obj = strategy_b_obj
            strategy_config = config_b
        
        # Select move (deterministic)
        move = strategy_obj.select_move(state, model, strategy_config)
        if move is None:
            raise ValueError(f"Move selection returned None for {strategy_obj.get_name()}")
        
        # Apply move
        move_sequence.append(move)
        state = apply_move_to_state(state, *move)
        
        if verbose >= 2:
            print("-", end="", flush=True)
    
    # Convert to TRMPH format
    trmph_moves = ''.join([rowcol_to_trmph(r, c, board_size) for r, c in move_sequence])
    trmph_str = f"#{board_size},{trmph_moves}"
    
    # Determine winner
    winner_enum = state.winner_enum
    if winner_enum is None:
        raise ValueError("Game is not over or winner missing")
    
    if winner_enum.name == 'BLUE':
        winner_strategy = strategy_a.name
        winner_char = 'b'
    elif winner_enum.name == 'RED':
        winner_strategy = strategy_b.name
        winner_char = 'r'
    else:
        raise ValueError(f"Unknown winner enum: {winner_enum}")
    
    return {
        'winner_strategy': winner_strategy,
        'winner_char': winner_char,
        'trmph_str': trmph_str,
        'move_sequence': move_sequence,
        'num_moves': len(move_sequence),
        'opening': opening
    }


def run_deterministic_tournament(
    model_path: str,
    strategy_configs: List[StrategyConfig],
    openings: List[OpeningPosition],
    verbose: int = 1
) -> TournamentResult:
    """
    Run a deterministic tournament using pre-generated opening positions.
    
    Args:
        model_path: Path to the model checkpoint
        strategy_configs: List of strategy configurations
        openings: List of opening positions to use
        verbose: Verbosity level
    
    Returns:
        TournamentResult with results
    """
    # Create tournament result tracking strategy names
    strategy_names = [config.name for config in strategy_configs]
    result = TournamentResult(strategy_names)
    
    # Preload the model for efficiency
    from hex_ai.inference.model_cache import preload_tournament_models, get_model_cache
    preload_tournament_models([model_path])
    model_cache = get_model_cache()
    model = model_cache.get_simple_model(model_path)
    
    # Create output directory and files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_dir = f"data/tournament_play/deterministic_tournament_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save opening positions
    openings_file = os.path.join(output_dir, "openings.txt")
    with open(openings_file, 'w') as f:
        for i, opening in enumerate(openings):
            f.write(f"Opening {i+1}: {opening.get_trmph_string()}\n")
    
    # Run round-robin between all strategy pairs
    for strategy_a, strategy_b in itertools.combinations(strategy_configs, 2):
        print(f"\nPlaying {len(openings)} games: {strategy_a.name} vs {strategy_b.name}")
        
        # Create output files for this strategy pair
        pair_name = f"{strategy_a.name}_vs_{strategy_b.name}"
        trmph_file = os.path.join(output_dir, f"{pair_name}.trmph")
        csv_file = os.path.join(output_dir, f"{pair_name}.csv")
        
        # Play games from each opening position
        for opening_idx, opening in enumerate(openings):
            if verbose >= 1:
                print(f"  Opening {opening_idx + 1}/{len(openings)}", end="", flush=True)
            
            # Game 1: Strategy A (Blue) vs Strategy B (Red)
            result_1 = play_deterministic_game(
                model, strategy_a, strategy_b, opening, verbose=verbose-1
            )
            
            # Game 2: Strategy B (Blue) vs Strategy A (Red)
            result_2 = play_deterministic_game(
                model, strategy_b, strategy_a, opening, verbose=verbose-1
            )
            
            # Log results to files
            import csv
            from pathlib import Path
            
            # Log TRMPH results
            append_trmph_winner_line(result_1['trmph_str'], result_1['winner_char'], trmph_file)
            append_trmph_winner_line(result_2['trmph_str'], result_2['winner_char'], trmph_file)
            
            # Log CSV results
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
            rows = [
                {
                    "timestamp": timestamp,
                    "strategy_a": strategy_a.name,
                    "strategy_b": strategy_b.name,
                    "opening_idx": opening_idx,
                    "opening_source": opening.source_game,
                    "game": "A_first",
                    "trmph": result_1['trmph_str'],
                    "winner": result_1['winner_char'],
                    "winner_strategy": result_1['winner_strategy'],
                    "num_moves": result_1['num_moves'],
                    "opening_length": opening.opening_length
                },
                {
                    "timestamp": timestamp,
                    "strategy_a": strategy_b.name,
                    "strategy_b": strategy_a.name,
                    "opening_idx": opening_idx,
                    "opening_source": opening.source_game,
                    "game": "B_first",
                    "trmph": result_2['trmph_str'],
                    "winner": result_2['winner_char'],
                    "winner_strategy": result_2['winner_strategy'],
                    "num_moves": result_2['num_moves'],
                    "opening_length": opening.opening_length
                }
            ]
            
            # Write CSV
            csv_path = Path(csv_file)
            write_header = not csv_path.exists()
            headers = list(rows[0].keys())
            
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                if write_header:
                    writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            
            # Record results for tournament tracking
            # Game 1: Strategy A vs Strategy B
            winner_1 = result_1['winner_strategy']
            loser_1 = strategy_b.name if winner_1 == strategy_a.name else strategy_a.name
            result.record_game(winner_1, loser_1)
            
            # Game 2: Strategy B vs Strategy A
            winner_2 = result_2['winner_strategy']
            loser_2 = strategy_a.name if winner_2 == strategy_b.name else strategy_b.name
            result.record_game(winner_2, loser_2)
            
            if verbose >= 1:
                print(f" - {result_1['winner_char']}/{result_2['winner_char']}", end="", flush=True)
        
        if verbose >= 1:
            print()  # New line after games
    
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a deterministic strategy tournament using pre-generated opening positions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare strategies using 100 diverse openings
  %(prog)s --model=current_best --strategies=policy,mcts_122,fixed_tree_13_8 --num-openings=100
  
  # Use specific opening file
  %(prog)s --model=current_best --strategies=mcts_100,mcts_200 --opening-file=data/deterministic_openings.txt
  
  # Compare with custom opening length
  %(prog)s --model=current_best --strategies=policy,mcts_122 --num-openings=200 --opening-length=5
        """
    )
    
    parser.add_argument('--model', type=str, default='current_best',
                       help='Model to use for all strategies (default: current_best)')
    parser.add_argument('--strategies', type=str, required=True,
                       help='Comma-separated list of strategies to compare')
    parser.add_argument('--num-openings', type=int, default=100,
                       help='Number of opening positions to generate (default: 100)')
    parser.add_argument('--opening-length', type=int, default=7,
                       help='Number of moves per opening (default: 7)')
    parser.add_argument('--opening-file', type=str,
                       help='File containing pre-generated openings (overrides num-openings)')
    parser.add_argument('--trmph-source', type=str, default='data/twoNetGames',
                       help='Directory containing TRMPH files for opening generation')
    parser.add_argument('--mcts-sims', type=str,
                       help='Comma-separated MCTS simulation counts (overrides strategy names)')
    parser.add_argument('--search-widths', type=str,
                       help='Semicolon-separated search width sets (e.g., "13,8;20,10")')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for opening generation (default: 42)')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (default: 1)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed for reproducible opening generation
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Validate model path
    try:
        model_path = get_model_path(args.model)
        if not validate_model_path(model_path):
            print(f"ERROR: Model file does not exist: {model_path}")
            sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Parse strategies
    strategy_names = [name.strip() for name in args.strategies.split(',')]
    
    # Parse optional parameters
    mcts_sims = None
    if args.mcts_sims:
        mcts_sims = [int(s.strip()) for s in args.mcts_sims.split(',')]
    
    search_widths = None
    if args.search_widths:
        search_widths = [s.strip() for s in args.search_widths.split(';')]
    
    # Parse strategy configurations
    try:
        strategy_configs = parse_strategy_configs(strategy_names, mcts_sims, search_widths)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Generate or load opening positions
    if args.opening_file and os.path.exists(args.opening_file):
        print(f"Loading openings from: {args.opening_file}")
        # TODO: Implement loading from file
        openings = []  # Placeholder
    else:
        print(f"Generating {args.num_openings} diverse openings...")
        
        # Find TRMPH files
        import glob
        trmph_files = glob.glob(os.path.join(args.trmph_source, "*.trmph"))
        if not trmph_files:
            print(f"ERROR: No TRMPH files found in {args.trmph_source}")
            sys.exit(1)
        
        # Generate diverse openings
        openings = generate_diverse_openings(
            trmph_files, 
            opening_length=args.opening_length,
            target_count=args.num_openings
        )
    
    if not openings:
        print("ERROR: No opening positions generated")
        sys.exit(1)
    
    # Print configuration
    print("\nDeterministic Tournament Configuration:")
    print(f"  Model: {args.model} ({os.path.basename(model_path)})")
    print(f"  Strategies: {[str(c) for c in strategy_configs]}")
    print(f"  Number of openings: {len(openings)}")
    print(f"  Opening length: {args.opening_length} moves")
    print(f"  Random seed: {args.seed}")
    print(f"  Temperature: 0.0 (deterministic)")
    print()
    
    # Run tournament
    result = run_deterministic_tournament(
        model_path=model_path,
        strategy_configs=strategy_configs,
        openings=openings,
        verbose=args.verbose
    )
    
    # Print results
    print("\nDeterministic Tournament Complete!")
    result.print_detailed_analysis()
    
    # Print output location
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_dir = f"data/tournament_play/deterministic_tournament_{timestamp}"
    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
