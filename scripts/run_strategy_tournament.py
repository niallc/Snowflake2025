#!/usr/bin/env python3
"""
Run a strategy tournament comparing different move selection strategies.

This script compares different strategies (policy, MCTS, fixed tree search) using
the same model, running round-robin tournaments between all strategy pairs.

Examples:

1. Compare policy vs MCTS vs fixed tree search:
   PYTHONPATH=. python scripts/run_strategy_tournament.py \
     --model=current_best \
     --strategies=policy,mcts,fixed_tree \
     --num-games=50

2. Compare MCTS with different simulation counts:
   PYTHONPATH=. python scripts/run_strategy_tournament.py \
     --model=current_best \
     --strategies=mcts_100,mcts_200 \
     --mcts-sims=100,200 \
     --num-games=50

3. Compare fixed tree search with different widths:
   PYTHONPATH=. python scripts/run_strategy_tournament.py \
     --model=current_best \
     --strategies=fixed_tree_13_8,fixed_tree_20_10 \
     --search-widths="13,8;20,10" \
     --num-games=50

4. Mixed strategy comparison:
   PYTHONPATH=. python scripts/run_strategy_tournament.py \
     --model=current_best \
     --strategies=policy,mcts_122,fixed_tree_13_8 \
     --mcts-sims=122 \
     --search-widths="13,8" \
     --num-games=100
"""

import argparse
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
import itertools

from hex_ai.inference.model_config import get_model_path, validate_model_path
from hex_ai.inference.tournament import (
    TournamentConfig, TournamentPlayConfig, run_round_robin_tournament,
    TournamentResult
)
from hex_ai.inference.move_selection import list_available_strategies


class StrategyConfig:
    """Configuration for a single strategy."""
    
    def __init__(self, name: str, strategy_type: str, config: Dict[str, Any]):
        self.name = name
        self.strategy_type = strategy_type
        self.config = config
    
    def get_play_config(self, base_config: TournamentPlayConfig) -> TournamentPlayConfig:
        """Create a TournamentPlayConfig for this strategy."""
        return TournamentPlayConfig(
            temperature=base_config.temperature,
            random_seed=base_config.random_seed,
            pie_rule=base_config.pie_rule,
            swap_threshold=base_config.swap_threshold,
            strategy=self.strategy_type,
            strategy_config=self.config
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


def play_strategy_vs_strategy_game(
    model,
    strategy_a: StrategyConfig,
    strategy_b: StrategyConfig,
    base_play_config: TournamentPlayConfig,
    board_size: int = 13,
    verbose: int = 0
) -> Dict[str, Any]:
    """
    Play a single game between two strategies using the same model.
    
    Args:
        model: The model to use for both strategies
        strategy_a: Strategy configuration for player A (Blue)
        strategy_b: Strategy configuration for player B (Red)
        base_play_config: Base configuration for the game
        board_size: Board size for the game
        verbose: Verbosity level
    
    Returns:
        Dictionary with game results
    """
    import numpy as np
    from hex_ai.inference.game_engine import HexGameState, apply_move_to_state
    from hex_ai.inference.move_selection import get_strategy, MoveSelectionConfig
    from hex_ai.config import EMPTY_PIECE
    from hex_ai.enums import Player, Piece
    from hex_ai.utils.format_conversion import rowcol_to_trmph
    
    # Initialize game state
    state = HexGameState(
        board=np.full((board_size, board_size), EMPTY_PIECE, dtype='U1'),
        _current_player=Player.BLUE
    )
    
    # Create strategy configurations
    config_a = MoveSelectionConfig(
        temperature=base_play_config.temperature,
        **strategy_a.config
    )
    config_b = MoveSelectionConfig(
        temperature=base_play_config.temperature,
        **strategy_b.config
    )
    
    # Get strategy objects
    strategy_a_obj = get_strategy(strategy_a.strategy_type)
    strategy_b_obj = get_strategy(strategy_b.strategy_type)
    
    # Play the game
    move_sequence = []
    while not state.game_over:
        # Determine which strategy to use based on current player
        current_player = state.current_player_enum
        if current_player == Player.BLUE:
            strategy_obj = strategy_a_obj
            strategy_config = config_a
        else:
            strategy_obj = strategy_b_obj
            strategy_config = config_b
        
        # Select move
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
        'num_moves': len(move_sequence)
    }


def run_strategy_tournament(
    model_path: str,
    strategy_configs: List[StrategyConfig],
    num_games: int,
    base_play_config: TournamentPlayConfig,
    verbose: int = 1
) -> TournamentResult:
    """
    Run a round-robin tournament between strategies using the same model.
    
    Args:
        model_path: Path to the model checkpoint
        strategy_configs: List of strategy configurations
        num_games: Number of games per strategy pair
        base_play_config: Base configuration for all games
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
    output_dir = f"data/tournament_play/strategy_tournament_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run round-robin between all strategy pairs
    for strategy_a, strategy_b in itertools.combinations(strategy_configs, 2):
        print(f"\nPlaying {num_games} games: {strategy_a.name} vs {strategy_b.name}")
        
        # Create output files for this strategy pair
        pair_name = f"{strategy_a.name}_vs_{strategy_b.name}"
        trmph_file = os.path.join(output_dir, f"{pair_name}.trmph")
        csv_file = os.path.join(output_dir, f"{pair_name}.csv")
        
        # Play games with each strategy going first
        for game_idx in range(num_games):
            if verbose >= 1:
                print(f"  Game {game_idx + 1}/{num_games}", end="", flush=True)
            
            # Game 1: Strategy A (Blue) vs Strategy B (Red)
            result_1 = play_strategy_vs_strategy_game(
                model, strategy_a, strategy_b, base_play_config, verbose=verbose-1
            )
            
            # Game 2: Strategy B (Blue) vs Strategy A (Red)
            result_2 = play_strategy_vs_strategy_game(
                model, strategy_b, strategy_a, base_play_config, verbose=verbose-1
            )
            
            # Log results to files
            from hex_ai.utils.tournament_logging import append_trmph_winner_line
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
                    "game": "A_first",
                    "trmph": result_1['trmph_str'],
                    "winner": result_1['winner_char'],
                    "winner_strategy": result_1['winner_strategy'],
                    "num_moves": result_1['num_moves'],
                    "temperature": base_play_config.temperature,
                    "seed": base_play_config.random_seed
                },
                {
                    "timestamp": timestamp,
                    "strategy_a": strategy_b.name,
                    "strategy_b": strategy_a.name,
                    "game": "B_first",
                    "trmph": result_2['trmph_str'],
                    "winner": result_2['winner_char'],
                    "winner_strategy": result_2['winner_strategy'],
                    "num_moves": result_2['num_moves'],
                    "temperature": base_play_config.temperature,
                    "seed": base_play_config.random_seed
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
        description='Run a strategy tournament comparing different move selection strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare policy vs MCTS vs fixed tree search
  %(prog)s --model=current_best --strategies=policy,mcts,fixed_tree_13_8 --num-games=50
  
  # Compare MCTS with different simulation counts
  %(prog)s --model=current_best --strategies=mcts_100,mcts_200 --num-games=50
  
  # Compare fixed tree search with different widths
  %(prog)s --model=current_best --strategies=fixed_tree_13_8,fixed_tree_20_10 --num-games=50
  
  # Mixed strategy comparison
  %(prog)s --model=current_best --strategies=policy,mcts_122,fixed_tree_13_8 --num-games=100
        """
    )
    
    parser.add_argument('--model', type=str, default='current_best',
                       help='Model to use for all strategies (default: current_best)')
    parser.add_argument('--strategies', type=str, required=True,
                       help='Comma-separated list of strategies to compare')
    parser.add_argument('--num-games', type=int, default=50,
                       help='Number of games per strategy pair (default: 50)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for move selection (default: 1.0)')
    parser.add_argument('--mcts-sims', type=str,
                       help='Comma-separated MCTS simulation counts (overrides strategy names)')
    parser.add_argument('--search-widths', type=str,
                       help='Semicolon-separated search width sets (e.g., "13,8;20,10")')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--no-pie-rule', action='store_true',
                       help='Disable pie rule (pie rule is enabled by default)')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (default: 1)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
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
    
    # Create base play configuration
    base_play_config = TournamentPlayConfig(
        temperature=args.temperature,
        random_seed=args.seed,
        pie_rule=not args.no_pie_rule
    )
    
    # Print configuration
    print("Strategy Tournament Configuration:")
    print(f"  Model: {args.model} ({os.path.basename(model_path)})")
    print(f"  Strategies: {[str(c) for c in strategy_configs]}")
    print(f"  Number of games per pair: {args.num_games}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Pie rule: {base_play_config.pie_rule}")
    print(f"  Random seed: {args.seed}")
    print()
    
    # Run tournament
    result = run_strategy_tournament(
        model_path=model_path,
        strategy_configs=strategy_configs,
        num_games=args.num_games,
        base_play_config=base_play_config,
        verbose=args.verbose
    )
    
    # Print results
    print("\nStrategy Tournament Complete!")
    result.print_detailed_analysis()
    
    # Print output location
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_dir = f"data/tournament_play/strategy_tournament_{timestamp}"
    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
