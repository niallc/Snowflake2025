#!/usr/bin/env python3
"""
Batch game strength evaluation script.

This script processes multiple games from TRMPH files and produces statistical
summaries of strength evaluation results across games.

Usage:
    python scripts/batch_evaluate_games.py --file data/sf25/sep5/streaming_selfplay_20250905_224248.trmph --sample-size 100
    python scripts/batch_evaluate_games.py --file data/sf25/sep5/streaming_selfplay_20250905_224248.trmph --sample-size 50 --no-mcts
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import random
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hex_ai.eval.strength_evaluator import (
    StrengthEvaluator, EvaluatorConfig, EvaluatorReport, GameRecord,
    AggregationMethod, GamePhase
)
from hex_ai.inference.game_engine import HexGameEngine
from hex_ai.inference.model_wrapper import ModelWrapper
from hex_ai.inference.model_config import get_model_path
from hex_ai.enums import Player
from hex_ai.utils.format_conversion import trmph_to_moves
from hex_ai.data_utils import extract_games_from_file
from hex_ai.config import BOARD_SIZE

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_trmph_game(trmph_string: str) -> GameRecord:
    """
    Parse a TRMPH string into a GameRecord.
    
    Args:
        trmph_string: TRMPH format game string (may include winner indicator at end)
        
    Returns:
        GameRecord object
    """
    # Strip winner indicator if present (b, r, or other single character at end)
    game_string = trmph_string.strip()
    if len(game_string) > 0 and game_string[-1] in 'br' and game_string[-2] == ' ':
        game_string = game_string[:-2].strip()
    
    # Parse moves
    moves = trmph_to_moves(game_string, BOARD_SIZE)
    
    # Convert to GameRecord format
    game_moves = []
    for i, (row, col) in enumerate(moves):
        player = Player.BLUE if i % 2 == 0 else Player.RED
        game_moves.append((row, col, player))
    
    return GameRecord(
        board_size=BOARD_SIZE,
        moves=game_moves,
        starting_player=Player.BLUE,
        metadata={"source": "trmph", "trmph_string": trmph_string}
    )


def evaluate_single_game(game: GameRecord, model_path: str, config: EvaluatorConfig) -> Tuple[EvaluatorReport, Dict[str, Any], float]:
    """Evaluate a single game."""
    # Initialize components (with reduced logging)
    engine = HexGameEngine()
    
    # Temporarily reduce logging level for ModelWrapper
    import logging
    old_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.WARNING)
    
    try:
        model_wrapper = ModelWrapper(model_path)
        
        # Create evaluator (with reduced logging)
        evaluator = StrengthEvaluator(engine, model_wrapper, config)
        
        # Evaluate game
        report = evaluator.evaluate_game(game)
        
        # Get phase detection stats
        phase_stats = evaluator.get_phase_detection_stats()
        
        return report, phase_stats, evaluator.evaluation_time
    finally:
        # Restore original logging level
        logging.getLogger().setLevel(old_level)


def calculate_statistics(scores: List[float]) -> Dict[str, float]:
    """Calculate statistics for a list of scores."""
    if not scores:
        return {"mean": 0.0, "std": 0.0, "q25": 0.0, "median": 0.0, "q75": 0.0, "uncertainty": 0.0}
    
    scores_array = np.array(scores)
    return {
        "mean": float(np.mean(scores_array)),
        "std": float(np.std(scores_array)),
        "q25": float(np.percentile(scores_array, 25)),
        "median": float(np.median(scores_array)),
        "q75": float(np.percentile(scores_array, 75)),
        "uncertainty": float((np.percentile(scores_array, 75) - np.percentile(scores_array, 25)) / 2)
    }


def create_batch_summary(all_summaries: List[Dict[str, Any]], config: EvaluatorConfig, 
                        source_file: str, sample_size: int) -> Dict[str, Any]:
    """Create statistical summary across all games."""
    
    # Collect all scores by category
    score_categories = {}
    
    for player in [0, 1]:
        for phase in ["opening", "middle", "end"]:
            for metric in ["policy", "value"]:
                key = f"player_{player}_{phase}_{metric}"
                scores = [summary[key] for summary in all_summaries if key in summary]
                score_categories[key] = calculate_statistics(scores)
    
    return {
        "metadata": {
            "source_file": source_file,
            "sample_size": sample_size,
            "total_games_evaluated": len(all_summaries),
            "evaluation_timestamp": datetime.now().isoformat(),
            "config": {
                "use_mcts": config.use_mcts,
                "mcts_sims": config.mcts_sims,
                "mcts_c_puct": config.mcts_c_puct,
                "enable_gumbel_root": config.enable_gumbel_root,
                "opening_plies": config.opening_plies,
                "endgame_value_thresh": config.endgame_value_thresh,
                "endgame_streak": config.endgame_streak,
                "aggregation": config.aggregation.value,
                "bucket_policy_thresholds": config.bucket_policy_thresholds,
                "bucket_value_thresholds": config.bucket_value_thresholds,
                "cache_size": config.cache_size
            }
        },
        "statistics": score_categories
    }


def analyze_phase_detection(all_phase_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze phase detection statistics across all games."""
    if not all_phase_stats:
        return {}
    
    # Aggregate statistics
    total_positions = sum(stats.get("total_positions", 0) for stats in all_phase_stats)
    total_opening = sum(stats.get("opening_positions", 0) for stats in all_phase_stats)
    total_endgame_candidates = sum(stats.get("endgame_candidates", 0) for stats in all_phase_stats)
    total_endgame_streaks = sum(stats.get("endgame_streaks", 0) for stats in all_phase_stats)
    
    # Calculate fractions
    middle_positions = total_positions - total_opening
    endgame_candidate_fraction = total_endgame_candidates / max(1, middle_positions)
    endgame_streak_fraction = total_endgame_streaks / max(1, middle_positions)
    
    # Collect all values for percentile analysis
    all_values = []
    middle_values = []
    for stats in all_phase_stats:
        all_values.extend(stats.get("all_values", []))
        middle_values.extend(stats.get("middle_values", []))
    
    # Calculate percentiles
    percentiles = [0.2, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.8]
    
    def calc_percentiles(values):
        if len(values) == 0:
            return {}
        values_array = np.array(values)
        result = {}
        for p in percentiles:
            if float(p) >= 0.1:  # Skip very small percentiles
                # Handle decimal percentiles properly
                if p < 1:
                    key = f"p{int(p*100)}"
                else:
                    key = f"p{int(p)}"
                result[key] = np.percentile(values_array, p)
        return result
    
    all_percentiles = calc_percentiles(all_values)
    middle_percentiles = calc_percentiles(middle_values)
    
    # Count extreme values
    extreme_high = sum(1 for v in middle_values if v >= 0.9)
    extreme_low = sum(1 for v in middle_values if v <= -0.9)
    extreme_abs = sum(1 for v in middle_values if abs(v) >= 0.9)
    
    return {
        "total_positions": total_positions,
        "opening_positions": total_opening,
        "middle_positions": middle_positions,
        "endgame_candidates": total_endgame_candidates,
        "endgame_streaks": total_endgame_streaks,
        "endgame_candidate_fraction": endgame_candidate_fraction,
        "endgame_streak_fraction": endgame_streak_fraction,
        "endgame_threshold": all_phase_stats[0].get("endgame_threshold", 0.9),
        "endgame_streak_required": all_phase_stats[0].get("endgame_streak_required", 3),
        "value_percentiles_all": all_percentiles,
        "value_percentiles_middle": middle_percentiles,
        "extreme_values": {
            "high_ge_0.9": extreme_high,
            "low_le_-0.9": extreme_low,
            "abs_ge_0.9": extreme_abs,
            "fraction_high": extreme_high / max(1, len(middle_values)),
            "fraction_low": extreme_low / max(1, len(middle_values)),
            "fraction_abs": extreme_abs / max(1, len(middle_values))
        }
    }


def print_batch_summary(batch_summary: Dict[str, Any], phase_analysis: Dict[str, Any]):
    """Print a formatted summary of batch evaluation results."""
    stats = batch_summary["statistics"]
    metadata = batch_summary["metadata"]
    
    print(f"\n=== BATCH STRENGTH EVALUATION SUMMARY ===")
    print(f"Source: {metadata['source_file']}")
    print(f"Games evaluated: {metadata['total_games_evaluated']}")
    print()
    
    # Phase detection analysis
    if phase_analysis:
        print("=== PHASE DETECTION ANALYSIS ===")
        print(f"Total positions analyzed: {phase_analysis['total_positions']}")
        print(f"Opening positions: {phase_analysis['opening_positions']} ({phase_analysis['opening_positions']/max(1, phase_analysis['total_positions']):.1%})")
        print(f"Middle positions: {phase_analysis['middle_positions']} ({phase_analysis['middle_positions']/max(1, phase_analysis['total_positions']):.1%})")
        print(f"Endgame candidates (|value| >= {phase_analysis['endgame_threshold']}): {phase_analysis['endgame_candidates']} ({phase_analysis['endgame_candidate_fraction']:.1%} of middle)")
        print(f"Endgame streaks (streak >= {phase_analysis['endgame_streak_required']}): {phase_analysis['endgame_streaks']} ({phase_analysis['endgame_streak_fraction']:.1%} of middle)")
        print()
        
        # Value distribution analysis
        if 'value_percentiles_middle' in phase_analysis and phase_analysis['value_percentiles_middle']:
            print("=== VALUE DISTRIBUTION (Middle/End Positions) ===")
            percentiles = phase_analysis['value_percentiles_middle']
            print("Percentiles:")
            for p in [0.2, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.8]:
                # Match the key generation logic from calc_percentiles
                if p < 1:
                    key = f"p{int(p*100)}"
                else:
                    key = f"p{int(p)}"
                if key in percentiles:
                    print(f"  {p*100:4.1f}%: {percentiles[key]:6.3f}")
            
            # Extreme values
            if 'extreme_values' in phase_analysis:
                ext = phase_analysis['extreme_values']
                print(f"\nExtreme values (middle positions only):")
                print(f"  Values >= 0.9:  {ext['high_ge_0.9']:4d} ({ext['fraction_high']:.1%})")
                print(f"  Values <= -0.9: {ext['low_le_-0.9']:4d} ({ext['fraction_low']:.1%})")
                print(f"  |Values| >= 0.9: {ext['abs_ge_0.9']:4d} ({ext['fraction_abs']:.1%})")
        print()
    
    print("Average loss per move with uncertainty (lower is better):")
    print("Format: mean (+- uncertainty)")
    print()
    
    for player in [0, 1]:
        print(f"Player {player}:")
        for phase in ["opening", "middle", "end"]:
            policy_key = f"player_{player}_{phase}_policy"
            value_key = f"player_{player}_{phase}_value"
            
            if policy_key in stats and value_key in stats:
                policy_stats = stats[policy_key]
                value_stats = stats[value_key]
                
                print(f"  {phase.capitalize():>8}: Policy={policy_stats['mean']:.4f} (+- {policy_stats['uncertainty']:.3f}), "
                      f"Value={value_stats['mean']:.4f} (+- {value_stats['uncertainty']:.3f})")
        print()


def save_results(batch_summary: Dict[str, Any], all_summaries: List[Dict[str, Any]], 
                all_phase_stats: List[Dict[str, Any]], output_dir: Path, source_file: str):
    """Save detailed and summary results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename based on source file and timestamp
    source_name = Path(source_file).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Analyze phase detection
    phase_analysis = analyze_phase_detection(all_phase_stats)
    
    # Save detailed results
    detailed_file = output_dir / f"{source_name}_detailed_{timestamp}.json"
    with open(detailed_file, 'w') as f:
        json.dump({
            "batch_summary": batch_summary,
            "phase_analysis": phase_analysis,
            "individual_game_summaries": all_summaries,
            "individual_phase_stats": all_phase_stats
        }, f, indent=2)
    
    # Save summary results
    summary_file = output_dir / f"{source_name}_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "batch_summary": batch_summary,
            "phase_analysis": phase_analysis
        }, f, indent=2)
    
    logger.info(f"Results saved to:")
    logger.info(f"  Detailed: {detailed_file}")
    logger.info(f"  Summary: {summary_file}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Batch evaluate Hex game strength across multiple games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate 100 games from a TRMPH file
  python scripts/batch_evaluate_games.py --file data/sf25/sep5/streaming_selfplay_20250905_224248.trmph --sample-size 100
  
  # Fast evaluation with neural networks only
  python scripts/batch_evaluate_games.py --file data/sf25/sep5/streaming_selfplay_20250905_224248.trmph --sample-size 50 --no-mcts
  
  # High-quality evaluation with more MCTS simulations
  python scripts/batch_evaluate_games.py --file data/sf25/sep5/streaming_selfplay_20250905_224248.trmph --sample-size 20 --mcts-sims 500
        """
    )
    
    # Input options
    parser.add_argument("--file", type=str, required=True,
                       help="TRMPH file containing games to evaluate")
    parser.add_argument("--sample-size", type=int, default=100,
                       help="Number of games to sample and evaluate (default: 100)")
    
    # Model options
    parser.add_argument("--model", type=str, default="current_best", 
                       help="Model to use for evaluation (default: current_best)")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="analysis/game_evals",
                       help="Directory to save results (default: analysis/game_evals)")
    
    # Evaluation parameters
    parser.add_argument("--opening-plies", type=int, default=12,
                       help="Number of opening plies (default: 12)")
    parser.add_argument("--endgame-thresh", type=float, default=0.90,
                       help="Endgame value threshold (default: 0.90)")
    parser.add_argument("--endgame-streak", type=int, default=3,
                       help="Endgame streak length (default: 3)")
    parser.add_argument("--use-mcts", action="store_true", default=True,
                       help="Use MCTS-based evaluation (default: True)")
    parser.add_argument("--no-mcts", dest="use_mcts", action="store_false",
                       help="Use neural network only evaluation (faster but lower quality)")
    
    # MCTS parameters
    parser.add_argument("--mcts-sims", type=int, default=200,
                       help="MCTS simulations per move (default: 200)")
    parser.add_argument("--c-puct", type=float, default=3.0,
                       help="MCTS C_PUCT parameter (default: 3.0)")
    parser.add_argument("--batch-cap", type=int, help="MCTS batch size limit")
    parser.add_argument("--enable-gumbel", action="store_true",
                       help="Enable Gumbel AlphaZero root selection")
    
    # Aggregation parameters
    parser.add_argument("--aggregation", type=str, default="mean",
                       choices=["mean", "median", "trimmed_mean"],
                       help="Aggregation method (default: mean)")
    parser.add_argument("--trimmed-fraction", type=float, default=0.1,
                       help="Trimmed mean fraction (default: 0.1)")
    parser.add_argument("--policy-small-thresh", type=float, default=0.10,
                       help="Policy bucket small threshold (default: 0.10)")
    parser.add_argument("--policy-big-thresh", type=float, default=0.30,
                       help="Policy bucket big threshold (default: 0.30)")
    parser.add_argument("--value-small-thresh", type=float, default=0.10,
                       help="Value bucket small threshold (default: 0.10)")
    parser.add_argument("--value-big-thresh", type=float, default=0.30,
                       help="Value bucket big threshold (default: 0.30)")
    
    # Performance parameters
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--batch-nn", action="store_true", default=True,
                       help="Use batched neural network inference (default: True)")
    parser.add_argument("--cache-size", type=int, default=60000,
                       help="Cache size for evaluations (default: 60000)")
    
    # Verbosity
    parser.add_argument("--verbose", "-v", action="count", default=0,
                       help="Increase verbosity")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose >= 1:
        logging.getLogger().setLevel(logging.INFO)
    
    try:
        # Get model path
        model_path = get_model_path(args.model)
        logger.info(f"Using model: {model_path}")
        
        # Create configuration
        config = EvaluatorConfig(
            opening_plies=args.opening_plies,
            endgame_value_thresh=args.endgame_thresh,
            endgame_streak=args.endgame_streak,
            use_mcts=args.use_mcts,
            mcts_sims=args.mcts_sims,
            mcts_c_puct=args.c_puct,
            mcts_batch_cap=args.batch_cap,
            enable_gumbel_root=args.enable_gumbel,
            aggregation=AggregationMethod(args.aggregation),
            trimmed_fraction=args.trimmed_fraction,
            bucket_policy_thresholds=(args.policy_small_thresh, args.policy_big_thresh),
            bucket_value_thresholds=(args.value_small_thresh, args.value_big_thresh),
            rng_seed=args.seed,
            batch_nn=args.batch_nn,
            cache_size=args.cache_size
        )
        
        # Set random seed if provided
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
        
        # Extract games from file
        logger.info(f"Extracting games from {args.file}")
        all_games = extract_games_from_file(Path(args.file))
        logger.info(f"Found {len(all_games)} games in file")
        
        if len(all_games) == 0:
            logger.error("No games found in file")
            return 1
        
        # Sample games
        sample_size = min(args.sample_size, len(all_games))
        if sample_size < len(all_games):
            sampled_games = random.sample(all_games, sample_size)
            logger.info(f"Sampled {sample_size} games from {len(all_games)} available")
        else:
            sampled_games = all_games
            logger.info(f"Using all {len(all_games)} games")
        
        # Print configuration info once
        logger.info(f"Configuration: {'MCTS' if config.use_mcts else 'Neural Network Only'}")
        if config.use_mcts:
            logger.info(f"MCTS sims: {config.mcts_sims}, C_PUCT: {config.mcts_c_puct}")
        logger.info(f"Cache size: {config.cache_size}")
        logger.info(f"Evaluating {len(sampled_games)} games...")
        
        # Evaluate games
        all_summaries = []
        all_phase_stats = []
        for i, trmph_string in enumerate(sampled_games):
            try:
                # Parse game
                game = parse_trmph_game(trmph_string)
                
                # Evaluate game
                report, phase_stats, eval_time = evaluate_single_game(game, model_path, config)
                
                # Create summary
                summary = StrengthEvaluator.create_summary_report(report)
                all_summaries.append(summary)
                all_phase_stats.append(phase_stats)
                
                # Simple progress indicator
                print(f"Game {i+1}/{len(sampled_games)}: {len(game.moves)} moves, {eval_time:.1f}s", end="")
                if i < len(sampled_games) - 1:
                    print(", ", end="", flush=True)
                else:
                    print()  # Final newline
                
            except Exception as e:
                logger.warning(f"Failed to evaluate game {i+1}: {e}")
                print(f"Game {i+1}/{len(sampled_games)}: FAILED", end="")
                if i < len(sampled_games) - 1:
                    print(", ", end="", flush=True)
                else:
                    print()  # Final newline
                continue
        
        if not all_summaries:
            logger.error("No games were successfully evaluated")
            return 1
        
        # Create batch summary
        batch_summary = create_batch_summary(all_summaries, config, args.file, sample_size)
        
        # Analyze phase detection
        phase_analysis = analyze_phase_detection(all_phase_stats)
        
        # Print results
        print_batch_summary(batch_summary, phase_analysis)
        
        # Save results
        output_dir = Path(args.output_dir)
        save_results(batch_summary, all_summaries, all_phase_stats, output_dir, args.file)
        
        logger.info(f"Successfully evaluated {len(all_summaries)} games")
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
