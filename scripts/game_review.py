#!/usr/bin/env python3
"""
Game Review Script - Detailed Analysis of Hex Games

This script analyzes Hex games to produce detailed review reports that help understand
mistakes, better options, and game flow. It compares actual moves against MCTS
recommendations and value network assessments.

Usage:
    python scripts/game_review.py --game "g7g6j4h6i6h7i7h9f7g8d8d10g9h8e9e10f9f5e5d4j8i10h10i9k9j11l10k12j12k11i11j10c6f3f4g3g4h3h4i3i5k2l2k3l3k4l4k5j2j3e3e4c4c5a6b6a7b7a8b8a9b9a10b10a11b12b11e11" --model best
    python scripts/game_review.py --file games.json --output-dir temp/reviews/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hex_ai.eval.strength_evaluator import GameRecord
from hex_ai.inference.game_engine import HexGameEngine, make_empty_hex_state
from hex_ai.inference.model_wrapper import ModelWrapper
from hex_ai.inference.model_config import get_model_path
from hex_ai.inference.mcts import BaselineMCTS, BaselineMCTSConfig
from hex_ai.enums import Player, Winner
from hex_ai.utils.format_conversion import trmph_to_moves, rowcol_to_trmph, normalize_game_input
from hex_ai.data_processing import parse_trmph_to_gamerecord
from hex_ai.value_utils import red_ref_signed_to_ptm_ref_signed
from hex_ai.config import BOARD_SIZE, DEFAULT_C_PUCT, DEFAULT_MCTS_SIMS, DEFAULT_BATCH_CAP

# Constants for analysis thresholds
MISTAKE_THRESHOLDS = {
    'minor': 0.05,
    'moderate': 0.15,
    'major': 0.30
}

WINNING_THRESHOLD = 0.05
GAME_PHASE_THRESHOLDS = {
    'opening_end': 12,
    'endgame_start_offset': 20
}

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MoveAnalysis:
    """Analysis of a single move in the game."""
    ply: int
    player: str
    move_played: Tuple[int, int]
    move_played_trmph: str
    
    # Value analysis
    value_before_move: float  # Value from perspective of player to move
    value_after_move: float   # Value from perspective of player to move
    value_change: float      # Change in value (positive = good for player)
    
    # MCTS analysis
    mcts_best_move: Tuple[int, int]
    mcts_best_move_trmph: str
    mcts_best_value: float
    mcts_move_played_value: float
    mcts_value_difference: float
    
    # Mistake analysis
    is_mistake: bool
    mistake_severity: str  # "minor", "moderate", "major"
    mistake_reason: str
    
    # Losing move analysis
    is_losing_move: bool
    was_winning_before: bool
    is_winning_after: bool
    alternative_winning_moves: List[Tuple[int, int]]
    
    # Additional context
    game_phase: str  # "opening", "middle", "endgame"
    position_trmph: str  # TRMPH representation of position before move


@dataclass
class GameReview:
    """Complete game review report."""
    game_metadata: Dict[str, Any]
    total_moves: int
    move_analyses: List[MoveAnalysis]
    
    # Summary statistics
    total_mistakes: int
    major_mistakes: int
    losing_moves: int
    mistake_by_player: Dict[str, int]
    mistake_by_phase: Dict[str, int]
    
    # Game flow analysis
    value_trajectory: List[float]
    critical_moments: List[Dict[str, Any]]


class GameReviewer:
    """Main class for analyzing games and producing review reports."""
    
    def __init__(self, model_path: str, mcts_sims: int = 39, c_puct: float = 1.0):
        """
        Initialize the game reviewer.
        
        Args:
            model_path: Path to the model checkpoint
            mcts_sims: Number of MCTS simulations for analysis
            c_puct: MCTS C_PUCT parameter
        """
        self.engine = HexGameEngine()
        self.model_wrapper = ModelWrapper(model_path)
        self.mcts_sims = mcts_sims
        self.c_puct = c_puct
        
        # Create MCTS configuration with Gumbel enabled
        self.mcts_config = BaselineMCTSConfig(
            sims=mcts_sims,
            c_puct=c_puct,
            batch_cap=DEFAULT_BATCH_CAP,
            add_root_noise=False,
            temperature_start=0.01,  # Very low but positive temperature
            temperature_end=0.01,
            enable_gumbel_root_selection=True,
            confidence_termination_threshold=0.95,
            enable_depth_discounting=False
        )
        
        logger.info(f"GameReviewer initialized with model: {model_path}")
        logger.info(f"MCTS config: {mcts_sims} sims, C_PUCT={c_puct}")
    
    def review_game(self, game: GameRecord) -> GameReview:
        """
        Analyze a complete game and produce a detailed review.
        
        Args:
            game: Game record to analyze
            
        Returns:
            Complete game review report
        """
        logger.info(f"Starting game review for {len(game.moves)} moves")
        
        # Reconstruct game states
        states = self._reconstruct_game_states(game)
        
        # Analyze each move
        move_analyses = []
        value_trajectory = []
        
        for i, (state, (row, col, player)) in enumerate(zip(states[:-1], game.moves)):
            logger.debug(f"Analyzing move {i+1}/{len(game.moves)}: {rowcol_to_trmph(row, col)} by {player.value}")
            
            try:
                # Analyze the move
                analysis = self._analyze_move(state, (row, col), player, i)
                move_analyses.append(analysis)
                value_trajectory.append(analysis.value_after_move)
            except Exception as e:
                logger.error(f"Failed to analyze move {i+1}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise
        
        # Create summary statistics
        summary = self._create_summary(move_analyses)
        
        # Identify critical moments
        critical_moments = self._identify_critical_moments(move_analyses)
        
        return GameReview(
            game_metadata=game.metadata or {},
            total_moves=len(game.moves),
            move_analyses=move_analyses,
            total_mistakes=summary['total_mistakes'],
            major_mistakes=summary['major_mistakes'],
            losing_moves=summary['losing_moves'],
            mistake_by_player=summary['mistake_by_player'],
            mistake_by_phase=summary['mistake_by_phase'],
            value_trajectory=value_trajectory,
            critical_moments=critical_moments
        )
    
    def _reconstruct_game_states(self, game: GameRecord) -> List:
        """Reconstruct all game states from the move sequence."""
        states = []
        state = make_empty_hex_state()
        states.append(state)
        
        for row, col, player in game.moves:
            if not state.is_valid_move(row, col):
                raise ValueError(f"Invalid move at ply {len(states)}: ({row}, {col})")
            state = state.make_move(row, col)
            states.append(state)
        
        return states
    
    def _analyze_move(self, state, move_played: Tuple[int, int], player: Player, ply: int) -> MoveAnalysis:
        """Analyze a single move in detail."""
        try:
            # Get value before the move (from player's perspective)
            value_before = self._get_position_value(state)
            
            # Get direct value network assessment of the played move (from player's perspective)
            value_network_assessment_played = self._get_move_value_direct(state, move_played)
            
            # Run Gumbel MCTS to get best move recommendation
            mcts_result = self._run_mcts_analysis(state)
            
            # Get direct value network assessment of the best move (from player's perspective)
            value_network_assessment_best = self._get_move_value_direct(state, mcts_result['best_move'])
            
            # Determine game phase
            game_phase = self._determine_game_phase(ply, len(state.move_history))
            
            # Analyze for mistakes using direct value network comparison
            is_mistake, mistake_severity, mistake_reason = self._analyze_mistake_direct(
                value_network_assessment_played, value_network_assessment_best
            )
            
            # Analyze for losing moves using direct value network assessment
            is_losing_move, was_winning_before, is_winning_after, alternative_winning_moves = self._analyze_losing_move_direct(
                value_network_assessment_played, mcts_result
            )
            
            return MoveAnalysis(
                ply=ply,
                player="blue" if player == Player.BLUE else "red",
                move_played=move_played,
                move_played_trmph=rowcol_to_trmph(move_played[0], move_played[1]),
                value_before_move=value_before,
                value_after_move=value_network_assessment_played,  # Use the move's value, not position after
                value_change=value_network_assessment_played - value_before,  # How much the move improved the position
                mcts_best_move=mcts_result['best_move'],
                mcts_best_move_trmph=rowcol_to_trmph(mcts_result['best_move'][0], mcts_result['best_move'][1]),
                mcts_best_value=value_network_assessment_best,  # Value network assessment of best move
                mcts_move_played_value=value_network_assessment_played,  # Value network assessment of played move
                mcts_value_difference=value_network_assessment_best - value_network_assessment_played,  # How much better the best move is
                is_mistake=is_mistake,
                mistake_severity=mistake_severity,
                mistake_reason=mistake_reason,
                is_losing_move=is_losing_move,
                was_winning_before=was_winning_before,
                is_winning_after=is_winning_after,
                alternative_winning_moves=alternative_winning_moves,
                game_phase=game_phase,
                position_trmph=state.to_trmph()
            )
        except Exception as e:
            logger.error(f"Error analyzing move {ply}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _get_position_value(self, state) -> float:
        """Get the value of a position from the perspective of the player to move."""
        if state.game_over:
            if state.winner == Winner.RED:
                return 1.0
            elif state.winner == Winner.BLUE:
                return -1.0
            else:
                return 0.0
        
        # Use value network
        _, value_signed = self.model_wrapper.predict(state.get_board_tensor())
        actor = state.current_player_enum
        return red_ref_signed_to_ptm_ref_signed(value_signed.item(), actor)
    
    def _get_move_value_direct(self, state, move: Tuple[int, int]) -> float:
        """Get the value network's direct assessment of a move from the player's perspective."""
        new_state = state.make_move(move[0], move[1])
        _, value_signed = self.model_wrapper.predict(new_state.get_board_tensor())
        
        # The value network always gives values from RED's perspective
        # After the move, it's the opponent's turn, so we need to flip the value
        # to get the value from the original player's perspective
        actor = state.current_player_enum
        value_ptm = red_ref_signed_to_ptm_ref_signed(value_signed.item(), actor)
        return value_ptm
    
    def _run_mcts_analysis(self, state) -> Dict[str, Any]:
        """Run Gumbel MCTS analysis to find the best move."""
        try:
            mcts = BaselineMCTS(self.engine, self.model_wrapper, self.mcts_config)
            result = mcts.run(state, verbose=0)
            
            # Get the best move from MCTS
            best_move = result.move
            
            # Get MCTS value estimate from tree data
            mcts_value_estimate = result.tree_data.get('v_ptm_ref_signed_best_child', None)
            
            # Build move_values dict for all legal moves (for finding alternatives)
            # We can get this from the root node
            move_values = {}
            root = result.root_node
            for i, move in enumerate(root.legal_moves):
                # Use Q-values (action values) for each move
                move_values[move] = float(root.Q[i]) if root.N[i] > 0 else float('-inf')
            
            return {
                'best_move': best_move,
                'best_value': move_values[best_move],
                'move_values': move_values,
                'mcts_value_estimate': mcts_value_estimate  # Optional MCTS value
            }
        except Exception as e:
            logger.error(f"Error in MCTS analysis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _determine_game_phase(self, ply: int, total_moves: int) -> str:
        """Determine the game phase based on move number."""
        if ply < GAME_PHASE_THRESHOLDS['opening_end']:
            return "opening"
        elif ply < max(0, total_moves - GAME_PHASE_THRESHOLDS['endgame_start_offset']):
            return "middle"
        else:
            return "endgame"
    
    def _analyze_mistake_direct(self, value_played: float, value_best: float) -> Tuple[bool, str, str]:
        """Analyze if a move is a mistake using direct value network comparison."""
        value_diff = value_best - value_played
        
        if value_diff < MISTAKE_THRESHOLDS['minor']:
            return False, "none", "Move is within acceptable range"
        elif value_diff < MISTAKE_THRESHOLDS['moderate']:
            return True, "minor", f"Small mistake: {value_diff:.3f} value loss"
        elif value_diff < MISTAKE_THRESHOLDS['major']:
            return True, "moderate", f"Moderate mistake: {value_diff:.3f} value loss"
        else:
            return True, "major", f"Major mistake: {value_diff:.3f} value loss"
    
    def _analyze_losing_move_direct(self, played_move_value: float, move_evaluation: Dict) -> Tuple[bool, bool, bool, List[Tuple[int, int]]]:
        """Analyze if a move is a losing move using direct value network assessment."""
        # A move is "losing" if the value network thinks it's negative
        # and there are alternative moves that are positive
        was_winning_before = played_move_value > WINNING_THRESHOLD
        is_winning_after = played_move_value > WINNING_THRESHOLD
        
        # Find alternative winning moves
        alternative_winning_moves = []
        for move, value in move_evaluation['move_values'].items():
            if value > WINNING_THRESHOLD:  # Alternative move that would maintain winning position
                alternative_winning_moves.append(move)
        
        # A move is a "losing move" if:
        # 1. The move itself is assessed as negative by the value network
        # 2. There are alternative moves that are positive
        is_losing_move = (played_move_value < -WINNING_THRESHOLD and len(alternative_winning_moves) > 0)
        
        return is_losing_move, was_winning_before, is_winning_after, alternative_winning_moves
    
    def _create_summary(self, move_analyses: List[MoveAnalysis]) -> Dict[str, Any]:
        """Create summary statistics from move analyses."""
        total_mistakes = sum(1 for analysis in move_analyses if analysis.is_mistake)
        major_mistakes = sum(1 for analysis in move_analyses if analysis.mistake_severity == "major")
        losing_moves = sum(1 for analysis in move_analyses if analysis.is_losing_move)
        
        # Mistakes by player
        mistake_by_player = {"blue": 0, "red": 0}
        for analysis in move_analyses:
            if analysis.is_mistake:
                player_key = "blue" if analysis.player == "blue" else "red"
                mistake_by_player[player_key] += 1
        
        # Mistakes by phase
        mistake_by_phase = {"opening": 0, "middle": 0, "endgame": 0}
        for analysis in move_analyses:
            if analysis.is_mistake:
                mistake_by_phase[analysis.game_phase] += 1
        
        return {
            'total_mistakes': total_mistakes,
            'major_mistakes': major_mistakes,
            'losing_moves': losing_moves,
            'mistake_by_player': mistake_by_player,
            'mistake_by_phase': mistake_by_phase
        }
    
    def _identify_critical_moments(self, move_analyses: List[MoveAnalysis]) -> List[Dict[str, Any]]:
        """Identify critical moments in the game."""
        critical_moments = []
        
        for analysis in move_analyses:
            if analysis.mistake_severity == "major":
                critical_moments.append({
                    'type': 'major_mistake',
                    'ply': analysis.ply,
                    'player': analysis.player,
                    'move_played': analysis.move_played_trmph,
                    'move_suggested': analysis.mcts_best_move_trmph,
                    'description': f"Major mistake by {analysis.player}: {analysis.mistake_reason}",
                    'suggestion': f"Played {analysis.move_played_trmph} (value: {analysis.mcts_move_played_value:+.3f}), should have played {analysis.mcts_best_move_trmph} (value: {analysis.mcts_best_value:+.3f})",
                    'position': analysis.position_trmph
                })
            elif analysis.is_losing_move:
                critical_moments.append({
                    'type': 'losing_move',
                    'ply': analysis.ply,
                    'player': analysis.player,
                    'move_played': analysis.move_played_trmph,
                    'move_suggested': analysis.mcts_best_move_trmph,
                    'description': f"Losing move by {analysis.player}: went from winning to not winning",
                    'suggestion': f"Played {analysis.move_played_trmph} (value: {analysis.mcts_move_played_value:+.3f}), should have played {analysis.mcts_best_move_trmph} (value: {analysis.mcts_best_value:+.3f})",
                    'alternatives': [rowcol_to_trmph(move[0], move[1]) for move in analysis.alternative_winning_moves],
                    'position': analysis.position_trmph
                })
        
        return critical_moments


def format_review_as_html(review: GameReview) -> str:
    """Format the game review as HTML."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hex Game Review</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .summary {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
            .move-analysis {{ margin: 10px 0; padding: 10px; border-left: 3px solid #ccc; }}
            .mistake {{ border-left-color: #ff6b6b; background-color: #ffe0e0; }}
            .major-mistake {{ border-left-color: #d63031; background-color: #ffcccc; }}
            .losing-move {{ border-left-color: #e17055; background-color: #ffe8e0; }}
            .critical {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; margin: 10px 0; }}
            .move-details {{ font-family: monospace; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <h1>Hex Game Review</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Total Moves:</strong> {review.total_moves}</p>
            <p><strong>Total Mistakes:</strong> {review.total_mistakes}</p>
            <p><strong>Major Mistakes:</strong> {review.major_mistakes}</p>
            <p><strong>Losing Moves:</strong> {review.losing_moves}</p>
            <p><strong>Mistakes by Player:</strong> Blue: {review.mistake_by_player['blue']}, Red: {review.mistake_by_player['red']}</p>
            <p><strong>Mistakes by Phase:</strong> Opening: {review.mistake_by_phase['opening']}, Middle: {review.mistake_by_phase['middle']}, Endgame: {review.mistake_by_phase['endgame']}</p>
        </div>
        
        <h2>Critical Moments</h2>
        {''.join([f'<div class="critical"><strong>{moment["type"].replace("_", " ").title()}</strong> at move {moment["ply"]+1}: {moment["description"]}<br>Move played: {moment["move_played"]}<br>Move suggested: {moment["move_suggested"]}<br>Suggestion: {moment["suggestion"]}<br>Position: {moment["position"]}</div>' for moment in review.critical_moments])}
        
        <h2>Move-by-Move Analysis</h2>
        {''.join([format_move_analysis_html(analysis) for analysis in review.move_analyses])}
    </body>
    </html>
    """
    return html


def format_move_analysis_html(analysis: MoveAnalysis) -> str:
    """Format a single move analysis as HTML."""
    css_class = ""
    if analysis.mistake_severity == "major":
        css_class = "major-mistake"
    elif analysis.is_mistake:
        css_class = "mistake"
    elif analysis.is_losing_move:
        css_class = "losing-move"
    
    return f"""
    <div class="move-analysis {css_class}">
        <h3>Move {analysis.ply + 1}: {analysis.move_played_trmph} by {analysis.player.upper()}</h3>
        <div class="move-details">
            <p><strong>Phase:</strong> {analysis.game_phase}</p>
            <p><strong>Position value before move:</strong> {analysis.value_before_move:+.3f}</p>
            <p><strong>Value of played move, {analysis.move_played_trmph}:</strong> {analysis.mcts_move_played_value:+.3f}</p>
            <p><strong>Value of MCTS best move, {analysis.mcts_best_move_trmph}:</strong> {analysis.mcts_best_value:+.3f}</p>
            <p><strong>Move improvement:</strong> {analysis.value_change:+.3f} (how much the move improved the position)</p>
            <p><strong>MCTS advantage:</strong> {analysis.mcts_value_difference:+.3f} (how much better the best move is)</p>
            {f'<p><strong>Mistake:</strong> {analysis.mistake_reason}</p>' if analysis.is_mistake else ''}
            {f'<p><strong>Losing Move:</strong> Went from winning to not winning</p>' if analysis.is_losing_move else ''}
            {f'<p><strong>Alternative Winning Moves:</strong> {", ".join([rowcol_to_trmph(move[0], move[1]) for move in analysis.alternative_winning_moves])}</p>' if analysis.alternative_winning_moves else ''}
        </div>
    </div>
    """


def format_review_as_json(review: GameReview) -> Dict[str, Any]:
    """Format the game review as JSON."""
    return {
        "game_metadata": review.game_metadata,
        "summary": {
            "total_moves": review.total_moves,
            "total_mistakes": review.total_mistakes,
            "major_mistakes": review.major_mistakes,
            "losing_moves": review.losing_moves,
            "mistake_by_player": review.mistake_by_player,
            "mistake_by_phase": review.mistake_by_phase
        },
        "critical_moments": review.critical_moments,
        "move_analyses": [asdict(analysis) for analysis in review.move_analyses],
        "value_trajectory": review.value_trajectory
    }


def create_index_html(reviews: List[GameReview]) -> str:
    """Create an HTML index page for multiple game reviews."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head><title>Hex Game Reviews</title></head>
    <body>
        <h1>Hex Game Reviews</h1>
        <ul>
            {''.join([f'<li><a href="game_{i+1}_review.html">Game {i+1}</a></li>' for i in range(len(reviews))])}
        </ul>
    </body>
    </html>
    """


def save_review_files(reviews: List[GameReview], output_dir: Path) -> None:
    """Save review files (JSON and HTML) to the specified directory."""
    for i, review in enumerate(reviews):
        # Find the next available file number to avoid overwriting
        file_counter = i + 1
        while True:
            json_file = output_dir / f"game_{file_counter}_review.json"
            html_file = output_dir / f"game_{file_counter}_review.html"
            
            # Check if both files exist
            if json_file.exists() or html_file.exists():
                file_counter += 1
            else:
                break
        
        # Save JSON
        with open(json_file, 'w') as f:
            json.dump(format_review_as_json(review), f, indent=2)
        
        # Save HTML
        with open(html_file, 'w') as f:
            f.write(format_review_as_html(review))
        
        logger.info(f"Saved review for game {i+1} to {json_file} and {html_file}")
    
    # Create index page for multiple games
    if len(reviews) > 1:
        index_file = output_dir / "index.html"
        with open(index_file, 'w') as f:
            f.write(create_index_html(reviews))
        logger.info(f"Created index page at {index_file}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Generate detailed game review reports for Hex games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Review a single TRMPH game
  python scripts/game_review.py --game "g7g6j4h6i6h7i7h9f7g8d8d10g9h8e9e10f9f5e5d4j8i10h10i9k9j11l10k12j12k11i11j10c6f3f4g3g4h3h4i3i5k2l2k3l3k4l4k5j2j3e3e4c4c5a6b6a7b7a8b8a9b9a10b10a11b12b11e11"
  
  # Review with custom MCTS parameters
  python scripts/game_review.py --game "..." --mcts-sims 200 --c-puct 1.5
  
  # Review multiple games and save to directory
  python scripts/game_review.py --file games.json --output-dir temp/reviews/
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--game", type=str, help="TRMPH game string to review")
    input_group.add_argument("--file", type=str, help="JSON file containing game(s) to review")
    
    # Model options
    parser.add_argument("--model", type=str, default="best", 
                       help="Model to use for analysis (default: best)")
    
    # Output options
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--html-output", type=str, help="Output HTML file")
    parser.add_argument("--output-dir", type=str, help="Output directory for multiple games")
    
    # Analysis parameters
    parser.add_argument("--mcts-sims", type=int, default=39,
                       help="MCTS simulations for analysis (default: 39)")
    parser.add_argument("--c-puct", type=float, default=1.0,
                       help="MCTS C_PUCT parameter (default: 1.0)")
    
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
        
        # Create reviewer
        reviewer = GameReviewer(model_path, args.mcts_sims, args.c_puct)
        
        # Parse input games
        games = []
        
        if args.game:
            # Single TRMPH game
            # Accept multiple input formats (e.g. LittleGolem "1.c2 2.e6 ...", raw TRMPH moves, etc.)
            # Normalize to bare TRMPH move stream (e.g. "c2e6...") then add the standard preamble.
            trmph_string = normalize_game_input(args.game, board_size=BOARD_SIZE)
            if not trmph_string.startswith("#13,"):
                trmph_string = f"#13,{trmph_string}"
            game = parse_trmph_to_gamerecord(trmph_string)
            games.append(game)
            logger.info(f"Parsed TRMPH game with {len(game.moves)} moves")
            
        elif args.file:
            # JSON file
            with open(args.file, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for i, game_data in enumerate(data):
                    try:
                        # Convert JSON game data to GameRecord
                        board_size = game_data.get("board_size", BOARD_SIZE)
                        moves = game_data.get("moves", [])
                        starting_player = Player(game_data.get("starting_player", Player.BLUE.value))
                        
                        game_moves = []
                        for move in moves:
                            if isinstance(move, dict):
                                row, col, player = move["row"], move["col"], Player(move["player"])
                            elif isinstance(move, list) and len(move) >= 3:
                                row, col, player = move[0], move[1], Player(move[2])
                            else:
                                raise ValueError(f"Invalid move format: {move}")
                            game_moves.append((row, col, player))
                        
                        game = GameRecord(
                            board_size=board_size,
                            moves=game_moves,
                            starting_player=starting_player,
                            metadata=game_data.get("metadata", {})
                        )
                        games.append(game)
                    except Exception as e:
                        logger.warning(f"Failed to parse game {i}: {e}")
            else:
                raise ValueError("JSON file must contain a list of games")
            
            logger.info(f"Parsed {len(games)} games from JSON file")
        
        if not games:
            logger.error("No valid games found in input")
            return 1
        
        # Review games
        reviews = []
        for i, game in enumerate(games):
            logger.info(f"Reviewing game {i+1}/{len(games)}")
            
            try:
                review = reviewer.review_game(game)
                reviews.append(review)
                
                # Print summary
                print(f"\n=== GAME {i+1} REVIEW SUMMARY ===")
                print(f"Total moves: {review.total_moves}")
                print(f"Total mistakes: {review.total_mistakes}")
                print(f"Major mistakes: {review.major_mistakes}")
                print(f"Losing moves: {review.losing_moves}")
                print(f"Mistakes by player: Blue: {review.mistake_by_player['blue']}, Red: {review.mistake_by_player['red']}")
                
            except Exception as e:
                logger.error(f"Failed to review game {i+1}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        # Output results
        if args.output_dir:
            # Save to specified directory
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_review_files(reviews, output_dir)
        
        elif args.output:
            # Single JSON output
            if len(reviews) == 1:
                with open(args.output, 'w') as f:
                    json.dump(format_review_as_json(reviews[0]), f, indent=2)
            else:
                with open(args.output, 'w') as f:
                    json.dump([format_review_as_json(review) for review in reviews], f, indent=2)
            logger.info(f"Results written to {args.output}")
        
        elif args.html_output:
            # Single HTML output
            if len(reviews) == 1:
                with open(args.html_output, 'w') as f:
                    f.write(format_review_as_html(reviews[0]))
            else:
                # Multiple games - create index page
                with open(args.html_output, 'w') as f:
                    f.write(create_index_html(reviews))
            logger.info(f"HTML results written to {args.html_output}")
        
        else:
            # Default behavior: save to analysis/game_reviews directory
            default_output_dir = Path("analysis/game_reviews")
            default_output_dir.mkdir(parents=True, exist_ok=True)
            save_review_files(reviews, default_output_dir)
        
        logger.info(f"Successfully reviewed {len(reviews)} games")
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
