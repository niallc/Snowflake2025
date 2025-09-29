"""
Two-stage tournament orchestration.

This module coordinates the knockout and round-robin stages of the tournament,
handling checkpoint discovery and strategy configuration.
"""

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from .checkpoint_discovery import CheckpointDiscovery, CheckpointInfo
from .knockout_tournament import KnockoutTournament, TournamentParticipant, MatchResult

logger = logging.getLogger(__name__)


class TwoStageTournament:
    """
    Orchestrates a 2-stage tournament: knockout elimination followed by round-robin.
    
    Stage 1: Knockout tournament to identify top k models from a directory of checkpoints
    Stage 2: Round-robin tournament with knockout winners + additional participants
    """
    
    def __init__(self,
                 knockout_dir: Optional[str] = None,
                 knockout_config: Optional[Dict[str, Any]] = None,
                 round_robin_participants: Optional[List[TournamentParticipant]] = None,
                 games_per_match: int = 50,
                 top_k: int = 2,
                 round_robin_games: int = 100):
        """
        Initialize the two-stage tournament.
        
        Args:
            knockout_dir: Directory containing checkpoints for knockout stage
            knockout_config: Configuration for knockout stage MCTS strategy
            round_robin_participants: Additional participants for round-robin stage
            games_per_match: Number of games per knockout match
            top_k: Number of winners from knockout stage to advance
            round_robin_games: Number of games per round-robin match
        """
        self.knockout_dir = knockout_dir
        self.knockout_config = knockout_config or self._get_default_knockout_config()
        self.round_robin_participants = round_robin_participants or []
        self.games_per_match = games_per_match
        self.top_k = top_k
        self.round_robin_games = round_robin_games
        
        # Tournament state
        self.knockout_winners: List[TournamentParticipant] = []
        
        logger.info(f"Initialized two-stage tournament: knockout_dir={knockout_dir}, top_k={top_k}")
    
    def _get_default_knockout_config(self) -> Dict[str, Any]:
        """Get default knockout configuration."""
        return {
            "enable_gumbel_root_selection": True,
            "mcts_sims": 220,
            "temperature": 1.0
        }
    
    def run_tournament(self) -> Dict[str, Any]:
        """
        Run the complete two-stage tournament.
        
        Returns:
            Dictionary containing tournament results and summary
        """
        logger.info("Starting two-stage tournament")
        
        results = {
            "knockout_results": None,
            "round_robin_results": None,
            "final_ranking": None
        }
        
        # Stage 1: Knockout tournament
        if self.knockout_dir:
            logger.info("Running knockout stage")
            knockout_results = self._run_knockout_stage()
            results["knockout_results"] = knockout_results
            self.knockout_winners = knockout_results["winners"]
            logger.info(f"Knockout stage complete. {len(self.knockout_winners)} winners advance")
        
        # Stage 2: Round-robin tournament
        if self.round_robin_participants or self.knockout_winners:
            logger.info("Running round-robin stage")
            round_robin_results = self._run_round_robin_stage()
            results["round_robin_results"] = round_robin_results
            results["final_ranking"] = round_robin_results.get("ranking", [])
            logger.info("Round-robin stage complete")
        
        logger.info("Two-stage tournament complete")
        
        # Generate and save tournament summary
        self._save_tournament_summary(results)
        
        return results
    
    def _run_knockout_stage(self) -> Dict[str, Any]:
        """Run the knockout elimination stage."""
        # Discover checkpoints
        discovery = CheckpointDiscovery(self.knockout_dir)
        checkpoints = discovery.discover_checkpoints()
        
        logger.info(f"Discovered {len(checkpoints)} checkpoints for knockout")
        
        # Create participants from checkpoints
        participants = []
        for checkpoint in checkpoints:
            participant = self._create_checkpoint_participant(checkpoint)
            participants.append(participant)
        
        # Create knockout tournament
        knockout_tournament = KnockoutTournament(
            participants=participants,
            games_per_match=self.games_per_match,
            match_executor=self._create_match_executor()
        )
        
        # Run knockout tournament
        elimination_order = knockout_tournament.run_tournament()
        
        # Get top k winners
        winners = elimination_order[-self.top_k:] if len(elimination_order) >= self.top_k else elimination_order
        
        return {
            "total_participants": len(participants),
            "elimination_order": [p.name for p in elimination_order],
            "winners": winners,
            "tournament_summary": knockout_tournament.get_tournament_summary()
        }
    
    def _run_round_robin_stage(self) -> Dict[str, Any]:
        """Run the round-robin ranking stage."""
        # Combine knockout winners with additional participants
        all_participants = self.knockout_winners + self.round_robin_participants
        
        if len(all_participants) < 2:
            raise ValueError(f"Round-robin stage requires at least 2 participants, got {len(all_participants)}")
        
        logger.info(f"Round-robin stage with {len(all_participants)} participants")
        
        # Convert participants to strategy configs
        strategy_configs = []
        for participant in all_participants:
            strategy_config = participant.to_strategy_config()
            strategy_configs.append(strategy_config)
        
        # Generate opening positions for round-robin stage
        openings = self._generate_round_robin_openings()
        
        # Run the round-robin tournament using existing infrastructure
        from scripts.run_tournament import run_tournament
        
        tournament_result = run_tournament(
            strategy_configs=strategy_configs,
            openings=openings,
            temperature=self.knockout_config.get("temperature", 1.0),
            verbose=1,
            seed=None
        )
        
        # Extract ranking from tournament results
        win_rates = tournament_result.win_rates()
        ranking = sorted(win_rates.items(), key=lambda x: x[1], reverse=True)
        ranking_names = [name for name, _ in ranking]
        
        return {
            "total_participants": len(all_participants),
            "ranking": ranking_names,
            "win_rates": win_rates,
            "elo_ratings": tournament_result.elo_ratings(),
            "total_games": tournament_result.total_games,
            "participants": [{"name": p.name, "metadata": p.metadata} for p in all_participants]
        }
    
    def _create_checkpoint_participant(self, checkpoint: CheckpointInfo) -> TournamentParticipant:
        """Create a tournament participant from a checkpoint."""
        # Create strategy config for this checkpoint
        strategy_config = {
            "strategy": "mcts",
            "model_path": str(checkpoint.file_path),
            **self.knockout_config
        }
        
        # Create metadata
        metadata = {
            "checkpoint_number": checkpoint.checkpoint_number,
            "epoch": checkpoint.epoch,
            "mini": checkpoint.mini,
            "file_path": str(checkpoint.file_path),
            "creation_time": checkpoint.creation_time
        }
        
        return TournamentParticipant(
            name=checkpoint.name,
            strategy_config=strategy_config,
            metadata=metadata
        )
    
    def _create_match_executor(self) -> Callable:
        """Create a match executor function for the knockout tournament."""
        def execute_match(p1: TournamentParticipant, p2: TournamentParticipant, games: int) -> MatchResult:
            """
            Execute a match between two participants using existing game execution infrastructure.
            """
            logger.info(f"Executing match: {p1.name} vs {p2.name} ({games} games)")
            
            # Convert TournamentParticipant to StrategyConfig
            strategy_a = p1.to_strategy_config()
            strategy_b = p2.to_strategy_config()
            
            # Generate opening positions for this match
            openings = self._generate_match_openings(games)
            
            # Get model cache
            from hex_ai.inference.model_cache import get_model_cache
            model_cache = get_model_cache()
            
            # Import play_deterministic_game function
            from .game_execution import play_deterministic_game
            
            # Track wins for each participant
            p1_wins = 0
            p2_wins = 0
            openings_used = []
            
            # Play games
            for i, opening in enumerate(openings):
                # Game 1: p1 (Blue) vs p2 (Red)
                result_1 = play_deterministic_game(
                    model_cache=model_cache,
                    strategy_a=strategy_a,
                    strategy_b=strategy_b,
                    opening=opening,
                    temperature=self.knockout_config.get("temperature", 1.0),
                    verbose=0,
                    strategy_a_is_blue=True
                )
                
                # Game 2: p2 (Blue) vs p1 (Red) 
                result_2 = play_deterministic_game(
                    model_cache=model_cache,
                    strategy_a=strategy_b,
                    strategy_b=strategy_a,
                    opening=opening,
                    temperature=self.knockout_config.get("temperature", 1.0),
                    verbose=0,
                    strategy_a_is_blue=True
                )
                
                # Record results
                openings_used.append(opening.get_trmph_string())
                
                # Count wins
                if result_1['winner_strategy'] == p1.name:
                    p1_wins += 1
                else:
                    p2_wins += 1
                    
                if result_2['winner_strategy'] == p1.name:
                    p1_wins += 1
                else:
                    p2_wins += 1
                
                # Progress reporting
                if i % 10 == 0:
                    print(".", end="", flush=True)
            
            # Log match result
            total_games = games * 2  # Each opening played twice
            winner_name = p1.name if p1_wins > p2_wins else p2.name
            p1_pct = (p1_wins / total_games) * 100
            p2_pct = (p2_wins / total_games) * 100
            print(f" {p1.name}:{p1_wins}/{total_games} ({p1_pct:.1f}%) {p2.name}:{p2_wins}/{total_games} ({p2_pct:.1f}%) -> {winner_name} wins")
            logger.info(f"Match complete: {p1.name} vs {p2.name} -> {winner_name} wins ({p1_wins}-{p2_wins})")
            
            return MatchResult(
                participant1=p1,
                participant2=p2,
                participant1_wins=p1_wins,
                participant2_wins=p2_wins,
                total_games=games * 2,  # Each opening played twice
                openings_used=openings_used
            )
        
        return execute_match
    
    
    def _generate_openings(self, num_games: int, stage_name: str = "tournament"):
        """
        Generate opening positions for tournament stages.
        
        Args:
            num_games: Number of games to generate openings for
            stage_name: Name of the stage for error messages
            
        Returns:
            List of opening positions
            
        Raises:
            ValueError: If insufficient openings can be generated
        """
        from .game_execution import generate_diverse_openings, find_trmph_files
        
        # Use same TRMPH files as existing tournament system
        trmph_files = find_trmph_files("data/sf25/sep28")
        
        if not trmph_files:
            raise ValueError(f"No TRMPH files found in data/sf25/sep28 for {stage_name} stage")
        
        # Generate 1.1x required openings (fail fast if insufficient)
        target_count = int(num_games * 1.1)
        openings = generate_diverse_openings(trmph_files, target_count=target_count)
        
        if len(openings) < num_games:
            raise ValueError(
                f"Insufficient openings generated for {stage_name} stage: "
                f"{len(openings)} < {num_games}. "
                f"Try using a different TRMPH source directory or reducing the number of games."
            )
        
        return openings[:num_games]
    
    def _generate_match_openings(self, num_games: int):
        """Generate opening positions for a match."""
        return self._generate_openings(num_games, "match")
    
    def _generate_round_robin_openings(self):
        """Generate opening positions for the round-robin stage."""
        # Calculate total games needed for round-robin
        # Each pair plays round_robin_games * 2 (A vs B and B vs A)
        num_participants = len(self.knockout_winners) + len(self.round_robin_participants)
        if num_participants < 2:
            return []
        
        # Number of unique pairs
        num_pairs = num_participants * (num_participants - 1) // 2
        total_games = num_pairs * self.round_robin_games * 2
        
        return self._generate_openings(total_games, "round-robin")
    
    def get_tournament_summary(self) -> Dict[str, Any]:
        """Get a summary of the tournament configuration and results."""
        summary = {
            "knockout_dir": self.knockout_dir,
            "knockout_config": self.knockout_config,
            "games_per_match": self.games_per_match,
            "top_k": self.top_k,
            "round_robin_games": self.round_robin_games,
            "round_robin_participants": len(self.round_robin_participants),
            "knockout_winners": [p.name for p in self.knockout_winners]
        }
        
        if self.knockout_dir:
            try:
                discovery = CheckpointDiscovery(self.knockout_dir)
                summary["checkpoint_summary"] = discovery.get_checkpoint_summary()
            except Exception as e:
                summary["checkpoint_discovery_error"] = str(e)
        
        return summary
    
    def _save_tournament_summary(self, results: Dict[str, Any]) -> None:
        """Save tournament summary to JSON file."""
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"data/tournament_play/two_stage_tournament_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create comprehensive summary
        summary = {
            "tournament_type": "two_stage",
            "timestamp": timestamp,
            "configuration": {
                "knockout_dir": self.knockout_dir,
                "knockout_config": self.knockout_config,
                "games_per_match": self.games_per_match,
                "top_k": self.top_k,
                "round_robin_games": self.round_robin_games,
                "round_robin_participants": len(self.round_robin_participants)
            },
            "results": results
        }
        
        # Save to JSON file
        summary_file = os.path.join(output_dir, "tournament_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Tournament summary saved to: {summary_file}")
        
        # Print final ranking
        if results.get("final_ranking"):
            print("\n" + "="*60)
            print("FINAL TOURNAMENT RANKING")
            print("="*60)
            for i, participant in enumerate(results["final_ranking"], 1):
                print(f"{i:2d}. {participant}")
            print("="*60)
