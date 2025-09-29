"""
Two-stage tournament orchestration.

This module coordinates the knockout and round-robin stages of the tournament,
handling checkpoint discovery and strategy configuration.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from .knockout_tournament import KnockoutTournament, TournamentParticipant, MatchResult
from .checkpoint_discovery import CheckpointDiscovery, CheckpointInfo
from .tournament import Tournament

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
        self.round_robin_tournament: Optional[Tournament] = None
        
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
            logger.warning("Not enough participants for round-robin stage")
            return {"ranking": [p.name for p in all_participants]}
        
        # Create round-robin tournament
        # Note: This will need to be adapted to work with the existing Tournament class
        # For now, we'll create a simple ranking based on participant names
        # TODO: Integrate with existing Tournament class
        
        logger.info(f"Round-robin stage with {len(all_participants)} participants")
        
        # Placeholder implementation
        ranking = [p.name for p in all_participants]
        
        return {
            "total_participants": len(all_participants),
            "ranking": ranking,
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
            Execute a match between two participants.
            
            This is a placeholder implementation that will need to be integrated
            with the existing game execution infrastructure.
            """
            # TODO: Integrate with existing game execution code
            # For now, return a placeholder result
            logger.info(f"Executing match: {p1.name} vs {p2.name} ({games} games)")
            
            # Placeholder: randomly assign winner
            import random
            p1_wins = random.randint(0, games)
            p2_wins = games - p1_wins
            
            return MatchResult(
                participant1=p1,
                participant2=p2,
                participant1_wins=p1_wins,
                participant2_wins=p2_wins,
                total_games=games,
                openings_used=[]  # TODO: Implement opening generation
            )
        
        return execute_match
    
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
