"""
Abstract knockout tournament implementation.

This module provides a strategy-agnostic knockout tournament system that can
execute tournament brackets without knowing about specific strategy types.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TournamentParticipant:
    """Represents a participant in a tournament."""
    name: str
    strategy_config: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_strategy_config(self):
        """
        Convert this participant to a StrategyConfig object.
        
        Returns:
            StrategyConfig object for use in tournament execution
        """
        from hex_ai.inference.strategy_config import StrategyConfig
        
        # Extract strategy type and model path
        strategy_type = self.strategy_config.get("strategy", "mcts")
        model_path = self.strategy_config.get("model_path")
        
        if not model_path:
            raise ValueError(
                f"Participant {self.name} missing model_path in strategy_config. "
                f"Each participant must have a 'model_path' field specifying the checkpoint file path."
            )
        
        # Create clean config with only MoveSelectionConfig parameters
        clean_config = {}
        for key, value in self.strategy_config.items():
            if key not in ["strategy", "model_path"]:
                clean_config[key] = value
        
        return StrategyConfig(
            name=self.name,
            strategy_type=strategy_type,
            config=clean_config,
            model_path=model_path,
            temperature=self.strategy_config.get("temperature")
        )


@dataclass
class MatchResult:
    """Result of a single match between two participants."""
    participant1: TournamentParticipant
    participant2: TournamentParticipant
    participant1_wins: int
    participant2_wins: int
    total_games: int
    openings_used: List[str]
    
    @property
    def winner(self) -> TournamentParticipant:
        """Returns the winning participant."""
        if self.participant1_wins > self.participant2_wins:
            return self.participant1
        elif self.participant2_wins > self.participant1_wins:
            return self.participant2
        else:
            # Tie: use first participant as winner (deterministic tiebreaker)
            return self.participant1
    
    @property
    def is_tie(self) -> bool:
        """Returns True if the match ended in a tie."""
        return self.participant1_wins == self.participant2_wins


class KnockoutTournament:
    """
    Abstract knockout tournament that executes tournament brackets.
    
    This class is strategy-agnostic and only knows how to execute matches
    between participants. It doesn't know about model checkpoints or
    specific strategy types.
    """
    
    def __init__(self, 
                 participants: List[TournamentParticipant],
                 games_per_match: int = 50,
                 semifinal_games_multiplier: int = 2,
                 final_games_multiplier: int = 4,
                 match_executor: Optional[Callable] = None,
                 top_k: Optional[int] = None):
        """
        Initialize the knockout tournament.
        
        Args:
            participants: List of tournament participants
            games_per_match: Number of games per match in early rounds
            semifinal_games_multiplier: Multiplier for semifinal games
            final_games_multiplier: Multiplier for final games
            match_executor: Function to execute matches between participants
            top_k: Optional number of top participants to advance (stops early when <= k remain)
        """
        if len(participants) < 2:
            raise ValueError(
                f"Tournament requires at least 2 participants, got {len(participants)}. "
                f"Please provide at least 2 participants to run a tournament."
            )
        
        self.participants = participants
        self.games_per_match = games_per_match
        self.semifinal_games_multiplier = semifinal_games_multiplier
        self.final_games_multiplier = final_games_multiplier
        self.match_executor = match_executor
        self.top_k = top_k
        
        # Tournament state
        self.current_round = 0
        self.active_participants = participants.copy()
        self.match_results: List[MatchResult] = []
        self.eliminated_participants: List[TournamentParticipant] = []
        
        # Track bracket positions to maintain diversity in later rounds
        # Each participant gets a bracket_id that represents their "side" of the tournament
        self.bracket_positions = {p.name: i for i, p in enumerate(participants)}
        
        logger.info(f"Initialized knockout tournament with {len(participants)} participants")
    
    def run_tournament(self) -> List[TournamentParticipant]:
        """
        Run the complete knockout tournament.
        
        Returns:
            List of participants in order of elimination (last is winner)
        """
        logger.info("Starting knockout tournament")
        
        while not self._should_stop_tournament():
            self.current_round += 1
            logger.info(f"Starting round {self.current_round} with {len(self.active_participants)} participants")
            
            # Determine games per match for this round
            games_this_round = self._get_games_for_round()
            
            # Execute matches for this round
            round_results = self._execute_round(games_this_round)
            
            # Update tournament state
            self._update_tournament_state(round_results)
            
            # Print round summary
            advancing_names = [p.name for p in self.active_participants]
            if self._should_stop_tournament():
                # Tournament complete
                print(f"\nEnd of Round {self.current_round}: Tournament complete! {len(self.active_participants)} participants advance to round-robin: {', '.join(advancing_names)}")
            else:
                # More rounds to come
                print(f"\nEnd of Round {self.current_round}: {len(self.active_participants)} participants advance to next round: {', '.join(advancing_names)}")
            
            logger.info(f"Round {self.current_round} complete. {len(self.active_participants)} participants remain")
        
        # Tournament complete - all remaining participants are winners
        winners = self.active_participants
        logger.info(f"Tournament complete! {len(winners)} participants advance: {[p.name for p in winners]}")
        
        elimination_order = self.eliminated_participants + winners
        return elimination_order
    
    def _should_stop_tournament(self) -> bool:
        """Determine if the tournament should stop based on remaining participants and top_k setting."""
        if self.top_k is not None and self.top_k > 1:
            # Stop when we have <= k participants remaining
            return len(self.active_participants) <= self.top_k
        else:
            # Default behavior: continue until 1 winner remains
            return len(self.active_participants) <= 1
    
    def _get_games_for_round(self) -> int:
        """Determine number of games for current round."""
        remaining = len(self.active_participants)
        
        if remaining == 4:  # Semifinals
            return self.games_per_match * self.semifinal_games_multiplier
        elif remaining == 2:  # Finals
            return self.games_per_match * self.final_games_multiplier
        else:
            return self.games_per_match
    
    def _execute_round(self, games_per_match: int) -> List[MatchResult]:
        """Execute all matches in the current round."""
        round_results = []
        
        # Pair participants for matches
        matches = self._pair_participants()
        
        for i, (p1, p2) in enumerate(matches):
            logger.info(f"Executing match {i+1}/{len(matches)}: {p1.name} vs {p2.name}")
            
            if self.match_executor:
                result = self.match_executor(p1, p2, games_per_match)
            else:
                # Default behavior: raise error if no executor provided
                raise ValueError(
                    "No match executor provided. "
                    "A match executor function is required to run matches between participants. "
                    "Please provide a match_executor function when creating the tournament."
                )
            
            round_results.append(result)
            self.match_results.append(result)
        
        return round_results
    
    def _pair_participants(self) -> List[tuple[TournamentParticipant, TournamentParticipant]]:
        """Pair participants for matches in current round using bracket-aware seeding."""
        if len(self.active_participants) < 2:
            return []
        
        # Handle odd number of participants by giving first participant a bye
        bye_participant = None
        if len(self.active_participants) % 2 != 0:
            # Give first participant a bye (advance automatically)
            bye_participant = self.active_participants[0]
            logger.info(f"Participant {bye_participant.name} gets a bye")
        
        participants_to_pair = self.active_participants.copy()
        if bye_participant:
            participants_to_pair = participants_to_pair[1:]  # Remove bye participant
        
        # For first round, use snake draft to maximize distance
        if self.current_round == 1:
            # Snake draft pairing: pair participants with maximum distance
            # For 6 participants: (1,6), (2,5), (3,4)
            # For 4 participants: (1,4), (2,3)
            # For 8 participants: (1,8), (2,7), (3,6), (4,5)
            pairs = []
            n = len(participants_to_pair)
            
            for i in range(n // 2):
                # Pair first with last, second with second-to-last, etc.
                pairs.append((participants_to_pair[i], participants_to_pair[n - 1 - i]))
            
            return pairs
        
        # For later rounds, use bracket-aware pairing to maintain diversity
        # Sort participants by their original bracket position to maintain structure
        participants_to_pair.sort(key=lambda p: self.bracket_positions[p.name])
        
        # Use snake draft again, but now on the sorted list
        # This ensures winners from different "sides" of the bracket don't meet immediately
        pairs = []
        n = len(participants_to_pair)
        
        for i in range(n // 2):
            # Pair first with last, second with second-to-last, etc.
            pairs.append((participants_to_pair[i], participants_to_pair[n - 1 - i]))
        
        return pairs
    
    def _update_tournament_state(self, round_results: List[MatchResult]):
        """Update tournament state after a round."""
        new_active = []
        
        # Handle match results
        for result in round_results:
            if result.is_tie:
                # In case of tie, use deterministic tiebreaker: first participant wins
                # This preserves tournament structure and avoids eliminating both participants
                winner = result.participant1
                loser = result.participant2
                logger.info(f"Tie in match {result.participant1.name} vs {result.participant2.name} - {winner.name} advances (tiebreaker)")
                
                new_active.append(winner)
                self.eliminated_participants.append(loser)
            else:
                winner = result.winner
                loser = result.participant2 if winner == result.participant1 else result.participant1
                
                new_active.append(winner)
                self.eliminated_participants.append(loser)
        
        # Handle bye participant (if any)
        if len(self.active_participants) % 2 != 0:
            # There was a bye participant in this round
            bye_participant = self.active_participants[0]
            new_active.append(bye_participant)
            logger.info(f"Bye participant {bye_participant.name} advances to next round")
        
        self.active_participants = new_active
    
    def get_tournament_summary(self) -> Dict[str, Any]:
        """Get a summary of the tournament results."""
        return {
            "total_participants": len(self.participants),
            "total_rounds": self.current_round,
            "total_matches": len(self.match_results),
            "elimination_order": [p.name for p in self.eliminated_participants],
            "winner": self.active_participants[0].name if self.active_participants else None,
            "match_results": [
                {
                    "participant1": r.participant1.name,
                    "participant2": r.participant2.name,
                    "score": f"{r.participant1_wins}-{r.participant2_wins}",
                    "winner": r.winner.name
                }
                for r in self.match_results
            ]
        }
