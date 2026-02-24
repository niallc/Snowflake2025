"""
Two-stage tournament orchestration.

This module coordinates the knockout and round-robin stages of the tournament,
handling checkpoint discovery and strategy configuration.
"""

import json
import itertools
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Tuple

from .checkpoint_discovery import CheckpointDiscovery, CheckpointInfo
from .knockout_tournament import KnockoutTournament, TournamentParticipant, MatchResult
from .game_execution import (
    DeterministicTournamentResult,
    OpeningPosition,
    find_trmph_files,
    generate_diverse_openings,
)

logger = logging.getLogger(__name__)


class TwoStageTournament:
    """
    Orchestrates a 2-stage tournament: knockout elimination followed by round-robin.
    
    Stage 1: Knockout tournament to identify top k models from a directory of checkpoints
    Stage 2: Round-robin tournament with knockout winners + additional participants
    """
    
    def __init__(self,
                 knockout_dir: Optional[str] = None,
                 knockout_participants: Optional[List[TournamentParticipant]] = None,
                 knockout_config: Optional[Dict[str, Any]] = None,
                 round_robin_participants: Optional[List[TournamentParticipant]] = None,
                 games_per_match: int = 50,
                 top_k: int = 2,
                 round_robin_games: int = 100,
                 epoch_range: Optional[Tuple[int, int]] = None,
                 mini_epoch_range: Optional[Tuple[int, int]] = None,
                 command_line: Optional[str] = None,
                 run_desc: Optional[str] = None,
                 seed: Optional[int] = None,
                 trmph_source: str = "data/sf25/sep28",
                 mps_empty_cache_per_pair: bool = False):
        """
        Initialize the two-stage tournament.
        
        Args:
            knockout_dir: Directory containing checkpoints for knockout stage (mutually exclusive with knockout_participants)
            knockout_participants: List of participants for knockout stage (mutually exclusive with knockout_dir)
            knockout_config: Configuration for knockout stage MCTS strategy
            round_robin_participants: Additional participants for round-robin stage
            games_per_match: Number of games per knockout match
            top_k: Number of winners from knockout stage to advance
            round_robin_games: Number of games per round-robin match
            epoch_range: Optional tuple of (start_epoch, end_epoch) to filter knockout checkpoints (only used with knockout_dir)
            mini_epoch_range: Optional tuple of (start_mini_epoch, end_mini_epoch) to filter knockout checkpoints (only used with knockout_dir)
            command_line: Command line that was used to run the tournament
            run_desc: Optional description of this tournament run (e.g., "Testing c_scale = 1.5")
            seed: Optional base seed for deterministic worker execution
            trmph_source: Directory containing TRMPH files for opening generation
        
        Raises:
            ValueError: If both knockout_dir and knockout_participants are provided, or if neither is provided
        """
        # Validate that exactly one of knockout_dir or knockout_participants is provided
        if knockout_dir is not None and knockout_participants is not None:
            raise ValueError(
                "Cannot specify both knockout_dir and knockout_participants. "
                "Use knockout_dir to discover checkpoints from a directory, "
                "or knockout_participants to provide participants directly."
            )
        # if knockout_dir is None and knockout_participants is None:
        #     raise ValueError(
        #         "Must specify either knockout_dir or knockout_participants. "
        #         "Use knockout_dir to discover checkpoints from a directory, "
        #         "or knockout_participants to provide participants directly."
        #     )
        
        self.knockout_dir = knockout_dir
        self.knockout_participants = knockout_participants
        self.knockout_config = knockout_config or self._get_default_knockout_config()
        self.round_robin_participants = round_robin_participants or []
        self.games_per_match = games_per_match
        self.top_k = top_k
        self.round_robin_games = round_robin_games
        self.epoch_range = epoch_range
        self.mini_epoch_range = mini_epoch_range
        self.command_line = command_line
        self.run_desc = run_desc
        self.seed = seed
        self.trmph_source = trmph_source
        self.mps_empty_cache_per_pair = mps_empty_cache_per_pair
        
        # Tournament state
        self.knockout_winners: List[TournamentParticipant] = []
        self.output_dir: Optional[str] = None
        self._worker_launch_count = 0
        
        logger.info(f"Initialized two-stage tournament: knockout_dir={knockout_dir}, knockout_participants={len(knockout_participants) if knockout_participants else 0}, top_k={top_k}")
    
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
        
        # Create output directory early for game streaming
        if self.output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = f"data/tournament_play/two_stage_tournament_{timestamp}"
            os.makedirs(self.output_dir, exist_ok=True)
        
        results = {
            "knockout_results": None,
            "round_robin_results": None,
            "final_ranking": None
        }
        
        # Stage 1: Knockout tournament
        if self.knockout_dir or self.knockout_participants:
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
        # Use provided participants directly, or discover from directory
        if self.knockout_participants is not None:
            # Use provided participants directly
            participants = self.knockout_participants
            logger.info(f"Using {len(participants)} provided participants for knockout")
            
            # Validate that we have enough participants for a tournament
            if len(participants) <= 1:
                raise ValueError(
                    f"Tournament requires at least 2 participants, got {len(participants)}. "
                    f"Please provide at least 2 participants to run a tournament."
                )
        else:
            # Discover checkpoints from directory (existing logic)
            discovery = CheckpointDiscovery(self.knockout_dir)
            
            # Filter by epoch and/or mini epoch ranges if specified
            if self.epoch_range or self.mini_epoch_range:
                checkpoints = discovery.get_checkpoints_by_combined_range(
                    epoch_range=self.epoch_range,
                    mini_epoch_range=self.mini_epoch_range
                )
                
                # Create descriptive filter message
                filter_parts = []
                if self.epoch_range:
                    start_epoch, end_epoch = self.epoch_range
                    filter_parts.append(f"epochs {start_epoch}-{end_epoch-1}")
                if self.mini_epoch_range:
                    start_mini, end_mini = self.mini_epoch_range
                    filter_parts.append(f"mini epochs {start_mini}-{end_mini-1}")
                
                filter_desc = " and ".join(filter_parts)
                logger.info(f"Filtered to {len(checkpoints)} checkpoints for {filter_desc}")
                
                # Validate that we have enough checkpoints for a tournament
                if len(checkpoints) <= 1:
                    self._raise_insufficient_checkpoints_error(discovery, self.epoch_range, self.mini_epoch_range, len(checkpoints))
            else:
                checkpoints = discovery.discover_checkpoints()
                logger.info(f"Discovered {len(checkpoints)} checkpoints for knockout")
                
                # Validate that we have enough checkpoints for a tournament
                if len(checkpoints) <= 1:
                    self._raise_insufficient_checkpoints_error(discovery, None, None, len(checkpoints))
            
            # Create participants from checkpoints
            participants = []
            for checkpoint in checkpoints:
                participant = self._create_checkpoint_participant(checkpoint)
                participants.append(participant)
        
        # Create knockout tournament
        knockout_tournament = KnockoutTournament(
            participants=participants,
            games_per_match=self.games_per_match,
            match_executor=self._create_match_executor(),
            top_k=self.top_k
        )
        
        # Run knockout tournament
        elimination_order = knockout_tournament.run_tournament()
        
        # Get top k winners (last k participants in elimination order)
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
        
        num_participants = len(all_participants)
        logger.info(f"Round-robin stage with {num_participants} participants")
        
        # Convert participants to strategy configs
        strategy_configs = []
        for participant in all_participants:
            strategy_config = participant.to_strategy_config()
            strategy_configs.append(strategy_config)
        
        # Generate opening positions for round-robin stage
        openings = self._generate_round_robin_openings()
        openings_per_pair = len(openings)
        games_per_pair = openings_per_pair * 2  # Each opening is played twice with swapped colors.
        num_pairs = (num_participants * (num_participants - 1)) // 2
        total_games_planned = num_pairs * games_per_pair
        logger.info(
            "Round-robin schedule: %d pairs, %d openings per pair, %d games per pair, %d total games",
            num_pairs,
            openings_per_pair,
            games_per_pair,
            total_games_planned,
        )
        
        # Run each pair in a fresh process to bound long-run memory growth.
        serialized_openings = self._serialize_openings(openings)
        unique_strategy_names = [config.name for config in strategy_configs]
        tournament_result = DeterministicTournamentResult(unique_strategy_names)

        for strategy_a, strategy_b in itertools.combinations(strategy_configs, 2):
            pair_payload = {
                "strategy_a": self._serialize_strategy_config(strategy_a),
                "strategy_b": self._serialize_strategy_config(strategy_b),
                "openings": serialized_openings,
                "temperature": self.knockout_config.get("temperature", 1.0),
                "seed": self._next_worker_seed(),
                "output_dir": self.output_dir,
                "command_line": self.command_line,
                "run_desc": self.run_desc,
                "mps_empty_cache_per_pair": self.mps_empty_cache_per_pair,
            }
            pair_result = self._run_tournament_worker("round_robin_pair", pair_payload)
            self._accumulate_round_robin_pair_result(tournament_result, pair_result)
        
        # Extract ranking from tournament results
        win_rates = tournament_result.win_rates()
        ranking = sorted(win_rates.items(), key=lambda x: x[1], reverse=True)
        ranking_names = [name for name, _ in ranking]
        
        # Calculate individual game counts for each participant
        participant_games = {}
        for name in ranking_names:
            # Count total games for this participant from actual tournament results
            total_games_for_participant = sum(tournament_result.results[name][op]['games'] for op in tournament_result.results[name])
            participant_games[name] = total_games_for_participant
        
        return {
            "total_participants": len(all_participants),
            "ranking": ranking_names,
            "win_rates": win_rates,
            "elo_ratings": tournament_result.elo_ratings(),
            "total_games": tournament_result.total_games,
            "participant_games": participant_games,
            "participants": [{"name": p.name, "metadata": p.metadata} for p in all_participants]
        }

    def _run_tournament_worker(self, mode: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one tournament unit in a fresh subprocess and return its JSON summary."""
        if self.output_dir is None:
            raise RuntimeError("Output directory must be initialized before launching worker processes.")

        with tempfile.TemporaryDirectory(prefix="tournament_worker_", dir=self.output_dir) as temp_dir:
            request_path = os.path.join(temp_dir, "request.json")
            response_path = os.path.join(temp_dir, "response.json")

            with open(request_path, "w") as request_file:
                json.dump(payload, request_file, default=str)

            command = [
                sys.executable,
                "-m",
                "hex_ai.inference.tournament_worker",
                "--mode",
                mode,
                "--input-json",
                request_path,
                "--output-json",
                response_path,
            ]

            logger.info("Starting worker for %s", mode)
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"Tournament worker failed (mode={mode}, exit_code={completed.returncode}).")

            if not os.path.exists(response_path):
                raise RuntimeError(f"Tournament worker completed without writing output: {response_path}")

            with open(response_path, "r") as response_file:
                return json.load(response_file)

    def _next_worker_seed(self) -> Optional[int]:
        """Return the next deterministic worker seed, or None when no base seed is configured."""
        if self.seed is None:
            return None

        next_seed = int(self.seed) + self._worker_launch_count
        self._worker_launch_count += 1
        return next_seed

    @staticmethod
    def _serialize_participant(participant: TournamentParticipant) -> Dict[str, Any]:
        """Serialize a tournament participant for worker process input."""
        return {
            "name": participant.name,
            "strategy_config": participant.strategy_config,
            "metadata": participant.metadata,
        }

    @staticmethod
    def _serialize_strategy_config(strategy_config) -> Dict[str, Any]:
        """Serialize a StrategyConfig for worker process input."""
        return {
            "name": strategy_config.name,
            "strategy_type": strategy_config.strategy_type,
            "config": strategy_config.config,
            "model_path": strategy_config.model_path,
            "original_name": strategy_config.original_name,
            "temperature": strategy_config.temperature,
        }

    @staticmethod
    def _serialize_openings(openings: List[OpeningPosition]) -> List[Dict[str, Any]]:
        """Serialize openings to JSON-friendly dictionaries."""
        serialized_openings = []
        for opening in openings:
            serialized_openings.append(
                {
                    "moves": [[int(row), int(col)] for row, col in opening.moves],
                    "source_game": opening.source_game,
                    "opening_length": opening.opening_length,
                }
            )
        return serialized_openings

    @staticmethod
    def _accumulate_round_robin_pair_result(
        tournament_result: DeterministicTournamentResult,
        pair_result: Dict[str, Any],
    ) -> None:
        """Merge one pair summary from a worker into the in-memory tournament aggregate."""
        strategy_a = pair_result["strategy_a"]
        strategy_b = pair_result["strategy_b"]
        a_vs_b = pair_result["a_vs_b"]
        b_vs_a = pair_result["b_vs_a"]

        tournament_result.results[strategy_a][strategy_b]["wins"] += int(a_vs_b["wins"])
        tournament_result.results[strategy_a][strategy_b]["losses"] += int(a_vs_b["losses"])
        tournament_result.results[strategy_a][strategy_b]["games"] += int(a_vs_b["games"])

        tournament_result.results[strategy_b][strategy_a]["wins"] += int(b_vs_a["wins"])
        tournament_result.results[strategy_b][strategy_a]["losses"] += int(b_vs_a["losses"])
        tournament_result.results[strategy_b][strategy_a]["games"] += int(b_vs_a["games"])

        tournament_result.total_games += int(pair_result["total_games"])
        for strategy_name, timing in pair_result.get("strategy_timings", {}).items():
            tournament_result.strategy_timings[strategy_name] += float(timing)
        for strategy_name, move_count in pair_result.get("strategy_move_counts", {}).items():
            tournament_result.strategy_move_counts[strategy_name] += int(move_count)
    
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
            """Execute one knockout match in a fresh subprocess."""
            openings = self._generate_match_openings(games)
            match_payload = {
                "participant1": self._serialize_participant(p1),
                "participant2": self._serialize_participant(p2),
                "games": games,
                "openings": self._serialize_openings(openings),
                "knockout_config": self.knockout_config,
                "seed": self._next_worker_seed(),
                "output_dir": self.output_dir,
                "command_line": self.command_line,
                "run_desc": self.run_desc,
            }
            worker_result = self._run_tournament_worker("knockout_match", match_payload)

            expected_total_games = games * 2
            total_games = int(worker_result["total_games"])
            if total_games != expected_total_games:
                raise RuntimeError(
                    f"Knockout worker returned invalid game count for {p1.name} vs {p2.name}: "
                    f"expected {expected_total_games}, got {total_games}"
                )

            return MatchResult(
                participant1=p1,
                participant2=p2,
                participant1_wins=int(worker_result["participant1_wins"]),
                participant2_wins=int(worker_result["participant2_wins"]),
                total_games=total_games,
                openings_used=worker_result.get(
                    "openings_used",
                    [opening.get_trmph_string() for opening in openings],
                ),
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
        # Use same TRMPH files as existing tournament system
        trmph_files = find_trmph_files(self.trmph_source)
        
        if not trmph_files:
            raise ValueError(f"No TRMPH files found in {self.trmph_source} for {stage_name} stage")
        
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
        # Generate openings for round-robin stage
        # The run_tournament function will use these openings for each strategy pair
        # Each pair plays round_robin_games * 2 (A vs B and B vs A)
        num_participants = len(self.knockout_winners) + len(self.round_robin_participants)
        if num_participants < 2:
            return []
        
        # Generate round_robin_games openings (not total games)
        # The run_tournament function handles the pairing and game execution
        return self._generate_openings(self.round_robin_games, "round-robin")
    
    def get_tournament_summary(self) -> Dict[str, Any]:
        """Get a summary of the tournament configuration and results."""
        summary = {
            "knockout_dir": self.knockout_dir,
            "knockout_participants_count": len(self.knockout_participants) if self.knockout_participants else None,
            "knockout_config": self.knockout_config,
            "games_per_match": self.games_per_match,
            "top_k": self.top_k,
            "round_robin_games": self.round_robin_games,
            "round_robin_participants": len(self.round_robin_participants),
            "knockout_winners": [p.name for p in self.knockout_winners],
            "epoch_range": self.epoch_range
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
        
        # Use existing output directory (created in run_tournament)
        output_dir = self.output_dir
        
        # Extract timestamp from output directory path
        timestamp = output_dir.split('_')[-1] if output_dir else "unknown"
        
        # Create comprehensive summary
        summary = {
            "tournament_type": "two_stage",
            "timestamp": timestamp,
            "configuration": {
                "knockout_dir": self.knockout_dir,
                "knockout_participants_count": len(self.knockout_participants) if self.knockout_participants else None,
                "knockout_config": self.knockout_config,
                "games_per_match": self.games_per_match,
                "top_k": self.top_k,
                "round_robin_games": self.round_robin_games,
                "round_robin_participants": len(self.round_robin_participants),
                "epoch_range": self.epoch_range
            },
            "results": results
        }
        
        # Save to JSON file
        summary_file = os.path.join(output_dir, "tournament_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Tournament summary saved to: {summary_file}")
        
        # Print final ranking with quantitative measures
        if results.get("final_ranking") and results.get("round_robin_results"):
            print("\n" + "="*80)
            print("FINAL TOURNAMENT RANKING")
            print("="*80)
            
            # Get quantitative data from round-robin results
            rr_results = results["round_robin_results"]
            win_rates = rr_results.get("win_rates", {})
            elo_ratings = rr_results.get("elo_ratings", {})
            participant_games = rr_results.get("participant_games", {})
            total_games = rr_results.get("total_games", 0)
            
            print(f"{'Rank':<4} {'Player':<25} {'Win Rate':<10} {'Elo':<8} {'Games':<6}")
            print("-" * 80)
            
            for i, participant in enumerate(results["final_ranking"], 1):
                win_rate = win_rates.get(participant, 0.0)
                elo = elo_ratings.get(participant, 0.0)
                games = participant_games.get(participant, 0)
                
                print(f"{i:<4} {participant:<25} {win_rate:<10.3f} {elo:<8.0f} {games:<6}")
            
            print(f"\nTotal games played: {total_games}")
            print("="*80)
    
    def _raise_insufficient_checkpoints_error(self, discovery: CheckpointDiscovery, 
                                            epoch_range: Optional[Tuple[int, int]], 
                                            mini_epoch_range: Optional[Tuple[int, int]],
                                            filtered_count: int) -> None:
        """
        Raise a helpful error when insufficient checkpoints are found.
        
        Args:
            discovery: The checkpoint discovery object
            epoch_range: The epoch range filter that was applied (if any)
            mini_epoch_range: The mini epoch range filter that was applied (if any)
            filtered_count: The number of checkpoints found after filtering
        """
        # Get directory contents for helpful error message
        checkpoint_dir = discovery.checkpoint_dir
        try:
            all_files = list(checkpoint_dir.iterdir())
            checkpoint_files = [f for f in all_files if f.is_file() and f.name.endswith('.pt.gz')]
            other_files = [f for f in all_files if f.is_file() and not f.name.endswith('.pt.gz')]
        except Exception as e:
            all_files = []
            checkpoint_files = []
            other_files = []
        
        # Get total checkpoints in directory for comparison
        # This should never fail since discover_checkpoints() was already called successfully
        # earlier in the flow and uses caching
        all_checkpoints = discovery.discover_checkpoints()
        total_checkpoints = len(all_checkpoints)
        
        # Build error message
        error_parts = [
            f"Tournament requires at least 2 checkpoints, but found {filtered_count} matching checkpoints after filtering.",
            f"",
            f"Directory: {checkpoint_dir}",
        ]
        
        # Add information about total vs filtered
        if filtered_count < total_checkpoints:
            error_parts.append(f"Total checkpoints in directory: {total_checkpoints}")
            error_parts.append(f"Filtered checkpoints: {filtered_count}")
            error_parts.append("")
        
        # Add filter information (only if filters were applied)
        if epoch_range or mini_epoch_range:
            error_parts.append("Applied filters:")
            if epoch_range:
                start_epoch, end_epoch = epoch_range
                error_parts.append(f"  Epoch range: {start_epoch}-{end_epoch-1}")
            if mini_epoch_range:
                start_mini, end_mini = mini_epoch_range
                error_parts.append(f"  Mini epoch range: {start_mini}-{end_mini-1}")
            error_parts.append("")
        
        # Add directory contents
        error_parts.extend([
            f"",
            f"Directory contents:"
        ])
        
        if checkpoint_files:
            error_parts.append(f"  Checkpoint files ({len(checkpoint_files)}):")
            for f in sorted(checkpoint_files):
                error_parts.append(f"    {f.name}")
        else:
            error_parts.append(f"  No checkpoint files found (looking for files ending in .pt.gz)")
        
        if other_files:
            error_parts.append(f"  Other files ({len(other_files)}):")
            for f in sorted(other_files):
                error_parts.append(f"    {f.name}")
        
        # Add helpful suggestions
        error_parts.extend([
            f"",
            f"Suggestions:",
            f"  1. Check that the directory contains checkpoint files with pattern 'epochN_miniJ.pt.gz'",
            f"  2. Verify the epoch and mini-epoch ranges include existing checkpoints",
            f"  3. Remove filters to use all available checkpoints",
            f"  4. Check that the directory path is correct"
        ])
        
        error_message = "\n".join(error_parts)
        raise ValueError(error_message)
