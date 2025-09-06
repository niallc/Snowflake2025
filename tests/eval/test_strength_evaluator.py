"""
Tests for the Hex Strength Evaluator.

This module contains comprehensive tests for the strength evaluator functionality,
including unit tests for individual components and integration tests for the
complete evaluation pipeline.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Tuple, Any

from hex_ai.eval.strength_evaluator import (
    StrengthEvaluator, EvaluatorConfig, EvaluatorReport, MoveEval, GameRecord,
    PolicySource, ValueSource, AggregationMethod, GamePhase, PhaseResults
)
from hex_ai.inference.game_engine import HexGameEngine, HexGameState, make_empty_hex_state
from hex_ai.inference.model_wrapper import ModelWrapper
from hex_ai.enums import Player, Winner, Piece
from hex_ai.config import BOARD_SIZE


class TestEvaluatorConfig:
    """Test the EvaluatorConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EvaluatorConfig()
        
        assert config.opening_plies == 12
        assert config.endgame_value_thresh == 0.90
        assert config.endgame_streak == 3
        assert config.use_value_prob_space is True
        assert config.policy_source == PolicySource.MCTS_PRIORS
        assert config.value_source == ValueSource.MCTS_Q
        assert config.mcts_sims == 200
        assert config.mcts_c_puct == 1.5
        assert config.aggregation == AggregationMethod.MEAN
        assert config.trimmed_fraction == 0.1
        assert config.bucket_policy_thresholds == (0.10, 0.30)
        assert config.bucket_value_thresholds == (0.10, 0.30)
        assert config.cache_size == 10000
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = EvaluatorConfig(
            opening_plies=10,
            endgame_value_thresh=0.85,
            mcts_sims=500,
            policy_source=PolicySource.POLICY_NET,
            value_source=ValueSource.VALUE_NET,
            aggregation=AggregationMethod.MEDIAN,
            cache_size=5000
        )
        
        assert config.opening_plies == 10
        assert config.endgame_value_thresh == 0.85
        assert config.mcts_sims == 500
        assert config.policy_source == PolicySource.POLICY_NET
        assert config.value_source == ValueSource.VALUE_NET
        assert config.aggregation == AggregationMethod.MEDIAN
        assert config.cache_size == 5000


class TestGameRecord:
    """Test the GameRecord class."""
    
    def test_game_record_creation(self):
        """Test creating a GameRecord."""
        moves = [
            (0, 0, Player.BLUE),
            (1, 1, Player.RED),
            (2, 2, Player.BLUE)
        ]
        
        game = GameRecord(
            board_size=13,
            moves=moves,
            starting_player=Player.BLUE,
            metadata={"test": "data"}
        )
        
        assert game.board_size == 13
        assert game.moves == moves
        assert game.starting_player == Player.BLUE
        assert game.metadata == {"test": "data"}
    
    def test_game_record_defaults(self):
        """Test GameRecord with default values."""
        moves = [(0, 0, Player.BLUE)]
        
        game = GameRecord(
            board_size=13,
            moves=moves,
            starting_player=Player.BLUE
        )
        
        assert game.metadata is None


class TestMoveEval:
    """Test the MoveEval class."""
    
    def test_move_eval_creation(self):
        """Test creating a MoveEval."""
        move_eval = MoveEval(
            ply_idx=5,
            actor=Player.BLUE,
            phase=GamePhase.MIDDLE,
            chosen_move=(3, 4),
            policy_prob_chosen=0.1,
            policy_prob_best=0.3,
            delta_policy=0.2,
            bucket_policy=-1,
            value_prob_after_chosen=0.4,
            value_prob_after_best=0.6,
            delta_value=0.2,
            bucket_value=0,
            evaluator_metadata={"mcts_sims": 200}
        )
        
        assert move_eval.ply_idx == 5
        assert move_eval.actor == Player.BLUE
        assert move_eval.phase == GamePhase.MIDDLE
        assert move_eval.chosen_move == (3, 4)
        assert move_eval.delta_policy == 0.2
        assert move_eval.delta_value == 0.2
        assert move_eval.bucket_policy == -1
        assert move_eval.bucket_value == 0


class TestPhaseResults:
    """Test the PhaseResults class."""
    
    def test_phase_results_creation(self):
        """Test creating PhaseResults."""
        results = PhaseResults(
            policy_score=0.15,
            value_score=0.12,
            policy_bucket_counts={0: 5, -1: 3, -2: 1},
            value_bucket_counts={0: 6, -1: 2, -2: 1},
            n=9
        )
        
        assert results.policy_score == 0.15
        assert results.value_score == 0.12
        assert results.policy_bucket_counts == {0: 5, -1: 3, -2: 1}
        assert results.value_bucket_counts == {0: 6, -1: 2, -2: 1}
        assert results.n == 9


class TestStrengthEvaluator:
    """Test the main StrengthEvaluator class."""
    
    @pytest.fixture
    def mock_engine(self):
        """Create a mock game engine."""
        engine = Mock(spec=HexGameEngine)
        return engine
    
    @pytest.fixture
    def mock_model_wrapper(self):
        """Create a mock model wrapper."""
        model = Mock(spec=ModelWrapper)
        
        # Mock predict method to return dummy policy and value
        def mock_predict(tensor):
            policy_logits = torch.randn(BOARD_SIZE * BOARD_SIZE)
            value_signed = torch.tensor([0.1])  # Slightly positive for Red
            return policy_logits, value_signed
        
        model.predict = mock_predict
        return model
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return EvaluatorConfig(
            mcts_sims=10,  # Small number for testing
            cache_size=100,
            rng_seed=42
        )
    
    @pytest.fixture
    def evaluator(self, mock_engine, mock_model_wrapper, config):
        """Create a test evaluator."""
        return StrengthEvaluator(mock_engine, mock_model_wrapper, config)
    
    @pytest.fixture
    def simple_game(self):
        """Create a simple test game."""
        moves = [
            (0, 0, Player.BLUE),
            (1, 1, Player.RED),
            (2, 2, Player.BLUE),
            (3, 3, Player.RED)
        ]
        
        return GameRecord(
            board_size=13,
            moves=moves,
            starting_player=Player.BLUE
        )
    
    def test_evaluator_initialization(self, mock_engine, mock_model_wrapper, config):
        """Test evaluator initialization."""
        evaluator = StrengthEvaluator(mock_engine, mock_model_wrapper, config)
        
        assert evaluator.engine == mock_engine
        assert evaluator.model_wrapper == mock_model_wrapper
        assert evaluator.cfg == config
        assert evaluator.cache_hits == 0
        assert evaluator.cache_misses == 0
        assert len(evaluator.policy_cache) == 0
        assert len(evaluator.value_cache) == 0
    
    def test_reconstruct_positions(self, evaluator, simple_game):
        """Test position reconstruction."""
        states = evaluator._reconstruct_positions(simple_game)
        
        assert len(states) == len(simple_game.moves) + 1  # +1 for initial empty state
        
        # Check that first state is empty
        first_state = states[0]
        assert first_state.current_player_enum == Player.BLUE
        assert len(first_state.move_history) == 0
        
        # Check that moves are applied correctly
        for i, (state, (row, col, player)) in enumerate(zip(states[1:], simple_game.moves)):
            assert len(state.move_history) == i + 1
            assert state.move_history[-1] == (row, col)
    
    def test_assign_phases(self, evaluator, simple_game):
        """Test phase assignment."""
        states = evaluator._reconstruct_positions(simple_game)
        phases = evaluator._assign_phases(states)
        
        assert len(phases) == len(simple_game.moves)
        
        # First few moves should be opening
        for i in range(min(evaluator.cfg.opening_plies, len(phases))):
            assert phases[i] == GamePhase.OPENING
    
    def test_calculate_delta_policy(self, evaluator):
        """Test policy delta calculation."""
        policy_dict = {
            (0, 0): 0.1,
            (1, 1): 0.3,
            (2, 2): 0.2
        }
        
        # Test with best move
        delta = evaluator._calculate_delta_policy(policy_dict, (1, 1))
        assert delta == 0.0  # Best move has no delta
        
        # Test with suboptimal move
        delta = evaluator._calculate_delta_policy(policy_dict, (0, 0))
        assert delta == 0.2  # 0.3 - 0.1 = 0.2
    
    def test_calculate_delta_value(self, evaluator):
        """Test value delta calculation."""
        value_dict = {
            (0, 0): 0.4,
            (1, 1): 0.6,
            (2, 2): 0.5
        }
        
        # Test with best move
        delta = evaluator._calculate_delta_value(value_dict, (1, 1))
        assert delta == 0.0  # Best move has no delta
        
        # Test with suboptimal move
        delta = evaluator._calculate_delta_value(value_dict, (0, 0))
        assert delta == 0.2  # 0.6 - 0.4 = 0.2
    
    def test_calculate_bucket(self, evaluator):
        """Test bucket calculation."""
        # Test big threshold
        bucket = evaluator._calculate_bucket(0.4, (0.1, 0.3))
        assert bucket == -2
        
        # Test small threshold
        bucket = evaluator._calculate_bucket(0.2, (0.1, 0.3))
        assert bucket == -1
        
        # Test no threshold
        bucket = evaluator._calculate_bucket(0.05, (0.1, 0.3))
        assert bucket == 0
    
    def test_aggregate_values_mean(self, evaluator):
        """Test value aggregation with mean."""
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = evaluator._aggregate_values(values)
        assert result == 0.3  # Mean of values
    
    def test_aggregate_values_median(self, evaluator):
        """Test value aggregation with median."""
        evaluator.cfg.aggregation = AggregationMethod.MEDIAN
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = evaluator._aggregate_values(values)
        assert result == 0.3  # Median of values
    
    def test_aggregate_values_trimmed_mean(self, evaluator):
        """Test value aggregation with trimmed mean."""
        evaluator.cfg.aggregation = AggregationMethod.TRIMMED_MEAN
        evaluator.cfg.trimmed_fraction = 0.2  # Trim 20% from each end
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = evaluator._aggregate_values(values)
        # Should trim 0.1 and 0.5, leaving [0.2, 0.3, 0.4] with mean 0.3
        assert result == 0.3
    
    def test_trimmed_mean_edge_cases(self, evaluator):
        """Test trimmed mean edge cases."""
        # Empty list
        result = evaluator._trimmed_mean([])
        assert result == 0.0
        
        # Single value
        result = evaluator._trimmed_mean([0.5])
        assert result == 0.5
        
        # Two values
        result = evaluator._trimmed_mean([0.3, 0.7])
        assert result == 0.5
    
    def test_logits_to_probs(self, evaluator):
        """Test logits to probabilities conversion."""
        logits = np.array([1.0, 2.0, 3.0])
        probs = evaluator._logits_to_probs(logits)
        
        # Check that probabilities sum to 1
        assert abs(np.sum(probs) - 1.0) < 1e-6
        
        # Check that highest logit has highest probability
        assert probs[2] > probs[1] > probs[0]
    
    def test_get_cache_stats(self, evaluator):
        """Test cache statistics."""
        stats = evaluator.get_cache_stats()
        
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "hit_rate" in stats
        assert "policy_cache_size" in stats
        assert "value_cache_size" in stats
        assert "mcts_cache_size" in stats
        
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["hit_rate"] == 0.0


class TestIntegration:
    """Integration tests for the complete evaluation pipeline."""
    
    @pytest.fixture
    def mock_engine(self):
        """Create a mock game engine."""
        engine = Mock(spec=HexGameEngine)
        return engine
    
    @pytest.fixture
    def mock_model_wrapper(self):
        """Create a mock model wrapper."""
        model = Mock(spec=ModelWrapper)
        
        # Mock predict method with more realistic outputs
        def mock_predict(tensor):
            # Create policy logits with some structure
            policy_logits = torch.randn(BOARD_SIZE * BOARD_SIZE)
            # Make some moves more likely
            policy_logits[0] = 2.0  # (0,0) move
            policy_logits[14] = 1.5  # (1,1) move
            
            # Value slightly positive for Red
            value_signed = torch.tensor([0.2])
            return policy_logits, value_signed
        
        model.predict = mock_predict
        return model
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return EvaluatorConfig(
            mcts_sims=5,  # Very small for testing
            cache_size=50,
            rng_seed=42,
            opening_plies=2,  # Small opening for testing
            ignore_early_noise_until=0  # Don't ignore any moves
        )
    
    @pytest.fixture
    def evaluator(self, mock_engine, mock_model_wrapper, config):
        """Create a test evaluator."""
        return StrengthEvaluator(mock_engine, mock_model_wrapper, config)
    
    @pytest.fixture
    def test_game(self):
        """Create a test game with known characteristics."""
        moves = [
            (0, 0, Player.BLUE),   # Opening
            (1, 1, Player.RED),    # Opening
            (2, 2, Player.BLUE),   # Middle
            (3, 3, Player.RED),    # Middle
            (4, 4, Player.BLUE),   # Middle
            (5, 5, Player.RED)     # Middle
        ]
        
        return GameRecord(
            board_size=13,
            moves=moves,
            starting_player=Player.BLUE
        )
    
    @patch('hex_ai.eval.strength_evaluator.BaselineMCTS')
    def test_evaluate_game_integration(self, mock_mcts_class, evaluator, test_game):
        """Test complete game evaluation."""
        # Mock MCTS
        mock_mcts = Mock()
        mock_root = Mock()
        mock_root.N = np.array([5, 3, 2, 1])  # Visit counts
        mock_root.Q = np.array([0.1, 0.2, 0.15, 0.05])  # Q values
        mock_mcts.search.return_value = mock_root
        mock_mcts_class.return_value = mock_mcts
        
        # Evaluate game
        report = evaluator.evaluate_game(test_game)
        
        # Check report structure
        assert isinstance(report, EvaluatorReport)
        assert report.per_phase_per_player is not None
        assert report.coverage is not None
        
        # Check coverage
        assert report.coverage["total_plies"] == len(test_game.moves)
        assert report.coverage["evaluated_plies"] > 0
        
        # Check that we have results for different phases
        phases_found = set()
        players_found = set()
        
        for (phase, player), results in report.per_phase_per_player.items():
            phases_found.add(phase)
            players_found.add(player)
            
            assert isinstance(results, PhaseResults)
            assert results.n > 0
            assert 0 <= results.policy_score <= 1
            assert 0 <= results.value_score <= 1
        
        # Should have both players
        assert Player.BLUE in players_found
        assert Player.RED in players_found
        
        # Should have opening phase (first 2 moves)
        assert GamePhase.OPENING in phases_found
    
    def test_evaluate_game_with_termination(self, evaluator, test_game):
        """Test game evaluation with early termination."""
        # Create a game that ends early
        moves = [
            (0, 0, Player.BLUE),
            (1, 1, Player.RED),
            (2, 2, Player.BLUE)
        ]
        
        short_game = GameRecord(
            board_size=13,
            moves=moves,
            starting_player=Player.BLUE
        )
        
        # Mock the model to return terminal values
        def mock_predict_terminal(tensor):
            policy_logits = torch.randn(BOARD_SIZE * BOARD_SIZE)
            value_signed = torch.tensor([0.95])  # Very high value (endgame)
            return policy_logits, value_signed
        
        evaluator.model_wrapper.predict = mock_predict_terminal
        
        # Evaluate game
        report = evaluator.evaluate_game(short_game)
        
        # Should still produce valid results
        assert isinstance(report, EvaluatorReport)
        assert report.coverage["total_plies"] == len(short_game.moves)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def mock_engine(self):
        """Create a mock game engine."""
        return Mock(spec=HexGameEngine)
    
    @pytest.fixture
    def mock_model_wrapper(self):
        """Create a mock model wrapper."""
        model = Mock(spec=ModelWrapper)
        model.predict.return_value = (torch.randn(BOARD_SIZE * BOARD_SIZE), torch.tensor([0.0]))
        return model
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return EvaluatorConfig(mcts_sims=1, cache_size=10)
    
    @pytest.fixture
    def evaluator(self, mock_engine, mock_model_wrapper, config):
        """Create a test evaluator."""
        return StrengthEvaluator(mock_engine, mock_model_wrapper, config)
    
    def test_empty_game(self, evaluator):
        """Test evaluation of empty game."""
        empty_game = GameRecord(
            board_size=13,
            moves=[],
            starting_player=Player.BLUE
        )
        
        report = evaluator.evaluate_game(empty_game)
        
        assert isinstance(report, EvaluatorReport)
        assert len(report.per_phase_per_player) == 0
        assert report.coverage["total_plies"] == 0
        assert report.coverage["evaluated_plies"] == 0
    
    def test_single_move_game(self, evaluator):
        """Test evaluation of single move game."""
        single_move_game = GameRecord(
            board_size=13,
            moves=[(0, 0, Player.BLUE)],
            starting_player=Player.BLUE
        )
        
        report = evaluator.evaluate_game(single_move_game)
        
        assert isinstance(report, EvaluatorReport)
        assert report.coverage["total_plies"] == 1
    
    def test_invalid_move_handling(self, evaluator):
        """Test handling of invalid moves."""
        invalid_game = GameRecord(
            board_size=13,
            moves=[(0, 0, Player.BLUE), (0, 0, Player.RED)],  # Duplicate move
            starting_player=Player.BLUE
        )
        
        # Should raise ValueError for invalid move
        with pytest.raises(ValueError):
            evaluator.evaluate_game(invalid_game)
    
    def test_model_error_handling(self, evaluator):
        """Test handling of model errors."""
        # Mock model to raise exception
        evaluator.model_wrapper.predict.side_effect = Exception("Model error")
        
        test_game = GameRecord(
            board_size=13,
            moves=[(0, 0, Player.BLUE), (1, 1, Player.RED)],
            starting_player=Player.BLUE
        )
        
        # Should handle model errors gracefully
        report = evaluator.evaluate_game(test_game)
        
        # Should still produce a report, possibly with warnings
        assert isinstance(report, EvaluatorReport)


if __name__ == "__main__":
    pytest.main([__file__])
