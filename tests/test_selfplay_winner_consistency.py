"""
Test self-play winner consistency with opening moves.

This test verifies that the self-play engine correctly handles winner labeling
when using opening strategies vs empty board starts.
"""

import pytest
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hex_ai.selfplay.selfplay_engine import SelfPlayEngine
from hex_ai.inference.game_engine import HexGameState
from hex_ai.inference.model_config import get_model_path
from hex_ai.utils.opening_strategies import create_pie_rule_strategy
from hex_ai.enums import Winner
from hex_ai.config import TRMPH_RED_WIN, TRMPH_BLUE_WIN


class TestSelfPlayWinnerConsistency:
    """Test that self-play winner labeling is consistent with opening moves."""
    
    def test_winner_consistency_with_opening_moves(self):
        """Test that winner labeling is consistent with and without opening moves."""
        # Initialize self-play engine with minimal MCTS for speed
        engine = SelfPlayEngine(
            model_path=get_model_path("current_best"),
            batch_size=32,
            cache_size=1000,
            temperature=1.0,
            verbose=0,  # Quiet for tests
            streaming_save=False,
            use_batched_inference=True,
            mcts_sims=1  # Minimal MCTS for speed
        )
        
        # Test 1: Game without opening move (empty board)
        game_data = engine._generate_single_game(board_size=13, opening_move=None)
        trmph = game_data['trmph']
        winner_label = game_data['winner']
        
        # Reconstruct board and recalculate winner
        state = HexGameState.from_trmph(trmph)
        calculated_winner = state._find_winner()
        
        # Verify consistency if game is over
        if calculated_winner is not None:
            expected_winner_label = TRMPH_BLUE_WIN if calculated_winner == Winner.BLUE else TRMPH_RED_WIN
            assert winner_label == expected_winner_label, \
                f"Winner label mismatch: {winner_label} vs {expected_winner_label}"
        
        # Test 2: Game with opening move
        opening_strategy = create_pie_rule_strategy()
        opening_move = opening_strategy.get_opening_move(0)  # First opening move (a2)
        
        game_data2 = engine._generate_single_game(board_size=13, opening_move=opening_move)
        trmph2 = game_data2['trmph']
        winner_label2 = game_data2['winner']
        
        # Reconstruct board and recalculate winner
        state2 = HexGameState.from_trmph(trmph2)
        calculated_winner2 = state2._find_winner()
        
        # Verify consistency if game is over
        if calculated_winner2 is not None:
            expected_winner_label2 = TRMPH_BLUE_WIN if calculated_winner2 == Winner.BLUE else TRMPH_RED_WIN
            assert winner_label2 == expected_winner_label2, \
                f"Winner label mismatch with opening move: {winner_label2} vs {expected_winner_label2}"
        
        # Test 3: Verify first piece ownership
        if len(state.move_history) > 0:
            first_move = state.move_history[0]
            # First move should be Blue's (empty board case)
            assert first_move is not None, "Empty board game should have moves"
        
        if len(state2.move_history) > 0:
            first_move2 = state2.move_history[0]
            # First move should be the opening move (Blue's opening move)
            assert first_move2 == opening_move, \
                f"Opening move mismatch: expected {opening_move}, got {first_move2}"
    
    def test_opening_strategy_integration(self):
        """Test that opening strategy integration works correctly."""
        # Test that opening strategy provides the expected moves
        strategy = create_pie_rule_strategy()
        
        # First few moves should be balanced moves
        expected_moves = [(1, 0), (2, 0), (3, 0), (4, 0)]  # a2, a3, a4, a5
        
        for i, expected_move in enumerate(expected_moves):
            opening_move = strategy.get_opening_move(i)
            assert opening_move == expected_move, \
                f"Opening move {i} mismatch: expected {expected_move}, got {opening_move}"
        
        # Test that network-chosen moves return None
        network_game_index = strategy.balanced_games + strategy.unbalanced_games
        network_move = strategy.get_opening_move(network_game_index)
        assert network_move is None, \
            f"Network game should return None, got {network_move}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
