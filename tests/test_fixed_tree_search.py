"""
Unit tests for fixed_tree_search module.

To run these tests, use:
    PYTHONPATH=. python -m pytest tests/test_fixed_tree_search.py -v

Or:
    PYTHONPATH=. python tests/test_fixed_tree_search.py
"""

import pytest
import torch
import numpy as np

try:
    from hex_ai.inference.fixed_tree_search import convert_model_logit_to_minimax_value
    from hex_ai.config import BLUE_PLAYER, RED_PLAYER
except ImportError:
    print("ERROR: Could not import convert_model_logit_to_minimax_value")
    print("Make sure to run with PYTHONPATH=.")
    print("Example: PYTHONPATH=. python -m pytest tests/test_fixed_tree_search.py -v")
    raise


class TestConvertModelLogitToMinimaxValue:
    """Test cases for the convert_model_logit_to_minimax_value function."""
    
    def test_blue_root_player_neutral_position(self):
        """Test neutral position (50% chance) from Blue's perspective."""
        # logit = 0.0 corresponds to sigmoid(0.0) = 0.5 (50% Red wins)
        result = convert_model_logit_to_minimax_value(0.0, root_player=BLUE_PLAYER)
        assert result == 0.0  # Neutral for Blue
        
    def test_red_root_player_neutral_position(self):
        """Test neutral position (50% chance) from Red's perspective."""
        # logit = 0.0 corresponds to sigmoid(0.0) = 0.5 (50% Red wins)
        result = convert_model_logit_to_minimax_value(0.0, root_player=RED_PLAYER)
        assert result == 0.0  # Neutral for Red
        
    def test_blue_root_player_blue_winning(self):
        """Test Blue winning position from Blue's perspective."""
        # logit = -2.0 corresponds to sigmoid(-2.0) ≈ 0.119 (11.9% Red wins, 88.1% Blue wins)
        result = convert_model_logit_to_minimax_value(-2.0, root_player=BLUE_PLAYER)
        # Compute expected using actual sigmoid value
        prob_red = torch.sigmoid(torch.tensor(-2.0)).item()
        expected = 1.0 - 2.0 * prob_red
        assert abs(result - expected) < 1e-10  # Use tighter tolerance for computed values
        assert result > 0.0  # Positive = good for Blue
        
    def test_blue_root_player_red_winning(self):
        """Test Red winning position from Blue's perspective."""
        # logit = 2.0 corresponds to sigmoid(2.0) ≈ 0.881 (88.1% Red wins, 11.9% Blue wins)
        result = convert_model_logit_to_minimax_value(2.0, root_player=BLUE_PLAYER)
        # Compute expected using actual sigmoid value
        prob_red = torch.sigmoid(torch.tensor(2.0)).item()
        expected = 1.0 - 2.0 * prob_red
        assert abs(result - expected) < 1e-10  # Use tighter tolerance for computed values
        assert result < 0.0  # Negative = bad for Blue
        
    def test_red_root_player_red_winning(self):
        """Test Red winning position from Red's perspective."""
        # logit = 2.0 corresponds to sigmoid(2.0) ≈ 0.881 (88.1% Red wins)
        result = convert_model_logit_to_minimax_value(2.0, root_player=RED_PLAYER)
        # Compute expected using actual sigmoid value
        prob_red = torch.sigmoid(torch.tensor(2.0)).item()
        expected = 2.0 * prob_red - 1.0
        assert abs(result - expected) < 1e-10  # Use tighter tolerance for computed values
        assert result > 0.0  # Positive = good for Red
        
    def test_red_root_player_blue_winning(self):
        """Test Blue winning position from Red's perspective."""
        # logit = -2.0 corresponds to sigmoid(-2.0) ≈ 0.119 (11.9% Red wins)
        result = convert_model_logit_to_minimax_value(-2.0, root_player=RED_PLAYER)
        # Compute expected using actual sigmoid value
        prob_red = torch.sigmoid(torch.tensor(-2.0)).item()
        expected = 2.0 * prob_red - 1.0
        assert abs(result - expected) < 1e-10  # Use tighter tolerance for computed values
        assert result < 0.0  # Negative = bad for Red
        
    def test_extreme_positive_logit(self):
        """Test very high logit (near certain Red win)."""
        # logit = 10.0 corresponds to sigmoid(10.0) ≈ 0.99995 (99.995% Red wins)
        result_blue = convert_model_logit_to_minimax_value(10.0, root_player=BLUE_PLAYER)
        result_red = convert_model_logit_to_minimax_value(10.0, root_player=RED_PLAYER)
        
        # Blue should see this as very bad
        assert result_blue < -0.99
        # Red should see this as very good
        assert result_red > 0.99
        
    def test_extreme_negative_logit(self):
        """Test very low logit (near certain Blue win)."""
        # logit = -10.0 corresponds to sigmoid(-10.0) ≈ 0.00005 (0.005% Red wins)
        result_blue = convert_model_logit_to_minimax_value(-10.0, root_player=BLUE_PLAYER)
        result_red = convert_model_logit_to_minimax_value(-10.0, root_player=RED_PLAYER)
        
        # Blue should see this as very good
        assert result_blue > 0.99
        # Red should see this as very bad
        assert result_red < -0.99
        
    def test_boundary_values(self):
        """Test that output values are properly bounded in [-1, 1]."""
        # Test a range of logits
        logits = [-10.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        for logit in logits:
            result_blue = convert_model_logit_to_minimax_value(logit, root_player=BLUE_PLAYER)
            result_red = convert_model_logit_to_minimax_value(logit, root_player=RED_PLAYER)
            
            # Check bounds
            assert -1.0 <= result_blue <= 1.0
            assert -1.0 <= result_red <= 1.0
            
    def test_symmetry_property(self):
        """Test that Blue and Red perspectives are symmetric for opposite logits."""
        # For logit L, Blue's value should equal Red's value for logit -L
        # This is because: Blue's value = 1 - 2*sigmoid(L), Red's value = 2*sigmoid(-L) - 1
        # And sigmoid(-L) = 1 - sigmoid(L), so Red's value = 2*(1-sigmoid(L)) - 1 = 1 - 2*sigmoid(L) = Blue's value
        logits = [-3.0, -1.0, 0.0, 1.0, 3.0]
        
        for logit in logits:
            blue_value = convert_model_logit_to_minimax_value(logit, root_player=BLUE_PLAYER)
            red_value_opposite = convert_model_logit_to_minimax_value(-logit, root_player=RED_PLAYER)
            
            # The symmetry should hold: Blue's value for logit L = Red's value for logit -L
            # Allow for floating point precision errors (typically ~1e-15 for double precision)
            assert abs(blue_value - red_value_opposite) < 1e-7  # Conservative tolerance for floating point precision
            
    def test_monotonicity(self):
        """Test that higher logits lead to better values for Red and worse for Blue."""
        logits = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        
        blue_values = [convert_model_logit_to_minimax_value(logit, root_player=BLUE_PLAYER) for logit in logits]
        red_values = [convert_model_logit_to_minimax_value(logit, root_player=RED_PLAYER) for logit in logits]
        
        # Blue values should be monotonically decreasing (higher logit = worse for Blue)
        for i in range(1, len(blue_values)):
            assert blue_values[i] <= blue_values[i-1]
            
        # Red values should be monotonically increasing (higher logit = better for Red)
        for i in range(1, len(red_values)):
            assert red_values[i] >= red_values[i-1]
            
    def test_specific_calculations(self):
        """Test specific known calculations."""
        # Test with logit = 1.0: sigmoid(1.0) ≈ 0.731
        result_blue = convert_model_logit_to_minimax_value(1.0, root_player=BLUE_PLAYER)
        result_red = convert_model_logit_to_minimax_value(1.0, root_player=RED_PLAYER)
        
        # Compute expected using actual sigmoid value
        prob_red = torch.sigmoid(torch.tensor(1.0)).item()
        expected_blue = 1.0 - 2.0 * prob_red
        expected_red = 2.0 * prob_red - 1.0
        
        assert abs(result_blue - expected_blue) < 1e-10
        assert abs(result_red - expected_red) < 1e-10
        
    def test_invalid_root_player(self):
        """Test that invalid root_player values raise appropriate errors."""
        with pytest.raises(ValueError):
            convert_model_logit_to_minimax_value(0.0, root_player=2)
            
        with pytest.raises(ValueError):
            convert_model_logit_to_minimax_value(0.0, root_player=-1)
            
    def test_numeric_types(self):
        """Test that function works with different numeric types."""
        # Test with numpy float
        result_np = convert_model_logit_to_minimax_value(np.float32(0.0), root_player=BLUE_PLAYER)
        assert result_np == 0.0
        
        # Test with torch tensor (should work after .item())
        result_torch = convert_model_logit_to_minimax_value(torch.tensor(0.0).item(), root_player=BLUE_PLAYER)
        assert result_torch == 0.0


if __name__ == "__main__":
    pytest.main([__file__]) 