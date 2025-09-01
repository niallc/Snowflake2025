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
    from hex_ai.value_utils import ValuePredictor
    from hex_ai.enums import Player as PlayerEnum
except ImportError:
    print("ERROR: Could not import ValuePredictor")
    print("Make sure to run with PYTHONPATH=.")
    print("Example: PYTHONPATH=. python -m pytest tests/test_fixed_tree_search.py -v")
    raise


class TestConvertToMinimaxValue:
    """Test cases for the convert_to_minimax_value function."""
    
    def test_blue_root_player_neutral_position(self):
        """Test neutral position (50% chance) from Blue's perspective."""
        # value_signed = 0.0 corresponds to 50% Red wins
        result = ValuePredictor.convert_to_minimax_value(0.0, root_player=PlayerEnum.BLUE)
        assert result == 0.0  # Neutral for Blue
        
    def test_red_root_player_neutral_position(self):
        """Test neutral position (50% chance) from Red's perspective."""
        # value_signed = 0.0 corresponds to 50% Red wins
        result = ValuePredictor.convert_to_minimax_value(0.0, root_player=PlayerEnum.RED)
        assert result == 0.0  # Neutral for Red
        
    def test_blue_root_player_blue_winning(self):
        """Test Blue winning position from Blue's perspective."""
        # value_signed = -0.8 corresponds to 10% Red wins (90% Blue wins)
        result = ValuePredictor.convert_to_minimax_value(-0.8, root_player=PlayerEnum.BLUE)
        assert result > 0.5  # Good for Blue
        
    def test_blue_root_player_red_winning(self):
        """Test Red winning position from Blue's perspective."""
        # value_signed = 0.8 corresponds to 90% Red wins (10% Blue wins)
        result = ValuePredictor.convert_to_minimax_value(0.8, root_player=PlayerEnum.BLUE)
        assert result < -0.5  # Bad for Blue
        
    def test_red_root_player_red_winning(self):
        """Test Red winning position from Red's perspective."""
        # value_signed = 0.8 corresponds to 90% Red wins
        result = ValuePredictor.convert_to_minimax_value(0.8, root_player=PlayerEnum.RED)
        assert result > 0.5  # Good for Red
        
    def test_red_root_player_blue_winning(self):
        """Test Blue winning position from Red's perspective."""
        # value_signed = -0.8 corresponds to 10% Red wins (90% Blue wins)
        result = ValuePredictor.convert_to_minimax_value(-0.8, root_player=PlayerEnum.RED)
        assert result < -0.5  # Bad for Red
        
    def test_extreme_positive_value_signed(self):
        """Test very high value_signed (near certain Red win)."""
        # value_signed = 0.99 corresponds to 99.5% Red wins
        result_blue = ValuePredictor.convert_to_minimax_value(0.99, root_player=PlayerEnum.BLUE)
        result_red = ValuePredictor.convert_to_minimax_value(0.99, root_player=PlayerEnum.RED)
        
        # Blue should see this as very bad
        assert result_blue <= -0.99
        # Red should see this as very good
        assert result_red >= 0.99
        
    def test_extreme_negative_value_signed(self):
        """Test very low value_signed (near certain Blue win)."""
        # value_signed = -0.99 corresponds to 0.5% Red wins (99.5% Blue wins)
        result_blue = ValuePredictor.convert_to_minimax_value(-0.99, root_player=PlayerEnum.BLUE)
        result_red = ValuePredictor.convert_to_minimax_value(-0.99, root_player=PlayerEnum.RED)
        
        # Blue should see this as very good
        assert result_blue >= 0.99
        # Red should see this as very bad
        assert result_red <= -0.99
        
    def test_boundary_values(self):
        """Test that output values are properly bounded in [-1, 1]."""
        # Test a range of value_signed values
        values = [-1.0, -0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 1.0]
        
        for value in values:
            result_blue = ValuePredictor.convert_to_minimax_value(value, root_player=PlayerEnum.BLUE)
            result_red = ValuePredictor.convert_to_minimax_value(value, root_player=PlayerEnum.RED)
            
            # Check bounds
            assert -1.0 <= result_blue <= 1.0
            assert -1.0 <= result_red <= 1.0
            
    def test_symmetry_property(self):
        """Test that Blue and Red perspectives are symmetric for opposite values."""
        # For value_signed V, Blue's value should equal Red's value for value_signed -V
        values = [-0.8, -0.5, 0.0, 0.5, 0.8]
        
        for value in values:
            blue_value = ValuePredictor.convert_to_minimax_value(value, root_player=PlayerEnum.BLUE)
            red_value_opposite = ValuePredictor.convert_to_minimax_value(-value, root_player=PlayerEnum.RED)
            
            # The symmetry should hold: Blue's value for V = Red's value for -V
            # Allow for floating point precision errors
            assert abs(blue_value - red_value_opposite) < 1e-7
            
    def test_monotonicity(self):
        """Test that higher value_signed leads to better values for Red and worse for Blue."""
        values = [-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8]
        
        blue_values = [ValuePredictor.convert_to_minimax_value(value, root_player=PlayerEnum.BLUE) for value in values]
        red_values = [ValuePredictor.convert_to_minimax_value(value, root_player=PlayerEnum.RED) for value in values]
        
        # Blue values should be monotonically decreasing (higher value_signed = worse for Blue)
        for i in range(1, len(blue_values)):
            assert blue_values[i] <= blue_values[i-1]
            
        # Red values should be monotonically increasing (higher value_signed = better for Red)
        for i in range(1, len(red_values)):
            assert red_values[i] >= red_values[i-1]
            
    def test_specific_calculations(self):
        """Test specific known calculations."""
        # Test with value_signed = 0.6: corresponds to 80% Red wins
        result_blue = ValuePredictor.convert_to_minimax_value(0.6, root_player=PlayerEnum.BLUE)
        result_red = ValuePredictor.convert_to_minimax_value(0.6, root_player=PlayerEnum.RED)
        
        # Compute expected using actual probability conversion
        prob_red = ValuePredictor.model_output_to_probability(0.6)
        expected_blue = 1.0 - 2.0 * prob_red
        expected_red = 2.0 * prob_red - 1.0
        
        assert abs(result_blue - expected_blue) < 1e-10
        assert abs(result_red - expected_red) < 1e-10
        
    def test_invalid_root_player(self):
        """Test that invalid root_player values raise appropriate errors."""
        with pytest.raises(TypeError):
            ValuePredictor.convert_to_minimax_value(0.0, root_player=2)
            
        with pytest.raises(TypeError):
            ValuePredictor.convert_to_minimax_value(0.0, root_player=-1)
            
    def test_numeric_types(self):
        """Test that function works with different numeric types."""
        # Test with numpy float
        result_np = ValuePredictor.convert_to_minimax_value(np.float32(0.0), root_player=PlayerEnum.BLUE)
        assert result_np == 0.0
        
        # Test with torch tensor (should work after .item())
        result_torch = ValuePredictor.convert_to_minimax_value(torch.tensor(0.0).item(), root_player=PlayerEnum.BLUE)
        assert result_torch == 0.0


if __name__ == "__main__":
    pytest.main([__file__]) 