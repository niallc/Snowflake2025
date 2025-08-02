"""
Test the new enum system for player and piece constants.

This demonstrates the benefits of using enums over simple integer constants.
"""

import pytest
from hex_ai.value_utils import (
    Player, Piece, Channel, Winner,
    player_to_int, int_to_player, piece_to_int, int_to_piece,
    get_opponent, is_blue, is_red, winner_to_color,
    player_to_winner, winner_to_player
)


class TestEnumSystem:
    """Test the enum system functionality."""
    
    def test_player_enum_basic(self):
        """Test basic Player enum functionality."""
        assert Player.BLUE.value == 0
        assert Player.RED.value == 1
        assert Player.BLUE != Player.RED
        
        # Type safety - can't assign invalid values
        with pytest.raises(ValueError):
            Player(42)  # Invalid player value
    
    def test_piece_enum_basic(self):
        """Test basic Piece enum functionality."""
        assert Piece.EMPTY.value == 0
        assert Piece.BLUE.value == 1
        assert Piece.RED.value == 2
        
        # Type safety - can't assign invalid values
        with pytest.raises(ValueError):
            Piece(99)  # Invalid piece value
    
    def test_channel_enum_basic(self):
        """Test basic Channel enum functionality."""
        assert Channel.BLUE.value == 0
        assert Channel.RED.value == 1
        assert Channel.PLAYER_TO_MOVE.value == 2
    
    def test_conversion_functions(self):
        """Test conversion functions for backward compatibility."""
        # Player conversions
        assert player_to_int(Player.BLUE) == 0
        assert player_to_int(Player.RED) == 1
        assert int_to_player(0) == Player.BLUE
        assert int_to_player(1) == Player.RED
        
        # Piece conversions
        assert piece_to_int(Piece.BLUE) == 1
        assert piece_to_int(Piece.RED) == 2
        assert int_to_piece(1) == Piece.BLUE
        assert int_to_piece(2) == Piece.RED
    
    def test_utility_functions(self):
        """Test utility functions for common operations."""
        # Opponent function
        assert get_opponent(Player.BLUE) == Player.RED
        assert get_opponent(Player.RED) == Player.BLUE
        
        # Color checking
        assert is_blue(Player.BLUE)
        assert is_blue(Winner.BLUE)
        assert not is_blue(Player.RED)
        assert not is_blue(Winner.RED)
        
        assert is_red(Player.RED)
        assert is_red(Winner.RED)
        assert not is_red(Player.BLUE)
        assert not is_red(Winner.BLUE)
        
        # Conversion between Player and Winner
        assert player_to_winner(Player.BLUE) == Winner.BLUE
        assert player_to_winner(Player.RED) == Winner.RED
        assert winner_to_player(Winner.BLUE) == Player.BLUE
        assert winner_to_player(Winner.RED) == Player.RED
    
    def test_backward_compatibility(self):
        """Test that the system maintains backward compatibility with integer constants."""
        from hex_ai.config import BLUE_PLAYER, RED_PLAYER
        
        # Should work with old integer constants
        assert is_blue(BLUE_PLAYER)
        assert is_red(RED_PLAYER)
        assert not is_blue(RED_PLAYER)
        assert not is_red(BLUE_PLAYER)
    
    def test_type_safety_benefits(self):
        """Demonstrate the type safety benefits of enums."""
        # This would be caught at runtime with enums
        with pytest.raises(ValueError):
            Player(999)  # Invalid player value
        
        # But with simple constants, this would silently work:
        # BLUE_PLAYER = 999  # This would break everything!
        
        # Enums prevent accidental reassignment
        # Player.BLUE = 999  # This would raise an AttributeError


class TestEnumUsageExamples:
    """Show practical examples of how enums improve code."""
    
    def test_game_logic_example(self):
        """Example of how enums improve game logic."""
        current_player = Player.BLUE
        
        # Clear and type-safe
        if current_player == Player.BLUE:
            next_player = Player.RED
        else:
            next_player = Player.BLUE
        
        # Even better with utility function
        next_player = get_opponent(current_player)
        
        assert next_player == Player.RED
        assert is_red(next_player)
    
    def test_board_representation_example(self):
        """Example of how enums improve board representation."""
        # Clear what each value means
        board_position = Piece.EMPTY  # Much clearer than board_position = 0
        
        # Type-safe channel access
        blue_channel = Channel.BLUE.value  # 0
        red_channel = Channel.RED.value    # 1
        player_channel = Channel.PLAYER_TO_MOVE.value  # 2
        
        # Clear intent in tensor operations
        # board_tensor[Channel.BLUE.value, row, col] = 1  # Much clearer than board_tensor[0, row, col] = 1
    
    def test_winner_detection_example(self):
        """Example of how enums improve winner detection."""
        winner = Winner.BLUE
        
        # Clear and type-safe
        if winner == Winner.BLUE:
            color = "blue"
        else:
            color = "red"
        
        # Even better with utility function
        color = winner_to_color(winner)
        
        assert color == "blue"
        assert is_blue(winner)


if __name__ == "__main__":
    pytest.main([__file__]) 