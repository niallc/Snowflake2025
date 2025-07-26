"""
Tests for move application utilities in hex_ai.value_utils.

These tests verify that moves are correctly applied to both tensors and game states,
and that the utilities handle edge cases and errors appropriately.
"""

import pytest
import torch
import numpy as np

from hex_ai.value_utils import (
    apply_move_to_tensor,
    apply_move_to_state,
    apply_move_to_state_trmph,
    apply_move_to_tensor_trmph,
    is_position_empty,
)
from hex_ai.inference.game_engine import HexGameState
from hex_ai.config import BOARD_SIZE, BLUE_PLAYER, RED_PLAYER, BLUE_PIECE, RED_PIECE, EMPTY_ONEHOT, PIECE_ONEHOT


class TestIsPositionEmpty:
    """Test the position emptiness checking utility."""
    
    def test_empty_position(self):
        """Test that empty positions return True."""
        board_tensor = torch.zeros(3, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
        assert is_position_empty(board_tensor, 0, 0) == True
        assert is_position_empty(board_tensor, 5, 5) == True
    
    def test_occupied_position(self):
        """Test that occupied positions return False."""
        board_tensor = torch.zeros(3, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
        board_tensor[0, 0, 0] = PIECE_ONEHOT  # Blue piece
        assert is_position_empty(board_tensor, 0, 0) == False
        
        board_tensor = torch.zeros(3, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
        board_tensor[1, 1, 1] = PIECE_ONEHOT  # Red piece
        assert is_position_empty(board_tensor, 1, 1) == False
    
    def test_invalid_values(self):
        """Test that invalid values raise exceptions."""
        board_tensor = torch.zeros(3, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
        board_tensor[0, 0, 0] = 0.5  # Invalid value
        
        with pytest.raises(ValueError, match="Invalid blue channel value"):
            is_position_empty(board_tensor, 0, 0)
        
        board_tensor = torch.zeros(3, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
        board_tensor[1, 0, 0] = 2.0  # Invalid value
        
        with pytest.raises(ValueError, match="Invalid red channel value"):
            is_position_empty(board_tensor, 0, 0)
    
    def test_floating_point_tolerance(self):
        """Test that small floating-point errors are handled correctly."""
        board_tensor = torch.zeros(3, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
        
        # Test with small floating-point errors (within tolerance)
        board_tensor[0, 0, 0] = EMPTY_ONEHOT + 1e-10  # Very small value, should be treated as EMPTY_ONEHOT
        assert is_position_empty(board_tensor, 0, 0) == True
        
        board_tensor[0, 0, 0] = EMPTY_ONEHOT
        board_tensor[1, 0, 0] = EMPTY_ONEHOT + 1e-10  # Very small value, should be treated as EMPTY_ONEHOT
        assert is_position_empty(board_tensor, 0, 0) == True
        
        # Test with values just outside tolerance
        board_tensor[0, 0, 0] = 1e-8  # Just outside default tolerance
        with pytest.raises(ValueError, match="Invalid blue channel value"):
            is_position_empty(board_tensor, 0, 0)
        
        # Test with custom tolerance
        assert is_position_empty(board_tensor, 0, 0, tolerance=1e-7) == True
    
    def test_occupied_with_floating_point_errors(self):
        """Test that occupied positions with small errors are still detected."""
        board_tensor = torch.zeros(3, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
        
        # Test with PIECE_ONEHOT + small error
        board_tensor[0, 0, 0] = PIECE_ONEHOT + 1e-10  # Should be treated as PIECE_ONEHOT
        assert is_position_empty(board_tensor, 0, 0) == False
        
        board_tensor[0, 0, 0] = EMPTY_ONEHOT
        board_tensor[1, 0, 0] = PIECE_ONEHOT - 1e-10  # Should be treated as PIECE_ONEHOT
        assert is_position_empty(board_tensor, 0, 0) == False
    
    def test_out_of_bounds(self):
        """Test that out-of-bounds positions raise exceptions."""
        board_tensor = torch.zeros(3, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
        
        with pytest.raises(IndexError, match="out of bounds"):
            is_position_empty(board_tensor, -1, 0)
        
        with pytest.raises(IndexError, match="out of bounds"):
            is_position_empty(board_tensor, BOARD_SIZE, 0)


class TestApplyMoveToTensor:
    """Test the core tensor-based move application function."""
    
    def test_apply_blue_move_to_empty_tensor(self):
        """Test applying a blue move to an empty 3-channel tensor."""
        # Create empty 3-channel tensor
        board_tensor = torch.zeros(3, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
        # Set player-to-move to blue
        board_tensor[2, :, :] = float(BLUE_PLAYER)
        
        # Apply blue move at (0, 0)
        new_tensor = apply_move_to_tensor(board_tensor, 0, 0, BLUE_PLAYER)
        
        # Check blue piece was placed
        assert new_tensor[0, 0, 0] == 1.0  # Blue channel
        assert new_tensor[1, 0, 0] == 0.0  # Red channel should be empty
        # Check player-to-move switched to red
        assert new_tensor[2, 0, 0] == float(RED_PLAYER)
        assert torch.all(new_tensor[2, :, :] == float(RED_PLAYER))
    
    def test_apply_red_move_to_empty_tensor(self):
        """Test applying a red move to an empty 3-channel tensor."""
        # Create empty 3-channel tensor
        board_tensor = torch.zeros(3, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
        # Set player-to-move to red
        board_tensor[2, :, :] = float(RED_PLAYER)
        
        # Apply red move at (1, 1)
        new_tensor = apply_move_to_tensor(board_tensor, 1, 1, RED_PLAYER)
        
        # Check red piece was placed
        assert new_tensor[0, 1, 1] == 0.0  # Blue channel should be empty
        assert new_tensor[1, 1, 1] == 1.0  # Red channel
        # Check player-to-move switched to blue
        assert new_tensor[2, 1, 1] == float(BLUE_PLAYER)
        assert torch.all(new_tensor[2, :, :] == float(BLUE_PLAYER))
    
    def test_apply_move_to_occupied_position(self):
        """Test that applying a move to an occupied position raises an error."""
        # Create tensor with blue piece at (0, 0)
        board_tensor = torch.zeros(3, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
        board_tensor[0, 0, 0] = 1.0  # Blue piece
        board_tensor[2, :, :] = float(RED_PLAYER)  # Red to move
        
        # Try to place red piece at same position
        with pytest.raises(ValueError, match="already occupied"):
            apply_move_to_tensor(board_tensor, 0, 0, RED_PLAYER)
    
    def test_apply_move_out_of_bounds(self):
        """Test that applying a move out of bounds raises an error."""
        board_tensor = torch.zeros(3, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
        
        # Test negative coordinates
        with pytest.raises(IndexError, match="out of bounds"):
            apply_move_to_tensor(board_tensor, -1, 0, BLUE_PLAYER)
        
        # Test coordinates beyond board size
        with pytest.raises(IndexError, match="out of bounds"):
            apply_move_to_tensor(board_tensor, BOARD_SIZE, 0, BLUE_PLAYER)
    
    def test_invalid_player(self):
        """Test that using an invalid player raises an error."""
        board_tensor = torch.zeros(3, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
        
        with pytest.raises(ValueError, match="Invalid player"):
            apply_move_to_tensor(board_tensor, 0, 0, 2)  # Invalid player ID
    
    def test_invalid_tensor_shape(self):
        """Test that using wrong tensor shape raises an error."""
        # Create 2-channel tensor instead of 3-channel
        board_tensor = torch.zeros(2, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
        
        with pytest.raises(ValueError, match="Expected tensor shape"):
            apply_move_to_tensor(board_tensor, 0, 0, BLUE_PLAYER)
    
    def test_tensor_immutability(self):
        """Test that the original tensor is not modified."""
        board_tensor = torch.zeros(3, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
        board_tensor[2, :, :] = float(BLUE_PLAYER)
        
        original_tensor = board_tensor.clone()
        apply_move_to_tensor(board_tensor, 0, 0, BLUE_PLAYER)
        
        # Original tensor should be unchanged
        assert torch.allclose(board_tensor, original_tensor)


class TestApplyMoveToState:
    """Test the game state move application function."""
    
    def test_apply_valid_move(self):
        """Test applying a valid move to a game state."""
        state = HexGameState()
        
        # Apply blue move
        new_state = apply_move_to_state(state, 0, 0)
        
        # Check the move was applied
        assert new_state.board[0, 0] == BLUE_PIECE
        assert new_state.current_player == RED_PLAYER
        assert new_state.move_history == [(0, 0)]
    
    def test_apply_invalid_move(self):
        """Test that applying an invalid move raises an error."""
        state = HexGameState()
        
        # Apply first move
        state = apply_move_to_state(state, 0, 0)
        
        # Try to apply move to same position
        with pytest.raises(ValueError, match="Invalid move"):
            apply_move_to_state(state, 0, 0)
    
    def test_apply_move_to_finished_game(self):
        """Test that applying move to finished game raises an error."""
        # Create a state with many moves (simulating near-finished game)
        state = HexGameState()
        for i in range(BOARD_SIZE * BOARD_SIZE - 1):  # Fill almost entire board
            row = i // BOARD_SIZE
            col = i % BOARD_SIZE
            if state.is_valid_move(row, col):
                state = apply_move_to_state(state, row, col)
        
        # Try to apply move when game is over
        if state.game_over:
            with pytest.raises(ValueError, match="Invalid move"):
                apply_move_to_state(state, 0, 0)


class TestApplyMoveToStateTrmph:
    """Test the TRMPH wrapper for game state move application."""
    
    def test_apply_valid_trmph_move(self):
        """Test applying a valid TRMPH move."""
        state = HexGameState()
        
        # Apply blue move using TRMPH
        new_state = apply_move_to_state_trmph(state, "a1")  # (0, 0)
        
        # Check the move was applied
        assert new_state.board[0, 0] == BLUE_PIECE
        assert new_state.current_player == RED_PLAYER
        assert new_state.move_history == [(0, 0)]
    
    def test_apply_invalid_trmph_move(self):
        """Test that applying an invalid TRMPH move raises an error."""
        state = HexGameState()
        
        with pytest.raises(ValueError, match="Invalid TRMPH move"):
            apply_move_to_state_trmph(state, "invalid")
    
    def test_apply_trmph_move_to_occupied_position(self):
        """Test that applying TRMPH move to occupied position raises an error."""
        state = HexGameState()
        
        # Apply first move
        state = apply_move_to_state_trmph(state, "a1")
        
        # Try to apply move to same position
        with pytest.raises(ValueError, match="Invalid move"):
            apply_move_to_state_trmph(state, "a1")


class TestApplyMoveToTensorTrmph:
    """Test the TRMPH wrapper for tensor move application."""
    
    def test_apply_valid_trmph_move_to_tensor(self):
        """Test applying a valid TRMPH move to a tensor."""
        board_tensor = torch.zeros(3, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
        board_tensor[2, :, :] = float(BLUE_PLAYER)
        
        # Apply blue move using TRMPH
        new_tensor = apply_move_to_tensor_trmph(board_tensor, "a1", BLUE_PLAYER)
        
        # Check the move was applied
        assert new_tensor[0, 0, 0] == 1.0  # Blue channel
        assert new_tensor[2, 0, 0] == float(RED_PLAYER)  # Player switched
    
    def test_apply_invalid_trmph_move_to_tensor(self):
        """Test that applying an invalid TRMPH move to tensor raises an error."""
        board_tensor = torch.zeros(3, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
        
        with pytest.raises(ValueError, match="Invalid TRMPH move"):
            apply_move_to_tensor_trmph(board_tensor, "invalid", BLUE_PLAYER)


class TestIntegration:
    """Integration tests to verify consistency between different approaches."""
    
    def test_tensor_and_state_consistency(self):
        """Test that tensor and state approaches produce consistent results."""
        # Start with empty state
        state = HexGameState()
        
        # Apply move using state approach
        new_state = apply_move_to_state(state, 0, 0)
        
        # Convert original state to tensor
        from hex_ai.utils.format_conversion import board_nxn_to_3nxn
        board_tensor = board_nxn_to_3nxn(state.board)
        
        # Apply same move using tensor approach
        new_tensor = apply_move_to_tensor(board_tensor, 0, 0, BLUE_PLAYER)
        
        # Convert new state to tensor for comparison
        new_state_tensor = board_nxn_to_3nxn(new_state.board)
        
        # The tensors should be equivalent (ignoring player-to-move channel differences)
        # since the state approach handles player-to-move differently
        assert torch.allclose(new_tensor[:2], new_state_tensor[:2])
    
    def test_trmph_consistency(self):
        """Test that TRMPH and direct approaches produce consistent results."""
        state = HexGameState()
        
        # Apply move using direct coordinates
        state_direct = apply_move_to_state(state, 0, 0)
        
        # Apply same move using TRMPH
        state_trmph = apply_move_to_state_trmph(state, "a1")
        
        # Results should be identical
        assert np.array_equal(state_direct.board, state_trmph.board)
        assert state_direct.current_player == state_trmph.current_player
        assert state_direct.move_history == state_trmph.move_history


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_sequence_of_moves(self):
        """Test applying a sequence of moves to verify player alternation."""
        state = HexGameState()
        
        # Apply several moves
        moves = [(0, 0), (1, 1), (2, 2), (3, 3)]
        for i, (row, col) in enumerate(moves):
            state = apply_move_to_state(state, row, col)
            expected_player = RED_PLAYER if i % 2 == 0 else BLUE_PLAYER
            assert state.current_player == expected_player
    
    def test_tensor_sequence_of_moves(self):
        """Test applying a sequence of moves to tensors."""
        board_tensor = torch.zeros(3, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
        board_tensor[2, :, :] = float(BLUE_PLAYER)
        
        # Apply several moves
        moves = [(0, 0), (1, 1), (2, 2)]
        players = [BLUE_PLAYER, RED_PLAYER, BLUE_PLAYER]
        
        for (row, col), player in zip(moves, players):
            board_tensor = apply_move_to_tensor(board_tensor, row, col, player)
            
            # Check piece was placed correctly
            if player == BLUE_PLAYER:
                assert board_tensor[0, row, col] == 1.0
                assert board_tensor[1, row, col] == 0.0
            else:
                assert board_tensor[0, row, col] == 0.0
                assert board_tensor[1, row, col] == 1.0 