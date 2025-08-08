"""
Unit tests for MCTS module.

To run these tests, use:
    PYTHONPATH=. python -m pytest tests/test_mcts.py -v

Or:
    PYTHONPATH=. python tests/test_mcts.py
"""

import pytest
import numpy as np
import copy
from unittest.mock import Mock, patch

try:
    from hex_ai.inference.mcts import MCTSNode, NeuralMCTS
    from hex_ai.inference.game_engine import HexGameState
    from hex_ai.inference.simple_model_inference import SimpleModelInference
    from hex_ai.config import BLUE_PLAYER, RED_PLAYER
except ImportError as e:
    print(f"ERROR: Could not import MCTS modules: {e}")
    print("Make sure to run with PYTHONPATH=.")
    print("Example: PYTHONPATH=. python -m pytest tests/test_mcts.py -v")
    raise


class TestMCTSNode:
    """Test cases for the MCTSNode class."""
    
    def test_node_initialization(self):
        """Test basic node initialization."""
        state = HexGameState()
        node = MCTSNode(state=state)
        
        assert node.state == state
        assert node.parent is None
        assert node.move is None
        assert node.visits == 0
        assert node.total_value == 0.0
        assert node.policy_priors is None
        assert len(node.children) == 0
    
    def test_node_with_parent_and_move(self):
        """Test node initialization with parent and move."""
        parent_state = HexGameState()
        child_state = HexGameState()
        parent = MCTSNode(state=parent_state)
        move = (6, 6)
        
        child = MCTSNode(state=child_state, parent=parent, move=move)
        
        assert child.parent == parent
        assert child.move == move
    
    def test_mean_value_calculation(self):
        """Test mean value calculation."""
        state = HexGameState()
        node = MCTSNode(state=state)
        
        # Initially should be 0
        assert node.mean_value == 0.0
        
        # After adding values
        node.visits = 3
        node.total_value = 1.5
        assert node.mean_value == 0.5
        
        # After more values
        node.visits = 5
        node.total_value = 2.5
        assert node.mean_value == 0.5
    
    def test_is_leaf(self):
        """Test leaf node detection."""
        state = HexGameState()
        node = MCTSNode(state=state)
        
        # Initially should be a leaf
        assert node.is_leaf()
        
        # After adding children, should not be a leaf
        child_state = HexGameState()
        node.children[(6, 6)] = MCTSNode(state=child_state)
        assert not node.is_leaf()
    
    def test_is_terminal(self):
        """Test terminal node detection."""
        state = HexGameState()
        node = MCTSNode(state=state)
        
        # Initially should not be terminal
        assert not node.is_terminal()
        
        # Make it terminal
        state.game_over = True
        state.winner = "blue"
        assert node.is_terminal()
    
    def test_update_statistics(self):
        """Test statistics update."""
        state = HexGameState()
        node = MCTSNode(state=state)
        
        # Update with a value
        node.update_statistics(1.0)
        assert node.visits == 1
        assert node.total_value == 1.0
        assert node.mean_value == 1.0
        
        # Update with another value
        node.update_statistics(-0.5)
        assert node.visits == 2
        assert node.total_value == 0.5
        assert node.mean_value == 0.25


class TestNeuralMCTS:
    """Test cases for the NeuralMCTS class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock model
        self.mock_model = Mock(spec=SimpleModelInference)
        
        # Create MCTS engine
        self.mcts = NeuralMCTS(model=self.mock_model, exploration_constant=1.4)
    
    def test_initialization(self):
        """Test MCTS engine initialization."""
        assert self.mcts.model == self.mock_model
        assert self.mcts.exploration_constant == 1.4
        assert self.mcts.win_value == 1.5  # Default value
        assert self.mcts.discount_factor == 0.98  # Default value
        assert self.mcts.stats['total_simulations'] == 0
        assert self.mcts.stats['total_inferences'] == 0
    
    def test_initialization_custom_win_value(self):
        """Test MCTS engine initialization with custom win value."""
        mcts_custom = NeuralMCTS(model=self.mock_model, win_value=2.0)
        assert mcts_custom.win_value == 2.0
    
    def test_initialization_custom_discount_factor(self):
        """Test MCTS engine initialization with custom discount factor."""
        mcts_custom = NeuralMCTS(model=self.mock_model, discount_factor=0.95)
        assert mcts_custom.discount_factor == 0.95
    
    def test_discount_factor_effect(self):
        """Test that discount factor affects backpropagation correctly."""
        # Create a simple tree structure
        root_state = HexGameState()
        root = MCTSNode(state=root_state, depth=0)
        
        # Create a child node
        child_state = root_state.make_move(6, 6)
        child = MCTSNode(state=child_state, parent=root, move=(6, 6), depth=1)
        root.children[(6, 6)] = child
        
        # Test with different discount factors
        test_cases = [
            (1.0, 1.0),  # No discount
            (0.5, 0.5),  # 50% discount
            (0.8, 0.8),  # 20% discount
        ]
        
        for discount_factor, expected_value in test_cases:
            # Create MCTS with this discount factor
            mcts = NeuralMCTS(model=self.mock_model, discount_factor=discount_factor)
            
            # Reset child state
            child.total_value = 0.0
            child.visits = 0
            
            # Backpropagate
            mcts._backpropagate(child, 1.0)
            
            # Verify the discount was applied correctly
            expected_discounted_value = 1.0 * (discount_factor ** 1)  # depth 1
            assert abs(child.total_value - expected_discounted_value) < 1e-10, \
                f"Discount factor {discount_factor} failed: expected {expected_discounted_value}, got {child.total_value}"
    
    def test_terminal_value_blue_win(self):
        """Test terminal value calculation for Blue win."""
        state = HexGameState()
        state.game_over = True
        state.winner = "blue"
        state.current_player = BLUE_PLAYER  # Blue's turn, so Red just moved
        
        # Value should be from perspective of player who just moved (Red)
        # Since Blue won, Red lost, so value should be -win_value
        value = self.mcts._terminal_value(state)
        assert value == -self.mcts.win_value
    
    def test_terminal_value_red_win(self):
        """Test terminal value calculation for Red win."""
        state = HexGameState()
        state.game_over = True
        state.winner = "red"
        state.current_player = RED_PLAYER  # Red's turn, so Blue just moved
        
        # Value should be from perspective of player who just moved (Blue)
        # Since Red won, Blue lost, so value should be -win_value
        value = self.mcts._terminal_value(state)
        assert value == -self.mcts.win_value
    
    def test_terminal_value_draw(self):
        """Test terminal value calculation for draw."""
        state = HexGameState()
        state.game_over = True
        state.winner = None  # Draw
        
        value = self.mcts._terminal_value(state)
        assert value == 0.0
    
    def test_terminal_value_non_terminal_error(self):
        """Test that terminal value raises error for non-terminal state."""
        state = HexGameState()
        state.game_over = False
        
        with pytest.raises(ValueError, match="Cannot get terminal value for non-terminal state"):
            self.mcts._terminal_value(state)
    
    def test_get_priors_for_legal_moves(self):
        """Test prior probability extraction for legal moves."""
        # Create mock policy logits (1D array, flattened from 13x13 board)
        policy_logits = np.zeros(13 * 13)  # 169 elements
        policy_logits[6 * 13 + 6] = 2.0  # High probability for center (6,6)
        policy_logits[0 * 13 + 0] = 1.0  # Medium probability for corner (0,0)
        
        legal_moves = [(6, 6), (0, 0), (12, 12)]
        
        priors = self.mcts._get_priors_for_legal_moves(policy_logits, legal_moves)
        
        # Should have priors for all legal moves
        assert len(priors) == 3
        assert (6, 6) in priors
        assert (0, 0) in priors
        assert (12, 12) in priors
        
        # Probabilities should sum to 1
        total_prob = sum(priors.values())
        assert abs(total_prob - 1.0) < 1e-10
        
        # Center should have highest probability
        assert priors[(6, 6)] > priors[(0, 0)]
        assert priors[(6, 6)] > priors[(12, 12)]
    
    def test_get_priors_zero_logits(self):
        """Test prior extraction when all logits are zero."""
        policy_logits = np.zeros(13 * 13)  # 1D array, 169 elements
        legal_moves = [(6, 6), (0, 0), (12, 12)]
        
        priors = self.mcts._get_priors_for_legal_moves(policy_logits, legal_moves)
        
        # Should use uniform distribution
        expected_prior = 1.0 / 3
        for move in legal_moves:
            assert abs(priors[move] - expected_prior) < 1e-10
    
    def test_temperature_scale_deterministic(self):
        """Test temperature scaling with temperature 0 (deterministic)."""
        visit_counts = [10, 5, 3]
        
        probabilities = self.mcts._temperature_scale(visit_counts, temperature=0)
        
        # Should put all probability on most visited
        assert probabilities[0] == 1.0
        assert probabilities[1] == 0.0
        assert probabilities[2] == 0.0
    
    def test_temperature_scale_stochastic(self):
        """Test temperature scaling with temperature 1 (stochastic)."""
        visit_counts = [10, 5, 3]
        
        probabilities = self.mcts._temperature_scale(visit_counts, temperature=1.0)
        
        # Should preserve relative proportions
        assert probabilities[0] > probabilities[1] > probabilities[2]
        assert abs(sum(probabilities) - 1.0) < 1e-10
    
    def test_temperature_scale_high_temperature(self):
        """Test temperature scaling with high temperature (more uniform)."""
        visit_counts = [10, 5, 3]
        
        probabilities = self.mcts._temperature_scale(visit_counts, temperature=2.0)
        
        # Should be more uniform than temperature=1.0
        high_temp_probs = self.mcts._temperature_scale(visit_counts, temperature=2.0)
        low_temp_probs = self.mcts._temperature_scale(visit_counts, temperature=1.0)
        
        # High temperature should be more uniform
        assert abs(high_temp_probs[0] - high_temp_probs[1]) < abs(low_temp_probs[0] - low_temp_probs[1])


class TestMCTSBackpropagation:
    """Test cases for MCTS backpropagation logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_model = Mock(spec=SimpleModelInference)
        self.mcts = NeuralMCTS(model=self.mock_model)
    
    def test_backpropagation_simple_tree(self):
        """Test backpropagation in a simple tree."""
        # Create a simple tree: root -> child1 -> child2
        root_state = HexGameState()
        child1_state = HexGameState()
        child2_state = HexGameState()
        
        root = MCTSNode(state=root_state, depth=0)
        child1 = MCTSNode(state=child1_state, parent=root, move=(6, 6), depth=1)
        child2 = MCTSNode(state=child2_state, parent=child1, move=(0, 0), depth=2)
        
        root.children[(6, 6)] = child1
        child1.children[(0, 0)] = child2
        
        # Set initial values
        root.visits = 5
        root.total_value = 2.0
        child1.visits = 3
        child1.total_value = 1.0
        child2.visits = 1
        child2.total_value = 0.5
        
        # Store initial values for comparison
        initial_root_value = root.total_value
        initial_child1_value = child1.total_value
        initial_child2_value = child2.total_value
        
        # Backpropagate a value from child2
        backprop_value = 1.0
        self.mcts._backpropagate(child2, backprop_value)
        
        # Check that all nodes were updated
        assert child2.visits == 2
        assert child1.visits == 4
        assert root.visits == 6
        
        # Check that values were updated (without hardcoding specific numbers)
        assert child2.total_value != initial_child2_value
        assert child1.total_value != initial_child1_value
        assert root.total_value != initial_root_value
        
        # Check that the discount factor was applied correctly
        # child2 should get the backprop value with discount applied
        expected_child2_discount = backprop_value * (self.mcts.discount_factor ** 2)
        assert abs(child2.total_value - (initial_child2_value + expected_child2_discount)) < 1e-10
        
        # child1 should get the negated backprop value with discount applied
        expected_child1_discount = -backprop_value * (self.mcts.discount_factor ** 1)
        assert abs(child1.total_value - (initial_child1_value + expected_child1_discount)) < 1e-10
        
        # root should get the backprop value with discount applied
        expected_root_discount = backprop_value * (self.mcts.discount_factor ** 0)
        assert abs(root.total_value - (initial_root_value + expected_root_discount)) < 1e-10
    
    def test_backpropagation_value_negation(self):
        """Test that values are correctly negated during backpropagation."""
        # Create a simple tree: root -> child
        root_state = HexGameState()
        child_state = HexGameState()
        
        root = MCTSNode(state=root_state, depth=0)
        child = MCTSNode(state=child_state, parent=root, move=(6, 6), depth=1)
        root.children[(6, 6)] = child
        
        # Store initial values
        initial_root_value = root.total_value
        initial_child_value = child.total_value
        
        # Backpropagate a positive value
        backprop_value = 1.0
        self.mcts._backpropagate(child, backprop_value)
        
        # Check that visits were incremented
        assert child.visits == 1
        assert root.visits == 1
        
        # Child should get the positive value with discount factor applied
        expected_child_value = backprop_value * (self.mcts.discount_factor ** 1)  # depth 1
        assert abs(child.total_value - expected_child_value) < 1e-10
        
        # Root should get the negated value with discount factor applied
        expected_root_value = -backprop_value * (self.mcts.discount_factor ** 0)  # depth 0
        assert abs(root.total_value - expected_root_value) < 1e-10
        
        # Verify that values changed from initial state
        assert child.total_value != initial_child_value
        assert root.total_value != initial_root_value


class TestMCTSStateIntegrity:
    """Test cases for state integrity during MCTS."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_model = Mock(spec=SimpleModelInference)
        self.mcts = NeuralMCTS(model=self.mock_model)
    
    def test_state_deep_copy_integrity(self):
        """Test that state modifications don't affect parent/siblings."""
        # Create initial state
        root_state = HexGameState()
        
        # Create a child state
        child_state = copy.deepcopy(root_state)
        child_state = child_state.make_move(6, 6)
        
        # Verify they are different
        assert root_state.move_history != child_state.move_history
        
        # Modify child state
        original_child_moves = child_state.move_history.copy()
        child_state = child_state.make_move(0, 0)
        
        # Parent should be unchanged
        assert root_state.move_history == []
        
        # Child should be changed
        assert len(child_state.move_history) == 2
        assert child_state.move_history[0] == (6, 6)
        assert child_state.move_history[1] == (0, 0)
    
    def test_node_expansion_state_independence(self):
        """Test that expanding a node doesn't affect other nodes."""
        # Create root state
        root_state = HexGameState()
        root = MCTSNode(state=root_state)
        
        # Mock the model to return fixed values
        self.mock_model.simple_infer.return_value = (
            np.ones(13 * 13) * 0.1,  # Policy logits (1D array)
            0.5  # Value
        )
        
        # Expand the root node
        value = self.mcts._expand_and_evaluate(root)
        
        # Check that children were created
        assert len(root.children) > 0
        
        # Check that root state is unchanged
        assert len(root.state.move_history) == 0
        
        # Check that child states are different
        for move, child in root.children.items():
            assert child.state.move_history == [move]
            # Compare move histories instead of full state objects
            assert child.state.move_history != root.state.move_history


class TestMCTSIntegration:
    """Integration tests for MCTS."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_model = Mock(spec=SimpleModelInference)
        self.mcts = NeuralMCTS(model=self.mock_model)
    
    def test_win_in_one_position(self):
        """Test MCTS on a simple 'win in one' position."""
        # Create a position where Blue can win in one move
        # This is a simplified test - in reality we'd need a more complex setup
        
        # Mock the model to return high value for winning moves
        def mock_infer(board_tensor):
            # Return high value for center move (simplified)
            policy = np.ones(13 * 13) * 0.01  # 1D array
            policy[6 * 13 + 6] = 0.5  # High probability for center (6,6)
            return policy, 0.8  # High value
        
        self.mock_model.simple_infer.side_effect = mock_infer
        
        # Create root state
        root_state = HexGameState()
        
        # Run a small number of simulations
        root = self.mcts.search(root_state, num_simulations=10)
        
        # Check that search completed
        assert root.visits == 10
        
        # Check that children were created
        assert len(root.children) > 0
        
        # Check that center move has highest visits (simplified expectation)
        center_visits = root.children.get((6, 6), MCTSNode(HexGameState())).visits
        assert center_visits > 0
    
    def test_move_selection_deterministic(self):
        """Test deterministic move selection."""
        # Create a root with children
        root_state = HexGameState()
        root = MCTSNode(state=root_state)
        
        # Add children with different visit counts
        child1 = MCTSNode(state=HexGameState(), parent=root, move=(6, 6))
        child1.visits = 10
        child1.total_value = 5.0
        
        child2 = MCTSNode(state=HexGameState(), parent=root, move=(0, 0))
        child2.visits = 5
        child2.total_value = 2.0
        
        root.children[(6, 6)] = child1
        root.children[(0, 0)] = child2
        
        # Select move deterministically
        selected_move = self.mcts.select_move(root, temperature=0)
        
        # Should select most visited move
        assert selected_move == (6, 6)
    
    def test_move_selection_stochastic(self):
        """Test stochastic move selection."""
        # Create a root with children
        root_state = HexGameState()
        root = MCTSNode(state=root_state)
        
        # Add children with different visit counts
        child1 = MCTSNode(state=HexGameState(), parent=root, move=(6, 6))
        child1.visits = 10
        child1.total_value = 5.0
        
        child2 = MCTSNode(state=HexGameState(), parent=root, move=(0, 0))
        child2.visits = 5
        child2.total_value = 2.0
        
        root.children[(6, 6)] = child1
        root.children[(0, 0)] = child2
        
        # Select move stochastically
        selected_move = self.mcts.select_move(root, temperature=1.0)
        
        # Should select one of the available moves
        assert selected_move in [(6, 6), (0, 0)]


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"]) 