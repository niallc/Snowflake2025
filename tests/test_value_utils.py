import pytest
from hex_ai.value_utils import Winner, ValuePerspective, value_target_from_winner, winner_from_value_target, model_output_to_prob, prob_to_model_output

def test_value_target_from_winner():
    assert value_target_from_winner(Winner.BLUE) == 0.0
    assert value_target_from_winner(Winner.RED) == 1.0
    with pytest.raises(ValueError):
        value_target_from_winner(None)

def test_winner_from_value_target():
    assert winner_from_value_target(0.0) == Winner.BLUE
    assert winner_from_value_target(1.0) == Winner.RED
    with pytest.raises(ValueError):
        winner_from_value_target(0.5)

def test_model_output_to_prob():
    # Model output is probability Red wins
    assert model_output_to_prob(0.0, ValuePerspective.TRAINING_TARGET) == 0.0
    assert model_output_to_prob(1.0, ValuePerspective.TRAINING_TARGET) == 1.0
    assert model_output_to_prob(0.2, ValuePerspective.BLUE_WIN_PROB) == pytest.approx(0.8)
    assert model_output_to_prob(0.2, ValuePerspective.RED_WIN_PROB) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        model_output_to_prob(0.5, None)

def test_prob_to_model_output():
    assert prob_to_model_output(0.0, ValuePerspective.TRAINING_TARGET) == 0.0
    assert prob_to_model_output(1.0, ValuePerspective.TRAINING_TARGET) == 1.0
    assert prob_to_model_output(0.8, ValuePerspective.BLUE_WIN_PROB) == pytest.approx(0.2)
    assert prob_to_model_output(0.2, ValuePerspective.RED_WIN_PROB) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        prob_to_model_output(0.5, None)

def test_model_output_to_prob_range():
    # Simulate a range of logits, convert to prob, then to blue win prob
    import torch
    from hex_ai.value_utils import model_output_to_prob, ValuePerspective
    logits = [-10, -2, 0, 2, 10]
    for logit in logits:
        prob_red_win = torch.sigmoid(torch.tensor(logit)).item()
        prob_blue_win = model_output_to_prob(prob_red_win, ValuePerspective.BLUE_WIN_PROB)
        assert 0.0 <= prob_blue_win <= 1.0, f"Probability out of range: {prob_blue_win} for logit {logit}"

def test_temperature_scaled_softmax():
    """Test temperature scaling utility with various temperature values."""
    import numpy as np
    from hex_ai.value_utils import temperature_scaled_softmax

    # Test logits with clear winner
    logits = np.array([1.0, 0.5, 0.1, 0.0])

        # Test temperature = 0 (greedy selection)
    probs = temperature_scaled_softmax(logits, 0.0)
    assert np.argmax(probs) == 0, "Temperature 0 should select argmax"
    assert np.sum(probs) == pytest.approx(1.0), "Probabilities should sum to 1"
    
    # Test temperature = 1.0 (standard softmax)
    probs = temperature_scaled_softmax(logits, 1.0)
    assert np.sum(probs) == pytest.approx(1.0), "Probabilities should sum to 1"
    assert np.argmax(probs) == 0, "Should still prefer highest logit"
    
    # Test temperature = 0.5 (more deterministic)
    probs_low = temperature_scaled_softmax(logits, 0.5)
    assert np.sum(probs_low) == pytest.approx(1.0), "Probabilities should sum to 1"
    # Lower temperature should make distribution sharper
    assert probs_low[0] > probs[0], "Lower temperature should increase probability of best move"
    
    # Test temperature = 2.0 (more random)
    probs_high = temperature_scaled_softmax(logits, 2.0)
    assert np.sum(probs_high) == pytest.approx(1.0), "Probabilities should sum to 1"
    # Higher temperature should make distribution flatter
    assert probs_high[0] < probs[0], "Higher temperature should decrease probability of best move"
    
    # Test with negative logits
    neg_logits = np.array([-1.0, -2.0, -3.0])
    probs_neg = temperature_scaled_softmax(neg_logits, 1.0)
    assert np.sum(probs_neg) == pytest.approx(1.0), "Probabilities should sum to 1"
    assert np.argmax(probs_neg) == 0, "Should prefer least negative logit"

def test_temperature_scaling_behavior():
    """Test that temperature scaling actually changes the distribution as expected."""
    import numpy as np
    from hex_ai.value_utils import temperature_scaled_softmax
    
    # Test logits with clear winner
    logits = np.array([2.0, 1.0, 0.5, 0.1])
    
    # Get probabilities with different temperatures
    probs_very_low = temperature_scaled_softmax(logits, 0.1)  # Very deterministic
    probs_low = temperature_scaled_softmax(logits, 0.5)       # More deterministic
    probs_standard = temperature_scaled_softmax(logits, 1.0)  # Standard softmax
    probs_high = temperature_scaled_softmax(logits, 2.0)      # More random
    probs_very_high = temperature_scaled_softmax(logits, 5.0) # Very random
    
    # Verify that lower temperature makes the distribution more peaked
    assert probs_very_low[0] > probs_low[0] > probs_standard[0] > probs_high[0] > probs_very_high[0], \
        "Lower temperature should increase probability of best move"
    
    # Verify that higher temperature makes the distribution more uniform
    # The ratio of best to worst should decrease with higher temperature
    ratio_very_low = probs_very_low[0] / probs_very_low[-1]
    ratio_low = probs_low[0] / probs_low[-1]
    ratio_standard = probs_standard[0] / probs_standard[-1]
    ratio_high = probs_high[0] / probs_high[-1]
    ratio_very_high = probs_very_high[0] / probs_very_high[-1]
    
    assert ratio_very_low > ratio_low > ratio_standard > ratio_high > ratio_very_high, \
        "Higher temperature should make distribution more uniform" 