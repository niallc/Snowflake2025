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