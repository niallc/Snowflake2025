from enum import Enum
from hex_ai.config import TRMPH_BLUE_WIN, TRMPH_RED_WIN, TRAINING_BLUE_WIN, TRAINING_RED_WIN
import torch
import numpy as np

# =============================
# Winner Mapping Utilities
# =============================
def trmph_winner_to_training_value(trmph_winner: str) -> float:
    """
    Map TRMPH winner annotation ("1" or "2") to training value (0.0 or 1.0).
    """
    if trmph_winner == TRMPH_BLUE_WIN:
        return TRAINING_BLUE_WIN
    elif trmph_winner == TRMPH_RED_WIN:
        return TRAINING_RED_WIN
    else:
        raise ValueError(f"Invalid TRMPH winner: {trmph_winner}")

def training_value_to_trmph_winner(training_value: float) -> str:
    """
    Map training value (0.0 or 1.0) to TRMPH winner annotation ("1" or "2").
    """
    if training_value == TRAINING_BLUE_WIN:
        return TRMPH_BLUE_WIN
    elif training_value == TRAINING_RED_WIN:
        return TRMPH_RED_WIN
    else:
        raise ValueError(f"Invalid training value: {training_value}")

def trmph_winner_to_clear_str(trmph_winner: str) -> str:
    """
    Map TRMPH winner annotation ("1" or "2") to clear string ("BLUE" or "RED").
    """
    if trmph_winner == TRMPH_BLUE_WIN:
        return "BLUE"
    elif trmph_winner == TRMPH_RED_WIN:
        return "RED"
    else:
        raise ValueError(f"Invalid TRMPH winner: {trmph_winner}") 
    
# --- Enums for clarity ---
class Winner(Enum):
    BLUE = 0
    RED = 1

class ValuePerspective(Enum):
    TRAINING_TARGET = 0  # 0.0 = Blue win, 1.0 = Red win
    BLUE_WIN_PROB = 1    # Probability Blue wins
    RED_WIN_PROB = 2     # Probability Red wins

# --- Conversion functions ---
def value_target_from_winner(winner: Winner) -> float:
    """Convert Winner to training value target (0.0 for Blue, 1.0 for Red)."""
    if winner == Winner.BLUE:
        return 0.0
    elif winner == Winner.RED:
        return 1.0
    else:
        raise ValueError(f"Unknown winner: {winner}")

def winner_from_value_target(value: float) -> Winner:
    """Convert training value target to Winner."""
    if value == 0.0:
        return Winner.BLUE
    elif value == 1.0:
        return Winner.RED
    else:
        raise ValueError(f"Invalid value target: {value}")

def model_output_to_prob(model_output: float, perspective: ValuePerspective) -> float:
    """
    Convert model output (sigmoid(logit), trained with 0.0=Blue win, 1.0=Red win)
    to probability for the given perspective.
    """
    if perspective == ValuePerspective.TRAINING_TARGET:
        return model_output
    elif perspective == ValuePerspective.BLUE_WIN_PROB:
        return 1.0 - model_output
    elif perspective == ValuePerspective.RED_WIN_PROB:
        return model_output
    else:
        raise ValueError(f"Unknown perspective: {perspective}")

def prob_to_model_output(prob: float, perspective: ValuePerspective) -> float:
    """
    Convert a probability for a given perspective to the model output convention (probability Red wins).
    """
    if perspective == ValuePerspective.TRAINING_TARGET:
        return prob
    elif perspective == ValuePerspective.BLUE_WIN_PROB:
        return 1.0 - prob
    elif perspective == ValuePerspective.RED_WIN_PROB:
        return prob
    else:
        raise ValueError(f"Unknown perspective: {perspective}") 

def get_win_prob_from_model_output(model_output: float, player) -> float:
    """
    Given the raw model output (logit) and a player ('blue', 'red', or Winner),
    return the probability that player wins (according to the model's convention).
    """
    from hex_ai.value_utils import ValuePerspective
    if isinstance(player, str):
        player = player.lower()
        if player == 'blue':
            perspective = ValuePerspective.BLUE_WIN_PROB
        elif player == 'red':
            perspective = ValuePerspective.RED_WIN_PROB
        else:
            raise ValueError(f"Unknown player: {player}")
    elif hasattr(player, 'name') and player.name in ('BLUE', 'RED'):
        perspective = ValuePerspective.BLUE_WIN_PROB if player.name == 'BLUE' else ValuePerspective.RED_WIN_PROB
    else:
        raise ValueError(f"Unknown player: {player}")
    prob_red_win = torch.sigmoid(torch.tensor(model_output)).item()
    from hex_ai.value_utils import model_output_to_prob
    return model_output_to_prob(prob_red_win, perspective)

def get_policy_probs_from_logits(policy_logits) -> np.ndarray:
    """
    Given raw policy logits (numpy array or torch tensor), return softmaxed probabilities as a numpy array.
    """
    import torch
    import numpy as np
    if not isinstance(policy_logits, np.ndarray):
        policy_logits = policy_logits.detach().cpu().numpy() if hasattr(policy_logits, 'detach') else np.array(policy_logits)
    policy_probs = torch.softmax(torch.tensor(policy_logits), dim=0).numpy()
    return policy_probs

def get_player_to_move_from_moves(moves: list) -> int:
    """
    Given a list of moves (e.g., ['a1', 'b2', ...]), return BLUE_PLAYER if it's blue's move, RED_PLAYER if it's red's move.
    Blue always starts, so even number of moves = Blue's turn, odd = Red's turn.
    """
    from hex_ai.config import BLUE_PLAYER, RED_PLAYER
    if len(moves) % 2 == 0:
        return BLUE_PLAYER
    else:
        return RED_PLAYER 

def winner_to_color(winner):
    """Map Winner enum or player int to color name string ('blue', 'red', or 'reset')."""
    if isinstance(winner, Winner):
        return 'blue' if winner == Winner.BLUE else 'red'
    from hex_ai.config import BLUE_PLAYER, RED_PLAYER
    if winner == BLUE_PLAYER:
        return 'blue'
    elif winner == RED_PLAYER:
        return 'red'
    else:
        return 'reset' 