from enum import Enum
from hex_ai.config import TRMPH_BLUE_WIN, TRMPH_RED_WIN, TRAINING_BLUE_WIN, TRAINING_RED_WIN
import torch
import numpy as np
from typing import List, Tuple

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

def temperature_scaled_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Apply temperature scaling to logits and return softmaxed probabilities.

    Args:
        logits: Raw logits (numpy array)
        temperature: Temperature parameter (higher = more random, lower = more deterministic)

    Returns:
        Temperature-scaled softmax probabilities

    Note:
        - temperature = 1.0: Standard softmax
        - temperature < 1.0: More deterministic (sharper distribution)
        - temperature > 1.0: More random (flatter distribution)
        - temperature = 0.0: Greedy selection (argmax)
    """
    import torch
    import numpy as np

    if temperature <= 0:
        # Greedy selection: return one-hot vector for argmax
        result = np.zeros_like(logits)
        result[np.argmax(logits)] = 1.0
        return result

    # Apply temperature scaling: logits / temperature
    scaled_logits = logits / temperature
    # Apply softmax
    probs = torch.softmax(torch.tensor(scaled_logits), dim=0).numpy()
    return probs

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

# =============================
# Policy Processing & Move Selection Utilities
# =============================

def policy_logits_to_probs(policy_logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Convert policy logits to probabilities using temperature-scaled softmax.

    Args:
        policy_logits: Raw policy logits (numpy array)
        temperature: Temperature parameter (default 1.0)

    Returns:
        Temperature-scaled softmax probabilities
    """
    return temperature_scaled_softmax(policy_logits, temperature)

def get_legal_policy_probs(policy_probs: np.ndarray, legal_moves: List[Tuple[int, int]], board_size: int) -> np.ndarray:
    """
    Extract policy probabilities for legal moves only.
    Raises ValueError if legal_moves is empty.
    """
    if not legal_moves:
        raise ValueError("No legal moves provided to get_legal_policy_probs.")
    move_indices = [row * board_size + col for row, col in legal_moves]
    legal_policy = np.array([policy_probs[idx] for idx in move_indices])
    return legal_policy

def select_top_k_moves(legal_policy: np.ndarray, legal_moves: List[Tuple[int, int]], k: int) -> List[Tuple[int, int]]:
    """
    Select top-k moves based on policy probabilities.
    Raises ValueError if legal_policy or legal_moves is empty.
    """
    if len(legal_policy) == 0 or len(legal_moves) == 0:
        raise ValueError("No legal moves available for top-k selection.")
    topk_idx = np.argsort(legal_policy)[::-1][:k]
    return [legal_moves[i] for i in topk_idx]

def sample_move_by_value(move_values: List[float], temperature: float = 1.0) -> int:
    """
    Sample a move index based on value logits using temperature scaling.

    Args:
        move_values: List of value logits for each move
        temperature: Temperature parameter for sampling (default 1.0)

    Returns:
        Index of the selected move
    """
    if len(move_values) == 0:
        raise ValueError("No moves to sample from")
    elif len(move_values) == 1:
        return 0

    # Convert value logits to probabilities using temperature scaling
    probs = temperature_scaled_softmax(np.array(move_values), temperature)
    chosen_idx = np.random.choice(len(move_values), p=probs)
    return chosen_idx

def get_top_k_moves_with_probs(policy_logits: np.ndarray,
                               legal_moves: List[Tuple[int, int]],
                               board_size: int, k: int, temperature: float = 1.0) -> List[Tuple[Tuple[int, int], float]]:
    """
    Get top-k moves with their probabilities, using centralized policy processing.
    
    Args:
        policy_logits: Raw policy logits
        legal_moves: List of (row, col) tuples for legal moves
        board_size: Board size
        k: Number of top moves to return
        temperature: Temperature for policy sampling
        
    Returns:
        List of ((row, col), probability) tuples for top-k moves
    """
    # Convert logits to probabilities
    policy_probs = policy_logits_to_probs(policy_logits, temperature)
    
    # Get legal move probabilities
    legal_policy = get_legal_policy_probs(policy_probs, legal_moves, board_size)
    
    # Select top-k moves
    topk_moves = select_top_k_moves(legal_policy, legal_moves, k)
    
    # Get corresponding probabilities
    move_indices = [row * board_size + col for row, col in topk_moves]
    move_probs = [float(policy_probs[idx]) for idx in move_indices]
    
    return list(zip(topk_moves, move_probs))

def select_policy_move(state, model, temperature: float = 1.0) -> Tuple[int, int]:
    """
    Select a move using policy head with centralized utilities.
    
    Args:
        state: Game state (must have .board and .get_legal_moves() methods)
        model: Model instance (must have .infer() method that returns (policy_logits, value_logit))
        temperature: Temperature for policy sampling (default 1.0)
        
    Returns:
        (row, col) tuple for the selected move
        
    Raises:
        ValueError: If no legal moves are available
    """
    policy_logits, _ = model.infer(state.board)
    
    # Use centralized utilities for policy processing
    policy_probs = policy_logits_to_probs(policy_logits, temperature)
    legal_moves = state.get_legal_moves()
    legal_policy = get_legal_policy_probs(policy_probs, legal_moves, state.board.shape[0])
    
    # Select the best move (top-1)
    best_moves = select_top_k_moves(legal_policy, legal_moves, 1)
    return best_moves[0] 