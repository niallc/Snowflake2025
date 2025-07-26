from enum import Enum
from hex_ai.config import (
    TRMPH_BLUE_WIN, TRMPH_RED_WIN,
    TRAINING_BLUE_WIN, TRAINING_RED_WIN,
    BLUE_PLAYER, RED_PLAYER,
    BOARD_SIZE, PIECE_ONEHOT, EMPTY_ONEHOT,
    BLUE_CHANNEL, RED_CHANNEL, PLAYER_CHANNEL
)
from hex_ai.inference.game_engine import HexGameState
from hex_ai.utils.format_conversion import trmph_move_to_rowcol
# Remove self-import - these are defined in this file
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
    return model_output_to_prob(prob_red_win, perspective)

def get_policy_probs_from_logits(policy_logits) -> np.ndarray:
    """
    Given raw policy logits (numpy array or torch tensor), return softmaxed probabilities as a numpy array.
    """
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
    if len(moves) % 2 == 0:
        return BLUE_PLAYER
    else:
        return RED_PLAYER

def winner_to_color(winner):
    """Map Winner enum or player int to color name string ('blue', 'red', or 'reset')."""
    if isinstance(winner, Winner):
        return 'blue' if winner == Winner.BLUE else 'red'
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

# =============================
# Move Application Utilities
# =============================

def is_position_empty(board_tensor: torch.Tensor, row: int, col: int, tolerance: float = 1e-9) -> bool:
    """
    Check if a position is empty in a 3-channel board tensor.
    
    Args:
        board_tensor: torch.Tensor of shape (3, BOARD_SIZE, BOARD_SIZE)
        row: Row index (0-(BOARD_SIZE-1))
        col: Column index (0-(BOARD_SIZE-1))
        tolerance: Floating-point tolerance for comparisons (default: 1e-9)
        
    Returns:
        True if the position is empty (both blue and red channels are approximately EMPTY_ONEHOT)
        
    Raises:
        ValueError: If the position has an invalid value (not approximately EMPTY_ONEHOT or PIECE_ONEHOT)
        IndexError: If coordinates are out of bounds
    """
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise IndexError(f"Position ({row}, {col}) is out of bounds")
    
    blue_val = board_tensor[BLUE_CHANNEL, row, col].item()
    red_val = board_tensor[RED_CHANNEL, row, col].item()
    
    # Check for invalid values (should only be approximately EMPTY_ONEHOT or PIECE_ONEHOT)
    if not (abs(blue_val - EMPTY_ONEHOT) < tolerance or abs(blue_val - PIECE_ONEHOT) < tolerance):
        raise ValueError(f"Invalid blue channel value at ({row}, {col}): {blue_val}")
    if not (abs(red_val - EMPTY_ONEHOT) < tolerance or abs(red_val - PIECE_ONEHOT) < tolerance):
        raise ValueError(f"Invalid red channel value at ({row}, {col}): {red_val}")
    
    # Position is empty if both channels are approximately EMPTY_ONEHOT
    return abs(blue_val - EMPTY_ONEHOT) < tolerance and abs(red_val - EMPTY_ONEHOT) < tolerance


def apply_move_to_tensor(board_tensor: torch.Tensor, row: int, col: int, player: int) -> torch.Tensor:
    """
    Apply a move to a 3-channel board tensor and return the new tensor.
    
    This is the core function that directly manipulates tensors for efficiency.
    The tensor should be in (3, BOARD_SIZE, BOARD_SIZE) format where:
    - channels[BLUE_CHANNEL] = blue pieces (0 or 1)
    - channels[RED_CHANNEL] = red pieces (0 or 1) 
    - channels[PLAYER_CHANNEL] = player-to-move (BLUE_PLAYER or RED_PLAYER)
    
    Args:
        board_tensor: torch.Tensor of shape (3, BOARD_SIZE, BOARD_SIZE)
        row: Row index (0-(BOARD_SIZE-1))
        col: Column index (0-(BOARD_SIZE-1))
        player: Player making the move (BLUE_PLAYER or RED_PLAYER)
        
    Returns:
        New board tensor with the move applied and player-to-move channel updated
        
    Raises:
        ValueError: If position is invalid or already occupied
        IndexError: If coordinates are out of bounds
    """
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise IndexError(f"Position ({row}, {col}) is out of bounds")
    
    if board_tensor.shape != (3, BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"Expected tensor shape (3, {BOARD_SIZE}, {BOARD_SIZE}), got {board_tensor.shape}")
    
    # Check if position is already occupied using utility function
    if not is_position_empty(board_tensor, row, col):
        raise ValueError(f"Position ({row}, {col}) is already occupied")
    
    # Create new tensor (don't modify original)
    new_tensor = board_tensor.clone()
    
    # Place the piece in the appropriate channel using constant
    if player == BLUE_PLAYER:
        new_tensor[BLUE_CHANNEL, row, col] = PIECE_ONEHOT  # Blue channel
    elif player == RED_PLAYER:
        new_tensor[RED_CHANNEL, row, col] = PIECE_ONEHOT  # Red channel
    else:
        raise ValueError(f"Invalid player: {player}")
    
    # Update player-to-move channel (switch to other player)
    next_player = RED_PLAYER if player == BLUE_PLAYER else BLUE_PLAYER
    new_tensor[PLAYER_CHANNEL, :, :] = float(next_player)
    
    return new_tensor


def apply_move_to_state(state, row: int, col: int) -> 'HexGameState':
    """
    Apply a move to a HexGameState and return the new state.
    
    This is the primary function for applying moves to game states.
    It handles the move validation and state updates.
    
    Args:
        state: HexGameState instance
        row: Row index (0-12)
        col: Column index (0-12)
        
    Returns:
        New HexGameState with the move applied
        
    Raises:
        ValueError: If move is invalid
    """
    if not state.is_valid_move(row, col):
        raise ValueError(f"Invalid move: ({row}, {col})")
    
    # Use the existing make_move method which handles all the game logic
    return state.make_move(row, col)


def apply_move_to_state_trmph(state, trmph_move: str) -> 'HexGameState':
    """
    Apply a TRMPH move to a HexGameState and return the new state.
    
    This is a wrapper that converts TRMPH to row,col coordinates.
    
    Args:
        state: HexGameState instance
        trmph_move: TRMPH format move (e.g., "a1", "b2")
        
    Returns:
        New HexGameState with the move applied
        
    Raises:
        ValueError: If TRMPH move is invalid or move is invalid
    """
    
    try:
        row, col = trmph_move_to_rowcol(trmph_move)
        return apply_move_to_state(state, row, col)
    except Exception as e:
        raise ValueError(f"Invalid TRMPH move '{trmph_move}': {e}")


def apply_move_to_tensor_trmph(board_tensor: torch.Tensor, trmph_move: str, player: int) -> torch.Tensor:
    """
    Apply a TRMPH move to a 3-channel board tensor and return the new tensor.
    
    This is a wrapper that converts TRMPH to row,col coordinates.
    
    Args:
        board_tensor: torch.Tensor of shape (3, BOARD_SIZE, BOARD_SIZE)
        trmph_move: TRMPH format move (e.g., "a1", "b2")
        player: Player making the move (BLUE_PLAYER=0 or RED_PLAYER=1)
        
    Returns:
        New board tensor with the move applied
        
    Raises:
        ValueError: If TRMPH move is invalid or move is invalid
    """
    
    try:
        row, col = trmph_move_to_rowcol(trmph_move)
        return apply_move_to_tensor(board_tensor, row, col, player)
    except Exception as e:
        raise ValueError(f"Invalid TRMPH move '{trmph_move}': {e}") 