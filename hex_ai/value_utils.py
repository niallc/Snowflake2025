from enum import Enum
from hex_ai.config import (
    TRMPH_BLUE_WIN, TRMPH_RED_WIN,
    TRAINING_BLUE_WIN, TRAINING_RED_WIN,
    BLUE_PLAYER, RED_PLAYER,
    BOARD_SIZE, PIECE_ONEHOT, EMPTY_ONEHOT,
    BLUE_CHANNEL, RED_CHANNEL
)
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

class Player(Enum):
    """Player constants for game logic and player-to-move channel."""
    BLUE = 0
    RED = 1

class Piece(Enum):
    """Piece constants for N×N board representation."""
    EMPTY = "e"
    BLUE = "b"
    RED = "r"

class Channel(Enum):
    """Channel indices for one-hot encoded board formats."""
    BLUE = 0
    RED = 1
    PLAYER_TO_MOVE = 2

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

# --- Backward compatibility functions ---
def player_to_int(player: Player) -> int:
    """Convert Player enum to integer for backward compatibility."""
    return player.value

def int_to_player(player_int: int) -> Player:
    """Convert integer to Player enum."""
    return Player(player_int)

def piece_to_int(piece: Piece) -> int:
    """Convert Piece enum to integer for backward compatibility."""
    return piece.value

def int_to_piece(piece_int: int) -> Piece:
    """Convert integer to Piece enum."""
    return Piece(piece_int)

def channel_to_int(channel: Channel) -> int:
    """Convert Channel enum to integer for backward compatibility."""
    return channel.value

def int_to_channel(channel_int: int) -> Channel:
    """Convert integer to Channel enum."""
    return Channel(channel_int)

# --- Validation functions to catch legacy formats ---
def validate_piece_value(piece_value) -> None:
    """Raise ValueError if piece_value is a legacy numeric value."""
    if isinstance(piece_value, int):
        raise ValueError(f"Legacy numeric piece value ({piece_value}) detected. Use character representation ('e', 'b', 'r') instead.")

def validate_trmph_winner(trmph_winner: str) -> None:
    """Raise ValueError if trmph_winner is a legacy value."""
    if trmph_winner == "1":  # Legacy TRMPH_BLUE_WIN
        raise ValueError(f"Legacy TRMPH_BLUE_WIN value ('1') detected. Use new TRMPH_BLUE_WIN ('b') instead.")
    elif trmph_winner == "2":  # Legacy TRMPH_RED_WIN
        raise ValueError(f"Legacy TRMPH_RED_WIN value ('2') detected. Use new TRMPH_RED_WIN ('r') instead.")

# --- Model output utilities ---
def model_output_to_prob(model_output: float, perspective: ValuePerspective) -> float:
    """
    Convert model output (sigmoid(logit)) to probability for the given perspective.
    
    The value head predicts Red's win probability because Red wins are labeled as 1.0 in training.
    model_output is the probability that Red wins (after applying sigmoid to the raw logit).
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
    Convert a probability for a given perspective to the model output convention.
    
    The value head predicts Red's win probability, so the model output convention
    is the probability that Red wins.
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
    return the probability that player wins.
    
    The value head predicts Red's win probability because Red wins are labeled as 1.0 in training.
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

def get_player_to_move_from_moves(moves: list) -> Player:
    """
    Given a list of moves (e.g., ['a1', 'b2', ...]), return Player.BLUE if it's blue's move, Player.RED if it's red's move.
    Blue always starts, so even number of moves = Blue's turn, odd = Red's turn.
    """
    if len(moves) % 2 == 0:
        return Player.BLUE
    else:
        return Player.RED

def winner_to_color(winner):
    """Map Winner enum, Player enum, or player int to color name string ('blue', 'red', or 'reset')."""
    if isinstance(winner, Winner):
        return 'blue' if winner == Winner.BLUE else 'red'
    if isinstance(winner, Player):
        return 'blue' if winner == Player.BLUE else 'red'
    if winner == BLUE_PLAYER:  # Backward compatibility
        return 'blue'
    elif winner == RED_PLAYER:  # Backward compatibility
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

def sample_moves_from_policy(policy_logits: np.ndarray, legal_moves: List[Tuple[int, int]], 
                            board_size: int, k: int, temperature: float = 1.0) -> List[Tuple[Tuple[int, int], float]]:
    """
    Sample k moves from policy logits with temperature scaling.
    
    Args:
        policy_logits: Raw policy logits
        legal_moves: List of (row, col) tuples for legal moves
        board_size: Board size
        k: Number of moves to sample
        temperature: Temperature for sampling (0.0 = deterministic top-k, higher = more random)
        
    Returns:
        List of ((row, col), probability) tuples for k sampled moves
    """
    if not legal_moves:
        return []
    
    # Convert logits to probabilities with temperature scaling
    policy_probs = temperature_scaled_softmax(policy_logits, temperature)
    
    # Get legal move probabilities
    move_indices = [row * board_size + col for row, col in legal_moves]
    legal_policy = np.array([policy_probs[idx] for idx in move_indices])
    
    # Normalize legal move probabilities to sum to 1
    legal_policy_sum = np.sum(legal_policy)
    if legal_policy_sum > 0:
        legal_policy = legal_policy / legal_policy_sum
    else:
        # If all probabilities are 0, use uniform distribution
        legal_policy = np.ones(len(legal_moves)) / len(legal_moves)
    
    # Sample k moves from the policy distribution
    if temperature == 0.0 or len(legal_moves) <= k:
        # Deterministic: take top-k moves
        topk_idx = np.argsort(legal_policy)[::-1][:k]
        sampled_moves = [legal_moves[i] for i in topk_idx]
    else:
        # Stochastic: sample k moves weighted by policy probabilities
        sampled_indices = np.random.choice(len(legal_moves), size=k, replace=False, p=legal_policy)
        sampled_moves = [legal_moves[i] for i in sampled_indices]
    
    # Get corresponding probabilities for the sampled moves
    move_indices = [row * board_size + col for row, col in sampled_moves]
    move_probs = [float(policy_probs[idx]) for idx in move_indices]
    
    return list(zip(sampled_moves, move_probs))


def get_top_k_moves_with_probs(policy_logits: np.ndarray,
                               legal_moves: List[Tuple[int, int]],
                               board_size: int, k: int,
                               temperature: float = 1.0) -> List[Tuple[Tuple[int, int], float]]:
    """
    Get top-k moves with their probabilities, using temperature-aware sampling.
    
    Args:
        policy_logits: Raw policy logits
        legal_moves: List of (row, col) tuples for legal moves
        board_size: Board size
        k: Number of top moves to return
        temperature: Temperature for policy sampling (0.0 = deterministic top-k, higher = more random)
        
    Returns:
        List of ((row, col), probability) tuples for top-k moves
    """
    return sample_moves_from_policy(policy_logits, legal_moves, board_size, k, temperature)

def select_policy_move(state, model, temperature: float = 1.0) -> Tuple[int, int]:
    """
    Select a move using policy head with centralized utilities.
    
    Args:
        state: Game state (must have .board and .get_legal_moves() methods)
        model: Model instance (must have .simple_infer() method that returns (policy_logits, value_logit))
        temperature: Temperature for policy sampling (default 1.0)
        
    Returns:
        (row, col) tuple for the selected move
        
    Raises:
        ValueError: If no legal moves are available
    """
    policy_logits, _ = model.simple_infer(state.board)
    
    # Use centralized utilities for policy processing
    policy_probs = policy_logits_to_probs(policy_logits, temperature)
    legal_moves = state.get_legal_moves()
    legal_policy = get_legal_policy_probs(policy_probs, legal_moves, state.board.shape[0])
    
    # Sample a move from the policy distribution
    if temperature == 0.0 or len(legal_moves) == 1:
        # Deterministic: take the best move
        best_moves = select_top_k_moves(legal_policy, legal_moves, 1)
        return best_moves[0]
    else:
        # Stochastic: sample a move weighted by policy probabilities
        # Normalize probabilities to sum to 1
        legal_policy_sum = np.sum(legal_policy)
        if legal_policy_sum > 0:
            legal_policy = legal_policy / legal_policy_sum
        else:
            # If all probabilities are 0, use uniform distribution
            legal_policy = np.ones(len(legal_moves)) / len(legal_moves)
        
        # Sample a move
        chosen_idx = np.random.choice(len(legal_moves), p=legal_policy)
        return legal_moves[chosen_idx] 

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


def apply_move_to_tensor(board_tensor: torch.Tensor, row: int, col: int, player) -> torch.Tensor:
    """
    Apply a move to a 3-channel board tensor and return the new tensor.
    
    This is the core function that directly manipulates tensors for efficiency.
    The tensor should be in (3, BOARD_SIZE, BOARD_SIZE) format where:
    - channels[Channel.BLUE] = blue pieces (0 or 1)
    - channels[Channel.RED] = red pieces (0 or 1) 
    - channels[Channel.PLAYER_TO_MOVE] = player-to-move (Player.BLUE or Player.RED)
    
    Args:
        board_tensor: torch.Tensor of shape (3, BOARD_SIZE, BOARD_SIZE)
        row: Row index (0-(BOARD_SIZE-1))
        col: Column index (0-(BOARD_SIZE-1))
        player: Player making the move (Player enum, or int for backward compatibility)
        
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
    
    # Convert player to Player enum if it's an integer
    if isinstance(player, int):
        player = int_to_player(player)
    
    # Create new tensor (don't modify original)
    new_tensor = board_tensor.clone()
    
    # Place the piece in the appropriate channel using enum
    if player == Player.BLUE:
        new_tensor[Channel.BLUE.value, row, col] = PIECE_ONEHOT  # Blue channel
    elif player == Player.RED:
        new_tensor[Channel.RED.value, row, col] = PIECE_ONEHOT  # Red channel
    else:
        raise ValueError(f"Invalid player: {player}")
    
    # Update player-to-move channel (switch to other player)
    next_player = Player.RED if player == Player.BLUE else Player.BLUE
    new_tensor[Channel.PLAYER_TO_MOVE.value, :, :] = float(next_player.value)
    
    return new_tensor



def get_top_k_legal_moves(model, state, top_k=20, temperature=1.0, return_probs=False):
    """
    Given a model and state, return the top-k legal moves (optionally with their probabilities).
    Args:
        model: Model instance (must have .simple_infer() method)
        state: Game state (must have .board and .get_legal_moves())
        top_k: Number of top moves to return
        temperature: Temperature for policy softmax
        return_probs: If True, return list of (move, prob) tuples; else just moves
    Returns:
        List of top-k moves [(row, col), ...] or [(row, col), prob] if return_probs
        Returns None if there are no legal moves.
    """
    policy_logits, _ = model.simple_infer(state.board)
    policy_probs = policy_logits_to_probs(policy_logits, temperature)
    legal_moves = state.get_legal_moves()
    if not legal_moves:
        return None
    legal_policy = get_legal_policy_probs(policy_probs, legal_moves, state.board.shape[0])
    if len(legal_policy) == 0:
        return None
    topk_moves = select_top_k_moves(legal_policy, legal_moves, top_k)
    if return_probs:
        # Map back to probabilities
        move_indices = [row * state.board.shape[0] + col for row, col in topk_moves]
        move_probs = [float(policy_probs[idx]) for idx in move_indices]
        return list(zip(topk_moves, move_probs))
    return topk_moves


# --- Utility functions for enum usage ---
def get_opponent(player: Player) -> Player:
    """Get the opponent of the given player."""
    return Player.RED if player == Player.BLUE else Player.BLUE

def is_blue(player) -> bool:
    """Check if the given player/winner is blue."""
    if isinstance(player, Player):
        return player == Player.BLUE
    if isinstance(player, Winner):
        return player == Winner.BLUE
    if isinstance(player, int):
        return player == BLUE_PLAYER  # Backward compatibility
    raise ValueError(f"Unknown player type: {type(player)}")

def is_red(player) -> bool:
    """Check if the given player/winner is red."""
    if isinstance(player, Player):
        return player == Player.RED
    if isinstance(player, Winner):
        return player == Winner.RED
    if isinstance(player, int):
        return player == RED_PLAYER  # Backward compatibility
    raise ValueError(f"Unknown player type: {type(player)}")

def player_to_winner(player: Player) -> Winner:
    """Convert Player enum to Winner enum."""
    return Winner.BLUE if player == Player.BLUE else Winner.RED

def winner_to_player(winner: Winner) -> Player:
    """Convert Winner enum to Player enum."""
    return Player.BLUE if winner == Winner.BLUE else Player.RED 