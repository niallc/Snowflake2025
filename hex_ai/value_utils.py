from hex_ai.enums import Winner, Player, Piece, Channel, ValuePerspective, channel_to_int, player_to_int
from hex_ai.config import (
    TRMPH_BLUE_WIN, TRMPH_RED_WIN,
    TRAINING_BLUE_WIN, TRAINING_RED_WIN,
    BOARD_SIZE, PIECE_ONEHOT, EMPTY_ONEHOT,
)
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

## Enums are now defined in hex_ai.enums and imported above

# --- Conversion functions ---
def value_target_from_winner(winner: Winner) -> float:
    """Convert Winner to training value target (0.0 for Blue, 1.0 for Red)."""
    if not isinstance(winner, Winner):
        raise ValueError(f"Invalid winner: {winner}")
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
    """Convert Player enum to integer (strict)."""
    if not isinstance(player, Player):
        raise TypeError(f"player must be Player enum, got {type(player)}")
    return int(player.value)

def int_to_player(player_int: int) -> Player:
    """Convert integer to Player enum (strict)."""
    if not isinstance(player_int, int):
        raise TypeError(f"player_int must be int, got {type(player_int)}")
    if player_int not in (Player.BLUE.value, Player.RED.value):
        raise ValueError(f"{player_int} is not a valid Player")
    return Player(player_int)

def piece_to_char(piece: Piece) -> str:
    """Convert Piece enum to its canonical character encoding ('e'|'b'|'r')."""
    if not isinstance(piece, Piece):
        raise TypeError(f"piece must be Piece enum, got {type(piece)}")
    return str(piece.value)

def char_to_piece(piece_char: str) -> Piece:
    """Convert character encoding to Piece enum."""
    if not isinstance(piece_char, str):
        raise TypeError(f"piece_char must be str, got {type(piece_char)}")
    mapping = {"e": Piece.EMPTY, "b": Piece.BLUE, "r": Piece.RED}
    if piece_char not in mapping:
        raise ValueError(f"Invalid piece char: {piece_char}")
    return mapping[piece_char]

def channel_to_int(channel: Channel) -> int:
    """Convert Channel enum to integer for backward compatibility."""
    if not isinstance(channel, Channel):
        raise TypeError(f"channel must be Channel enum, got {type(channel)}")
    return int(channel.value)

def int_to_channel(channel_int: int) -> Channel:
    """Convert integer to Channel enum."""
    if not isinstance(channel_int, int):
        raise TypeError(f"channel_int must be int, got {type(channel_int)}")
    if channel_int not in (Channel.BLUE.value, Channel.RED.value, Channel.PLAYER_TO_MOVE.value):
        raise ValueError(f"Invalid channel int: {channel_int}")
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

# =============================
# Value Prediction Utilities (New Enhanced Value Head)
# =============================

class ValuePredictor:
    """
    Centralized utility for handling value predictions from the enhanced value head.
    
    The new value head outputs values in [-1, 1] range with tanh activation.
    This utility provides a clean interface for converting these values to various
    probability formats needed by different parts of the codebase.
    """
    
    @staticmethod
    def model_output_to_probability(model_output: float) -> float:
        """
        Convert model output from [-1, 1] range to [0, 1] probability.
        
        Args:
            model_output: Raw model output in [-1, 1] range (tanh activated)
            
        Returns:
            Probability in [0, 1] range representing Red's win probability
        """
        return (model_output + 1) / 2
    
    @staticmethod
    def model_output_to_probability_tensor(model_output: torch.Tensor) -> torch.Tensor:
        """
        Convert model output tensor from [-1, 1] range to [0, 1] probability.
        
        Args:
            model_output: Raw model output tensor in [-1, 1] range (tanh activated)
            
        Returns:
            Probability tensor in [0, 1] range representing Red's win probability
        """
        return (model_output + 1) / 2
    
    @staticmethod
    def get_win_probability(model_output: float, player: Player) -> float:
        """
        Get win probability for a specific player from model output.
        
        Args:
            model_output: Raw model output in [-1, 1] range (tanh activated)
            player: Player to get win probability for
            
        Returns:
            Win probability for the specified player in [0, 1] range
        """
        prob_red_win = ValuePredictor.model_output_to_probability(model_output)
        
        if player == Player.RED:
            return prob_red_win
        elif player == Player.BLUE:
            return 1.0 - prob_red_win
        else:
            raise ValueError(f"Invalid player: {player}")
    
    @staticmethod
    def get_win_probability_tensor(model_output: torch.Tensor, player: Player) -> torch.Tensor:
        """
        Get win probability tensor for a specific player from model output.
        
        Args:
            model_output: Raw model output tensor in [-1, 1] range (tanh activated)
            player: Player to get win probability for
            
        Returns:
            Win probability tensor for the specified player in [0, 1] range
        """
        prob_red_win = ValuePredictor.model_output_to_probability_tensor(model_output)
        
        if player == Player.RED:
            return prob_red_win
        elif player == Player.BLUE:
            return 1.0 - prob_red_win
        else:
            raise ValueError(f"Invalid player: {player}")
    
    @staticmethod
    def convert_to_minimax_value(model_output: float, root_player: Player) -> float:
        """
        Convert model output to minimax-friendly value from root player's perspective.
        
        Args:
            model_output: Raw model output in [-1, 1] range (tanh activated)
            root_player: Player whose perspective to use for minimax value
            
        Returns:
            Minimax value in [-1, 1] range where positive = good for root player
        """
        prob_red_win = ValuePredictor.model_output_to_probability(model_output)
        
        if root_player == Player.BLUE:
            # For Blue: positive values = Blue wins (good), negative values = Red wins (bad)
            # Convert from Red's win probability to Blue's perspective
            return 1.0 - 2.0 * prob_red_win  # Maps [0,1] to [1,-1]
        else:  # root_player == Player.RED
            # For Red: negative values = Red wins (good), positive values = Blue wins (bad)
            # Convert from Red's win probability to Red's perspective
            return 2.0 * prob_red_win - 1.0  # Maps [0,1] to [-1,1]
    
    @staticmethod
    def batch_convert_to_probabilities(model_outputs: torch.Tensor) -> torch.Tensor:
        """
        Convert batch of model outputs from [-1, 1] range to [0, 1] probabilities.
        
        Args:
            model_outputs: Batch of raw model outputs in [-1, 1] range (tanh activated)
            
        Returns:
            Batch of probabilities in [0, 1] range representing Red's win probabilities
        """
        return ValuePredictor.model_output_to_probability_tensor(model_outputs)
    
    @staticmethod
    def validate_model_output(model_output: float) -> bool:
        """
        Validate that model output is in the expected [-1, 1] range.
        
        Args:
            model_output: Raw model output to validate
            
        Returns:
            True if output is in valid range, False otherwise
        """
        return -1.0 <= model_output <= 1.0
    
    @staticmethod
    def validate_model_output_tensor(model_output: torch.Tensor) -> bool:
        """
        Validate that model output tensor is in the expected [-1, 1] range.
        
        Args:
            model_output: Raw model output tensor to validate
            
        Returns:
            True if all outputs are in valid range, False otherwise
        """
        return torch.all((model_output >= -1.0) & (model_output <= 1.0)).item()


# Legacy compatibility functions (deprecated but kept for backward compatibility)
def model_output_to_prob(model_output: float, perspective: ValuePerspective) -> float:
    """
    Convert model output (sigmoid(logit)) to probability for the given perspective.
    
    DEPRECATED: This function assumes sigmoid-based model outputs. Use ValuePredictor
    for the new tanh-based value head.
    
    The value head predicts Red's win probability because Red wins are labeled as 1.0 in training.
    model_output is the probability that Red wins (after applying sigmoid to the raw logit).
    """
    if perspective is None:
        raise ValueError("perspective cannot be None")
    elif perspective == ValuePerspective.TRAINING_TARGET:
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
    
    DEPRECATED: This function assumes sigmoid-based model outputs. Use ValuePredictor
    for the new tanh-based value head.
    
    The value head predicts Red's win probability, so the model output convention
    is the probability that Red wins.
    """
    if perspective is None:
        raise ValueError("perspective cannot be None")
    elif perspective == ValuePerspective.TRAINING_TARGET:
        return prob
    elif perspective == ValuePerspective.BLUE_WIN_PROB:
        return 1.0 - prob
    elif perspective == ValuePerspective.RED_WIN_PROB:
        return prob
    else:
        raise ValueError(f"Unknown perspective: {perspective}")

def get_win_prob_from_model_output(model_output: float, player) -> float:
    """
    Given the raw model output (logit) and a player (Winner or Player only),
    return the probability that player wins.
    
    DEPRECATED: This function assumes sigmoid-based model outputs. Use ValuePredictor
    for the new tanh-based value head.
    
    The value head predicts Red's win probability because Red wins are labeled as 1.0 in training.
    """
    if isinstance(player, Player):
        perspective = ValuePerspective.BLUE_WIN_PROB if player == Player.BLUE else ValuePerspective.RED_WIN_PROB
    elif isinstance(player, Winner):
        perspective = ValuePerspective.BLUE_WIN_PROB if player == Winner.BLUE else ValuePerspective.RED_WIN_PROB
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
        - temperature < 0.02: Uses deterministic selection to avoid numerical issues
    """

    if temperature <= 0:
        # Greedy selection: return one-hot vector for argmax
        result = np.zeros_like(logits)
        result[np.argmax(logits)] = 1.0
        return result
    
    # For very low temperatures, use deterministic selection to avoid numerical issues
    if temperature < 0.02:
        result = np.zeros_like(logits)
        result[np.argmax(logits)] = 1.0
        return result

    # Apply temperature scaling: logits / temperature
    scaled_logits = logits / temperature
    # Apply softmax
    probs = torch.softmax(torch.tensor(scaled_logits), dim=0).numpy()
    
    # Validate that we got reasonable probabilities
    if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
        raise ValueError(f"Invalid probabilities detected: NaN or Inf values in softmax output")
    
    # Note: This check is redundant since softmax should never return all zeros for finite input
    # But it's kept as a safety check for edge cases
    if np.sum(probs) == 0:
        raise ValueError(f"All probabilities are zero! Logits min/max: {logits.min():.6f}/{logits.max():.6f}, temperature: {temperature}")
    
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

def winner_to_color(winner) -> str:
    """
    Map Winner enum, Player enum, or player int to color name string ('blue' or 'red').
    
    Args:
        winner: Winner enum, Player enum, or player int (0 for Blue, 1 for Red)
        
    Returns:
        Color name string: 'blue' or 'red'
        
    Raises:
        TypeError: If winner is not a Winner enum, Player enum, or int
        ValueError: If winner is an invalid int value
    """
    if isinstance(winner, Winner):
        return 'blue' if winner == Winner.BLUE else 'red'
    if isinstance(winner, Player):
        return 'blue' if winner == Player.BLUE else 'red'
    # Backward compatibility for legacy int inputs 0/1
    if isinstance(winner, int):
        if winner == player_to_int(Player.BLUE):
            return 'blue'
        if winner == player_to_int(Player.RED):
            return 'red'
        raise ValueError(f"Invalid player integer: {winner}. Expected 0 (Blue) or 1 (Red)")
    
    raise TypeError(f"Invalid winner type: {type(winner)}. Expected Winner enum, Player enum, or int (0/1), got {type(winner)}")

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


def policy_logits_to_probs_2d(policy_logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Convert policy logits to 2D probabilities using temperature-scaled softmax.
    
    This function is specifically designed for MCTS and other applications that prefer
    2D coordinate access. It flattens the logits, applies softmax, and reshapes back to 2D.

    Args:
        policy_logits: Raw policy logits (numpy array, typically 13x13 for Hex)
        temperature: Temperature parameter (default 1.0)

    Returns:
        2D temperature-scaled softmax probabilities with same shape as input
    """
    if temperature <= 0:
        # Greedy selection: return one-hot vector for argmax
        result = np.zeros_like(policy_logits)
        result[np.argmax(policy_logits)] = 1.0
        return result

    # Store original shape
    original_shape = policy_logits.shape
    
    # Flatten logits for softmax
    logits_flat = policy_logits.flatten()
    
    # Apply temperature scaling: logits / temperature
    scaled_logits = logits_flat / temperature
    
    # Apply softmax to flattened array
    probs_flat = torch.softmax(torch.tensor(scaled_logits), dim=0).numpy()
    
    # Reshape back to original shape
    probs = probs_flat.reshape(original_shape)
    
    # Validate that we got reasonable probabilities
    if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
        raise ValueError(f"Invalid probabilities detected: NaN or Inf values in softmax output")
    
    # Note: This check is redundant since softmax should never return all zeros for finite input
    # But it's kept as a safety check for edge cases
    if np.sum(probs) == 0:
        raise ValueError(f"All probabilities are zero! Logits min/max: {policy_logits.min():.6f}/{policy_logits.max():.6f}, temperature: {temperature}")
    
    return probs

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
    
    blue_val = board_tensor[Channel.BLUE.value, row, col].item()
    red_val = board_tensor[Channel.RED.value, row, col].item()
    
    # Check for invalid values (should only be approximately EMPTY_ONEHOT or PIECE_ONEHOT)
    if not (abs(blue_val - EMPTY_ONEHOT) < tolerance or abs(blue_val - PIECE_ONEHOT) < tolerance):
        raise ValueError(f"Invalid blue channel value at ({row}, {col}): {blue_val}")
    if not (abs(red_val - EMPTY_ONEHOT) < tolerance or abs(red_val - PIECE_ONEHOT) < tolerance):
        raise ValueError(f"Invalid red channel value at ({row}, {col}): {red_val}")
    
    # Position is empty if both channels are approximately EMPTY_ONEHOT
    return abs(blue_val - EMPTY_ONEHOT) < tolerance and abs(red_val - EMPTY_ONEHOT) < tolerance


def apply_move_to_tensor(board_tensor: torch.Tensor, row: int, col: int, player: Player) -> torch.Tensor:
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
        player: Player making the move (Player enum only)
        
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
    
    # Strict enum-only API
    if not isinstance(player, Player):
        raise TypeError(f"player must be Player, got {type(player)}")
    
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
    """Check if the given player/winner is blue (strict inputs)."""
    if isinstance(player, Player):
        return player == Player.BLUE
    if isinstance(player, Winner):
        return player == Winner.BLUE
    raise TypeError(f"Unknown player type: {type(player)}")

def is_red(player) -> bool:
    """Check if the given player/winner is red (strict inputs)."""
    if isinstance(player, Player):
        return player == Player.RED
    if isinstance(player, Winner):
        return player == Winner.RED
    raise TypeError(f"Unknown player type: {type(player)}")

def player_to_winner(player: Player) -> Winner:
    """Convert Player enum to Winner enum."""
    return Winner.BLUE if player == Player.BLUE else Winner.RED

def winner_to_player(winner: Winner) -> Player:
    """Convert Winner enum to Player enum."""
    return Player.BLUE if winner == Winner.BLUE else Player.RED 