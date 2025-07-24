import torch
import numpy as np
from typing import Union, Tuple, List
from hex_ai.inference.model_wrapper import ModelWrapper
from hex_ai.utils import format_conversion as fc
from hex_ai.inference.board_display import display_hex_board
from hex_ai.config import (
    PLAYER_CHANNEL, LEGACY_MODEL_CHANNELS, CURRENT_MODEL_CHANNELS,
    BLUE_PLAYER, RED_PLAYER, BLUE_CHANNEL, RED_CHANNEL,
    BLUE_PIECE, RED_PIECE, EMPTY_PIECE,
    TRAINING_BLUE_WIN, TRAINING_RED_WIN,
    TRMPH_BLUE_WIN, TRMPH_RED_WIN,
    trmph_winner_to_training_value, trmph_winner_to_clear_str
)
from hex_ai.data_utils import create_board_from_moves, preprocess_example_for_model, get_player_to_move_from_board
from hex_ai.value_utils import model_output_to_prob, ValuePerspective

class SimpleModelInference:
    def __init__(self, checkpoint_path: str, device: str = None, model_type: str = "resnet18", model_instance=None):
        print(f"SimpleModelInference.__init__() called with checkpoint_path={checkpoint_path}, device={device}, model_type={model_type}, model_instance={type(model_instance) if model_instance is not None else None}")
        self.model = ModelWrapper(checkpoint_path, device=device, model_type=model_type, model_instance=model_instance)
        self.board_size = fc.BOARD_SIZE
        
        # Detect if this is a legacy model (2-channel input) or current model (3-channel input)
        if model_instance is not None:
            # Check the first layer's input channels to determine if it's legacy
            first_layer = list(self.model.model.children())[0]
            if hasattr(first_layer, 'in_channels'):
                self.is_legacy = (first_layer.in_channels == LEGACY_MODEL_CHANNELS)  # Legacy models have 2 channels, current models have 3
            else:
                # Fallback: assume legacy if model_instance was provided
                self.is_legacy = True
        else:
            # Current model always uses 3 channels
            self.is_legacy = False
        
        print(f"Detected {'legacy' if self.is_legacy else 'current'} model (expects {LEGACY_MODEL_CHANNELS if self.is_legacy else CURRENT_MODEL_CHANNELS} channels)")

    def trmph_to_2nxn(self, trmph: str) -> torch.Tensor:
        board_nxn = fc.parse_trmph_to_board(trmph, board_size=self.board_size)
        # TODO: This function is legacy and only used internally; prefer board_nxn_to_2nxn for inference
        return fc.board_nxn_to_2nxn(board_nxn)

    def _is_finished_position(self, board_2ch: np.ndarray) -> Tuple[bool, int]:
        """
        Check if a position is finished (one player has won).
        Returns (is_finished, winner) where winner is BLUE_PLAYER or RED_PLAYER.
        """
        # For now, use a simple heuristic: if one player has significantly more pieces
        # and the board has a reasonable number of pieces, it's likely a finished position
        blue_pieces = np.sum(board_2ch[BLUE_CHANNEL])  # Use BLUE_CHANNEL constant
        red_pieces = np.sum(board_2ch[RED_CHANNEL])    # Use RED_CHANNEL constant
        total_pieces = blue_pieces + red_pieces
        
        # If board has a reasonable number of pieces (>10), check for winner
        if total_pieces > 10:
            # Simple heuristic: player with more pieces likely won
            if blue_pieces > red_pieces:
                return True, BLUE_PLAYER  # Use BLUE_PLAYER constant
            elif red_pieces > blue_pieces:
                return True, RED_PLAYER   # Use RED_PLAYER constant
        
        return False, -1

    def _create_board_with_correct_player_channel(self, board_2ch: np.ndarray) -> torch.Tensor:
        """
        Create a 3-channel board tensor with the correct player-to-move channel.
        For finished positions, sets the player-to-move to the loser (next player),
        which is how the model was trained.
        """
        is_finished, winner = self._is_finished_position(board_2ch)
        
        if is_finished:
            # For finished positions, set player-to-move to the loser (next player)
            # This matches how the model was trained on final positions
            # NOTE: This is the key insight - training data has player-to-move = loser
            loser = RED_PLAYER if winner == BLUE_PLAYER else BLUE_PLAYER
            player_to_move = loser
        else:
            # For non-finished positions, use normal logic
            player_to_move = get_player_to_move_from_board(board_2ch)
        
        # Create player-to-move channel
        # NOTE: player_to_move is BLUE_PLAYER (0) or RED_PLAYER (1)
        player_channel = np.full((1, self.board_size, self.board_size), float(player_to_move), dtype=board_2ch.dtype)
        board_3ch = np.concatenate([board_2ch, player_channel], axis=0)
        return torch.tensor(board_3ch, dtype=torch.float32)

    def infer(self, board: Union[str, np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, float]:
        """
        Accepts a board in trmph string, (N,N) np.ndarray, or (2,N,N) or (3,N,N) torch.Tensor format.
        Uses the same preprocessing pipeline as training to ensure consistency.
        Returns (policy_probs, value_estimate, raw_value)
        """
        # Convert input to the same format used in training
        if isinstance(board, str):
            # For TRMPH strings, use the same pipeline as training
            board_2ch = create_board_from_moves(fc.split_trmph_moves(fc.strip_trmph_preamble(board)))
        elif isinstance(board, np.ndarray):
            if board.shape == (self.board_size, self.board_size):
                # NxN format - convert to 2-channel using same logic as training
                # NOTE: NxN boards use BLUE_PIECE (1) and RED_PIECE (2)
                board_2ch = np.zeros((2, self.board_size, self.board_size), dtype=np.float32)
                board_2ch[BLUE_CHANNEL] = (board == BLUE_PIECE).astype(np.float32)  # Blue channel
                board_2ch[RED_CHANNEL] = (board == RED_PIECE).astype(np.float32)   # Red channel
            elif board.shape == (2, self.board_size, self.board_size):
                # Already 2-channel format
                board_2ch = board
            elif board.shape == (3, self.board_size, self.board_size):
                # 3-channel format - extract first 2 channels
                board_2ch = board[:2]
            else:
                raise ValueError(f"Numpy array must have shape ({self.board_size}, {self.board_size}), (2, {self.board_size}, {self.board_size}), or (3, {self.board_size}, {self.board_size})")
        elif isinstance(board, torch.Tensor):
            if board.shape == (2, self.board_size, self.board_size):
                # Already 2-channel format
                board_2ch = board.cpu().numpy()
            elif board.shape == (3, self.board_size, self.board_size):
                # 3-channel format - extract first 2 channels
                board_2ch = board[:2].cpu().numpy()
            else:
                raise ValueError(f"Tensor must have shape (2, {self.board_size}, {self.board_size}) or (3, {self.board_size}, {self.board_size})")
        else:
            raise TypeError("Board must be a trmph string, (N,N) np.ndarray, or (2,N,N) or (3,N,N) torch.Tensor")

        # Create the 3-channel board with correct player-to-move channel
        board_3ch = self._create_board_with_correct_player_channel(board_2ch)
        
        # For legacy models, we need to remove the player-to-move channel
        if self.is_legacy:
            input_tensor = board_3ch[:2]  # Remove player-to-move channel
        else:
            input_tensor = board_3ch

        policy_logits, value_logit = self.model.predict(input_tensor)
        policy_probs = torch.softmax(policy_logits, dim=0).numpy()
        raw_value = value_logit.item()
        prob_red_win = torch.sigmoid(torch.tensor(raw_value)).item()
        value_blue_win = model_output_to_prob(prob_red_win, ValuePerspective.BLUE_WIN_PROB)
        return policy_probs, value_blue_win, raw_value

    def get_top_k_moves(self, policy_probs: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """
        Returns the top-k moves as (trmph_move, probability) tuples.
        """
        topk_indices = np.argsort(policy_probs)[::-1][:k]
        moves = [(fc.tensor_to_trmph(idx, self.board_size), float(policy_probs[idx])) for idx in topk_indices]
        return moves

    def display_board(self, board: Union[str, np.ndarray, torch.Tensor]):
        """
        Display the board using display_hex_board.
        Accepts trmph string, (N,N) np.ndarray, or (2,N,N) torch.Tensor.
        """
        if isinstance(board, str):
            board_nxn = fc.parse_trmph_to_board(board, board_size=self.board_size)
        elif isinstance(board, np.ndarray):
            board_nxn = board
        elif isinstance(board, torch.Tensor):
            board_nxn = fc.board_2nxn_to_nxn(board)
        else:
            raise TypeError("Board must be a trmph string, (N,N) np.ndarray, or (2,N,N) torch.Tensor")
        display_hex_board(board_nxn) 