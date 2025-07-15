import torch
import numpy as np
from typing import Union, Tuple, List
from hex_ai.inference.model_wrapper import ModelWrapper
from hex_ai.utils import format_conversion as fc
from hex_ai.inference.board_display import display_hex_board

class SimpleModelInference:
    def __init__(self, checkpoint_path: str, device: str = None, model_type: str = "resnet18"):
        self.model = ModelWrapper(checkpoint_path, device=device, model_type=model_type)
        self.board_size = fc.BOARD_SIZE

    def trmph_to_2nxn(self, trmph: str) -> torch.Tensor:
        board_nxn = fc.parse_trmph_to_board(trmph, board_size=self.board_size)
        board_2nxn = fc.board_nxn_to_2nxn(board_nxn)
        return board_2nxn

    def infer(self, board: Union[str, np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, float]:
        """
        Accepts a board in trmph string, (N,N) np.ndarray, or (2,N,N) or (3,N,N) torch.Tensor format.
        TODO: Update to always use (3,N,N) input, constructing player-to-move channel as in training.
        Returns (policy_probs, value_estimate)
        """
        if isinstance(board, str):
            board_2nxn = self.trmph_to_2nxn(board)
        elif isinstance(board, np.ndarray):
            board_2nxn = fc.board_nxn_to_2nxn(board)
        elif isinstance(board, torch.Tensor):
            if board.shape == (2, self.board_size, self.board_size):
                board_2nxn = board
            elif board.shape == (3, self.board_size, self.board_size):
                board_2nxn = board
            else:
                raise ValueError(f"Tensor must have shape (2, {self.board_size}, {self.board_size}) or (3, {self.board_size}, {self.board_size})")
        else:
            raise TypeError("Board must be a trmph string, (N,N) np.ndarray, or (2,N,N) or (3,N,N) torch.Tensor")
        # TODO: Add player-to-move channel for inference, as in training pipeline.
        policy_logits, value_logit = self.model.predict(board_2nxn)
        policy_probs = torch.softmax(policy_logits, dim=0).numpy()
        value = torch.sigmoid(value_logit).item()  # Probability blue wins
        return policy_probs, value

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