import torch
import numpy as np
from typing import Union, Tuple, List
from hex_ai.inference.model_wrapper import ModelWrapper
from hex_ai.utils import format_conversion as fc
from hex_ai.inference.board_display import display_hex_board

class SimpleModelInference:
    def __init__(self, checkpoint_path: str, device: str = None, model_type: str = "resnet18"):
        print(f"SimpleModelInference.__init__() called with checkpoint_path={checkpoint_path}, device={device}, model_type={model_type}")
        self.model = ModelWrapper(checkpoint_path, device=device, model_type=model_type)
        self.board_size = fc.BOARD_SIZE

    def trmph_to_2nxn(self, trmph: str) -> torch.Tensor:
        board_nxn = fc.parse_trmph_to_board(trmph, board_size=self.board_size)
        # TODO: This function is legacy and only used internally; prefer board_nxn_to_3nxn for inference
        return fc.board_nxn_to_2nxn(board_nxn)

    def infer(self, board: Union[str, np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, float]:
        """
        Accepts a board in trmph string, (N,N) np.ndarray, or (2,N,N) or (3,N,N) torch.Tensor format.
        Always converts to (3,N,N) input, constructing player-to-move channel as in training.
        Returns (policy_probs, value_estimate)
        """
        if isinstance(board, str):
            board_nxn = fc.parse_trmph_to_board(board, board_size=self.board_size)
            input_tensor = fc.board_nxn_to_3nxn(board_nxn)
        elif isinstance(board, np.ndarray):
            if board.shape == (self.board_size, self.board_size):
                input_tensor = fc.board_nxn_to_3nxn(board)
            elif board.shape == (2, self.board_size, self.board_size):
                input_tensor = fc.board_2nxn_to_3nxn(torch.tensor(board, dtype=torch.float32))
            elif board.shape == (3, self.board_size, self.board_size):
                input_tensor = torch.tensor(board, dtype=torch.float32)
            else:
                raise ValueError(f"Numpy array must have shape ({self.board_size}, {self.board_size}), (2, {self.board_size}, {self.board_size}), or (3, {self.board_size}, {self.board_size})")
        elif isinstance(board, torch.Tensor):
            if board.shape == (2, self.board_size, self.board_size):
                input_tensor = fc.board_2nxn_to_3nxn(board)
            elif board.shape == (3, self.board_size, self.board_size):
                input_tensor = board
            else:
                raise ValueError(f"Tensor must have shape (2, {self.board_size}, {self.board_size}) or (3, {self.board_size}, {self.board_size})")
        else:
            raise TypeError("Board must be a trmph string, (N,N) np.ndarray, or (2,N,N) or (3,N,N) torch.Tensor")

        policy_logits, value_logit = self.model.predict(input_tensor)
        policy_probs = torch.softmax(policy_logits, dim=0).numpy()
        raw_value = value_logit.item()
        value = torch.sigmoid(value_logit).item()  # Probability blue wins
        return policy_probs, value, raw_value

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