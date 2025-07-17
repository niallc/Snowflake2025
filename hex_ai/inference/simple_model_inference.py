import torch
import numpy as np
from typing import Union, Tuple, List
from hex_ai.inference.model_wrapper import ModelWrapper
from hex_ai.utils import format_conversion as fc
from hex_ai.inference.board_display import display_hex_board

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
                self.is_legacy = (first_layer.in_channels == 2)
            else:
                # Fallback: assume legacy if model_instance was provided
                self.is_legacy = True
        else:
            # Current model always uses 3 channels
            self.is_legacy = False
        
        print(f"Detected {'legacy' if self.is_legacy else 'current'} model (expects {2 if self.is_legacy else 3} channels)")

    def trmph_to_2nxn(self, trmph: str) -> torch.Tensor:
        board_nxn = fc.parse_trmph_to_board(trmph, board_size=self.board_size)
        # TODO: This function is legacy and only used internally; prefer board_nxn_to_3nxn for inference
        return fc.board_nxn_to_2nxn(board_nxn)

    def infer(self, board: Union[str, np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, float]:
        """
        Accepts a board in trmph string, (N,N) np.ndarray, or (2,N,N) or (3,N,N) torch.Tensor format.
        Converts to appropriate input format based on model type (2-channel for legacy, 3-channel for current).
        Returns (policy_probs, value_estimate)
        """
        if isinstance(board, str):
            board_nxn = fc.parse_trmph_to_board(board, board_size=self.board_size)
            if self.is_legacy:
                input_tensor = fc.board_nxn_to_2nxn(board_nxn)
            else:
                input_tensor = fc.board_nxn_to_3nxn(board_nxn)
        elif isinstance(board, np.ndarray):
            if board.shape == (self.board_size, self.board_size):
                if self.is_legacy:
                    input_tensor = fc.board_nxn_to_2nxn(board)
                else:
                    input_tensor = fc.board_nxn_to_3nxn(board)
            elif board.shape == (2, self.board_size, self.board_size):
                if self.is_legacy:
                    input_tensor = torch.tensor(board, dtype=torch.float32)
                else:
                    input_tensor = fc.board_2nxn_to_3nxn(torch.tensor(board, dtype=torch.float32))
            elif board.shape == (3, self.board_size, self.board_size):
                if self.is_legacy:
                    raise ValueError(f"Legacy model expects 2-channel input, but got 3-channel numpy array")
                else:
                    input_tensor = torch.tensor(board, dtype=torch.float32)
            else:
                raise ValueError(f"Numpy array must have shape ({self.board_size}, {self.board_size}), (2, {self.board_size}, {self.board_size}), or (3, {self.board_size}, {self.board_size})")
        elif isinstance(board, torch.Tensor):
            if board.shape == (2, self.board_size, self.board_size):
                if self.is_legacy:
                    input_tensor = board
                else:
                    input_tensor = fc.board_2nxn_to_3nxn(board)
            elif board.shape == (3, self.board_size, self.board_size):
                if self.is_legacy:
                    raise ValueError(f"Legacy model expects 2-channel input, but got 3-channel tensor")
                else:
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