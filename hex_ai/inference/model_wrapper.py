import gzip
import os
import torch
from hex_ai.models import create_model
from typing import Optional, Tuple, List, Union
from hex_ai.training_utils import get_device

# NOTE: As of July 2025, the model expects (3, N, N) input: blue, red, player-to-move channels.
# TODO: Update all inference logic to construct and use 3-channel input, matching the training pipeline.
class ModelWrapper:
    """
    Wrapper for loading a trained Hex AI model and running inference.
    Designed for easy extension to ensembles and prediction caching.
    Now supports passing a model instance (e.g., for legacy models).
    """
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        model_type: str = "resnet18"
    ):
        """
        Args:
            checkpoint_path: Path to the model checkpoint (.pt or .pth file, possibly .gz)
            device: 'cpu', 'cuda', 'mps', or None for auto-detect
            model_type: Model architecture type (default: 'resnet18')
        """
        self.device = self._detect_device(device)
        self.model = self._load_model(checkpoint_path, model_type)
        self.model.eval()
        self.model.to(self.device)
        # self.prediction_cache = {}  # Uncomment to enable prediction caching

    def _detect_device(self, device: Optional[str]) -> torch.device:
        if device is not None:
            return torch.device(device)
        return get_device()

    def _load_model(self, checkpoint_path: str, model_type: str):
        model = create_model(model_type)
        try:
            if checkpoint_path.endswith('.gz'):
                with gzip.open(checkpoint_path, 'rb') as f:
                    checkpoint = torch.load(f, map_location=self.device, weights_only=False)
            else:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {e}")
        return model



    def eval(self):
        """Set the underlying model to evaluation mode (passthrough)."""
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        """Set the underlying model to train/eval mode (passthrough)."""
        self.model.train(mode)
        return self

    def predict(self, board_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference on a single board tensor.
        Args:
            board_tensor: torch.Tensor of shape (3, N, N) or (batch_size=1, 3, N, N)
        Returns:
            policy_logits: torch.Tensor of shape (169,) or (N*N,)
            value_logit: torch.Tensor of shape (1,)
        """
        self.model.eval()
        with torch.no_grad():
            if board_tensor.dim() == 3:
                board_tensor = board_tensor.unsqueeze(0)  # Add batch dim
            board_tensor = board_tensor.to(self.device, dtype=torch.float32)
            policy_logits, value_logit = self.model(board_tensor)
            # Remove batch dimension for single input
            return policy_logits[0].cpu(), value_logit[0].cpu()

    def batch_predict(self, board_tensors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference on a batch of board tensors.
        Args:
            board_tensors: torch.Tensor of shape (batch_size, 3, N, N)
        Returns:
            policy_logits: torch.Tensor of shape (batch_size, N*N)
            value_logits: torch.Tensor of shape (batch_size, 1)
        """
        self.model.eval()
        with torch.no_grad():
            board_tensors = board_tensors.to(self.device, dtype=torch.float32)
            policy_logits, value_logits = self.model(board_tensors)
            return policy_logits.cpu(), value_logits.cpu()

    # Future: Add prediction caching here for MCTS, e.g. by hashing board_tensor
    # def predict(self, board_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     key = board_tensor.cpu().numpy().tobytes()
    #     if key in self.prediction_cache:
    #         return self.prediction_cache[key]
    #     ...
    #     self.prediction_cache[key] = (policy_logits, value_logit)
    #     return policy_logits, value_logit 