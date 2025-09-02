import gzip
import os
import torch
import logging
import time
from hex_ai.models import create_model
from typing import Optional, Tuple, List, Union, Dict, Any
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
        self._logged_batch_predict_info = False
        # Log effective device details for diagnostics
        try:
            param_device = next(self.model.parameters()).device
        except StopIteration:
            param_device = torch.device("unknown")
        logging.getLogger(__name__).info(
            f"ModelWrapper initialized: wrapper_device={self.device}, param_device={param_device}, "
            f"mps_available={torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}, "
            f"cuda_available={torch.cuda.is_available()}"
        )
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
            value_signed: torch.Tensor of shape (1,)
        """
        self.model.eval()
        with torch.no_grad():
            if board_tensor.dim() == 3:
                board_tensor = board_tensor.unsqueeze(0)  # Add batch dim
            before_device = getattr(board_tensor, 'device', 'na')
            board_tensor = board_tensor.to(self.device, dtype=torch.float32)
            if logging.getLogger(__name__).isEnabledFor(logging.DEBUG):
                logging.getLogger(__name__).debug(
                    f"predict: input_device_before={before_device}, input_device_after={board_tensor.device}, model_device={next(self.model.parameters()).device}"
                )
            policy_logits, value_signed = self.model(board_tensor)
            # Remove batch dimension for single input
            return policy_logits[0].cpu(), value_signed[0].cpu()

    def batch_predict(self, board_tensors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference on a batch of board tensors.
        Args:
            board_tensors: torch.Tensor of shape (batch_size, 3, N, N)
        Returns:
            policy_logits: torch.Tensor of shape (batch_size, N*N)
            value_signed: torch.Tensor of shape (batch_size, 1)
        """
        self.model.eval()
        with torch.no_grad():
            before_device = getattr(board_tensors, 'device', 'na')
            board_tensors = board_tensors.to(self.device, dtype=torch.float32)
            logger = logging.getLogger(__name__)
            if not self._logged_batch_predict_info:
                logger.info(
                    f"batch_predict runtime: batch_shape={tuple(board_tensors.shape)}, input_device_before={before_device}, "
                    f"input_device_after={board_tensors.device}, model_device={next(self.model.parameters()).device}"
                )
                self._logged_batch_predict_info = True
            elif logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"batch_predict: input_device_before={before_device}, input_device_after={board_tensors.device}, model_device={next(self.model.parameters()).device}"
                )
            policy_logits, value_signed = self.model(board_tensors)
            return policy_logits.cpu(), value_signed.cpu()

    def infer_timed(self, board_tensors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Run inference on a batch of board tensors with detailed timing information.
        Args:
            board_tensors: torch.Tensor of shape (batch_size, 3, N, N)
        Returns:
            policy_logits: torch.Tensor of shape (batch_size, N*N) on CPU
            value_signed: torch.Tensor of shape (batch_size, 1) on CPU
            timing_info: Dict with timing details (h2d_ms, forward_ms, d2h_ms, batch_size, device, param_dtype)
        """
        self.model.eval()
        
        # Get device info
        param_device = next(self.model.parameters()).device
        param_dtype = str(next(self.model.parameters()).dtype)
        
        # Time host to device transfer
        h2d_start = time.perf_counter()
        before_device = getattr(board_tensors, 'device', 'na')
        board_tensors = board_tensors.to(self.device, dtype=torch.float32)
        h2d_ms = (time.perf_counter() - h2d_start) * 1000.0
        
        # Time the actual neural network forward pass (pure inference)
        pure_forward_start = time.perf_counter()
        with torch.no_grad():
            policy_logits, value_signed = self.model(board_tensors)
        pure_forward_ms = (time.perf_counter() - pure_forward_start) * 1000.0
        
        # Time device synchronization
        sync_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            sync_ms = (time.perf_counter() - sync_start) * 1000.0
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Simple host read approach - no MPS events for now
            _ = (policy_logits.sum() + value_signed.sum()).item()
            sync_ms = 0.0  # No separate sync needed
        else:
            sync_ms = (time.perf_counter() - sync_start) * 1000.0
        
        # Total forward time (including sync)
        forward_ms = pure_forward_ms + sync_ms
        
        # Time device to host transfer
        d2h_start = time.perf_counter()
        policy_cpu = policy_logits.cpu()
        value_cpu = value_signed.cpu()
        d2h_ms = (time.perf_counter() - d2h_start) * 1000.0
        
        timing_info = {
            "h2d_ms": h2d_ms,
            "forward_ms": forward_ms,
            "pure_forward_ms": pure_forward_ms,
            "sync_ms": sync_ms,
            "d2h_ms": d2h_ms,
            "batch_size": int(board_tensors.shape[0]),
            "device": str(self.device),
            "param_dtype": param_dtype,
        }
        
        return policy_cpu, value_cpu, timing_info

    def get_device(self) -> torch.device:
        """Return the torch.device used by the wrapper."""
        return self.device

    def get_param_device(self) -> torch.device:
        """Return the device of the first model parameter (effective model device)."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.device

    def get_device_info(self) -> dict:
        """Return a snapshot of device-related diagnostics for inference."""
        info = {
            'wrapper_device': str(self.device),
            'param_device': str(self.get_param_device()),
            'cuda_available': torch.cuda.is_available(),
            'mps_available': bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()),
            'default_dtype': str(torch.get_default_dtype()),
        }
        try:
            if torch.cuda.is_available():
                info['cuda_device_name'] = torch.cuda.get_device_name(0)
        except Exception:
            pass
        return info

    # Future: Add prediction caching here for MCTS, e.g. by hashing board_tensor
    # def predict(self, board_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     key = board_tensor.cpu().numpy().tobytes()
    #     if key in self.prediction_cache:
    #         return self.prediction_cache[key]
    #     ...
    #     self.prediction_cache[key] = (policy_logits, value_logit)
    #     return policy_logits, value_logit 