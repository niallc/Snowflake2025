"""
Utility functions for the Hex AI project.

This module contains helper functions for data processing, model utilities,
and other common operations used throughout the project.
"""

import torch
import numpy as np
import time
from typing import List, Optional, Dict, Tuple
import logging

from .config import BOARD_SIZE, POLICY_LOSS_WEIGHT
from .model_spec import (
    DEFAULT_MODEL_TYPE,
    load_checkpoint_payload,
    model_spec_from_model,
    resolve_model_spec_from_checkpoint_payload,
)


# =============================================================
#  Hyperparameter Configuration
# =============================================================

# Default hyperparameter sweep configuration
# This is the central place where all hyperparameters are defined
DEFAULT_HYPERPARAMETER_SWEEP = {
    "batch_size": [256],
    "max_grad_norm": [2.0],  # Updated default for AdamW
    "weight_decay": [1e-4],
    "value_learning_rate_factor": [1],  # Value head learns slower if this is < 1
    "value_weight_decay_factor": [1],  # Value head gets more regularization if this is > 1
    # Keep scripted defaults aligned with Trainer defaults in hex_ai.config.
    "policy_weight": [POLICY_LOSS_WEIGHT],
    "learning_rate": [8e-4],  # Updated default for AdamW
    
    # AdamW optimizer parameters
    "betas": [(0.9, 0.999)],  # Coefficients for computing running averages
    "eps": [1e-8],  # Term added to denominator for numerical stability
    
    # New KataGo-inspired architecture parameters
    "model_type": [DEFAULT_MODEL_TYPE],
    "num_blocks": [7],  # Number of residual blocks - 6 blocks ≈ ResNet-18
    "trunk_channels": [128],  # Number of channels in trunk
    "board_size": [BOARD_SIZE],  # Configured training board size
    "dropout_prob": [0],  # Legacy parameter (not used in current architecture)
    "use_policy_search_targets": [True],  # Consume per-position MCTS policy targets by default
    "soft_target_legal_mix_alpha": [0.017],  # Default legal-uniform mixing for non-one-hot policy targets
    
    # Note: Value head parameters (bottleneck_channels=32, hidden_dim=256, k_outputs=4) 
    # are currently fixed in the architecture but could be made configurable later
}

# Short labels for parameters (used in experiment naming)
HYPERPARAMETER_SHORT_LABELS = {
    "learning_rate": "lr",
    "batch_size": "bs",
    "max_grad_norm": "mgn",
    "dropout_prob": "do",
    "weight_decay": "wd",
    "value_learning_rate_factor": "vlrf",
    "value_weight_decay_factor": "vwdf",
    "policy_weight": "pw",
    "value_weight": "vw",
    "model_type": "mt",
    "num_blocks": "nb",
    "trunk_channels": "tc",
    "board_size": "n",
    "betas": "betas",
    "eps": "eps",
    "use_policy_search_targets": "psrch",
    "soft_target_legal_mix_alpha": "stmix",
}


def create_hyperparameter_sweep(overrides: Dict = None) -> Dict:
    """
    Create a hyperparameter sweep configuration with optional overrides.
    
    Args:
        overrides: Dictionary of parameter overrides. Keys should match parameter names,
                  values should be lists (even single values should be in lists).
                  
    Returns:
        Dictionary with hyperparameter sweep configuration
        
    Examples:
        # Use default configuration
        sweep = create_hyperparameter_sweep()
        
        # Override learning rate
        sweep = create_hyperparameter_sweep({"learning_rate": [1e-4]})
        
        # Override multiple parameters
        sweep = create_hyperparameter_sweep({
            "learning_rate": [1e-4, 5e-5],
            "batch_size": [128, 256]
        })
    """
    sweep = DEFAULT_HYPERPARAMETER_SWEEP.copy()
    
    if overrides:
        for param, values in overrides.items():
            if param not in sweep:
                raise ValueError(f"Unknown hyperparameter: {param}. "
                               f"Available parameters: {list(sweep.keys())}")
            if not isinstance(values, list):
                raise ValueError(f"Override values must be lists, got {type(values)} for {param}")
            sweep[param] = values
    
    return sweep


def get_single_hyperparameter_config(overrides: Dict = None) -> Dict:
    """
    Get a single hyperparameter configuration (first value from each sweep parameter).
    
    Args:
        overrides: Dictionary of parameter overrides
        
    Returns:
        Dictionary with single values for each parameter
        
    Examples:
        # Get default single config
        config = get_single_hyperparameter_config()
        
        # Get config with overridden learning rate
        config = get_single_hyperparameter_config({"learning_rate": [1e-4]})
    """
    sweep = create_hyperparameter_sweep(overrides)
    return {param: values[0] for param, values in sweep.items()}


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def save_checkpoint(model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   filepath: str,
                   compress: bool = True):
    """
    Save a model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        filepath: Path to save the checkpoint
        compress: Whether to save as gzipped file (.pt.gz)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_spec': model_spec_from_model(model).to_dict(),
    }
    
    if compress:
        # Ensure path has .pt.gz extension
        if not filepath.endswith('.pt.gz'):
            filepath = filepath.replace('.pt', '.pt.gz') if filepath.endswith('.pt') else filepath + '.pt.gz'
        
        # Save as gzipped file
        import gzip
        with gzip.open(filepath, 'wb') as f:
            torch.save(checkpoint, f)
    else:
        # Save as uncompressed file
        torch.save(checkpoint, filepath)


def load_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   filepath: str) -> Tuple[int, float]:
    """
    Load a model checkpoint.
    
    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        filepath: Path to the checkpoint file
        
    Returns:
        Tuple of (epoch, loss)
    """
    checkpoint = load_checkpoint_payload(filepath)
    checkpoint_model_spec = resolve_model_spec_from_checkpoint_payload(checkpoint)
    runtime_model_spec = model_spec_from_model(model)
    if checkpoint_model_spec != runtime_model_spec:
        raise ValueError(
            "Checkpoint model spec does not match runtime model spec. "
            f"checkpoint={checkpoint_model_spec.to_dict()} "
            f"runtime={runtime_model_spec.to_dict()} "
            f"path={filepath}"
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def get_device() -> str:
    """
    Get the appropriate device for training or inference.
    Returns:
        str: 'cuda' if available, else 'mps' (Apple Silicon GPU) if available, else 'cpu'.
    Note:
        This function should be used everywhere device selection is needed for consistency.
        All scripts and modules should import and use this function instead of direct torch.cuda/mps/cpu checks.
    """
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    logging.getLogger(__name__).debug(f"[get_device] Selected device: {device}")
    return device


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed) 


class TrainingUtilities:
    """
    Generic training utilities that can be used across different training frameworks.
    
    This class contains utility functions that are not specific to any particular
    trainer implementation and can be reused in various training scenarios.
    """
    
    @staticmethod
    def move_batch_to_device(boards: torch.Tensor, policies: torch.Tensor, 
                           values: torch.Tensor, move_stage: torch.Tensor, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move batch data to the specified device."""
        return boards.to(device), policies.to(device), values.to(device), move_stage.to(device)
    
    @staticmethod
    def calculate_batch_timing(state: Dict) -> Dict:
        """Calculate timing metrics for the current batch."""
        data_load_end = time.time()
        batch_data_time = data_load_end - state['data_load_start']
        state['batch_data_times'].append(batch_data_time)
        batch_start_time = time.time()
        
        return {
            'data_load_end': data_load_end,
            'batch_data_time': batch_data_time,
            'batch_start_time': batch_start_time
        }
    
    
    @staticmethod
    def calculate_statistics(values: List[float]) -> Dict[str, float]:
        """Calculate mean, min, max, std statistics for a list of values."""
        if not values:
            return {}
        return {
            'mean': float(np.mean(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'std': float(np.std(values))
        }
    
    @staticmethod
    def format_epoch_id(epoch: int, mini_epoch: int) -> str:
        """Format epoch and mini-epoch into a string identifier."""
        return f"{epoch}_mini{mini_epoch}"
    
    @staticmethod
    def calculate_mini_epoch_time(batch_times: List[float]) -> float:
        """Calculate total time for a mini-epoch from batch times."""
        return sum(batch_times)
    
    @staticmethod
    def should_log_progress(batch_idx: int, epoch: int, mini_epoch: int, 
                          next_log_batch: int, start_time: float, last_time_log: float) -> bool:
        """Determine if we should log progress at this batch."""
        now = time.time()
        
        # For first epoch, log for all powers of 2
        if epoch == 0 and mini_epoch == 0:
            return batch_idx + 1 == next_log_batch
        
        # For later epochs, only log for batch >= 64
        if batch_idx + 1 >= 64 and batch_idx + 1 == next_log_batch:
            return True
            
        # After 3 minutes, switch to time-based logging every 180 seconds
        return now - last_time_log > 180
    
    @staticmethod
    def calculate_weight_statistics(model: torch.nn.Module) -> Optional[Dict[str, float]]:
        """Calculate weight statistics for a model."""
        weight_norms = []
        for p in model.parameters():
            if p.data is not None:
                weight_norms.append(p.data.norm(2).item())
        if weight_norms:
            return {
                'mean': float(np.mean(weight_norms)),
                'std': float(np.std(weight_norms))
            }
        return None
    
    @staticmethod
    def calculate_learning_rate_statistics(optimizer: torch.optim.Optimizer) -> Optional[Dict[str, float]]:
        """Calculate learning rate statistics from optimizer parameter groups."""
        lr_values = [group['lr'] for group in optimizer.param_groups if 'lr' in group]
        if lr_values:
            return TrainingUtilities.calculate_statistics(lr_values)
        return None
    
    @staticmethod
    def get_gpu_memory_usage() -> Optional[float]:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available() and hasattr(torch.cuda, 'memory_allocated'):
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return None
    
