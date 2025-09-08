"""
Utility functions for the Hex AI project.

This module contains helper functions for data processing, model utilities,
and other common operations used throughout the project.
"""

import torch
import numpy as np
import time
import math
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import logging

from .config import BOARD_SIZE, NUM_PLAYERS, POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE
from hex_ai.value_utils import ValuePredictor


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
    # Check if file is gzipped by reading the first two bytes
    def is_gzipped(filepath):
        with open(filepath, 'rb') as f:
            return f.read(2) == b'\x1f\x8b'
    
    if is_gzipped(filepath):
        import gzip
        with gzip.open(filepath, 'rb') as f:
            checkpoint = torch.load(f, weights_only=False)
    else:
        checkpoint = torch.load(filepath, weights_only=False)
        
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def validate_board_shape(tensor: torch.Tensor) -> bool:
    """
    Validate that a tensor has the correct board shape.
    
    Args:
        tensor: Tensor to validate
        
    Returns:
        True if shape is correct, False otherwise
    """
    expected_shape = (NUM_PLAYERS, BOARD_SIZE, BOARD_SIZE)
    return tensor.shape == expected_shape


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


def create_sample_data(batch_size: int = 8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create sample data for testing purposes.
    
    Args:
        batch_size: Number of samples to create
        
    Returns:
        Tuple of (boards, policies, values) tensors
    """
    # Create random board states (2 channels for 2 players)
    boards = torch.randn(batch_size, NUM_PLAYERS, BOARD_SIZE, BOARD_SIZE)
    
    # Create random policy targets (169 possible moves)
    policies = torch.randn(batch_size, POLICY_OUTPUT_SIZE)
    policies = torch.softmax(policies, dim=1)  # Convert to probabilities
    
    # Create random value targets (single value per board)
    values = torch.randn(batch_size, VALUE_OUTPUT_SIZE)
    # TODO: Check here and elsewhere whether we're correctly using [-1, 1], vs. [0, 1].
    values = torch.sigmoid(values)  # Convert to [0, 1] range (targets are still in [0,1])
    
    return boards, policies, values


class GradientMonitor:
    """Monitor gradients during training to help debug value head issues."""
    
    def __init__(self, model, log_interval=100):
        self.model = model
        self.log_interval = log_interval
        self.gradient_history = {
            'policy_head': [],
            'value_head': [],
            'shared_layers': []
        }
        self.batch_count = 0
    
    def compute_gradient_norms(self):
        """Compute gradient norms for different parts of the model."""
        policy_norms = []
        value_norms = []
        shared_norms = []
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                norm = param.grad.norm().item()
                
                if 'policy_head' in name:
                    policy_norms.append(norm)
                elif 'value_head' in name:
                    value_norms.append(norm)
                else:
                    shared_norms.append(norm)
        
        return {
            'policy_head': np.mean(policy_norms) if policy_norms else 0.0,
            'value_head': np.mean(value_norms) if value_norms else 0.0,
            'shared_layers': np.mean(shared_norms) if shared_norms else 0.0
        }
    
    def log_gradients(self, batch_idx):
        """Log gradient norms if it's time to do so."""
        self.batch_count += 1
        if self.batch_count % self.log_interval == 0:
            norms = self.compute_gradient_norms()
            self.gradient_history['policy_head'].append(norms['policy_head'])
            self.gradient_history['value_head'].append(norms['value_head'])
            self.gradient_history['shared_layers'].append(norms['shared_layers'])
            
            print(f"[Batch {batch_idx}] Gradient norms - Policy: {norms['policy_head']:.6f}, "
                  f"Value: {norms['value_head']:.6f}, Shared: {norms['shared_layers']:.6f}")
    
    def get_summary(self):
        """Get summary statistics of gradient norms."""
        summary = {}
        for key, values in self.gradient_history.items():
            if values:
                summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        return summary


class ActivationMonitor:
    """Monitor activations during training to help debug value head issues."""
    
    def __init__(self, model, log_interval=100):
        self.model = model
        self.log_interval = log_interval
        self.activation_hooks = []
        self.activation_history = {}
        self.batch_count = 0
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture activations."""
        def hook_fn(name):
            def hook(module, input, output):
                if self.batch_count % self.log_interval == 0:
                    if isinstance(output, torch.Tensor):
                        # Compute activation statistics
                        with torch.no_grad():
                            mean_act = output.mean().item()
                            std_act = output.std().item()
                            max_act = output.max().item()
                            min_act = output.min().item()
                            
                            if name not in self.activation_history:
                                self.activation_history[name] = []
                            
                            self.activation_history[name].append({
                                'mean': mean_act,
                                'std': std_act,
                                'max': max_act,
                                'min': min_act
                            })
            return hook
        
        # Register hooks for key layers
        for name, module in self.model.named_modules():
            if any(key in name for key in ['value_head', 'policy_head', 'layer4', 'global_pool']):
                hook = module.register_forward_hook(hook_fn(name))
                self.activation_hooks.append(hook)
    
    def log_activations(self, batch_idx):
        """Log activation statistics if it's time to do so."""
        self.batch_count += 1
        if self.batch_count % self.log_interval == 0:
            print(f"[Batch {batch_idx}] Activation monitoring active")
    
    def get_summary(self):
        """Get summary statistics of activations."""
        summary = {}
        for layer_name, activations in self.activation_history.items():
            if activations:
                summary[layer_name] = {
                    'mean_activation': np.mean([a['mean'] for a in activations]),
                    'std_activation': np.mean([a['std'] for a in activations]),
                    'max_activation': np.max([a['max'] for a in activations]),
                    'min_activation': np.min([a['min'] for a in activations]),
                    'count': len(activations)
                }
        return summary
    
    def cleanup(self):
        """Remove hooks to prevent memory leaks."""
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks = []


class ValueHeadAnalyzer:
    """Analyze value head performance and behavior."""
    
    def __init__(self, model, dataset, device):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.analysis_results = {}
    
    def analyze_value_predictions(self, num_samples=1000):
        """Analyze value predictions across different position types."""
        self.model.eval()
        predictions = []
        targets = []
        position_types = []
        
        with torch.no_grad():
            for i in range(min(num_samples, len(self.dataset))):
                board, policy, value = self.dataset[i]
                board = board.unsqueeze(0).to(self.device)
                
                policy_pred, value_pred = self.model(board)
                # Convert from [-1, 1] to [0, 1] range using centralized utility
                value_prob = ValuePredictor.model_output_to_probability(value_pred.item())
                
                predictions.append(value_prob)
                targets.append(value.item())
                
                # Determine position type based on board state
                board_np = board[0].cpu().numpy()
                blue_pieces = np.sum(board_np[0] > 0)
                red_pieces = np.sum(board_np[1] > 0)
                total_pieces = blue_pieces + red_pieces
                
                if total_pieces < 10:
                    position_types.append('early')
                elif total_pieces < 100:
                    position_types.append('mid')
                else:
                    position_types.append('late')
        
        # Analyze predictions by position type
        for pos_type in ['early', 'mid', 'late']:
            mask = [t == pos_type for t in position_types]
            if any(mask):
                type_preds = [p for p, m in zip(predictions, mask) if m]
                type_targets = [t for t, m in zip(targets, mask) if m]
                
                mse = np.mean((np.array(type_preds) - np.array(type_targets)) ** 2)
                accuracy = np.mean([abs(p - t) < 0.3 for p, t in zip(type_preds, type_targets)])
                
                self.analysis_results[f'{pos_type}_game'] = {
                    'mse': mse,
                    'accuracy': accuracy,
                    'mean_prediction': np.mean(type_preds),
                    'std_prediction': np.std(type_preds),
                    'sample_count': len(type_preds)
                }
        
        return self.analysis_results
    
    def test_simple_positions(self):
        """Test value predictions on simple, known positions."""
        self.model.eval()
        test_positions = {
            'empty_board': torch.zeros(1, 3, 13, 13).to(self.device),
            'blue_winning': self._create_winning_position('blue'),
            'red_winning': self._create_winning_position('red')
        }
        
        results = {}
        with torch.no_grad():
            for name, board in test_positions.items():
                policy_pred, value_pred = self.model(board)
                # Convert from [-1, 1] to [0, 1] range using centralized utility
                value_prob = ValuePredictor.model_output_to_probability(value_pred.item())
                results[name] = value_prob
        
        return results
    
    def _create_winning_position(self, winner):
        """Create a simple winning position for testing."""
        board = torch.zeros(1, 3, 13, 13)
        if winner == 'blue':
            # Create a simple blue winning pattern
            for i in range(13):
                board[0, 0, i, 0] = 1.0  # Blue pieces in first column
            board[0, 2, :, :] = 0.0  # Blue to move
        else:
            # Create a simple red winning pattern
            for i in range(13):
                board[0, 1, 0, i] = 1.0  # Red pieces in first row
            board[0, 2, :, :] = 1.0  # Red to move
        
        return board.to(self.device)


class TrainingUtilities:
    """
    Generic training utilities that can be used across different training frameworks.
    
    This class contains utility functions that are not specific to any particular
    trainer implementation and can be reused in various training scenarios.
    """
    
    @staticmethod
    def move_batch_to_device(boards: torch.Tensor, policies: torch.Tensor, 
                           values: torch.Tensor, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move batch data to the specified device."""
        return boards.to(device), policies.to(device), values.to(device)
    
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
    def calculate_gradient_norm(model: torch.nn.Module) -> float:
        """Calculate gradient norm for all model parameters."""
        total_norm = 0.0
        param_count = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        return total_norm ** (1. / 2) if param_count > 0 else 0.0
    
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
    
    @staticmethod
    def get_checkpoints_to_keep(max_epoch: int, max_checkpoints: int) -> set:
        """Determine which checkpoints to keep based on epoch number."""
        # Always keep last 3, and 2, 5, 10, 20, 40, 60, 100, ...
        keep = set()
        if max_epoch < 4:
            keep.update(range(1, max_epoch + 1))
        else:
            keep.update([max_epoch, max_epoch - 1, max_epoch - 2])
            for k in [2, 5, 10, 20, 40, 60, 100, 140, 200, 300]:
                if k <= max_epoch:
                    keep.add(k)
        return keep
