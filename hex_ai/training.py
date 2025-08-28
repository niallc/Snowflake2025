"""
Training module for Hex AI.

This module contains the training infrastructure including the Trainer class,
loss functions, and training utilities.
"""

import logging
import math
import os
import pickle
import re
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from .config import VERBOSE_LEVEL
from .models import TwoHeadedResNet
from .config import (
    LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, POLICY_LOSS_WEIGHT, VALUE_LOSS_WEIGHT,
    BOARD_SIZE, POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE
)
from hex_ai.data_pipeline import discover_processed_files
from hex_ai.training_utils import get_device
from hex_ai.training_logger import TrainingLogger, get_memory_usage, get_gpu_memory_usage, get_weight_statistics, get_gradient_norm
from hex_ai.system_utils import get_system_info, calculate_optimal_batch_size
from hex_ai.error_handling import get_board_state_error_tracker
from hex_ai.value_utils import ValuePredictor

logger = logging.getLogger(__name__)

# Value loss gets ~5.7x more weight to balance cross-entropy vs MSE scales
# Note about analysis of training runs that use different loss weights:
# The analysis script *should* use fixed values for the policy and value weights.
# The point is to produce a standardized loss calculation to make the loss that we see comparable across different training loss functions.
# The policy loss will always be higher than the value loss so it's not fair to compare the balanced run against the others with the loss that *it trained with* because that is higher by *construction*.
# Even with both better policy loss AND better value loss, if we weight the value loss higher we'll get a greater total loss.

# So we need to make sure that the loss calculate for the PNG *does* use this fixed weight of the separate policy and training loss.
# summarizing briefly: 
POLICY_LOSS_WEIGHT = 0.15
VALUE_LOSS_WEIGHT = 0.85

class PolicyValueLoss(nn.Module):
    """Combined loss for policy and value heads with support for missing policy targets."""
    
    def __init__(self, policy_weight: float = POLICY_LOSS_WEIGHT, value_weight: float = VALUE_LOSS_WEIGHT):
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.policy_loss = nn.CrossEntropyLoss()
        self.value_loss = nn.MSELoss()
    
    def forward(self, policy_pred: torch.Tensor, value_pred: torch.Tensor,
                policy_target: torch.Tensor, value_target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined policy and value loss.
        
        This function handles the case where policy_target might be None (indicating
        no valid policy target, such as for final game positions). When policy_target
        is None, the policy loss is set to a constant (zero) tensor, which results
        in zero gradients for the policy head while still allowing gradients to flow
        through the value head and shared features.
        
        Args:
            policy_pred: Predicted policy logits (batch_size, policy_output_size)
            value_pred: Predicted value (batch_size, 1)
            policy_target: Target policy probabilities (batch_size, policy_output_size) or None
            value_target: Target value (batch_size, 1)
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses
        """
        # Value loss is computed on the raw value predictions
        # The value head now outputs values in [-1, 1] range with tanh activation
        # We need to convert to [0, 1] range for comparison with targets
        value_prob = ValuePredictor.model_output_to_probability_tensor(value_pred.squeeze())
        value_loss = self.value_loss(value_prob, value_target.squeeze())
        
        # Policy loss: handle None targets by using constant loss (zero gradient)
        if policy_target is None:
            # Create a constant tensor with zero gradient for policy loss
            # This ensures no gradients flow to the policy head when there's no target
            policy_loss = torch.tensor(0.0, device=policy_pred.device, requires_grad=True)
        else:
            # Convert one-hot policy targets to class indices for CrossEntropyLoss
            # CrossEntropyLoss expects class indices, not one-hot vectors
            policy_class_target = policy_target.argmax(dim=1)
            policy_loss = self.policy_loss(policy_pred, policy_class_target)
        
        # Combine losses with weights
        total_loss = (self.policy_weight * policy_loss + 
                     self.value_weight * value_loss)
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
        
        return total_loss, loss_dict


class MixedPrecisionTrainer:
    """Wrapper for mixed precision training capabilities."""
    
    def __init__(self, device: str):
        logger.debug(f"[MixedPrecisionTrainer.__init__] device argument = {device} (type: {type(device)})")
        device_str = str(device)
        logger.debug(f"[MixedPrecisionTrainer.__init__] device_str = {device_str}")
        self.device = device
        self.use_mixed_precision = device_str in ['cuda', 'mps']
        
        if self.use_mixed_precision:
            try:
                if device_str == 'cuda':
                    self.autocast = autocast
                    self.scaler = GradScaler()
                    logger.info("Mixed precision training enabled for CUDA GPU")
                elif device_str == 'mps':
                    # MPS uses torch.autocast with device_type="mps"
                    self.autocast = lambda: torch.autocast(device_type="mps")
                    # MPS doesn't need GradScaler, but we'll keep the interface
                    self.scaler = None
                    logger.info("Mixed precision training enabled for MPS GPU")
            except ImportError:
                logger.warning("PyTorch AMP not available, falling back to full precision")
                self.use_mixed_precision = False
        else:
            logger.info("Mixed precision disabled (CPU training)")
            self.use_mixed_precision = False
    
    def autocast_context(self):
        """Get autocast context if available."""
        if self.use_mixed_precision:
            return self.autocast()
        else:
            # Return a no-op context manager
            return nullcontext()
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training."""
        if self.use_mixed_precision and self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(self, optimizer: optim.Optimizer):
        """Step optimizer with proper scaling."""
        if self.use_mixed_precision and self.scaler is not None:
            self.scaler.step(optimizer)
        else:
            optimizer.step()
    
    def update_scaler(self):
        """Update gradient scaler."""
        if self.use_mixed_precision and self.scaler is not None:
            self.scaler.update()


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in validation loss to be considered an improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_val_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.stopped_epoch = 0
        
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info("Restored best model weights")
            return True
        return False
    
    def get_best_val_loss(self) -> float:
        """Get the best validation loss achieved."""
        return self.best_val_loss


class Trainer:
    """Training manager for Hex AI models."""
    
    def __init__(self, model: TwoHeadedResNet, 
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 learning_rate: float = LEARNING_RATE,
                 device: str = None,
                 enable_system_analysis: bool = True,
                 enable_csv_logging: bool = True,
                 experiment_name: Optional[str] = None,
                 policy_weight: float = POLICY_LOSS_WEIGHT,
                 value_weight: float = VALUE_LOSS_WEIGHT,
                 weight_decay: float = 1e-4,
                 max_grad_norm: float = 20.0,
                 value_learning_rate_factor: float = 0.1,
                 value_weight_decay_factor: float = 5.0,
                 log_interval_batches: int = 200,
                 run_timestamp: Optional[str] = None,
                 shutdown_handler=None):
        """
        Args:
            model: The neural network model to train.
            train_loader: DataLoader for the training dataset.
            val_loader: Optional DataLoader for the validation dataset.
            learning_rate: Learning rate for the optimizer.
            device: Device to use for training (e.g., 'cuda', 'cpu').
            enable_system_analysis: Whether to run system analysis.
            enable_csv_logging: Whether to enable CSV logging.
            experiment_name: Optional name for the experiment.
            policy_weight: Weight for the policy loss.
            value_weight: Weight for the value loss.
            weight_decay: Weight decay for the optimizer.
            max_grad_norm: If not None, clip gradients to this max norm after backward(). Default: 20.0
            value_learning_rate_factor: Factor to multiply learning rate for value head (default: 0.1)
            value_weight_decay_factor: Factor to multiply weight decay for value head (default: 5.0)
            log_interval_batches: How often (in batches) to log progress during training (default: 200)
            run_timestamp: Optional timestamp for the entire run to use in log filenames
        """
        if device is None:
            device = get_device()
        logger.debug(f"[Trainer.__init__] device argument = {device}")
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.run_timestamp = run_timestamp
        self.shutdown_handler = shutdown_handler
        
        # Store hyperparameters for logging
        self.value_learning_rate_factor = value_learning_rate_factor
        self.value_weight_decay_factor = value_weight_decay_factor
        self.original_learning_rate = learning_rate  # Store the original learning rate
        
        # Initialize mixed precision
        self.mixed_precision = MixedPrecisionTrainer(device)
        
        # Create parameter groups for different learning rates and weight decay
        # Separate the value head parameters from the rest
        value_head_params = list(model.value_head.parameters())
        value_head_param_ids = {id(p) for p in value_head_params}
        other_params = [p for p in model.parameters() if id(p) not in value_head_param_ids]
        
        param_groups = [
            {
                'params': other_params,
                'lr': learning_rate,
                'weight_decay': weight_decay
            },
            {
                'params': value_head_params,
                'lr': learning_rate * value_learning_rate_factor,
                'weight_decay': weight_decay * value_weight_decay_factor
            }
        ]
        
        # Optimizer and loss
        self.optimizer = optim.Adam(param_groups)
        self.criterion = PolicyValueLoss(policy_weight=policy_weight, value_weight=value_weight)
        
        # Learning rate scheduler (ReduceLROnPlateau)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-5
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        self.start_time = None
        
        # CSV logging
        self.csv_logger = None
        if enable_csv_logging:
            # Use timestamped CSV filename if run_timestamp is provided
            if self.run_timestamp:
                csv_log_file = f"checkpoints/bookkeeping/training_metrics_{self.run_timestamp}.csv"
            else:
                csv_log_file = "checkpoints/bookkeeping/training_metrics.csv"
            self.csv_logger = TrainingLogger(log_file=csv_log_file, experiment_name=experiment_name)
        
        # System analysis
        if enable_system_analysis:
            self._run_system_analysis()
        
        logger.info(f"Initialized trainer with streaming DataLoader (batches unknown)")
        if val_loader:
            logger.info(f"Validation set with streaming DataLoader (batches unknown)")
        
        # Log parameter group info
        logger.info(f"Value head learning rate: {learning_rate * value_learning_rate_factor:.6f} (factor: {value_learning_rate_factor})")
        logger.info(f"Value head weight decay: {weight_decay * value_weight_decay_factor:.6f} (factor: {value_weight_decay_factor})")
        self.log_interval_batches = log_interval_batches
    
    def _run_system_analysis(self):
        """Run system analysis and log recommendations."""
        try:
            system_info = get_system_info()
            _, batch_analysis = calculate_optimal_batch_size()
            
            logger.info("=== System Analysis ===")
            logger.info(f"Platform: {system_info['platform']}")
            logger.info(f"Memory: {system_info['memory_available_gb']:.1f} GB available")
            logger.info(f"GPU: {'Available' if system_info['gpu_available'] else 'Not available'}")
            
            # Warn if batch size is suboptimal
            if batch_analysis['optimal_batch_size'] > self.train_loader.batch_size:
                logger.warning(f"Consider increasing batch size to {batch_analysis['optimal_batch_size']} for better efficiency")
            
            # Warn about GPU usage
            if not system_info['gpu_available'] and self.device in ['cuda', 'mps']:
                logger.warning(f"{self.device.upper()} device requested but no GPU available, falling back to CPU")
            
        except ImportError as e:
            logger.warning(f"System analysis unavailable: {e}")
        except Exception as e:
            logger.warning(f"System analysis failed: {e}")
    
    def train_epoch(self, batch_callback=None) -> Dict[str, float]:
        print(f"Trainer.train_epoch() called")
        print(f"self.current_epoch = {self.current_epoch}")
        if(self.current_epoch == 0):
            print(f"VERBOSE_LEVEL = {VERBOSE_LEVEL}")
            print(f"self.max_grad_norm = {self.max_grad_norm}")
            print(f"self.train_loader.batch_size = {self.train_loader.batch_size}")
            print(f"self.train_loader.dataset = {self.train_loader.dataset}", flush=True)

        # Enhanced logging setup
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Use timestamped filenames if run_timestamp is provided
        if self.run_timestamp:
            verbose_log_file = log_dir / f'run_training_verbose_{self.run_timestamp}.txt'
            diagnostic_log_file = log_dir / f'training_diagnostics_{self.run_timestamp}.txt'
        else:
            print("WARNING: No run_timestamp found; using default log filenames.", flush=True)
            verbose_log_file = log_dir / 'run_training_verbose.txt'
            diagnostic_log_file = log_dir / 'training_diagnostics.txt'
        
        def log_and_print(msg):
            print(msg)
            with open(verbose_log_file, 'a') as f:
                f.write(msg + '\n')
        
        def log_diagnostic(msg):
            with open(diagnostic_log_file, 'a') as f:
                f.write(f"[Epoch {self.current_epoch}] {msg}\n")

        self.model.train()
        epoch_losses = []
        epoch_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': []
        }
        
        # Enhanced diagnostics tracking
        gradient_norms = []
        learning_rates = []
        loss_spikes = []  # Track when loss increases significantly
        batch_times = []
        batch_data_times = []
        epoch_start_time = time.time()
        data_load_start = time.time()
        n_batches = None  # Unknown for streaming datasets
        # For streaming datasets, log every self.log_interval_batches batches
        log_interval = self.log_interval_batches
        special_batches = {0, 1, 3, 9, 29}  # 1st, 2nd, 4th, 10th, 30th (0-based)
        
        # Track previous loss for spike detection
        prev_total_loss = None
        
        try:
            for batch_idx, (boards, policies, values) in enumerate(self.train_loader):
                # DEBUG: Dump first batch of epoch 0 for inspection
                if self.current_epoch == 0 and batch_idx == 0:
                    import pickle, os
                    os.makedirs('analysis/debugging/value_head_performance', exist_ok=True)
                    now = datetime.now()
                    date_str = now.strftime("%Y-%m-%d")
                    time_str = now.strftime("%H-%M")
                    debug_filename = f'analysis/debugging/value_head_performance/batch0_epoch0_{date_str}_{time_str}.pkl'
                    with open(debug_filename, 'wb') as f:
                        pickle.dump({
                            'boards': boards.cpu(),
                            'policies': policies.cpu(),
                            'values': values.cpu()
                        }, f)
                    print("[DEBUG] Dumped first batch of epoch 0 to analysis/debugging/value_head_performance/batch0_epoch0.pkl")
                # Time data loading
                data_load_end = time.time()
                batch_data_time = data_load_end - data_load_start
                batch_data_times.append(batch_data_time)
                batch_start_time = time.time()
                # Move to device
                boards = boards.to(self.device)
                policies = policies.to(self.device)
                values = values.to(self.device)
                # Forward pass with mixed precision
                self.optimizer.zero_grad()
                with self.mixed_precision.autocast_context():
                    policy_pred, value_pred = self.model(boards)
                    total_loss, loss_dict = self.criterion(policy_pred, value_pred, policies, values)
                
                # Enhanced loss spike detection
                if prev_total_loss is not None:
                    loss_change = total_loss.item() - prev_total_loss
                    loss_change_ratio = abs(loss_change) / prev_total_loss if prev_total_loss > 0 else 0
                    if loss_change_ratio > 0.5:  # 50% change threshold
                        loss_spikes.append({
                            'batch': batch_idx,
                            'prev_loss': prev_total_loss,
                            'current_loss': total_loss.item(),
                            'change': loss_change,
                            'change_ratio': loss_change_ratio
                        })
                        log_diagnostic(f"LOSS SPIKE: batch={batch_idx}, prev={prev_total_loss:.4f}, current={total_loss.item():.4f}, change={loss_change:.4f}, ratio={loss_change_ratio:.3f}")
                
                prev_total_loss = total_loss.item()
                
                # Backward pass with scaling
                scaled_loss = self.mixed_precision.scale_loss(total_loss)
                scaled_loss.backward()
                if batch_callback is not None:
                    batch_callback(self, batch_idx)
                
                # Enhanced gradient analysis
                grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float('inf'))
                gradient_norms.append(grad_norm_before_clip)
                
                # Gradient clipping (configurable)
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                    grad_norm_after_clip = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float('inf'))
                    if grad_norm_before_clip > self.max_grad_norm:
                        log_diagnostic(f"GRADIENT CLIPPED: batch={batch_idx}, before={grad_norm_before_clip:.4f}, after={grad_norm_after_clip:.4f}, max_norm={self.max_grad_norm}")
                
                # Optimizer step with scaling
                self.mixed_precision.step_optimizer(self.optimizer)
                self.mixed_precision.update_scaler()
                
                # Track learning rates
                current_lr = self.optimizer.param_groups[0]['lr']
                learning_rates.append(current_lr)
                
                # Track metrics
                epoch_losses.append(loss_dict['total_loss'])
                for key in epoch_metrics:
                    epoch_metrics[key].append(loss_dict[key])
                
                # Enhanced logging: print every k batches, and at special batches for epoch 0
                should_log = (
                    (batch_idx % log_interval == 0)
                    or (
                        self.current_epoch == 0
                        and batch_idx in special_batches
                    )
                )
                if should_log:
                    cum_epoch_time = time.time() - epoch_start_time
                    batch_size = boards.size(0) if hasattr(boards, 'size') else 'N/A'
                    msg = (f"[Epoch {self.current_epoch}][Batch {batch_idx+1}/{n_batches}] "
                           f"Total Loss: {loss_dict['total_loss']:.4f}, "
                           f"Policy: {loss_dict['policy_loss']:.4f}, "
                           f"Value: {loss_dict['value_loss']:.4f}, "
                           f"Grad Norm: {grad_norm_before_clip:.4f}, "
                           f"LR: {current_lr:.6f}, "
                           f"Batch time ({batch_size}): {time.time() - batch_start_time:.3f}s, "
                           f"Data load: {batch_data_time:.3f}s, "
                           f"Cumulative epoch time: {cum_epoch_time:.1f}s")
                    log_and_print(msg)
                
                # Prepare for next batch data timing
                data_load_start = time.time()
                batch_end_time = time.time()
                batch_times.append(batch_end_time - batch_start_time)
        except IndexError:
            log_and_print("[INFO] No more data in dataset. Ending epoch early.")
            pass
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        # Compute epoch averages
        epoch_avg = {key: np.mean(values) for key, values in epoch_metrics.items()}
        # Add timing info to epoch_avg
        epoch_avg['epoch_time'] = epoch_time
        epoch_avg['batch_time_mean'] = np.mean(batch_times) if batch_times else 0
        epoch_avg['batch_time_min'] = np.min(batch_times) if batch_times else 0
        epoch_avg['batch_time_max'] = np.max(batch_times) if batch_times else 0
        epoch_avg['data_load_time_mean'] = np.mean(batch_data_times) if batch_data_times else 0
        
        # Enhanced diagnostics summary
        if gradient_norms:
            epoch_avg['grad_norm_mean'] = np.mean(gradient_norms)
            epoch_avg['grad_norm_max'] = np.max(gradient_norms)
            epoch_avg['grad_norm_min'] = np.min(gradient_norms)
            epoch_avg['grad_norm_std'] = np.std(gradient_norms)
        
        if learning_rates:
            epoch_avg['lr_mean'] = np.mean(learning_rates)
            epoch_avg['lr_min'] = np.min(learning_rates)
            epoch_avg['lr_max'] = np.max(learning_rates)
            epoch_avg['lr_std'] = np.std(learning_rates)
        
        epoch_avg['loss_spikes_count'] = len(loss_spikes)
        
        # Log timing summary
        samples_per_sec = float('nan')  # Unknown for streaming datasets
        summary_msg = (f"[Epoch {self.current_epoch}] DONE: epoch_time={epoch_time:.2f}s, "
                       f"batch_time_mean={epoch_avg['batch_time_mean']:.3f}s, "
                       f"batch_time_min={epoch_avg['batch_time_min']:.3f}s, "
                       f"batch_time_max={epoch_avg['batch_time_max']:.3f}s, "
                       f"data_load_time_mean={epoch_avg['data_load_time_mean']:.3f}s, "
                       f"samples/sec={samples_per_sec:.1f}")
        log_and_print(summary_msg)
        
        # Enhanced diagnostic summary
        if gradient_norms:
            diagnostic_summary = (f"DIAGNOSTICS: grad_norm_mean={epoch_avg['grad_norm_mean']:.4f}, "
                                f"grad_norm_max={epoch_avg['grad_norm_max']:.4f}, "
                                f"grad_norm_std={epoch_avg['grad_norm_std']:.4f}, "
                                f"lr_mean={epoch_avg['lr_mean']:.6f}, "
                                f"loss_spikes={epoch_avg['loss_spikes_count']}")
            log_diagnostic(diagnostic_summary)
        
        return epoch_avg
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if not self.val_loader:
            return {}
        
        self.model.eval()
        val_losses = []
        val_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': []
        }
        
        with torch.no_grad():
            for boards, policies, values in self.val_loader:
                # Move to device
                boards = boards.to(self.device)
                policies = policies.to(self.device)
                values = values.to(self.device)
                
                # Forward pass with mixed precision
                with self.mixed_precision.autocast_context():
                    policy_pred, value_pred = self.model(boards)
                    total_loss, loss_dict = self.criterion(policy_pred, value_pred, policies, values)
                
                # Track metrics
                val_losses.append(loss_dict['total_loss'])
                for key in val_metrics:
                    val_metrics[key].append(loss_dict[key])
        
        # Compute validation averages
        val_avg = {key: np.mean(values) for key, values in val_metrics.items()}
        return val_avg
    
    # NOTE: The train() method has been removed as it was dead code.
    # The hyperparameter sweep uses train_on_batches() via MiniEpochOrchestrator.
    # If you need epoch-based training, use MiniEpochOrchestrator or implement
    # a new training loop that calls train_on_batches().
    
    def save_checkpoint(self, path: Path, train_metrics: Dict, val_metrics: Dict, compress: bool = True):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save the checkpoint
            train_metrics: Training metrics for this checkpoint
            val_metrics: Validation metrics for this checkpoint
            compress: Whether to save as gzipped file (.pt.gz)
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_val_loss': self.best_val_loss,
            'mixed_precision': self.mixed_precision.use_mixed_precision
        }
        
        if compress:
            # Ensure path has .pt.gz extension
            if not str(path).endswith('.pt.gz'):
                path = path.with_suffix('.pt.gz')
            
            # Save as gzipped file
            import gzip
            with gzip.open(path, 'wb') as f:
                torch.save(checkpoint, f)
        else:
            # Save as uncompressed file
            torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path, override_checkpoint_hyperparameters: bool = False):
        """
        Load model checkpoint.
        
        Args:
            path: Path to the checkpoint file
            override_checkpoint_hyperparameters: If True, reset optimizer state to use current hyperparameters
                                               instead of checkpoint hyperparameters. This ensures clean
                                               hyperparameter experiments but may affect training stability.
        """
        # Check if file is gzipped by reading the first two bytes
        def is_gzipped(filepath):
            with open(filepath, 'rb') as f:
                return f.read(2) == b'\x1f\x8b'
        
        if is_gzipped(path):
            import gzip
            with gzip.open(path, 'rb') as f:
                checkpoint = torch.load(f, map_location=self.device, weights_only=False)
        else:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if override_checkpoint_hyperparameters:
            logger.warning("Overriding checkpoint hyperparameters - optimizer state will be reset")
            logger.info(f"Using hyperparameter learning rate: {self.original_learning_rate}")
            logger.info(f"Using hyperparameter value_learning_rate_factor: {self.value_learning_rate_factor}")
            logger.info(f"Using hyperparameter value_weight_decay_factor: {self.value_weight_decay_factor}")
            # Don't load optimizer state - let it use current hyperparameters
        else:
            # Load optimizer state (preserves checkpoint hyperparameters)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} with checkpoint hyperparameters")
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']

    def _save_checkpoint_smart(self, save_path: Path, epoch: int, 
                              train_metrics: Dict, val_metrics: Dict,
                              max_checkpoints: int, compress_checkpoints: bool):
        # Save regular checkpoint
        fname = f"epoch{epoch+1}_mini1.pt"
        path = save_path / fname
        self.save_checkpoint(path, train_metrics, val_metrics)
        # Save best model if needed
        if val_metrics and val_metrics['total_loss'] < self.best_val_loss:
            best_path = save_path / "best_model.pt"
            self.save_checkpoint(best_path, train_metrics, val_metrics)
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints(save_path, max_checkpoints, compress_checkpoints)

    def _get_checkpoints_to_keep(self, max_epoch: int, max_checkpoints: int) -> set:
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

    def _cleanup_old_checkpoints(self, save_path: Path, max_checkpoints: int, compress_checkpoints: bool):
        # Find all checkpoint files except best_model.pt
        all_ckpts = [f for f in os.listdir(save_path) if re.match(r"epoch\d+_mini\d+\.pt", f)]
        # Extract epoch numbers
        epoch_nums = []
        for fname in all_ckpts:
            m = re.match(r"epoch(\d+)_mini(\d+)\.pt", fname)
            if m:
                epoch_nums.append((int(m.group(1)), fname))
        if not epoch_nums:
            return
        max_epoch = max(e for e, _ in epoch_nums)
        keep_epochs = self._get_checkpoints_to_keep(max_epoch, max_checkpoints)
        for e, fname in epoch_nums:
            if e not in keep_epochs:
                try:
                    os.remove(os.path.join(save_path, fname))
                except Exception:
                    pass

    def train_on_batches(self, batch_iterable, epoch=None, mini_epoch=None, val_metrics=None) -> Dict[str, float]:
        """
        Train the model on a provided iterable of batches (mini-epoch).

        This is a lower-level training method that processes a specific set of batches
        without managing the overall training loop. It's designed for:
        - Mini-epoch orchestration (see MiniEpochOrchestrator)
        - Custom training loops that need fine-grained control
        - Integration with external training frameworks
        - Debugging and experimentation

        Unlike train(), this method:
        - Does NOT manage epochs, checkpointing, or validation
        - Does NOT reset model/optimizer state between calls
        - Can be called multiple times within a single epoch
        - Returns metrics for just the processed batches

        Args:
            batch_iterable: An iterable yielding (boards, policies, values) batches.
            epoch: Current epoch number (int, optional, for debugging/dumping purposes)
            mini_epoch: Current mini-epoch number (int, optional, for debugging/dumping purposes)

        Returns:
            Dictionary of average losses for the mini-epoch (policy_loss, value_loss, total_loss).

        Usage:
            # For standard training, use train() instead
            # For mini-epoch orchestration:
            orchestrator = MiniEpochOrchestrator(trainer, train_loader, val_loader, mini_epoch_batches=500)
            orchestrator.run()
        """
        self.model.train()
        mini_epoch_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': []
        }
        # Track gradient norms for diagnostic purposes
        gradient_norms = []
        
        # --- Progress logging setup ---
        start_time = time.time()
        next_log_batch = 1
        log_interval_sec = 180  # 3 minutes
        last_time_log = start_time
        # Only print device info the first time this method is called per Trainer instance
        if not hasattr(self, '_train_on_batches_logged_device'):
            print("[train_on_batches] Device info:")
            print(f"  self.device = {self.device}")
            print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
            if hasattr(torch.backends, 'mps'):
                print(f"  torch.backends.mps.is_available() = {torch.backends.mps.is_available()}")
                print(f"  torch.backends.mps.is_built() = {torch.backends.mps.is_built()}")
            print(f"  Model device: {next(self.model.parameters()).device}")
            self._train_on_batches_logged_device = True
        data_load_start = time.time()
        batch_data_times = []  # Track data loading times for each batch
        batch_times = [] # Initialize batch_times here
        for batch_idx, (boards, policies, values) in enumerate(batch_iterable):
            # --- DEBUG: Dump first batch of epoch 0, mini_epoch 0 ---
            if (epoch == 0 and mini_epoch == 0 and batch_idx == 0):
                os.makedirs('analysis/debugging/value_head_performance', exist_ok=True)
                now = datetime.now()
                date_str = now.strftime("%Y-%m-%d")
                time_str = now.strftime("%H-%M")
                debug_filename = f'analysis/debugging/value_head_performance/batch0_epoch0_{date_str}_{time_str}.pkl'
                # Save both device and cpu/numpy forms for debugging
                batch_dict = {
                    'boards': boards,
                    'policies': policies,
                    'values': values,
                    'boards_cpu': boards.cpu(),
                    'policies_cpu': policies.cpu(),
                    'values_cpu': values.cpu(),
                    'boards_np': boards.cpu().numpy(),
                    'policies_np': policies.cpu().numpy(),
                    'values_np': values.cpu().numpy(),
                    'meta': {
                        'epoch': epoch,
                        'mini_epoch': mini_epoch,
                        'batch_idx': batch_idx,
                        'shape': {
                            'boards': tuple(boards.shape),
                            'policies': tuple(policies.shape),
                            'values': tuple(values.shape),
                        },
                        'dtype': {
                            'boards': str(boards.dtype),
                            'policies': str(policies.dtype),
                            'values': str(values.dtype),
                        },
                    }
                }
                with open(debug_filename, 'wb') as f:
                    pickle.dump(batch_dict, f)
                print(f"[DEBUG] Dumped first batch of epoch 0, mini_epoch 0 to {debug_filename}")
                # TODO: Remove or refactor this debug dumping logic after value head debugging is complete.
            # Time data loading
            data_load_end = time.time()
            batch_data_time = data_load_end - data_load_start
            batch_data_times.append(batch_data_time)
            batch_start_time = time.time()
            # Move to device
            boards = boards.to(self.device)
            policies = policies.to(self.device)
            values = values.to(self.device)
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            with self.mixed_precision.autocast_context():
                policy_pred, value_pred = self.model(boards)
                total_loss, loss_dict = self.criterion(policy_pred, value_pred, policies, values)
            # Backward pass: compute gradients for this batch
            scaled_loss = self.mixed_precision.scale_loss(total_loss)
            scaled_loss.backward()
            
            # Calculate gradient norm before clipping (for diagnostic purposes)
            pre_clip_gradient_norm = None
            try:
                total_norm = 0.0
                param_count = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                if param_count > 0:
                    pre_clip_gradient_norm = total_norm ** (1. / 2)
                    gradient_norms.append(pre_clip_gradient_norm)
            except Exception as e:
                print(f"[train_on_batches] Warning: Failed to calculate pre-clip gradient norm: {e}")
            
            # Clip gradients to avoid exploding gradients (if configured)
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                
                # Calculate gradient norm after clipping (to verify clipping worked)
                post_clip_gradient_norm = None
                try:
                    total_norm = 0.0
                    param_count = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                            param_count += 1
                    if param_count > 0:
                        post_clip_gradient_norm = total_norm ** (1. / 2)
                except Exception as e:
                    print(f"[train_on_batches] Warning: Failed to calculate post-clip gradient norm: {e}")
                
                # Store both values for logging
                if pre_clip_gradient_norm is not None and post_clip_gradient_norm is not None:
                    gradient_norms.append(post_clip_gradient_norm)  # Use post-clip for statistics
                    # Store both values for debugging
                    if not hasattr(self, 'gradient_clipping_debug'):
                        self.gradient_clipping_debug = []
                    self.gradient_clipping_debug.append({
                        'pre_clip': pre_clip_gradient_norm,
                        'post_clip': post_clip_gradient_norm,
                        'clipped': pre_clip_gradient_norm > post_clip_gradient_norm
                    })
            
            # Optimizer step: update model parameters using accumulated gradients
            self.mixed_precision.step_optimizer(self.optimizer)
            # Update mixed precision scaler (if used)
            self.mixed_precision.update_scaler()
            # Track losses for this batch
            for key in mini_epoch_metrics:
                mini_epoch_metrics[key].append(loss_dict[key])
            # --- Progress logging ---
            now = time.time()
            should_log = False
            if self.current_epoch == 0 and mini_epoch == 0:
                # For first epoch, log for all powers of 2
                if batch_idx + 1 == next_log_batch:
                    should_log = True
            else:
                # For later epochs, only log for batch >= 64
                if batch_idx + 1 >= 64 and batch_idx + 1 == next_log_batch:
                    should_log = True
            # After 3 minutes, switch to time-based logging every log_interval_sec
            if not should_log and (now - last_time_log > log_interval_sec):
                should_log = True
                last_time_log = now
            if should_log:
                elapsed = now - start_time
                print(
                    f"[train_on_batches] Batch {batch_idx+1}: "
                    f"total_loss={loss_dict['total_loss']:.4f}, "
                    f"policy_loss={loss_dict['policy_loss']:.4f}, "
                    f"value_loss={loss_dict['value_loss']:.4f} "
                    f"(elapsed {elapsed:.1f}s)"
                )
                if batch_idx + 1 == next_log_batch:
                    exp_backoff = 2
                    if batch_idx > 64:
                        exp_backoff = 1.5
                    next_log_batch *= math.floor(exp_backoff)  # Exponential backoff
            # --- DUMP FINAL BATCH FOR DETAILED DEBUGGING ---
            is_last_batch = (batch_idx == len(batch_iterable) - 1)
            if (epoch == 0 and mini_epoch == 0 and is_last_batch):
                os.makedirs('analysis/debugging/value_head_performance', exist_ok=True)
                now = datetime.now()
                date_str = now.strftime("%Y-%m-%d")
                time_str = now.strftime("%H-%M")
                debug_filename = f'analysis/debugging/value_head_performance/batch{batch_idx}_epoch{epoch}_mini{mini_epoch}_{date_str}_{time_str}_detailed.pkl'
                batch_dict = {
                    'boards': boards.cpu(),
                    'policies': policies.cpu(),
                    'values': values.cpu(),
                    'policy_logits': policy_pred.detach().cpu(),
                    'value_logits': value_pred.detach().cpu(),
                    'policy_logits_np': policy_pred.detach().cpu().numpy(),
                    'value_logits_np': value_pred.detach().cpu().numpy(),
                    'policies_np': policies.cpu().numpy(),
                    'values_np': values.cpu().numpy(),
                    'meta': {
                        'epoch': epoch,
                        'mini_epoch': mini_epoch,
                        'batch_idx': batch_idx,
                        'shape': {
                            'boards': tuple(boards.shape),
                            'policies': tuple(policies.shape),
                            'values': tuple(values.shape),
                        },
                        'dtype': {
                            'boards': str(boards.dtype),
                            'policies': str(policies.dtype),
                            'values': str(values.dtype),
                        },
                        'loss_dict': loss_dict,
                    }
                }
                with open(debug_filename, 'wb') as f:
                    pickle.dump(batch_dict, f)
                print(f"[DEBUG] Dumped detailed final batch to {debug_filename}")
            # Prepare for next batch data timing
            data_load_start = time.time()
            batch_end_time = time.time()
            batch_times.append(batch_end_time - batch_start_time)
        # Compute averages for the mini-epoch
        mini_epoch_avg = {key: float(np.mean(values)) if values else float('nan') for key, values in mini_epoch_metrics.items()}
        
        # Calculate diagnostic metrics for stability analysis
        gradient_norm = None
        post_clip_gradient_norm = None
        gradient_stats = None
        weight_stats = None
        lr_stats = None
        gpu_memory_mb = None
        best_val_loss = None
        
        # Update best validation loss if validation metrics are provided
        if val_metrics and 'total_loss' in val_metrics:
            current_val_loss = val_metrics['total_loss']
            if not hasattr(self, 'best_val_loss') or self.best_val_loss is None:
                self.best_val_loss = current_val_loss
            elif current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
            best_val_loss = self.best_val_loss
        
        # Calculate gradient statistics from collected norms
        if gradient_norms:
            gradient_norm = gradient_norms[-1]  # Use the last gradient norm (post-clip)
            post_clip_gradient_norm = gradient_norm  # This is the post-clip value
            gradient_stats = {
                'mean': float(np.mean(gradient_norms)),
                'min': float(np.min(gradient_norms)),
                'max': float(np.max(gradient_norms)),
                'std': float(np.std(gradient_norms))
            }
        
        # Calculate weight statistics
        weight_norms = []
        for p in self.model.parameters():
            if p.data is not None:
                weight_norms.append(p.data.norm(2).item())
        if weight_norms:
            weight_stats = {
                'mean': float(np.mean(weight_norms)),
                'std': float(np.std(weight_norms))
            }
        
        # Calculate learning rate statistics
        lr_values = [group['lr'] for group in self.optimizer.param_groups if 'lr' in group]
        if lr_values:
            lr_stats = {
                'mean': float(np.mean(lr_values)),
                'min': float(np.min(lr_values)),
                'max': float(np.max(lr_values)),
                'std': float(np.std(lr_values))
            }
        
        # Calculate GPU memory usage
        if torch.cuda.is_available() and hasattr(torch.cuda, 'memory_allocated'):
            gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS doesn't have memory_allocated, so we'll skip this
            pass
        
        # CSV logging for mini-epoch
        if self.csv_logger:
            # Extract hyperparameters as in MiniEpochOrchestrator
            hp = {
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'batch_size': self.train_loader.batch_size,
                'dataset_size': 'N/A',
                'network_structure': f"ResNet{getattr(self.model, 'resnet_depth', '?')}",
                'policy_weight': getattr(self.criterion, 'policy_weight', ''),
                'value_weight': getattr(self.criterion, 'value_weight', ''),
                'total_loss_weight': getattr(self.criterion, 'policy_weight', 0) + getattr(self.criterion, 'value_weight', 0),
                'dropout_prob': getattr(self.model, 'dropout', type('dummy', (), {'p': ''})) .p if hasattr(self.model, 'dropout') else '',
                'weight_decay': self.optimizer.param_groups[0].get('weight_decay', 0.0),
                'max_grad_norm': getattr(self, 'max_grad_norm', ''),
                'value_learning_rate_factor': getattr(self, 'value_learning_rate_factor', ''),
                'value_weight_decay_factor': getattr(self, 'value_weight_decay_factor', '')
            }
            epoch_id = f"{epoch+1}_mini{mini_epoch+1}" if epoch is not None and mini_epoch is not None else "unknown"
            
            # Calculate training time for this mini-epoch
            mini_epoch_time = sum(batch_times)
            
            self.csv_logger.log_mini_epoch(
                epoch=epoch_id,
                train_metrics=mini_epoch_avg,
                val_metrics=val_metrics,  # Pass val_metrics here
                hyperparams=hp,
                training_time=mini_epoch_time,
                epoch_time=mini_epoch_time,
                samples_per_second=0.0,  # Would need to calculate based on samples processed
                memory_usage_mb=0.0,  # Would need to calculate system memory usage
                gpu_memory_mb=gpu_memory_mb,
                gradient_norm=gradient_norm,
                post_clip_gradient_norm=post_clip_gradient_norm,
                weight_stats=weight_stats,
                gradient_stats=gradient_stats,  # Add the missing parameter
                lr_stats=lr_stats,  # Add the missing parameter
                best_val_loss=best_val_loss,  # Add the best validation loss
                notes="train_on_batches"
            )
        
        return mini_epoch_avg
