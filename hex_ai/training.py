"""
Training module for Hex AI models.

This module provides training loops, loss computation, checkpointing,
and progress tracking for the two-headed ResNet models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from pathlib import Path
import time

from .models import TwoHeadedResNet
from .config import (
    BOARD_SIZE, POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE,
    LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS
)
from hex_ai.data_pipeline import StreamingProcessedDataset, discover_processed_files

from hex_ai.training_utils import get_device
DEVICE = get_device()

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
        # Value loss is always computed (MSE)
        value_loss = self.value_loss(value_pred.squeeze(), value_target.squeeze())
        
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
                    from torch.cuda.amp import autocast, GradScaler
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
            from contextlib import nullcontext
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
                 device: str = DEVICE,
                 enable_system_analysis: bool = True,
                 enable_csv_logging: bool = True,
                 experiment_name: Optional[str] = None,
                 policy_weight: float = POLICY_LOSS_WEIGHT,
                 value_weight: float = VALUE_LOSS_WEIGHT,
                 weight_decay: float = 1e-4):
        logger.debug(f"[Trainer.__init__] device argument = {device}")
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Initialize mixed precision
        self.mixed_precision = MixedPrecisionTrainer(device)
        
        # Optimizer and loss
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
            from .training_logger import TrainingLogger
            self.csv_logger = TrainingLogger(experiment_name=experiment_name)
        
        # System analysis
        if enable_system_analysis:
            self._run_system_analysis()
        
        logger.info(f"Initialized trainer with {len(train_loader)} training batches")
        if val_loader:
            logger.info(f"Validation set with {len(val_loader)} batches")
    
    def _run_system_analysis(self):
        """Run system analysis and log recommendations."""
        try:
            from .system_utils import get_system_info, calculate_optimal_batch_size
            
            system_info = get_system_info()
            optimal_batch_size, batch_analysis = calculate_optimal_batch_size()
            
            logger.info("=== System Analysis ===")
            logger.info(f"Platform: {system_info['platform']}")
            logger.info(f"Memory: {system_info['memory_available_gb']:.1f} GB available")
            logger.info(f"GPU: {'Available' if system_info['gpu_available'] else 'Not available'}")
            logger.info(f"Old notion of optimal batch size, from calculate_optimal_batch_size: {batch_analysis['optimal_batch_size']}")
            logger.info(f"Current batch size: {self.train_loader.batch_size}")
            
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
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch, with detailed timing logs."""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': []
        }
        batch_times = []
        batch_data_times = []
        epoch_start_time = time.time()
        data_load_start = time.time()
        for batch_idx, (boards, policies, values) in enumerate(self.train_loader):
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
            # Backward pass with scaling
            scaled_loss = self.mixed_precision.scale_loss(total_loss)
            scaled_loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            # Optimizer step with scaling
            self.mixed_precision.step_optimizer(self.optimizer)
            self.mixed_precision.update_scaler()
            # Track metrics
            epoch_losses.append(loss_dict['total_loss'])
            for key in epoch_metrics:
                epoch_metrics[key].append(loss_dict[key])
            # Log progress - adjust frequency based on dataset size and verbosity
            from .config import VERBOSE_LEVEL
            if VERBOSE_LEVEL >= 2:  # Only log batches if verbose level is 2 or higher
                # With only 100 or fewer batches, log every 5 batches
                # Between 101 and 2000 batches, log every 50 batches
                # Above 2000 batches, log every 200 batches
                log_interval = 5 if len(self.train_loader) <= 100 else 50 if len(self.train_loader) <= 2000 else 200
                if batch_idx % log_interval == 0:
                    logger.info(f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                              f"Loss: {loss_dict['total_loss']:.4f}, Batch time: {time.time() - batch_start_time:.3f}s, Data load: {batch_data_time:.3f}s")
            # Prepare for next batch data timing
            data_load_start = time.time()
            batch_end_time = time.time()
            batch_times.append(batch_end_time - batch_start_time)
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
        # Log timing summary
        logger.info(f"Epoch {self.current_epoch} timing: epoch_time={epoch_time:.2f}s, batch_time_mean={epoch_avg['batch_time_mean']:.3f}s, batch_time_min={epoch_avg['batch_time_min']:.3f}s, batch_time_max={epoch_avg['batch_time_max']:.3f}s, data_load_time_mean={epoch_avg['data_load_time_mean']:.3f}s")
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
    
    def train(self, num_epochs: int, save_dir: str = "checkpoints", 
              max_checkpoints: int = 5, compress_checkpoints: bool = True,
              early_stopping: Optional[EarlyStopping] = None) -> Dict:
        """Train the model for specified number of epochs."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Start timing
        self.start_time = datetime.now()
        
        # Initialize loss tracking
        train_losses = []
        val_losses = []
        train_policy_losses = []
        train_value_losses = []
        val_policy_losses = []
        val_value_losses = []
        epoch_times = []  # Store epoch times for later analysis
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Checkpoint management: max {max_checkpoints} checkpoints, compression: {compress_checkpoints}")
        if early_stopping:
            logger.info(f"Early stopping enabled with patience {early_stopping.patience}")
        
        early_stopped = False
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Track losses
            train_losses.append(train_metrics['total_loss'])
            train_policy_losses.append(train_metrics['policy_loss'])
            train_value_losses.append(train_metrics['value_loss'])
            
            if val_metrics:
                val_losses.append(val_metrics['total_loss'])
                val_policy_losses.append(val_metrics['policy_loss'])
                val_value_losses.append(val_metrics['value_loss'])
            else:
                val_losses.append(float('inf'))  # No validation data
                val_policy_losses.append(float('inf'))
                val_value_losses.append(float('inf'))
            
            # Learning rate scheduler step (use validation loss if available, else training loss)
            val_loss_for_scheduler = val_metrics['total_loss'] if val_metrics else train_metrics['total_loss']
            self.scheduler.step(val_loss_for_scheduler)
            logger.info(f"Current learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Calculate timing and performance metrics
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            total_training_time = (datetime.now() - self.start_time).total_seconds()
            
            # Calculate samples per second
            total_samples = len(self.train_loader.dataset)
            samples_per_second = total_samples / epoch_time if epoch_time > 0 else 0
            
            # Get memory usage
            from .training_logger import get_memory_usage, get_gpu_memory_usage, get_weight_statistics, get_gradient_norm
            memory_usage_mb = get_memory_usage()
            gpu_memory_mb = get_gpu_memory_usage()
            weight_stats = get_weight_statistics(self.model)
            gradient_norm = get_gradient_norm(self.model)
            
            # Prepare hyperparameters for logging
            hyperparams = {
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'batch_size': self.train_loader.batch_size,
                'dataset_size': len(self.train_loader.dataset),
                'network_structure': f"ResNet{self.model.resnet_depth}",
                'policy_weight': self.criterion.policy_weight,
                'value_weight': self.criterion.value_weight,
                'total_loss_weight': self.criterion.policy_weight + self.criterion.value_weight,
                'dropout_prob': self.model.dropout.p,
                'weight_decay': self.optimizer.param_groups[0].get('weight_decay', 0.0)
            }
            
            # Log to CSV if enabled
            if self.csv_logger:
                self.csv_logger.log_epoch(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    hyperparams=hyperparams,
                    training_time=total_training_time,
                    epoch_time=epoch_time,
                    samples_per_second=samples_per_second,
                    memory_usage_mb=memory_usage_mb,
                    gpu_memory_mb=gpu_memory_mb,
                    gradient_norm=gradient_norm,
                    weight_stats=weight_stats,
                    notes=f"Epoch {epoch} completed"
                )
            
            # Log results with memory info
            memory_usage_mb = get_memory_usage()
            logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['total_loss']:.4f} | Memory: {memory_usage_mb:.1f}MB")
            if val_metrics:
                logger.info(f"Epoch {epoch}: Val Loss: {val_metrics['total_loss']:.4f}")
            
            # Log memory warning if usage is high
            if memory_usage_mb > 8000:  # 8GB threshold
                logger.warning(f"High memory usage: {memory_usage_mb:.1f}MB - consider reducing batch size")
            
            # Save checkpoint with smart management
            self._save_checkpoint_smart(save_path, epoch, train_metrics, val_metrics, 
                                      max_checkpoints, compress_checkpoints)
            
            # Update best model if validation loss improved
            if val_metrics and val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.save_checkpoint(save_path / "best_model.pt", train_metrics, val_metrics)
                logger.info(f"New best model saved with val loss: {self.best_val_loss:.4f}")
            
            # Check early stopping
            if early_stopping and val_metrics:
                if early_stopping(val_metrics['total_loss'], self.model):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    early_stopped = True
                    break
        
        # Log experiment summary
        total_training_time = (datetime.now() - self.start_time).total_seconds()
        if self.csv_logger:
            self.csv_logger.log_experiment_summary(
                best_val_loss=self.best_val_loss,
                total_epochs=epoch + 1,
                total_training_time=total_training_time,
                early_stopped=early_stopped,
                notes=f"Training completed with {epoch + 1} epochs"
            )
        
        logger.info("Training completed!")
        
        # Print compact summary if we used compact logging
        from hex_ai.error_handling import get_board_state_error_tracker
        error_tracker = get_board_state_error_tracker()
        stats = error_tracker.get_stats()
        if stats['total_samples'] > 0:
            print(f"\nData loading summary: {stats['total_samples']} samples, {stats['error_count']} errors ({stats['error_rate']:.2%})")
        
        # Return comprehensive results
        return {
            "best_val_loss": self.best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_policy_losses": train_policy_losses,
            "train_value_losses": train_value_losses,
            "val_policy_losses": val_policy_losses,
            "val_value_losses": val_value_losses,
            "epochs_trained": len(train_losses),
            "early_stopped": early_stopped,
            "total_training_time": total_training_time,
            "epoch_times": epoch_times, # Return epoch times for analysis
            "data_stats": stats
        }
    
    def save_checkpoint(self, path: Path, train_metrics: Dict, val_metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_val_loss': self.best_val_loss,
            'mixed_precision': self.mixed_precision.use_mixed_precision
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def _save_checkpoint_smart(self, save_path: Path, epoch: int, 
                              train_metrics: Dict, val_metrics: Dict,
                              max_checkpoints: int, compress_checkpoints: bool):
        """Save checkpoint with smart management to control storage usage."""
        import gzip
        import pickle
        
        # Create checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_val_loss': self.best_val_loss,
            'mixed_precision': self.mixed_precision.use_mixed_precision
        }
        
        # Save current checkpoint
        checkpoint_file = save_path / f"checkpoint_epoch_{epoch}.pt"
        if compress_checkpoints:
            # Save compressed checkpoint
            with gzip.open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
        else:
            # Save uncompressed checkpoint
            torch.save(checkpoint_data, checkpoint_file)
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints(save_path, max_checkpoints, compress_checkpoints)
    
    def _cleanup_old_checkpoints(self, save_path: Path, max_checkpoints: int, compress_checkpoints: bool):
        """Remove old checkpoints using smart retention strategy."""
        # Get all checkpoint files
        if compress_checkpoints:
            checkpoint_files = list(save_path.glob("checkpoint_epoch_*.pt"))
        else:
            checkpoint_files = list(save_path.glob("checkpoint_epoch_*.pt"))
        
        # Sort by epoch number
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        if len(checkpoint_files) <= max_checkpoints:
            return  # No cleanup needed
        
        # Get epochs from checkpoint filenames
        epochs = [int(f.stem.split('_')[-1]) for f in checkpoint_files]
        max_epoch = max(epochs)
        
        # Determine which checkpoints to keep based on smart strategy
        keep_epochs = self._get_checkpoints_to_keep(max_epoch, max_checkpoints)
        
        # Find checkpoints to delete
        files_to_delete = []
        for checkpoint_file in checkpoint_files:
            epoch = int(checkpoint_file.stem.split('_')[-1])
            if epoch not in keep_epochs:
                files_to_delete.append(checkpoint_file)
        
        # Delete old checkpoints
        for file_path in files_to_delete:
            try:
                file_path.unlink()
                logger.debug(f"Deleted checkpoint: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}: {e}")
        
        logger.info(f"Smart cleanup: deleted {len(files_to_delete)} checkpoints, keeping epochs {keep_epochs}")
    
    def _get_checkpoints_to_keep(self, max_epoch: int, max_checkpoints: int) -> set:
        """Get the epochs to keep based on smart retention strategy."""
        if max_checkpoints <= 5:
            # For small numbers, keep all recent checkpoints
            return set(range(max(0, max_epoch - max_checkpoints + 1), max_epoch + 1))
        
        # Smart strategy for larger numbers
        keep_epochs = set()
        
        # Always keep the latest few
        keep_epochs.update([max_epoch, max_epoch - 1, max_epoch - 2])
        
        # Add strategic samples from earlier epochs
        if max_epoch >= 20:
            # Your specific scheme for N=20: [20, 19, 18, 16, 13, 9, 4]
            if 4 <= max_epoch:
                keep_epochs.add(4)
            if max_epoch == 20:
                keep_epochs.update([16, 13, 9, 4])
            else:
                # Generalize the pattern: recent + strategic samples
                keep_epochs.update([max_epoch - 4, max_epoch - 7, max_epoch - 11, max_epoch - 16])
        else:
            # For smaller numbers, sample more densely
            step = max(1, max_epoch // max_checkpoints)
            for i in range(0, max_epoch - 2, step):
                if len(keep_epochs) < max_checkpoints:
                    keep_epochs.add(i)
        
        # Always keep epoch 0 (baseline)
        if 0 <= max_epoch:
            keep_epochs.add(0)
        
        # Ensure we don't exceed max_checkpoints
        keep_epochs = set(sorted(keep_epochs)[-max_checkpoints:])
        
        return keep_epochs


def create_trainer(model: TwoHeadedResNet, 
                  train_shard_files: List[Path],
                  val_shard_files: Optional[List[Path]] = None,
                  batch_size: int = BATCH_SIZE,
                  learning_rate: float = LEARNING_RATE,
                  device: str = DEVICE,
                  enable_system_analysis: bool = True) -> Trainer:
    """Create a trainer with data loaders from processed shard files."""
    # Use StreamingProcessedDataset for all data loading
    train_dataset = StreamingProcessedDataset(train_shard_files)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = None
    if val_shard_files:
        val_dataset = StreamingProcessedDataset(val_shard_files)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    trainer = Trainer(model, train_loader, val_loader, learning_rate, device, enable_system_analysis)
    return trainer


def train_model(model: TwoHeadedResNet,
                train_shard_files: List[Path],
                val_shard_files: Optional[List[Path]] = None,
                num_epochs: int = NUM_EPOCHS,
                batch_size: int = BATCH_SIZE,
                learning_rate: float = LEARNING_RATE,
                save_dir: str = "checkpoints",
                device: str = DEVICE,
                enable_system_analysis: bool = True,
                early_stopping_patience: Optional[int] = None) -> Dict:
    """Convenience function to train a model with processed shard files."""
    
    trainer = create_trainer(model, train_shard_files, val_shard_files, batch_size, learning_rate, device, enable_system_analysis)
    
    # Set up early stopping if requested
    early_stopping = None
    if early_stopping_patience is not None:
        early_stopping = EarlyStopping(patience=early_stopping_patience)
    
    return trainer.train(num_epochs, save_dir, early_stopping=early_stopping) 


def resume_training(checkpoint_path: str, 
                   num_epochs: int = 10,
                   save_dir: str = "checkpoints",
                   max_checkpoints: int = 5,
                   compress_checkpoints: bool = True) -> Dict:
    """
    Resume training from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        num_epochs: Number of additional epochs to train
        save_dir: Directory to save new checkpoints
        max_checkpoints: Maximum number of checkpoints to keep
        compress_checkpoints: Whether to compress checkpoints
        
    Returns:
        Training results dictionary
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint to get model and training state
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract model and training info
    model_state = checkpoint['model_state_dict']
    optimizer_state = checkpoint['optimizer_state_dict']
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    
    # Create model (we need to know the architecture)
    from .models import TwoHeadedResNet
    model = TwoHeadedResNet()  # Default architecture
    
    # Create trainer with dummy data (will be overridden)
    dummy_shard_files = [Path("dummy_shard.pkl.gz")]
    trainer = create_trainer(model, dummy_shard_files, enable_system_analysis=False)
    
    # Load the checkpoint state
    trainer.model.load_state_dict(model_state)
    trainer.optimizer.load_state_dict(optimizer_state)
    trainer.current_epoch = start_epoch
    trainer.best_val_loss = best_val_loss
    
    logger.info(f"Resuming training from epoch {start_epoch}")
    logger.info(f"Best validation loss so far: {best_val_loss:.4f}")
    
    # Continue training
    return trainer.train(num_epochs, save_dir, max_checkpoints, compress_checkpoints) 