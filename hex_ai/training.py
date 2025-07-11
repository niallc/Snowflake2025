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

from .models import TwoHeadedResNet
from .dataset import HexDataset
from .config import (
    BOARD_SIZE, POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE,
    LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, DEVICE
)

logger = logging.getLogger(__name__)


class PolicyValueLoss(nn.Module):
    """Combined loss for policy and value heads."""
    
    def __init__(self, policy_weight: float = 1.0, value_weight: float = 1.0):
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.policy_loss = nn.CrossEntropyLoss()
        self.value_loss = nn.MSELoss()
    
    def forward(self, policy_pred: torch.Tensor, value_pred: torch.Tensor,
                policy_target: torch.Tensor, value_target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined policy and value loss.
        
        Args:
            policy_pred: Predicted policy logits (batch_size, policy_output_size)
            value_pred: Predicted value (batch_size, 1)
            policy_target: Target policy probabilities (batch_size, policy_output_size)
            value_target: Target value (batch_size, 1)
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses
        """
        # Policy loss (cross-entropy)
        policy_loss = self.policy_loss(policy_pred, policy_target)
        
        # Value loss (MSE)
        value_loss = self.value_loss(value_pred.squeeze(), value_target.squeeze())
        
        # Combined loss
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
        self.device = device
        self.use_mixed_precision = device == 'cuda'
        
        if self.use_mixed_precision:
            try:
                from torch.cuda.amp import autocast, GradScaler
                self.autocast = autocast
                self.scaler = GradScaler()
                logger.info("Mixed precision training enabled for GPU")
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
        if self.use_mixed_precision:
            return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(self, optimizer: optim.Optimizer):
        """Step optimizer with proper scaling."""
        if self.use_mixed_precision:
            self.scaler.step(optimizer)
        else:
            optimizer.step()
    
    def update_scaler(self):
        """Update gradient scaler."""
        if self.use_mixed_precision:
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
                 enable_system_analysis: bool = True):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Initialize mixed precision
        self.mixed_precision = MixedPrecisionTrainer(device)
        
        # Optimizer and loss
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = PolicyValueLoss()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        
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
            logger.info(f"Optimal batch size: {batch_analysis['optimal_batch_size']}")
            logger.info(f"Current batch size: {self.train_loader.batch_size}")
            
            # Warn if batch size is suboptimal
            if batch_analysis['optimal_batch_size'] > self.train_loader.batch_size:
                logger.warning(f"Consider increasing batch size to {batch_analysis['optimal_batch_size']} for better efficiency")
            
            # Warn about GPU usage
            if not system_info['gpu_available'] and self.device == 'cuda':
                logger.warning("CUDA device requested but no GPU available, falling back to CPU")
            
        except ImportError as e:
            logger.warning(f"System analysis unavailable: {e}")
        except Exception as e:
            logger.warning(f"System analysis failed: {e}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': []
        }
        
        for batch_idx, (boards, policies, values) in enumerate(self.train_loader):
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
            
            # Optimizer step with scaling
            self.mixed_precision.step_optimizer(self.optimizer)
            self.mixed_precision.update_scaler()
            
            # Track metrics
            epoch_losses.append(loss_dict['total_loss'])
            for key in epoch_metrics:
                epoch_metrics[key].append(loss_dict[key])
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                          f"Loss: {loss_dict['total_loss']:.4f}")
        
        # Compute epoch averages
        epoch_avg = {key: np.mean(values) for key, values in epoch_metrics.items()}
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
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Checkpoint management: max {max_checkpoints} checkpoints, compression: {compress_checkpoints}")
        if early_stopping:
            logger.info(f"Early stopping enabled with patience {early_stopping.patience}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Log results
            logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['total_loss']:.4f}")
            if val_metrics:
                logger.info(f"Epoch {epoch}: Val Loss: {val_metrics['total_loss']:.4f}")
            
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
                    break
        
        logger.info("Training completed!")
        return {"best_val_loss": self.best_val_loss}
    
    def save_checkpoint(self, path: Path, train_metrics: Dict, val_metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_val_loss': self.best_val_loss
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
            'best_val_loss': self.best_val_loss
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
        """Remove old checkpoints to keep storage under control."""
        # Get all checkpoint files
        if compress_checkpoints:
            checkpoint_files = list(save_path.glob("checkpoint_epoch_*.pt"))
        else:
            checkpoint_files = list(save_path.glob("checkpoint_epoch_*.pt"))
        
        # Sort by epoch number
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        # Keep only the most recent max_checkpoints
        if len(checkpoint_files) > max_checkpoints:
            files_to_delete = checkpoint_files[:-max_checkpoints]
            for file_path in files_to_delete:
                try:
                    file_path.unlink()
                    logger.debug(f"Deleted old checkpoint: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete {file_path}: {e}")
            
            logger.info(f"Cleaned up {len(files_to_delete)} old checkpoints, keeping {max_checkpoints} most recent")


def create_trainer(model: TwoHeadedResNet, 
                  train_data: List[str],
                  val_data: Optional[List[str]] = None,
                  batch_size: int = BATCH_SIZE,
                  learning_rate: float = LEARNING_RATE,
                  device: str = DEVICE,
                  enable_system_analysis: bool = True) -> Trainer:
    """Create a trainer with data loaders."""
    
    # Create datasets
    train_dataset = HexDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = None
    if val_data:
        val_dataset = HexDataset(val_data)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, learning_rate, device, enable_system_analysis)
    return trainer


def train_model(model: TwoHeadedResNet,
                train_data: List[str],
                val_data: Optional[List[str]] = None,
                num_epochs: int = NUM_EPOCHS,
                batch_size: int = BATCH_SIZE,
                learning_rate: float = LEARNING_RATE,
                save_dir: str = "checkpoints",
                device: str = DEVICE,
                enable_system_analysis: bool = True,
                early_stopping_patience: Optional[int] = None) -> Dict:
    """Convenience function to train a model."""
    
    trainer = create_trainer(model, train_data, val_data, batch_size, learning_rate, device, enable_system_analysis)
    
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
    dummy_data = [("http://www.trmph.com/hex/board#13,a1b2c3", "1")]
    trainer = create_trainer(model, dummy_data, enable_system_analysis=False)
    
    # Load the checkpoint state
    trainer.model.load_state_dict(model_state)
    trainer.optimizer.load_state_dict(optimizer_state)
    trainer.current_epoch = start_epoch
    trainer.best_val_loss = best_val_loss
    
    logger.info(f"Resuming training from epoch {start_epoch}")
    logger.info(f"Best validation loss so far: {best_val_loss:.4f}")
    
    # Continue training
    return trainer.train(num_epochs, save_dir, max_checkpoints, compress_checkpoints) 