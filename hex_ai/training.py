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


class Trainer:
    """Training manager for Hex AI models."""
    
    def __init__(self, model: TwoHeadedResNet, 
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 learning_rate: float = LEARNING_RATE,
                 device: str = DEVICE):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer and loss
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = PolicyValueLoss()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        
        logger.info(f"Initialized trainer with {len(train_loader)} training batches")
        if val_loader:
            logger.info(f"Validation set with {len(val_loader)} batches")
    
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
            
            # Forward pass
            self.optimizer.zero_grad()
            policy_pred, value_pred = self.model(boards)
            
            # Compute loss
            total_loss, loss_dict = self.criterion(policy_pred, value_pred, policies, values)
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
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
                
                # Forward pass
                policy_pred, value_pred = self.model(boards)
                
                # Compute loss
                total_loss, loss_dict = self.criterion(policy_pred, value_pred, policies, values)
                
                # Track metrics
                val_losses.append(loss_dict['total_loss'])
                for key in val_metrics:
                    val_metrics[key].append(loss_dict[key])
        
        # Compute validation averages
        val_avg = {key: np.mean(values) for key, values in val_metrics.items()}
        return val_avg
    
    def train(self, num_epochs: int, save_dir: str = "checkpoints") -> Dict:
        """Train the model for specified number of epochs."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
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
            
            # Save checkpoint
            self.save_checkpoint(save_path / f"checkpoint_epoch_{epoch}.pt", 
                               train_metrics, val_metrics)
            
            # Save best model
            if val_metrics and val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.save_checkpoint(save_path / "best_model.pt", 
                                   train_metrics, val_metrics)
                logger.info(f"New best model saved with val loss: {self.best_val_loss:.4f}")
            
            # Track history
            self.training_history.append({
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics
            })
        
        # Save training history
        with open(save_path / "training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info("Training completed!")
        return {
            'history': self.training_history,
            'best_val_loss': self.best_val_loss
        }
    
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
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


def create_trainer(model: TwoHeadedResNet, 
                  train_data: List[str],
                  val_data: Optional[List[str]] = None,
                  batch_size: int = BATCH_SIZE,
                  learning_rate: float = LEARNING_RATE,
                  device: str = DEVICE) -> Trainer:
    """Create a trainer with data loaders."""
    
    # Create datasets
    train_dataset = HexDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = None
    if val_data:
        val_dataset = HexDataset(val_data)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, learning_rate, device)
    return trainer


def train_model(model: TwoHeadedResNet,
                train_data: List[str],
                val_data: Optional[List[str]] = None,
                num_epochs: int = NUM_EPOCHS,
                batch_size: int = BATCH_SIZE,
                learning_rate: float = LEARNING_RATE,
                save_dir: str = "checkpoints",
                device: str = DEVICE) -> Dict:
    """Convenience function to train a model."""
    
    trainer = create_trainer(model, train_data, val_data, batch_size, learning_rate, device)
    return trainer.train(num_epochs, save_dir) 