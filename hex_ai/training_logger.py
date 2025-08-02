"""
Training logger for comprehensive CSV-based metric tracking.

This module provides CSV logging for all training metrics and hyperparameters
to enable detailed analysis of training runs without storage bloat.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class TrainingLogger:
    """Comprehensive CSV logger for training metrics and hyperparameters."""
    
    def __init__(self, log_file: str = "checkpoints/bookkeeping/training_metrics.csv", 
                 experiment_name: Optional[str] = None):
        """
        Initialize the training logger.
        
        Args:
            log_file: Path to the CSV file for logging
            experiment_name: Name of the current experiment
        """
        self.log_file = Path(log_file)
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create directory if it doesn't exist
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Define CSV headers
        self.headers = [
            # Experiment identification
            'experiment_name', 'date', 'timestamp', 'epoch',
            
            # Hyperparameters
            'learning_rate', 'batch_size', 'dataset_size', 'network_structure',
            'policy_weight', 'value_weight', 'total_loss_weight',
            'dropout_prob', 'weight_decay', 'max_grad_norm',
            'value_learning_rate_factor', 'value_weight_decay_factor',
            
            # Training metrics
            'policy_loss', 'value_loss', 'total_loss',
            'val_policy_loss', 'val_value_loss', 'val_total_loss',
            
            # Performance metrics
            'training_time', 'epoch_time', 'samples_per_second',
            'memory_usage_mb', 'gpu_memory_mb',
            
            # Training state
            'early_stopped', 'best_val_loss', 'epochs_trained',
            
            # Model statistics
            'gradient_norm', 'gradient_norm_mean', 'gradient_norm_max', 'gradient_norm_min', 'gradient_norm_std',
            'weight_norm_mean', 'weight_norm_std',
            'lr_mean', 'lr_min', 'lr_max', 'lr_std',
            'loss_spikes_count',
            
            # Additional info
            'notes'
        ]
        
        # Initialize CSV file if it doesn't exist
        self._initialize_csv()
        
        logger.info(f"Training logger initialized: {self.log_file}")
    
    def _initialize_csv(self):
        """Initialize the CSV file with headers if it doesn't exist."""
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
            logger.info(f"Created new training metrics file: {self.log_file}")
    
    def log_epoch(self, 
                  epoch: int,
                  train_metrics: Dict[str, float],
                  val_metrics: Optional[Dict[str, float]],
                  hyperparams: Dict[str, Any],
                  training_time: float,
                  epoch_time: float,
                  samples_per_second: float,
                  memory_usage_mb: float,
                  gpu_memory_mb: Optional[float] = None,
                  gradient_norm: Optional[float] = None,
                  weight_stats: Optional[Dict[str, float]] = None,
                  notes: str = "",
                  gradient_stats: Optional[Dict[str, float]] = None,
                  lr_stats: Optional[Dict[str, float]] = None,
                  loss_spikes_count: int = 0) -> None:
        """
        Log metrics for a single epoch.
        
        Args:
            epoch: Current epoch number
            train_metrics: Training loss metrics
            val_metrics: Validation loss metrics (optional)
            hyperparams: Hyperparameters dictionary
            training_time: Total training time so far
            epoch_time: Time for this epoch
            samples_per_second: Training speed
            memory_usage_mb: CPU memory usage
            gpu_memory_mb: GPU memory usage (optional)
            gradient_norm: L2 norm of gradients (optional)
            weight_stats: Model weight statistics (optional)
            notes: Additional notes
            gradient_stats: Detailed gradient statistics (optional)
            lr_stats: Learning rate statistics (optional)
            loss_spikes_count: Number of loss spikes detected (optional)
        """
        # Prepare row data
        row = {
            # Experiment identification
            'experiment_name': self.experiment_name,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            
            # Hyperparameters
            'learning_rate': hyperparams.get('learning_rate', ''),
            'batch_size': hyperparams.get('batch_size', ''),
            'dataset_size': hyperparams.get('dataset_size', ''),
            'network_structure': hyperparams.get('network_structure', ''),
            'policy_weight': hyperparams.get('policy_weight', ''),
            'value_weight': hyperparams.get('value_weight', ''),
            'total_loss_weight': hyperparams.get('total_loss_weight', ''),
            'dropout_prob': hyperparams.get('dropout_prob', ''),
            'weight_decay': hyperparams.get('weight_decay', ''),
            'max_grad_norm': hyperparams.get('max_grad_norm', ''),
            'value_learning_rate_factor': hyperparams.get('value_learning_rate_factor', ''),
            'value_weight_decay_factor': hyperparams.get('value_weight_decay_factor', ''),
            
            # Training metrics
            'policy_loss': train_metrics.get('policy_loss', ''),
            'value_loss': train_metrics.get('value_loss', ''),
            'total_loss': train_metrics.get('total_loss', ''),
            'val_policy_loss': val_metrics.get('policy_loss', '') if val_metrics else '',
            'val_value_loss': val_metrics.get('value_loss', '') if val_metrics else '',
            'val_total_loss': val_metrics.get('total_loss', '') if val_metrics else '',
            
            # Performance metrics
            'training_time': training_time,
            'epoch_time': epoch_time,
            'samples_per_second': samples_per_second,
            'memory_usage_mb': memory_usage_mb,
            'gpu_memory_mb': gpu_memory_mb or '',
            
            # Training state
            'early_stopped': False,  # Will be updated later if needed
            'best_val_loss': '',  # Will be updated by trainer
            'epochs_trained': epoch + 1,
            
            # Model statistics
            'gradient_norm': gradient_norm or '',
            'gradient_norm_mean': gradient_stats.get('mean', '') if gradient_stats else '',
            'gradient_norm_max': gradient_stats.get('max', '') if gradient_stats else '',
            'gradient_norm_min': gradient_stats.get('min', '') if gradient_stats else '',
            'gradient_norm_std': gradient_stats.get('std', '') if gradient_stats else '',
            'weight_norm_mean': weight_stats.get('mean', '') if weight_stats else '',
            'weight_norm_std': weight_stats.get('std', '') if weight_stats else '',
            'lr_mean': lr_stats.get('mean', '') if lr_stats else '',
            'lr_min': lr_stats.get('min', '') if lr_stats else '',
            'lr_max': lr_stats.get('max', '') if lr_stats else '',
            'lr_std': lr_stats.get('std', '') if lr_stats else '',
            'loss_spikes_count': loss_spikes_count,
            
            # Additional info
            'notes': notes
        }
        
        # Write to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            # Write headers if file is empty (first write)
            if self.log_file.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(row)
        
        logger.debug(f"Logged epoch {epoch} metrics")
    
    def log_mini_epoch(self, 
                  epoch: str,
                  train_metrics: Dict[str, float],
                  val_metrics: Optional[Dict[str, float]],
                  hyperparams: Dict[str, Any],
                  training_time: float = 0.0,
                  epoch_time: float = 0.0,
                  samples_per_second: float = 0.0,
                  memory_usage_mb: float = 0.0,
                  gpu_memory_mb: Optional[float] = None,
                  gradient_norm: Optional[float] = None,
                  post_clip_gradient_norm: Optional[float] = None,
                  weight_stats: Optional[Dict[str, float]] = None,
                  notes: str = "",
                  gradient_stats: Optional[Dict[str, float]] = None,
                  lr_stats: Optional[Dict[str, float]] = None,
                  loss_spikes_count: int = 0,
                  best_val_loss: Optional[float] = None) -> None:
        """
        Log metrics for a single mini-epoch (or chunk).
        Args are the same as log_epoch, but epoch can be a string.
        """
        row = {
            # Experiment identification
            'experiment_name': self.experiment_name,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            # Hyperparameters
            'learning_rate': hyperparams.get('learning_rate', ''),
            'batch_size': hyperparams.get('batch_size', ''),
            'dataset_size': hyperparams.get('dataset_size', ''),
            'network_structure': hyperparams.get('network_structure', ''),
            'policy_weight': hyperparams.get('policy_weight', ''),
            'value_weight': hyperparams.get('value_weight', ''),
            'total_loss_weight': hyperparams.get('total_loss_weight', ''),
            'dropout_prob': hyperparams.get('dropout_prob', ''),
            'weight_decay': hyperparams.get('weight_decay', ''),
            'max_grad_norm': hyperparams.get('max_grad_norm', ''),
            'value_learning_rate_factor': hyperparams.get('value_learning_rate_factor', ''),
            'value_weight_decay_factor': hyperparams.get('value_weight_decay_factor', ''),
            # Training metrics
            'policy_loss': train_metrics.get('policy_loss', ''),
            'value_loss': train_metrics.get('value_loss', ''),
            'total_loss': train_metrics.get('total_loss', ''),
            'val_policy_loss': val_metrics.get('policy_loss', '') if val_metrics else '',
            'val_value_loss': val_metrics.get('value_loss', '') if val_metrics else '',
            'val_total_loss': val_metrics.get('total_loss', '') if val_metrics else '',
            # Performance metrics
            'training_time': training_time,
            'epoch_time': epoch_time,
            'samples_per_second': samples_per_second,
            'memory_usage_mb': memory_usage_mb,
            'gpu_memory_mb': gpu_memory_mb or '',
            # Training state
            'early_stopped': False,
            'best_val_loss': best_val_loss or '',
            'epochs_trained': '',
            # Model statistics
            'gradient_norm': gradient_norm or '',
            'post_clip_gradient_norm': post_clip_gradient_norm or '',
            'gradient_norm_mean': gradient_stats.get('mean', '') if gradient_stats else '',
            'gradient_norm_max': gradient_stats.get('max', '') if gradient_stats else '',
            'gradient_norm_min': gradient_stats.get('min', '') if gradient_stats else '',
            'gradient_norm_std': gradient_stats.get('std', '') if gradient_stats else '',
            'weight_norm_mean': weight_stats.get('mean', '') if weight_stats else '',
            'weight_norm_std': weight_stats.get('std', '') if weight_stats else '',
            'lr_mean': lr_stats.get('mean', '') if lr_stats else '',
            'lr_min': lr_stats.get('min', '') if lr_stats else '',
            'lr_max': lr_stats.get('max', '') if lr_stats else '',
            'lr_std': lr_stats.get('std', '') if lr_stats else '',
            'loss_spikes_count': loss_spikes_count,
            # Additional info
            'notes': notes
        }
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            if self.log_file.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(row)
        logger.debug(f"Logged mini-epoch {epoch} metrics")
    
    def log_experiment_summary(self, 
                              best_val_loss: float,
                              total_epochs: int,
                              total_training_time: float,
                              early_stopped: bool = False,
                              notes: str = "") -> None:
        """
        Log a summary row for the experiment.
        
        Args:
            best_val_loss: Best validation loss achieved
            total_epochs: Total epochs trained
            total_training_time: Total training time
            early_stopped: Whether training was stopped early
            notes: Additional notes
        """
        row = {
            # Experiment identification
            'experiment_name': self.experiment_name,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'timestamp': datetime.now().isoformat(),
            'epoch': 'SUMMARY',
            
            # Hyperparameters (empty for summary)
            'learning_rate': '',
            'batch_size': '',
            'dataset_size': '',
            'network_structure': '',
            'policy_weight': '',
            'value_weight': '',
            'total_loss_weight': '',
            'dropout_prob': '',
            'weight_decay': '',
            
            # Training metrics (empty for summary)
            'policy_loss': '',
            'value_loss': '',
            'total_loss': '',
            'val_policy_loss': '',
            'val_value_loss': '',
            'val_total_loss': '',
            
            # Performance metrics
            'training_time': total_training_time,
            'epoch_time': '',
            'samples_per_second': '',
            'memory_usage_mb': '',
            'gpu_memory_mb': '',
            
            # Training state
            'early_stopped': early_stopped,
            'best_val_loss': best_val_loss,
            'epochs_trained': total_epochs,
            
            # Model statistics (empty for summary)
            'gradient_norm': '',
            'weight_norm_mean': '',
            'weight_norm_std': '',
            
            # Additional info
            'notes': f"EXPERIMENT SUMMARY: {notes}"
        }
        
        # Write to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            # Write headers if file is empty (first write)
            if self.log_file.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(row)
        
        logger.info(f"Logged experiment summary: best_val_loss={best_val_loss:.6f}, epochs={total_epochs}")
    
    def get_experiment_data(self) -> List[Dict]:
        """Get all data for the current experiment."""
        data = []
        with open(self.log_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['experiment_name'] == self.experiment_name:
                    data.append(row)
        return data
    
    def get_latest_metrics(self) -> Optional[Dict]:
        """Get the latest metrics for the current experiment."""
        data = self.get_experiment_data()
        if data:
            return data[-1]
        return None


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def get_gpu_memory_usage() -> Optional[float]:
    """Get GPU memory usage in MB if available."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
    except ImportError:
        pass
    return None


def get_weight_statistics(model) -> Dict[str, float]:
    """Get statistics about model weights."""
    import torch
    import numpy as np
    
    all_weights = []
    for param in model.parameters():
        if param.requires_grad:
            all_weights.extend(param.data.cpu().numpy().flatten())
    
    if all_weights:
        weights_array = np.array(all_weights)
        return {
            'mean': float(np.mean(weights_array)),
            'std': float(np.std(weights_array))
        }
    else:
        return {'mean': 0.0, 'std': 0.0}


def get_gradient_norm(model) -> Optional[float]:
    """Get L2 norm of gradients."""
    import torch
    
    total_norm = 0.0
    param_count = 0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    if param_count > 0:
        return total_norm ** (1. / 2)
    return None 