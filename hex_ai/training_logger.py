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
            'gradient_norm', 'post_clip_gradient_norm', 'gradient_norm_mean', 
            'gradient_norm_max', 'gradient_norm_min', 'gradient_norm_std',
            'weight_norm_mean', 'weight_norm_std',
            'lr_mean', 'lr_min', 'lr_max', 'lr_std',
            'loss_spikes_count',
            
            # Enhanced performance tracking (based on recommendations)
            'policy_loss_ma', 'value_loss_ma', 'total_loss_ma',  # Moving averages
            'policy_loss_es', 'value_loss_es', 'total_loss_es',  # Exponential smoothing
            'composite_score', 'trend_slope', 'trend_direction',  # Composite metrics
            'performance_alert', 'alert_message',  # Performance alerts
            
            # Additional info
            'notes'
        ]
        
        # Initialize CSV file if it doesn't exist
        self._initialize_csv()
        
        # Initialize performance tracking
        self._init_performance_tracking()
        
        logger.info(f"Training logger initialized: {self.log_file}")
    
    def _init_performance_tracking(self):
        """Initialize performance tracking variables."""
        # Performance tracking parameters
        self.ma_window_size = 20  # Moving average window
        self.es_alpha = 0.3  # Exponential smoothing alpha
        self.alert_threshold = 0.05  # 5% change threshold for alerts
        
        # Storage for performance tracking
        self.policy_losses = []
        self.value_losses = []
        self.total_losses = []
        self.val_policy_losses = []
        self.val_value_losses = []
        self.val_total_losses = []
        
        # Smoothed metrics
        self.policy_loss_ma = []
        self.value_loss_ma = []
        self.total_loss_ma = []
        self.policy_loss_es = []
        self.value_loss_es = []
        self.total_loss_es = []
        
        # Trend analysis
        self.trend_slope = None
        self.trend_direction = 'stable'
        
        # Composite score
        self.composite_scores = []
        
        # Performance alerts
        self.performance_alerts = []
        self.alert_messages = []
    
    def _initialize_csv(self):
        """Initialize the CSV file with headers if it doesn't exist."""
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
            logger.info(f"Created new training metrics file: {self.log_file}")
        else:
            # Check if existing file has the correct headers
            self._check_and_update_headers()
    
    def _check_and_update_headers(self):
        """Check if existing CSV file has the correct headers and update if needed."""
        try:
            with open(self.log_file, 'r', newline='') as f:
                reader = csv.reader(f)
                existing_headers = next(reader, [])
            
            if existing_headers != self.headers:
                logger.warning(f"CSV file {self.log_file} has outdated headers. Updating...")
                
                # Read all existing data
                with open(self.log_file, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    existing_data = list(reader)
                
                # Write back with new headers
                with open(self.log_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.headers)
                    writer.writeheader()
                    
                    # Write existing data, filling missing columns with empty strings
                    for row in existing_data:
                        # Ensure all required fields are present
                        for header in self.headers:
                            if header not in row:
                                row[header] = ''
                        writer.writerow(row)
                
                logger.info(f"Updated CSV headers in {self.log_file}")
            else:
                logger.debug(f"CSV file {self.log_file} has correct headers")
                
        except Exception as e:
            logger.error(f"Error checking/updating CSV headers: {e}")
            # If we can't read the file, create a backup and start fresh
            backup_path = self.log_file.with_suffix('.csv.backup')
            logger.warning(f"Creating backup of problematic file: {backup_path}")
            self.log_file.rename(backup_path)
            
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
                  post_clip_gradient_norm: Optional[float] = None,
                  weight_stats: Optional[Dict[str, float]] = None,
                  notes: str = "",
                  gradient_stats: Optional[Dict[str, float]] = None,
                  lr_stats: Optional[Dict[str, float]] = None,
                  loss_spikes_count: int = 0,
                  best_val_loss: Optional[float] = None) -> None:
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
            'best_val_loss': best_val_loss or '',  # Will be updated by trainer
            'epochs_trained': epoch + 1,
            
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
        
        # Update performance tracking
        self._update_performance_tracking(train_metrics, val_metrics)
        
        # Add enhanced performance metrics to row
        row.update({
            'policy_loss_ma': self.policy_loss_ma[-1] if self.policy_loss_ma else '',
            'value_loss_ma': self.value_loss_ma[-1] if self.value_loss_ma else '',
            'total_loss_ma': self.total_loss_ma[-1] if self.total_loss_ma else '',
            'policy_loss_es': self.policy_loss_es[-1] if self.policy_loss_es else '',
            'value_loss_es': self.value_loss_es[-1] if self.value_loss_es else '',
            'total_loss_es': self.total_loss_es[-1] if self.total_loss_es else '',
            'composite_score': self.composite_scores[-1] if self.composite_scores else '',
            'trend_slope': self.trend_slope if self.trend_slope is not None else '',
            'trend_direction': self.trend_direction,
            'performance_alert': self.performance_alerts[-1] if self.performance_alerts else False,
            'alert_message': self.alert_messages[-1] if self.alert_messages else ''
        })
        
        # Write to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            # Write headers if file is empty (first write)
            if self.log_file.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(row)
        
        logger.debug(f"Logged epoch {epoch} metrics")
    
    def _update_performance_tracking(self, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]] = None):
        """Update performance tracking with new metrics."""
        # Store raw metrics
        self.policy_losses.append(train_metrics.get('policy_loss', 0.0))
        self.value_losses.append(train_metrics.get('value_loss', 0.0))
        self.total_losses.append(train_metrics.get('total_loss', 0.0))
        
        if val_metrics:
            self.val_policy_losses.append(val_metrics.get('policy_loss', 0.0))
            self.val_value_losses.append(val_metrics.get('value_loss', 0.0))
            self.val_total_losses.append(val_metrics.get('total_loss', 0.0))
        else:
            self.val_policy_losses.append(0.0)
            self.val_value_losses.append(0.0)
            self.val_total_losses.append(0.0)
        
        # Update moving averages
        self._update_moving_averages()
        
        # Update exponential smoothing
        self._update_exponential_smoothing()
        
        # Update trend analysis
        self._update_trend_analysis()
        
        # Update composite score
        self._update_composite_score()
        
        # Check for alerts
        self._check_performance_alerts()
    
    def _update_moving_averages(self):
        """Update moving averages for all loss types."""
        if len(self.policy_losses) >= self.ma_window_size:
            # Policy loss moving average
            recent_policy = self.policy_losses[-self.ma_window_size:]
            self.policy_loss_ma.append(sum(recent_policy) / len(recent_policy))
            
            # Value loss moving average
            recent_value = self.value_losses[-self.ma_window_size:]
            self.value_loss_ma.append(sum(recent_value) / len(recent_value))
            
            # Total loss moving average
            recent_total = self.total_losses[-self.ma_window_size:]
            self.total_loss_ma.append(sum(recent_total) / len(recent_total))
        else:
            # Not enough data yet, use current values
            self.policy_loss_ma.append(self.policy_losses[-1])
            self.value_loss_ma.append(self.value_losses[-1])
            self.total_loss_ma.append(self.total_losses[-1])
    
    def _update_exponential_smoothing(self):
        """Update exponential smoothing for all loss types."""
        if len(self.policy_loss_es) == 0:
            # First update, use current values
            self.policy_loss_es.append(self.policy_losses[-1])
            self.value_loss_es.append(self.value_losses[-1])
            self.total_loss_es.append(self.total_losses[-1])
        else:
            # Apply exponential smoothing
            self.policy_loss_es.append(
                self.es_alpha * self.policy_losses[-1] + 
                (1 - self.es_alpha) * self.policy_loss_es[-1]
            )
            self.value_loss_es.append(
                self.es_alpha * self.value_losses[-1] + 
                (1 - self.es_alpha) * self.value_loss_es[-1]
            )
            self.total_loss_es.append(
                self.es_alpha * self.total_losses[-1] + 
                (1 - self.es_alpha) * self.total_loss_es[-1]
            )
    
    def _update_trend_analysis(self):
        """Update linear trend analysis."""
        if len(self.policy_losses) < 10:  # Need at least 10 points for trend
            return
        
        # Use moving average for trend analysis if available
        if len(self.policy_loss_ma) >= 10:
            trend_data = self.policy_loss_ma[-10:]
        else:
            trend_data = self.policy_losses[-10:]
        
        # Simple linear trend calculation
        n = len(trend_data)
        x_sum = sum(range(n))
        y_sum = sum(trend_data)
        xy_sum = sum(i * val for i, val in enumerate(trend_data))
        x2_sum = sum(i * i for i in range(n))
        
        # Calculate slope
        denominator = n * x2_sum - x_sum * x_sum
        if denominator != 0:
            self.trend_slope = (n * xy_sum - x_sum * y_sum) / denominator
            
            # Determine trend direction
            if self.trend_slope < -0.001:
                self.trend_direction = 'improving'
            elif self.trend_slope > 0.001:
                self.trend_direction = 'degrading'
            else:
                self.trend_direction = 'stable'
    
    def _update_composite_score(self):
        """Calculate composite performance score."""
        if len(self.policy_losses) < 2:
            self.composite_scores.append(0.5)  # Neutral score
            return
        
        # Normalize losses to 0-1 scale
        policy_norm = self._normalize_to_0_1(self.policy_losses)
        value_norm = self._normalize_to_0_1(self.value_losses)
        
        # Use moving averages for stability if available
        if len(self.policy_loss_ma) > 0:
            policy_ma_norm = self._normalize_to_0_1(self.policy_loss_ma)
            value_ma_norm = self._normalize_to_0_1(self.value_loss_ma)
        else:
            policy_ma_norm = policy_norm
            value_ma_norm = value_norm
        
        # Composite score (higher is better)
        # Weight policy loss more heavily (60%) as it's typically more important
        score = 1 - (0.6 * policy_ma_norm + 0.4 * value_ma_norm)
        self.composite_scores.append(score)
    
    def _normalize_to_0_1(self, values: list) -> float:
        """Normalize a value to 0-1 scale based on min/max of the series."""
        if len(values) < 2:
            return 0.5
        
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return 0.5
        
        return (values[-1] - min_val) / (max_val - min_val)
    
    def _check_performance_alerts(self):
        """Check for performance alerts."""
        if len(self.policy_loss_ma) < 2:
            self.performance_alerts.append(False)
            self.alert_messages.append('')
            return
        
        # Check for performance degradation (5% increase in moving average)
        current_ma = self.policy_loss_ma[-1]
        previous_ma = self.policy_loss_ma[-2]
        
        if current_ma > previous_ma * (1 + self.alert_threshold):
            self.performance_alerts.append(True)
            self.alert_messages.append(
                f"Performance degradation: MA increased by {((current_ma/previous_ma - 1) * 100):.1f}%"
            )
        elif current_ma < previous_ma * (1 - self.alert_threshold):
            self.performance_alerts.append(True)
            self.alert_messages.append(
                f"Performance improvement: MA decreased by {((1 - current_ma/previous_ma) * 100):.1f}%"
            )
        else:
            self.performance_alerts.append(False)
            self.alert_messages.append('')
    
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
            
            # Enhanced performance tracking
            'policy_loss_ma': '',
            'value_loss_ma': '',
            'total_loss_ma': '',
            'policy_loss_es': '',
            'value_loss_es': '',
            'total_loss_es': '',
            'composite_score': '',
            'trend_slope': '',
            'trend_direction': '',
            'performance_alert': '',
            'alert_message': '',
            
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