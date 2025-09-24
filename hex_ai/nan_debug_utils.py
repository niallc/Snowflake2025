"""
Unified NaN detection and debugging utilities for training and validation.

This module provides consistent NaN checking across training and validation,
with detailed debugging information and data dumping capabilities.
"""

import torch
import numpy as np
import pickle
import gzip
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

def check_for_nan_and_debug(
    policy_pred: torch.Tensor,
    value_pred: torch.Tensor, 
    total_loss: torch.Tensor,
    loss_dict: Dict[str, float],
    boards: torch.Tensor,
    policies: torch.Tensor,
    values: torch.Tensor,
    move_stage: torch.Tensor,
    context: str = "unknown",
    batch_idx: Optional[int] = None,
    epoch: Optional[int] = None,
    mini_epoch: Optional[int] = None
) -> None:
    """
    Comprehensive NaN check with detailed debugging information.
    
    Args:
        policy_pred: Model policy predictions
        value_pred: Model value predictions
        total_loss: Total loss tensor
        loss_dict: Dictionary of individual loss components
        boards: Input board tensors
        policies: Target policy tensors
        values: Target value tensors
        move_stage: Move stage tensors
        context: Context string (e.g., "training", "validation")
        batch_idx: Batch index for debugging
        epoch: Epoch number for debugging
        mini_epoch: Mini-epoch number for debugging
    """
    
    # Check for NaN in model outputs
    policy_pred_has_nan = torch.isnan(policy_pred).any()
    value_pred_has_nan = torch.isnan(value_pred).any()
    total_loss_has_nan = torch.isnan(total_loss)
    
    # Check for NaN in individual loss components
    loss_components_has_nan = any(np.isnan(v) for v in loss_dict.values())
    
    # If no NaN found, return early
    if not (policy_pred_has_nan or value_pred_has_nan or total_loss_has_nan or loss_components_has_nan):
        return
    
    # NaN detected - create detailed debug information
    debug_info = create_nan_debug_info(
        policy_pred, value_pred, total_loss, loss_dict,
        boards, policies, values, move_stage,
        context, batch_idx, epoch, mini_epoch
    )
    
    # Dump debug data to file
    debug_file = dump_nan_debug_data(debug_info)
    
    # Create comprehensive error message
    error_msg = create_nan_error_message(
        policy_pred_has_nan, value_pred_has_nan, total_loss_has_nan, 
        loss_components_has_nan, loss_dict, debug_file, context
    )
    
    # Raise error with detailed information
    raise RuntimeError(error_msg)

def create_nan_debug_info(
    policy_pred: torch.Tensor,
    value_pred: torch.Tensor,
    total_loss: torch.Tensor,
    loss_dict: Dict[str, float],
    boards: torch.Tensor,
    policies: torch.Tensor,
    values: torch.Tensor,
    move_stage: torch.Tensor,
    context: str,
    batch_idx: Optional[int],
    epoch: Optional[int],
    mini_epoch: Optional[int]
) -> Dict[str, Any]:
    """Create comprehensive debug information for NaN cases."""
    
    # Basic statistics
    debug_info = {
        'timestamp': time.time(),
        'context': context,
        'batch_idx': batch_idx,
        'epoch': epoch,
        'mini_epoch': mini_epoch,
        
        # Model outputs
        'policy_pred_stats': {
            'min': float(policy_pred.min()),
            'max': float(policy_pred.max()),
            'mean': float(policy_pred.mean()),
            'std': float(policy_pred.std()),
            'has_nan': bool(torch.isnan(policy_pred).any()),
            'nan_count': int(torch.isnan(policy_pred).sum()),
            'shape': list(policy_pred.shape)
        },
        
        'value_pred_stats': {
            'min': float(value_pred.min()),
            'max': float(value_pred.max()),
            'mean': float(value_pred.mean()),
            'std': float(value_pred.std()),
            'has_nan': bool(torch.isnan(value_pred).any()),
            'nan_count': int(torch.isnan(value_pred).sum()),
            'shape': list(value_pred.shape)
        },
        
        # Loss information
        'total_loss': {
            'value': float(total_loss),
            'has_nan': bool(torch.isnan(total_loss))
        },
        
        'loss_dict': loss_dict,
        'loss_components_has_nan': any(np.isnan(v) for v in loss_dict.values()),
        
        # Input data statistics
        'input_stats': {
            'boards': {
                'min': float(boards.min()),
                'max': float(boards.max()),
                'has_nan': bool(torch.isnan(boards).any()),
                'shape': list(boards.shape)
            },
            'policies': {
                'min': float(policies.min()),
                'max': float(policies.max()),
                'has_nan': bool(torch.isnan(policies).any()),
                'sum_min': float(policies.sum(dim=1).min()),
                'sum_max': float(policies.sum(dim=1).max()),
                'shape': list(policies.shape)
            },
            'values': {
                'min': float(values.min()),
                'max': float(values.max()),
                'has_nan': bool(torch.isnan(values).any()),
                'shape': list(values.shape)
            },
            'move_stage': {
                'min': float(move_stage.min()),
                'max': float(move_stage.max()),
                'has_nan': bool(torch.isnan(move_stage).any()),
                'shape': list(move_stage.shape)
            }
        }
    }
    
    # Add sample data for debugging (first few samples)
    debug_info['sample_data'] = {
        'boards_sample': boards[:3].cpu().numpy() if boards.numel() > 0 else None,
        'policies_sample': policies[:3].cpu().numpy() if policies.numel() > 0 else None,
        'values_sample': values[:3].cpu().numpy() if values.numel() > 0 else None,
        'move_stage_sample': move_stage[:3].cpu().numpy() if move_stage.numel() > 0 else None,
        'policy_pred_sample': policy_pred[:3].cpu().numpy() if policy_pred.numel() > 0 else None,
        'value_pred_sample': value_pred[:3].cpu().numpy() if value_pred.numel() > 0 else None
    }
    
    return debug_info

def dump_nan_debug_data(debug_info: Dict[str, Any]) -> str:
    """Dump debug information to a compressed file."""
    
    # Create temp directory if it doesn't exist
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    # Create filename with timestamp
    timestamp = int(time.time())
    debug_file = temp_dir / f"nan_debug_{debug_info['context']}_{timestamp}.pkl.gz"
    
    # Dump data
    with gzip.open(debug_file, 'wb') as f:
        pickle.dump(debug_info, f)
    
    return str(debug_file)

def create_nan_error_message(
    policy_pred_has_nan: bool,
    value_pred_has_nan: bool, 
    total_loss_has_nan: bool,
    loss_components_has_nan: bool,
    loss_dict: Dict[str, float],
    debug_file: str,
    context: str
) -> str:
    """Create a comprehensive error message for NaN cases."""
    
    error_parts = [
        f"NaN detected in {context}!",
        f"policy_pred has NaN: {policy_pred_has_nan}",
        f"value_pred has NaN: {value_pred_has_nan}",
        f"total_loss is NaN: {total_loss_has_nan}",
        f"loss components have NaN: {loss_components_has_nan}"
    ]
    
    if loss_components_has_nan:
        nan_components = [k for k, v in loss_dict.items() if np.isnan(v)]
        error_parts.append(f"NaN components: {nan_components}")
    
    error_parts.extend([
        f"Debug data saved to: {debug_file}",
        "This indicates numerical instability. Check learning rate, gradient clipping, and model architecture."
    ])
    
    return " | ".join(error_parts)

def calculate_validation_metrics_statistics(val_metrics: Dict[str, List[float]]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate comprehensive statistics for validation metrics.
    
    Args:
        val_metrics: Dictionary mapping metric names to lists of values from each batch
        
    Returns:
        Dictionary with statistics for each metric
    """
    return {
        key: {
            'count': len(values),
            'has_nan': any(np.isnan(v) for v in values),
            'nan_count': sum(1 for v in values if np.isnan(v)),
            'min': float(np.nanmin(values)) if values else float('nan'),
            'max': float(np.nanmax(values)) if values else float('nan'),
            'mean': float(np.nanmean(values)) if values else float('nan'),
            'std': float(np.nanstd(values)) if values else float('nan')
        }
        for key, values in val_metrics.items()
    }

def create_batch_analysis(val_metrics: Dict[str, List[float]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Create batch-by-batch analysis for validation metrics.
    
    Args:
        val_metrics: Dictionary mapping metric names to lists of values from each batch
        
    Returns:
        Dictionary with batch-by-batch analysis for each metric
    """
    return {
        key: [
            {
                'batch_idx': i,
                'value': float(v),
                'is_nan': bool(np.isnan(v))
            }
            for i, v in enumerate(values)
        ]
        for key, values in val_metrics.items()
    }

def find_nan_batch_indices(val_metrics: Dict[str, List[float]]) -> List[int]:
    """
    Find batch indices that contain NaN in any metric.
    
    Args:
        val_metrics: Dictionary mapping metric names to lists of values from each batch
        
    Returns:
        List of batch indices that have NaN in any metric
    """
    return list(set([
        i for key, values in val_metrics.items()
        for i, v in enumerate(values) if np.isnan(v)
    ]))

def check_model_outputs_for_nan(
    policy_pred: torch.Tensor,
    value_pred: torch.Tensor,
    context: str = "unknown"
) -> None:
    """
    Check model outputs for NaN and raise error if found.
    
    This is a lighter-weight check for cases where we only have model outputs.
    """
    
    policy_pred_has_nan = torch.isnan(policy_pred).any()
    value_pred_has_nan = torch.isnan(value_pred).any()
    
    if policy_pred_has_nan or value_pred_has_nan:
        error_msg = (
            f"NaN detected in {context} model outputs! "
            f"policy_pred has NaN: {policy_pred_has_nan}, "
            f"value_pred has NaN: {value_pred_has_nan}. "
            f"This indicates numerical instability in the model forward pass."
        )
        raise RuntimeError(error_msg)


# ============================================================================
# TEMPORARY ENHANCED NaN DETECTION AND LOGGING SYSTEM
# ============================================================================
# This section contains temporary code for debugging NaN validation issues.
# All functions and classes in this section should be removed after the issue is resolved.

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
import csv
from datetime import datetime


class FirstNaNDetector:
    """
    TEMPORARY: Detects and logs the very first NaN occurrence with maximum detail.
    
    This class is designed to capture comprehensive information about the first
    NaN occurrence during training/validation to help debug the validation NaN issue.
    """
    
    def __init__(self, log_dir: str = "temp", enabled: bool = True):
        self.enabled = enabled
        self.nan_detected = False
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup first-NaN logger
        self.first_nan_logger = self._setup_logger(
            "first_nan", 
            self.log_dir / "first_nan_detection.log"
        )
        
        # Setup batch monitoring logger
        self.batch_monitor_logger = self._setup_logger(
            "batch_monitor",
            self.log_dir / "batch_monitoring_summary.log"
        )
        
        # Setup trend tracking logger
        self.trend_logger = self._setup_logger(
            "trend_tracker",
            self.log_dir / "trend_analysis.log"
        )
        
        self.batch_count = 0
        self.summary_interval = 100  # Log summary every 100 batches
        self.batch_stats = []
        self.validation_batch_count = 0
        self.validation_batch_stats = []
        
        # Track trends leading up to NaN
        self.recent_policy_losses = []
        self.recent_value_losses = []
        self.recent_total_losses = []
        self.recent_policy_pred_stats = []
        self.recent_value_pred_stats = []
        
        self.first_nan_logger.info("=" * 80)
        self.first_nan_logger.info("TEMPORARY NaN DETECTION SYSTEM INITIALIZED")
        self.first_nan_logger.info("=" * 80)
        self.first_nan_logger.info(f"Log directory: {self.log_dir}")
        self.first_nan_logger.info(f"Summary interval: {self.summary_interval} batches")
        self.first_nan_logger.info("=" * 80)
    
    def _setup_logger(self, name: str, log_file: Path) -> logging.Logger:
        """Setup a logger for specific purpose."""
        logger = logging.getLogger(f"temp_nan_{name}")
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.propagate = False  # Don't propagate to root logger
        
        return logger
    
    def check_and_log_first_nan(self, 
                                policy_pred: torch.Tensor,
                                value_pred: torch.Tensor,
                                total_loss: torch.Tensor,
                                loss_dict: Dict[str, float],
                                boards: torch.Tensor,
                                policies: torch.Tensor,
                                values: torch.Tensor,
                                move_stage: torch.Tensor,
                                context: str,
                                batch_idx: Optional[int] = None,
                                epoch: Optional[int] = None,
                                mini_epoch: Optional[int] = None) -> bool:
        """
        Check for NaN and log comprehensive details if this is the first occurrence.
        Returns True if NaN detected, False otherwise.
        """
        if not self.enabled or self.nan_detected:
            return False
            
        # Check for NaN using existing logic
        policy_pred_has_nan = torch.isnan(policy_pred).any()
        value_pred_has_nan = torch.isnan(value_pred).any()
        total_loss_has_nan = torch.isnan(total_loss)
        loss_components_has_nan = any(np.isnan(v) for v in loss_dict.values())
        
        has_nan = policy_pred_has_nan or value_pred_has_nan or total_loss_has_nan or loss_components_has_nan
        
        if has_nan and not self.nan_detected:
            self.nan_detected = True
            self._log_first_nan_occurrence(
                policy_pred, value_pred, total_loss, loss_dict,
                boards, policies, values, move_stage,
                context, batch_idx, epoch, mini_epoch,
                policy_pred_has_nan, value_pred_has_nan, 
                total_loss_has_nan, loss_components_has_nan
            )
            return True
        
        return False
    
    def _log_first_nan_occurrence(self, 
                                 policy_pred: torch.Tensor,
                                 value_pred: torch.Tensor,
                                 total_loss: torch.Tensor,
                                 loss_dict: Dict[str, float],
                                 boards: torch.Tensor,
                                 policies: torch.Tensor,
                                 values: torch.Tensor,
                                 move_stage: torch.Tensor,
                                 context: str,
                                 batch_idx: Optional[int],
                                 epoch: Optional[int],
                                 mini_epoch: Optional[int],
                                 policy_pred_has_nan: bool,
                                 value_pred_has_nan: bool,
                                 total_loss_has_nan: bool,
                                 loss_components_has_nan: bool):
        """Log comprehensive details of the first NaN occurrence."""
        
        self.first_nan_logger.info("🚨 FIRST NaN DETECTED! 🚨")
        self.first_nan_logger.info("=" * 80)
        self.first_nan_logger.info(f"Context: {context}")
        self.first_nan_logger.info(f"Epoch: {epoch}")
        self.first_nan_logger.info(f"Mini-epoch: {mini_epoch}")
        self.first_nan_logger.info(f"Batch index: {batch_idx}")
        self.first_nan_logger.info(f"Timestamp: {datetime.now().isoformat()}")
        self.first_nan_logger.info("=" * 80)
        
        # Log NaN detection details
        self.first_nan_logger.info("NaN DETECTION RESULTS:")
        self.first_nan_logger.info(f"  policy_pred has NaN: {policy_pred_has_nan}")
        self.first_nan_logger.info(f"  value_pred has NaN: {value_pred_has_nan}")
        self.first_nan_logger.info(f"  total_loss is NaN: {total_loss_has_nan}")
        self.first_nan_logger.info(f"  loss components have NaN: {loss_components_has_nan}")
        
        if loss_components_has_nan:
            nan_components = [k for k, v in loss_dict.items() if np.isnan(v)]
            self.first_nan_logger.info(f"  NaN components: {nan_components}")
        
        # Log comprehensive statistics
        self._log_tensor_statistics("POLICY PREDICTIONS", policy_pred, self.first_nan_logger)
        self._log_tensor_statistics("VALUE PREDICTIONS", value_pred, self.first_nan_logger)
        self._log_tensor_statistics("INPUT BOARDS", boards, self.first_nan_logger)
        self._log_tensor_statistics("TARGET POLICIES", policies, self.first_nan_logger)
        self._log_tensor_statistics("TARGET VALUES", values, self.first_nan_logger)
        self._log_tensor_statistics("MOVE STAGE", move_stage, self.first_nan_logger)
        
        # Log loss details
        self.first_nan_logger.info("LOSS DETAILS:")
        for key, value in loss_dict.items():
            self.first_nan_logger.info(f"  {key}: {value} (NaN: {np.isnan(value)})")
        
        # Log recent trends
        self._log_recent_trends()
        
        # Dump comprehensive data to files
        self._dump_first_nan_data(
            policy_pred, value_pred, total_loss, loss_dict,
            boards, policies, values, move_stage,
            context, batch_idx, epoch, mini_epoch
        )
        
        self.first_nan_logger.info("=" * 80)
        self.first_nan_logger.info("FIRST NaN LOGGING COMPLETE")
        self.first_nan_logger.info("=" * 80)
    
    def _log_tensor_statistics(self, name: str, tensor: torch.Tensor, logger: logging.Logger):
        """Log comprehensive tensor statistics."""
        logger.info(f"{name} STATISTICS:")
        logger.info(f"  Shape: {list(tensor.shape)}")
        logger.info(f"  Dtype: {tensor.dtype}")
        logger.info(f"  Device: {tensor.device}")
        logger.info(f"  Has NaN: {torch.isnan(tensor).any()}")
        logger.info(f"  NaN count: {torch.isnan(tensor).sum()}")
        logger.info(f"  Has Inf: {torch.isinf(tensor).any()}")
        logger.info(f"  Inf count: {torch.isinf(tensor).sum()}")
        
        if not torch.isnan(tensor).all():
            logger.info(f"  Min: {tensor.min().item()}")
            logger.info(f"  Max: {tensor.max().item()}")
            
            # Handle different tensor types for mean and std
            if tensor.dtype in [torch.float16, torch.float32, torch.float64]:
                logger.info(f"  Mean: {tensor.mean().item()}")
                logger.info(f"  Std: {tensor.std().item()}")
            else:
                # For integer tensors, convert to float for statistics
                tensor_float = tensor.float()
                logger.info(f"  Mean: {tensor_float.mean().item()}")
                logger.info(f"  Std: {tensor_float.std().item()}")
            
            # Log extreme values
            abs_tensor = torch.abs(tensor.float())
            extreme_threshold = 1e6
            extreme_count = (abs_tensor > extreme_threshold).sum()
            if extreme_count > 0:
                logger.info(f"  Values > {extreme_threshold}: {extreme_count}")
                logger.info(f"  Max absolute value: {abs_tensor.max().item()}")
    
    def _log_recent_trends(self):
        """Log recent trends leading up to NaN."""
        self.first_nan_logger.info("RECENT TRENDS LEADING TO NaN:")
        
        if self.recent_policy_losses:
            self.first_nan_logger.info(f"  Recent policy losses: {self.recent_policy_losses[-10:]}")
        if self.recent_value_losses:
            self.first_nan_logger.info(f"  Recent value losses: {self.recent_value_losses[-10:]}")
        if self.recent_total_losses:
            self.first_nan_logger.info(f"  Recent total losses: {self.recent_total_losses[-10:]}")
        
        # Log trend analysis
        if len(self.recent_total_losses) >= 5:
            recent_avg = np.mean(self.recent_total_losses[-5:])
            older_avg = np.mean(self.recent_total_losses[-10:-5]) if len(self.recent_total_losses) >= 10 else recent_avg
            trend = "increasing" if recent_avg > older_avg else "decreasing"
            self.first_nan_logger.info(f"  Loss trend: {trend} (recent: {recent_avg:.6f}, older: {older_avg:.6f})")
    
    def _dump_first_nan_data(self, 
                            policy_pred: torch.Tensor,
                            value_pred: torch.Tensor,
                            total_loss: torch.Tensor,
                            loss_dict: Dict[str, float],
                            boards: torch.Tensor,
                            policies: torch.Tensor,
                            values: torch.Tensor,
                            move_stage: torch.Tensor,
                            context: str,
                            batch_idx: Optional[int],
                            epoch: Optional[int],
                            mini_epoch: Optional[int]):
        """Dump comprehensive data for first NaN occurrence."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive debug info
        debug_info = {
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'epoch': epoch,
            'mini_epoch': mini_epoch,
            'batch_idx': batch_idx,
            
            # Model outputs (first few samples only to keep size manageable)
            'policy_pred_sample': policy_pred[:3].cpu().numpy() if policy_pred.numel() > 0 else None,
            'value_pred_sample': value_pred[:3].cpu().numpy() if value_pred.numel() > 0 else None,
            
            # Input data (first few samples only)
            'boards_sample': boards[:3].cpu().numpy() if boards.numel() > 0 else None,
            'policies_sample': policies[:3].cpu().numpy() if policies.numel() > 0 else None,
            'values_sample': values[:3].cpu().numpy() if values.numel() > 0 else None,
            'move_stage_sample': move_stage[:3].cpu().numpy() if move_stage.numel() > 0 else None,
            
            # Loss information
            'total_loss': float(total_loss),
            'loss_dict': loss_dict,
            
            # Recent trends
            'recent_policy_losses': self.recent_policy_losses[-20:],
            'recent_value_losses': self.recent_value_losses[-20:],
            'recent_total_losses': self.recent_total_losses[-20:],
            
            # Batch statistics
            'validation_batch_stats': self.validation_batch_stats[-50:] if context == "validation" else [],
            'batch_stats': self.batch_stats[-50:]
        }
        
        # Save to compressed pickle file
        debug_file = self.log_dir / f"first_nan_comprehensive_{timestamp}.pkl.gz"
        with gzip.open(debug_file, 'wb') as f:
            pickle.dump(debug_info, f)
        
        self.first_nan_logger.info(f"Comprehensive debug data saved to: {debug_file}")
        
        # Also save a JSON summary for easy reading
        json_file = self.log_dir / f"first_nan_summary_{timestamp}.json"
        json_summary = {
            'timestamp': debug_info['timestamp'],
            'context': context,
            'epoch': epoch,
            'mini_epoch': mini_epoch,
            'batch_idx': batch_idx,
            'total_loss': debug_info['total_loss'],
            'loss_dict': debug_info['loss_dict'],
            'recent_trends': {
                'policy_losses': debug_info['recent_policy_losses'],
                'value_losses': debug_info['recent_value_losses'],
                'total_losses': debug_info['recent_total_losses']
            }
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_summary, f, indent=2)
        
        self.first_nan_logger.info(f"JSON summary saved to: {json_file}")
    
    def log_batch_summary(self, 
                         batch_idx: int,
                         epoch: Optional[int],
                         mini_epoch: Optional[int],
                         loss_dict: Dict[str, float],
                         policy_pred: torch.Tensor,
                         value_pred: torch.Tensor,
                         boards: torch.Tensor,
                         policies: torch.Tensor,
                         values: torch.Tensor,
                         context: str = "unknown"):
        """Log batch-level summary statistics."""
        
        if not self.enabled:
            return
        
        # Update counters
        if context == "validation":
            self.validation_batch_count += 1
        else:
            self.batch_count += 1
        
        # Collect statistics
        batch_stat = {
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'batch_idx': batch_idx,
            'epoch': epoch,
            'mini_epoch': mini_epoch,
            'loss_dict': loss_dict.copy(),
            'policy_pred_stats': self._get_tensor_stats(policy_pred),
            'value_pred_stats': self._get_tensor_stats(value_pred),
            'input_stats': {
                'boards': self._get_tensor_stats(boards),
                'policies': self._get_tensor_stats(policies),
                'values': self._get_tensor_stats(values)
            }
        }
        
        # Store in appropriate list
        if context == "validation":
            self.validation_batch_stats.append(batch_stat)
        else:
            self.batch_stats.append(batch_stat)
        
        # Update recent trends
        self.recent_policy_losses.append(loss_dict.get('policy_loss', 0.0))
        self.recent_value_losses.append(loss_dict.get('value_loss', 0.0))
        self.recent_total_losses.append(loss_dict.get('total_loss', 0.0))
        
        # Keep only recent history (last 100 batches)
        if len(self.recent_policy_losses) > 100:
            self.recent_policy_losses = self.recent_policy_losses[-100:]
        if len(self.recent_value_losses) > 100:
            self.recent_value_losses = self.recent_value_losses[-100:]
        if len(self.recent_total_losses) > 100:
            self.recent_total_losses = self.recent_total_losses[-100:]
        
        # Log summary every N batches
        if (context == "validation" and self.validation_batch_count % self.summary_interval == 0) or \
           (context != "validation" and self.batch_count % self.summary_interval == 0):
            self._log_periodic_summary(batch_stat, context)
    
    def _get_tensor_stats(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Get basic statistics for a tensor."""
        if tensor.numel() == 0:
            return {'empty': True}
        
        stats = {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'has_nan': bool(torch.isnan(tensor).any()),
            'nan_count': int(torch.isnan(tensor).sum()),
            'has_inf': bool(torch.isinf(tensor).any()),
            'inf_count': int(torch.isinf(tensor).sum())
        }
        
        if not torch.isnan(tensor).all():
            stats.update({
                'min': float(tensor.min()),
                'max': float(tensor.max())
            })
            
            # Handle different tensor types for mean and std
            if tensor.dtype in [torch.float16, torch.float32, torch.float64]:
                stats.update({
                    'mean': float(tensor.mean()),
                    'std': float(tensor.std())
                })
            else:
                # For integer tensors, convert to float for statistics
                tensor_float = tensor.float()
                stats.update({
                    'mean': float(tensor_float.mean()),
                    'std': float(tensor_float.std())
                })
        
        return stats
    
    def _log_periodic_summary(self, batch_stat: Dict[str, Any], context: str):
        """Log periodic summary of batch statistics."""
        logger = self.batch_monitor_logger
        
        logger.info(f"=== {context.upper()} BATCH SUMMARY ===")
        logger.info(f"Batch: {batch_stat['batch_idx']}, Epoch: {batch_stat['epoch']}, Mini-epoch: {batch_stat['mini_epoch']}")
        logger.info(f"Losses: {batch_stat['loss_dict']}")
        
        # Log prediction statistics
        policy_stats = batch_stat['policy_pred_stats']
        value_stats = batch_stat['value_pred_stats']
        
        logger.info(f"Policy pred - Min: {policy_stats.get('min', 'N/A'):.6f}, Max: {policy_stats.get('max', 'N/A'):.6f}, Mean: {policy_stats.get('mean', 'N/A'):.6f}")
        logger.info(f"Value pred - Min: {value_stats.get('min', 'N/A'):.6f}, Max: {value_stats.get('max', 'N/A'):.6f}, Mean: {value_stats.get('mean', 'N/A'):.6f}")
        
        # Log recent trends
        if len(self.recent_total_losses) >= 10:
            recent_avg = np.mean(self.recent_total_losses[-10:])
            logger.info(f"Recent {context} loss average: {recent_avg:.6f}")
        
        logger.info("=" * 50)
    
    def log_averaging_nan(self, 
                         val_avg: Dict[str, float],
                         val_metrics: Dict[str, List[float]],
                         epoch: Optional[int],
                         mini_epoch: Optional[int]):
        """Log when NaN appears in validation averaging step."""
        
        self.first_nan_logger.info("🚨 NaN DETECTED IN VALIDATION AVERAGING! 🚨")
        self.first_nan_logger.info("=" * 80)
        self.first_nan_logger.info(f"Epoch: {epoch}, Mini-epoch: {mini_epoch}")
        self.first_nan_logger.info(f"Timestamp: {datetime.now().isoformat()}")
        self.first_nan_logger.info("=" * 80)
        
        # Log averaged values
        self.first_nan_logger.info("VALIDATION AVERAGES:")
        for key, value in val_avg.items():
            self.first_nan_logger.info(f"  {key}: {value} (NaN: {np.isnan(value)})")
        
        # Log batch-level statistics
        self.first_nan_logger.info("BATCH-LEVEL STATISTICS:")
        for key, values in val_metrics.items():
            nan_count = sum(1 for v in values if np.isnan(v))
            self.first_nan_logger.info(f"  {key}: {len(values)} batches, {nan_count} NaN batches")
            if nan_count > 0:
                nan_indices = [i for i, v in enumerate(values) if np.isnan(v)]
                self.first_nan_logger.info(f"    NaN batch indices: {nan_indices[:10]}...")  # Show first 10
        
        # Dump averaging debug data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        averaging_debug = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'mini_epoch': mini_epoch,
            'val_avg': val_avg,
            'val_metrics': val_metrics,
            'validation_batch_stats': self.validation_batch_stats[-100:]  # Last 100 validation batches
        }
        
        debug_file = self.log_dir / f"averaging_nan_{timestamp}.pkl.gz"
        with gzip.open(debug_file, 'wb') as f:
            pickle.dump(averaging_debug, f)
        
        self.first_nan_logger.info(f"Averaging debug data saved to: {debug_file}")
        self.first_nan_logger.info("=" * 80)
    
    def _log_first_nan_components(self, 
                                  components: Dict[str, torch.Tensor],
                                  nan_components: List[str],
                                  policy_pred: torch.Tensor,
                                  value_pred: torch.Tensor,
                                  policy_target: torch.Tensor,
                                  value_target: torch.Tensor,
                                  board: torch.Tensor):
        """
        TEMPORARY: Log detailed analysis of the first NaN occurrence in loss components.
        
        This method captures the exact moment and calculation where the first NaN appears
        in any loss component, providing comprehensive debugging information.
        """
        self.nan_detected = True
        
        self.first_nan_logger.info("🚨 FIRST NaN DETECTED IN LOSS COMPONENTS 🚨")
        self.first_nan_logger.info("=" * 80)
        self.first_nan_logger.info(f"Timestamp: {datetime.now().isoformat()}")
        self.first_nan_logger.info(f"NaN components: {nan_components}")
        self.first_nan_logger.info("=" * 80)
        
        # Log each component individually
        self.first_nan_logger.info("LOSS COMPONENT ANALYSIS:")
        for name, component in components.items():
            is_nan = torch.isnan(component)
            is_inf = torch.isinf(component)
            self.first_nan_logger.info(f"  {name}:")
            self.first_nan_logger.info(f"    Value: {component.item()}")
            self.first_nan_logger.info(f"    Is NaN: {is_nan}")
            self.first_nan_logger.info(f"    Is Inf: {is_inf}")
            if is_nan:
                self.first_nan_logger.info(f"    *** THIS COMPONENT IS NaN ***")
            self.first_nan_logger.info("")
        
        # Log input tensor statistics
        self.first_nan_logger.info("INPUT TENSOR STATISTICS:")
        self._log_tensor_statistics("policy_pred", policy_pred, self.first_nan_logger)
        self._log_tensor_statistics("value_pred", value_pred, self.first_nan_logger)
        self._log_tensor_statistics("policy_target", policy_target, self.first_nan_logger)
        self._log_tensor_statistics("value_target", value_target, self.first_nan_logger)
        self._log_tensor_statistics("board", board, self.first_nan_logger)
        
        # Log recent trends
        self._log_recent_trends()
        
        # Save comprehensive debug data
        self._dump_first_nan_components_data(
            components, nan_components, policy_pred, value_pred,
            policy_target, value_target, board
        )
        
        self.first_nan_logger.info("=" * 80)
        self.first_nan_logger.info("FIRST NaN COMPONENT ANALYSIS COMPLETE")
        self.first_nan_logger.info("=" * 80)
    
    def _dump_first_nan_components_data(self,
                                        components: Dict[str, torch.Tensor],
                                        nan_components: List[str],
                                        policy_pred: torch.Tensor,
                                        value_pred: torch.Tensor,
                                        policy_target: torch.Tensor,
                                        value_target: torch.Tensor,
                                        board: torch.Tensor):
        """TEMPORARY: Save comprehensive debug data for first NaN in components."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive debug data
        debug_data = {
            'timestamp': datetime.now().isoformat(),
            'nan_components': nan_components,
            'components': {name: component.item() for name, component in components.items()},
            'component_tensors': {name: component.detach().cpu() for name, component in components.items()},
            'policy_pred': policy_pred.detach().cpu(),
            'value_pred': value_pred.detach().cpu(),
            'policy_target': policy_target.detach().cpu(),
            'value_target': value_target.detach().cpu(),
            'board': board.detach().cpu(),
            'recent_trends': {
                'policy_losses': self.recent_policy_losses[-20:],
                'value_losses': self.recent_value_losses[-20:],
                'total_losses': self.recent_total_losses[-20:],
                'policy_pred_stats': self.recent_policy_pred_stats[-20:],
                'value_pred_stats': self.recent_value_pred_stats[-20:]
            },
            'batch_stats': self.batch_stats[-50:] if self.batch_stats else []
        }
        
        # Save as compressed pickle
        debug_file = self.log_dir / f"first_nan_components_{timestamp}.pkl.gz"
        with gzip.open(debug_file, 'wb') as f:
            pickle.dump(debug_data, f)
        
        # Save as JSON for human readability
        json_file = self.log_dir / f"first_nan_components_{timestamp}.json"
        json_data = {
            'timestamp': debug_data['timestamp'],
            'nan_components': debug_data['nan_components'],
            'components': debug_data['components'],
            'recent_trends': debug_data['recent_trends'],
            'batch_count': len(debug_data['batch_stats'])
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        self.first_nan_logger.info(f"Component debug data saved to: {debug_file}")
        self.first_nan_logger.info(f"Component summary saved to: {json_file}")
    
    def finalize_logging(self):
        """Finalize logging and create summary reports."""
        if not self.enabled:
            return
        
        # Create final summary
        summary_file = self.log_dir / "nan_detection_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("NaN DETECTION SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"NaN detected: {self.nan_detected}\n")
            f.write(f"Total batches processed: {self.batch_count}\n")
            f.write(f"Total validation batches processed: {self.validation_batch_count}\n")
            f.write(f"Log directory: {self.log_dir}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            
            if self.nan_detected:
                f.write("\n🚨 NaN WAS DETECTED DURING THIS RUN 🚨\n")
                f.write("Check first_nan_detection.log for details.\n")
            else:
                f.write("\n✅ No NaN detected during this run.\n")
        
        self.first_nan_logger.info(f"Final summary saved to: {summary_file}")


# Global instance for easy access
_global_first_nan_detector = None

def get_global_first_nan_detector() -> Optional[FirstNaNDetector]:
    """Get the global first-NaN detector instance."""
    return _global_first_nan_detector

def initialize_global_first_nan_detector(log_dir: str = "temp", enabled: bool = True) -> FirstNaNDetector:
    """Initialize the global first-NaN detector."""
    global _global_first_nan_detector
    _global_first_nan_detector = FirstNaNDetector(log_dir=log_dir, enabled=enabled)
    return _global_first_nan_detector

def cleanup_global_first_nan_detector():
    """Cleanup the global first-NaN detector."""
    global _global_first_nan_detector
    if _global_first_nan_detector:
        _global_first_nan_detector.finalize_logging()
        _global_first_nan_detector = None
