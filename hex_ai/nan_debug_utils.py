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
