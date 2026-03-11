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

from hex_ai.models import compute_move_stage, compute_value_loss, MAX_LOG_COSH_INPUT_ABS
from hex_ai.nan_debug_utils import (
    check_for_nan_and_debug, calculate_validation_metrics_statistics, create_batch_analysis, find_nan_batch_indices,
    initialize_global_first_nan_detector, get_global_first_nan_detector, cleanup_global_first_nan_detector
)

from .config import VERBOSE_LEVEL
from .models import TwoHeadedResNet
from .config import (
    LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, POLICY_LOSS_WEIGHT, VALUE_LOSS_WEIGHT,
    BOARD_SIZE, POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE
)
from hex_ai.data_pipeline import discover_training_data_files_all
from hex_ai.training_utils import get_device, TrainingUtilities
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


# =============================================================================
# LOSS FUNCTION HYPERPARAMETERS
# =============================================================================

# Policy entropy regularization weight
DEFAULT_ENTROPY_WEIGHT = 1e-3
# Rationale: Encourages policy uncertainty to prevent overconfidence and logit explosion

# Label smoothing factor for policy targets over legal moves
DEFAULT_LABEL_SMOOTHING = 0.1
# Rationale: Prevents overconfidence by smoothing targets over legal moves

# Targets with support strictly larger than this threshold are treated as
# non-one-hot distributions and bypass extra label smoothing.
TARGET_SUPPORT_EPSILON = 1e-6

# Optional training-time regularization for non-one-hot targets:
# mix a small amount of uniform legal mass into soft targets.
# Default adds 1.7% legal-uniform mass; set 0.0 for strict behavior.
DEFAULT_SOFT_TARGET_LEGAL_MIX_ALPHA = 0.017

# L2 penalty on centered logits to prevent explosion
DEFAULT_LOGITS_L2_LAMBDA = 1e-5
# Rationale: Directly penalizes logit scale/variance to prevent gradient explosion
# GPT recommendation: start at 1e-6, increase to 5e-6, 1e-5, or rarely 3e-5 if needed


def get_legal_moves_from_board_tensor(board: torch.Tensor) -> torch.Tensor:
    """
    Determine legal moves from board state tensor.

    Legal moves in Hex are the empty cells, derived from the blue/red stone
    occupancy channels. The helper is kept at module scope so analysis code can
    reuse the exact training semantics without instantiating the trainer.
    """
    blue_channel = board[:, 0]
    red_channel = board[:, 1]
    empty_positions = (blue_channel == 0) & (red_channel == 0)
    return empty_positions.view(board.shape[0], -1)


def is_one_hot_like_target(
    target: torch.Tensor,
    *,
    support_epsilon: float = TARGET_SUPPORT_EPSILON,
) -> torch.Tensor:
    """
    Return a per-row mask indicating whether targets are one-hot-like.

    A row is considered one-hot-like when exactly one action has mass above
    `support_epsilon`. This matches the training-time branch that keeps legacy
    label smoothing for played-move targets while leaving richer search targets
    as soft distributions.
    """
    if target.ndim != 2:
        raise RuntimeError(
            f"CRITICAL BUG: target must be 2D in is_one_hot_like_target, got shape {tuple(target.shape)}"
        )
    support_count = (target > support_epsilon).sum(dim=1)
    return support_count == 1


def materialize_policy_targets_for_loss(
    policy_target: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    label_smoothing: float,
    soft_target_legal_mix_alpha: float,
    support_epsilon: float = TARGET_SUPPORT_EPSILON,
) -> torch.Tensor:
    """
    Normalize and regularize policy targets exactly as training does pre-loss.

    This intentionally excludes the illegal-mass debug path, which remains in
    `PolicyValueLoss._compute_policy_loss()` so training still emits the richer
    diagnostics when malformed data is encountered.
    """
    if policy_target.ndim != 2 or policy_target.shape != legal_mask.shape:
        raise RuntimeError(
            "CRITICAL BUG: policy_target/legal_mask shape mismatch in "
            "materialize_policy_targets_for_loss(). "
            f"Got policy_target={tuple(policy_target.shape)}, legal_mask={tuple(legal_mask.shape)}"
        )

    target = policy_target
    if not torch.isfinite(target).all():
        raise RuntimeError(
            "CRITICAL BUG: policy_target contains non-finite values in "
            "materialize_policy_targets_for_loss()."
        )
    if (target < 0).any():
        raise RuntimeError(
            "CRITICAL BUG: policy_target contains negative values in "
            "materialize_policy_targets_for_loss()."
        )

    target_mass = target.sum(dim=1, keepdim=True)
    if (target_mass <= 0).any():
        raise RuntimeError(
            "CRITICAL BUG: policy_target has non-positive probability mass in "
            "materialize_policy_targets_for_loss(). Zero-vector targets should "
            "have been filtered before this call."
        )
    target = target / target_mass.clamp_min(1e-12)

    one_hot_like_mask = is_one_hot_like_target(
        target, support_epsilon=support_epsilon
    )
    legal_counts = legal_mask.sum(dim=1).clamp_min(1)
    uniform = legal_mask.to(dtype=target.dtype) / legal_counts.unsqueeze(1).to(
        dtype=target.dtype
    )

    if soft_target_legal_mix_alpha > 0:
        alpha = float(soft_target_legal_mix_alpha)
        soft_target_mask = (~one_hot_like_mask).unsqueeze(1)
        if soft_target_mask.any():
            mixed_target = (1 - alpha) * target + alpha * uniform
            target = torch.where(soft_target_mask, mixed_target, target)

    if label_smoothing > 0:
        epsilon = float(label_smoothing)
        one_hot_like_mask_expanded = one_hot_like_mask.unsqueeze(1)
        if one_hot_like_mask_expanded.any():
            smoothed_target = (1 - epsilon) * target + epsilon * uniform
            target = torch.where(one_hot_like_mask_expanded, smoothed_target, target)

    return target

class PolicyValueLoss(nn.Module):
    """Combined loss for policy and value heads with support for missing policy targets."""
    
    def __init__(self, policy_weight: float = POLICY_LOSS_WEIGHT, value_weight: float = VALUE_LOSS_WEIGHT, 
                 entropy_weight: float = DEFAULT_ENTROPY_WEIGHT, label_smoothing: float = DEFAULT_LABEL_SMOOTHING, 
                 logits_l2_lambda: float = DEFAULT_LOGITS_L2_LAMBDA,
                 soft_target_legal_mix_alpha: float = DEFAULT_SOFT_TARGET_LEGAL_MIX_ALPHA):
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.label_smoothing = label_smoothing
        self.logits_l2_lambda = logits_l2_lambda
        self.soft_target_legal_mix_alpha = float(soft_target_legal_mix_alpha)
        if not (0.0 <= self.soft_target_legal_mix_alpha < 1.0):
            raise ValueError(
                "soft_target_legal_mix_alpha must be in [0, 1), "
                f"got {self.soft_target_legal_mix_alpha}"
            )
        self.policy_loss = nn.CrossEntropyLoss()
        # Value loss is now handled by the new compute_value_loss function
    
    def _get_legal_moves_from_board(self, board: torch.Tensor) -> torch.Tensor:
        """
        Determine legal moves from board state.
        
        In Hex, legal moves are simply any position that doesn't already have a piece on it.
        
        Args:
            board: Board tensor of shape (batch_size, 3, height, width) where channels are [blue_channel, red_channel, player_channel]
            
        Returns:
            Legal moves mask of shape (batch_size, height * width) where True indicates legal moves
        """
        return get_legal_moves_from_board_tensor(board)
    
    def _debug_illegal_targets(self, board: torch.Tensor, policy_target: torch.Tensor, 
                              legal_mask: torch.Tensor, target_indices: torch.Tensor, 
                              target_is_legal: torch.Tensor):
        """
        Debug illegal targets by displaying board states and move information.
        
        This method provides detailed debugging information when illegal targets are detected,
        including visual board representations and move analysis.
        """
        try:
            from hex_ai.utils.format_conversion import board_2nxn_to_nxn, rowcol_to_trmph
            from hex_ai.inference.board_display import display_hex_board
            import numpy as np
            import pickle
            import gzip
            from pathlib import Path
            from datetime import datetime
            
            print("\n" + "="*80)
            print("CRITICAL BUG: ILLEGAL TARGETS DETECTED")
            print("="*80)
            
            # Find the first illegal target for detailed analysis
            illegal_indices = (~target_is_legal).nonzero(as_tuple=False).squeeze(-1)
            if illegal_indices.numel() > 0:
                sample_idx = illegal_indices[0].item()
                
                print(f"\nAnalyzing first illegal target (sample {sample_idx}):")
                print(f"Target index: {target_indices[sample_idx].item()}")
                print(f"Target is legal: {target_is_legal[sample_idx].item()}")
                
                # Get board state
                board_sample = board[sample_idx]  # (3, height, width)
                blue_channel = board_sample[0].cpu().numpy()  # (height, width)
                red_channel = board_sample[1].cpu().numpy()   # (height, width)
                player_channel = board_sample[2].cpu().numpy()  # (height, width)
                
                # Convert to display format
                board_2d = np.zeros_like(blue_channel, dtype=str)
                board_2d[:] = 'e'  # empty
                board_2d[blue_channel == 1] = 'b'  # blue
                board_2d[red_channel == 1] = 'r'   # red
                
                print(f"\nBoard state (sample {sample_idx}):")
                print("Blue channel (1=blue piece, 0=empty):")
                print(blue_channel)
                print("Red channel (1=red piece, 0=empty):")
                print(red_channel)
                print("Player channel (1=blue to move, -1=red to move):")
                print(player_channel)
                
                # Display the board visually
                print(f"\nVisual board representation:")
                display_hex_board(board_2d)
                
                # Analyze legal moves
                legal_moves_flat = legal_mask[sample_idx].cpu().numpy()  # (height*width,)
                legal_moves_2d = legal_moves_flat.reshape(blue_channel.shape)
                
                print(f"\nLegal moves mask (True=legal, False=illegal):")
                print(legal_moves_2d.astype(int))
                
                # Find the target position
                target_idx = target_indices[sample_idx].item()
                height, width = blue_channel.shape
                target_row = target_idx // width
                target_col = target_idx % width
                
                print(f"\nTarget analysis:")
                print(f"Target index: {target_idx}")
                print(f"Target position: row={target_row}, col={target_col}")
                print(f"Target is legal: {legal_moves_2d[target_row, target_col]}")
                print(f"Target position has blue piece: {blue_channel[target_row, target_col] == 1}")
                print(f"Target position has red piece: {red_channel[target_row, target_col] == 1}")
                
                # Convert to TRMPH format for easier debugging
                try:
                    trmph_move = rowcol_to_trmph(target_row, target_col, height)
                    print(f"Target move in TRMPH format: {trmph_move}")
                except Exception as e:
                    print(f"Could not convert to TRMPH format: {e}")
                
                # Show all legal moves
                legal_positions = np.where(legal_moves_2d)
                print(f"\nAll legal positions: {list(zip(legal_positions[0], legal_positions[1]))}")
                
                # Show policy target distribution
                policy_target_sample = policy_target[sample_idx].cpu().numpy()
                print(f"\nPolicy target distribution (top 5 values):")
                top_indices = np.argsort(policy_target_sample)[-5:][::-1]
                for idx in top_indices:
                    row, col = idx // width, idx % width
                    print(f"  Index {idx} (row={row}, col={col}): {policy_target_sample[idx]:.6f}")
                
                # Check if there are other illegal targets
                if illegal_indices.numel() > 1:
                    print(f"\nOther illegal targets in this batch:")
                    for i in range(1, min(illegal_indices.numel(), 5)):  # Show up to 5
                        other_idx = illegal_indices[i].item()
                        other_target = target_indices[other_idx].item()
                        other_row = other_target // width
                        other_col = other_target % width
                        print(f"  Sample {other_idx}: target={other_target} (row={other_row}, col={other_col})")
                
                # Save the problematic sample data for further analysis
                self._save_problematic_sample(board, policy_target, legal_mask, target_indices, 
                                            target_is_legal, sample_idx)
            
            print("\n" + "="*80)
            print("END OF ILLEGAL TARGET DEBUG INFO")
            print("="*80)
            
        except Exception as e:
            print(f"Error in debug function: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_problematic_sample(self, board: torch.Tensor, policy_target: torch.Tensor, 
                                legal_mask: torch.Tensor, target_indices: torch.Tensor, 
                                target_is_legal: torch.Tensor, sample_idx: int):
        """
        Save the problematic training sample data to a pickle file for further analysis.
        
        This allows us to reconstruct the problem and trace it back to the original record.
        """
        try:
            import pickle
            import gzip
            from pathlib import Path
            from datetime import datetime
            
            # Create temp directory if it doesn't exist
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            
            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"illegal_target_sample_{timestamp}.pkl.gz"
            filepath = temp_dir / filename
            
            # Prepare the sample data
            sample_data = {
                'timestamp': timestamp,
                'sample_idx': sample_idx,
                'batch_size': board.shape[0],
                'board_size': board.shape[2],  # Assuming square board
                
                # The problematic sample data
                'problematic_sample': {
                    'board': board[sample_idx].cpu().numpy(),  # (3, height, width)
                    'policy_target': policy_target[sample_idx].cpu().numpy(),  # (height*width,)
                    'legal_mask': legal_mask[sample_idx].cpu().numpy(),  # (height*width,)
                    'target_index': target_indices[sample_idx].item(),
                    'target_is_legal': target_is_legal[sample_idx].item(),
                },
                
                # The entire batch for context
                'full_batch': {
                    'board': board.cpu().numpy(),  # (batch_size, 3, height, width)
                    'policy_target': policy_target.cpu().numpy(),  # (batch_size, height*width)
                    'legal_mask': legal_mask.cpu().numpy(),  # (batch_size, height*width)
                    'target_indices': target_indices.cpu().numpy(),  # (batch_size,)
                    'target_is_legal': target_is_legal.cpu().numpy(),  # (batch_size,)
                },
                
                # Analysis information
                'analysis': {
                    'illegal_count': int((~target_is_legal).sum().item()),
                    'total_samples': board.shape[0],
                    'board_shape': list(board.shape),
                    'policy_shape': list(policy_target.shape),
                }
            }
            
            # Save to compressed pickle file
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(sample_data, f)
            
            print(f"\n💾 SAVED PROBLEMATIC SAMPLE DATA")
            print(f"   File: {filepath}")
            print(f"   Sample {sample_idx} has illegal target at index {target_indices[sample_idx].item()}")
            print(f"   This data can be used to reconstruct the problem and trace back to the original record")
            
            # Also save a human-readable summary
            summary_file = temp_dir / f"illegal_target_summary_{timestamp}.txt"
            with open(summary_file, 'w') as f:
                f.write(f"ILLEGAL TARGET SAMPLE ANALYSIS\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Sample Index: {sample_idx}\n")
                f.write(f"Target Index: {target_indices[sample_idx].item()}\n")
                f.write(f"Target Is Legal: {target_is_legal[sample_idx].item()}\n")
                f.write(f"Illegal Count: {int((~target_is_legal).sum().item())}\n")
                f.write(f"Total Samples: {board.shape[0]}\n")
                f.write(f"Board Shape: {list(board.shape)}\n")
                f.write(f"Policy Shape: {list(policy_target.shape)}\n")
                f.write(f"\nData saved to: {filepath}\n")
                f.write(f"Use this data to reconstruct the problem and trace back to the original record.\n")
            
            print(f"   Summary: {summary_file}")
            
        except Exception as e:
            print(f"❌ Failed to save problematic sample data: {e}")
            import traceback
            traceback.print_exc()
    
    def _apply_label_smoothing(self, policy_target: torch.Tensor, board: torch.Tensor) -> torch.Tensor:
        """
        Apply label smoothing over legal moves.
        
        Args:
            policy_target: Original one-hot policy target of shape (batch_size, policy_output_size)
            board: Board tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Smoothed policy target of shape (batch_size, policy_output_size)
        """
        if self.label_smoothing <= 0:
            return policy_target
        
        # Get legal moves mask
        legal_moves = self._get_legal_moves_from_board(board)  # (batch_size, height * width)
        
        # Count legal moves per batch
        num_legal_moves = legal_moves.sum(dim=1, keepdim=True)  # (batch_size, 1)
        
        # Create smoothed targets
        smoothed_target = torch.zeros_like(policy_target)
        
        # For each batch
        for batch_idx in range(policy_target.shape[0]):
            batch_legal = legal_moves[batch_idx]  # (height * width,)
            batch_target = policy_target[batch_idx]  # (height * width,)
            batch_num_legal = num_legal_moves[batch_idx].item()
            
            if batch_num_legal == 0:
                # No legal moves (shouldn't happen in normal training)
                continue
            
            # Find the chosen move (where target is 1.0)
            chosen_move_idx = batch_target.argmax().item()
            
            # Apply label smoothing: (1-ε) for chosen move, ε/num_legal for other legal moves
            epsilon = self.label_smoothing
            smoothed_target[batch_idx, chosen_move_idx] = 1.0 - epsilon
            
            # Distribute epsilon evenly among all legal moves
            epsilon_per_legal = epsilon / batch_num_legal
            smoothed_target[batch_idx, batch_legal] += epsilon_per_legal
            
            # Ensure the chosen move gets the correct total probability
            smoothed_target[batch_idx, chosen_move_idx] = 1.0 - epsilon + epsilon_per_legal
        
        return smoothed_target

    @staticmethod
    def _is_one_hot_like_target(
        target: torch.Tensor,
        *,
        support_epsilon: float = TARGET_SUPPORT_EPSILON,
    ) -> torch.Tensor:
        """
        Return a per-row mask indicating whether targets are one-hot-like.

        A row is considered one-hot-like when exactly one action has mass above
        `support_epsilon`. This allows us to keep legacy label smoothing for
        played-move one-hot targets while preserving richer soft search targets.
        """
        return is_one_hot_like_target(target, support_epsilon=support_epsilon)
    
    def forward(self, policy_pred: torch.Tensor, value_pred: torch.Tensor,
                policy_target: torch.Tensor, value_target: torch.Tensor, 
                board: torch.Tensor = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined policy and value loss with direct logit scale penalty.
        
        This implements the approach from GPT that directly penalizes logit scale
        to prevent explosion, combined with entropy regularization and label smoothing.
        
        Args:
            policy_pred: Predicted policy logits (batch_size, policy_output_size) - RAW logits before masking
            value_pred: Predicted value (batch_size, 1) in [-1,1] range
            policy_target: Target policy probabilities (batch_size, policy_output_size) or None
            value_target: Target value (batch_size, 1) in [-1,1] range (signed)
            board: Board tensor for legal move detection (batch_size, 3, height, width)
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses
        """
        # Value loss using new log-cosh loss with label smoothing
        value_loss = compute_value_loss(value_pred, value_target, smooth=0.95)

        # Positions with zero-vector policy targets should not contribute to policy-head updates.
        # This covers post-game terminal positions (no next move target) and masked samples.
        # Trainable terminal-winning moves (provenance code 'T') still have normal non-zero
        # policy targets and continue to train the policy head.
        batch_size = policy_target.shape[0]
        zero_vectors = (policy_target.sum(dim=1) == 0.0)  # (batch_size,)
        terminal_count = zero_vectors.sum().item()
        non_terminal_indices = ~zero_vectors
        
        # ----- Logit L2 on legal moves only, centered over legal positions -----
        # This directly penalizes the scale/variance of legal logits to prevent explosion
        # Apply this regardless of whether we have policy targets
        if board is None:
            raise RuntimeError(
                "CRITICAL BUG: board tensor is None in PolicyValueLoss.forward(). "
                "The board tensor is required for legal move masking and L2 regularization. "
                "This indicates a bug in the training pipeline where boards are not being passed to the loss function. "
                "Check that all calls to criterion() include the board tensor as the 5th argument."
            )
        
        legal_mask = self._get_legal_moves_from_board(board)
        
        # Compute per-sample mean over legal entries only
        with torch.no_grad():
            legal_sum = (policy_pred * legal_mask).sum(dim=1, keepdim=True)
            legal_counts = legal_mask.sum(dim=1, keepdim=True).clamp_min(1)
            legal_mean = legal_sum / legal_counts
        
        # Center logits over legal positions only, zero elsewhere
        centered = (policy_pred - legal_mean) * legal_mask
        # Normalize by number of legal entries so batches with many illegals aren't over-penalized.
        # Only retain policy-regularization signal for samples with real policy targets.
        logits_l2_per_sample = centered.pow(2).sum(dim=1) / legal_counts.squeeze(1)
        if non_terminal_indices.any():
            logits_l2 = logits_l2_per_sample[non_terminal_indices].mean()
            logits_l2_loss = self.logits_l2_lambda * logits_l2
        else:
            logits_l2 = torch.tensor(0.0, dtype=policy_pred.dtype, device=policy_pred.device)
            logits_l2_loss = torch.tensor(
                0.0, dtype=policy_pred.dtype, device=policy_pred.device, requires_grad=True
            )
        
        # TEMPORARY: Enhanced NaN detection for logits L2 calculation
        # TODO: Remove after confirming training stability (3+ successful runs)
        from hex_ai.nan_debug_utils import get_global_first_nan_detector
        first_nan_detector = get_global_first_nan_detector()
        
        if first_nan_detector and torch.isnan(logits_l2_loss) and not first_nan_detector.nan_detected:
            first_nan_detector.first_nan_logger.info("🚨 NaN DETECTED IN LOGITS L2 CALCULATION 🚨")
            first_nan_detector.first_nan_logger.info(f"policy_pred range: [{policy_pred.min().item():.6f}, {policy_pred.max().item():.6f}]")
            first_nan_detector.first_nan_logger.info(f"legal_mask sum: {legal_mask.sum()}")
            first_nan_detector.first_nan_logger.info(f"legal_sum: {legal_sum.flatten()}")
            first_nan_detector.first_nan_logger.info(f"legal_counts: {legal_counts.flatten()}")
            first_nan_detector.first_nan_logger.info(f"legal_mean: {legal_mean.flatten()}")
            first_nan_detector.first_nan_logger.info(f"legal_mean has NaN: {torch.isnan(legal_mean).any()}")
            first_nan_detector.first_nan_logger.info(f"legal_mean has Inf: {torch.isinf(legal_mean).any()}")
            first_nan_detector.first_nan_logger.info(f"centered range: [{centered.min().item():.6f}, {centered.max().item():.6f}]")
            first_nan_detector.first_nan_logger.info(f"centered has NaN: {torch.isnan(centered).any()}")
            first_nan_detector.first_nan_logger.info(f"centered has Inf: {torch.isinf(centered).any()}")
            centered_squared = centered.pow(2)
            first_nan_detector.first_nan_logger.info(f"centered_squared range: [{centered_squared.min().item():.6f}, {centered_squared.max().item():.6f}]")
            first_nan_detector.first_nan_logger.info(f"centered_squared has NaN: {torch.isnan(centered_squared).any()}")
            first_nan_detector.first_nan_logger.info(f"centered_squared has Inf: {torch.isinf(centered_squared).any()}")
            squared_sum = centered_squared.sum(dim=1)
            first_nan_detector.first_nan_logger.info(f"squared_sum range: [{squared_sum.min().item():.6f}, {squared_sum.max().item():.6f}]")
            first_nan_detector.first_nan_logger.info(f"squared_sum has NaN: {torch.isnan(squared_sum).any()}")
            first_nan_detector.first_nan_logger.info(f"squared_sum has Inf: {torch.isinf(squared_sum).any()}")
            legal_counts_squeezed = legal_counts.squeeze(1)
            first_nan_detector.first_nan_logger.info(f"legal_counts_squeezed: {legal_counts_squeezed}")
            first_nan_detector.first_nan_logger.info(f"legal_counts_squeezed has NaN: {torch.isnan(legal_counts_squeezed).any()}")
            first_nan_detector.first_nan_logger.info(f"legal_counts_squeezed has Inf: {torch.isinf(legal_counts_squeezed).any()}")
            normalized = squared_sum / legal_counts_squeezed
            first_nan_detector.first_nan_logger.info(f"normalized range: [{normalized.min().item():.6f}, {normalized.max().item():.6f}]")
            first_nan_detector.first_nan_logger.info(f"normalized has NaN: {torch.isnan(normalized).any()}")
            first_nan_detector.first_nan_logger.info(f"normalized has Inf: {torch.isinf(normalized).any()}")
            first_nan_detector.first_nan_logger.info(f"logits_l2: {logits_l2.item()}")
            first_nan_detector.first_nan_logger.info(f"logits_l2 is NaN: {torch.isnan(logits_l2)}")
            first_nan_detector.first_nan_logger.info(f"logits_l2_lambda: {self.logits_l2_lambda}")
            first_nan_detector.first_nan_logger.info(f"logits_l2_loss: {logits_l2_loss.item()}")
            first_nan_detector.first_nan_logger.info(f"logits_l2_loss is NaN: {torch.isnan(logits_l2_loss)}")
        
        # Policy loss: handle zero-vector policy targets (terminal/masked samples)
        if terminal_count > 0:
            # Mixed batch - process only non-terminal moves
            if non_terminal_indices.any():
                # Process only non-terminal moves
                non_terminal_policy_pred = policy_pred[non_terminal_indices]
                non_terminal_policy_target = policy_target[non_terminal_indices]
                non_terminal_board = board[non_terminal_indices]
                
                policy_loss, entropy_loss = self._compute_policy_loss(
                    non_terminal_policy_pred, non_terminal_policy_target, non_terminal_board
                )
            else:
                # All samples are terminal moves (terminal_count == batch_size)
                # Create zero tensors that are properly connected to the computation graph
                # Use policy_pred to ensure proper dtype and device, then create zero scalar
                policy_loss = torch.tensor(0.0, dtype=policy_pred.dtype, device=policy_pred.device, requires_grad=True)
                entropy_loss = torch.tensor(0.0, dtype=policy_pred.dtype, device=policy_pred.device, requires_grad=True)
        else:
            # No terminal moves - process normally
            policy_loss, entropy_loss = self._compute_policy_loss(policy_pred, policy_target, board)
        
        # ----- Total loss -----
        total_loss = (self.policy_weight * policy_loss + 
                     self.value_weight * value_loss +
                     self.entropy_weight * entropy_loss +
                     logits_l2_loss)
        
        # TEMPORARY: Enhanced NaN detection for individual loss components
        # TODO: Remove after confirming training stability (3+ successful runs)
        # This will help identify exactly which component produces the first NaN
        from hex_ai.nan_debug_utils import get_global_first_nan_detector
        first_nan_detector = get_global_first_nan_detector()
        
        if first_nan_detector:
            # Check each component individually for NaN
            components = {
                'policy_loss': policy_loss,
                'value_loss': value_loss, 
                'entropy_loss': entropy_loss,
                'logits_l2_loss': logits_l2_loss,
                'total_loss': total_loss
            }
            
            # Check for NaN in any component
            nan_components = []
            for name, component in components.items():
                if torch.isnan(component):
                    nan_components.append(name)
            
            if nan_components and not first_nan_detector.nan_detected:
                # This is the first NaN - log detailed component analysis
                first_nan_detector._log_first_nan_components(
                    components, nan_components, policy_pred, value_pred, 
                    policy_target, value_target, board
                )
        
        # Create loss dictionary first
        loss_dict = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'logits_l2_loss': logits_l2_loss.item()
        }
        
        # CRITICAL: Check for NaN values and fail fast with detailed debugging
        check_for_nan_and_debug(
            policy_pred=policy_pred,
            value_pred=value_pred,
            total_loss=total_loss,
            loss_dict=loss_dict,
            boards=board,
            policies=policy_target,
            values=value_target,
            move_stage=torch.zeros(policy_pred.shape[0], 1, device=policy_pred.device),  # Placeholder
            context="loss_computation"
        )
        
        return total_loss, loss_dict

    def _compute_policy_loss(self, policy_pred: torch.Tensor, policy_target: torch.Tensor, board: torch.Tensor = None):
        """
        Compute policy loss for non-terminal moves.
        
        Args:
            policy_pred: Policy predictions from model
            policy_target: Policy targets (should not contain terminal moves)
            board: Board state for legal move masking
            
        Returns:
            Tuple of (policy_loss, entropy_loss)
        """
        # ----- Get legal moves mask -----
        if board is None:
            raise RuntimeError(
                "CRITICAL BUG: board tensor is None in _compute_policy_loss(). "
                "The board tensor is required for legal move masking in policy loss computation. "
                "This indicates a bug in the training pipeline where boards are not being passed to the loss function. "
                "Check that all calls to criterion() include the board tensor as the 5th argument."
            )
        
        legal_mask = self._get_legal_moves_from_board(board)  # (batch_size, height * width)
        
        # ----- Apply legal mask to logits -----
        logits = policy_pred
        # Use -1e4 instead of -1e9 to avoid overflow in float16 (Half precision)
        # float16 range is approximately -65504 to 65504
        logits = logits.masked_fill(~legal_mask.bool(), -1e4)
        
        # ----- Center logits to prevent mean drift -----
        # Subtract the mean of legal move logits from all logits
        # This keeps logits centered around 0 without changing probabilities
        with torch.no_grad():
            denom = legal_mask.sum(dim=1, keepdim=True).clamp_min(1)
            mean_legal = (logits.clamp_min(-1e3) * legal_mask).sum(dim=1, keepdim=True) / denom
        logits = torch.where(legal_mask, logits - mean_legal, logits)
        
        # ----- Policy loss with optional soft targets over legal moves -----
        if policy_target.ndim != 2 or policy_target.shape != logits.shape:
            raise RuntimeError(
                "CRITICAL BUG: policy_target shape mismatch in _compute_policy_loss(). "
                f"Expected {tuple(logits.shape)}, got {tuple(policy_target.shape)}"
            )

        target = policy_target.to(dtype=logits.dtype)
        if not torch.isfinite(target).all():
            raise RuntimeError(
                "CRITICAL BUG: policy_target contains non-finite values in _compute_policy_loss()."
            )
        if (target < 0).any():
            raise RuntimeError(
                "CRITICAL BUG: policy_target contains negative values in _compute_policy_loss()."
            )

        target_mass = target.sum(dim=1, keepdim=True)  # (batch_size, 1)
        if (target_mass <= 0).any():
            raise RuntimeError(
                "CRITICAL BUG: policy_target has non-positive probability mass in _compute_policy_loss(). "
                "Zero-vector targets should have been filtered before this call."
            )
        target = target / target_mass.clamp_min(1e-12)

        B, _V = logits.shape
        batch = torch.arange(B, device=logits.device)
        illegal_mass = (target * (~legal_mask).float()).sum(dim=1)  # (batch_size,)
        illegal_target_samples = illegal_mass > 1e-6
        bad_count = int(illegal_target_samples.sum().item())
        if bad_count > 0:
            target_indices = target.argmax(dim=1)
            target_is_legal = legal_mask[batch, target_indices]
            self._debug_illegal_targets(
                board, target, legal_mask, target_indices, target_is_legal
            )
            max_illegal_mass = float(illegal_mass[illegal_target_samples].max().item())
            raise RuntimeError(
                f"CRITICAL BUG: Found {bad_count} samples with illegal policy-target mass out of {B} total samples "
                f"(max illegal mass={max_illegal_mass:.6e}). "
                "This indicates a data pipeline issue where policy targets include illegal moves. "
                "Training stopped to prevent silent failures. Check debug output above for details."
            )

        target = materialize_policy_targets_for_loss(
            target,
            legal_mask,
            label_smoothing=self.label_smoothing,
            soft_target_legal_mix_alpha=self.soft_target_legal_mix_alpha,
            support_epsilon=TARGET_SUPPORT_EPSILON,
        )

        logp = torch.log_softmax(logits, dim=1)
        policy_loss = -(target * logp).sum(dim=1).mean()
        
        # ----- Entropy bonus (encourages spread) -----
        if self.entropy_weight > 0:
            p = torch.softmax(logits, dim=1)
            # Use a larger minimum value for float16 compatibility
            min_val = 1e-6 if p.dtype == torch.float16 else 1e-12
            entropy = -(p * torch.log(p.clamp_min(min_val))).sum(dim=1).mean()
            entropy_loss = -self.entropy_weight * entropy
            
            # TEMPORARY: Enhanced NaN detection for entropy calculation
            # TODO: Remove after confirming training stability (3+ successful runs)
            from hex_ai.nan_debug_utils import get_global_first_nan_detector
            first_nan_detector = get_global_first_nan_detector()
            
            if first_nan_detector and torch.isnan(entropy_loss) and not first_nan_detector.nan_detected:
                first_nan_detector.first_nan_logger.info("🚨 NaN DETECTED IN ENTROPY CALCULATION 🚨")
                first_nan_detector.first_nan_logger.info(f"logits range: [{logits.min().item():.6f}, {logits.max().item():.6f}]")
                first_nan_detector.first_nan_logger.info(f"softmax p range: [{p.min().item():.6f}, {p.max().item():.6f}]")
                first_nan_detector.first_nan_logger.info(f"p has NaN: {torch.isnan(p).any()}")
                first_nan_detector.first_nan_logger.info(f"p has Inf: {torch.isinf(p).any()}")
                first_nan_detector.first_nan_logger.info(f"p sum per sample: {p.sum(dim=1)}")
                first_nan_detector.first_nan_logger.info(f"min_val: {min_val}")
                p_clamped = p.clamp_min(min_val)
                first_nan_detector.first_nan_logger.info(f"p_clamped range: [{p_clamped.min().item():.6f}, {p_clamped.max().item():.6f}]")
                log_p = torch.log(p_clamped)
                first_nan_detector.first_nan_logger.info(f"log(p_clamped) range: [{log_p.min().item():.6f}, {log_p.max().item():.6f}]")
                first_nan_detector.first_nan_logger.info(f"log(p_clamped) has NaN: {torch.isnan(log_p).any()}")
                first_nan_detector.first_nan_logger.info(f"log(p_clamped) has Inf: {torch.isinf(log_p).any()}")
                entropy_term = -(p * log_p)
                first_nan_detector.first_nan_logger.info(f"entropy_term range: [{entropy_term.min().item():.6f}, {entropy_term.max().item():.6f}]")
                first_nan_detector.first_nan_logger.info(f"entropy_term has NaN: {torch.isnan(entropy_term).any()}")
                first_nan_detector.first_nan_logger.info(f"entropy_term has Inf: {torch.isinf(entropy_term).any()}")
                entropy_sum = entropy_term.sum(dim=1)
                first_nan_detector.first_nan_logger.info(f"entropy_sum range: [{entropy_sum.min().item():.6f}, {entropy_sum.max().item():.6f}]")
                first_nan_detector.first_nan_logger.info(f"entropy_sum has NaN: {torch.isnan(entropy_sum).any()}")
                first_nan_detector.first_nan_logger.info(f"entropy_sum has Inf: {torch.isinf(entropy_sum).any()}")
                first_nan_detector.first_nan_logger.info(f"entropy: {entropy.item()}")
                first_nan_detector.first_nan_logger.info(f"entropy is NaN: {torch.isnan(entropy)}")
                first_nan_detector.first_nan_logger.info(f"entropy_loss: {entropy_loss.item()}")
                first_nan_detector.first_nan_logger.info(f"entropy_loss is NaN: {torch.isnan(entropy_loss)}")
        else:
            entropy_loss = torch.zeros((), device=policy_pred.device, dtype=policy_pred.dtype)
        
        return policy_loss, entropy_loss


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
                    # TODO: Semi-urgent: Are we sure we can't use mixed precision with autocast?
                    #       I think I was using it before. I *definitely* used it for a long time with MPS.
                    # MPS mixed precision is problematic with autocast - disable for now
                    # MPS autocast can create float16 tensors without proper GradScaler support
                    # which leads to scalar type mismatches during backward pass
                    self.use_mixed_precision = False
                    self.scaler = None
                    logger.warning("WARNING: Mixed precision temprarily disabled for MPS due to scalar type issues. Using full precision.")
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
                 entropy_weight: float = DEFAULT_ENTROPY_WEIGHT,
                 label_smoothing: float = DEFAULT_LABEL_SMOOTHING,
                 logits_l2_lambda: float = DEFAULT_LOGITS_L2_LAMBDA,
                 weight_decay: float = 1e-4,
                 max_grad_norm: float = 20.0,
                 value_learning_rate_factor: float = 1.0,
                 value_weight_decay_factor: float = 1.0,
                 log_interval_batches: int = 200,
                 run_timestamp: Optional[str] = None,
                 shutdown_handler=None,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 use_policy_search_targets: bool = False,
                 soft_target_legal_mix_alpha: float = DEFAULT_SOFT_TARGET_LEGAL_MIX_ALPHA):
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
            entropy_weight: Weight for the policy entropy regularization (default: 1e-3).
            label_smoothing: Label smoothing factor for policy targets over legal moves (default: 0.1).
            logits_l2_lambda: L2 penalty on centered logits to prevent explosion (default: 1e-5).
            weight_decay: Weight decay for the optimizer.
            max_grad_norm: If not None, clip gradients to this max norm after backward(). Default: 20.0
            value_learning_rate_factor: Factor to multiply learning rate for value head (default: 1.0, no effect)
            value_weight_decay_factor: Factor to multiply weight decay for value head (default: 1.0, no effect)
            log_interval_batches: How often (in batches) to log progress during training (default: 200)
            run_timestamp: Optional timestamp for the entire run to use in log filenames
            betas: Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
            eps: Term added to the denominator to improve numerical stability (default: 1e-8)
            use_policy_search_targets: If True, training expects data loader policy
                tensors to come from per-position MCTS search targets.
            soft_target_legal_mix_alpha: Optional alpha in [0,1) used to mix
                uniform-legal mass into non-one-hot policy targets at training time.
                0.0 disables mixing and keeps strict original targets.

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
        self.use_policy_search_targets = bool(use_policy_search_targets)
        self.soft_target_legal_mix_alpha = float(soft_target_legal_mix_alpha)

        
        # Store hyperparameters for logging
        self.value_learning_rate_factor = value_learning_rate_factor
        self.value_weight_decay_factor = value_weight_decay_factor
        self.original_learning_rate = learning_rate  # Store the original learning rate
        self.betas = betas
        self.eps = eps
        
        # Note: Numerical stability warmup is now handled via epoch/mini_epoch/batch_idx checks
        
        # Initialize mixed precision
        self.mixed_precision = MixedPrecisionTrainer(device)
        
        # Create parameter groups for different learning rates and weight decay
        # Use the model's built-in policy head stability mechanisms
        
        def separate_params_by_weight_decay(params):
            """
            Separate parameters into those that should and shouldn't have weight decay.
            
            Weight decay should NOT be applied to:
            - BatchNorm/LayerNorm weight and bias parameters
            - All bias parameters (including Linear layer biases)
            
            Weight decay SHOULD be applied to:
            - Conv2d weight parameters
            - Linear weight parameters
            """
            weight_decay_params = []
            no_weight_decay_params = []
            
            for param in params:
                # Get the parameter name to check its type
                param_name = None
                for name, p in model.named_parameters():
                    if id(p) == id(param):
                        param_name = name
                        break
                
                if param_name is None:
                    raise RuntimeError(
                        f"CRITICAL BUG: Could not find parameter name for parameter {param}. "
                        f"This indicates a bug in the parameter separation logic. "
                        f"All model parameters should have identifiable names for proper weight decay handling."
                    )
                
                # Check if this is a norm layer parameter or bias
                if any(norm_type in param_name for norm_type in ['bn', 'norm', 'batch_norm', 'layer_norm']):
                    no_weight_decay_params.append(param)
                elif param_name.endswith('.bias'):
                    no_weight_decay_params.append(param)
                else:
                    weight_decay_params.append(param)
            
            return weight_decay_params, no_weight_decay_params
        
        # Get policy head parameter groups (with higher weight decay for final layer)
        policy_final_params = model.get_policy_head_final_layer_params()
        policy_other_params = model.get_policy_head_other_params()
        
        # Get value head parameters
        value_head_params = list(model.value_head.parameters())
        value_head_param_ids = {id(p) for p in value_head_params}
        
        # Get all other parameters (excluding policy head and value head)
        other_param_ids = {id(p) for p in policy_final_params + policy_other_params + value_head_params}
        trunk_params = [p for p in model.parameters() if id(p) not in other_param_ids]
        
        # Separate each parameter group by weight decay eligibility
        trunk_weight_decay, trunk_no_weight_decay = separate_params_by_weight_decay(trunk_params)
        policy_other_weight_decay, policy_other_no_weight_decay = separate_params_by_weight_decay(policy_other_params)
        policy_final_weight_decay, policy_final_no_weight_decay = separate_params_by_weight_decay(policy_final_params)
        value_head_weight_decay, value_head_no_weight_decay = separate_params_by_weight_decay(value_head_params)
        
        # Create parameter groups with proper weight decay separation
        param_groups = []
        
        # Trunk parameters
        if trunk_weight_decay:
            param_groups.append({
                'params': trunk_weight_decay,
                'lr': learning_rate,
                'weight_decay': weight_decay
            })
        if trunk_no_weight_decay:
            param_groups.append({
                'params': trunk_no_weight_decay,
                'lr': learning_rate,
                'weight_decay': 0.0
            })
        
        # Policy head other parameters
        if policy_other_weight_decay:
            param_groups.append({
                'params': policy_other_weight_decay,
                'lr': learning_rate,
                'weight_decay': weight_decay
            })
        if policy_other_no_weight_decay:
            param_groups.append({
                'params': policy_other_no_weight_decay,
                'lr': learning_rate,
                'weight_decay': 0.0
            })
        
        # Policy head final layer parameters (higher weight decay)
        if policy_final_weight_decay:
            param_groups.append({
                'params': policy_final_weight_decay,
                'lr': learning_rate,
                'weight_decay': weight_decay * 2.0  # Higher weight decay for policy final layer
            })
        if policy_final_no_weight_decay:
            param_groups.append({
                'params': policy_final_no_weight_decay,
                'lr': learning_rate,
                'weight_decay': 0.0
            })
        
        # Value head parameters
        if value_head_weight_decay:
            param_groups.append({
                'params': value_head_weight_decay,
                'lr': learning_rate * value_learning_rate_factor,
                'weight_decay': weight_decay * value_weight_decay_factor
            })
        if value_head_no_weight_decay:
            param_groups.append({
                'params': value_head_no_weight_decay,
                'lr': learning_rate * value_learning_rate_factor,
                'weight_decay': 0.0
            })
        
        # Optimizer and loss
        self.optimizer = optim.AdamW(param_groups, betas=betas, eps=eps)
        self.criterion = PolicyValueLoss(policy_weight=policy_weight, value_weight=value_weight, 
                                        entropy_weight=entropy_weight, label_smoothing=label_smoothing,
                                        logits_l2_lambda=logits_l2_lambda,
                                        soft_target_legal_mix_alpha=soft_target_legal_mix_alpha)
        
        # Learning rate scheduler (ReduceLROnPlateau)
        # TODO: NOTE, I have increased min_lr to =2e-5 (from 1e-5), as a temporary check to see if learning becomes more faster.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, min_lr=2e-5
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
        logger.info(
            "Trainer policy target source: %s",
            "policy_search_target (MCTS distribution)"
            if self.use_policy_search_targets
            else "policy (played move one-hot)",
        )
        if self.use_policy_search_targets and label_smoothing > 0:
            logger.info(
                "Label smoothing is enabled (%.4f) while using policy_search_target; "
                "only one-hot-like targets will be smoothed over legal moves "
                "(non-one-hot search targets are left unchanged).",
                label_smoothing,
            )
        if self.use_policy_search_targets and self.soft_target_legal_mix_alpha > 0:
            logger.info(
                "Applying soft-target legal-uniform mixing with alpha=%.6f on non-one-hot targets.",
                self.soft_target_legal_mix_alpha,
            )
        
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
    
    
    def validate(self, epoch: int = None, mini_epoch: int = None) -> Dict[str, float]:
        """Validate the model."""
        if not self.val_loader:
            return {}
        
        # TEMPORARY: Initialize enhanced NaN detection if not already done
        # TODO: Remove after confirming training stability
        if get_global_first_nan_detector() is None:
            initialize_global_first_nan_detector(log_dir="temp", enabled=True)
        
        first_nan_detector = get_global_first_nan_detector()
        
        self.model.eval()
        val_losses = []
        val_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'entropy_loss': [],
            'logits_l2_loss': []
        }
        
        with torch.no_grad():
            for batch_idx, (boards, policies, values, move_stage) in enumerate(self.val_loader):
                    
                # Move to device
                boards, policies, values, move_stage = TrainingUtilities.move_batch_to_device(boards, policies, values, move_stage, self.device)
                
                # Forward pass with mixed precision
                with self.mixed_precision.autocast_context():
                    policy_pred, value_pred = self.model(boards, move_stage)
                    total_loss, loss_dict = self.criterion(policy_pred, value_pred, policies, values, boards)
                
                
                # Check for NaN in validation
                if first_nan_detector:
                    first_nan_detector.check_and_log_first_nan(
                        policy_pred=policy_pred,
                        value_pred=value_pred,
                        total_loss=total_loss,
                        loss_dict=loss_dict,
                        boards=boards,
                        policies=policies,
                        values=values,
                        move_stage=move_stage,
                        context="validation",
                        batch_idx=batch_idx,
                        epoch=epoch,
                        mini_epoch=mini_epoch
                    )
                
                # Check for NaN in validation
                check_for_nan_and_debug(
                    policy_pred=policy_pred,
                    value_pred=value_pred,
                    total_loss=total_loss,
                    loss_dict=loss_dict,
                    boards=boards,
                    policies=policies,
                    values=values,
                    move_stage=move_stage,
                    context="validation",
                    batch_idx=batch_idx,
                    epoch=epoch,
                    mini_epoch=mini_epoch
                )
                
                # Track metrics
                val_losses.append(loss_dict['total_loss'])
                for key in val_metrics:
                    val_metrics[key].append(loss_dict[key])
        
        # Check if val_metrics is empty before computing averages
        val_metrics_empty = all(len(values) == 0 for values in val_metrics.values())
        if val_metrics_empty:
            error_msg = (
                f"Validation dataset produced no data. "
                f"Epoch: {epoch}, Mini-epoch: {mini_epoch}. "
                f"This usually indicates the validation dataset has reached its max_examples_unaugmented limit "
                f"and needs to be reset. Check that validation dataset reset is working properly."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Compute validation averages
        val_avg = {key: float(np.mean(values)) if values else float('nan') for key, values in val_metrics.items()}
        
        # Check for NaN in validation averages
        val_avg_has_nan = any(np.isnan(v) for v in val_avg.values())
        if val_avg_has_nan and first_nan_detector and not first_nan_detector.nan_detected:
            first_nan_detector.log_averaging_nan(val_avg, val_metrics, epoch, mini_epoch)
        
        # Handle NaN in validation averages
        if val_avg_has_nan:
            self._handle_validation_nan_debug(val_avg, val_metrics, epoch, mini_epoch)
        
        return val_avg
    
    def _handle_validation_nan_debug(self, val_avg: Dict[str, float], val_metrics: Dict[str, List[float]], 
                                   epoch: Optional[int], mini_epoch: Optional[int]) -> None:
        """
        Handle NaN values in validation averages by capturing debug information and exiting.
        
        This method is called when validation averages contain NaN values, which indicates
        numerical instability. It captures comprehensive debug information and exits
        the training process to prevent further corruption.
        """
        import time
        import pickle
        import gzip
        import numpy as np
        from pathlib import Path
        
        # Create comprehensive debug information
        debug_info = {
            'timestamp': time.time(),
            'context': 'validation_averages',
            'epoch': epoch,
            'mini_epoch': mini_epoch,
            
            # Validation averages that contain NaN
            'val_avg': val_avg,
            'val_avg_has_nan': {k: np.isnan(v) for k, v in val_avg.items()},
            
            # Individual validation metrics (lists of values from each batch)
            'val_metrics': val_metrics,
            'val_metrics_stats': calculate_validation_metrics_statistics(val_metrics),
            'batch_analysis': create_batch_analysis(val_metrics),
            'nan_batch_indices': find_nan_batch_indices(val_metrics),
            
            # Model state information
            'model_info': {
                'device': str(next(self.model.parameters()).device),
                'dtype': str(next(self.model.parameters()).dtype),
                'num_parameters': sum(p.numel() for p in self.model.parameters()),
                'num_trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            },
            
            # Optimizer state
            'optimizer_info': {
                'type': type(self.optimizer).__name__,
                'learning_rate': self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else None,
                'momentum': self.optimizer.param_groups[0].get('momentum', None) if self.optimizer.param_groups else None,
                'weight_decay': self.optimizer.param_groups[0].get('weight_decay', None) if self.optimizer.param_groups else None
            },
            
            # Training configuration
            'training_config': {
                'mixed_precision': self.mixed_precision.use_mixed_precision if hasattr(self, 'mixed_precision') else None,
                'gradient_clipping': getattr(self, 'gradient_clipping', None),
                'best_val_loss': getattr(self, 'best_val_loss', None)
            },
            
            # Data loader information
            'data_loader_info': {
                'val_loader_exists': self.val_loader is not None,
                'val_loader_type': type(self.val_loader).__name__ if self.val_loader else None,
                'batch_size': getattr(self.val_loader, 'batch_size', None) if self.val_loader else None,
                'num_workers': getattr(self.val_loader, 'num_workers', None) if self.val_loader else None,
                'shuffle': getattr(self.val_loader, 'shuffle', None) if self.val_loader else None
            },
            
        }
        
        # Create temp directory if it doesn't exist
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        # Create filename with timestamp
        timestamp = int(time.time())
        debug_file = temp_dir / f"validation_nan_debug_epoch{epoch}_mini{mini_epoch}_{timestamp}.pkl.gz"
        
        # Dump debug data
        with gzip.open(debug_file, 'wb') as f:
            pickle.dump(debug_info, f)
        
        # Create detailed error message
        nan_components = [k for k, v in val_avg.items() if np.isnan(v)]
        error_msg = (
            f"CRITICAL: NaN detected in validation averages! "
            f"NaN components: {nan_components} | "
            f"Epoch: {epoch}, Mini-epoch: {mini_epoch} | "
            f"Debug data saved to: {debug_file} | "
            f"This indicates numerical instability in validation. "
            f"Training will exit to prevent further corruption. "
            f"See write_ups/nan_validation_investigation.md for detailed analysis and next steps."
        )
        
        # Log the error before raising
        logger.error(error_msg)
        
        # Raise error to exit training
        raise RuntimeError(error_msg)
        
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
        
        # Explicitly release memory to prevent accumulation
        del checkpoint
    
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
            print("Changing learning rate in load_checkpoint...", end="")
            logger.warning("Overriding checkpoint hyperparameters - optimizer state will be reset")
            # logger.info(f"Using hyperparameter learning rate: {self.original_learning_rate}")
            logger.info(f"Using hyperparameter value_learning_rate_factor: {self.value_learning_rate_factor}")
            logger.info(f"Using hyperparameter value_weight_decay_factor: {self.value_weight_decay_factor}")
            # Don't load optimizer state - let it use current hyperparameters
            # Get the actual learning rate from the optimizer to confirm
            actual_lr = self.optimizer.param_groups[0]['lr']
            actual_value_lr = self.optimizer.param_groups[-1]['lr']  # Value head is typically the last group
            print(f"Value learning rate (read from optimizer) now {actual_lr:.6f} (value head: {actual_value_lr:.6f}).")
        else:
            # Load optimizer state (preserves checkpoint hyperparameters)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} with checkpoint hyperparameters")
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        # Explicitly release memory to prevent accumulation
        del checkpoint
        import gc
        gc.collect()  # Force garbage collection





    def _update_loss_metrics(self, state: Dict, loss_dict: Dict) -> None:
        """Update loss metrics in the training state."""
        for key in state['mini_epoch_metrics']:
            state['mini_epoch_metrics'][key].append(loss_dict[key])


    def _apply_gradient_clipping(self, state: Dict) -> None:
        """Apply gradient clipping and track gradient norms."""
        # Calculate gradient norm before clipping (for diagnostic purposes)
        pre_clip_gradient_norm = None
        try:
            pre_clip_gradient_norm = get_gradient_norm(self.model)
            if pre_clip_gradient_norm > 0:
                state['gradient_norms'].append(pre_clip_gradient_norm)
        except Exception as e:
            print(f"[train_on_batches] Warning: Failed to calculate pre-clip gradient norm: {e}")
        
        # Clip gradients to avoid exploding gradients (if configured)
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
            
            # Calculate gradient norm after clipping (to verify clipping worked)
            post_clip_gradient_norm = None
            try:
                post_clip_gradient_norm = get_gradient_norm(self.model)
            except Exception as e:
                print(f"[train_on_batches] Warning: Failed to calculate post-clip gradient norm: {e}")
            
            # Store both values for logging
            if pre_clip_gradient_norm is not None and post_clip_gradient_norm is not None:
                state['gradient_norms'].append(post_clip_gradient_norm)  # Use post-clip for statistics

    def _handle_progress_logging(self, batch_idx: int, epoch: int, mini_epoch: int, state: Dict) -> None:
        """Handle progress logging for the current batch."""
        now = time.time()
        should_log = TrainingUtilities.should_log_progress(
            batch_idx, epoch, mini_epoch, state['next_log_batch'], 
            state['start_time'], state['last_time_log']
        )
        if should_log:
            state['last_time_log'] = now
            elapsed = now - state['start_time']
            print(
                f"[train_on_batches] Batch {batch_idx+1}: "
                f"total_loss={state['mini_epoch_metrics']['total_loss'][-1]:.4f}, "
                f"policy_loss={state['mini_epoch_metrics']['policy_loss'][-1]:.4f}, "
                f"value_loss={state['mini_epoch_metrics']['value_loss'][-1]:.4f}, "
                f"entropy_loss={state['mini_epoch_metrics']['entropy_loss'][-1]:.4f}, "
                f"logits_l2_loss={state['mini_epoch_metrics']['logits_l2_loss'][-1]:.6f} "
                f"(elapsed {elapsed:.1f}s)"
            )
            if batch_idx + 1 == state['next_log_batch']:
                exp_backoff = 2
                if batch_idx > 64:
                    exp_backoff = 1.5
                state['next_log_batch'] *= math.floor(exp_backoff)  # Exponential backoff

    def _log_csv_metrics(self, epoch: int, mini_epoch: int, state: Dict, val_metrics: Optional[Dict], 
                        diagnostics: Dict, mini_epoch_avg: Dict) -> None:
        """Log metrics to CSV with simplified parameter passing."""
        if not self.csv_logger:
            return
            
        csv_data = self._prepare_csv_logging_data(
            epoch, mini_epoch, state['batch_times'], val_metrics,
            diagnostics['gradient_norm'], diagnostics['post_clip_gradient_norm'],
            diagnostics['weight_stats'], diagnostics['gradient_stats'], 
            diagnostics['lr_stats'], diagnostics['gpu_memory_mb'], 
            diagnostics['best_val_loss']
        )
        
        # Add training metrics to the CSV data
        csv_data['train_metrics'] = mini_epoch_avg
        csv_data['val_metrics'] = val_metrics
        
        self.csv_logger.log_mini_epoch(**csv_data)



    def _prepare_csv_logging_data(self, epoch: int, mini_epoch: int, batch_times: List[float], 
                                val_metrics: Optional[Dict], gradient_norm: Optional[float],
                                post_clip_gradient_norm: Optional[float], weight_stats: Optional[Dict],
                                gradient_stats: Optional[Dict], lr_stats: Optional[Dict],
                                gpu_memory_mb: Optional[float], best_val_loss: Optional[float]) -> Dict:
        """Prepare data for CSV logging."""
        # Extract hyperparameters
        hp = self._get_hyperparameter_summary()
        
        if epoch is None or mini_epoch is None:
            raise RuntimeError(
                f"CRITICAL BUG: epoch or mini_epoch is None in CSV logging. "
                f"epoch={epoch}, mini_epoch={mini_epoch}. "
                f"This indicates a bug in the training pipeline where epoch/mini_epoch are not being passed properly. "
                f"Check that all calls to _prepare_csv_logging_data() include valid epoch and mini_epoch values."
            )
        epoch_id = TrainingUtilities.format_epoch_id(epoch, mini_epoch)
        mini_epoch_time = TrainingUtilities.calculate_mini_epoch_time(batch_times)
        
        return {
            'epoch': epoch_id,
            'hyperparams': hp,
            'training_time': mini_epoch_time,
            'epoch_time': mini_epoch_time,
            'samples_per_second': 0.0,  # Would need to calculate based on samples processed
            'memory_usage_mb': 0.0,  # Would need to calculate system memory usage
            'gpu_memory_mb': gpu_memory_mb,
            'gradient_norm': gradient_norm,
            'post_clip_gradient_norm': post_clip_gradient_norm,
            'weight_stats': weight_stats,
            'gradient_stats': gradient_stats,
            'lr_stats': lr_stats,
            'best_val_loss': best_val_loss,
            'notes': "train_on_batches"
        }

    def _initialize_training_state(self) -> Dict:
        """Initialize training state variables and return them."""
        return {
            'mini_epoch_metrics': {
                'policy_loss': [],
                'value_loss': [],
                'total_loss': [],
                'entropy_loss': [],
                'logits_l2_loss': []
            },
            'gradient_norms': [],
            'start_time': time.time(),
            'next_log_batch': 1,
            'last_time_log': time.time(),
            'data_load_start': time.time(),
            'batch_data_times': [],
            'batch_times': []
        }

    def _process_single_batch(self, batch_idx: int, boards: torch.Tensor, policies: torch.Tensor, 
                            values: torch.Tensor, state: Dict, move_stage: torch.Tensor, epoch: int = None, mini_epoch: int = None) -> Dict:
        """Process a single batch and return updated state."""
        
        # Calculate timing metrics
        timing = TrainingUtilities.calculate_batch_timing(state)
        
        # Move to device
        boards, policies, values, move_stage = TrainingUtilities.move_batch_to_device(boards, policies, values, move_stage, self.device)
        
        # Forward pass with mixed precision
        self.optimizer.zero_grad(set_to_none=True)
        with self.mixed_precision.autocast_context():
            policy_pred, value_pred = self.model(boards, move_stage)
            total_loss, loss_dict = self.criterion(policy_pred, value_pred, policies, values, boards)
        
        
        # Check for NaN in training
        first_nan_detector = get_global_first_nan_detector()
        if first_nan_detector:
            first_nan_detector.check_and_log_first_nan(
                policy_pred=policy_pred,
                value_pred=value_pred,
                total_loss=total_loss,
                loss_dict=loss_dict,
                boards=boards,
                policies=policies,
                values=values,
                move_stage=move_stage,
                context="training",
                batch_idx=batch_idx,
                epoch=epoch,
                mini_epoch=mini_epoch
            )
        
        # CRITICAL: Check for NaN in training with detailed debugging (existing system)
        check_for_nan_and_debug(
            policy_pred=policy_pred,
            value_pred=value_pred,
            total_loss=total_loss,
            loss_dict=loss_dict,
            boards=boards,
            policies=policies,
            values=values,
            move_stage=move_stage,
            context="training",
            batch_idx=batch_idx,
            epoch=epoch,
            mini_epoch=mini_epoch
        )
        
        
        # Backward pass: compute gradients for this batch
        scaled_loss = self.mixed_precision.scale_loss(total_loss)
        scaled_loss.backward()
        
        # Apply gradient clipping and track norms
        self._apply_gradient_clipping(state)
        
        # CRITICAL: Check for NaN and extreme values in gradients after backward pass
        has_nan_grad = False
        has_extreme_grad = False
        max_grad_norm = 0.0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                max_grad_norm = max(max_grad_norm, grad_norm)
                
                if torch.isnan(param.grad).any():
                    has_nan_grad = True
                    print(f"WARNING: NaN gradient detected in parameter {name}")
                
        
        if has_nan_grad:
            raise RuntimeError(
                f"NaN detected in gradients after backward pass! "
                f"This indicates gradient explosion. "
                f"Check learning rate (current: {self.optimizer.param_groups[0]['lr']}), "
                f"gradient clipping (current: {self.max_grad_norm}), "
                f"and model architecture."
            )
        
        if has_extreme_grad:
            raise RuntimeError(
                f"Extreme gradient norms detected! Max gradient norm: {max_grad_norm:.6f}. "
                f"This indicates potential gradient explosion. "
                f"Check learning rate (current: {self.optimizer.param_groups[0]['lr']}), "
                f"gradient clipping (current: {self.max_grad_norm}), "
                f"and model architecture."
            )
        
        # Optimizer step: update model parameters using accumulated gradients
        self.mixed_precision.step_optimizer(self.optimizer)
        # Update mixed precision scaler (if used)
        self.mixed_precision.update_scaler()
        
        # Track losses for this batch
        self._update_loss_metrics(state, loss_dict)
        
        # Prepare for next batch data timing
        state['data_load_start'] = time.time()
        batch_end_time = time.time()
        state['batch_times'].append(batch_end_time - timing['batch_start_time'])
        
        return state


    def _get_hyperparameter_summary(self) -> Dict[str, any]:
        """Get a summary of hyperparameters for logging."""
        return {
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'batch_size': self.train_loader.batch_size,
            'dataset_size': 'N/A',
            'network_structure': f"ResNet{getattr(self.model, 'num_blocks', '?')}",
            'policy_weight': getattr(self.criterion, 'policy_weight', ''),
            'value_weight': getattr(self.criterion, 'value_weight', ''),
            'total_loss_weight': getattr(self.criterion, 'policy_weight', 0) + getattr(self.criterion, 'value_weight', 0),
            'dropout_prob': 'N/A (not used in current architecture)',
            'weight_decay': self.optimizer.param_groups[0].get('weight_decay', 0.0),
            'max_grad_norm': getattr(self, 'max_grad_norm', ''),
            'value_learning_rate_factor': getattr(self, 'value_learning_rate_factor', ''),
            'value_weight_decay_factor': getattr(self, 'value_weight_decay_factor', ''),
            'betas': getattr(self, 'betas', ''),
            'eps': getattr(self, 'eps', '')
        }


    def _calculate_diagnostic_metrics(self, val_metrics: Optional[Dict], gradient_norms: List[float]) -> Dict:
        """Calculate diagnostic metrics for stability analysis."""
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
            gradient_stats = TrainingUtilities.calculate_statistics(gradient_norms)
        
        # Calculate weight statistics
        weight_stats = TrainingUtilities.calculate_weight_statistics(self.model)
        
        # Calculate learning rate statistics
        lr_stats = TrainingUtilities.calculate_learning_rate_statistics(self.optimizer)
        
        # Calculate GPU memory usage
        gpu_memory_mb = TrainingUtilities.get_gpu_memory_usage()

        
        return {
            'gradient_norm': gradient_norm,
            'post_clip_gradient_norm': post_clip_gradient_norm,
            'gradient_stats': gradient_stats,
            'weight_stats': weight_stats,
            'lr_stats': lr_stats,
            'gpu_memory_mb': gpu_memory_mb,
            'best_val_loss': best_val_loss
        }

    def train_on_batches(self, batch_iterable, epoch=None, mini_epoch=None, val_metrics=None) -> Dict[str, float]:
        """
        Train the model on a provided iterable of batches (mini-epoch).

        This is a lower-level training method that processes a specific set of batches
        without managing the overall training loop. It's designed for:
        - Mini-epoch orchestration (see MiniEpochOrchestrator)
        - Custom training loops that need fine-grained control
        - Integration with external training frameworks
        - Debugging and experimentation
        
        Args:
            batch_iterable: Iterable of batches to train on
            epoch: Current epoch number (for logging and numerical stability warmup)
            mini_epoch: Current mini-epoch number (for logging and numerical stability warmup)
            val_metrics: Validation metrics from previous validation

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
            # For mini-epoch orchestration:
            orchestrator = MiniEpochOrchestrator(trainer, train_loader, val_loader, mini_epoch_batches=500)
            orchestrator.run()
        """
        self.model.train()
        
        # Initialize training state
        state = self._initialize_training_state()
        
        # Reset policy monitoring flags for this mini-epoch
        if hasattr(self, '_policy_monitoring_logged'):
            self._policy_monitoring_logged = {'training': False, 'validation': False}
        
        # Note: Numerical stability warmup is handled via epoch/mini_epoch/batch_idx checks
        
        # Process each batch
        for batch_idx, batch_data in enumerate(batch_iterable):
            # Validate batch format - should always be 4 elements now
            if len(batch_data) != 4:
                raise ValueError(
                    f"Invalid batch format: expected 4 elements (boards, policies, values, move_stage), "
                    f"got {len(batch_data)}. This suggests the data pipeline is not providing move_stage. "
                    f"Check that StreamingMixedShardDataset._transform_example() returns 4 elements."
                )
            
            boards, policies, values, move_stage = batch_data
            state = self._process_single_batch(batch_idx, boards, policies, values, state, move_stage, epoch, mini_epoch)
            
            # Progress logging
            self._handle_progress_logging(batch_idx, epoch, mini_epoch, state)
        
        # Compute averages for the mini-epoch
        mini_epoch_avg = {
            key: float(np.mean(values)) if values else float('nan') 
            for key, values in state['mini_epoch_metrics'].items()
        }
        
        # CRITICAL: Check for NaN in mini-epoch averages
        for key, value in mini_epoch_avg.items():
            if np.isnan(value):
                raise RuntimeError(
                    f"NaN detected in mini-epoch average for {key}={value}! "
                    f"This indicates numerical instability has occurred during training. "
                    f"Check learning rate, gradient clipping, and model architecture."
                )
        
        # Calculate diagnostic metrics
        diagnostics = self._calculate_diagnostic_metrics(val_metrics, state['gradient_norms'])
        
        # CSV logging for mini-epoch
        self._log_csv_metrics(epoch, mini_epoch, state, val_metrics, diagnostics, mini_epoch_avg)
        
        # Learning rate scheduler step (ReduceLROnPlateau)
        if val_metrics and 'total_loss' in val_metrics:
            self.scheduler.step(val_metrics['total_loss'])
        
        
        return mini_epoch_avg
