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
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from hex_ai.models import compute_move_stage, compute_value_loss, MAX_LOG_COSH_INPUT_ABS

from .config import VERBOSE_LEVEL
from .models import TwoHeadedResNet
from .config import (
    LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, POLICY_LOSS_WEIGHT, VALUE_LOSS_WEIGHT,
    BOARD_SIZE, POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE
)
from hex_ai.data_pipeline import discover_processed_files
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
# NUMERICAL STABILITY MONITORING CONSTANTS
# =============================================================================

# Policy logit thresholds for detecting loss of uncertainty
MAX_POLICY_LOGIT_ABS = 20.0  # Restored to normal value after fixing global pooling bug
# Rationale: Logits > 20 create softmax probabilities > 0.9999, indicating
# complete loss of uncertainty. The network becomes overconfident and can't
# express doubt, leading to training instability.

# Value output thresholds for detecting tanh saturation
MAX_VALUE_OUTPUT_ABS = 0.999
# Rationale: Values > 0.999 indicate tanh saturation, causing severe gradient
# vanishing (gradients < 0.002). This prevents the network from learning
# effectively from these positions.

# Loss value thresholds for detecting numerical instability
MAX_TOTAL_LOSS_ABS = 100.0
MAX_POLICY_LOSS_ABS = 50.0
MAX_VALUE_LOSS_ABS = 10.0
# Rationale: These thresholds detect unusual numerical behavior that may lead to NaN.
# With proper L2 penalty on logits, losses should stay within normal ranges.

# Gradient norm thresholds for detecting gradient explosion
MAX_GRADIENT_NORM = 100.0
# Rationale: Gradient norms > 100 indicate potential gradient explosion,
# which can cause numerical instability and NaN values.

# Warmup period for numerical stability checks
NUMERICAL_STABILITY_WARMUP_BATCHES = 20
# Rationale: Skip extreme value checks for the first N batches to allow the network
# to stabilize from random initialization. Early batches often have extreme values
# that are not indicative of actual training problems.

# =============================================================================
# LOSS FUNCTION HYPERPARAMETERS
# =============================================================================

# Policy entropy regularization weight
DEFAULT_ENTROPY_WEIGHT = 1e-3
# Rationale: Encourages policy uncertainty to prevent overconfidence and logit explosion

# Label smoothing factor for policy targets over legal moves
DEFAULT_LABEL_SMOOTHING = 0.1
# Rationale: Prevents overconfidence by smoothing targets over legal moves

# L2 penalty on centered logits to prevent explosion
DEFAULT_LOGITS_L2_LAMBDA = 1e-5
# Rationale: Directly penalizes logit scale/variance to prevent gradient explosion
# GPT recommendation: start at 1e-6, increase to 5e-6, 1e-5, or rarely 3e-5 if needed

class PolicyValueLoss(nn.Module):
    """Combined loss for policy and value heads with support for missing policy targets."""
    
    def __init__(self, policy_weight: float = POLICY_LOSS_WEIGHT, value_weight: float = VALUE_LOSS_WEIGHT, 
                 entropy_weight: float = DEFAULT_ENTROPY_WEIGHT, label_smoothing: float = DEFAULT_LABEL_SMOOTHING, 
                 logits_l2_lambda: float = DEFAULT_LOGITS_L2_LAMBDA):
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.label_smoothing = label_smoothing
        self.logits_l2_lambda = logits_l2_lambda
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
        # Extract blue and red channels (ignore player channel)
        blue_channel = board[:, 0]  # (batch_size, height, width)
        red_channel = board[:, 1]   # (batch_size, height, width)
        
        # A position is legal if both blue and red channels are 0 (empty)
        empty_positions = (blue_channel == 0) & (red_channel == 0)  # (batch_size, height, width)
        
        # Flatten to match policy output shape
        legal_moves = empty_positions.view(board.shape[0], -1)  # (batch_size, height * width)
        
        return legal_moves
    
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
        
        # ----- Logit L2 on centered logits (pre-mask) -----
        # This directly penalizes the scale/variance of logits to prevent explosion
        # Apply this regardless of whether we have policy targets
        with torch.no_grad():
            mean_per_row = policy_pred.mean(dim=1, keepdim=True)
        centered_logits = policy_pred - mean_per_row
        logits_l2 = (centered_logits.pow(2).mean())  # scalar
        logits_l2_loss = self.logits_l2_lambda * logits_l2
        
        # Policy loss: handle terminal moves by detecting zero vectors (original approach)
        # Terminal moves are represented as zero vectors in the data pipeline
        
        # Simple validation: check for zero vectors (terminal moves)
        batch_size = policy_target.shape[0]
        zero_vectors = (policy_target.sum(dim=1) == 0.0)  # (batch_size,)
        terminal_count = zero_vectors.sum().item()
        
        if terminal_count > 0:
            # Mixed batch - process only non-terminal moves
            non_terminal_indices = ~zero_vectors
            if non_terminal_indices.any():
                # Process only non-terminal moves
                non_terminal_policy_pred = policy_pred[non_terminal_indices]
                non_terminal_policy_target = policy_target[non_terminal_indices]
                non_terminal_board = board[non_terminal_indices] if board is not None else None
                
                policy_loss, entropy_loss = self._compute_policy_loss(
                    non_terminal_policy_pred, non_terminal_policy_target, non_terminal_board
                )
            else:
                # All samples are terminal moves (terminal_count == batch_size)
                policy_loss = torch.zeros((), device=policy_pred.device, dtype=policy_pred.dtype)
                entropy_loss = torch.zeros((), device=policy_pred.device, dtype=policy_pred.dtype)
        else:
            # No terminal moves - process normally
            policy_loss, entropy_loss = self._compute_policy_loss(policy_pred, policy_target, board)
        
        # ----- Total loss -----
        total_loss = (self.policy_weight * policy_loss + 
                     self.value_weight * value_loss +
                     self.entropy_weight * entropy_loss +
                     logits_l2_loss)
        
        # CRITICAL: Check for NaN values and fail fast
        if torch.isnan(total_loss) or torch.isnan(policy_loss) or torch.isnan(value_loss) or torch.isnan(entropy_loss) or torch.isnan(logits_l2_loss):
            raise RuntimeError(
                f"NaN detected in loss computation! "
                f"total_loss={total_loss.item()}, policy_loss={policy_loss.item()}, value_loss={value_loss.item()}, "
                f"entropy_loss={entropy_loss.item()}, logits_l2_loss={logits_l2_loss.item()}. "
                f"This indicates numerical instability. Check learning rate, gradient clipping, and model architecture."
            )
        
        # Check for extreme loss values that indicate numerical instability
        if (abs(total_loss.item()) > MAX_TOTAL_LOSS_ABS or 
            abs(policy_loss.item()) > MAX_POLICY_LOSS_ABS or 
            abs(value_loss.item()) > MAX_VALUE_LOSS_ABS):
            raise RuntimeError(
                f"Extreme loss values detected! "
                f"total_loss={total_loss.item():.6f}, policy_loss={policy_loss.item():.6f}, value_loss={value_loss.item():.6f}, "
                f"entropy_loss={entropy_loss.item():.6f}, logits_l2_loss={logits_l2_loss.item():.6f}. "
                f"These values are unusually high and may indicate numerical instability. "
                f"Check learning rate, gradient clipping, and model architecture."
            )
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'logits_l2_loss': logits_l2_loss.item()
        }
        
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
        legal_mask = None
        if board is not None:
            legal_mask = self._get_legal_moves_from_board(board)  # (batch_size, height * width)
        
        # ----- Apply legal mask to logits -----
        logits = policy_pred
        if legal_mask is not None:
            # Use -1e4 instead of -1e9 to avoid overflow in float16 (Half precision)
            # float16 range is approximately -65504 to 65504
            logits = logits.masked_fill(~legal_mask.bool(), -1e4)
        
        # ----- Policy loss with label smoothing over legal moves -----
        B, V = logits.shape
        target_indices = policy_target.argmax(dim=1)  # (batch_size,)
        
        # CRITICAL: Check for target-mask mismatch and handle illegal targets
        if legal_mask is not None:
            batch = torch.arange(B, device=logits.device)
            target_is_legal = legal_mask[batch, target_indices]  # (B,)
            bad_count = int((~target_is_legal).sum().item())
            
            if bad_count > 0:
                # CRITICAL: This is a bug that needs to be found and fixed!
                # Don't mask the error - fail immediately with detailed debugging info
                self._debug_illegal_targets(board, policy_target, legal_mask, target_indices, target_is_legal)
                raise RuntimeError(
                    f"CRITICAL BUG: Found {bad_count} samples with illegal targets out of {B} total samples! "
                    f"This indicates a data pipeline issue where policy targets point to illegal moves. "
                    f"Training stopped to prevent silent failures. Check debug output above for details."
                )
            else:
                # All targets are legal - proceed with normal computation
                if self.label_smoothing > 0:
                    # Build smoothed targets strictly over legal moves
                    target = torch.zeros_like(logits)
                    target[batch, target_indices] = 1.0
                    
                    legal_counts = legal_mask.sum(dim=1).clamp_min(1)
                    epsilon = self.label_smoothing
                    # Uniform over legal moves
                    uniform = legal_mask.float() / legal_counts.unsqueeze(1)
                    # Final smoothed distribution
                    target = (1 - epsilon) * target + epsilon * uniform
                    
                    logp = torch.log_softmax(logits, dim=1)
                    policy_loss = -(target * logp).sum(dim=1).mean()
                else:
                    policy_loss = F.cross_entropy(logits, target_indices, reduction='mean')
        else:
            # No legal mask - proceed with original logic
            if self.label_smoothing > 0:
                # Build smoothed targets over all moves
                target = torch.zeros_like(logits)
                target[batch, target_indices] = 1.0
                uniform = torch.full_like(logits, 1.0 / V)
                target = (1 - self.label_smoothing) * target + self.label_smoothing * uniform
                
                logp = torch.log_softmax(logits, dim=1)
                policy_loss = -(target * logp).sum(dim=1).mean()
            else:
                policy_loss = F.cross_entropy(logits, target_indices, reduction='mean')
        
        # ----- Entropy bonus (encourages spread) -----
        if self.entropy_weight > 0:
            p = torch.softmax(logits, dim=1)
            # Use a larger minimum value for float16 compatibility
            min_val = 1e-6 if p.dtype == torch.float16 else 1e-12
            entropy = -(p * torch.log(p.clamp_min(min_val))).sum(dim=1).mean()
            entropy_loss = -self.entropy_weight * entropy
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
                 eps: float = 1e-8):
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
                    # Fallback: assume it should have weight decay
                    weight_decay_params.append(param)
                    continue
                
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
                                        logits_l2_lambda=logits_l2_lambda)
        
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
    
    
    def validate(self, epoch: int = None, mini_epoch: int = None) -> Dict[str, float]:
        """Validate the model."""
        if not self.val_loader:
            return {}
        
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
            for boards, policies, values, move_stage in self.val_loader:
                    
                # Move to device
                boards, policies, values, move_stage = TrainingUtilities.move_batch_to_device(boards, policies, values, move_stage, self.device)
                
                # Forward pass with mixed precision
                with self.mixed_precision.autocast_context():
                    policy_pred, value_pred = self.model(boards, move_stage)
                    total_loss, loss_dict = self.criterion(policy_pred, value_pred, policies, values, boards)
                
                # CRITICAL: Check for NaN in validation
                if torch.isnan(policy_pred).any() or torch.isnan(value_pred).any() or torch.isnan(total_loss):
                    raise RuntimeError(
                        f"NaN detected in validation! "
                        f"policy_pred has NaN: {torch.isnan(policy_pred).any()}, "
                        f"value_pred has NaN: {torch.isnan(value_pred).any()}, "
                        f"total_loss is NaN: {torch.isnan(total_loss)}. "
                        f"This indicates numerical instability in the model."
                    )
                
                # Check for extreme values in validation
                policy_max_abs = torch.abs(policy_pred).max().item()
                value_max_abs = torch.abs(value_pred).max().item()
                
                # Check if we're in early training phase
                is_early_training = (epoch == 1 and mini_epoch == 1)
                
                if (not is_early_training and 
                    (policy_max_abs > MAX_POLICY_LOGIT_ABS or value_max_abs > MAX_VALUE_OUTPUT_ABS or abs(total_loss.item()) > MAX_TOTAL_LOSS_ABS)):
                    policy_range = f"[{policy_pred.min().item():.6f}, {policy_pred.max().item():.6f}]"
                    value_range = f"[{value_pred.min().item():.6f}, {value_pred.max().item():.6f}]"
                    
                    raise RuntimeError(
                        f"Extreme values detected in validation! "
                        f"policy_pred max_abs: {policy_max_abs:.6f}, range: {policy_range}, "
                        f"value_pred max_abs: {value_max_abs:.6f}, range: {value_range}, "
                        f"total_loss: {total_loss.item():.6f}. "
                        f"Policy logits > 20 or value outputs > 0.999 indicate numerical instability."
                    )
                
                # Track metrics
                val_losses.append(loss_dict['total_loss'])
                for key in val_metrics:
                    val_metrics[key].append(loss_dict[key])
        
        # Compute validation averages
        val_avg = {key: float(np.mean(values)) if values else float('nan') for key, values in val_metrics.items()}
        return val_avg
        
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
        keep_epochs = TrainingUtilities.get_checkpoints_to_keep(max_epoch, max_checkpoints)
        for e, fname in epoch_nums:
            if e not in keep_epochs:
                try:
                    os.remove(os.path.join(save_path, fname))
                except Exception:
                    pass



    def _update_loss_metrics(self, state: Dict, loss_dict: Dict) -> None:
        """Update loss metrics in the training state."""
        for key in state['mini_epoch_metrics']:
            state['mini_epoch_metrics'][key].append(loss_dict[key])

    def _apply_gradient_clipping(self, state: Dict) -> None:
        """Apply gradient clipping and track gradient norms."""
        # Calculate gradient norm before clipping (for diagnostic purposes)
        pre_clip_gradient_norm = None
        try:
            pre_clip_gradient_norm = TrainingUtilities.calculate_gradient_norm(self.model)
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
                post_clip_gradient_norm = TrainingUtilities.calculate_gradient_norm(self.model)
            except Exception as e:
                print(f"[train_on_batches] Warning: Failed to calculate post-clip gradient norm: {e}")
            
            # Store both values for logging
            if pre_clip_gradient_norm is not None and post_clip_gradient_norm is not None:
                state['gradient_norms'].append(post_clip_gradient_norm)  # Use post-clip for statistics
                # Store both values for debugging
                if not hasattr(self, 'gradient_clipping_debug'):
                    self.gradient_clipping_debug = []
                self.gradient_clipping_debug.append({
                    'pre_clip': pre_clip_gradient_norm,
                    'post_clip': post_clip_gradient_norm,
                    'clipped': pre_clip_gradient_norm > post_clip_gradient_norm
                })

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
        
        # Create a single metrics dictionary for cleaner CSV logging
        metrics_dict = {
            'epoch': csv_data['epoch_id'],
            'train_metrics': mini_epoch_avg,
            'val_metrics': val_metrics,
            'hyperparams': csv_data['hyperparams'],
            'training_time': csv_data['training_time'],
            'epoch_time': csv_data['epoch_time'],
            'samples_per_second': csv_data['samples_per_second'],
            'memory_usage_mb': csv_data['memory_usage_mb'],
            'gpu_memory_mb': csv_data['gpu_memory_mb'],
            'gradient_norm': csv_data['gradient_norm'],
            'post_clip_gradient_norm': csv_data['post_clip_gradient_norm'],
            'weight_stats': csv_data['weight_stats'],
            'gradient_stats': csv_data['gradient_stats'],
            'lr_stats': csv_data['lr_stats'],
            'best_val_loss': csv_data['best_val_loss'],
            'notes': csv_data['notes']
        }
        
        self.csv_logger.log_mini_epoch(**metrics_dict)



    def _prepare_csv_logging_data(self, epoch: int, mini_epoch: int, batch_times: List[float], 
                                val_metrics: Optional[Dict], gradient_norm: Optional[float],
                                post_clip_gradient_norm: Optional[float], weight_stats: Optional[Dict],
                                gradient_stats: Optional[Dict], lr_stats: Optional[Dict],
                                gpu_memory_mb: Optional[float], best_val_loss: Optional[float]) -> Dict:
        """Prepare data for CSV logging."""
        # Extract hyperparameters
        hp = self._get_hyperparameter_summary()
        
        epoch_id = f"{epoch+1}_mini{mini_epoch+1}" if epoch is not None and mini_epoch is not None else "unknown"
        mini_epoch_time = sum(batch_times)
        
        return {
            'epoch_id': epoch_id,
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
        # Check if we're in early training phase (first few batches of first mini-epoch of first epoch)
        is_early_training = (epoch == 1 and mini_epoch == 1 and batch_idx < NUMERICAL_STABILITY_WARMUP_BATCHES)
        
        # Calculate timing metrics
        timing = TrainingUtilities.calculate_batch_timing(state)
        
        # Move to device
        boards, policies, values, move_stage = TrainingUtilities.move_batch_to_device(boards, policies, values, move_stage, self.device)
        
        # Forward pass with mixed precision
        self.optimizer.zero_grad()
        with self.mixed_precision.autocast_context():
            policy_pred, value_pred = self.model(boards, move_stage)
            total_loss, loss_dict = self.criterion(policy_pred, value_pred, policies, values, boards)
        
        # CRITICAL: Check for NaN and extreme values in model outputs
        if torch.isnan(policy_pred).any() or torch.isnan(value_pred).any():
            # Get diagnostic information
            policy_range = f"[{policy_pred.min().item():.6f}, {policy_pred.max().item():.6f}]"
            value_range = f"[{value_pred.min().item():.6f}, {value_pred.max().item():.6f}]"
            move_stage_range = f"[{move_stage.min().item():.6f}, {move_stage.max().item():.6f}]"
            
            raise RuntimeError(
                f"NaN detected in model outputs! "
                f"policy_pred has NaN: {torch.isnan(policy_pred).any()}, range: {policy_range}, "
                f"value_pred has NaN: {torch.isnan(value_pred).any()}, range: {value_range}, "
                f"move_stage range: {move_stage_range}. "
                f"This indicates numerical instability in the model forward pass. "
                f"Check learning rate, gradient clipping, and model architecture."
            )
        
        # Check for extreme values that indicate numerical instability
        policy_max_abs = torch.abs(policy_pred).max().item()
        value_max_abs = torch.abs(value_pred).max().item()
        
        # TEMPORARY: Log values during early training for diagnostic purposes
        if is_early_training:
            logger.info(f"EARLY_TRAINING_DEBUG: Epoch {epoch}, Mini-epoch {mini_epoch}, Batch {batch_idx}: "
                       f"policy_max_abs={policy_max_abs:.6f}, value_max_abs={value_max_abs:.6f}, "
                       f"policy_range=[{policy_pred.min().item():.3f}, {policy_pred.max().item():.3f}], "
                       f"value_range=[{value_pred.min().item():.3f}, {value_pred.max().item():.3f}]")
        elif epoch == 1 and mini_epoch == 1 and batch_idx < 30:  # Log first 10 batches after warmup
            logger.info(f"POST_WARMUP_DEBUG: Epoch {epoch}, Mini-epoch {mini_epoch}, Batch {batch_idx}: "
                       f"policy_max_abs={policy_max_abs:.6f}, value_max_abs={value_max_abs:.6f}, "
                       f"policy_range=[{policy_pred.min().item():.3f}, {policy_pred.max().item():.3f}], "
                       f"value_range=[{value_pred.min().item():.3f}, {value_pred.max().item():.3f}]")
        
        # Enhanced debugging for extreme values
        if policy_max_abs > 30.0 or value_max_abs > 0.8:  # Lower threshold for more debugging
            # Get gradient norm info if available
            grad_norm_info = ""
            if hasattr(self, 'gradient_clipping_debug') and self.gradient_clipping_debug:
                latest_grad = self.gradient_clipping_debug[-1]
                grad_norm_info = f", pre_clip_grad_norm={latest_grad['pre_clip']:.3f}, post_clip_grad_norm={latest_grad['post_clip']:.3f}"
            
            logger.warning(f"LARGE_VALUES_DEBUG: Epoch {epoch}, Mini-epoch {mini_epoch}, Batch {batch_idx}: "
                          f"policy_max_abs={policy_max_abs:.6f}, value_max_abs={value_max_abs:.6f}, "
                          f"policy_range=[{policy_pred.min().item():.3f}, {policy_pred.max().item():.3f}], "
                          f"value_range=[{value_pred.min().item():.3f}, {value_pred.max().item():.3f}]{grad_norm_info}")
        
        # Check for extreme values that indicate numerical instability (skip during early training)
        if (not is_early_training and 
            (policy_max_abs > MAX_POLICY_LOGIT_ABS or value_max_abs > MAX_VALUE_OUTPUT_ABS)):
            policy_range = f"[{policy_pred.min().item():.6f}, {policy_pred.max().item():.6f}]"
            value_range = f"[{value_pred.min().item():.6f}, {value_pred.max().item():.6f}]"
            
            # Enhanced debug information
            logger.error(f"EXTREME_VALUES_DETECTED: Epoch {epoch}, Mini-epoch {mini_epoch}, Batch {batch_idx}")
            logger.error(f"Policy stats: max_abs={policy_max_abs:.6f}, mean={policy_pred.mean().item():.6f}, "
                        f"std={policy_pred.std().item():.6f}, range={policy_range}")
            logger.error(f"Value stats: max_abs={value_max_abs:.6f}, mean={value_pred.mean().item():.6f}, "
                        f"std={value_pred.std().item():.6f}, range={value_range}")
            
            # Save debug information to errors directory
            self._save_debug_info(epoch, mini_epoch, batch_idx, policy_pred, value_pred, 
                                policy_max_abs, value_max_abs, "extreme_model_outputs")
            
            raise RuntimeError(
                f"Extreme values detected in model outputs! "
                f"policy_pred max_abs: {policy_max_abs:.6f}, range: {policy_range}, "
                f"value_pred max_abs: {value_max_abs:.6f}, range: {value_range}. "
                f"Policy logits > {MAX_POLICY_LOGIT_ABS} indicate loss of uncertainty (softmax saturation). "
                f"Value outputs > {MAX_VALUE_OUTPUT_ABS} indicate tanh saturation (gradient vanishing). "
                f"Check learning rate, gradient clipping, and model architecture. "
                f"Debug info saved to checkpoints/bookkeeping/errors/"
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
                
                if grad_norm > MAX_GRADIENT_NORM:  # Extreme gradient norm
                    has_extreme_grad = True
                    print(f"WARNING: Extreme gradient norm {grad_norm:.6f} in parameter {name}")
        
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

    def _save_debug_info(self, epoch: int, mini_epoch: int, batch_idx: int, 
                        policy_pred: torch.Tensor, value_pred: torch.Tensor,
                        policy_max_abs: float, value_max_abs: float, error_type: str):
        """Save debug information to errors directory for analysis."""
        try:
            import os
            from pathlib import Path
            import json
            from datetime import datetime
            
            # Create errors directory if it doesn't exist
            errors_dir = Path("checkpoints/bookkeeping/errors")
            errors_dir.mkdir(parents=True, exist_ok=True)
            
            # Create debug filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_file = errors_dir / f"debug_{error_type}_{timestamp}.json"
            
            # Collect debug information
            debug_info = {
                "timestamp": timestamp,
                "error_type": error_type,
                "epoch": epoch,
                "mini_epoch": mini_epoch,
                "batch_idx": batch_idx,
                "policy_max_abs": policy_max_abs,
                "value_max_abs": value_max_abs,
                "policy_range": [policy_pred.min().item(), policy_pred.max().item()],
                "value_range": [value_pred.min().item(), value_pred.max().item()],
                "policy_mean": policy_pred.mean().item(),
                "value_mean": value_pred.mean().item(),
                "policy_std": policy_pred.std().item(),
                "value_std": value_pred.std().item(),
                "thresholds": {
                    "MAX_POLICY_LOGIT_ABS": MAX_POLICY_LOGIT_ABS,
                    "MAX_VALUE_OUTPUT_ABS": MAX_VALUE_OUTPUT_ABS
                }
            }
            
            # Save to JSON file
            with open(debug_file, 'w') as f:
                json.dump(debug_info, f, indent=2)
                
            logger.info(f"Debug info saved to {debug_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save debug info: {e}")

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
