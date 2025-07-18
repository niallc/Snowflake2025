"""
Training utilities for Hex AI models.

This module provides reusable utilities for:
- Loading and processing the new data format
- Creating train/validation splits
- Managing hyperparameter experiments
- Data loading with variable shard sizes
- Experiment tracking and results management
"""

import torch
import torch.nn as nn
import numpy as np
import gzip
import pickle
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
import random

from .models import TwoHeadedResNet
from .config import BOARD_SIZE, POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE
from hex_ai.data_utils import get_player_to_move_from_board
from hex_ai.error_handling import check_data_loading_errors

logger = logging.getLogger(__name__)


def _validate_example_format(example, filename):
    """Validate example format early to fail fast."""
    if not isinstance(example, tuple):
        raise ValueError(f"Example in {filename} must be tuple, got {type(example)}")
    if len(example) != 3:
        raise ValueError(f"Example in {filename} must have 3 elements, got {len(example)}")
    
    board_state, policy_target, value_target = example
    
    # Validate board state
    if not isinstance(board_state, np.ndarray):
        raise ValueError(f"Board state in {filename} must be numpy array, got {type(board_state)}")
    # Accept both (BOARD_SIZE, BOARD_SIZE) and (2, BOARD_SIZE, BOARD_SIZE) formats
    if board_state.shape not in [(BOARD_SIZE, BOARD_SIZE), (2, BOARD_SIZE, BOARD_SIZE)]:
        raise ValueError(f"Board state in {filename} shape must be ({BOARD_SIZE}, {BOARD_SIZE}) or (2, {BOARD_SIZE}, {BOARD_SIZE}), got {board_state.shape}")
    
    # Validate policy target (can be None for final moves)
    if policy_target is not None and not isinstance(policy_target, np.ndarray):
        raise ValueError(f"Policy target in {filename} must be numpy array or None, got {type(policy_target)}")
    if policy_target is not None and policy_target.shape != (POLICY_OUTPUT_SIZE,):
        raise ValueError(f"Policy target in {filename} shape must be ({POLICY_OUTPUT_SIZE},), got {policy_target.shape}")
    
    # Validate value target
    if not isinstance(value_target, (int, float, np.number)):
        raise ValueError(f"Value target in {filename} must be numeric, got {type(value_target)}")


class StreamingProcessedDataset(torch.utils.data.Dataset):
    """Streaming dataset that loads data in chunks to reduce memory usage."""
    
    def __init__(self, data_files: List[Path], chunk_size: int = 100000, shuffle_files: bool = True, max_examples: Optional[int] = None):
        """
        Initialize streaming dataset.
        
        Args:
            data_files: List of paths to processed data files
            chunk_size: Number of examples to load at once
            shuffle_files: Whether to shuffle the order of files
            max_examples: Maximum number of examples to provide (None for unlimited)
        """
        self.data_files = data_files
        if shuffle_files:
            random.shuffle(self.data_files)
        
        self.chunk_size = chunk_size
        self.max_examples = max_examples
        self.current_chunk = []
        self.current_file_idx = 0
        self.current_example_idx = 0
        self.total_examples_loaded = 0
        
        # Load first chunk
        self._load_next_chunk()
    
    def _load_next_chunk(self):
        """Load the next chunk of examples from files."""
        self.current_chunk = []
        files_attempted = 0
        files_with_errors = 0
        error_details = []
        while len(self.current_chunk) < self.chunk_size and self.current_file_idx < len(self.data_files):
            file_path = self.data_files[self.current_file_idx]
            files_attempted += 1
            try:
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
                if 'examples' in data:
                    file_examples = data['examples']
                    remaining_in_chunk = self.chunk_size - len(self.current_chunk)
                    examples_to_add = file_examples[:remaining_in_chunk]
                    self.current_chunk.extend(examples_to_add)
                    self.current_file_idx += 1
                else:
                    files_with_errors += 1
                    error_details.append((str(file_path), "Missing 'examples' key"))
                    self.current_file_idx += 1
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
                files_with_errors += 1
                error_details.append((str(file_path), str(e)))
                self.current_file_idx += 1
                continue
        # Shuffle the chunk for better randomization
        if len(self.current_chunk) > 0:
            random.shuffle(self.current_chunk)
        
        # After loading, check error thresholds
        if files_attempted > 0:
            error_log_dir = str(self.data_files[0].parent) if self.data_files else "."
            check_data_loading_errors(files_attempted, files_with_errors, error_details, error_log_dir)
        # Compact logging for verbose level 2
        import logging
        if logging.getLogger().getEffectiveLevel() <= logging.INFO:
            print(".", end="", flush=True)
            # Add newline every 50 chunks for readability
            if (self.total_examples_loaded + len(self.current_chunk)) % (50 * self.chunk_size) == 0:
                print()  # Newline
    
    def __len__(self):
        """
        Return the number of samples (for DataLoader compatibility).
        If max_examples is set, returns that value. Otherwise returns an estimate.
        """
        if self.max_examples is not None:
            return self.max_examples
        else:
            # Return an estimate based on the number of valid files and chunk size
            num_valid_files = len([f for f in self.data_files if f.exists()])
            return num_valid_files * self.chunk_size
    
    def __getitem__(self, idx):
        """Get a single training sample."""
        # Check if we've reached the maximum number of examples
        if self.max_examples is not None and self.total_examples_loaded >= self.max_examples:
            # Instead of raising an error, cycle back to the beginning
            # This allows training to continue with the limited dataset
            self.current_file_idx = 0
            self.current_example_idx = 0
            self.total_examples_loaded = 0
            random.shuffle(self.data_files)  # Reshuffle for next epoch
            self._load_next_chunk()
            self.current_example_idx = 0
        
        if self.current_example_idx >= len(self.current_chunk):
            if self.current_file_idx >= len(self.data_files):
                # We've exhausted all files, start over
                self.current_file_idx = 0
                self.current_example_idx = 0
                random.shuffle(self.data_files)  # Reshuffle for next epoch
            self._load_next_chunk()
            self.current_example_idx = 0
        if len(self.current_chunk) == 0:
            raise IndexError("No more data available")
        sample = self.current_chunk[self.current_example_idx]
        self.current_example_idx += 1
        self.total_examples_loaded += 1
        
        # Get error tracker for board state validation
        from hex_ai.error_handling import get_board_state_error_tracker
        error_tracker = get_board_state_error_tracker()
        
        board_state = torch.FloatTensor(sample[0])
        # Add player-to-move channel
        board_np = board_state.numpy()
        from hex_ai.inference.board_utils import BLUE_PLAYER, RED_PLAYER
        
        # Get current file info for error tracking
        current_file = self.data_files[self.current_file_idx - 1] if self.current_file_idx > 0 else "unknown"
        sample_info = f"chunk_idx={self.current_example_idx-1}, total_loaded={self.total_examples_loaded}"
        
        # Set context for error tracking
        error_tracker._current_file = str(current_file)
        error_tracker._current_sample = sample_info
        
        try:
            player_to_move = get_player_to_move_from_board(board_np, error_tracker)
        except Exception as e:
            # If get_player_to_move_from_board still raises an exception (shouldn't happen with error tracker)
            error_tracker.record_error(
                board_state=board_np,
                error_msg=str(e),
                file_info=str(current_file),
                sample_info=sample_info,
                raw_sample=sample,
                file_path=str(current_file)
            )
            # Use default value
            player_to_move = BLUE_PLAYER
        
        # Record successful processing
        error_tracker.record_success()
        
        player_channel = np.full((board_np.shape[1], board_np.shape[2]), float(player_to_move), dtype=np.float32)
        board_3ch = np.concatenate([board_np, player_channel[None, ...]], axis=0)
        board_state = torch.from_numpy(board_3ch)
        policy_target = torch.FloatTensor(sample[1]) if sample[1] is not None else torch.zeros(POLICY_OUTPUT_SIZE, dtype=torch.float32)
        value_target = torch.FloatTensor([sample[2]])
        return board_state, policy_target, value_target


class StreamingAugmentedProcessedDataset(StreamingProcessedDataset):
    """Streaming dataset that applies data augmentation to create 4x more training examples."""
    
    def __init__(self, data_files: List[Path], enable_augmentation: bool = True, **kwargs):
        """
        Initialize streaming augmented dataset.
        
        Args:
            data_files: List of paths to processed data files
            enable_augmentation: Whether to apply augmentation (default: True)
            **kwargs: Additional arguments passed to StreamingProcessedDataset
        """
        super().__init__(data_files, **kwargs)
        self.enable_augmentation = enable_augmentation
        
        # For augmented datasets, we need to adjust the max_examples limit
        # since each __getitem__ call returns 4 examples instead of 1
        if enable_augmentation and self.max_examples is not None:
            # Store the original limit for internal tracking
            self.original_max_examples = self.max_examples
            # Adjust the limit for the base class (divide by 4 since we return 4x examples)
            self.max_examples = self.max_examples // 4
            logger.info(f"StreamingAugmentedProcessedDataset: Adjusted max_examples from {self.original_max_examples:,} to {self.max_examples:,} (will provide {self.original_max_examples:,} effective examples with augmentation)")
        else:
            self.original_max_examples = self.max_examples
        
        if enable_augmentation:
            logger.info(f"StreamingAugmentedProcessedDataset: Will create 4x training examples through augmentation")
        else:
            logger.info(f"StreamingAugmentedProcessedDataset: Augmentation disabled, using original examples")
    
    def __len__(self):
        """
        Return the number of samples (for DataLoader compatibility).
        For augmented datasets, this should return the effective number of examples
        that will be provided (4x the base examples).
        """
        if self.enable_augmentation and self.original_max_examples is not None:
            # Return the original max_examples (the effective number with augmentation)
            return self.original_max_examples
        else:
            base_len = super().__len__()
            if self.enable_augmentation:
                # Return the effective number of examples (4x the base examples)
                return base_len * 4
            else:
                return base_len
    
    def __getitem__(self, idx):
        """Get augmented training examples."""
        if not self.enable_augmentation:
            return super().__getitem__(idx)
        
        # Get original example from streaming dataset
        original_example = super().__getitem__(idx)
        board_3ch, policy, value = original_example
        
        # Extract 2-channel board for augmentation
        board_2ch = board_3ch[:2].numpy()  # Remove player-to-move channel
        
        # Skip empty boards (no pieces to augment)
        if np.sum(board_2ch) == 0:
            return [original_example]  # Return single example for empty boards
        
        # Create all 4 augmented examples
        from hex_ai.data_utils import create_augmented_example_with_player_to_move
        try:
            # Get error tracker for board state validation
            from hex_ai.error_handling import get_board_state_error_tracker
            error_tracker = get_board_state_error_tracker()
            
            # Get current file info for error tracking
            current_file = self.data_files[self.current_file_idx - 1] if self.current_file_idx > 0 else "unknown"
            sample_info = f"augmented_chunk_idx={self.current_example_idx-1}, total_loaded={self.total_examples_loaded}"
            
            # Set context for error tracking
            error_tracker._current_file = str(current_file)
            error_tracker._current_sample = sample_info
            
            augmented_examples = create_augmented_example_with_player_to_move(board_2ch, policy.numpy(), value.item(), error_tracker)
        except Exception as e:
            logger.error(f"Error in create_augmented_example_with_player_to_move for idx {idx}: {e}")
            raise
        
        # Convert to tensors and create 3-channel boards
        tensor_examples = []
        for i, (aug_board_2ch, aug_policy, aug_value, aug_player) in enumerate(augmented_examples):
            try:
                # Create player-to-move channel
                player_channel = np.full((aug_board_2ch.shape[1], aug_board_2ch.shape[2]), float(aug_player), dtype=np.float32)
                board_3ch = np.concatenate([aug_board_2ch, player_channel[None, ...]], axis=0)
                
                # Convert to tensors
                board_tensor = torch.from_numpy(board_3ch)
                policy_tensor = torch.FloatTensor(aug_policy)
                value_tensor = torch.FloatTensor([aug_value])
                
                tensor_examples.append((board_tensor, policy_tensor, value_tensor))
            except Exception as e:
                logger.error(f"Error processing augmentation {i} for idx {idx}: {e}")
                raise
        
        # Return all 4 examples (DataLoader will handle batching)
        return tensor_examples





def discover_processed_files(data_dir: str = "data/processed") -> List[Path]:
    """
    Discover all processed data files in the specified directory.
    
    Args:
        data_dir: Directory containing processed data files
        
    Returns:
        List of paths to processed data files
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found")
    
    # Find all .pkl.gz files
    data_files = list(data_path.glob("*.pkl.gz"))
    
    if not data_files:
        raise FileNotFoundError(f"No processed data files found in {data_dir}")
    
    logger.info(f"Found {len(data_files)} processed data files")
    return data_files


def create_train_val_split(data_files: List[Path], 
                          train_ratio: float = 0.8,
                          random_seed: Optional[int] = None,
                          max_files_per_split: Optional[int] = None) -> Tuple[List[Path], List[Path]]:
    """
    Create train/validation split of data files.
    
    Args:
        data_files: List of all data files
        train_ratio: Ratio of files to use for training (0.0 to 1.0)
        random_seed: Random seed for reproducible splits
        max_files_per_split: Maximum number of files per split (for efficiency)
        
    Returns:
        Tuple of (train_files, val_files)
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Shuffle files for random split
    shuffled_files = data_files.copy()
    random.shuffle(shuffled_files)
    
    # Split files
    split_idx = int(len(shuffled_files) * train_ratio)
    train_files = shuffled_files[:split_idx]
    val_files = shuffled_files[split_idx:]
    
    # Limit number of files if specified (for efficiency)
    if max_files_per_split is not None:
        train_files = train_files[:max_files_per_split]
        val_files = val_files[:max_files_per_split]
        logger.info(f"Limited to {max_files_per_split} files per split for efficiency")
    
    logger.info(f"Split {len(data_files)} files: {len(train_files)} train, {len(val_files)} validation")
    
    return train_files, val_files


def estimate_dataset_size(data_files: List[Path], max_files: Optional[int] = None) -> int:
    """
    Estimate the total number of training examples across all files.
    
    Args:
        data_files: List of data files to analyze
        max_files: Maximum number of files to check (None for all files)
        
    Returns:
        Estimated total number of examples
    """
    total_examples = 0
    files_checked = 0
    
    # Limit the number of files to check for speed
    files_to_check = data_files[:max_files] if max_files else data_files
    
    for file_path in files_to_check:
        try:
            with gzip.open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if 'examples' in data:
                total_examples += len(data['examples'])
            elif 'processing_stats' in data:
                # Fallback to processing stats if available
                stats = data['processing_stats']
                if 'examples_generated' in stats:
                    total_examples += stats['examples_generated']
            
            files_checked += 1
                    
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            continue
    
    # If we only checked a subset, estimate the total
    if max_files and files_checked < len(data_files):
        estimated_total = int(total_examples * len(data_files) / files_checked)
        logger.info(f"Estimated {estimated_total:,} total examples from {files_checked} sample files")
        return estimated_total
    
    return total_examples
