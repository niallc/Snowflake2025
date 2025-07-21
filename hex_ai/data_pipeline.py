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
from .config import BOARD_SIZE, POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE, PLAYER_CHANNEL
from hex_ai.data_utils import get_player_to_move_from_board
from hex_ai.error_handling import check_data_loading_errors

logger = logging.getLogger(__name__)


def _validate_example_format(example, filename):
    """Validate example format early to fail fast."""
    if not isinstance(example, dict):
        raise ValueError(f"Example in {filename} must be dictionary, got {type(example)}")
    
    # Check for required keys
    if not all(key in example for key in ['board', 'policy', 'value']):
        raise ValueError(f"Example in {filename} missing required keys: {example.keys()}")
    
    board_state = example['board']
    policy_target = example['policy']
    value_target = example['value']
    
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


class StreamingAugmentedProcessedDataset(torch.utils.data.Dataset):
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
        self.max_examples = kwargs.get('max_examples', None)
        # Effective number of examples after augmentation (for reporting and __len__)
        if self.enable_augmentation and self.max_examples is not None:
            self.effective_max_examples = self.max_examples * 4
        else:
            self.effective_max_examples = self.max_examples
        if enable_augmentation:
            logger.info(f"StreamingAugmentedProcessedDataset: Will create 4x training examples through augmentation (max_examples={self.max_examples}, effective={self.effective_max_examples})")
        else:
            logger.info(f"StreamingAugmentedProcessedDataset: Augmentation disabled, using original examples (max_examples={self.max_examples})")

    def __len__(self):
        """
        Return the number of samples (for DataLoader compatibility).
        For augmented datasets, this should return the effective number of examples
        that will be provided (4x the base examples).
        """
        if self.effective_max_examples is not None:
            return self.effective_max_examples
        else:
            base_len = super().__len__()
            if self.enable_augmentation:
                return base_len * 4
            else:
                return base_len

    def _start_new_epoch(self):
        """Reset counters and reshuffle files for a new epoch."""
        if self.enable_augmentation and self.max_examples is not None:
            print(f"Max samples ({self.max_examples}, effective {self.effective_max_examples} with augmentation) reached, starting next epoch")
        else:
            print(f"Max samples ({self.max_examples}) reached, starting next epoch")
        self.epoch_file_list = self._get_shuffled_file_list()
        self.current_file_idx = 0
        self.current_example_idx = 0
        self.total_examples_loaded = 0
        self._load_next_chunk()
        self.current_example_idx = 0

    def __getitem__(self, idx):
        """Get augmented training examples."""
        if not self.enable_augmentation:
            return super().__getitem__(idx)
        # The logic for when to start a new epoch is based on pre-augmentation max_examples
        if self.max_examples is not None and self.total_examples_loaded >= self.max_examples:
            self._start_new_epoch()
        # Get original example from streaming dataset
        original_example = super().__getitem__(idx)
        board_3ch, policy, value = original_example
        # Extract 2-channel board for augmentation
        board_2ch = board_3ch[:PLAYER_CHANNEL].numpy()  # Remove player-to-move channel
        # Skip empty boards (no pieces to augment)
        if np.sum(board_2ch) == 0:
            return [original_example]  # Return single example for empty boards
        # Create all 4 augmented examples
        from hex_ai.data_utils import create_augmented_example_with_player_to_move
        try:
            from hex_ai.error_handling import get_board_state_error_tracker
            error_tracker = get_board_state_error_tracker()
            current_file = self.data_files[self.current_file_idx - 1] if self.current_file_idx > 0 else "unknown"
            sample_info = f"augmented_chunk_idx={self.current_example_idx-1}, total_loaded={self.total_examples_loaded}"
            error_tracker._current_file = str(current_file)
            error_tracker._current_sample = sample_info
            augmented_examples = create_augmented_example_with_player_to_move(board_2ch, policy.numpy(), value.item(), error_tracker)
        except Exception as e:
            logger.error(f"Error in create_augmented_example_with_player_to_move for idx {idx}: {e}")
            raise
        tensor_examples = []
        for i, (aug_board_2ch, aug_policy, aug_value, aug_player) in enumerate(augmented_examples):
            try:
                player_channel = np.full((aug_board_2ch.shape[1], aug_board_2ch.shape[2]), float(aug_player), dtype=np.float32)
                board_3ch = np.concatenate([aug_board_2ch, player_channel[None, ...]], axis=0)
                board_tensor = torch.from_numpy(board_3ch)
                policy_tensor = torch.FloatTensor(aug_policy)
                value_tensor = torch.FloatTensor([aug_value])
                tensor_examples.append((board_tensor, policy_tensor, value_tensor))
            except Exception as e:
                logger.error(f"Error processing augmentation {i} for idx {idx}: {e}")
                raise
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
    
    # Check if this is shuffled data directory
    if (data_path / "shuffling_progress.json").exists():
        # Shuffled data: look for shuffled_*.pkl.gz files
        data_files = list(data_path.glob("shuffled_*.pkl.gz"))
        logger.info(f"Found {len(data_files)} shuffled data files")
    else:
        # Original processed data: look for *_processed.pkl.gz files
        data_files = list(data_path.glob("*_processed.pkl.gz"))
        logger.info(f"Found {len(data_files)} processed data files")
    
    if not data_files:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    
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
