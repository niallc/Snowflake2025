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
from time import sleep
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
    
    def __init__(self, data_files: List[Path], enable_augmentation: bool = True, chunk_size: int = 1024, **kwargs):
        """
        Initialize streaming augmented dataset.
        
        Args:
            data_files: List of paths to processed data files
            enable_augmentation: Whether to apply augmentation (default: True)
            chunk_size: Number of examples to load per chunk
            **kwargs: Additional arguments (max_examples, shuffle_files)
        """
        super().__init__()
        self.data_files = data_files
        self.enable_augmentation = enable_augmentation
        self.chunk_size = chunk_size
        self.max_examples = kwargs.get('max_examples', None)
        self.shuffle_files = kwargs.get('shuffle_files', False)
        # Effective number of examples after augmentation (for reporting and __len__)
        if self.enable_augmentation and self.max_examples is not None:
            self.effective_max_examples = self.max_examples * 4
        else:
            self.effective_max_examples = self.max_examples
        if enable_augmentation:
            logger.info(f"StreamingAugmentedProcessedDataset: Will create 4x training examples through augmentation (max_examples={self.max_examples}, effective={self.effective_max_examples})")
        else:
            logger.info(f"StreamingAugmentedProcessedDataset: Augmentation disabled, using original examples (max_examples={self.max_examples})")
        # Initialize epoch state
        self.epoch_file_list = self._get_shuffled_file_list()
        self.current_file_idx = 0  # Index in epoch_file_list
        self.current_chunk = []    # List of raw examples in current chunk
        self.current_chunk_idx = 0 # Index within current chunk
        self.total_examples_loaded = 0  # Across epoch
        self._load_next_chunk()    # Load first chunk

    def __len__(self):
        """
        Return the number of samples (for DataLoader compatibility).
        For augmented datasets, this should return the effective number of examples
        that will be provided (4x the base examples).
        """
        if self.effective_max_examples is not None:
            return self.effective_max_examples
        else:
            # If not specified, estimate from files (not recommended for streaming)
            # TODO: Optionally scan files to estimate total length
            raise NotImplementedError("Length is undefined when max_examples is not set.")

    def _get_shuffled_file_list(self):
        """Return a new shuffled list of files for the epoch."""
        file_list = self.data_files.copy()
        if self.shuffle_files:
            random.shuffle(file_list)
        return file_list

    def _start_new_epoch(self):
        """Reset counters and reshuffle files for a new epoch."""
        if self.enable_augmentation and self.max_examples is not None:
            print(f"Max samples ({self.max_examples}, effective {self.effective_max_examples} with augmentation) reached, starting next epoch")
        else:
            print(f"Max samples ({self.max_examples}) reached, starting next epoch")
        self.epoch_file_list = self._get_shuffled_file_list()
        self.current_file_idx = 0
        self.current_chunk = []
        self.current_chunk_idx = 0
        self.total_examples_loaded = 0
        self._load_next_chunk()
        self.current_chunk_idx = 0

    def __getitem__(self, idx):
        """
        Get augmented training examples for the given global index.
        Maps idx to the correct chunk and example, loading new chunks as needed.
        """
        # TODO: Support non-augmented mode for validation
        if not self.enable_augmentation:
            raise NotImplementedError("Non-augmented mode is not yet supported.")
        # --- Epoch boundary logic ---
        num_restarts = 0
        exceeded_max_samples = self.total_examples_loaded >= self.max_examples if self.max_examples is not None else False
        start_new_epoch = self.max_examples is not None and exceeded_max_samples
        while start_new_epoch:
            self._start_new_epoch()
            if num_restarts > 0:
                logger.info(f"Restarting epoch, take {num_restarts} / max=100. Sleeping for 1 second.")
                sleep(0.5)
            num_restarts += 1
            if num_restarts > 200:
                logger.warning("WARNING: Restarting epoch failed too many times. Continuing to avoid infinite loop.")
                break
            exceeded_max_samples = self.total_examples_loaded >= self.max_examples if self.max_examples is not None else False
            start_new_epoch = self.max_examples is not None and exceeded_max_samples
        # --- End epoch boundary logic ---
        # Map global idx to chunk and example
        # Each original example produces 4 augmented examples
        base_example_idx = idx // 4
        aug_idx = idx % 4
        # Find the chunk containing base_example_idx
        chunk_start = 0
        chunk_end = len(self.current_chunk)
        # If current chunk does not contain the desired example, load next chunk(s)
        while not (chunk_start <= base_example_idx < chunk_end):
            self._load_next_chunk()
            chunk_start += chunk_end
            chunk_end = chunk_start + len(self.current_chunk)
        # Index within current chunk
        local_idx = base_example_idx - chunk_start
        example = self.current_chunk[local_idx]
        # Validate and augment
        _validate_example_format(example, filename="<stream>")
        board_state = example['board']
        policy = example['policy']
        value = example['value']
        # Extract 2-channel board for augmentation
        board_2ch = board_state[:PLAYER_CHANNEL] if board_state.shape[0] > 1 else board_state
        # Skip empty boards (no pieces to augment)
        if np.sum(board_2ch) == 0:
            # Only one example for empty boards
            if aug_idx == 0:
                self.total_examples_loaded += 1
                return self._tensorize_example(board_state, policy, value)
            else:
                # For aug_idx > 0, return the first example again (or raise IndexError)
                raise IndexError("No augmented examples for empty board.")
        # Create all 4 augmented examples
        from hex_ai.data_utils import create_augmented_example_with_player_to_move
        try:
            from hex_ai.error_handling import get_board_state_error_tracker
            error_tracker = get_board_state_error_tracker()
            current_file = self.epoch_file_list[self.current_file_idx - 1] if self.current_file_idx > 0 else "unknown"
            sample_info = f"augmented_chunk_idx={local_idx}, total_loaded={self.total_examples_loaded}"
            error_tracker._current_file = str(current_file)
            error_tracker._current_sample = sample_info
            augmented_examples = create_augmented_example_with_player_to_move(board_2ch, policy, value, error_tracker)
        except Exception as e:
            logger.error(f"Error in create_augmented_example_with_player_to_move for idx {idx}: {e}")
            raise
        if aug_idx >= len(augmented_examples):
            raise IndexError(f"Requested augmentation {aug_idx} but only {len(augmented_examples)} available.")
        aug_board_2ch, aug_policy, aug_value, aug_player = augmented_examples[aug_idx]
        tensor_example = self._tensorize_example(aug_board_2ch, aug_policy, aug_value, aug_player)
        self.total_examples_loaded += 1
        return tensor_example

    def _tensorize_example(self, board_2ch, policy, value, player=None):
        """Convert numpy arrays to torch tensors, adding player channel if needed."""
        if player is not None:
            player_channel = np.full((board_2ch.shape[1], board_2ch.shape[2]), float(player), dtype=np.float32)
            board_3ch = np.concatenate([board_2ch, player_channel[None, ...]], axis=0)
        else:
            board_3ch = board_2ch
        board_tensor = torch.from_numpy(board_3ch).float()
        policy_tensor = torch.FloatTensor(policy)
        value_tensor = torch.FloatTensor([value])
        return (board_tensor, policy_tensor, value_tensor)

    def _load_next_chunk(self):
        """Load the next chunk of examples from files (no repeats within epoch)."""
        self.current_chunk = []
        files_attempted = 0
        files_with_errors = 0
        error_details = []
        # Only use files from the current epoch's shuffled list
        while len(self.current_chunk) < self.chunk_size and self.current_file_idx < len(self.epoch_file_list):
            file_path = self.epoch_file_list[self.current_file_idx]
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
        import logging
        if logging.getLogger().getEffectiveLevel() <= logging.INFO:
            print(".", end="", flush=True)
            if (self.total_examples_loaded + len(self.current_chunk)) % (50 * self.chunk_size) == 0:
                print()  # Newline



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
