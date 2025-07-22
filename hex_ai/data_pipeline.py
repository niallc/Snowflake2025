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
from .config import BOARD_SIZE, POLICY_OUTPUT_SIZE, PLAYER_CHANNEL
from hex_ai.data_utils import get_player_to_move_from_board, create_augmented_example_with_player_to_move
from hex_ai.error_handling import check_data_loading_errors, get_board_state_error_tracker

logger = logging.getLogger(__name__)

AUGMENTATION_FACTOR = 4  # Number of augmentations per unaugmented board (rotations/reflections)


class StreamingAugmentedProcessedDataset(torch.utils.data.Dataset):
    """Streaming dataset that applies data augmentation to create 4x more training examples.
    Refactored for clean chunked access. See write_ups/chunk_loading_in_torch_dataset.md for design.
    """
    def __init__(self, data_files: List[Path], enable_augmentation: bool = True, chunk_size: int = 100000, verbose: bool = False, **kwargs):
        super().__init__()
        self.data_files = data_files
        self.enable_augmentation = enable_augmentation
        self.chunk_size = chunk_size
        self.max_examples_unaugmented = kwargs.get('max_examples_unaugmented', None)
        self.shuffle_files = kwargs.get('shuffle_files', False)
        self.verbose = verbose
        self.augmentation_factor = AUGMENTATION_FACTOR if enable_augmentation else 1
        self.total_examples = None
        self.current_chunk = None
        self.current_chunk_number = None
        self.current_file_index = 0
        self.current_file = None
        self.current_file_examples = None
        self.current_file_example_index = 0
        self.logger = logging.getLogger(__name__)
        # Set policy shape to match real data format
        from hex_ai.config import BOARD_SIZE
        self.policy_shape = (BOARD_SIZE * BOARD_SIZE,)
        self.max_examples_augmented = (
            self.max_examples_unaugmented * self.augmentation_factor
            if self.max_examples_unaugmented is not None else None
        )
        self.epoch_file_list = self._get_shuffled_file_list()
        self.example_index_map = self._build_example_index_map()
        self._last_accessed_index = None
        self._random_access_warned = False
        if self.verbose:
            print(f"[INIT] Ready for chunked access. Files: {self.epoch_file_list}")

    def __len__(self):
        if self.max_examples_augmented is not None:
            return self.max_examples_augmented
        else:
            raise NotImplementedError("Length is undefined when max_examples_unaugmented is not set.")

    def __getitem__(self, idx):
        # Warn if random access is detected (not strictly sequential)
        if self._last_accessed_index is not None and idx != self._last_accessed_index + 1:
            if not self._random_access_warned:
                logger.warning("Random access is supported but not as thoroughly tested as sequential access. Please report any issues.")
                self._random_access_warned = True
        self._last_accessed_index = idx
        chunk_number, index_in_chunk = self._map_index(idx)
        if self.current_chunk_number != chunk_number:
            self.current_chunk = self._load_chunk(chunk_number)
            self.current_chunk_number = chunk_number
        return self.current_chunk[index_in_chunk]

    def _map_index(self, idx):
        chunk_number = idx // self.chunk_size
        index_in_chunk = idx % self.chunk_size
        return chunk_number, index_in_chunk

    def _get_shuffled_file_list(self):
        file_list = self.data_files.copy()
        if self.shuffle_files:
            random.shuffle(file_list)
        return file_list

    def _build_example_index_map(self):
        """Build a flat list of (file_idx, example_idx) for all unaugmented examples up to max_examples_unaugmented."""
        index_map = []
        total = 0
        for file_idx, file_path in enumerate(self.epoch_file_list):
            try:
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
                if 'examples' in data:
                    file_examples = data['examples']
                    for example_idx in range(len(file_examples)):
                        if self.max_examples_unaugmented is not None and total >= self.max_examples_unaugmented:
                            return index_map
                        index_map.append((file_idx, example_idx))
                        total += 1
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
                continue
        return index_map

    def _load_chunk(self, chunk_number):
        """Load the specified chunk (by chunk_number) as a list of tensorized examples."""
        start_idx = chunk_number * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.__len__())
        chunk_examples = []
        for global_idx in range(start_idx, end_idx):
            file_idx, example_idx = self.example_index_map[global_idx // self.augmentation_factor]
            file_path = self.epoch_file_list[file_idx]
            try:
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
                ex = data['examples'][example_idx]
                error_tracker = get_board_state_error_tracker()
                error_tracker._current_file = str(file_path)
                error_tracker._current_sample = f"file_idx={file_idx}, example_idx={example_idx}"
                # Use helper for augmentation/tensorization
                tensorized = self.get_augmented_tensor_for_index(
                    ex, global_idx % self.augmentation_factor, error_tracker)
                chunk_examples.append(tensorized)
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
                raise
        return chunk_examples

    def get_augmented_tensor_for_index(self, ex, aug_idx, error_tracker):
        """
        Given a raw example, augmentation index, and error tracker, return the tensorized (board, policy, value).
        Handles both augmented and non-augmented cases.
        """
        board = ex['board']
        policy = ex['policy']
        value = ex['value']
        board_2ch = board[:PLAYER_CHANNEL] if board.shape[0] > 1 else board
        if self.enable_augmentation:
            augmented_examples = create_augmented_example_with_player_to_move(
                board_2ch, policy, value, error_tracker)
            aug = augmented_examples[aug_idx]
            return self._transform_example(*aug)
        else:
            return self._transform_example(board_2ch, policy, value, None)

    def _normalize_policy(self, policy):
        """Return a zero vector if policy is None, else return as-is."""
        if policy is None:
            return np.zeros(self.policy_shape, dtype=np.float32)
        return policy

    def _transform_example(self, board_2ch, policy, value, player=None):
        # Apply tensorization and add player channel if needed
        if player is not None:
            player_channel = np.full((board_2ch.shape[1], board_2ch.shape[2]), float(player), dtype=np.float32)
            board_3ch = np.concatenate([board_2ch, player_channel[None, ...]], axis=0)
        else:
            board_3ch = board_2ch
        board_tensor = torch.from_numpy(board_3ch).float()
        # Normalize policy before tensorization
        policy = self._normalize_policy(policy)
        policy_tensor = torch.FloatTensor(policy)
        value_tensor = torch.FloatTensor([value])
        return board_tensor, policy_tensor, value_tensor


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
