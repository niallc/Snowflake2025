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
    """Sequential dataset for pre-shuffled data. Loads as few files as needed to reach max_examples_unaugmented, and returns examples sequentially. Augmentation is applied on-the-fly if enabled."""
    def __init__(self, data_files: List[Path], enable_augmentation: bool = True, chunk_size: int = 100000, verbose: bool = False, **kwargs):
        super().__init__()
        self.data_files = data_files
        self.enable_augmentation = enable_augmentation
        self.max_examples_unaugmented = kwargs.get('max_examples_unaugmented', None)
        self.verbose = verbose
        self.augmentation_factor = AUGMENTATION_FACTOR if enable_augmentation else 1
        self.logger = logging.getLogger(__name__)
        self.policy_shape = (BOARD_SIZE * BOARD_SIZE,)
        self.examples = []
        total_loaded = 0
        file_count = 0
        self.logger.info(f"[StreamingAugmentedProcessedDataset] Initializing with {len(self.data_files)} files, enable_augmentation={self.enable_augmentation}, max_examples_unaugmented={self.max_examples_unaugmented}")
        for file_path in self.data_files:
            file_count += 1
            try:
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
                file_examples = data['examples'] if 'examples' in data else []
                n_to_add = len(file_examples)
                if self.max_examples_unaugmented is not None:
                    n_to_add = min(n_to_add, self.max_examples_unaugmented - total_loaded)
                self.examples.extend(file_examples[:n_to_add])
                total_loaded += n_to_add
                self.logger.info(f"  Loaded {n_to_add} examples from {file_path.name} (file {file_count}/{len(self.data_files)}), total loaded: {total_loaded}")
                if (
                    self.max_examples_unaugmented is not None
                    and total_loaded >= self.max_examples_unaugmented
                ):
                    self.logger.info(f"  Reached max_examples_unaugmented ({self.max_examples_unaugmented}), stopping file loading.")
                    break
            except Exception as e:
                self.logger.warning(f"Error loading {file_path}: {e}")
                continue
        self.total_unaugmented = len(self.examples)
        self.max_examples_augmented = self.total_unaugmented * self.augmentation_factor
        self.logger.info(f"[StreamingAugmentedProcessedDataset] Initialization complete: {self.total_unaugmented} examples loaded from {file_count} files (augmentation: {self.enable_augmentation})")
        if self.verbose:
            print(f"[INIT] Loaded {self.total_unaugmented} examples from {file_count} files (augmentation: {self.enable_augmentation})")

    def __len__(self):
        return self.max_examples_augmented

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.max_examples_augmented:
            raise IndexError(f"Index {idx} out of range for dataset of length {self.max_examples_augmented}")
        ex_idx = idx // self.augmentation_factor
        aug_idx = idx % self.augmentation_factor
        ex = self.examples[ex_idx]
        error_tracker = get_board_state_error_tracker()
        error_tracker._current_file = "sequential_in_memory"
        error_tracker._current_sample = f"example_idx={ex_idx}"
        return self.get_augmented_tensor_for_index(ex, aug_idx, error_tracker)

    def get_augmented_tensor_for_index(self, ex, aug_idx, error_tracker):
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
            # Always add player channel, even if augmentation is off
            player = get_player_to_move_from_board(board_2ch)
            return self._transform_example(board_2ch, policy, value, player)

    def _normalize_policy(self, policy):
        if policy is None:
            return np.zeros(self.policy_shape, dtype=np.float32)
        return policy

    def _transform_example(self, board_2ch, policy, value, player=None):
        if player is not None:
            player_channel = np.full((board_2ch.shape[1], board_2ch.shape[2]), float(player), dtype=np.float32)
            board_3ch = np.concatenate([board_2ch, player_channel[None, ...]], axis=0)
        else:
            board_3ch = board_2ch
        board_tensor = torch.from_numpy(board_3ch).float()
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
