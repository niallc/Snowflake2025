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
        import torch, time, os, psutil
        start_time = time.time()
        print("\n[StreamingAugmentedProcessedDataset.__init__] START")
        print(f"  Number of files: {len(data_files)} | enable_augmentation={enable_augmentation} | chunk_size={chunk_size}")
        print(f"  PID: {os.getpid()} | RAM used: {psutil.Process(os.getpid()).memory_info().rss / 1e6:.1f} MB")
        print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
        if hasattr(torch.backends, 'mps'):
            print(f"  torch.backends.mps.is_available(): {torch.backends.mps.is_available()}")
            print(f"  torch.backends.mps.is_built(): {torch.backends.mps.is_built()}")
        print(f"  torch.get_num_threads(): {torch.get_num_threads()}")
        print(f"  torch version: {torch.__version__}")
        super().__init__()
        self.data_files = data_files
        self.enable_augmentation = enable_augmentation
        self.chunk_size = chunk_size
        self.max_examples_unaugmented = kwargs.get('max_examples_unaugmented', None)
        self.shuffle_files = kwargs.get('shuffle_files', False)
        self.verbose = verbose
        self.augmentation_factor = AUGMENTATION_FACTOR if enable_augmentation else 1
        self.logger = logging.getLogger(__name__)
        self.policy_shape = (BOARD_SIZE * BOARD_SIZE,)
        self.epoch_file_list = self.data_files.copy()
        if self.shuffle_files:
            random.shuffle(self.epoch_file_list)
        self.file_example_counts = []
        total_examples = 0
        file_count = 0
        file_times = []
        print("  Scanning files for example counts...")
        for file_path in self.epoch_file_list:
            t0 = time.time()
            try:
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
                n_examples = len(data['examples']) if 'examples' in data else 0
            except Exception as e:
                print(f"    [WARN] Could not read {file_path}: {e}")
                n_examples = 0
            self.file_example_counts.append(n_examples)
            total_examples += n_examples
            file_count += 1
            t1 = time.time()
            file_times.append(t1-t0)
            print(f"    File {file_count:4d}/{len(self.epoch_file_list)}: {file_path.name:40s} | {n_examples:6d} examples | {t1-t0:.2f}s | RAM: {psutil.Process(os.getpid()).memory_info().rss / 1e6:.1f} MB")
            if file_count % 10 == 0 or file_count == len(self.epoch_file_list):
                elapsed = time.time() - start_time
                print(f"      ... Scanned {file_count} files, {total_examples} total examples so far, elapsed {elapsed:.1f}s")
        self.cumulative_counts = [0]
        for count in self.file_example_counts:
            self.cumulative_counts.append(self.cumulative_counts[-1] + count)
        self.total_unaugmented = sum(self.file_example_counts)
        if self.max_examples_unaugmented is not None:
            self.total_unaugmented = min(self.total_unaugmented, self.max_examples_unaugmented)
        self.max_examples_augmented = self.total_unaugmented * self.augmentation_factor
        self.current_chunk = None
        self.current_chunk_number = None
        self._last_accessed_index = None
        self._random_access_warned = False
        if self.verbose:
            print(f"[INIT] Ready for chunked access. Files: {self.epoch_file_list}")
        end_time = time.time()
        print(f"[StreamingAugmentedProcessedDataset.__init__] END | Total files: {file_count} | Total examples: {total_examples} | Init time: {end_time-start_time:.2f}s | RAM: {psutil.Process(os.getpid()).memory_info().rss / 1e6:.1f} MB\n")

    def __len__(self):
        return self.max_examples_augmented

    def __getitem__(self, idx):
        # Warn if random access is detected (not strictly sequential)
        if self._last_accessed_index is not None and idx != self._last_accessed_index + 1:
            if not self._random_access_warned:
                self.logger.warning("Random access is supported but not as thoroughly tested as sequential access. Please report any issues.")
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

    def _find_file_and_local_index(self, global_example_idx):
        # Binary search over cumulative_counts to find file
        left, right = 0, len(self.cumulative_counts) - 1
        while left < right:
            mid = (left + right) // 2
            if self.cumulative_counts[mid+1] <= global_example_idx:
                left = mid + 1
            else:
                right = mid
        file_idx = left
        local_idx = global_example_idx - self.cumulative_counts[file_idx]
        return file_idx, local_idx

    def _load_chunk(self, chunk_number):
        start_idx = chunk_number * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.__len__())
        chunk_examples = []
        # For each global example index in this chunk, group by file
        needed = []  # List of (file_idx, local_idx, aug_idx)
        for global_aug_idx in range(start_idx, end_idx):
            global_example_idx = global_aug_idx // self.augmentation_factor
            if global_example_idx >= self.total_unaugmented:
                break
            aug_idx = global_aug_idx % self.augmentation_factor
            file_idx, local_idx = self._find_file_and_local_index(global_example_idx)
            needed.append((file_idx, local_idx, aug_idx))
        # Group needed by file_idx
        from collections import defaultdict
        file_to_indices = defaultdict(list)
        for file_idx, local_idx, aug_idx in needed:
            file_to_indices[file_idx].append((local_idx, aug_idx))
        # Load each file only once, extract all needed examples
        for file_idx in sorted(file_to_indices.keys()):
            file_path = self.epoch_file_list[file_idx]
            with gzip.open(file_path, 'rb') as f:
                data = pickle.load(f)
            file_examples = data['examples']
            for local_idx, aug_idx in file_to_indices[file_idx]:
                ex = file_examples[local_idx]
                error_tracker = get_board_state_error_tracker()
                error_tracker._current_file = str(file_path)
                error_tracker._current_sample = f"file_idx={file_idx}, example_idx={local_idx}"
                tensorized = self.get_augmented_tensor_for_index(
                    ex, aug_idx, error_tracker)
                chunk_examples.append(tensorized)
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
