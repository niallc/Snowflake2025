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
import random
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import sleep
import psutil
from .models import TwoHeadedResNet
from .config import BOARD_SIZE, POLICY_OUTPUT_SIZE, PLAYER_CHANNEL, DEFAULT_POOL_SIZE, DEFAULT_REFILL_THRESHOLD, DEFAULT_MAX_MEMORY_GB, VALIDATION_DATA_COMPRESSION_RATIO, MAX_TEMP_MEMORY_GB
from hex_ai.data_utils import get_player_to_move_from_board, create_augmented_example_with_player_to_move
from hex_ai.error_handling import check_data_loading_errors, get_board_state_error_tracker

logger = logging.getLogger(__name__)

AUGMENTATION_FACTOR = 4  # Number of augmentations per unaugmented board (rotations/reflections)
# TODO: Refine ths as I doubt the actual validation gets nearly this big
MAX_VALIDATION_MEMORY_GB = 9.0


def shuffle_data_files(data_files: List[Path], shuffle_shards: bool = True, random_seed: Optional[int] = None) -> List[Path]:
    """
    Utility function to shuffle data files consistently.
    
    Args:
        data_files: List of data file paths
        shuffle_shards: Whether to shuffle the shards
        random_seed: Random seed for reproducible shuffling
        
    Returns:
        List of data file paths (shuffled if requested)
    """
    if not shuffle_shards:
        return data_files
    
    if random_seed is not None:
        random.seed(random_seed)
    
    shuffled_files = data_files.copy()
    random.shuffle(shuffled_files)
    return shuffled_files


class ShardLogger:
    """
    Tracks and logs data shard transitions during training.
    """
    def __init__(self, log_shard_transitions: bool = True):
        self.log_shard_transitions = log_shard_transitions
        self.current_shard = None
        self.shard_transitions = []
        self.logger = logging.getLogger(__name__)
    
    def log_shard_start(self, file_path: Path, file_idx: int, total_shards: int, approx_batch_count: int = None):
        """Log when a new shard starts being processed."""
        if not self.log_shard_transitions:
            return
        
        # Get relative path from project root
        try:
            # Try to get relative path from current working directory (project root)
            relative_path = file_path.relative_to(Path.cwd())
            # Extract just the directory part (parent of the file)
            relative_dir = relative_path.parent
            dir_info = f" from {relative_dir}"
        except ValueError:
            # If we can't get relative path, fall back to absolute path
            dir_info = f" from {file_path.parent}"
        
        shard_info = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_idx': file_idx,
            'total_shards': total_shards,
            'approx_batch_count': approx_batch_count,
            'timestamp': None  # Could add timestamp if needed
        }
        
        self.current_shard = shard_info
        self.shard_transitions.append(shard_info)
        
        # Log to console and file
        log_msg = f"[SHARD_START] Processing shard {file_idx+1}/{total_shards}: {file_path.name}{dir_info}"
        if approx_batch_count is not None:
            log_msg += f" (batch {approx_batch_count})"
        
        self.logger.info(log_msg)
    
    def get_current_shard_info(self):
        """Get information about the currently active shard."""
        return self.current_shard
    
    def get_shard_transitions(self):
        """Get all recorded shard transitions."""
        return self.shard_transitions


class StreamingMixedShardDataset(torch.utils.data.IterableDataset):
    """
    Streaming dataset that maintains a mixed pool of positions from multiple shards.
    Loads shards proportionally from multiple directories and maintains a large in-memory pool
    to eliminate blockwise learning. Designed for use with torch DataLoader.

    Args:
        data_dirs: List of data directories to load from
        shard_ranges: List of shard ranges for each directory (e.g., ["251-300", "all"])
        pool_size: Target number of positions to maintain in memory (default: 1M)
        refill_threshold: Refill pool when it drops below this many positions (default: 750K)
        max_memory_gb: Maximum memory usage before graceful shutdown (default: 5.0)
        enable_augmentation: Whether to apply augmentation on-the-fly
        max_examples_unaugmented: Stop after yielding this many (unaugmented) examples
        verbose: Verbose level (2=default, 3=detailed pool/shard info)
        random_seed: Random seed for reproducible behavior
        is_validation: Whether this is a validation dataset (enables special validation behavior)
    """
    
    def __init__(self,
                 data_dirs: List[str],
                 shard_ranges: List[str],
                 pool_size: int = DEFAULT_POOL_SIZE,
                 refill_threshold: int = DEFAULT_REFILL_THRESHOLD,
                 max_memory_gb: float = DEFAULT_MAX_MEMORY_GB,
                 enable_augmentation: bool = True,
                 max_examples_unaugmented: Optional[int] = None,
                 verbose: int = 2,
                 random_seed: Optional[int] = None,
                 is_validation: bool = False):
        super().__init__()
        
        # Validate inputs
        if len(data_dirs) != len(shard_ranges):
            raise ValueError(f"Number of data_dirs ({len(data_dirs)}) must match number of shard_ranges ({len(shard_ranges)})")
        
        if pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {pool_size}")
        
        if refill_threshold >= pool_size:
            raise ValueError(f"refill_threshold ({refill_threshold}) must be less than pool_size ({pool_size})")
        
        if max_memory_gb <= 0:
            raise ValueError(f"max_memory_gb must be positive, got {max_memory_gb}")
        
        # Store configuration
        self.data_dirs = data_dirs
        self.shard_ranges = shard_ranges
        self.pool_size = pool_size
        self.refill_threshold = refill_threshold
        self.max_memory_gb = max_memory_gb
        self.enable_augmentation = enable_augmentation
        self.max_examples_unaugmented = max_examples_unaugmented
        self.verbose = verbose
        self.random_seed = random_seed
        self.is_validation = is_validation
        
        # Set up random seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Initialize data structures
        self.augmentation_factor = AUGMENTATION_FACTOR if enable_augmentation else 1
        self.logger = logging.getLogger(__name__)
        self.policy_shape = (BOARD_SIZE * BOARD_SIZE,)
        
        # Position pool and shard management
        self.position_pool: List[Dict] = []
        self.shard_queues: List[List[Path]] = []  # One queue per directory
        self.loaded_shards: set = set()  # Track loaded shards to prevent duplicates
        self.directory_weights: List[float] = []  # Proportional weights for each directory
        
        # Statistics and monitoring
        self.total_positions_yielded = 0
        self.total_shards_loaded = 0
        self.approx_batch_count = 0
        self._memory_warning_logged = False  # Track if we've already logged the memory warning
        self._shards_exhausted_logged = False  # Track if we've already logged that shards are exhausted
        
        # Initialize shard discovery and weighting
        self._discover_shards()
        self._calculate_directory_weights()
        
        # Validation-specific initialization
        if self.is_validation:
            self._initialize_validation_dataset()
        
        if self.verbose:
            dataset_type = "validation" if self.is_validation else "training"
            self.logger.info(f"[StreamingMixedShardDataset] Initialized {dataset_type} dataset with {len(self.data_dirs)} directories, "
                           f"pool_size={self.pool_size:,}, refill_threshold={self.refill_threshold:,}")
            for i, (dir_path, weight, shard_count) in enumerate(zip(self.data_dirs, self.directory_weights, [len(q) for q in self.shard_queues])):
                if self.verbose >= 3:
                    self.logger.info(f"  Directory {i+1}: {dir_path} (weight={weight:.3f}, {shard_count} shards)")
    
    def _discover_shards(self):
        """Discover and organize shards from all directories."""
        from hex_ai.data_collection import parse_shard_range
        
        self.shard_queues = []
        
        for i, (data_dir, shard_range) in enumerate(zip(self.data_dirs, self.shard_ranges)):
            try:
                # Skip directories with "None" shard range
                if shard_range.lower() == "none":
                    if self.verbose >= 2:
                        self.logger.info(f"Directory {i+1}: Skipping {data_dir} (range: {shard_range})")
                    self.shard_queues.append([])  # Empty queue for this directory
                    continue
                
                # Parse shard range for this directory
                start, end = parse_shard_range(shard_range, data_dir)
                
                if end is None:  # 'all' case
                    skip_files = 0
                    max_files = None
                else:
                    skip_files = start
                    max_files = end - start + 1
                
                # Discover files in this directory
                dataset_type = "validation" if self.is_validation else "training"
                process_context = f"{dataset_type} dataset initialization"
                data_files = discover_training_data_files(data_dir, skip_files=skip_files, max_files=max_files, process_context=process_context)
                
                if not data_files:
                    raise RuntimeError(f"No data files found in {data_dir} with range {shard_range}")
                
                self.shard_queues.append(data_files)
                
                if self.verbose >= 3:
                    self.logger.info(f"Directory {i+1}: Found {len(data_files)} shards in {data_dir} (range: {shard_range})")
                    
            except Exception as e:
                self.logger.error(f"Failed to discover shards in {data_dir}: {e}")
                raise RuntimeError(f"Failed to discover shards in {data_dir}: {e}")
        
        # Estimate total positions and games after shard discovery
        self._estimate_total_data()
        
        # Store original state for reset functionality
        self._original_shard_queues = [queue.copy() for queue in self.shard_queues]
    
    def _estimate_total_data(self):
        """Estimate total positions and games by sampling a few shards from each directory."""
        import pickle
        import gzip
        import random
        
        total_estimated_positions = 0
        total_estimated_games = 0
        
        for i, (data_dir, shard_queue) in enumerate(zip(self.data_dirs, self.shard_queues)):
            if not shard_queue:
                continue
                
            # Sample up to 3 shards from this directory to estimate
            sample_size = min(3, len(shard_queue))
            sample_shards = random.sample(shard_queue, sample_size)
            
            dir_positions = 0
            dir_games = 0
            
            for shard_path in sample_shards:
                try:
                    with gzip.open(shard_path, 'rb') as f:
                        data = pickle.load(f)
                        
                    if isinstance(data, dict) and 'examples' in data:
                        positions = len(data['examples'])
                        dir_positions += positions
                        
                        # Estimate games from positions (rough estimate: 60-80 positions per game)
                        # We'll use 70 as a middle ground
                        estimated_games = positions // 70
                        dir_games += estimated_games
                        
                except Exception as e:
                    if self.verbose >= 2:
                        self.logger.warning(f"Could not sample shard {shard_path}: {e}")
                    continue
            
            if sample_size > 0:
                # Average the sample and scale to total shards in this directory
                avg_positions_per_shard = dir_positions / sample_size
                avg_games_per_shard = dir_games / sample_size
                
                total_shards_in_dir = len(shard_queue)
                estimated_dir_positions = int(avg_positions_per_shard * total_shards_in_dir)
                estimated_dir_games = int(avg_games_per_shard * total_shards_in_dir)
                
                total_estimated_positions += estimated_dir_positions
                total_estimated_games += estimated_dir_games
                
                if self.verbose:
                    self.logger.info(f"Directory {i+1} ({data_dir}): ~{estimated_dir_positions:,} positions, ~{estimated_dir_games:,} games "
                                   f"({total_shards_in_dir} shards, sampled {sample_size})")
        
        self.estimated_total_positions = total_estimated_positions
        self.estimated_total_games = total_estimated_games
        
        if self.verbose:
            self.logger.info(f"Estimated total training data: ~{total_estimated_positions:,} positions from ~{total_estimated_games:,} games")
    
    def get_data_summary(self) -> dict:
        """Get a summary of the estimated training data."""
        return {
            'estimated_total_positions': getattr(self, 'estimated_total_positions', 0),
            'estimated_total_games': getattr(self, 'estimated_total_games', 0),
            'total_shards': sum(len(queue) for queue in self.shard_queues),
            'directories': len(self.data_dirs)
        }
    
    def reset(self):
        """Reset dataset for new epoch."""
        if self.verbose >= 2:
            dataset_type = "validation" if self.is_validation else "training"
            self.logger.info(f"[StreamingMixedShardDataset] Resetting {dataset_type} dataset for new epoch...")
        
        # Reset counters
        self.total_positions_yielded = 0
        self.total_shards_loaded = 0
        self.approx_batch_count = 0
        self._memory_warning_logged = False
        self._shards_exhausted_logged = False
        
        if self.is_validation:
            # For validation datasets, reset the position index
            if hasattr(self, 'validation_positions'):
                self.validation_position_index = 0
                if self.verbose >= 2:
                    self.logger.info(f"[StreamingMixedShardDataset] Validation dataset reset: {len(self.validation_positions):,} positions available")
        else:
            # Training dataset reset logic (original)
            # Restore original shard queues
            self.shard_queues = [queue.copy() for queue in self._original_shard_queues]
            
            # Clear loaded shards tracking
            self.loaded_shards = set()
            
            # Clear position pool
            self.position_pool = []
            
            # Shuffle shard queues for this epoch (using same random seed for reproducibility)
            for queue in self.shard_queues:
                random.shuffle(queue)
            
            # Refill initial pool
            self._refill_pool()
            
            if self.verbose >= 2:
                total_shards = sum(len(queue) for queue in self.shard_queues)
                self.logger.info(f"[StreamingMixedShardDataset] Training dataset reset complete: {total_shards} shards available, pool size: {len(self.position_pool):,}")
    
    def _calculate_directory_weights(self):
        """Calculate proportional weights for each directory based on shard counts."""
        shard_counts = [len(queue) for queue in self.shard_queues]
        total_shards = sum(shard_counts)
        
        if total_shards == 0:
            raise RuntimeError("No shards found in any directory")
        
        self.directory_weights = [count / total_shards for count in shard_counts]
        
        if self.verbose:
            self.logger.info(f"Directory weights: {[f'{w:.3f}' for w in self.directory_weights]}")
    
    def _initialize_validation_dataset(self):
        """
        Initialize validation dataset with improved memory estimation and guards.
        Estimates memory usage from disk sizes before loading, then loads and shuffles data.
        """
        # Step 1: Estimate total disk size and memory usage before loading
        total_disk_size = 0
        for i, (data_dir, shard_queue) in enumerate(zip(self.data_dirs, self.shard_queues)):
            if not shard_queue:
                continue
            for shard_path in shard_queue:
                if shard_path.exists():
                    total_disk_size += shard_path.stat().st_size
        
        # Estimate memory usage from disk size using compression ratio
        estimated_temp_memory_gb = (total_disk_size * VALIDATION_DATA_COMPRESSION_RATIO) / (1024**3)
        
        # Check if temporary loading would exceed memory limit
        if estimated_temp_memory_gb > MAX_TEMP_MEMORY_GB:
            raise RuntimeError(f"Validation data would use {estimated_temp_memory_gb:.1f}GB during loading, exceeds {MAX_TEMP_MEMORY_GB}GB limit")
        
        # Step 2: Load all validation data into memory for shuffling
        all_validation_positions = []
        
        for i, (data_dir, shard_queue) in enumerate(zip(self.data_dirs, self.shard_queues)):
            if not shard_queue:
                continue
                
            for shard_path in shard_queue:
                try:
                    with gzip.open(shard_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    if isinstance(data, dict) and 'examples' in data:
                        all_validation_positions.extend(data['examples'])
                        
                except Exception as e:
                    self.logger.error(f"Failed to load validation shard {shard_path}: {e}")
                    raise RuntimeError(f"Failed to load validation shard {shard_path}: {e}")
        
        # Step 3: Calculate usage fraction before shuffling/limiting
        total_loaded_positions = len(all_validation_positions)
        
        # Calculate usage fraction: how much of the loaded data we'll actually use
        if self.max_examples_unaugmented is not None and total_loaded_positions > self.max_examples_unaugmented:
            # We loaded more than we need, so we'll use a fraction
            usage_fraction = self.max_examples_unaugmented / total_loaded_positions
        else:
            # We'll use all loaded data
            usage_fraction = 1.0
        
        # Estimate final memory usage based on usage fraction
        estimated_final_memory_gb = estimated_temp_memory_gb * usage_fraction
        
        # Step 4: Check final memory usage against validation limit
        if estimated_final_memory_gb > MAX_VALIDATION_MEMORY_GB:
            raise RuntimeError(f"Validation data would use {estimated_final_memory_gb:.1f}GB, exceeds {MAX_VALIDATION_MEMORY_GB}GB limit")
        
        # Step 5: Shuffle validation data deterministically
        if self.random_seed is not None:
            random.seed(self.random_seed)
        random.shuffle(all_validation_positions)
        
        # Step 6: Limit to max_validation_examples if specified
        if self.max_examples_unaugmented is not None and len(all_validation_positions) > self.max_examples_unaugmented:
            all_validation_positions = all_validation_positions[:self.max_examples_unaugmented]
        
        # Step 7: Get final position count and check limits
        actual_positions = len(all_validation_positions)
        
        if actual_positions > 5_000_000:
            raise RuntimeError(f"Validation data would have {actual_positions:,} positions, exceeds 5M limit")
        
        # Store shuffled validation data
        self.validation_positions = all_validation_positions
        self.validation_position_index = 0
        
        # Small validation set warning
        if len(all_validation_positions) < 10_000:
            self.logger.warning(f"WARNING: Only {len(all_validation_positions):,} validation samples (recommended: >= 10,000)")
        
        if self.verbose:
            self.logger.info(f"Validation dataset initialized: {len(all_validation_positions):,} positions, "
                           f"estimated {estimated_final_memory_gb:.2f}GB memory usage")
    
    def _monitor_memory(self) -> bool:
        """
        Monitor memory usage and return True if within limits, False if should shutdown.
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_gb = memory_info.rss / (1024**3)  # Convert bytes to GB
            
            if memory_gb > self.max_memory_gb:
                self.logger.error(f"Memory usage ({memory_gb:.2f}GB) exceeds limit ({self.max_memory_gb}GB). Shutting down gracefully.")
                return False
            
            if self.verbose >= 2 and memory_gb > self.max_memory_gb * 0.8 and not self._memory_warning_logged:
                self.logger.warning(f"Memory usage is high: {memory_gb:.2f}GB (limit: {self.max_memory_gb}GB)")
                self._memory_warning_logged = True
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to monitor memory: {e}")
            raise RuntimeError(f"Memory monitoring failed: {e}")

    def __iter__(self):
        """
        Main iteration logic - yields positions from the mixed pool.
        For validation datasets, yields from pre-shuffled validation data.
        """
        # Reset statistics
        self.total_positions_yielded = 0
        self.total_shards_loaded = 0
        self.approx_batch_count = 0
        
        # Handle validation datasets differently
        if self.is_validation:
            yield from self._iterate_validation_data()
            return
        
        # Training dataset logic (original)
        # Initial pool fill
        self._refill_pool()
        
        # Main iteration loop
        while self.position_pool and (self.max_examples_unaugmented is None or 
                                    self.total_positions_yielded < self.max_examples_unaugmented):
            
            # Check memory limits
            if not self._monitor_memory():
                break
            
            # Refill pool if needed and shards are available
            if len(self.position_pool) < self.refill_threshold:
                if self._has_available_shards():
                    self._refill_pool()
                else:
                    # No more shards available - log this once
                    if self.verbose >= 2 and not self._shards_exhausted_logged:
                        self.logger.info(f"[StreamingMixedShardDataset] No more shards available, continuing with remaining {len(self.position_pool):,} positions")
                        self._shards_exhausted_logged = True
            
            # Yield positions from pool
            if self.position_pool:
                position = self.position_pool.pop(0)  # Remove from front of pool
                yield self._process_position(position)
                self.total_positions_yielded += 1
                
                # Update batch count (approximate)
                if self.total_positions_yielded % 256 == 0:
                        self.approx_batch_count += 1
            
            if self.verbose >= 5:
                self.logger.info(f"[StreamingMixedShardDataset] Iteration complete: "
                               f"yielded {self.total_positions_yielded:,} positions from {self.total_shards_loaded} shards")
    
    def _iterate_validation_data(self):
        """
        Iterate through pre-shuffled validation data.
        Yields positions from the in-memory validation dataset in shuffled order.
        """
        if not hasattr(self, 'validation_positions'):
            raise RuntimeError("Validation dataset not properly initialized")
        
        for position in self.validation_positions:
            yield self._process_position(position)
            self.total_positions_yielded += 1
            
            # Update batch count (approximate)
            if self.total_positions_yielded % 256 == 0:
                self.approx_batch_count += 1
        
        if self.verbose >= 3:
            self.logger.info(f"[StreamingMixedShardDataset] Validation iteration complete: "
                           f"yielded {self.total_positions_yielded:,} positions")
    
    def _refill_pool(self):
        """Load new shards and add positions to the pool."""
        if self.verbose >= 3:
            self.logger.info(f"[StreamingMixedShardDataset] Refilling pool (current size: {len(self.position_pool):,})")
        
        # Calculate how many positions we need to add
        positions_needed = self.pool_size - len(self.position_pool)
        if positions_needed <= 0:
            return
        
        # Load shards proportionally until we have enough positions
        positions_added = 0
        shards_loaded_this_refill = 0
        
        while positions_added < positions_needed and self._has_available_shards():
            # Select directory to load from based on weights
            selected_dir_idx = self._select_directory_for_loading()
            if selected_dir_idx is None:
                break  # No more shards available
            
            # Load next shard from selected directory
            shard_path = self.shard_queues[selected_dir_idx][0]  # Get first shard from queue
            
            try:
                # Load shard data
                with gzip.open(shard_path, 'rb') as f:
                    data = pickle.load(f)
                
                file_examples = data['examples'] if 'examples' in data else []
                
                if not file_examples:
                    self.logger.warning(f"Shard {shard_path} contains no examples, skipping")
                    self.shard_queues[selected_dir_idx].pop(0)  # Remove empty shard
                    continue
                
                # Add positions to pool
                for example in file_examples:
                    if positions_added >= positions_needed:
                        break
                    self.position_pool.append(example)
                    positions_added += 1
                
                # Mark shard as loaded and remove from queue
                self.loaded_shards.add(str(shard_path))
                self.shard_queues[selected_dir_idx].pop(0)
                self.total_shards_loaded += 1
                shards_loaded_this_refill += 1
                
                if self.verbose >= 3:
                    self.logger.info(f"Loaded shard {shard_path.name}: {len(file_examples)} examples "
                                   f"(added {min(positions_added, positions_needed)} to pool)")
                
            except Exception as e:
                self.logger.error(f"Failed to load shard {shard_path}: {e}")
                raise RuntimeError(f"Failed to load shard {shard_path}: {e}")
        
        # Shuffle the entire pool after refilling
        if positions_added > 0:
            random.shuffle(self.position_pool)
            if self.verbose >= 3:
                self.logger.info(f"Shuffled pool after adding {positions_added:,} positions "
                               f"(total pool size: {len(self.position_pool):,})")
        
        if self.verbose >= 3:
            self.logger.info(f"Pool refill complete: added {positions_added:,} positions from {shards_loaded_this_refill} shards")
    
    def _has_available_shards(self) -> bool:
        """Check if any directories still have unloaded shards."""
        return any(len(queue) > 0 for queue in self.shard_queues)
    
    def _select_directory_for_loading(self) -> Optional[int]:
        """
        Select which directory to load the next shard from based on proportional weights.
        Returns the directory index, or None if no directories have available shards.
        """
        available_dirs = [i for i, queue in enumerate(self.shard_queues) if len(queue) > 0]
        
        if not available_dirs:
            return None
        
        # If only one directory has shards, use it
        if len(available_dirs) == 1:
            return available_dirs[0]
        
        # Calculate current loading ratios for available directories
        current_ratios = []
        for dir_idx in available_dirs:
            # Count how many shards we've loaded from this directory
            loaded_from_dir = sum(1 for shard_path in self.loaded_shards 
                                if str(shard_path).startswith(self.data_dirs[dir_idx]))
            total_shards_in_dir = len(self.shard_queues[dir_idx]) + loaded_from_dir
            
            if total_shards_in_dir > 0:
                current_ratio = loaded_from_dir / total_shards_in_dir
            else:
                current_ratio = 0.0
            
            current_ratios.append(current_ratio)
        
        # Find directory that's furthest behind its target weight
        target_ratios = [self.directory_weights[i] for i in available_dirs]
        deficits = [target - current for target, current in zip(target_ratios, current_ratios)]
        
        # Select directory with largest deficit
        max_deficit_idx = max(range(len(deficits)), key=lambda i: deficits[i])
        return available_dirs[max_deficit_idx]
    
    def _process_position(self, position: Dict):
        """Process a single position (augmentation, tensor conversion, etc.)."""
        board = position['board']
        policy = position['policy']
        value = position['value']
        player_to_move = position.get('player_to_move', None)
        
        # Convert integer player_to_move to Player enum if needed (for backward compatibility)
        if player_to_move is not None and isinstance(player_to_move, int):
            from hex_ai.value_utils import Player
            player_to_move = Player(player_to_move)
        
        board_2ch = board[:PLAYER_CHANNEL] if board.shape[0] > 1 else board
        
        if self.enable_augmentation:
            # Apply augmentation
            error_tracker = get_board_state_error_tracker()
            error_tracker._current_file = "mixed_pool"
            error_tracker._current_sample = f"pool_position_{self.total_positions_yielded}"
            
            augmented_examples = create_augmented_example_with_player_to_move(
                board_2ch, policy, value, error_tracker)
            
            # Select one augmentation randomly
            aug_idx = random.randint(0, len(augmented_examples) - 1)
            aug = augmented_examples[aug_idx]
            return self._transform_example(*aug)
        else:
            if player_to_move is None:
                raise ValueError("Missing 'player_to_move' in example during data loading. All examples must have this field.")
            return self._transform_example(board_2ch, policy, value, player_to_move)

    def _normalize_policy(self, policy):
        """Normalize policy tensor, handling None values."""
        if policy is None:
            return np.zeros(self.policy_shape, dtype=np.float32)
        return policy

    def _transform_example(self, board_2ch, policy, value, player=None):
        """Transform example data into tensors."""
        if player is not None:
            # Player should always be a Player enum
            if not hasattr(player, 'value'):
                raise ValueError(f"Expected Player enum, got {type(player)}. This indicates a bug in the data pipeline.")
            player_value = player.value
            player_channel = np.full((board_2ch.shape[1], board_2ch.shape[2]), player_value, dtype=np.float32)
            board_3ch = np.concatenate([board_2ch, player_channel[None, ...]], axis=0)
        else:
            board_3ch = board_2ch
        
        # Compute move_stage from board state
        # Count stones from both player planes (channels 0 and 1)
        stones_plane = (board_3ch[0] > 0).astype(np.int32) + (board_3ch[1] > 0).astype(np.int32)
        stones_on_board = int(stones_plane.sum())
        
        # Normalize by board area
        board_area = board_3ch.shape[1] * board_3ch.shape[2]  # height * width
        move_stage = stones_on_board / board_area  # Python float in [0,1]
        
        board_tensor = torch.from_numpy(board_3ch).float()
        policy = self._normalize_policy(policy)
        policy_tensor = torch.FloatTensor(policy)
        
        # CRITICAL FIX: Convert value targets from [0,1] to [-1,1] range
        # Training data has values in [0,1] range (0.0 = Blue win, 1.0 = Red win)
        # But the model outputs values in [-1,1] range with tanh activation
        # MCTS expects [-1,1] range values
        value_signed = 2.0 * value - 1.0  # Convert [0,1] -> [-1,1]
        value_tensor = torch.FloatTensor([value_signed])
        
        move_stage_tensor = torch.tensor(move_stage, dtype=torch.float32)
        
        return board_tensor, policy_tensor, value_tensor, move_stage_tensor
    
    def __len__(self):
        # HACK: PyTorch DataLoader sometimes calls __len__ even for IterableDataset
        import warnings
        warnings.warn(
            "__len__ called on StreamingMixedShardDataset. Returning a large dummy value. "
            "This is a workaround for PyTorch DataLoader compatibility.",
            RuntimeWarning
        )
        return 10**12


def discover_training_data_files(data_dir: str = "data/processed", skip_files: int = 0, max_files: Optional[int] = None, process_context: str = "training data loading") -> List[Path]:
    """
    Discover training data files in the specified directory.
    
    This function finds either:
    - Shuffled training data files (shuffled_*.pkl.gz) if shuffling_progress.json exists
    - Ordered position files (*_processed.pkl.gz) otherwise
    
    Args:
        data_dir: Directory containing data files
        skip_files: Number of files to skip from the beginning (sorted by name)
        max_files: Maximum number of files to use after skipping (None = use all remaining)
        process_context: Context string describing what process is calling this function
        
    Returns:
        List of paths to data files
    """
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found")
    
    # Check if this is shuffled data directory
    if (data_path / "shuffling_progress.json").exists():
        # Shuffled data: look for shuffled_*.pkl.gz files
        data_files = list(data_path.glob("shuffled_*.pkl.gz"))
        logger.info(f"[DATA_DISCOVERY] {process_context}: Found {len(data_files)} shuffled data files in {data_dir}")
    else:
        # Original processed data: look for *_processed.pkl.gz files
        data_files = list(data_path.glob("*_processed.pkl.gz"))
        logger.info(f"[DATA_DISCOVERY] {process_context}: Looking for processed data files in {data_dir}")
        logger.info(f"[DATA_DISCOVERY] {process_context}: Found {len(data_files)} processed data files (not shuffled)")
        if len(data_files) == 0:
            logger.warning(f"[DATA_DISCOVERY] {process_context}: No processed data files found in {data_dir}")
            logger.warning(f"[DATA_DISCOVERY] {process_context}: This directory does not contain the expected processed data files")
            logger.warning(f"[DATA_DISCOVERY] {process_context}: Expected files matching pattern: *_processed.pkl.gz")
            logger.warning(f"[DATA_DISCOVERY] {process_context}: Do you want to quit this run and try again? (Ctrl+C to quit)")
            sleep(5)
    
    if not data_files:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    
    # Sort files by name for consistent ordering
    data_files.sort()
    
    # Skip the first N files if requested
    if skip_files > 0:
        if skip_files >= len(data_files):
            raise ValueError(f"Cannot skip {skip_files} files when only {len(data_files)} files exist in {data_dir}")
        data_files = data_files[skip_files:]
        logger.info(f"Skipped first {skip_files} files from {data_dir}, using {len(data_files)} remaining files")
    
    # Limit to max_files if requested
    if max_files is not None and max_files > 0:
        if max_files < len(data_files):
            data_files = data_files[:max_files]
            logger.info(f"Limited to first {max_files} files from {data_dir}, using {len(data_files)} files total")
    
    return data_files


# ============================================================================
# Data Shuffling Utilities
# ============================================================================

# Configuration constants for data shuffling
DEFAULT_NUM_BUCKETS = 500
BUCKET_ID_FORMAT_WIDTH = 4  # For :04d format in filenames


class DataShuffler:
    """Handles the two-phase data shuffling process."""
    
    def __init__(self, 
                 input_dir: str = "data/processed/step1_unshuffled",
                 output_dir: str = "data/processed/shuffled",
                 temp_dir: str = "data/processed/temp_buckets",
                 num_buckets: int = DEFAULT_NUM_BUCKETS,
                 resume_enabled: bool = True,
                 cleanup_temp: bool = True,
                 validation_enabled: bool = True):
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.num_buckets = num_buckets
        self.resume_enabled = resume_enabled
        self.cleanup_temp = cleanup_temp
        self.validation_enabled = validation_enabled
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'total_examples': 0,
            'buckets_completed': 0,
            'start_time': time.time()
        }
        
        # Progress tracking
        self.progress_file = self.output_dir / "shuffling_progress.json"
        self.progress = self._load_progress()
        
        # Shutdown handler
        from hex_ai.file_utils import GracefulShutdown
        self.shutdown_handler = GracefulShutdown()
    
    def _load_progress(self) -> Dict[str, Any]:
        """Load existing progress if available."""
        if not self.progress_file.exists() or not self.resume_enabled:
            return self._create_new_progress()
        
        try:
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
            
            logger.info("Resuming from previous run")
            logger.info(f"Files processed: {len(progress.get('processed_files', []))}")
            logger.info(f"Buckets completed: {len(progress.get('completed_buckets', []))}")
            
            # Restore stats from saved progress
            saved_stats = progress.get('stats', {})
            if saved_stats:
                self.stats.update(saved_stats)
                logger.info(f"Restored stats: {saved_stats}")
            
            # Fix incorrect phase state: if we have processed files but no completed buckets,
            # we should still be in distribution phase, not completed
            if (len(progress.get('processed_files', [])) > 0 and 
                len(progress.get('completed_buckets', [])) == 0 and
                progress.get('current_phase') == 'completed'):
                logger.warning("Detected incorrect phase state: files processed but no buckets completed, fixing to 'distribution'")
                progress['current_phase'] = 'distribution'
            
            return progress
        except Exception as e:
            logger.warning(f"Failed to load progress file: {e}")
            return self._create_new_progress()
    
    def _create_new_progress(self) -> Dict[str, Any]:
        """Create new progress tracking structure."""
        return {
            'started_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'processed_files': [],
            'completed_buckets': [],
            'current_phase': 'distribution',
            'stats': self.stats.copy()
        }
    
    def _save_progress(self):
        """Save current progress."""
        self.progress.update({
            'last_updated': datetime.now().isoformat(),
            'stats': self.stats.copy()
        })
        
        try:
            # Ensure output directory exists before saving progress
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            temp_progress_file = self.progress_file.with_suffix('.tmp')
            final_progress_file = self.progress_file
            
            # Step 1: Write to temporary file
            try:
                with open(temp_progress_file, 'w') as f:
                    json.dump(self.progress, f, indent=2)
                
                # Verify the temp file was actually written and has content
                if not temp_progress_file.exists():
                    raise RuntimeError(f"Temp progress file was not created: {temp_progress_file}")
                
                temp_size = temp_progress_file.stat().st_size
                if temp_size == 0:
                    raise RuntimeError(f"Temp progress file is empty (0 bytes): {temp_progress_file}")
                
                logger.debug(f"Successfully wrote {temp_size} bytes to temp progress file: {temp_progress_file}")
                
            except Exception as e:
                raise RuntimeError(f"Failed to write temp progress file {temp_progress_file}: {e}")
            
            # Step 2: Rename temp file to final file
            try:
                # Check if final file already exists and remove it first
                if final_progress_file.exists():
                    final_progress_file.unlink()
                    logger.debug(f"Removed existing progress file: {final_progress_file}")
                
                # Perform the rename
                temp_progress_file.rename(final_progress_file)
                logger.debug(f"Successfully renamed {temp_progress_file} -> {final_progress_file}")
                
            except Exception as e:
                # Provide detailed context about the rename failure
                temp_exists = temp_progress_file.exists()
                final_exists = final_progress_file.exists()
                temp_size = temp_progress_file.stat().st_size if temp_exists else "N/A"
                final_size = final_progress_file.stat().st_size if final_exists else "N/A"
                
                error_context = (
                    f"Failed to rename progress file:\n"
                    f"  Source: {temp_progress_file} (exists: {temp_exists}, size: {temp_size})\n"
                    f"  Target: {final_progress_file} (exists: {final_exists}, size: {final_size})\n"
                    f"  Output dir: {self.output_dir} (exists: {self.output_dir.exists()})\n"
                    f"  Error: {e}"
                )
                
                raise RuntimeError(f"Critical error: Failed to save progress file.\n{error_context}")
                
        except Exception as e:
            logger.error(f"Failed to save progress file: {e}")
            # CRASH instead of continuing
            raise
    
    def _load_pkl_gz(self, file_path: Path) -> Dict[str, Any]:
        """Load data from a .pkl.gz file."""
        try:
            with gzip.open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise
    
    def _write_bucket_file(self, bucket_idx: int, examples: List[Dict], source_files: List[str]):
        """Write examples to a bucket file."""
        # Use input filename in bucket filename to avoid overwriting
        input_filename = Path(source_files[0]).stem  # Remove .pkl.gz extension
        bucket_file = self.temp_dir / f"{input_filename}_bucket_{bucket_idx:0{BUCKET_ID_FORMAT_WIDTH}d}.pkl.gz"
        
        bucket_data = {
            'examples': examples,
            'bucket_id': bucket_idx,
            'source_files': source_files,
            'created_at': datetime.now().isoformat(),
            'num_examples': len(examples)
        }
        
        try:
            from hex_ai.file_utils import atomic_write_pickle_gz
            atomic_write_pickle_gz(bucket_data, bucket_file)
            logger.debug(f"Written bucket {bucket_idx} from {input_filename} with {len(examples)} examples")
        except Exception as e:
            logger.error(f"Failed to write bucket {bucket_idx} from {input_filename}: {e}")
            raise
    
    def _write_shuffled_file(self, bucket_idx: int, examples: List[Dict], source_files: List[str]):
        """Write shuffled examples to final output file."""
        shuffled_file = self.output_dir / f"shuffled_{bucket_idx:0{BUCKET_ID_FORMAT_WIDTH}d}.pkl.gz"
        
        shuffled_data = {
            'examples': examples,
            'shuffling_stats': {
                'num_buckets': self.num_buckets,
                'bucket_id': bucket_idx,
                'total_examples': len(examples),
                'shuffled_at': datetime.now().isoformat(),
                'source_files': source_files
            }
        }
        
        try:
            from hex_ai.file_utils import atomic_write_pickle_gz
            atomic_write_pickle_gz(shuffled_data, shuffled_file)
            logger.info(f"Written shuffled file {bucket_idx} with {len(examples)} examples")
        except Exception as e:
            logger.error(f"Failed to write shuffled file {bucket_idx}: {e}")
            raise
    
    def _distribute_single_file(self, input_file):
        try:
            data = self._load_pkl_gz(input_file)
            examples = data['examples']
            bucket_examples = [[] for _ in range(self.num_buckets)]
            for example_idx, example in enumerate(examples):
                bucket_idx = example_idx % self.num_buckets
                bucket_examples[bucket_idx].append(example)
            for bucket_idx, examples_for_bucket in enumerate(bucket_examples):
                if examples_for_bucket:
                    self._write_bucket_file(bucket_idx, examples_for_bucket, [str(input_file)])
            return str(input_file), len(examples)
        except Exception as e:
            logger.error(f"Error processing {input_file}: {e}")
            raise

    def _distribute_to_buckets(self, input_files):
        logger.info(f"Starting Phase 1: Distribution to {self.num_buckets} buckets (parallelized)")
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(self._distribute_single_file, input_file): input_file for input_file in input_files}
            for future in as_completed(futures):
                input_file = futures[future]
                try:
                    file_name, num_examples = future.result()
                    logger.info(f"Processed {file_name} with {num_examples} examples")
                    self.progress['processed_files'].append(str(input_file))
                    self.stats['files_processed'] += 1
                    self.stats['total_examples'] += num_examples
                    self._save_progress()
                except Exception as e:
                    logger.error(f"Error in parallel processing of {input_file}: {e}")
                    raise
    
    def _consolidate_and_shuffle_bucket(self, bucket_idx: int):
        """Phase 2: Consolidate and shuffle a single bucket.
        
        This method:
        1. Loads all bucket files for this bucket index from different input files
        2. Concatenates all examples into a single list
        3. Shuffles the examples to break any remaining correlations
        4. Writes the final shuffled file
        5. Optionally cleans up the temporary bucket files
        
        Returns the bucket_idx for progress tracking in the main process.
        """
        logger.info(f"Processing bucket {bucket_idx}")
        
        # Find all bucket files for this bucket index
        bucket_pattern = f"*_bucket_{bucket_idx:0{BUCKET_ID_FORMAT_WIDTH}d}.pkl.gz"
        bucket_files = list(self.temp_dir.glob(bucket_pattern))
        
        if not bucket_files:
            logger.warning(f"No files found for bucket {bucket_idx}")
            return bucket_idx
        
        # Load and consolidate examples from all bucket files
        all_examples = []
        source_files = []
        
        for bucket_file in bucket_files:
            try:
                data = self._load_pkl_gz(bucket_file)
                examples = data['examples']
                file_source_files = data.get('source_files', [])
                
                all_examples.extend(examples)
                source_files.extend(file_source_files)
                    
            except Exception as e:
                logger.error(f"Error loading {bucket_file}: {e}")
                raise  # Make file processing errors fatal
        
        if not all_examples:
            logger.warning(f"No examples found for bucket {bucket_idx}")
            return bucket_idx
        
        # Shuffle examples to break any remaining correlations
        # This is the final randomization step that addresses value head fingerprinting
        logger.info(f"Shuffling {len(all_examples)} examples in bucket {bucket_idx}")
        random.shuffle(all_examples)
        
        # Write final shuffled file with complete source tracking
        self._write_shuffled_file(bucket_idx, all_examples, source_files)
        
        # Clean up temporary bucket files if requested
        if self.cleanup_temp:
            for bucket_file in bucket_files:
                try:
                    bucket_file.unlink()
                except Exception as e:
                    logger.error(f"Failed to delete {bucket_file}: {e}")
                    raise  # Make cleanup errors fatal
        
        return bucket_idx
    
    def _consolidate_and_shuffle_all_buckets(self):
        logger.info(f"Starting Phase 2: Consolidation and shuffling of {self.num_buckets} buckets (parallelized)")
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(self._consolidate_and_shuffle_bucket, bucket_idx): bucket_idx for bucket_idx in range(self.num_buckets)}
            for future in as_completed(futures):
                bucket_idx = futures[future]
                try:
                    completed_bucket_idx = future.result()
                    logger.info(f"Bucket {completed_bucket_idx} consolidated and shuffled")
                    # Update progress tracking in main process to avoid race conditions
                    self.progress['completed_buckets'].append(completed_bucket_idx)
                    self.stats['buckets_completed'] += 1
                    self._save_progress()
                except Exception as e:
                    logger.error(f"Error in parallel processing of bucket {bucket_idx}: {e}")
                    raise
        logger.info("Phase 2 completed")
    
    def _validate_output(self):
        """Validate the shuffled output data.
        
        Currently only counts total examples to verify no data was lost.
        The distribution is guaranteed to be even by construction (≤169 moves per game,
        {self.num_buckets} buckets), so no distribution validation is needed.
        """
        if not self.validation_enabled:
            return
        
        logger.info("Validating shuffled output...")
        
        shuffled_files = list(self.output_dir.glob("shuffled_*.pkl.gz"))
        total_examples = 0
        
        for shuffled_file in shuffled_files:
            try:
                data = self._load_pkl_gz(shuffled_file)
                examples = data['examples']
                total_examples += len(examples)
                
            except Exception as e:
                logger.error(f"Error validating {shuffled_file}: {e}")
                raise  # Make validation errors fatal
        
        # Report validation results
        logger.info(f"Validation complete:")
        logger.info(f"  Total shuffled files: {len(shuffled_files)}")
        logger.info(f"  Total examples: {total_examples}")
    
    def shuffle_data(self):
        """Main method to run the complete shuffling process."""
        logger.info("Starting data shuffling process")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Number of buckets: {self.num_buckets}")
        
        # Find input files
        input_files = list(self.input_dir.glob("*.pkl.gz"))
        if not input_files:
            logger.error(f"No .pkl.gz files found in {self.input_dir}")
            return
        
        logger.info(f"Found {len(input_files)} input files")
        
        try:
            # Phase 1: Distribution
            if self.progress['current_phase'] == 'distribution':
                self._distribute_to_buckets(input_files)
                self.progress['current_phase'] = 'consolidation'
                self._save_progress()
            
            # Phase 2: Consolidation and shuffling
            if self.progress['current_phase'] == 'consolidation':
                self._consolidate_and_shuffle_all_buckets()
                self.progress['current_phase'] = 'completed'
                self._save_progress()
            
            # Validation
            if self.progress['current_phase'] == 'completed':
                self._validate_output()
            
            # Final statistics
            elapsed_time = time.time() - self.stats['start_time']
            logger.info("=" * 60)
            logger.info("SHUFFLING COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"Files processed: {self.stats['files_processed']}")
            logger.info(f"Total examples: {self.stats['total_examples']}")
            logger.info(f"Buckets completed: {self.stats['buckets_completed']}")
            logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
            logger.info(f"Output files: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error during shuffling process: {e}")
            raise




