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
import psutil
from .models import TwoHeadedResNet
from .config import BOARD_SIZE, POLICY_OUTPUT_SIZE, PLAYER_CHANNEL, DEFAULT_POOL_SIZE, DEFAULT_REFILL_THRESHOLD, DEFAULT_MAX_MEMORY_GB
from hex_ai.data_utils import get_player_to_move_from_board, create_augmented_example_with_player_to_move
from hex_ai.error_handling import check_data_loading_errors, get_board_state_error_tracker

logger = logging.getLogger(__name__)

AUGMENTATION_FACTOR = 4  # Number of augmentations per unaugmented board (rotations/reflections)


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
                 random_seed: Optional[int] = None):
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
        
        if self.verbose:
            self.logger.info(f"[StreamingMixedShardDataset] Initialized with {len(self.data_dirs)} directories, "
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
                # Parse shard range for this directory
                start, end = parse_shard_range(shard_range, data_dir)
                
                if end is None:  # 'all' case
                    skip_files = 0
                    max_files = None
                else:
                    skip_files = start
                    max_files = end - start + 1
                
                # Discover files in this directory
                data_files = discover_processed_files(data_dir, skip_files=skip_files, max_files=max_files)
                
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
    
    def _calculate_directory_weights(self):
        """Calculate proportional weights for each directory based on shard counts."""
        shard_counts = [len(queue) for queue in self.shard_queues]
        total_shards = sum(shard_counts)
        
        if total_shards == 0:
            raise RuntimeError("No shards found in any directory")
        
        self.directory_weights = [count / total_shards for count in shard_counts]
        
        if self.verbose:
            self.logger.info(f"Directory weights: {[f'{w:.3f}' for w in self.directory_weights]}")
    
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
        """
        # Reset statistics
        self.total_positions_yielded = 0
        self.total_shards_loaded = 0
        self.approx_batch_count = 0
        
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
        
        board_tensor = torch.from_numpy(board_3ch).float()
        policy = self._normalize_policy(policy)
        policy_tensor = torch.FloatTensor(policy)
        value_tensor = torch.FloatTensor([value])
        return board_tensor, policy_tensor, value_tensor
    
    def __len__(self):
        # HACK: PyTorch DataLoader sometimes calls __len__ even for IterableDataset
        import warnings
        warnings.warn(
            "__len__ called on StreamingMixedShardDataset. Returning a large dummy value. "
            "This is a workaround for PyTorch DataLoader compatibility.",
            RuntimeWarning
        )
        return 10**12


def discover_processed_files(data_dir: str = "data/processed", skip_files: int = 0, max_files: Optional[int] = None) -> List[Path]:
    """
    Discover all processed data files in the specified directory.
    
    Args:
        data_dir: Directory containing processed data files
        skip_files: Number of files to skip from the beginning (sorted by name)
        max_files: Maximum number of files to use after skipping (None = use all remaining)
        
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
        logger.info(f"WARNING: Failed to find shuffled files. Found {len(data_files)} processed data files")
        logger.info(f"WARNING: Do you want to quit this run and try again? (Ctrl+C to quit)")
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




