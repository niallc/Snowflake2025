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
    
    def __init__(self, data_files: List[Path], chunk_size: int = 100000, shuffle_files: bool = True):
        """
        Initialize streaming dataset.
        
        Args:
            data_files: List of paths to processed data files
            chunk_size: Number of examples to load at once
            shuffle_files: Whether to shuffle the order of files
        """
        self.data_files = data_files
        if shuffle_files:
            random.shuffle(self.data_files)
        
        self.chunk_size = chunk_size
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
        # After loading, check error thresholds
        if files_attempted > 0:
            error_log_dir = str(self.data_files[0].parent) if self.data_files else "."
            check_data_loading_errors(files_attempted, files_with_errors, error_details, error_log_dir)
        logger.info(f"Loaded chunk of {len(self.current_chunk)} examples (total loaded: {self.total_examples_loaded + len(self.current_chunk)})")
    
    def __len__(self):
        """
        Return an estimated number of samples (for DataLoader compatibility).
        NOTE: This is an estimate based on the number of valid files and chunk size,
        not the true number of available samples. For small datasets or tests, use
        dataset.true_len() for the exact count.
        """
        num_valid_files = len([f for f in self.data_files if f.exists()])
        return num_valid_files * self.chunk_size
    
    def __getitem__(self, idx):
        """Get a single training sample."""
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
        board_state = torch.FloatTensor(sample[0])
        # Add player-to-move channel
        board_np = board_state.numpy()
        from hex_ai.inference.board_utils import BLUE_PLAYER, RED_PLAYER
        player_to_move = get_player_to_move_from_board(board_np)
        player_channel = np.full((board_np.shape[1], board_np.shape[2]), float(player_to_move), dtype=np.float32)
        board_3ch = np.concatenate([board_np, player_channel[None, ...]], axis=0)
        board_state = torch.from_numpy(board_3ch)
        policy_target = torch.FloatTensor(sample[1]) if sample[1] is not None else torch.zeros(POLICY_OUTPUT_SIZE, dtype=torch.float32)
        value_target = torch.FloatTensor([sample[2]])
        return board_state, policy_target, value_target





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


def create_experiment_config(experiment_name: str,
                           hyperparams: Dict,
                           dataset_info: Dict,
                           device: str) -> Dict:
    """
    Create a standardized experiment configuration.
    
    Args:
        experiment_name: Name of the experiment
        hyperparams: Hyperparameters for the experiment
        dataset_info: Information about the dataset
        device: Device being used for training
        
    Returns:
        Experiment configuration dictionary
    """
    return {
        'experiment_name': experiment_name,
        'hyperparameters': hyperparams,
        'dataset_info': dataset_info,
        'device': device,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0'  # For tracking format changes
    }





# [MOVED] run_hyperparameter_experiment has been moved to scripts/run_hyperparameter_experiment.py


# run_hyperparameter_tuning has been moved to scripts/run_hyperparameter_tuning.py and removed from this file.


def create_summary_csv(experiment_results: List[Dict], results_path: Path):
    """
    Create a summary CSV file with all experiment results.
    
    Args:
        experiment_results: List of experiment result dictionaries
        results_path: Path to save the summary CSV
    """
    import csv
    from datetime import datetime
    
    summary_file = results_path / "experiment_summary.csv"
    
    # Define headers for summary CSV
    headers = [
        'experiment_name', 'timestamp', 'date',
        'learning_rate', 'batch_size', 'dropout_prob', 'weight_decay',
        'policy_weight', 'value_weight',
        'best_val_loss', 'best_train_loss', 'final_val_loss', 'final_train_loss',
        'epochs_trained', 'early_stopped', 'training_time',
        'device', 'dataset_size'
    ]
    
    with open(summary_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for exp in experiment_results:
            row = {
                'experiment_name': exp['experiment_name'],
                'timestamp': datetime.now().isoformat(),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'learning_rate': exp['hyperparameters'].get('learning_rate', ''),
                'batch_size': exp['hyperparameters'].get('batch_size', ''),
                'dropout_prob': exp['hyperparameters'].get('dropout_prob', ''),
                'weight_decay': exp['hyperparameters'].get('weight_decay', ''),
                'policy_weight': exp['hyperparameters'].get('policy_weight', ''),
                'value_weight': exp['hyperparameters'].get('value_weight', ''),
                'best_val_loss': exp['best_val_loss'],
                'best_train_loss': exp['best_train_loss'],
                'final_val_loss': exp['final_val_loss'],
                'final_train_loss': exp['final_train_loss'],
                'epochs_trained': exp['epochs_trained'],
                'early_stopped': exp['early_stopped'],
                'training_time': exp['training_time'],
                'device': exp.get('device', ''),
                'dataset_size': exp.get('dataset_size', '')
            }
            writer.writerow(row)
    
    logger.info(f"Created experiment summary CSV: {summary_file}") 