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
from .training import Trainer, EarlyStopping
from .config import BOARD_SIZE, POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE

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


class NewProcessedDataset(torch.utils.data.Dataset):
    """Dataset for loading the new processed data format."""
    
    def __init__(self, data_files: List[Path], shuffle_files: bool = True, max_examples: Optional[int] = None):
        """
        Initialize dataset with new format data files.
        
        Args:
            data_files: List of paths to processed data files
            shuffle_files: Whether to shuffle the order of files
        """
        self.data_files = data_files
        if shuffle_files:
            random.shuffle(self.data_files)
        
        self.max_examples = max_examples
        
        # Load examples from files
        self.examples = []
        self.file_stats = {}
        
        for i, file_path in enumerate(self.data_files):
            if i % 10 == 0:  # Log progress every 10 files
                logger.info(f"Loading data files: {i+1}/{len(self.data_files)}")
            
            try:
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
                logger.debug(f"Successfully loaded {file_path.name}")
                logger.debug(f"Data keys: {list(data.keys())}")
            except (EOFError, OSError) as e:
                logger.error(f"Corrupted file {file_path.name}: {e}")
                continue
            except pickle.UnpicklingError as e:
                logger.error(f"Invalid pickle data in {file_path.name}: {e}")
                continue
            except KeyError as e:
                logger.error(f"Missing required key '{e}' in {file_path.name}")
                continue
            except ValueError as e:
                logger.error(f"Invalid data format in {file_path.name}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error loading {file_path.name}: {e}")
                logger.debug(f"Exception type: {type(e).__name__}")
                continue
            
            # Process the loaded data
            logger.debug(f"Checking for 'examples' key in {file_path.name}")
            if 'examples' not in data:
                logger.warning(f"No 'examples' key found in {file_path}")
                logger.debug(f"Available keys in {file_path.name}: {list(data.keys())}")
                continue
            logger.debug(f"'examples' key found in {file_path.name}")
            
            file_examples = data['examples']
            logger.debug(f"Found {len(file_examples)} examples in {file_path.name}")
            if len(file_examples) == 0:
                logger.warning(f"File {file_path.name} has 0 examples")
                continue
            
            # Validate first example to fail fast
            if len(file_examples) > 0:
                try:
                    _validate_example_format(file_examples[0], file_path.name)
                    logger.debug(f"Data format validated for {file_path.name}")
                except ValueError as e:
                    logger.error(f"Data format error in {file_path.name}: {e}")
                    continue
            
            logger.info(f"Processing {len(file_examples)} examples from {file_path.name}")
            
            # Limit examples if max_examples is specified
            if self.max_examples is not None and len(self.examples) + len(file_examples) > self.max_examples:
                remaining = self.max_examples - len(self.examples)
                if remaining > 0:
                    file_examples = file_examples[:remaining]
                    logger.info(f"Limited to {remaining} examples from {file_path.name}")
                else:
                    break  # We've reached the limit
            
            logger.debug(f"Adding {len(file_examples)} examples from {file_path.name}")
            logger.debug(f"Before adding: {len(self.examples)} total examples")
            self.examples.extend(file_examples)
            logger.debug(f"After adding: {len(self.examples)} total examples")
            
            # Store file statistics
            self.file_stats[file_path] = {
                'num_examples': len(file_examples),
                'source_file': data.get('source_file', 'unknown'),
                'processing_stats': data.get('processing_stats', {})
            }
            logger.debug(f"Added file stats for {file_path.name}: {len(file_examples)} examples")
            
            logger.debug(f"Loaded {len(file_examples)} examples from {file_path}")
            
            # Stop if we've reached the limit
            if self.max_examples is not None and len(self.examples) >= self.max_examples:
                logger.info(f"Reached target of {self.max_examples} examples, stopping data loading")
                break
            
            # Log progress every few files
            if i % 5 == 0 and i > 0:
                logger.info(f"Loaded {len(self.examples)} examples so far from {i+1} files")
        
        logger.info(f"Loaded {len(self.examples)} total examples from {len(self.data_files)} files")
        if len(self.examples) == 0:
            logger.error("WARNING: No examples loaded! This will cause training to fail.")
            logger.error(f"Files attempted: {[f.name for f in self.data_files]}")
            logger.error(f"File stats: {self.file_stats}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Get a single training example."""
        if idx >= len(self.examples):
            raise IndexError(f"Index {idx} out of range")
        
        example = self.examples[idx]
        
        # Debug: Check example structure
        if idx == 0:  # Only log for first example to avoid spam
            logger.debug(f"Example structure: type={type(example)}, len={len(example) if hasattr(example, '__len__') else 'N/A'}")
            for i, item in enumerate(example):
                logger.debug(f"  example[{i}]: type={type(item)}, value={item if item is not None else 'None'}")
        
        # Convert numpy arrays to tensors
        board_state = torch.FloatTensor(example[0])
        
        # MODIFIED: Add player-to-move channel
        board_np = board_state.numpy()
        from hex_ai.data_utils import get_player_to_move_from_board
        
        try:
            player_to_move = get_player_to_move_from_board(board_np)
        except Exception as e:
            # Use default value if we can't determine
            from hex_ai.inference.board_utils import BLUE_PLAYER
            player_to_move = BLUE_PLAYER
        
        # Create player-to-move channel
        player_channel = np.full((board_np.shape[1], board_np.shape[2]), float(player_to_move), dtype=np.float32)
        board_3ch = np.concatenate([board_np, player_channel[None, ...]], axis=0)
        board_state = torch.from_numpy(board_3ch)
        
        # Handle None policy targets (final moves)
        # Note that on the final move of any game there is no next move to predict,
        # so we need to have some placeholder value. As of 2025-07-13, we use
        # all zeroes, generated by upstream code on last moves, when it 
        # is given a None policy label.
        if example[1] is None:
            policy_target = torch.zeros(POLICY_OUTPUT_SIZE, dtype=torch.float32)
            # Only log at very high verbosity levels
            from .config import VERBOSE_LEVEL
            if VERBOSE_LEVEL >= 3:
                logger.debug(f"Using zero tensor for None policy target at index {idx}")
        else:
            policy_target = torch.FloatTensor(example[1])
        
        value_target = torch.FloatTensor([example[2]])
        
        # Ensure all tensors are valid
        if board_state is None or policy_target is None or value_target is None:
            raise ValueError(f"Invalid example at index {idx}")
        
        return board_state, policy_target, value_target





def discover_processed_files_legacy(data_dir: str = "data/processed") -> List[Path]:
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


def estimate_dataset_size_legacy(data_files: List[Path], max_files: Optional[int] = None) -> int:
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


def create_experiment_config_legacy(experiment_name: str,
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





def run_hyperparameter_tuning_legacy(experiments: List[Dict],
                            data_dir: str = "data/processed",
                            results_dir: str = "checkpoints/hyperparameter_tuning",
                            train_ratio: float = 0.8,
                            num_epochs: int = 10,
                            early_stopping_patience: Optional[int] = None,
                            random_seed: Optional[int] = None,
                            max_examples_per_split: Optional[int] = None,
                            experiment_name: Optional[str] = None) -> Dict:
    """
    Run a complete hyperparameter tuning experiment.
    
    Args:
        experiments: List of experiment configurations
        data_dir: Directory containing processed data
        results_dir: Directory to save results
        train_ratio: Ratio of data to use for training
        num_epochs: Number of epochs per experiment
        early_stopping_patience: Early stopping patience
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing all experiment results
    """
    import time
    
    # Setup
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Use experiment name if provided, otherwise use timestamp
    if experiment_name is None:
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Discover data files
    data_files = discover_processed_files_legacy(data_dir)
    
    # Create train/val split (limit files for efficiency)
    max_files_per_split = 5 if max_examples_per_split is not None else None  # Use only 5 files for quick exploration
    logger.info(f"Creating train/val split with max_files_per_split={max_files_per_split}")
    train_files, val_files = create_train_val_split(
        data_files, train_ratio, random_seed, max_files_per_split
    )
    logger.info(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
    logger.info(f"Sample train files: {[f.name for f in train_files[:3]]}")
    logger.info(f"Sample val files: {[f.name for f in val_files[:3]]}")
    
    # Estimate dataset size (use sampling for speed)
    logger.info("Estimating dataset sizes...")
    total_examples = estimate_dataset_size_legacy(data_files, max_files=10)  # Sample 10 files
    train_examples = estimate_dataset_size_legacy(train_files, max_files=5)  # Sample 5 files
    val_examples = estimate_dataset_size_legacy(val_files, max_files=5)     # Sample 5 files
    
    # If max_examples_per_split is specified, limit the data
    if max_examples_per_split is not None:
        logger.info(f"Limiting data to ~{max_examples_per_split:,} examples per split for quick exploration")
        # We'll limit the data in the dataset creation, not here
    
    dataset_info = {
        'total_files': len(data_files),
        'train_files': len(train_files),
        'val_files': len(val_files),
        'total_examples': total_examples,
        'train_examples': train_examples,
        'val_examples': val_examples
    }
    
    logger.info(f"Dataset info: {dataset_info}")
    
    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA GPU: {device_name}")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple MPS GPU")
    else:
        device = "cpu"
        logger.info("Using CPU")
    
    # Save overall configuration
    overall_config = {
        'experiment_name': experiment_name,
        'dataset_info': dataset_info,
        'device': device,
        'num_experiments': len(experiments),
        'train_ratio': train_ratio,
        'num_epochs': num_epochs,
        'early_stopping_patience': early_stopping_patience,
        'random_seed': random_seed,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(results_path / "overall_config.json", "w") as f:
        json.dump(overall_config, f, indent=2)
    
    # Load data once and reuse across experiments
    logger.info(f"\nLoading data once for all experiments...")
    train_dataset = NewProcessedDataset(train_files, shuffle_files=True, max_examples=max_examples_per_split)
    val_dataset = NewProcessedDataset(val_files, shuffle_files=False, max_examples=max_examples_per_split) if val_files else None
    
    logger.info(f"Loaded {len(train_dataset)} training examples")
    if val_dataset:
        logger.info(f"Loaded {len(val_dataset)} validation examples")
    
    # Run experiments
    all_results = []
    total_start_time = time.time()
    
    logger.info(f"\nStarting {len(experiments)} experiments...")
    
    for i, exp_config in enumerate(experiments):
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment {i+1}/{len(experiments)}: {exp_config['experiment_name']}")
        logger.info(f"{'='*60}")
        
        # Add device and max_examples_per_split to experiment config
        exp_config['device'] = device
        if max_examples_per_split is not None:
            exp_config['max_examples_per_split'] = max_examples_per_split
        
        try:
            results = run_hyperparameter_experiment_legacy(
                exp_config,
                train_dataset,
                val_dataset,
                results_path,
                num_epochs,
                early_stopping_patience,
                exp_config['experiment_name'] # Pass experiment_name here
            )
            all_results.append(results)
            
        except Exception as e:
            logger.error(f"Experiment {exp_config['experiment_name']} failed: {e}")
            continue
    
    # Save overall results
    total_time = time.time() - total_start_time
    overall_results = {
        'total_training_time': total_time,
        'num_experiments': len(experiments),
        'successful_experiments': len(all_results),
        'device': device,
        'experiments': all_results
    }
    
    with open(results_path / "overall_results.json", "w") as f:
        json.dump(overall_results, f, indent=2, default=str)
    
    # Create summary CSV with all experiment results
    if all_results:
        create_summary_csv(all_results, results_path)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("HYPERPARAMETER TUNING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total training time: {total_time:.1f}s")
    logger.info(f"Successful experiments: {len(all_results)}/{len(experiments)}")
    
    if all_results:
        best_exp = min(all_results, key=lambda x: x['best_val_loss'])
        logger.info(f"\nBest experiment: {best_exp['experiment_name']}")
        logger.info(f"Best validation loss: {best_exp['best_val_loss']:.6f}")
        logger.info(f"Hyperparameters: {best_exp['hyperparameters']}")
    
    logger.info(f"\nAll results saved to: {results_path}")
    
    return overall_results


def run_hyperparameter_experiment_legacy(experiment_config: Dict,
                                train_dataset: NewProcessedDataset,
                                val_dataset: Optional[NewProcessedDataset],
                                results_dir: Path,
                                num_epochs: int = 10,
                                early_stopping_patience: Optional[int] = None,
                                experiment_name: Optional[str] = None) -> Dict:
    """
    Run a single hyperparameter experiment using pre-loaded datasets.
    
    Args:
        experiment_config: Experiment configuration
        train_dataset: Pre-loaded training dataset
        val_dataset: Pre-loaded validation dataset (can be None)
        results_dir: Directory to save results
        num_epochs: Number of training epochs
        early_stopping_patience: Early stopping patience (None to disable)
        
    Returns:
        Dictionary containing experiment results
    """
    exp_name = experiment_config['experiment_name']
    hyperparams = experiment_config['hyperparameters']
    
    logger.info(f"Starting experiment: {exp_name}")
    logger.info(f"Hyperparameters: {hyperparams}")
    
    # Create experiment directory
    exp_dir = results_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment config
    with open(exp_dir / "config.json", "w") as f:
        json.dump(experiment_config, f, indent=2)
    
    # Create dataloaders from pre-loaded datasets
    logger.info(f"Creating dataloaders from pre-loaded datasets")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hyperparams['batch_size'],
        # NOTE: Performance was good with shuffle=True but shuffling should not
        # be necessary. Also want to align with the non-legacy training.
        shuffle=False, 
        num_workers=0,  # Use 0 to avoid multiprocessing issues
        pin_memory=False  # Disable pin_memory for MPS
    )
    
    val_loader = None
    if val_dataset:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=hyperparams['batch_size'],
            shuffle=False,
            num_workers=0,  # Use 0 to avoid multiprocessing issues
            pin_memory=False  # Disable pin_memory for MPS
        )
    
    # Create model
    model = TwoHeadedResNet(dropout_prob=hyperparams.get('dropout_prob', 0.1))
    device = torch.device(experiment_config['device'])
    model = model.to(device)
    
    # Create trainer
    csv_log_file = exp_dir / "training_metrics.csv"
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=hyperparams['learning_rate'],
        device=device,
        enable_system_analysis=True,
        enable_csv_logging=True,
        experiment_name=experiment_name or exp_name,  # Use provided experiment_name or fall back to exp_name
        policy_weight=hyperparams.get('policy_weight', 0.15),
        value_weight=hyperparams.get('value_weight', 0.85),
        weight_decay=hyperparams.get('weight_decay', 1e-4),
        max_grad_norm=hyperparams.get('max_grad_norm', 20.0)
    )
    
    # Update CSV logger to use experiment-specific file
    if trainer.csv_logger:
        trainer.csv_logger.log_file = csv_log_file
        trainer.csv_logger.experiment_name = experiment_name or exp_name
    
    # Setup early stopping
    early_stopping = None
    if early_stopping_patience is not None:
        early_stopping = EarlyStopping(patience=early_stopping_patience)
    
    # Train
    training_results = trainer.train(
        num_epochs=num_epochs,
        save_dir=str(exp_dir),
        early_stopping=early_stopping
    )
    
    # Extract key metrics
    best_val_loss = min(training_results['val_losses']) if training_results['val_losses'] else float('inf')
    best_train_loss = min(training_results['train_losses']) if training_results['train_losses'] else float('inf')
    final_val_loss = training_results['val_losses'][-1] if training_results['val_losses'] else float('inf')
    final_train_loss = training_results['train_losses'][-1] if training_results['train_losses'] else float('inf')
    
    experiment_results = {
        'experiment_name': exp_name,
        'hyperparameters': hyperparams,
        'best_val_loss': best_val_loss,
        'best_train_loss': best_train_loss,
        'final_val_loss': final_val_loss,
        'final_train_loss': final_train_loss,
        'epochs_trained': len(training_results['train_losses']),
        'early_stopped': training_results.get('early_stopped', False),
        'training_time': training_results.get('total_time', 0),
        'all_metrics': training_results
    }
    
    # Save experiment results
    with open(exp_dir / "experiment_results.json", "w") as f:
        json.dump(experiment_results, f, indent=2, default=str)
    
    logger.info(f"Experiment {exp_name} completed:")
    logger.info(f"  Best val loss: {best_val_loss:.6f}")
    logger.info(f"  Final val loss: {final_val_loss:.6f}")
    logger.info(f"  Epochs trained: {len(training_results['train_losses'])}")
    
    return experiment_results


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


def run_hyperparameter_tuning_current_data(experiments: List[Dict],
                            data_dir: str = "data/processed",
                            results_dir: str = "checkpoints/hyperparameter_tuning",
                            train_ratio: float = 0.8,
                            num_epochs: int = 10,
                            early_stopping_patience: Optional[int] = None,
                            random_seed: Optional[int] = None,
                            max_examples_per_split: Optional[int] = None,
                            experiment_name: Optional[str] = None) -> Dict:
    """
    Run a complete hyperparameter tuning experiment using current data pipeline.
    
    This version uses StreamingProcessedDataset instead of the legacy NewProcessedDataset.
    
    Args:
        experiments: List of experiment configurations
        data_dir: Directory containing processed data
        results_dir: Directory to save results
        train_ratio: Ratio of data to use for training
        num_epochs: Number of epochs per experiment
        early_stopping_patience: Early stopping patience
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing all experiment results
    """
    import time
    from hex_ai.data_pipeline import StreamingProcessedDataset, discover_processed_files, estimate_dataset_size, create_train_val_split
    
    # Setup
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Use experiment name if provided, otherwise use timestamp
    if experiment_name is None:
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Discover data files using current pipeline
    data_files = discover_processed_files(data_dir)
    
    # Create train/val split (limit files for efficiency)
    max_files_per_split = 5 if max_examples_per_split is not None else None  # Use only 5 files for quick exploration
    logger.info(f"Creating train/val split with max_files_per_split={max_files_per_split}")
    train_files, val_files = create_train_val_split(
        data_files, train_ratio, random_seed, max_files_per_split
    )
    logger.info(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
    logger.info(f"Sample train files: {[f.name for f in train_files[:3]]}")
    logger.info(f"Sample val files: {[f.name for f in val_files[:3]]}")
    
    # Estimate dataset size (use sampling for speed)
    logger.info("Estimating dataset sizes...")
    total_examples = estimate_dataset_size(data_files, max_files=10)  # Sample 10 files
    train_examples = estimate_dataset_size(train_files, max_files=5)  # Sample 5 files
    val_examples = estimate_dataset_size(val_files, max_files=5)     # Sample 5 files
    
    # If max_examples_per_split is specified, limit the data
    if max_examples_per_split is not None:
        logger.info(f"Limiting data to ~{max_examples_per_split:,} examples per split for quick exploration")
        # We'll limit the data in the dataset creation, not here
    
    dataset_info = {
        'total_files': len(data_files),
        'train_files': len(train_files),
        'val_files': len(val_files),
        'total_examples': total_examples,
        'train_examples': train_examples,
        'val_examples': val_examples
    }
    
    logger.info(f"Dataset info: {dataset_info}")
    
    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA GPU: {device_name}")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple MPS GPU")
    else:
        device = "cpu"
        logger.info("Using CPU")
    
    # Save overall configuration
    overall_config = {
        'experiment_name': experiment_name,
        'dataset_info': dataset_info,
        'device': device,
        'num_experiments': len(experiments),
        'train_ratio': train_ratio,
        'num_epochs': num_epochs,
        'early_stopping_patience': early_stopping_patience,
        'random_seed': random_seed,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(results_path / "overall_config.json", "w") as f:
        json.dump(overall_config, f, indent=2)
    
    # Load data once and reuse across experiments using current pipeline
    logger.info(f"\nLoading data once for all experiments...")
    train_dataset = StreamingProcessedDataset(train_files, chunk_size=max_examples_per_split or 100000)
    val_dataset = StreamingProcessedDataset(val_files, chunk_size=max_examples_per_split or 100000) if val_files else None
    
    logger.info(f"Created streaming datasets")
    if val_dataset:
        logger.info(f"Created validation dataset")
    
    # Run experiments
    all_results = []
    total_start_time = time.time()
    
    logger.info(f"\nStarting {len(experiments)} experiments...")
    
    for i, exp_config in enumerate(experiments):
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment {i+1}/{len(experiments)}: {exp_config['experiment_name']}")
        logger.info(f"{'='*60}")
        
        # Add device and max_examples_per_split to experiment config
        exp_config['device'] = device
        if max_examples_per_split is not None:
            exp_config['max_examples_per_split'] = max_examples_per_split
        
        try:
            results = run_hyperparameter_experiment_current_data(
                exp_config,
                train_dataset,
                val_dataset,
                results_path,
                num_epochs,
                early_stopping_patience,
                exp_config['experiment_name'] # Pass experiment_name here
            )
            all_results.append(results)
            
        except Exception as e:
            logger.error(f"Experiment {exp_config['experiment_name']} failed: {e}")
            continue
    
    # Save overall results
    total_time = time.time() - total_start_time
    overall_results = {
        'total_training_time': total_time,
        'num_experiments': len(experiments),
        'successful_experiments': len(all_results),
        'device': device,
        'experiments': all_results
    }
    
    with open(results_path / "overall_results.json", "w") as f:
        json.dump(overall_results, f, indent=2, default=str)
    
    # Create summary CSV with all experiment results
    if all_results:
        create_summary_csv(all_results, results_path)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("HYPERPARAMETER TUNING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total training time: {total_time:.1f}s")
    logger.info(f"Successful experiments: {len(all_results)}/{len(experiments)}")
    
    if all_results:
        best_exp = min(all_results, key=lambda x: x['best_val_loss'])
        logger.info(f"\nBest experiment: {best_exp['experiment_name']}")
        logger.info(f"Best validation loss: {best_exp['best_val_loss']:.6f}")
        logger.info(f"Hyperparameters: {best_exp['hyperparameters']}")
    
    logger.info(f"\nAll results saved to: {results_path}")
    
    return overall_results


def run_hyperparameter_experiment_current_data(experiment_config: Dict,
                                train_dataset,
                                val_dataset,
                                results_dir: Path,
                                num_epochs: int = 10,
                                early_stopping_patience: Optional[int] = None,
                                experiment_name: Optional[str] = None) -> Dict:
    """
    Run a single hyperparameter experiment using current data pipeline.
    
    Args:
        experiment_config: Experiment configuration
        train_dataset: Streaming training dataset
        val_dataset: Streaming validation dataset (can be None)
        results_dir: Directory to save results
        num_epochs: Number of training epochs
        early_stopping_patience: Early stopping patience (None to disable)
        
    Returns:
        Dictionary containing experiment results
    """
    exp_name = experiment_config['experiment_name']
    hyperparams = experiment_config['hyperparameters']
    
    logger.info(f"Starting experiment: {exp_name}")
    logger.info(f"Hyperparameters: {hyperparams}")
    
    # Create experiment directory
    exp_dir = results_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment config
    with open(exp_dir / "config.json", "w") as f:
        json.dump(experiment_config, f, indent=2)
    
    # Create dataloaders from streaming datasets
    logger.info(f"Creating dataloaders from streaming datasets")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hyperparams['batch_size'],
        shuffle=True,  # NOTE: Consider setting to False if data is already shuffled
        num_workers=0,  # Use 0 to avoid multiprocessing issues
        pin_memory=False  # Disable pin_memory for MPS
    )
    val_loader = None
    if val_dataset:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=hyperparams['batch_size'],
            shuffle=False,
            num_workers=0,  # Use 0 to avoid multiprocessing issues
            pin_memory=False  # Disable pin_memory for MPS
        )

    # Log config/environment info
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f'hyperparam_tuning_config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    # Create model
    model = TwoHeadedResNet(dropout_prob=hyperparams.get('dropout_prob', 0.1))
    device = torch.device(experiment_config['device'])
    model = model.to(device)
    
    # Create trainer
    csv_log_file = exp_dir / "training_metrics.csv"
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=hyperparams['learning_rate'],
        device=device,
        enable_system_analysis=True,
        enable_csv_logging=True,
        experiment_name=experiment_name or exp_name,  # Use provided experiment_name or fall back to exp_name
        policy_weight=hyperparams.get('policy_weight', 0.15),
        value_weight=hyperparams.get('value_weight', 0.85),
        weight_decay=hyperparams.get('weight_decay', 1e-4),
        max_grad_norm=hyperparams.get('max_grad_norm', 20.0),
        value_learning_rate_factor=hyperparams.get('value_learning_rate_factor', 0.1),
        value_weight_decay_factor=hyperparams.get('value_weight_decay_factor', 5.0)
    )
    
    # Update CSV logger to use experiment-specific file
    if trainer.csv_logger:
        trainer.csv_logger.log_file = csv_log_file
        trainer.csv_logger.experiment_name = experiment_name or exp_name
    
    # Setup early stopping
    early_stopping = None
    if early_stopping_patience is not None:
        early_stopping = EarlyStopping(patience=early_stopping_patience)
    
    ### Extra config/environment logging to compare with non-legacy training
    def log_and_print(msg):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')

    log_and_print(f"**************************************************")
    log_and_print(f"==== Hex AI Hyperparam Tuning Experiment Config (legacy) ====")
    log_and_print(f"**************************************************")
    log_and_print(f"Timestamp: {datetime.now().isoformat()}")
    log_and_print(f"Experiment: {exp_name}")
    log_and_print(f"Model class: {model.__class__.__name__}")
    log_and_print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    try:
        log_and_print(f"Model input channels: {model.input_channels if hasattr(model, 'input_channels') else 'unknown'}")
    except Exception as e:
        log_and_print(f"Model input channels: unknown (error: {e})")
    log_and_print(f"Train DataLoader: batch_size={hyperparams['batch_size']}")
    log_and_print(f"Val DataLoader: batch_size={hyperparams['batch_size']}")
    log_and_print(f"Train dataset size: {len(train_dataset)}")
    log_and_print(f"Val dataset size: {len(val_dataset) if val_dataset else 0}")
    log_and_print(f"Learning rate: {hyperparams['learning_rate']}")
    log_and_print(f"Value learning rate factor: {hyperparams.get('value_learning_rate_factor', 'N/A')}")
    log_and_print(f"Dropout: {hyperparams.get('dropout_prob', 'N/A')}")
    log_and_print(f"Weight decay: {hyperparams.get('weight_decay', 'N/A')}")
    log_and_print(f"Value weight decay factor: {hyperparams.get('value_weight_decay_factor', 'N/A')}")
    log_and_print(f"Policy loss weight: {hyperparams.get('policy_weight', 'N/A')}")
    log_and_print(f"Value loss weight: {hyperparams.get('value_weight', 'N/A')}")
    log_and_print(f"Max grad norm: {hyperparams.get('max_grad_norm', 'N/A')}")
    log_and_print(f"Device: {device}")
    log_and_print(f"Mixed precision: {trainer.mixed_precision.use_mixed_precision}")
    log_and_print(f"Results dir: {exp_dir}")
    log_and_print(f"==========================================")
    ### END: Extra config/environment logging to compare with non-legacy training

    # Train
    training_results = trainer.train(
        num_epochs=num_epochs,
        save_dir=str(exp_dir),
        early_stopping=early_stopping
    )
    
    # Extract key metrics
    best_val_loss = min(training_results['val_losses']) if training_results['val_losses'] else float('inf')
    best_train_loss = min(training_results['train_losses']) if training_results['train_losses'] else float('inf')
    final_val_loss = training_results['val_losses'][-1] if training_results['val_losses'] else float('inf')
    final_train_loss = training_results['train_losses'][-1] if training_results['train_losses'] else float('inf')
    
    experiment_results = {
        'experiment_name': exp_name,
        'hyperparameters': hyperparams,
        'best_val_loss': best_val_loss,
        'best_train_loss': best_train_loss,
        'final_val_loss': final_val_loss,
        'final_train_loss': final_train_loss,
        'epochs_trained': len(training_results['train_losses']),
        'early_stopped': training_results.get('early_stopped', False),
        'training_time': training_results.get('total_time', 0),
        'all_metrics': training_results
    }
    
    # Save experiment results
    with open(exp_dir / "experiment_results.json", "w") as f:
        json.dump(experiment_results, f, indent=2, default=str)
    
    logger.info(f"Experiment {exp_name} completed:")
    logger.info(f"  Best val loss: {best_val_loss:.6f}")
    logger.info(f"  Final val loss: {final_val_loss:.6f}")
    logger.info(f"  Epochs trained: {len(training_results['train_losses'])}")
    
    return experiment_results 