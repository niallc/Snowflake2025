"""
Training orchestration module for Hex AI.

This module provides high-level training orchestration including hyperparameter tuning,
experiment management, and data pipeline coordination.
"""

import csv
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import torch
import torch.nn as nn
import numpy as np
import gzip
import pickle
import logging
import random
import re

from .models import TwoHeadedResNet
from .training import Trainer, EarlyStopping
from .config import BOARD_SIZE, POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE
from hex_ai.mini_epoch_orchestrator import MiniEpochOrchestrator
from hex_ai.data_pipeline import StreamingSequentialShardDataset, discover_processed_files, create_train_val_split, estimate_dataset_size

logger = logging.getLogger(__name__)


def create_datasets(train_files, val_files, max_examples_unaugmented,
                    max_validation_examples, batch_size=256, log_shard_transitions=True):
    """
    Create DataLoader objects from StreamingSequentialShardDataset for train and val sets.
    Returns (train_loader, val_loader).
    """
    try:
        train_dataset = StreamingSequentialShardDataset(
            train_files, 
            enable_augmentation=True, 
            max_examples_unaugmented=max_examples_unaugmented,
            log_shard_transitions=log_shard_transitions
        )
        val_dataset = StreamingSequentialShardDataset(
            val_files, 
            enable_augmentation=False, # Validation dataset is not augmented
            max_examples_unaugmented=max_validation_examples,
            log_shard_transitions=log_shard_transitions
        ) if val_files else None
        
        # Create DataLoaders from the datasets
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size,
            num_workers=0,  # Streaming datasets don't work well with multiple workers
            pin_memory=False  # Streaming datasets don't benefit from pin_memory
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=batch_size,
                num_workers=0,
                pin_memory=False
            )
    except Exception as e:
        logger.error(f"Failed to create training/validation dataloaders: {e}")
        raise
    return train_loader, val_loader

def find_latest_checkpoint_for_epoch(experiment_dir: Path, target_epoch: int) -> Optional[Path]:
    """
    Find the latest checkpoint file for a specific epoch.
    
    Args:
        experiment_dir: Directory containing checkpoint files
        target_epoch: The epoch to find checkpoints for
        
    Returns:
        Path to the latest checkpoint file for that epoch, or None if not found
    """
    if not experiment_dir.exists():
        return None
    
    # Look for checkpoint files matching the pattern epoch{epoch}_mini{mini}.pt*
    pattern = f"epoch{target_epoch}_mini*.pt*"
    checkpoint_files = list(experiment_dir.glob(pattern))
    
    if not checkpoint_files:
        return None
    
    # Return the latest one (highest mini-epoch number)
    return max(checkpoint_files, key=lambda f: f.name)


def run_single_experiment(
    exp_config, 
    train_loader, 
    val_loader, 
    results_path, 
    num_epochs, 
    mini_epoch_samples, 
    device, 
    resume_from: Optional[str] = None,
    shutdown_handler=None,
    run_timestamp: Optional[str] = None,
    override_checkpoint_hyperparameters: bool = False
):
    """
    Run a single experiment: instantiate Trainer, Orchestrator, and run training.
    Returns a result dict (can be expanded later).
    
    Args:
        exp_config: Experiment configuration
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        results_path: Path to save results
        num_epochs: Number of training epochs
        mini_epoch_samples: Number of samples per mini-epoch
        device: Device to use for training
        resume_from: Path to checkpoint file to resume from
        shutdown_handler: Handler for graceful shutdown
        run_timestamp: Optional timestamp for the run
        override_checkpoint_hyperparameters: If True, reset optimizer state to use current hyperparameters
                                           instead of checkpoint hyperparameters
    """
    # Determine checkpoint path and start epoch
    checkpoint_path = None
    start_epoch = 0
    
    if resume_from:
        checkpoint_path = Path(resume_from)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {resume_from}")
        
        # Extract epoch from checkpoint filename
        # Expected format: epoch{N}_mini{M}.pt.gz
        match = re.search(r'epoch(\d+)_mini', checkpoint_path.name)
        if match:
            completed_epoch = int(match.group(1))
            start_epoch = completed_epoch  # Start from the completed epoch (will continue from where we left off)
            # Adjust num_epochs to ensure we train for the full requested duration
            # If we want 3 total epochs and completed 2, we need to train for 3 more epochs (2, 3, 4)
            num_epochs = completed_epoch + num_epochs
            logger.info(f"Resuming from epoch {start_epoch} using checkpoint: {checkpoint_path}")
        else:
            raise ValueError(f"Could not extract epoch from checkpoint filename: {checkpoint_path.name}")
    
    # Create model and trainer
    # Filter hyperparameters for model vs trainer
    model_params = {}
    trainer_params = {}
    
    # Model parameters
    if 'resnet_depth' in exp_config['hyperparameters']:
        model_params['resnet_depth'] = exp_config['hyperparameters']['resnet_depth']
    if 'dropout_prob' in exp_config['hyperparameters']:
        model_params['dropout_prob'] = exp_config['hyperparameters']['dropout_prob']
    
    # Trainer parameters (everything else except batch_size which is used for DataLoader creation)
    trainer_params = {k: v for k, v in exp_config['hyperparameters'].items() 
                     if k not in model_params and k != 'batch_size'}
    
    model = TwoHeadedResNet(**model_params).to(device)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        run_timestamp=run_timestamp,
        **trainer_params
    )
    
    # Load checkpoint if resuming
    if checkpoint_path:
        trainer.load_checkpoint(checkpoint_path, override_checkpoint_hyperparameters=override_checkpoint_hyperparameters)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    # Create experiment-specific checkpoint directory
    experiment_name = exp_config.get('experiment_name', 'unknown_experiment')
    experiment_checkpoint_dir = results_path / experiment_name
    experiment_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created experiment checkpoint directory: {experiment_checkpoint_dir}")
    
    # Create orchestrator
    orchestrator = MiniEpochOrchestrator(
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=experiment_checkpoint_dir,
        num_epochs=num_epochs,
        mini_epoch_samples=mini_epoch_samples,
        start_epoch=start_epoch
    )
    
    # Run training
    try:
        result = orchestrator.run()
        return result
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def select_device():
    """
    Select the best available device (cuda, mps, or cpu).
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def prepare_experiment_config(exp_config, device, max_examples_unaugmented, max_validation_examples, enable_augmentation):
    """
    Update experiment config with device, augmentation, and max_examples settings.
    """
    exp_config['device'] = device
    if max_examples_unaugmented is not None:
        exp_config['max_examples_unaugmented'] = max_examples_unaugmented
    if max_validation_examples is not None:
        exp_config['max_validation_examples'] = max_validation_examples
    exp_config['enable_augmentation'] = enable_augmentation
    return exp_config

def discover_and_split_multiple_data(
    data_dirs: List[str], 
    train_ratio: float = 0.8, 
    random_seed: Optional[int] = None,
    skip_files: Optional[List[int]] = None
) -> Tuple[List[Path], List[Path], List[Path], List[Dict]]:
    """
    Discover and split data from multiple directories.
    
    Args:
        data_dirs: List of data directories
        train_ratio: Ratio for train/val split
        random_seed: Random seed for reproducible splits
        fail_fast: Whether to fail on first error
        skip_files: Optional list of files to skip from each directory (one value per directory)
        
    Returns:
        Tuple of (train_files, val_files, all_files, data_source_info)
    """
    
    all_data_files = []
    data_source_info = []
    
    # Validate skip_files if provided
    if skip_files is not None:
        if len(skip_files) != len(data_dirs):
            raise ValueError(f"Number of skip_files values ({len(skip_files)}) must match number of data directories ({len(data_dirs)})")
    else:
        # Use 0 skip files for all directories if not provided
        skip_files = [0] * len(data_dirs)
    
    # Discover files from each directory
    for i, data_dir in enumerate(data_dirs):
        try:
            data_files = discover_processed_files(data_dir, skip_files=skip_files[i])
        except Exception as e:
            logger.error(f"Failed to discover data in {data_dir}: {e}")
            raise
        
        if not data_files:
            logger.warning(f"No data files found in {data_dir}")
            continue
        
        # Estimate dataset size for this directory
        try:
            total_examples = estimate_dataset_size(data_files)
        except Exception as e:
            logger.warning(f"Failed to estimate dataset size for {data_dir}: {e}")
            total_examples = 0
        
        # Add files to the combined list
        all_data_files.extend(data_files)
        
        # Record data source information
        data_source_info.append({
            'directory': data_dir,
            'files_count': len(data_files),
            'total_examples': total_examples,
            'skip_files': skip_files[i]
        })
        
        logger.info(f"Added {len(data_files)} files from {data_dir} (examples: {total_examples:,}, skipped: {skip_files[i]})")
    
    if not all_data_files:
        raise ValueError("No data files found in any of the specified directories")
    
    # Create train/val split across all files
    train_files, val_files = create_train_val_split(
        all_data_files, train_ratio, random_seed, max_files_per_split=None
    )
    
    logger.info(f"Final split: {len(train_files)} train files, {len(val_files)} validation files")
    
    return train_files, val_files, all_data_files, data_source_info


def save_experiment_metadata(
    results_path: Path, 
    experiment_name: str, 
    data_source_info: List[Dict], 
    hyperparameters: Dict, 
    training_config: Dict
) -> None:
    """
    Save detailed metadata about the experiment including data sources.
    
    Args:
        results_path: Path to results directory
        experiment_name: Name of the experiment
        data_source_info: Information about data sources used
        hyperparameters: Model hyperparameters
        training_config: Training configuration
    """
    from datetime import datetime
    
    metadata = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'hyperparameters': hyperparameters,
        'training_config': training_config,
        'data_sources': data_source_info,
        'total_examples': sum(src['total_examples'] for src in data_source_info)
    }
    
    metadata_file = results_path / experiment_name / "experiment_metadata.json"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Saved experiment metadata to {metadata_file}")


def save_overall_results(results_path, overall_results, data_source_info=None):
    """
    Save the overall results dict to disk as overall_results.json.
    
    Args:
        results_path: Path to results directory
        overall_results: Results dictionary to save
        data_source_info: Optional data source information to include
    """
    if data_source_info:
        overall_results['data_sources'] = data_source_info
    
    with open(results_path / "overall_results.json", "w") as f:
        json.dump(overall_results, f, indent=2, default=str)

# Refactored main function

def run_hyperparameter_tuning_current_data(
    experiments: List[Dict],
    data_dirs: Union[str, List[str]],  # Updated to support multiple directories
    results_dir: str = "checkpoints/hyperparameter_tuning",
    train_ratio: float = 0.8,
    num_epochs: int = 10,
    early_stopping_patience: Optional[int] = None,
    random_seed: Optional[int] = None,
    max_examples_unaugmented: Optional[int] = None,
    max_validation_examples: Optional[int] = None,
    experiment_name: Optional[str] = None,
    enable_augmentation: bool = True,
    mini_epoch_samples: int = 128000,
    resume_from: Optional[str] = None,  # New: Resume from checkpoint file
    skip_files: Optional[List[int]] = None,  # New: Skip first N files from each directory
    shutdown_handler=None,
    run_timestamp: Optional[str] = None,
    override_checkpoint_hyperparameters: bool = False
) -> Dict:
    """
    Orchestrates the full hyperparameter sweep using modular helpers for data, dataset, and experiment logic.
    
    Args:
        experiments: List of experiment configurations
        data_dirs: Single data directory (str) or list of data directories (List[str])
        data_weights: Optional weights for each data directory (must match number of data_dirs)
        results_dir: Directory to save results
        train_ratio: Ratio for train/val split
        num_epochs: Number of training epochs
        early_stopping_patience: Early stopping patience
        random_seed: Random seed for reproducibility
        max_examples_unaugmented: Maximum training examples
        max_validation_examples: Maximum validation examples
        experiment_name: Optional experiment name
        enable_augmentation: Whether to enable data augmentation
        fail_fast: Whether to fail on first error
        mini_epoch_batches: Number of mini-epoch batches
        resume_from: Optional path to resume from (file or directory)
        resume_epoch: Optional epoch to resume from (only used with directory)
        shutdown_handler: Shutdown handler for graceful termination
        
    Returns:
        Dictionary containing overall results
    """
    # Handle backward compatibility: convert single directory to list
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
        # data_weights = None  # Single directory doesn't need weights
        skip_files = None # Single directory doesn't need per-directory skip
    
    if max_validation_examples is None:
        max_validation_examples = max_examples_unaugmented
    
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    if experiment_name is None:
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"\nGrabbing data from {len(data_dirs)} directories with random seed {random_seed}...")
    
    # Use new multi-directory data discovery
    train_files, val_files, data_files, data_source_info = discover_and_split_multiple_data(
        data_dirs, train_ratio, random_seed, skip_files=skip_files
    )
    
    if train_files is None or val_files is None:
        return {'error': 'Failed to discover or split data'}
    
    # Get batch_size from hyperparameters for the first experiment (they should all be the same)
    batch_size = experiments[0]['hyperparameters'].get('batch_size', 256) if experiments else 256
    
    train_loader, val_loader = create_datasets(train_files, val_files, max_examples_unaugmented, max_validation_examples, batch_size)
    
    # Remove or guard len() usage for streaming datasets
    logger.info(f"\nStreaming training dataset: {len(train_files)} files, up to {max_examples_unaugmented} examples; validation: {len(val_files)} files, up to {max_validation_examples} examples.")
    
    if train_loader is None:
        return {'error': 'Failed to create datasets'}
    
    device = select_device()
    logger.info(f"Using device {device}...")
    logger.info(f"Starting {len(experiments)} experiments...")

    all_results = []
    total_start_time = time.time()
    
    for i, exp_config in enumerate(experiments):
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment {i+1}/{len(experiments)}: {exp_config['experiment_name']}")
        logger.info(f"{'='*60}")
        
        exp_config = prepare_experiment_config(exp_config, device, max_examples_unaugmented, max_validation_examples, enable_augmentation)
        
        try:
            result = run_single_experiment(
                exp_config,
                train_loader,
                val_loader,
                results_path,
                num_epochs,
                mini_epoch_samples,
                device,
                resume_from=resume_from,
                shutdown_handler=shutdown_handler,
                run_timestamp=run_timestamp,
                override_checkpoint_hyperparameters=override_checkpoint_hyperparameters
            )
            
            # Save experiment metadata with data source information
            save_experiment_metadata(
                results_path,
                exp_config['experiment_name'],
                data_source_info,
                exp_config['hyperparameters'],
                {
                    'num_epochs': num_epochs,
                    'mini_epoch_samples': mini_epoch_samples,
                    'max_examples_unaugmented': max_examples_unaugmented,
                    'max_validation_examples': max_validation_examples,
                    'enable_augmentation': enable_augmentation,
                    'train_ratio': train_ratio,
                    'random_seed': random_seed,
                    'resumed_from': resume_from
                }
            )
            
            all_results.append(result)
            
        except Exception as e:
            logger.error(f"Experiment {exp_config['experiment_name']} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.error(f"Fail-fast mode enabled: stopping sweep after failure in experiment {exp_config['experiment_name']}")
            raise
    
    total_time = time.time() - total_start_time
    overall_results = {
        'total_training_time': total_time,
        'num_experiments': len(experiments),
        'successful_experiments': len(all_results),
        'experiments': all_results
    }
    
    logger.info(f"\nSaving overall results to {results_path}...")
    save_overall_results(results_path, overall_results, data_source_info)
    logger.info(f"\nOverall results saved to {results_path}.")
    
    return overall_results 