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
from .training import Trainer
from .config import BOARD_SIZE, POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE, DEFAULT_POOL_SIZE, DEFAULT_REFILL_THRESHOLD, DEFAULT_MAX_MEMORY_GB
from hex_ai.mini_epoch_orchestrator import MiniEpochOrchestrator
from hex_ai.data_pipeline import discover_training_data_files_all
from hex_ai.error_handling import GracefulShutdownRequested
from hex_ai.validation_defaults import resolve_validation_config, log_validation_summary

logger = logging.getLogger(__name__)


def create_datasets(data_dirs: List[str], 
                   shard_ranges: List[str],
                   validation_dirs: List[str],
                   validation_shard_ranges: List[str],
                   train_ratio: float = 0.8,
                   max_examples_unaugmented: Optional[int] = None,
                   max_validation_examples: Optional[int] = None,
                   batch_size: int = 256,
                   pool_size: int = DEFAULT_POOL_SIZE,
                   refill_threshold: int = DEFAULT_REFILL_THRESHOLD,
                   max_memory_gb: float = DEFAULT_MAX_MEMORY_GB,
                   random_seed: Optional[int] = None,
                   verbose: int = 2,
                   shutdown_handler=None):
    """
    Create DataLoader objects from StreamingMixedShardDataset for train and val sets.
    Returns (train_loader, val_loader).
    """
    from hex_ai.data_pipeline import StreamingMixedShardDataset
    
    try:
        logger.info("Creating training dataset...")
        train_dataset = StreamingMixedShardDataset(
            data_dirs=data_dirs,
            shard_ranges=shard_ranges,
            pool_size=pool_size,
            refill_threshold=refill_threshold,
            max_memory_gb=max_memory_gb,
            enable_augmentation=True,
            max_examples_unaugmented=max_examples_unaugmented,
            verbose=verbose,
            random_seed=random_seed,
            shutdown_handler=shutdown_handler
        )
        logger.info("Training dataset created. Done.")
        
        if max_validation_examples and validation_dirs:
            logger.info("Creating validation dataset...")
            val_dataset = StreamingMixedShardDataset(
                data_dirs=validation_dirs,
                shard_ranges=validation_shard_ranges,
                pool_size=pool_size,
                refill_threshold=refill_threshold,
                max_memory_gb=max_memory_gb,
                enable_augmentation=False,  # Validation dataset is not augmented
                max_examples_unaugmented=max_validation_examples,
                verbose=verbose,
                random_seed=random_seed,
                is_validation=True,  # Enable validation-specific behavior
                shutdown_handler=shutdown_handler
            )
            logger.info("Validation dataset created. Done.")
        else:
            val_dataset = None
        
        # Log data summary after shard discovery
        train_summary = train_dataset.get_data_summary()
        logger.info("=" * 60)
        logger.info("TRAINING DATA SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Estimated total positions: ~{train_summary['estimated_total_positions']:,}")
        logger.info(f"Estimated total games: ~{train_summary['estimated_total_games']:,}")
        logger.info(f"Total shards: {train_summary['total_shards']}")
        logger.info(f"Data directories: {train_summary['directories']}")
        if max_examples_unaugmented is not None and train_summary['estimated_total_positions'] > 0:
            estimated_positions = train_summary['estimated_total_positions']
            coverage = min(1.0, max_examples_unaugmented / estimated_positions)
            logger.info(
                f"Per-epoch training cap: {max_examples_unaugmented:,} unaugmented samples "
                f"(~{coverage:.1%} of estimated available positions)"
            )
            if coverage < 0.5:
                logger.warning(
                    "Per-epoch cap is significantly below estimated available training data. "
                    "Increase max_examples_unaugmented/--max_samples if you want longer epochs."
                )
        logger.info("=" * 60)
        
        # Log validation data summary if validation dataset exists
        if val_dataset is not None:
            val_summary = val_dataset.get_data_summary()
            logger.info("VALIDATION DATA SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Estimated total positions: ~{val_summary['estimated_total_positions']:,}")
            logger.info(f"Estimated total games: ~{val_summary['estimated_total_games']:,}")
            logger.info(f"Total shards: {val_summary['total_shards']}")
            logger.info(f"Data directories: {val_summary['directories']}")
            logger.info("=" * 60)
        
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
    override_checkpoint_hyperparameters: bool = False,
    max_mini_epochs: Optional[int] = None,
    resume_mode: str = "next_epoch",
    target_end_epoch: Optional[int] = None,
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
        max_mini_epochs: Optional per-process mini-epoch cap for chunked restarts
        resume_mode: Resume policy:
            - "next_epoch": resume from next epoch (legacy behavior)
            - "same_epoch": resume from same epoch and skip completed mini-epochs
        target_end_epoch: Optional absolute end epoch number (1-based, inclusive)
    """
    # Determine checkpoint path and start epoch
    checkpoint_path = None
    start_epoch = 0
    start_mini_epoch = 0
    
    if resume_from:
        checkpoint_path = Path(resume_from)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {resume_from}")
        
        # Extract epoch + mini from checkpoint filename
        # Expected format: epoch{N}_mini{M}.pt.gz
        match = re.search(r'epoch(\d+)_mini(\d+)', checkpoint_path.name)
        if match:
            completed_epoch = int(match.group(1))
            completed_mini = int(match.group(2))

            if resume_mode == "next_epoch":
                # Legacy behavior: continue at next epoch boundary.
                start_epoch = completed_epoch
                start_mini_epoch = 0
            elif resume_mode == "same_epoch":
                # Chunked-restart behavior: continue within the same epoch.
                # Orchestrator epoch loop is 0-based; checkpoint filenames are 1-based.
                start_epoch = max(0, completed_epoch - 1)
                start_mini_epoch = max(0, completed_mini)
            else:
                raise ValueError(f"Unsupported resume_mode: {resume_mode}")

            if target_end_epoch is not None:
                if target_end_epoch < 1:
                    raise ValueError(
                        f"target_end_epoch must be >= 1, got {target_end_epoch}"
                    )
                if target_end_epoch <= start_epoch:
                    logger.info(
                        f"Training already complete for target_end_epoch={target_end_epoch}; "
                        f"start_epoch={start_epoch}, checkpoint={checkpoint_path}"
                    )
                    return {
                        'total_batches': 0,
                        'epochs_completed': 0,
                        'mini_epochs_trained': 0,
                        'stopped_due_to_max_mini_epochs': False,
                        'already_complete': True,
                    }
                num_epochs = target_end_epoch
            else:
                # Adjust num_epochs to ensure we train for the full requested duration.
                # If we want 3 total epochs and completed 2, we need to train for 3 more epochs (2, 3, 4).
                num_epochs = completed_epoch + num_epochs

            logger.info(
                f"Resuming from checkpoint {checkpoint_path} "
                f"(completed epoch={completed_epoch}, mini={completed_mini}, "
                f"resume_mode={resume_mode}, start_epoch={start_epoch}, start_mini={start_mini_epoch}, "
                f"end_epoch={num_epochs})"
            )
        else:
            raise ValueError(
                f"Could not extract epoch/mini from checkpoint filename: {checkpoint_path.name}"
            )
    
    # Create model and trainer
    # Filter hyperparameters for model vs trainer
    model_params = {}
    trainer_params = {}
    
    # Model parameters - expect only the correct parameter names
    required_model_params = {'num_blocks', 'trunk_channels'}
    for param in required_model_params:
        if param in exp_config['hyperparameters']:
            model_params[param] = exp_config['hyperparameters'][param]
        else:
            raise ValueError(f"Missing required model parameter: {param}. "
                           f"Available hyperparameters: {list(exp_config['hyperparameters'].keys())}")
    
    # Handle legacy dropout_prob parameter (no longer used in model)
    if 'dropout_prob' in exp_config['hyperparameters']:
        # Log a warning but don't fail - this is a legacy parameter
        logger.warning(f"Legacy parameter 'dropout_prob' is no longer used in the model architecture. "
                      f"Value {exp_config['hyperparameters']['dropout_prob']} will be ignored.")
    
    # Check for legacy parameter names and provide clear error
    legacy_params = {'resnet_depth'}
    for legacy_param in legacy_params:
        if legacy_param in exp_config['hyperparameters']:
            raise ValueError(f"Legacy parameter '{legacy_param}' is no longer supported. "
                           f"Use 'num_blocks' instead.")
    
    # Note: use_value_bottleneck is no longer a parameter in the new KataGo-inspired architecture
    # The value head now has a fixed bottleneck design
    
    # Trainer parameters (everything else except batch_size, model parameters, and legacy parameters)
    model_param_keys = {'num_blocks', 'trunk_channels'}  # Keys that map to model parameters
    legacy_params = {'dropout_prob'}  # Legacy parameters that should be ignored
    trainer_params = {k: v for k, v in exp_config['hyperparameters'].items() 
                     if k not in model_param_keys and k not in legacy_params and k != 'batch_size'}
    
    model = TwoHeadedResNet(**model_params).to(device)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        run_timestamp=run_timestamp,
        shutdown_handler=shutdown_handler,
        **trainer_params
    )
    
    # Load checkpoint if resuming
    if checkpoint_path:
        trainer.load_checkpoint(checkpoint_path, override_checkpoint_hyperparameters=override_checkpoint_hyperparameters)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    # Use results_path directly (no extra directory nesting)
    experiment_name = exp_config.get('experiment_name', 'unknown_experiment')
    results_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using checkpoint directory: {results_path}")
    
    # Create orchestrator
    orchestrator = MiniEpochOrchestrator(
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=results_path,
        num_epochs=num_epochs,
        mini_epoch_samples=mini_epoch_samples,
        start_epoch=start_epoch,
        start_mini_epoch=start_mini_epoch,
        max_mini_epochs=max_mini_epochs,
        shutdown_handler=shutdown_handler
    )
    
    # Run training
    try:
        result = orchestrator.run()
        return result
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # TEMPORARY: Cleanup enhanced NaN detection logging
        # TODO: Remove after confirming training stability (3+ successful runs without NaN issues)
        from hex_ai.nan_debug_utils import cleanup_global_first_nan_detector
        cleanup_global_first_nan_detector()

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



def save_experiment_metadata(
    results_path: Path, 
    experiment_name: str, 
    hyperparameters: Dict, 
    training_config: Dict
) -> None:
    """
    Save detailed metadata about the experiment.
    
    Args:
        results_path: Path to results directory
        experiment_name: Name of the experiment
        hyperparameters: Model hyperparameters
        training_config: Training configuration
        
    Raises:
        Exception: If metadata saving fails (no silent failures)
    """
    from datetime import datetime
    
    try:
        metadata = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'hyperparameters': hyperparameters,
            'training_config': training_config
        }
        
        # Save metadata directly in results_path (no extra directory)
        metadata_file = results_path / "experiment_metadata.json"
        results_path.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved experiment metadata to {metadata_file}")
        
    except Exception as e:
        error_msg = f"Failed to save experiment metadata to {results_path}: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def save_overall_results(results_path, overall_results):
    """
    Save the overall results dict to disk as overall_results.json.
    
    Args:
        results_path: Path to results directory
        overall_results: Results dictionary to save
    """
    
    with open(results_path / "overall_results.json", "w") as f:
        json.dump(overall_results, f, indent=2, default=str)

def run_hyperparameter_tuning_current_data(
    experiments: List[Dict],
    data_dirs: Union[str, List[str]],  # Single directory or list of directories
    validation_dirs: Optional[List[str]] = None,  # Override validation directories
    validation_shard_ranges: Optional[List[str]] = None,  # Override validation ranges
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
    resume_from: Optional[str] = None,  # Resume from checkpoint file
    shard_ranges: Optional[List[str]] = None,  # Shard ranges for each directory (e.g., ["251-300", "all"])
    shuffle_shards: bool = True,  # Control whether to shuffle data shards
    pool_size: int = DEFAULT_POOL_SIZE,  # Pool size for mixed dataset
    refill_threshold: int = DEFAULT_REFILL_THRESHOLD,  # Refill threshold for mixed dataset
    max_memory_gb: float = DEFAULT_MAX_MEMORY_GB,  # Memory limit for mixed dataset
    verbose: int = 2,  # Verbose level (2=default, 3=detailed pool/shard info)
    shutdown_handler=None,
    run_timestamp: Optional[str] = None,
    override_checkpoint_hyperparameters: bool = False,
    max_mini_epochs: Optional[int] = None,
    resume_mode: str = "next_epoch",
    target_end_epoch: Optional[int] = None,
) -> Dict:
    """
    Orchestrates the full hyperparameter sweep using modular helpers for data, dataset, and experiment logic.
    
    Args:
        experiments: List of experiment configurations
        data_dirs: Single data directory (str) or list of data directories (List[str])
        validation_dirs: Override validation directories (defaults to hardcoded values)
        validation_shard_ranges: Override validation shard ranges (defaults to hardcoded values)
        results_dir: Directory to save results
        train_ratio: Ratio for train/val split
        num_epochs: Number of training epochs
        early_stopping_patience: Early stopping patience
        random_seed: Random seed for reproducibility
        max_examples_unaugmented: Maximum training examples
        max_validation_examples: Maximum validation examples
        experiment_name: Optional experiment name
        enable_augmentation: Whether to enable data augmentation
        mini_epoch_samples: Number of samples per mini-epoch
        resume_from: Optional path to resume from (file or directory)
        shard_ranges: Optional list of shard ranges for each directory (e.g., ["251-300", "all"])
        shuffle_shards: Whether to shuffle data shards before train/val split (default: True)
        pool_size: Target number of positions to maintain in memory (default: 1M)
        refill_threshold: Refill pool when it drops below this many positions (default: 750K)
        max_memory_gb: Maximum memory usage before graceful shutdown (default: 5.0)
        verbose: Verbose level (2=default, 3=detailed pool/shard info)
        shutdown_handler: Shutdown handler for graceful termination
        run_timestamp: Optional timestamp for the run
        override_checkpoint_hyperparameters: Whether to override checkpoint hyperparameters
        max_mini_epochs: Optional per-process mini-epoch cap for chunked restarts
        resume_mode: Resume mode for checkpoint continuation ("next_epoch" or "same_epoch")
        target_end_epoch: Optional absolute end epoch number (1-based, inclusive)
        
    Returns:
        Dictionary containing overall results
    """
    # Handle backward compatibility: convert single directory to list
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    
    if max_validation_examples is None:
        max_validation_examples = max_examples_unaugmented
    
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    if experiment_name is None:
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"\nGrabbing data from {len(data_dirs)} directories with random seed {random_seed}...")
    
    # Validate directories and shard ranges
    if shard_ranges is None:
        shard_ranges = ["all"] * len(data_dirs)
    
    # Validate that we have the right number of shard ranges
    if len(shard_ranges) != len(data_dirs):
        raise ValueError(f"Number of shard_ranges ({len(shard_ranges)}) must match number of data_dirs ({len(data_dirs)})")
    
    # Resolve validation configuration
    resolved_validation_dirs, resolved_validation_ranges = resolve_validation_config(
        validation_dirs=validation_dirs,
        validation_shard_ranges=validation_shard_ranges,
        no_validation=False  # We don't support no_validation in this function
    )
    
    # Validate validation configuration if not empty
    if resolved_validation_dirs and resolved_validation_ranges:
        if len(resolved_validation_dirs) != len(resolved_validation_ranges):
            raise ValueError(f"Number of validation directories ({len(resolved_validation_dirs)}) must match number of validation shard ranges ({len(resolved_validation_ranges)})")
    
    # Validate training shard ranges
    from hex_ai.data_collection import validate_shard_ranges
    validate_shard_ranges(data_dirs, shard_ranges, context_name="training", logger=logger)
    
    # Validate validation shard ranges if not empty
    if resolved_validation_dirs and resolved_validation_ranges:
        validate_shard_ranges(resolved_validation_dirs, resolved_validation_ranges, context_name="validation", logger=logger)
    
    # Log validation summary
    log_validation_summary(resolved_validation_dirs, resolved_validation_ranges)
    
    # Get batch_size from hyperparameters for the first experiment (they should all be the same)
    batch_size = experiments[0]['hyperparameters'].get('batch_size', 256) if experiments else 256
    
    # Create datasets using the new mixed shard approach
    logger.info(f"Using StreamingMixedShardDataset with pool_size={pool_size:,}, refill_threshold={refill_threshold:,}")
    logger.info("Creating training and validation datasets...")
    train_loader, val_loader = create_datasets(
        data_dirs=data_dirs,
        shard_ranges=shard_ranges,
        validation_dirs=resolved_validation_dirs,
        validation_shard_ranges=resolved_validation_ranges,
        train_ratio=train_ratio,
        max_examples_unaugmented=max_examples_unaugmented,
        max_validation_examples=max_validation_examples,
        batch_size=batch_size,
        pool_size=pool_size,
        refill_threshold=refill_threshold,
        max_memory_gb=max_memory_gb,
        random_seed=random_seed,
        verbose=verbose,
        shutdown_handler=shutdown_handler
    )
    logger.info("Datasets created. Done.")
    
    # Log dataset information
    logger.info(f"\nStreaming mixed dataset: up to {max_examples_unaugmented} training examples, up to {max_validation_examples} validation examples.")
    
    if train_loader is None:
        return {'error': 'Failed to create datasets'}
    
    device = select_device()
    logger.info(f"Using device {device}...")
    logger.info(f"Starting {len(experiments)} experiments...")
    logger.info("Initializing training...")

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
                override_checkpoint_hyperparameters=override_checkpoint_hyperparameters,
                max_mini_epochs=max_mini_epochs,
                resume_mode=resume_mode,
                target_end_epoch=target_end_epoch,
            )
            
            # Save experiment metadata with data source information
            save_experiment_metadata(
                results_path,
                exp_config['experiment_name'],
                exp_config['hyperparameters'],
                {
                    'num_epochs': num_epochs,
                    'mini_epoch_samples': mini_epoch_samples,
                    'max_examples_unaugmented': max_examples_unaugmented,
                    'max_validation_examples': max_validation_examples,
                    'enable_augmentation': enable_augmentation,
                    'train_ratio': train_ratio,
                    'random_seed': random_seed,
                    'resumed_from': resume_from,
                    'max_mini_epochs': max_mini_epochs,
                    'resume_mode': resume_mode,
                    'target_end_epoch': target_end_epoch,
                }
            )
            
            all_results.append(result)
            
        except GracefulShutdownRequested:
            logger.info(f"Experiment {exp_config['experiment_name']} interrupted due to graceful shutdown request")
            raise
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
    save_overall_results(results_path, overall_results)
    logger.info(f"\nOverall results saved to {results_path}.")
    
    return overall_results 
