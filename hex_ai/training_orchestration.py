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

from .models import TwoHeadedResNet
from .training import Trainer, EarlyStopping
from .config import BOARD_SIZE, POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE
from hex_ai.mini_epoch_orchestrator import MiniEpochOrchestrator
from hex_ai.data_pipeline import StreamingSequentialShardDataset, discover_processed_files, create_train_val_split

logger = logging.getLogger(__name__)


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


def create_summary_csv(experiment_results: List[Dict], results_path: Path):
    """
    Create a summary CSV file with all experiment results.
    
    Args:
        experiment_results: List of experiment result dictionaries
        results_path: Path to save the summary CSV
    """
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


def discover_and_split_data(data_dir, train_ratio, random_seed, fail_fast):
    """
    Discover processed files and split into train/val sets.
    Returns (train_files, val_files, all_files).
    """
    try:
        data_files = discover_processed_files(data_dir)
    except Exception as e:
        logger.error(f"Failed to discover processed files in {data_dir}: {e}")
        if fail_fast:
            raise
        else:
            return None, None, None
    try:
        logger.info(f"Creating train/val split (no file limit with streaming)")
        train_files, val_files = create_train_val_split(
            data_files, train_ratio, random_seed, max_files_per_split=None
        )
    except Exception as e:
        logger.error(f"Failed to create train/val split: {e}")
        if fail_fast:
            raise
        else:
            return None, None, data_files
    return train_files, val_files, data_files

def create_datasets(train_files, val_files, max_examples_unaugmented, max_validation_examples, fail_fast):
    """
    Create StreamingSequentialShardDataset objects for train and val sets.
    Returns (train_dataset, val_dataset).
    """
    try:
        train_dataset = StreamingSequentialShardDataset(
            train_files, 
            enable_augmentation=True, 
            max_examples_unaugmented=max_examples_unaugmented
        )
        val_dataset = StreamingSequentialShardDataset(
            val_files, 
            enable_augmentation=False, # Validation dataset is not augmented
            max_examples_unaugmented=max_validation_examples
        ) if val_files else None
    except Exception as e:
        logger.error(f"Failed to create training/validation datasets: {e}")
        if fail_fast:
            raise
        else:
            return None, None
    return train_dataset, val_dataset

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
    train_dataset, 
    val_dataset, 
    results_path, 
    num_epochs, 
    mini_epoch_batches, 
    device, 
    resume_from: Optional[str] = None,
    resume_epoch: Optional[int] = None,
    shutdown_handler=None
):
    """
    Run a single experiment: instantiate Trainer, Orchestrator, and run training.
    Returns a result dict (can be expanded later).
    
    Args:
        exp_config: Experiment configuration
        train_dataset: Training dataset
        val_dataset: Validation dataset
        results_path: Path to save results
        num_epochs: Number of training epochs
        mini_epoch_batches: Number of mini-epoch batches
        device: Device to use for training
        resume_from: Optional path to resume from (file or directory)
        resume_epoch: Optional epoch to resume from (only used with directory)
        shutdown_handler: Optional shutdown handler
    """
    model = TwoHeadedResNet(dropout_prob=exp_config['hyperparameters'].get('dropout_prob', 0.1))
    
    # Handle resume logic
    start_epoch = 0
    checkpoint_path = None
    
    if resume_from:
        resume_path = Path(resume_from)
        
        if resume_path.is_file():
            # Resume from specific checkpoint file
            checkpoint_path = resume_path
            logger.info(f"Resuming from checkpoint file: {checkpoint_path}")
            
        elif resume_path.is_dir():
            # Resume from experiment directory
            if resume_epoch is not None:
                # Find latest checkpoint for specified epoch
                checkpoint_path = find_latest_checkpoint_for_epoch(resume_path, resume_epoch)
                if checkpoint_path:
                    logger.info(f"Found checkpoint for epoch {resume_epoch}: {checkpoint_path}")
                else:
                    raise FileNotFoundError(f"No checkpoint found for epoch {resume_epoch} in {resume_path}")
            else:
                # Find latest checkpoint overall
                checkpoint_files = list(resume_path.glob("epoch*_mini*.pt*"))
                if checkpoint_files:
                    checkpoint_path = max(checkpoint_files, key=lambda f: f.name)
                    logger.info(f"Found latest checkpoint: {checkpoint_path}")
                else:
                    raise FileNotFoundError(f"No checkpoint files found in {resume_path}")
        else:
            raise FileNotFoundError(f"Resume path does not exist: {resume_from}")
        
        # Load the checkpoint
        if checkpoint_path and checkpoint_path.exists():
            try:
                logger.info(f"Loading checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                # Load model state
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # Determine start epoch from checkpoint filename
                # Extract epoch number from filename like "epoch2_mini36.pt.gz"
                filename = checkpoint_path.name
                if filename.startswith("epoch") and "_mini" in filename:
                    epoch_part = filename.split("_mini")[0]
                    start_epoch = int(epoch_part.replace("epoch", ""))
                    logger.info(f"Resuming from epoch {start_epoch}")
                else:
                    # Fallback to checkpoint metadata
                    start_epoch = checkpoint.get('epoch', 0) + 1
                    logger.info(f"Resuming from epoch {start_epoch} (from checkpoint metadata)")
                
            except Exception as e:
                logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
                raise
    
    trainer = Trainer(
        model=model,
        train_loader=torch.utils.data.DataLoader(
            train_dataset,
            batch_size=exp_config['hyperparameters']['batch_size'],
            shuffle=False,
            num_workers=0,
            drop_last=False
        ),
        val_loader=torch.utils.data.DataLoader(
            val_dataset,
            batch_size=exp_config['hyperparameters']['batch_size'],
            shuffle=False,
            num_workers=0,
            drop_last=False
        ) if val_dataset else None,
        learning_rate=exp_config['hyperparameters']['learning_rate'],
        device=device,
        enable_system_analysis=True,
        enable_csv_logging=True,
        experiment_name=exp_config['experiment_name'],
        policy_weight=exp_config['hyperparameters'].get('policy_weight', 0.15),
        value_weight=exp_config['hyperparameters'].get('value_weight', 0.85),
        weight_decay=exp_config['hyperparameters'].get('weight_decay', 1e-4),
        max_grad_norm=exp_config['hyperparameters'].get('max_grad_norm', 20.0),
        value_learning_rate_factor=exp_config['hyperparameters'].get('value_learning_rate_factor', 0.1),
        value_weight_decay_factor=exp_config['hyperparameters'].get('value_weight_decay_factor', 5.0),
        log_interval_batches=exp_config['hyperparameters'].get('log_interval_batches', 200)
    )
    
    exp_dir = results_path / exp_config['experiment_name']
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    orchestrator = MiniEpochOrchestrator(
        trainer=trainer,
        train_loader=trainer.train_loader,
        val_loader=trainer.val_loader,
        mini_epoch_batches=mini_epoch_batches,
        num_epochs=num_epochs,
        checkpoint_dir=exp_dir,
        log_interval=1,
        shutdown_handler=shutdown_handler,
        start_epoch=start_epoch  # Start from specific epoch if resuming
    )
    
    orchestrator.run()
    logger.info(f"Experiment {exp_config['experiment_name']} completed.")
    
    # Collect relevant metrics for sweep summary
    best_val_loss = trainer.best_val_loss
    best_train_loss = min(trainer.training_history) if trainer.training_history else None
    final_val_loss = trainer.best_val_loss  # Could be last val loss if tracked separately
    final_train_loss = trainer.training_history[-1] if trainer.training_history else None
    
    return {
        'experiment_name': exp_config['experiment_name'],
        'hyperparameters': exp_config['hyperparameters'],
        'best_val_loss': best_val_loss,
        'best_train_loss': best_train_loss,
        'final_val_loss': final_val_loss,
        'final_train_loss': final_train_loss,
        'resumed_from': str(checkpoint_path) if checkpoint_path else None,
        'start_epoch': start_epoch,
    }

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
    data_weights: Optional[List[float]] = None,
    train_ratio: float = 0.8, 
    random_seed: Optional[int] = None,
    fail_fast: bool = True
) -> Tuple[List[Path], List[Path], List[Path], List[Dict]]:
    """
    Discover and split data from multiple directories.
    
    Args:
        data_dirs: List of data directories
        data_weights: Optional weights for each directory (must sum to 1.0)
        train_ratio: Ratio for train/val split
        random_seed: Random seed for reproducible splits
        fail_fast: Whether to fail on first error
        
    Returns:
        Tuple of (train_files, val_files, all_files, data_source_info)
    """
    from hex_ai.data_pipeline import discover_processed_files, estimate_dataset_size
    
    all_data_files = []
    data_source_info = []
    
    # Validate data weights if provided
    if data_weights is not None:
        if len(data_weights) != len(data_dirs):
            raise ValueError(f"Number of data weights ({len(data_weights)}) must match number of data directories ({len(data_dirs)})")
        if abs(sum(data_weights) - 1.0) > 1e-6:
            raise ValueError(f"Data weights must sum to 1.0, got {sum(data_weights)}")
    
    for i, data_dir in enumerate(data_dirs):
        try:
            data_files = discover_processed_files(data_dir)
            weight = data_weights[i] if data_weights else 1.0 / len(data_dirs)
            
            all_data_files.extend(data_files)
            data_source_info.append({
                'directory': data_dir,
                'files': data_files,
                'weight': weight,
                'examples_estimated': estimate_dataset_size(data_files)
            })
            
            logger.info(f"Data source {i+1}/{len(data_dirs)}: {data_dir}")
            logger.info(f"  - Files: {len(data_files)}")
            logger.info(f"  - Weight: {weight:.3f}")
            logger.info(f"  - Estimated examples: {data_source_info[-1]['examples_estimated']:,}")
            
        except Exception as e:
            logger.error(f"Failed to discover data in {data_dir}: {e}")
            if fail_fast:
                raise
            else:
                continue
    
    if not all_data_files:
        raise FileNotFoundError(f"No data files found in any of the provided directories: {data_dirs}")
    
    # Create train/val split across all files
    from hex_ai.data_pipeline import create_train_val_split
    train_files, val_files = create_train_val_split(
        all_data_files, train_ratio, random_seed, max_files_per_split=None
    )
    
    logger.info(f"Combined data sources: {len(all_data_files)} total files")
    logger.info(f"  - Train: {len(train_files)} files")
    logger.info(f"  - Validation: {len(val_files)} files")
    
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
        'created_at': datetime.now().isoformat(),
        'hyperparameters': hyperparameters,
        'training_config': training_config,
        'data_sources': data_source_info,
        'total_examples': sum(src['examples_estimated'] for src in data_source_info),
        'data_weights': [src['weight'] for src in data_source_info]
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
    data_weights: Optional[List[float]] = None,  # New: Optional weighting
    results_dir: str = "checkpoints/hyperparameter_tuning",
    train_ratio: float = 0.8,
    num_epochs: int = 10,
    early_stopping_patience: Optional[int] = None,
    random_seed: Optional[int] = None,
    max_examples_unaugmented: Optional[int] = None,
    max_validation_examples: Optional[int] = None,
    experiment_name: Optional[str] = None,
    enable_augmentation: bool = True,
    fail_fast: bool = True,
    mini_epoch_batches: int = 500,
    resume_from: Optional[str] = None,  # New: Resume from checkpoint
    resume_epoch: Optional[int] = None,  # New: Resume from specific epoch
    shutdown_handler=None
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
        data_weights = None  # Single directory doesn't need weights
    
    if max_validation_examples is None:
        max_validation_examples = max_examples_unaugmented
    
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    if experiment_name is None:
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"\nGrabbing data from {len(data_dirs)} directories with random seed {random_seed}...")
    
    # Use new multi-directory data discovery
    train_files, val_files, data_files, data_source_info = discover_and_split_multiple_data(
        data_dirs, data_weights, train_ratio, random_seed, fail_fast
    )
    
    if train_files is None or val_files is None:
        return {'error': 'Failed to discover or split data'}
    
    train_dataset, val_dataset = create_datasets(train_files, val_files, max_examples_unaugmented, max_validation_examples, fail_fast)
    
    # Remove or guard len() usage for streaming datasets
    logger.info(f"\nStreaming training dataset: {len(train_files)} files, up to {max_examples_unaugmented} examples; validation: {len(val_files)} files, up to {max_validation_examples} examples.")
    
    if train_dataset is None:
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
                train_dataset,
                val_dataset,
                results_path,
                num_epochs,
                mini_epoch_batches,
                device,
                resume_from=resume_from,
                resume_epoch=resume_epoch,
                shutdown_handler=shutdown_handler
            )
            
            # Save experiment metadata with data source information
            save_experiment_metadata(
                results_path,
                exp_config['experiment_name'],
                data_source_info,
                exp_config['hyperparameters'],
                {
                    'num_epochs': num_epochs,
                    'mini_epoch_batches': mini_epoch_batches,
                    'max_examples_unaugmented': max_examples_unaugmented,
                    'max_validation_examples': max_validation_examples,
                    'enable_augmentation': enable_augmentation,
                    'train_ratio': train_ratio,
                    'random_seed': random_seed,
                    'resumed_from': resume_from,
                    'resume_epoch': resume_epoch
                }
            )
            
            all_results.append(result)
            
        except Exception as e:
            logger.error(f"Experiment {exp_config['experiment_name']} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if fail_fast:
                logger.error(f"Fail-fast mode enabled: stopping sweep after failure in experiment {exp_config['experiment_name']}")
                raise
            else:
                continue
    
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