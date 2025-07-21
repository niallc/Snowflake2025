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
    if not isinstance(example, dict):
        raise ValueError(f"Example in {filename} must be dictionary, got {type(example)}")
    
    # Check for required keys
    if not all(key in example for key in ['board', 'policy', 'value']):
        raise ValueError(f"Example in {filename} missing required keys: {example.keys()}")
    
    board_state = example['board']
    policy_target = example['policy']
    value_target = example['value']
    
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




def augmented_collate_fn(batch):
    """
    Custom collate function for AugmentedProcessedDataset.
    Each item in the batch is a list of 4 augmented examples (tuples).
    This function flattens them into a single batch.
    """
    # Flatten the batch (each item is a list of 4 examples)
    flattened_batch = []
    for item in batch:
        if isinstance(item, list):
            flattened_batch.extend(item)
        else:
            flattened_batch.append(item)
    boards, policies, values = zip(*flattened_batch)
    boards_batch = torch.stack(boards)
    policies_batch = torch.stack(policies)
    values_batch = torch.stack(values)
    return boards_batch, policies_batch, values_batch


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


def run_hyperparameter_tuning_current_data(
    experiments: List[Dict],
    data_dir: str,
    results_dir: str = "checkpoints/hyperparameter_tuning",
    train_ratio: float = 0.8,
    num_epochs: int = 10,
    early_stopping_patience: Optional[int] = None,
    random_seed: Optional[int] = None,
    max_examples_per_split: Optional[int] = None,
    max_validation_examples: Optional[int] = None,
    experiment_name: Optional[str] = None,
    enable_augmentation: bool = True,
    fail_fast: bool = True
) -> Dict:
    """
    Run a complete hyperparameter tuning experiment using current data pipeline.
    
    Args:
        experiments: List of experiment configurations
        data_dir: Directory containing processed data
        results_dir: Directory to save results
        train_ratio: Ratio of data to use for training
        num_epochs: Number of epochs per experiment
        early_stopping_patience: Early stopping patience
        random_seed: Random seed for reproducibility
        max_examples_per_split: Maximum examples for training dataset
        max_validation_examples: Maximum examples for validation dataset (defaults to max_examples_per_split if None)
        enable_augmentation: Whether to use data augmentation for training (default: True)
        fail_fast: If True, any serious failure (e.g., import error, data loading error, experiment failure) will immediately stop the sweep and raise the error. If False, will continue to next experiment. Default: True.
    
    Returns:
        Dictionary containing all experiment results
    """
    import time
    from hex_ai.data_pipeline import StreamingProcessedDataset, StreamingAugmentedProcessedDataset, discover_processed_files, create_train_val_split
    
    # Use max_examples_per_split for validation if not specified
    if max_validation_examples is None:
        max_validation_examples = max_examples_per_split
    
    # Setup
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Use experiment name if provided, otherwise use timestamp
    if experiment_name is None:
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Discover data files using current pipeline
    try:
        data_files = discover_processed_files(data_dir)
    except Exception as e:
        logger.error(f"Failed to discover processed files in {data_dir}: {e}")
        if fail_fast:
            raise
        else:
            return {'error': str(e), 'stage': 'discover_processed_files'}
    
    # Create train/val split (no file limit needed with streaming)
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
            return {'error': str(e), 'stage': 'create_train_val_split'}
    
    # If max_examples_per_split is specified, limit the data
    if max_examples_per_split is not None:
        logger.info(f"Limiting training data to ~{max_examples_per_split:,} examples for quick exploration")
    if max_validation_examples is not None:
        logger.info(f"Limiting validation data to ~{max_validation_examples:,} examples for quick exploration")
    
    dataset_info = {
        'total_files': len(data_files),
        'train_files': len(train_files),
        'val_files': len(val_files),
        'max_training_examples': max_examples_per_split,
        'max_validation_examples': max_validation_examples
    }
    
    logger.info(f"Dataset summary: {len(data_files)} total files, {len(train_files)} train, {len(val_files)} validation")
    if max_examples_per_split:
        logger.info(f"Training will use streaming with max_examples={max_examples_per_split:,}")
    if max_validation_examples:
        logger.info(f"Validation will use streaming with max_examples={max_validation_examples:,}")
    
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
        'enable_augmentation': enable_augmentation,
        'max_training_examples': max_examples_per_split,
        'max_validation_examples': max_validation_examples,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(results_path / "overall_config.json", "w") as f:
        json.dump(overall_config, f, indent=2)
    
    # Load data once and reuse across experiments using current pipeline
    logger.info(f"\nLoading data once for all experiments...")
    
    # Create training dataset with optional augmentation
    try:
        if enable_augmentation:
            logger.info("Using StreamingAugmentedProcessedDataset for training data (4x augmentation)")
            train_dataset = StreamingAugmentedProcessedDataset(
                train_files, 
                enable_augmentation=True, 
                chunk_size=100000,  # Use fixed chunk size for memory efficiency
                max_examples=max_examples_per_split
            )
            # Note: Validation dataset is not augmented
            val_dataset = StreamingProcessedDataset(
                val_files, 
                chunk_size=100000,  # Use fixed chunk size for memory efficiency
                max_examples=max_validation_examples
            ) if val_files else None
        else:
            logger.info("Using standard StreamingProcessedDataset for training data (no augmentation)")
            train_dataset = StreamingProcessedDataset(
                train_files, 
                chunk_size=100000,  # Use fixed chunk size for memory efficiency
                max_examples=max_examples_per_split
            )
            val_dataset = StreamingProcessedDataset(
                val_files, 
                chunk_size=100000,  # Use fixed chunk size for memory efficiency
                max_examples=max_validation_examples
            ) if val_files else None
    except Exception as e:
        logger.error(f"Failed to create training/validation datasets: {e}")
        if fail_fast:
            raise
        else:
            return {'error': str(e), 'stage': 'create_datasets'}
    
    logger.info(f"Created training dataset")
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
        if max_validation_examples is not None:
            exp_config['max_validation_examples'] = max_validation_examples
        exp_config['enable_augmentation'] = enable_augmentation
        
        try:
            results = run_hyperparameter_experiment_current_data(
                exp_config,
                train_dataset,
                val_dataset,
                results_path,
                num_epochs,
                early_stopping_patience,
                exp_config['experiment_name'], # Pass experiment_name here
                enable_augmentation
            )
            all_results.append(results)
            
        except Exception as e:
            logger.error(f"Experiment {exp_config['experiment_name']} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if fail_fast:
                logger.error(f"Fail-fast mode enabled: stopping sweep after failure in experiment {exp_config['experiment_name']}")
                raise
            else:
                continue
    
    # Save overall results
    total_time = time.time() - total_start_time
    overall_results = {
        'total_training_time': total_time,
        'num_experiments': len(experiments),
        'successful_experiments': len(all_results),
        'device': device,
        'enable_augmentation': enable_augmentation,
        'max_training_examples': max_examples_per_split,
        'max_validation_examples': max_validation_examples,
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
    logger.info(f"Data augmentation: {'Enabled' if enable_augmentation else 'Disabled'}")
    if max_examples_per_split:
        logger.info(f"Training: streaming with max_examples={max_examples_per_split:,}")
    else:
        logger.info(f"Training: streaming with unlimited examples")
    if max_validation_examples:
        logger.info(f"Validation: streaming with max_examples={max_validation_examples:,}")
    else:
        logger.info(f"Validation: streaming with unlimited examples")
    
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
                                experiment_name: Optional[str] = None,
                                enable_augmentation: bool = True) -> Dict:
    """
    Run a single hyperparameter experiment using current data pipeline.
    
    Args:
        experiment_config: Experiment configuration
        train_dataset: Streaming training dataset
        val_dataset: Streaming validation dataset (can be None)
        results_dir: Directory to save results
        num_epochs: Number of training epochs
        early_stopping_patience: Early stopping patience (None to disable)
        enable_augmentation: Whether to use data augmentation for training (default: True)
        
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
    
    # Use custom collate_fn if augmentation is enabled
    if enable_augmentation:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=hyperparams['batch_size'],
            shuffle=False,  
            num_workers=0,  # Use 0 to avoid multiprocessing issues
            pin_memory=False,  # Disable pin_memory for MPS
            collate_fn=augmented_collate_fn # Use custom collate_fn for augmented data
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
    else:
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