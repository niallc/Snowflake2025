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
from hex_ai.mini_epoch_orchestrator import MiniEpochOrchestrator

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
    max_examples_unaugmented: Optional[int] = None,
    max_validation_examples: Optional[int] = None,
    experiment_name: Optional[str] = None,
    enable_augmentation: bool = True,
    fail_fast: bool = True,
    mini_epoch_batches: int = 500
) -> Dict:
    """
    Run a complete hyperparameter tuning experiment using current data pipeline and mini-epoch orchestration.
    Uses MiniEpochOrchestrator to perform validation and checkpointing every mini-epoch.
    """
    import time
    from hex_ai.data_pipeline import StreamingAugmentedProcessedDataset, discover_processed_files, create_train_val_split
    
    # Use max_examples_unaugmented for validation if not specified
    if max_validation_examples is None:
        max_validation_examples = max_examples_unaugmented
    
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
    
    # If max_examples_unaugmented is specified, limit the data
    if max_examples_unaugmented is not None:
        logger.info(f"Limiting training data to ~{max_examples_unaugmented:,} examples for quick exploration")
    if max_validation_examples is not None:
        logger.info(f"Limiting validation data to ~{max_validation_examples:,} examples for quick exploration")
    
    dataset_info = {
        'total_files': len(data_files),
        'train_files': len(train_files),
        'val_files': len(val_files),
        'max_training_examples': max_examples_unaugmented,
        'max_validation_examples': max_validation_examples
    }
    
    logger.info(f"Dataset summary: {len(data_files)} total files, {len(train_files)} train, {len(val_files)} validation")
    if max_examples_unaugmented:
        logger.info(f"Training will use streaming with max_examples={max_examples_unaugmented:,}")
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
        'max_training_examples': max_examples_unaugmented,
        'max_validation_examples': max_validation_examples,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(results_path / "overall_config.json", "w") as f:
        json.dump(overall_config, f, indent=2)
    
    # Load data once and reuse across experiments using current pipeline
    logger.info(f"\nLoading data once for all experiments...")
    
    # Create training dataset with optional augmentation
    try:
        train_dataset = StreamingAugmentedProcessedDataset(
            train_files, 
            enable_augmentation=True, 
            chunk_size=100000,  # Use fixed chunk size for memory efficiency
            max_examples=max_examples_unaugmented
        )
        # Note: Validation dataset is not augmented
        val_dataset = StreamingAugmentedProcessedDataset(
            val_files, 
            enable_augmentation=False, # Validation dataset is not augmented
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
    
    all_results = []
    total_start_time = time.time()
    logger.info(f"\nStarting {len(experiments)} experiments...")
    
    for i, exp_config in enumerate(experiments):
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment {i+1}/{len(experiments)}: {exp_config['experiment_name']}")
        logger.info(f"{'='*60}")
        exp_config['device'] = device
        if max_examples_unaugmented is not None:
            exp_config['max_examples_unaugmented'] = max_examples_unaugmented
        if max_validation_examples is not None:
            exp_config['max_validation_examples'] = max_validation_examples
        exp_config['enable_augmentation'] = enable_augmentation
        try:
            # Instantiate Trainer for this experiment
            model = TwoHeadedResNet(dropout_prob=exp_config['hyperparameters'].get('dropout_prob', 0.1))
            trainer = Trainer(
                model=model,
                train_loader=torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=exp_config['hyperparameters']['batch_size'],
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False
                ),
                val_loader=torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=exp_config['hyperparameters']['batch_size'],
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False
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
                value_weight_decay_factor=exp_config['hyperparameters'].get('value_weight_decay_factor', 5.0)
            )
            # Orchestrate mini-epoch training
            exp_dir = results_path / exp_config['experiment_name']
            exp_dir.mkdir(parents=True, exist_ok=True)
            orchestrator = MiniEpochOrchestrator(
                trainer=trainer,
                train_loader=trainer.train_loader,
                val_loader=trainer.val_loader,
                mini_epoch_batches=mini_epoch_batches,
                num_epochs=num_epochs,
                checkpoint_dir=exp_dir,
                log_interval=1
            )
            orchestrator.run()
            # Optionally, collect results/metrics from trainer or orchestrator
            # For now, just log completion
            logger.info(f"Experiment {exp_config['experiment_name']} completed.")
            all_results.append({'experiment_name': exp_config['experiment_name']})
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
        'experiments': all_results
    }
    with open(results_path / "overall_results.json", "w") as f:
        json.dump(overall_results, f, indent=2, default=str)
    return overall_results 