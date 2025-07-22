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
from hex_ai.data_pipeline import StreamingAugmentedProcessedDataset, discover_processed_files, create_train_val_split

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
    Create StreamingAugmentedProcessedDataset objects for train and val sets.
    Returns (train_dataset, val_dataset).
    """
    try:
        train_dataset = StreamingAugmentedProcessedDataset(
            train_files, 
            enable_augmentation=True, 
            max_examples_unaugmented=max_examples_unaugmented
        )
        val_dataset = StreamingAugmentedProcessedDataset(
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

def run_single_experiment(exp_config, train_dataset, val_dataset, results_path, num_epochs, mini_epoch_batches, device, shutdown_handler=None):
    """
    Run a single experiment: instantiate Trainer, Orchestrator, and run training.
    Returns a result dict (can be expanded later).
    """
    from .models import TwoHeadedResNet
    from .training import Trainer
    import torch
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
        shutdown_handler=shutdown_handler
    )
    orchestrator.run()
    logger.info(f"Experiment {exp_config['experiment_name']} completed.")
    return {'experiment_name': exp_config['experiment_name']}

def select_device():
    """
    Select the best available device (cuda, mps, or cpu).
    """
    import torch
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

def save_overall_results(results_path, overall_results):
    """
    Save the overall results dict to disk as overall_results.json.
    """
    with open(results_path / "overall_results.json", "w") as f:
        json.dump(overall_results, f, indent=2, default=str)

# Refactored main function

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
    mini_epoch_batches: int = 500,
    shutdown_handler=None
) -> Dict:
    """
    Orchestrates the full hyperparameter sweep using modular helpers for data, dataset, and experiment logic.
    """
    import time
    from pathlib import Path
    from datetime import datetime
    if max_validation_examples is None:
        max_validation_examples = max_examples_unaugmented
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    if experiment_name is None:
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"\nGrabbing data from {data_dir} with random seed {random_seed}...")
    train_files, val_files, data_files = discover_and_split_data(data_dir, train_ratio, random_seed, fail_fast)
    if train_files is None or val_files is None:
        return {'error': 'Failed to discover or split data'}
    train_dataset, val_dataset = create_datasets(train_files, val_files, max_examples_unaugmented, max_validation_examples, fail_fast)
    logger.info(f"\nCollected {len(train_dataset)} training examples and {len(val_dataset)} validation examples.")
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
                shutdown_handler=shutdown_handler
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
    save_overall_results(results_path, overall_results)
    logger.info(f"\nOverall results saved to {results_path}.")
    return overall_results 