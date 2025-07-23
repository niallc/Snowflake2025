#!/usr/bin/env python3
"""
Training script with integrated monitoring for value head debugging.

USAGE:
  python scripts/train_with_monitoring.py \
    --data_dir data/processed/shuffled \
    --save_dir checkpoints/monitored_training \
    --batch_size 256 \
    --learning_rate 0.001 \
    --num_epochs 5 \
    --enable_augmentation

Recommended: Use the same data directory and batch size as in scripts/hyperparam_sweep.py for comparable results.

This script adds gradient and activation monitoring to the training process
to help diagnose value head performance issues.
"""

import argparse
import sys
import os
from pathlib import Path
import json
import torch
import numpy as np
import logging
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hex_ai.models import TwoHeadedResNet
from hex_ai.training import Trainer, PolicyValueLoss
from hex_ai.training_utils import GradientMonitor, ActivationMonitor, ValueHeadAnalyzer
from hex_ai.data_pipeline import StreamingSequentialShardDataset, discover_processed_files, create_train_val_split
from hex_ai.training_utils import get_device

# ========== Logging Setup (copied from hyperparam_sweep.py) ==========
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / ('monitored_training_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.log')

file_handler = logging.FileHandler(log_file, mode='a')
formatter = logging.Formatter('%(asctime)s %(levelname)s:%(name)s: %(message)s')
file_handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.addHandler(file_handler)
root_logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)
# ========== End Logging Setup ==========

def create_monitored_trainer(model, train_loader, val_loader, device, **kwargs):
    """Create a trainer with monitoring capabilities."""
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        **kwargs
    )
    
    # Add monitoring
    gradient_monitor = GradientMonitor(model, log_interval=50)
    activation_monitor = ActivationMonitor(model, log_interval=50)
    
    return trainer, gradient_monitor, activation_monitor


def train_with_monitoring(trainer, gradient_monitor, activation_monitor, num_epochs, save_dir):
    """Train with monitoring and save analysis results."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    monitoring_results = {
        'gradient_history': [],
        'activation_history': [],
        'activation_summary': None,
        'value_analysis': None
    }
    
    logging.info("Starting training with monitoring...")
    
    for epoch in range(num_epochs):
        logging.info(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        
        # Training
        train_metrics = trainer.train_epoch()
        
        # Log gradients and activations during training (focus on value head)
        for batch_idx, (boards, policies, values) in enumerate(trainer.train_loader):
            if batch_idx % 50 == 0:
                gradient_monitor.log_gradients(batch_idx)
                activation_monitor.log_activations(batch_idx)
                # Save value head focused gradient and activation stats
                grad_norms = gradient_monitor.compute_gradient_norms()
                activation_stats = {}
                # Only keep value head activations if available
                for layer_name, activations in activation_monitor.activation_history.items():
                    if 'value_head' in layer_name and activations:
                        last = activations[-1]
                        activation_stats[layer_name] = last
                monitoring_results['gradient_history'].append({
                    'batch_idx': batch_idx,
                    'value_head': grad_norms.get('value_head', None),
                    'policy_head': grad_norms.get('policy_head', None),
                    'shared_layers': grad_norms.get('shared_layers', None)
                })
                monitoring_results['activation_history'].append({
                    'batch_idx': batch_idx,
                    'value_head': activation_stats
                })
        
        # Validation
        val_metrics = trainer.validate()
        
        # Save monitoring data every few epochs
        if (epoch + 1) % 5 == 0:
            gradient_summary = gradient_monitor.get_summary()
            activation_summary = activation_monitor.get_summary()
            monitoring_results['activation_summary'] = activation_summary
            
            logging.info(f"Epoch {epoch + 1} - Gradient Summary:")
            for key, values in gradient_summary.items():
                logging.info(f"  {key}: mean={values['mean']:.6f}, std={values['std']:.6f}")
    
    # Final analysis
    logging.info("\n=== Final Analysis ===")
    
    # Value head analysis
    value_analyzer = ValueHeadAnalyzer(trainer.model, trainer.train_loader.dataset, trainer.device)
    value_analysis = value_analyzer.analyze_value_predictions(num_samples=1000)
    simple_position_results = value_analyzer.test_simple_positions()
    
    monitoring_results['value_analysis'] = {
        'position_type_analysis': value_analysis,
        'simple_position_results': simple_position_results
    }
    
    # Save results
    results_file = save_path / "monitoring_results.json"
    with open(results_file, 'w') as f:
        json.dump(monitoring_results, f, indent=2, default=str)
    
    logging.info(f"Monitoring results saved to {results_file}")
    
    # Print summary
    logging.info("\n=== Value Head Performance Summary ===")
    for pos_type, results in value_analysis.items():
        logging.info(f"{pos_type}: MSE={results['mse']:.4f}, Accuracy={results['accuracy']:.2%}")
    
    logging.info("\n=== Simple Position Results ===")
    for position, value in simple_position_results.items():
        logging.info(f"{position}: {value:.4f}")
    
    # Cleanup
    activation_monitor.cleanup()
    
    return monitoring_results


def main():
    parser = argparse.ArgumentParser(description="Train with monitoring for value head debugging")
    parser.add_argument('--data_dir', type=str, default="data/processed/shuffled", help='Data directory (recommended: data/processed/shuffled)')
    parser.add_argument('--save_dir', type=str, default="checkpoints/monitored_training", help='Save directory (recommended: checkpoints/monitored_training)')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size (default: 256, matches sweep)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001, matches sweep)')
    parser.add_argument('--device', type=str, default=None, help='Device to use')
    parser.add_argument('--max_examples', type=int, default=5000000, help='Max examples to use (default: 5,000,000, matches sweep)')
    parser.add_argument('--enable_augmentation', action='store_true', help='Enable data augmentation (recommended: use for consistency with sweep)')
    
    args = parser.parse_args()
    
    # Setup
    device = get_device() if args.device is None else torch.device(args.device)
    logging.info(f"Using device: {device}")
    
    # Data loading
    logging.info("Loading data...")
    data_files = discover_processed_files(args.data_dir)
    train_files, val_files = create_train_val_split(data_files, train_ratio=0.8)

    # Fallback: Use StreamingAugmentedProcessedDataset instead of StreamingSequentialShardDataset
    # This is less memory efficient but avoids issues with __len__ and is known to work.
    from hex_ai.data_pipeline import StreamingAugmentedProcessedDataset
    train_dataset = StreamingAugmentedProcessedDataset(
        train_files,
        enable_augmentation=args.enable_augmentation,
        max_examples_unaugmented=args.max_examples
    )

    val_dataset = StreamingAugmentedProcessedDataset(
        val_files,
        enable_augmentation=args.enable_augmentation,
        max_examples_unaugmented=args.max_examples // 5
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Model
    logging.info("Creating model...")
    model = TwoHeadedResNet(dropout_prob=0.1)
    
    # Trainer with monitoring
    trainer, gradient_monitor, activation_monitor = create_monitored_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        enable_csv_logging=True,
        experiment_name="monitored_training"
    )
    
    # Train with monitoring
    results = train_with_monitoring(
        trainer=trainer,
        gradient_monitor=gradient_monitor,
        activation_monitor=activation_monitor,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir
    )
    
    logging.info(f"\nTraining complete! Results saved to {args.save_dir}")


if __name__ == "__main__":
    main() 