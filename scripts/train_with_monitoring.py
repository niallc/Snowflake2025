#!/usr/bin/env python3
"""
Training script with integrated monitoring for value head debugging.

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

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hex_ai.models import TwoHeadedResNet
from hex_ai.training import Trainer, PolicyValueLoss
from hex_ai.training_utils import GradientMonitor, ActivationMonitor, ValueHeadAnalyzer
from hex_ai.data_pipeline import StreamingSequentialShardDataset, discover_processed_files, create_train_val_split
from hex_ai.training_utils import get_device


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
        'activation_summary': None,
        'value_analysis': None
    }
    
    print("Starting training with monitoring...")
    
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        
        # Training
        train_metrics = trainer.train_epoch()
        
        # Log gradients during training (this happens in train_epoch)
        for batch_idx, (boards, policies, values) in enumerate(trainer.train_loader):
            # This is a simplified version - in practice, you'd integrate this into the actual training loop
            if batch_idx % 50 == 0:
                gradient_monitor.log_gradients(batch_idx)
                activation_monitor.log_activations(batch_idx)
        
        # Validation
        val_metrics = trainer.validate()
        
        # Save monitoring data every few epochs
        if (epoch + 1) % 5 == 0:
            gradient_summary = gradient_monitor.get_summary()
            activation_summary = activation_monitor.get_summary()
            
            monitoring_results['gradient_history'].append({
                'epoch': epoch + 1,
                'gradient_summary': gradient_summary,
                'activation_summary': activation_summary
            })
            
            print(f"Epoch {epoch + 1} - Gradient Summary:")
            for key, values in gradient_summary.items():
                print(f"  {key}: mean={values['mean']:.6f}, std={values['std']:.6f}")
    
    # Final analysis
    print("\n=== Final Analysis ===")
    
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
    
    print(f"Monitoring results saved to {results_file}")
    
    # Print summary
    print("\n=== Value Head Performance Summary ===")
    for pos_type, results in value_analysis.items():
        print(f"{pos_type}: MSE={results['mse']:.4f}, Accuracy={results['accuracy']:.2%}")
    
    print("\n=== Simple Position Results ===")
    for position, value in simple_position_results.items():
        print(f"{position}: {value:.4f}")
    
    # Cleanup
    activation_monitor.cleanup()
    
    return monitoring_results


def main():
    parser = argparse.ArgumentParser(description="Train with monitoring for value head debugging")
    parser.add_argument('--data_dir', type=str, default="data/processed", help='Data directory')
    parser.add_argument('--save_dir', type=str, default="checkpoints/monitored_training", help='Save directory')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default=None, help='Device to use')
    parser.add_argument('--max_examples', type=int, default=100000, help='Max examples to use')
    parser.add_argument('--enable_augmentation', action='store_true', help='Enable data augmentation')
    
    args = parser.parse_args()
    
    # Setup
    device = get_device() if args.device is None else torch.device(args.device)
    print(f"Using device: {device}")
    
    # Data loading
    print("Loading data...")
    data_files = discover_processed_files(args.data_dir)
    train_files, val_files = create_train_val_split(data_files, train_ratio=0.8)
    
    train_dataset = StreamingSequentialShardDataset(
        train_files, 
        enable_augmentation=args.enable_augmentation,
        max_examples_unaugmented=args.max_examples
    )
    
    val_dataset = StreamingSequentialShardDataset(
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
    print("Creating model...")
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
    
    print(f"\nTraining complete! Results saved to {args.save_dir}")


if __name__ == "__main__":
    main() 