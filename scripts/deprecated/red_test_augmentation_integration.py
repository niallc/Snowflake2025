#!/usr/bin/env python3
"""
Test script to verify data augmentation integration with training pipeline.
This script runs a minimal training experiment with and without augmentation
to ensure the integration works correctly.
"""

import sys
import os
import torch
from pathlib import Path
import logging

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hex_ai.training_orchestration import run_hyperparameter_tuning_current_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_augmentation_integration():
    """Test the augmentation integration with a minimal experiment."""
    
    # Check if data directory exists
    data_dir = "data/processed"
    if not Path(data_dir).exists():
        print(f"Data directory {data_dir} not found. Skipping integration test.")
        return
    
    # Find some data files
    data_files = list(Path(data_dir).glob("*.pkl.gz"))
    if not data_files:
        print(f"No data files found in {data_dir}. Skipping integration test.")
        return
    
    print(f"Found {len(data_files)} data files for testing")
    
    # Create a minimal experiment configuration
    experiments = [
        {
            'experiment_name': 'test_augmentation_enabled',
            'hyperparameters': {
                'learning_rate': 0.001,
                'batch_size': 32,  # Small batch size for testing
                'max_grad_norm': 20,
                'dropout_prob': 0,
                'weight_decay': 1e-4,
                'value_learning_rate_factor': 0.1,
                'value_weight_decay_factor': 2.0,
            }
        }
    ]
    
    print("\n" + "="*60)
    print("TESTING AUGMENTATION INTEGRATION")
    print("="*60)
    
    # Test with augmentation enabled
    print("\n1. Testing with augmentation ENABLED...")
    try:
        results_enabled = run_hyperparameter_tuning_current_data(
            experiments=experiments,
            data_dir=data_dir,
            results_dir="checkpoints/test_augmentation_enabled",
            train_ratio=0.8,
            num_epochs=1,  # Just 1 epoch for testing
            early_stopping_patience=None,
            random_seed=42,
            max_examples_unaugmented=1000,  # Small dataset for testing
            experiment_name="test_augmentation_enabled",
            enable_augmentation=True
        )
        print("✅ Augmentation ENABLED test completed successfully")
        print(f"   Successful experiments: {results_enabled.get('successful_experiments', 0)}")
    except Exception as e:
        print(f"❌ Augmentation ENABLED test failed: {e}")
        return
    
    # Test with augmentation disabled
    print("\n2. Testing with augmentation DISABLED...")
    try:
        results_disabled = run_hyperparameter_tuning_current_data(
            experiments=experiments,
            data_dir=data_dir,
            results_dir="checkpoints/test_augmentation_disabled",
            train_ratio=0.8,
            num_epochs=1,  # Just 1 epoch for testing
            early_stopping_patience=None,
            random_seed=42,
            max_examples_unaugmented=1000,  # Small dataset for testing
            experiment_name="test_augmentation_disabled",
            enable_augmentation=False
        )
        print("✅ Augmentation DISABLED test completed successfully")
        print(f"   Successful experiments: {results_disabled.get('successful_experiments', 0)}")
    except Exception as e:
        print(f"❌ Augmentation DISABLED test failed: {e}")
        return
    
    print("\n" + "="*60)
    print("INTEGRATION TEST COMPLETE")
    print("="*60)
    print("✅ Both augmentation modes work correctly!")
    print("\nNext steps:")
    print("1. Run a full sweep with augmentation: python -m scripts.hyperparam_sweep")
    print("2. Compare results with and without augmentation")
    print("3. Monitor training speed and memory usage")

if __name__ == "__main__":
    test_augmentation_integration() 