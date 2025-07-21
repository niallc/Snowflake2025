#!/usr/bin/env python3
"""
Test script to verify that shuffled data can be loaded and processed correctly.
"""

import sys
import logging
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hex_ai.data_pipeline import discover_processed_files, create_train_val_split, StreamingAugmentedProcessedDataset
from hex_ai.training_utils_legacy import run_hyperparameter_tuning_current_data

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_shuffled_data_loading():
    """Test that shuffled data can be loaded correctly."""
    print("Testing shuffled data loading...")
    
    # Test file discovery
    data_files = discover_processed_files("data/processed/shuffled")
    print(f"✓ Discovered {len(data_files)} shuffled files")
    
    # Test train/val split
    train_files, val_files = create_train_val_split(data_files, train_ratio=0.8, random_seed=42)
    print(f"✓ Created split: {len(train_files)} train, {len(val_files)} validation")
    
    # Test dataset creation with a small sample
    try:
        # Use just a few files for testing
        test_files = train_files[:2]
        dataset = StreamingAugmentedProcessedDataset(test_files, chunk_size=1000, max_examples=100)
        print(f"✓ Created dataset with {len(dataset)} examples")
        
        # Test loading a few examples
        for i in range(min(3, len(dataset))):
            example = dataset[i]
            print(f"✓ Example {i}: board shape {example[0].shape}, value {example[2]}")
            
    except Exception as e:
        print(f"✗ Error creating dataset: {e}")
        return False
    
    return True

def test_hyperparameter_tuning_integration():
    """Test that the hyperparameter tuning function works with shuffled data."""
    print("\nTesting hyperparameter tuning integration...")
    
    # Create a minimal experiment
    experiments = [{
        'experiment_name': 'test_shuffled_data',
        'hyperparameters': {
            'learning_rate': 0.001,
            'batch_size': 64,
            'max_grad_norm': 20,
            'dropout_prob': 0,
            'weight_decay': 1e-4,
            'value_learning_rate_factor': 0.001,
            'value_weight_decay_factor': 10.0,
        }
    }]
    
    try:
        # This should not actually run training, just test the setup
        # We'll limit it to very few examples and epochs
        results = run_hyperparameter_tuning_current_data(
            experiments=experiments,
            data_dir="data/processed/shuffled",
            results_dir="checkpoints/test_shuffled",
            train_ratio=0.8,
            num_epochs=1,  # Just 1 epoch for testing
            early_stopping_patience=None,
            random_seed=42,
            max_examples_per_split=1000,  # Very small for testing
            max_validation_examples=200,
            enable_augmentation=False  # Disable for faster testing
        )
        
        print("✓ Hyperparameter tuning function completed successfully")
        print(f"✓ Results: {len(results.get('experiments', []))} experiments completed")
        
    except Exception as e:
        print(f"✗ Error in hyperparameter tuning: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING SHUFFLED DATA INTEGRATION")
    print("=" * 60)
    
    success1 = test_shuffled_data_loading()
    success2 = test_hyperparameter_tuning_integration()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✓ ALL TESTS PASSED - Shuffled data integration is working!")
    else:
        print("✗ SOME TESTS FAILED - Please check the errors above")
    print("=" * 60) 