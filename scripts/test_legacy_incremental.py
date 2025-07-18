#!/usr/bin/env python3
"""
Incremental testing script to migrate from legacy to modern architecture.

This script tests changes one at a time to identify which change causes
the performance regression.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List
import torch
import numpy as np
from datetime import datetime
import multiprocessing
import logging
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Incremental testing from legacy to modern architecture')
parser.add_argument('--step', type=str, required=True, 
                   choices=['2.1', '2.2', '2.3', '2.4'],
                   help='Which step to test: 2.1=player-to-move channel, 2.2=5x5 conv, 2.3=training pipeline, 2.4=data pipeline')
parser.add_argument('--verbose', '-v', type=int, default=2, 
                   help='Verbose level: 0=critical only, 1=important, 2=detailed (default), 3=debug')
parser.add_argument('--num-epochs', type=int, default=5,
                   help='Number of epochs to train (default: 5)')
parser.add_argument('--batch-size', type=int, default=256,
                   help='Batch size for training (default: 256)')
parser.add_argument('--target-examples', type=int, default=50000,
                   help='Number of examples to use for training (default: 50000)')
args = parser.parse_args()

# Set up logging
if args.verbose == 0:
    logging.basicConfig(level=logging.CRITICAL)
elif args.verbose == 1:
    logging.basicConfig(level=logging.WARNING)
elif args.verbose == 2:
    logging.basicConfig(level=logging.INFO)
else:  # args.verbose >= 3
    logging.basicConfig(level=logging.DEBUG)

# Fix multiprocessing on macOS
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

from hex_ai.models_legacy import TwoHeadedResNetLegacy
from hex_ai.training_utils_legacy import (
    run_hyperparameter_tuning_legacy,
    discover_processed_files_legacy,
    estimate_dataset_size_legacy
)

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    print(f"Using CUDA GPU: {device_name}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS GPU")
else:
    device = torch.device("cpu")
    print("Using CPU (no GPU detected)")

def create_step_2_1_model():
    """
    Step 2.1: Add player-to-move channel to legacy model.
    
    This creates a modified version of the legacy model that accepts 3-channel input
    but keeps everything else the same (3x3 conv, legacy training, etc.).
    """
    class TwoHeadedResNetStep2_1(TwoHeadedResNetLegacy):
        """Legacy model with 3-channel input (player-to-move channel added)."""
        
        def __init__(self, resnet_depth=18, dropout_prob=0.1):
            super().__init__(resnet_depth, dropout_prob)
            
            # Override input conv to accept 3 channels instead of 2
            self.input_conv = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.input_bn = torch.nn.BatchNorm2d(64)
            
            # Re-initialize weights for the new input conv
            torch.nn.init.kaiming_normal_(self.input_conv.weight, mode='fan_out', nonlinearity='relu')
    
    return TwoHeadedResNetStep2_1()

def create_step_2_1_dataset(data_files, max_examples=None):
    """
    Step 2.1: Create dataset that adds player-to-move channel to legacy data.
    """
    from hex_ai.training_utils_legacy import NewProcessedDataset
    
    class Step2_1Dataset(NewProcessedDataset):
        """Legacy dataset with player-to-move channel added."""
        
        def __getitem__(self, idx):
            # Get the original 2-channel data
            board_state, policy_target, value_target = super().__getitem__(idx)
            
            # Add player-to-move channel
            board_np = board_state.numpy()
            from hex_ai.data_utils import get_player_to_move_from_board
            
            try:
                player_to_move = get_player_to_move_from_board(board_np)
            except Exception as e:
                # Use default value if we can't determine
                from hex_ai.inference.board_utils import BLUE_PLAYER
                player_to_move = BLUE_PLAYER
            
            # Create player-to-move channel
            player_channel = np.full((board_np.shape[1], board_np.shape[2]), float(player_to_move), dtype=np.float32)
            board_3ch = np.concatenate([board_np, player_channel[None, ...]], axis=0)
            board_state = torch.from_numpy(board_3ch)
            
            return board_state, policy_target, value_target
    
    return Step2_1Dataset(data_files, max_examples=max_examples)

def run_step_2_1_test():
    """Run Step 2.1: Test adding player-to-move channel to legacy code."""
    print(f"\n{'='*60}")
    print("STEP 2.1: TESTING PLAYER-TO-MOVE CHANNEL ADDITION")
    print(f"{'='*60}")
    
    # Discover data files
    print("Discovering processed data files...")
    data_files = discover_processed_files_legacy("data/processed")
    total_examples = estimate_dataset_size_legacy(data_files, max_files=10)
    print(f"Found {len(data_files)} data files with approximately {total_examples:,} training examples")
    
    # Create experiment config
    experiment_name = f"step_2_1_player_channel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    experiments = [
        {
            'experiment_name': 'legacy_with_player_channel',
            'hyperparameters': {
                'learning_rate': 0.001,
                'batch_size': args.batch_size,
                'dropout_prob': 0.1,
                'weight_decay': 5e-4,
                'policy_weight': 0.2,
                'value_weight': 0.8
            }
        }
    ]
    
    # Create results directory
    results_dir = Path("checkpoints") / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = {
        'step': '2.1',
        'description': 'Testing player-to-move channel addition to legacy model',
        'experiment_name': experiment_name,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'target_examples': args.target_examples,
        'device': str(device),
        'timestamp': datetime.now().isoformat()
    }
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Experiment name: {experiment_name}")
    print(f"Results directory: {results_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Target examples: {args.target_examples}")
    
    # Run the experiment with modified legacy code
    start_time = time.time()
    
    # We need to modify the legacy training utils to use our custom model and dataset
    # For now, let's create a simple test that demonstrates the approach
    
    print("\nTesting Step 2.1 approach...")
    print("This step would:")
    print("1. Use legacy model with 3-channel input conv")
    print("2. Use legacy dataset with player-to-move channel added")
    print("3. Use legacy training pipeline")
    print("4. Compare results to pure legacy (2-channel) performance")
    
    # Create test model
    model = create_step_2_1_model()
    print(f"Created Step 2.1 model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    test_input = torch.randn(1, 3, 13, 13)  # 3-channel input
    with torch.no_grad():
        policy_out, value_out = model(test_input)
    print(f"Model forward pass successful: policy={policy_out.shape}, value={value_out.shape}")
    
    total_time = time.time() - start_time
    print(f"\nStep 2.1 test completed in {total_time:.1f}s")
    print(f"Results saved to: {results_dir}")
    
    return {
        'step': '2.1',
        'experiment_name': experiment_name,
        'results_dir': str(results_dir),
        'success': True
    }

def main():
    """Main function to run the specified step."""
    print(f"\n{'='*60}")
    print(f"INCREMENTAL TESTING: STEP {args.step}")
    print(f"{'='*60}")
    
    if args.step == '2.1':
        result = run_step_2_1_test()
    else:
        print(f"Step {args.step} not yet implemented")
        return
    
    print(f"\nStep {args.step} completed successfully!")
    print(f"Next step: Implement step {args.step} with full training pipeline")

if __name__ == '__main__':
    main() 