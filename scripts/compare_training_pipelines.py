#!/usr/bin/env python3
"""
Compare the overfit training pipeline with the main training pipeline on the same small dataset.
This helps identify differences that might explain why the main training isn't working.
"""

import argparse
import torch
import numpy as np
import time
import psutil
import os
from pathlib import Path

# Add the parent directory to the path so we can import from hex_ai
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hex_ai.data_pipeline import StreamingProcessedDataset
from hex_ai.models import create_model
from hex_ai.training_utils import get_device
from hex_ai.training import PolicyValueLoss, Trainer
from hex_ai.utils.format_conversion import tensor_to_rowcol, tensor_to_trmph


def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024


def run_overfit_pipeline(data_files, max_samples=500, epochs=50, device='cpu'):
    """Run the overfit pipeline on the given data."""
    print(f"\n=== RUNNING OVERFIT PIPELINE ===")
    print(f"Data files: {data_files}")
    print(f"Max samples: {max_samples}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    # Load dataset
    dataset = StreamingProcessedDataset(data_files, chunk_size=10000, shuffle_files=False)
    
    # Load samples
    all_samples = []
    for i in range(min(max_samples, len(dataset))):
        try:
            sample = dataset[i]
            all_samples.append(sample)
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            break
    
    print(f"Loaded {len(all_samples)} samples")
    
    # Extract tensors
    boards = torch.stack([b for b, p, v in all_samples])
    policies = torch.stack([p for b, p, v in all_samples])
    values = torch.stack([v for b, p, v in all_samples])
    
    # Create model
    model = create_model()
    model = model.to(device)
    
    # Create dataloader
    from torch.utils.data import TensorDataset, DataLoader
    dataset_tensor = TensorDataset(boards, policies, values)
    dataloader = DataLoader(dataset_tensor, batch_size=32, shuffle=True)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = PolicyValueLoss()
    
    print(f"  Overfit pipeline loss weights: policy={criterion.policy_weight}, value={criterion.value_weight}")
    print(f"  Overfit pipeline optimizer: {type(optimizer).__name__}, lr={optimizer.param_groups[0]['lr']}, weight_decay={optimizer.param_groups[0].get('weight_decay', 0)}")
    
    # Training loop
    best_accuracy = 0.0
    training_history = []
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        num_batches = 0
        
        for batch_idx, (boards_batch, policy_targets, value_targets) in enumerate(dataloader):
            boards_batch = boards_batch.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            
            # Forward pass
            policy_output, value_output = model(boards_batch)
            
            # Compute loss
            total_loss, loss_dict = criterion(policy_output, value_output, policy_targets, value_targets)
            
            # Compute accuracy
            policy_probs = torch.softmax(policy_output, dim=1)
            predicted_moves = torch.argmax(policy_probs, dim=1)
            target_moves = torch.argmax(policy_targets, dim=1)
            correct_predictions += (predicted_moves == target_moves).sum().item()
            total_predictions += predicted_moves.size(0)
            
            # Accumulate losses
            total_policy_loss += loss_dict['policy_loss']
            total_value_loss += loss_dict['value_loss']
            num_batches += 1
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Clear memory
            del policy_output, value_output, policy_probs, predicted_moves, target_moves
            del boards_batch, policy_targets, value_targets
        
        epoch_time = time.time() - epoch_start_time
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        avg_policy_loss = total_policy_loss / num_batches if num_batches > 0 else 0.0
        avg_value_loss = total_value_loss / num_batches if num_batches > 0 else 0.0
        
        training_history.append({
            'epoch': epoch,
            'accuracy': accuracy,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'time': epoch_time
        })
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        
        if epoch % 10 == 0 or epoch in [1, 5, 10, 20, 30, 40]:
            print(f"  Epoch {epoch:3d}: policy_loss={avg_policy_loss:.6f} value_loss={avg_value_loss:.6f} accuracy={accuracy:.3f} time={epoch_time:.1f}s")
    
    end_time = time.time()
    end_memory = get_memory_usage()
    
    print(f"\nOverfit Pipeline Results:")
    print(f"  Best accuracy: {best_accuracy:.3f}")
    print(f"  Final policy loss: {training_history[-1]['policy_loss']:.6f}")
    print(f"  Final value loss: {training_history[-1]['value_loss']:.6f}")
    print(f"  Total time: {end_time - start_time:.1f}s")
    print(f"  Memory usage: {end_memory - start_memory:.2f}GB")
    
    return {
        'best_accuracy': best_accuracy,
        'final_policy_loss': training_history[-1]['policy_loss'],
        'final_value_loss': training_history[-1]['value_loss'],
        'training_history': training_history,
        'total_time': end_time - start_time,
        'memory_usage': end_memory - start_memory
    }


def run_main_pipeline(data_files, max_samples=500, epochs=50, device='cpu'):
    """Run the main training pipeline on the same data."""
    print(f"\n=== RUNNING MAIN PIPELINE ===")
    print(f"Data files: {data_files}")
    print(f"Max samples: {max_samples}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    # Create a limited dataset by creating a custom dataset that only returns the first max_samples
    class LimitedDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, max_samples):
            self.base_dataset = base_dataset
            self.max_samples = min(max_samples, len(base_dataset))
        
        def __len__(self):
            return self.max_samples
        
        def __getitem__(self, idx):
            return self.base_dataset[idx]
    
    # Load dataset
    base_dataset = StreamingProcessedDataset(data_files, chunk_size=10000, shuffle_files=False)
    limited_dataset = LimitedDataset(base_dataset, max_samples)
    
    print(f"Limited dataset size: {len(limited_dataset)}")
    
    # Create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(limited_dataset, batch_size=32, shuffle=True)
    
    # Create model
    model = create_model()
    model = model.to(device)
    
    # Create trainer (this is the key difference - using the main training pipeline)
    trainer = Trainer(
        model=model,
        train_loader=dataloader,
        val_loader=None,  # No validation for this test
        learning_rate=0.001,
        device=device,
        enable_system_analysis=False,
        enable_csv_logging=False,
        weight_decay=0.0  # Disable weight decay for comparison
    )
    
    # Disable mixed precision and gradient clipping for this test
    trainer.mixed_precision.use_mixed_precision = False
    print("  Disabled mixed precision and gradient clipping for comparison")
    print(f"  Main pipeline loss weights: policy={trainer.criterion.policy_weight}, value={trainer.criterion.value_weight}")
    print(f"  Main pipeline optimizer: {type(trainer.optimizer).__name__}, lr={trainer.optimizer.param_groups[0]['lr']}, weight_decay={trainer.optimizer.param_groups[0].get('weight_decay', 0)}")
    
    # Training loop (simplified to match overfit pipeline)
    best_accuracy = 0.0
    training_history = []
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        num_batches = 0
        
        for batch_idx, (boards_batch, policy_targets, value_targets) in enumerate(dataloader):
            boards_batch = boards_batch.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            
            # Forward pass (without mixed precision)
            policy_output, value_output = model(boards_batch)
            
            # Compute loss
            total_loss, loss_dict = trainer.criterion(policy_output, value_output, policy_targets, value_targets)
            
            # Compute accuracy
            policy_probs = torch.softmax(policy_output, dim=1)
            predicted_moves = torch.argmax(policy_probs, dim=1)
            target_moves = torch.argmax(policy_targets, dim=1)
            correct_predictions += (predicted_moves == target_moves).sum().item()
            total_predictions += predicted_moves.size(0)
            
            # Accumulate losses
            total_policy_loss += loss_dict['policy_loss']
            total_value_loss += loss_dict['value_loss']
            num_batches += 1
            
            # Backward pass (without gradient clipping)
            trainer.optimizer.zero_grad()
            total_loss.backward()
            trainer.optimizer.step()  # Direct step, no scaling
            
            # Clear memory
            del policy_output, value_output, policy_probs, predicted_moves, target_moves
            del boards_batch, policy_targets, value_targets
        
        epoch_time = time.time() - epoch_start_time
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        avg_policy_loss = total_policy_loss / num_batches if num_batches > 0 else 0.0
        avg_value_loss = total_value_loss / num_batches if num_batches > 0 else 0.0
        
        training_history.append({
            'epoch': epoch,
            'accuracy': accuracy,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'time': epoch_time
        })
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        
        if epoch % 10 == 0 or epoch in [1, 5, 10, 20, 30, 40]:
            print(f"  Epoch {epoch:3d}: policy_loss={avg_policy_loss:.6f} value_loss={avg_value_loss:.6f} accuracy={accuracy:.3f} time={epoch_time:.1f}s")
    
    end_time = time.time()
    end_memory = get_memory_usage()
    
    print(f"\nMain Pipeline Results:")
    print(f"  Best accuracy: {best_accuracy:.3f}")
    print(f"  Final policy loss: {training_history[-1]['policy_loss']:.6f}")
    print(f"  Final value loss: {training_history[-1]['value_loss']:.6f}")
    print(f"  Total time: {end_time - start_time:.1f}s")
    print(f"  Memory usage: {end_memory - start_memory:.2f}GB")
    
    return {
        'best_accuracy': best_accuracy,
        'final_policy_loss': training_history[-1]['policy_loss'],
        'final_value_loss': training_history[-1]['value_loss'],
        'training_history': training_history,
        'total_time': end_time - start_time,
        'memory_usage': end_memory - start_memory
    }


def main():
    parser = argparse.ArgumentParser(description='Compare overfit and main training pipelines')
    parser.add_argument('pkl_path', help='Path to .pkl.gz file with training data')
    parser.add_argument('--max-samples', type=int, default=500, help='Maximum number of samples to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--device', default='auto', help='Device to use (auto, cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)
    
    print(f"Comparing training pipelines on {args.pkl_path}")
    print(f"Max samples: {args.max_samples}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {device}")
    print(f"Initial memory: {get_memory_usage():.2f} GB")
    
    data_files = [Path(args.pkl_path)]
    
    # Run both pipelines
    overfit_results = run_overfit_pipeline(data_files, args.max_samples, args.epochs, device)
    
    # Clear memory between runs
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    main_results = run_main_pipeline(data_files, args.max_samples, args.epochs, device)
    
    # Compare results
    print(f"\n=== COMPARISON ===")
    print(f"Overfit Pipeline:")
    print(f"  Best accuracy: {overfit_results['best_accuracy']:.3f}")
    print(f"  Final policy loss: {overfit_results['final_policy_loss']:.6f}")
    print(f"  Final value loss: {overfit_results['final_value_loss']:.6f}")
    print(f"  Time: {overfit_results['total_time']:.1f}s")
    print(f"  Memory: {overfit_results['memory_usage']:.2f}GB")
    
    print(f"\nMain Pipeline:")
    print(f"  Best accuracy: {main_results['best_accuracy']:.3f}")
    print(f"  Final policy loss: {main_results['final_policy_loss']:.6f}")
    print(f"  Final value loss: {main_results['final_value_loss']:.6f}")
    print(f"  Time: {main_results['total_time']:.1f}s")
    print(f"  Memory: {main_results['memory_usage']:.2f}GB")
    
    print(f"\nDifferences:")
    print(f"  Accuracy difference: {overfit_results['best_accuracy'] - main_results['best_accuracy']:.3f}")
    print(f"  Policy loss difference: {overfit_results['final_policy_loss'] - main_results['final_policy_loss']:.6f}")
    print(f"  Value loss difference: {overfit_results['final_value_loss'] - main_results['final_value_loss']:.6f}")
    print(f"  Time difference: {overfit_results['total_time'] - main_results['total_time']:.1f}s")
    print(f"  Memory difference: {overfit_results['memory_usage'] - main_results['memory_usage']:.2f}GB")


if __name__ == "__main__":
    main() 