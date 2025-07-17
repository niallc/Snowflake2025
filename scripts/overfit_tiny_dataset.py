#!/usr/bin/env python3
"""
Overfit the model on a tiny dataset (e.g., one game from a .pkl.gz file) to sanity-check the model and training loop.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import gzip
import numpy as np
import argparse
import sys
import os
import gc
import psutil
import time

# Add the parent directory to the path so we can import from hex_ai
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hex_ai.models import TwoHeadedResNet
from hex_ai.training import PolicyValueLoss
from scripts.lib.board_viz_utils import visualize_board_with_policy

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def get_gpu_memory_usage():
    """Get GPU memory usage if available"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS doesn't have a direct memory API, but we can check if it's being used
        return "MPS (no direct memory API)"
    else:
        return "No GPU"

def get_tensor_memory_usage(tensors):
    """Calculate total memory usage of tensors in GB"""
    total_memory = 0
    for name, tensor in tensors.items():
        if tensor is not None:
            # Calculate memory based on dtype and shape
            if tensor.dtype == torch.float32:
                element_size = 4
            elif tensor.dtype == torch.float64:
                element_size = 8
            elif tensor.dtype == torch.int64:
                element_size = 8
            elif tensor.dtype == torch.int32:
                element_size = 4
            else:
                element_size = 4  # default assumption
            
            memory_bytes = tensor.numel() * element_size
            total_memory += memory_bytes / 1024 / 1024 / 1024
    return total_memory

def load_samples_from_pkl(pkl_path):
    """Load samples from a .pkl.gz file"""
    print(f"Loading samples from {pkl_path}...")
    
    with gzip.open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Type of loaded data: {type(data)}")
    
    # Handle the dict format with 'examples' key
    if isinstance(data, dict) and 'examples' in data:
        examples = data['examples']
        print(f"Found {len(examples)} examples in data['examples']")
        
        # Extract board states, policy targets, and value targets
        boards = []
        policy_targets = []
        value_targets = []
        
        for board, policy, value in examples:
            boards.append(board)
            policy_targets.append(policy)
            value_targets.append(value)
        
        return boards, policy_targets, value_targets
    else:
        # Handle the original list-of-dicts format
        print(f"Loaded {len(data)} samples from {pkl_path}")
        
        boards = []
        policy_targets = []
        value_targets = []
        
        for sample in data:
            boards.append(sample['board'])
            policy_targets.append(sample['policy_target'])
            value_targets.append(sample['value_target'])
        
        return boards, policy_targets, value_targets

def main():
    parser = argparse.ArgumentParser(description='Overfit on a tiny dataset to verify model and training')
    parser.add_argument('pkl_path', help='Path to .pkl.gz file with training data')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=10000, help='Batch size (default: use all data as one batch)')
    parser.add_argument('--early-stopping-patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--device', default='auto', help='Device to use (auto, cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    print(f"Initial memory usage: {get_memory_usage():.2f} GB")
    print(f"GPU memory: {get_gpu_memory_usage()}")
    
    # Load samples
    boards, policy_targets, value_targets = load_samples_from_pkl(args.pkl_path)
    
    print(f"Dataset length: {len(boards)}")
    print(f"Loaded {len(boards)} samples")
    print(f"Memory after loading samples: {get_memory_usage():.2f} GB")
    
    # Convert to tensors
    boards_tensor = torch.tensor(np.array(boards), dtype=torch.float32)
    policy_targets_tensor = torch.tensor(np.array(policy_targets), dtype=torch.float32)
    value_targets_tensor = torch.tensor(np.array(value_targets), dtype=torch.float32)
    
    print(f"Board tensor shape: {boards_tensor.shape}, size: {boards_tensor.element_size() * boards_tensor.numel() / 1024 / 1024:.2f} MB")
    print(f"Policy tensor shape: {policy_targets_tensor.shape}, size: {policy_targets_tensor.element_size() * policy_targets_tensor.numel() / 1024 / 1024:.2f} MB")
    print(f"Value tensor shape: {value_targets_tensor.shape}, size: {value_targets_tensor.element_size() * value_targets_tensor.numel() / 1024 / 1024:.2f} MB")
    print(f"Memory after creating tensors: {get_memory_usage():.2f} GB")
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create model
    model = TwoHeadedResNet()
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Memory after model creation: {get_memory_usage():.2f} GB")
    
    # Create dataset and dataloader
    dataset = TensorDataset(boards_tensor, policy_targets_tensor, value_targets_tensor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = PolicyValueLoss()
    
    print(f"Starting training with {len(boards)} samples...")
    print(f"Memory before training loop: {get_memory_usage():.2f} GB")
    
    # Training loop
    best_accuracy = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, (boards, policy_targets, value_targets) in enumerate(dataloader):
            boards = boards.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            
            # Forward pass
            policy_output, value_output = model(boards)
            
            # Compute loss
            total_loss, loss_dict = criterion(policy_output, value_output, policy_targets, value_targets)
            
            # Compute accuracy
            policy_probs = torch.softmax(policy_output, dim=1)
            predicted_moves = torch.argmax(policy_probs, dim=1)
            target_moves = torch.argmax(policy_targets, dim=1)
            correct_predictions += (predicted_moves == target_moves).sum().item()
            total_predictions += predicted_moves.size(0)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        # Early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch} with best accuracy: {best_accuracy:.3f}")
            break
        
        if epoch % 100 == 0 or epoch in [1,2,4,6,10,30,65] or epoch == args.epochs:
            # Track all tensors in memory
            tensor_memory = get_tensor_memory_usage({
                'boards': boards_tensor,
                'policy_targets': policy_targets_tensor,
                'value_targets': value_targets_tensor,
                'policy_output': policy_output,
                'value_output': value_output,
                'policy_probs': policy_probs,
                'predicted_moves': predicted_moves,
                'target_moves': target_moves
            })
            
            print(f"Epoch {epoch:6d}: total_loss={loss_dict['total_loss']:.6f} policy_loss={loss_dict['policy_loss']:.6f} value_loss={loss_dict['value_loss']:.6f} accuracy={accuracy:.3f} memory={get_memory_usage():.2f}GB tensor_memory={tensor_memory:.2f}GB gpu={get_gpu_memory_usage()}")
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Final analysis
    print(f"\nFinal Results:")
    print(f"Best accuracy achieved: {best_accuracy:.3f}")
    print(f"Final memory usage: {get_memory_usage():.2f} GB")
    
    # Move-by-move analysis on a few samples
    print(f"\nMove-by-move analysis (first 5 samples):")
    model.eval()
    with torch.no_grad():
        for i in range(min(5, len(boards))):
            board = boards_tensor[i:i+1].to(device)
            policy_target = policy_targets_tensor[i:i+1].to(device)
            
            policy_output, value_output = model(board)
            policy_probs = torch.softmax(policy_output, dim=1)
            
            predicted_move = torch.argmax(policy_probs, dim=1).item()
            target_move = torch.argmax(policy_target, dim=1).item()
            
            print(f"Sample {i}: Predicted={predicted_move}, Target={target_move}, Correct={predicted_move == target_move}")
            
            # Display board with policy target
            visualize_board_with_policy(boards[i], policy_targets[i])

if __name__ == "__main__":
    main() 