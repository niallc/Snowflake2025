#!/usr/bin/env python3
"""
Overfit the model on a tiny dataset (e.g., one game from a .pkl.gz file) to sanity-check the model and training loop.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gc
import psutil
import os
import time
from pathlib import Path

# Add the parent directory to the path so we can import from hex_ai
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hex_ai.data_pipeline import StreamingProcessedDataset
from hex_ai.models import create_model
from hex_ai.training_utils import get_device
from hex_ai.training import PolicyValueLoss
from hex_ai.utils.format_conversion import tensor_to_rowcol, tensor_to_trmph
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


def calculate_policy_accuracy(policy_pred, policy_target):
    """Calculate accuracy of policy predictions."""
    if policy_target is None:
        return 0.0
    pred_move = torch.argmax(policy_pred, dim=1)
    target_move = torch.argmax(policy_target, dim=1)
    correct = (pred_move == target_move).float().mean()
    return correct.item()


def get_top_policy_predictions(policy_probs, k=3):
    """Get top k policy predictions with probabilities."""
    top_probs, top_indices = torch.topk(policy_probs, k, dim=1)
    return top_indices, top_probs


def main():
    parser = argparse.ArgumentParser(description='Overfit on a tiny dataset to verify model and training')
    parser.add_argument('pkl_path', help='Path to .pkl.gz file with training data')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--early-stopping-patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--device', default='auto', help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay (L2 regularization)')
    parser.add_argument('--early-stop-accuracy', type=float, default=0.95, help='Stop training if accuracy exceeds this value')
    parser.add_argument('--max-samples', type=int, default=1000, help='Maximum number of samples to load (for memory control)')
    
    args = parser.parse_args()
    
    print(f"Initial memory usage: {get_memory_usage():.2f} GB")
    print(f"GPU memory: {get_gpu_memory_usage()}")
    
    # Load the tiny dataset using StreamingProcessedDataset (the correct way)
    dataset = StreamingProcessedDataset([Path(args.pkl_path)], chunk_size=10000, shuffle_files=False)
    print(f"Dataset length: {len(dataset)}")
    
    # Load a limited number of samples to control memory usage
    print(f"Loading up to {args.max_samples} samples from dataset...")
    all_samples = []
    for i in range(min(args.max_samples, len(dataset))):
        try:
            sample = dataset[i]
            all_samples.append(sample)
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            break
    
    print(f"Loaded {len(all_samples)} samples")
    print(f"Memory after loading samples: {get_memory_usage():.2f} GB")
    
    # Extract boards, policies, and values
    boards = torch.stack([b for b, p, v in all_samples])
    policies = torch.stack([p for b, p, v in all_samples])
    values = torch.stack([v for b, p, v in all_samples])
    
    print(f"Board tensor shape: {boards.shape}, size: {boards.element_size() * boards.numel() / 1024 / 1024:.2f} MB")
    print(f"Policy tensor shape: {policies.shape}, size: {policies.element_size() * policies.numel() / 1024 / 1024:.2f} MB")
    print(f"Value tensor shape: {values.shape}, size: {values.element_size() * values.numel() / 1024 / 1024:.2f} MB")
    print(f"Memory after creating tensors: {get_memory_usage():.2f} GB")
    
    # Determine device
    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create model
    model = create_model()
    model.dropout.p = args.dropout
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Memory after model creation: {get_memory_usage():.2f} GB")
    
    # Create dataset and dataloader with reasonable batch size
    from torch.utils.data import TensorDataset, DataLoader
    dataset_tensor = TensorDataset(boards, policies, values)
    dataloader = DataLoader(dataset_tensor, batch_size=args.batch_size, shuffle=True)
    
    print(f"Number of batches: {len(dataloader)}")
    
    # Setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = PolicyValueLoss()
    
    print(f"Starting training with {len(boards)} samples...")
    print(f"Memory before training loop: {get_memory_usage():.2f} GB")
    
    # Training loop
    best_accuracy = 0.0
    patience_counter = 0
    training_history = []
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        num_batches = 0
        
        for batch_idx, (boards_batch, policy_targets, value_targets) in enumerate(dataloader):
            # Move to device
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
            
            # Clear intermediate tensors to save memory
            del policy_output, value_output, policy_probs, predicted_moves, target_moves
            del boards_batch, policy_targets, value_targets
        
        epoch_time = time.time() - epoch_start_time
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        avg_policy_loss = total_policy_loss / num_batches if num_batches > 0 else 0.0
        avg_value_loss = total_value_loss / num_batches if num_batches > 0 else 0.0
        
        # Store training history
        training_history.append({
            'epoch': epoch,
            'accuracy': accuracy,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'time': epoch_time
        })
        
        # Early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch} with best accuracy: {best_accuracy:.3f}")
            break
        
        if accuracy >= args.early_stop_accuracy:
            print(f"Early stopping at epoch {epoch}: accuracy {accuracy:.3f} >= {args.early_stop_accuracy}")
            break
        
        # Logging with different frequencies
        if epoch % 50 == 0 or epoch in [1,2,4,6,10,20,30,40] or epoch == args.epochs:
            print(f"Epoch {epoch:3d}: policy_loss={avg_policy_loss:.6f} value_loss={avg_value_loss:.6f} accuracy={accuracy:.3f} time={epoch_time:.1f}s memory={get_memory_usage():.2f}GB gpu={get_gpu_memory_usage()}")
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Final analysis
    print(f"\nFinal Results:")
    print(f"Best accuracy achieved: {best_accuracy:.3f}")
    print(f"Final memory usage: {get_memory_usage():.2f} GB")
    
    # Training summary
    print(f"\nTraining Summary:")
    print(f"Total epochs: {len(training_history)}")
    print(f"Final accuracy: {training_history[-1]['accuracy']:.3f}")
    print(f"Final policy loss: {training_history[-1]['policy_loss']:.6f}")
    print(f"Final value loss: {training_history[-1]['value_loss']:.6f}")
    
    # Show accuracy progression
    if len(training_history) > 10:
        print(f"\nAccuracy progression (every 10 epochs):")
        for i in range(0, len(training_history), 10):
            epoch_info = training_history[i]
            print(f"  Epoch {epoch_info['epoch']:3d}: accuracy={epoch_info['accuracy']:.3f} policy_loss={epoch_info['policy_loss']:.6f}")
    
    # Move-by-move analysis on a few samples
    print(f"\nMove-by-move analysis (first 5 samples):")
    model.eval()
    with torch.no_grad():
        for i in range(min(5, len(boards))):
            board = boards[i:i+1].to(device)
            policy_target = policies[i:i+1].to(device)
            
            policy_output, value_output = model(board)
            policy_probs = torch.softmax(policy_output, dim=1)
            
            predicted_move = torch.argmax(policy_probs, dim=1).item()
            target_move = torch.argmax(policy_target, dim=1).item()
            
            # Get top 3 predictions
            top_indices, top_probs = get_top_policy_predictions(policy_probs, k=3)
            
            print(f"\nSample {i}:")
            print(f"  Target move: {target_move} ({tensor_to_trmph(target_move)})")
            print(f"  Predicted move: {predicted_move} ({tensor_to_trmph(predicted_move)}) - Correct: {'✓' if predicted_move == target_move else '✗'}")
            print(f"  Top 3 predictions:")
            for j in range(3):
                move_idx = top_indices[0, j].item()
                prob = top_probs[0, j].item()
                move_str = tensor_to_trmph(move_idx)
                print(f"    {j+1}. {move_idx:3d} ({move_str:4s}): {prob:.4f}")
            print(f"  Value prediction: {value_output[0].item():.3f} (target: {values[i].item():.3f})")
            
            # Display board with policy target (convert to numpy for visualization)
            try:
                visualize_board_with_policy(boards[i].cpu().numpy(), policies[i].cpu().numpy())
            except Exception as e:
                print(f"  Visualization error: {e}")

if __name__ == "__main__":
    main() 