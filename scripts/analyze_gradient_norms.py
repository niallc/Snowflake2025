#!/usr/bin/env python3
"""
Analyze gradient norms during training to understand the impact of gradient clipping.
"""

import argparse
import torch
import numpy as np
import time
import os
from pathlib import Path

# Add the parent directory to the path so we can import from hex_ai
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hex_ai.data_pipeline import StreamingProcessedDataset
from hex_ai.models import create_model
from hex_ai.training_utils import get_device
from hex_ai.training import PolicyValueLoss


def analyze_gradient_norms(data_files, max_samples=1000, epochs=5, device='cpu'):
    """Analyze gradient norms during training."""
    print(f"Analyzing gradient norms on {max_samples} samples for {epochs} epochs")
    print(f"Device: {device}")
    
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    criterion = PolicyValueLoss()
    
    # Track gradient norms
    gradient_norms = []
    clipped_gradient_norms = []
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}:")
        epoch_grad_norms = []
        epoch_clipped_norms = []
        
        for batch_idx, (boards_batch, policy_targets, value_targets) in enumerate(dataloader):
            boards_batch = boards_batch.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            
            # Forward pass
            policy_output, value_output = model(boards_batch)
            total_loss, loss_dict = criterion(policy_output, value_output, policy_targets, value_targets)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Calculate gradient norm before clipping
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            epoch_grad_norms.append(total_norm.item())
            
            # Calculate gradient norm after clipping to 1.0
            clipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            epoch_clipped_norms.append(clipped_norm.item())
            
            # Optimizer step
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: grad_norm={total_norm:.4f}, clipped_norm={clipped_norm:.4f}, "
                      f"clipping_ratio={clipped_norm/total_norm:.3f}")
        
        # Epoch statistics
        avg_grad_norm = np.mean(epoch_grad_norms)
        avg_clipped_norm = np.mean(epoch_clipped_norms)
        avg_clipping_ratio = avg_clipped_norm / avg_grad_norm
        
        print(f"  Epoch {epoch} stats:")
        print(f"    Avg gradient norm: {avg_grad_norm:.4f}")
        print(f"    Avg clipped norm: {avg_clipped_norm:.4f}")
        print(f"    Avg clipping ratio: {avg_clipping_ratio:.3f}")
        print(f"    Max gradient norm: {np.max(epoch_grad_norms):.4f}")
        print(f"    Min gradient norm: {np.min(epoch_grad_norms):.4f}")
        
        gradient_norms.extend(epoch_grad_norms)
        clipped_gradient_norms.extend(epoch_clipped_norms)
    
    # Overall statistics
    print(f"\n=== OVERALL STATISTICS ===")
    print(f"Total batches: {len(gradient_norms)}")
    print(f"Overall avg gradient norm: {np.mean(gradient_norms):.4f}")
    print(f"Overall avg clipped norm: {np.mean(clipped_gradient_norms):.4f}")
    print(f"Overall avg clipping ratio: {np.mean(clipped_gradient_norms) / np.mean(gradient_norms):.3f}")
    print(f"Max gradient norm seen: {np.max(gradient_norms):.4f}")
    print(f"Min gradient norm seen: {np.min(gradient_norms):.4f}")
    
    # Count how often gradients are clipped
    clipping_ratios = [clipped / original for original, clipped in zip(gradient_norms, clipped_gradient_norms)]
    heavily_clipped = sum(1 for ratio in clipping_ratios if ratio < 0.5)
    moderately_clipped = sum(1 for ratio in clipping_ratios if 0.5 <= ratio < 0.9)
    lightly_clipped = sum(1 for ratio in clipping_ratios if 0.9 <= ratio < 1.0)
    not_clipped = sum(1 for ratio in clipping_ratios if ratio >= 1.0)
    
    print(f"\nClipping frequency:")
    print(f"  Heavily clipped (<50%): {heavily_clipped} batches ({heavily_clipped/len(clipping_ratios)*100:.1f}%)")
    print(f"  Moderately clipped (50-90%): {moderately_clipped} batches ({moderately_clipped/len(clipping_ratios)*100:.1f}%)")
    print(f"  Lightly clipped (90-100%): {lightly_clipped} batches ({lightly_clipped/len(clipping_ratios)*100:.1f}%)")
    print(f"  Not clipped (100%): {not_clipped} batches ({not_clipped/len(clipping_ratios)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Analyze gradient norms during training')
    parser.add_argument('pkl_path', help='Path to .pkl.gz file with training data')
    parser.add_argument('--max-samples', type=int, default=1000, help='Maximum number of samples to use')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--device', default='auto', help='Device to use (auto, cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)
    
    data_files = [Path(args.pkl_path)]
    
    analyze_gradient_norms(data_files, args.max_samples, args.epochs, device)


if __name__ == "__main__":
    main() 