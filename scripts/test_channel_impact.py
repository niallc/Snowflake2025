#!/usr/bin/env python3
"""
Test the impact of the 3rd channel (player-to-move) on training performance.

This script compares training with 2 channels vs 3 channels on the same data,
using a simplified training loop to isolate the effect of the channel change.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hex_ai.models import TwoHeadedResNet
from hex_ai.data_pipeline import StreamingAugmentedProcessedDataset


def create_model(num_channels, device):
    """Create a model with the specified number of input channels."""
    model = TwoHeadedResNet(
        input_channels=num_channels,
        board_size=13,
        num_residual_blocks=8,
        num_filters=64,
        policy_head_size=256,
        value_head_size=256,
        dropout_rate=0.0,  # No dropout for this test
    ).to(device)
    
    # Zero-initialize value head bias
    if hasattr(model.value_head, 'bias') and model.value_head.bias is not None:
        model.value_head.bias.data.zero_()
    
    return model


def train_single_epoch(model, dataloader, optimizer, device, num_channels):
    """Train for one epoch and return metrics."""
    model.train()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_policy_correct = 0
    total_samples = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if num_channels == 2:
            # For 2-channel test, use only the first two channels
            boards = batch[0][:, :2, :, :].to(device)  # Shape: (batch, 2, 13, 13)
        else:
            # For 3-channel test, use all channels
            boards = batch[0].to(device)  # Shape: (batch, 3, 13, 13)
        
        policy_targets = batch[1].to(device)  # Shape: (batch, 169)
        value_targets = batch[2].to(device)   # Shape: (batch, 1)
        
        optimizer.zero_grad()
        
        policy_output, value_output = model(boards)
        
        # Policy loss (cross entropy)
        policy_loss = nn.CrossEntropyLoss()(policy_output, policy_targets.argmax(dim=1))
        
        # Value loss (MSE)
        value_loss = nn.MSELoss()(value_output.squeeze(), value_targets.squeeze())
        
        # Combined loss
        loss = policy_loss + value_loss
        loss.backward()
        
        # No gradient clipping for this test
        optimizer.step()
        
        # Calculate accuracy
        policy_pred = policy_output.argmax(dim=1)
        policy_target = policy_targets.argmax(dim=1)
        policy_correct = (policy_pred == policy_target).sum().item()
        
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_policy_correct += policy_correct
        total_samples += boards.size(0)
        
        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}: Policy Loss: {policy_loss.item():.4f}, "
                  f"Value Loss: {value_loss.item():.4f}, "
                  f"Policy Acc: {policy_correct/boards.size(0)*100:.1f}%")
    
    avg_policy_loss = total_policy_loss / len(dataloader)
    avg_value_loss = total_value_loss / len(dataloader)
    avg_policy_acc = total_policy_correct / total_samples * 100
    
    return {
        'policy_loss': avg_policy_loss,
        'value_loss': avg_value_loss,
        'policy_accuracy': avg_policy_acc,
        'total_samples': total_samples
    }


def main():
    parser = argparse.ArgumentParser(description='Test impact of 3rd channel on training')
    parser.add_argument('data_file', help='Path to .pkl.gz data file')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max-samples', type=int, default=10000, help='Max samples to use')
    parser.add_argument('--device', default='auto', help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--output-dir', default='analysis/debugging/channel_test', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Device selection
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
    print(f"Testing with data file: {args.data_file}")
    print(f"Max samples: {args.max_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = StreamingAugmentedProcessedDataset(
        data_files=[Path(args.data_file)],
        chunk_size=10000,
        shuffle_files=True,
        enable_augmentation=False # Disable augmentation for this test
    )
    
    # Create a limited dataset to control sample count
    class LimitedDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, max_samples):
            self.base_dataset = base_dataset
            self.max_samples = min(max_samples, len(base_dataset))
        
        def __len__(self):
            return self.max_samples
        
        def __getitem__(self, idx):
            return self.base_dataset[idx]
    
    limited_dataset = LimitedDataset(dataset, args.max_samples)
    
    dataloader = torch.utils.data.DataLoader(
        limited_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  # Dataset already shuffles
        num_workers=0   # Keep it simple for debugging
    )
    
    print(f"Dataset size: {len(limited_dataset)} samples")
    print(f"Number of batches: {len(dataloader)}")
    
    results = {}
    
    # Test both 2-channel and 3-channel models
    for num_channels in [2, 3]:
        print(f"\n{'='*50}")
        print(f"Testing {num_channels}-channel model")
        print(f"{'='*50}")
        
        # Create model
        model = create_model(num_channels, device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
        
        # Train for specified epochs
        epoch_results = []
        start_time = time.time()
        
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            epoch_start = time.time()
            
            metrics = train_single_epoch(model, dataloader, optimizer, device, num_channels)
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
            print(f"  Value Loss: {metrics['value_loss']:.4f}")
            print(f"  Policy Accuracy: {metrics['policy_accuracy']:.2f}%")
            
            epoch_results.append(metrics)
        
        total_time = time.time() - start_time
        print(f"\nTotal training time: {total_time:.2f}s")
        
        results[f'{num_channels}_channels'] = {
            'epoch_results': epoch_results,
            'total_time': total_time,
            'final_policy_loss': epoch_results[-1]['policy_loss'],
            'final_value_loss': epoch_results[-1]['value_loss'],
            'final_policy_accuracy': epoch_results[-1]['policy_accuracy']
        }
    
    # Compare results
    print(f"\n{'='*50}")
    print("COMPARISON RESULTS")
    print(f"{'='*50}")
    
    two_ch = results['2_channels']
    three_ch = results['3_channels']
    
    print(f"2-channel final policy loss:  {two_ch['final_policy_loss']:.4f}")
    print(f"3-channel final policy loss:  {three_ch['final_policy_loss']:.4f}")
    print(f"Difference: {three_ch['final_policy_loss'] - two_ch['final_policy_loss']:.4f}")
    
    print(f"\n2-channel final policy accuracy:  {two_ch['final_policy_accuracy']:.2f}%")
    print(f"3-channel final policy accuracy:  {three_ch['final_policy_accuracy']:.2f}%")
    print(f"Difference: {three_ch['final_policy_accuracy'] - two_ch['final_policy_accuracy']:.2f}%")
    
    print(f"\n2-channel final value loss:  {two_ch['final_value_loss']:.4f}")
    print(f"3-channel final value loss:  {three_ch['final_value_loss']:.4f}")
    print(f"Difference: {three_ch['final_value_loss'] - two_ch['final_value_loss']:.4f}")
    
    # Save results
    results_file = output_dir / f"channel_test_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Determine if 3rd channel is problematic
    policy_loss_diff = three_ch['final_policy_loss'] - two_ch['final_policy_loss']
    policy_acc_diff = three_ch['final_policy_accuracy'] - two_ch['final_policy_accuracy']
    
    if policy_loss_diff > 0.5 or policy_acc_diff < -10:
        print(f"\n⚠️  WARNING: 3rd channel appears to be causing significant performance degradation!")
        print(f"   Policy loss increased by {policy_loss_diff:.4f}")
        print(f"   Policy accuracy decreased by {policy_acc_diff:.2f}%")
    else:
        print(f"\n✅ 3rd channel appears to be working fine")
        print(f"   Policy loss difference: {policy_loss_diff:.4f}")
        print(f"   Policy accuracy difference: {policy_acc_diff:.2f}%")


if __name__ == '__main__':
    main() 