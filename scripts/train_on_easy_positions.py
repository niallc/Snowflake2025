#!/usr/bin/env python3
"""
Train on easy positions only to test value head learning capability.

This script creates datasets of only final positions (no player turn dependency)
and penultimate positions (with player turn information) to isolate whether
the value head can learn simple cases.
"""

import argparse
import sys
import os
from pathlib import Path
import json
import torch
import numpy as np
import gzip
import pickle
from collections import defaultdict

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hex_ai.models import TwoHeadedResNet
from hex_ai.training import Trainer, PolicyValueLoss
from hex_ai.data_pipeline import discover_processed_files
from hex_ai.training_utils import get_device


class EasyPositionDataset(torch.utils.data.Dataset):
    """Dataset containing only easy positions (final and penultimate moves)."""
    
    def __init__(self, data_files, position_types=['final', 'penultimate'], max_examples=None):
        self.position_types = position_types
        self.examples = []
        
        print(f"Loading easy positions: {position_types}")
        
        for file_path in data_files:
            try:
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                file_examples = data.get('examples', [])
                
                for ex in file_examples:
                    # Check if this is a final or penultimate position
                    if self._is_easy_position(ex):
                        self.examples.append(ex)
                        
                        if max_examples and len(self.examples) >= max_examples:
                            break
                
                if max_examples and len(self.examples) >= max_examples:
                    break
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        print(f"Loaded {len(self.examples)} easy positions")
    
    def _is_easy_position(self, example):
        """Determine if this is a final or penultimate position."""
        board = example['board']
        
        # Count pieces
        blue_pieces = np.sum(board[0] > 0)
        red_pieces = np.sum(board[1] > 0)
        total_pieces = blue_pieces + red_pieces
        
        # Final position: very few empty spaces (game is over)
        empty_spaces = 169 - total_pieces
        if empty_spaces <= 5:  # Very few empty spaces
            return 'final' in self.position_types
        
        # Penultimate position: game is almost over
        if empty_spaces <= 10:  # Few empty spaces
            return 'penultimate' in self.position_types
        
        return False
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        board = torch.tensor(ex['board'], dtype=torch.float32)
        policy = torch.tensor(ex['policy'], dtype=torch.float32)
        value = torch.tensor(ex['value'], dtype=torch.float32)
        
        return board, policy, value


def create_easy_position_datasets(data_files, train_ratio=0.8, max_examples=None):
    """Create train/val splits for easy positions."""
    # Load all easy positions
    all_easy_dataset = EasyPositionDataset(data_files, max_examples=max_examples)
    
    # Split into train/val
    total_examples = len(all_easy_dataset)
    train_size = int(total_examples * train_ratio)
    val_size = total_examples - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        all_easy_dataset, [train_size, val_size]
    )
    
    print(f"Easy position split: {len(train_dataset)} train, {len(val_dataset)} val")
    
    return train_dataset, val_dataset


def analyze_easy_positions(dataset, num_samples=1000):
    """Analyze the characteristics of easy positions."""
    print(f"Analyzing {min(num_samples, len(dataset))} easy positions...")
    
    piece_counts = []
    value_distribution = []
    policy_entropy = []
    
    for i in range(min(num_samples, len(dataset))):
        board, policy, value = dataset[i]
        
        # Count pieces
        blue_pieces = torch.sum(board[0] > 0).item()
        red_pieces = torch.sum(board[1] > 0).item()
        total_pieces = blue_pieces + red_pieces
        piece_counts.append(total_pieces)
        
        # Value distribution
        value_distribution.append(value.item())
        
        # Policy entropy (measure of move uncertainty)
        policy_probs = torch.softmax(policy, dim=0)
        entropy = -torch.sum(policy_probs * torch.log(policy_probs + 1e-8)).item()
        policy_entropy.append(entropy)
    
    analysis = {
        'piece_counts': {
            'mean': np.mean(piece_counts),
            'std': np.std(piece_counts),
            'min': np.min(piece_counts),
            'max': np.max(piece_counts)
        },
        'value_distribution': {
            'mean': np.mean(value_distribution),
            'std': np.std(value_distribution),
            'zeros': np.sum(np.array(value_distribution) == 0.0),
            'ones': np.sum(np.array(value_distribution) == 1.0)
        },
        'policy_entropy': {
            'mean': np.mean(policy_entropy),
            'std': np.std(policy_entropy)
        }
    }
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Train on easy positions only")
    parser.add_argument('--data_dir', type=str, default="data/processed", help='Data directory')
    parser.add_argument('--save_dir', type=str, default="checkpoints/easy_positions", help='Save directory')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default=None, help='Device to use')
    parser.add_argument('--max_examples', type=int, default=50000, help='Max examples to use')
    parser.add_argument('--position_types', nargs='+', default=['final', 'penultimate'], 
                       choices=['final', 'penultimate'], help='Types of positions to include')
    parser.add_argument('--analyze_only', action='store_true', help='Only analyze, don\'t train')
    
    args = parser.parse_args()
    
    # Setup
    device = get_device() if args.device is None else torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    data_files = discover_processed_files(args.data_dir)
    
    # Create datasets
    train_dataset, val_dataset = create_easy_position_datasets(
        data_files, 
        max_examples=args.max_examples
    )
    
    # Analyze easy positions
    print("\n=== Easy Position Analysis ===")
    train_analysis = analyze_easy_positions(train_dataset)
    val_analysis = analyze_easy_positions(val_dataset)
    
    print("Train dataset characteristics:")
    print(f"  Piece counts: {train_analysis['piece_counts']['mean']:.1f} ± {train_analysis['piece_counts']['std']:.1f}")
    print(f"  Value distribution: {train_analysis['value_distribution']['mean']:.3f} ± {train_analysis['value_distribution']['std']:.3f}")
    print(f"  Value zeros: {train_analysis['value_distribution']['zeros']}, ones: {train_analysis['value_distribution']['ones']}")
    print(f"  Policy entropy: {train_analysis['policy_entropy']['mean']:.3f} ± {train_analysis['policy_entropy']['std']:.3f}")
    
    if args.analyze_only:
        print("\nAnalysis complete!")
        return
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Model
    print("\nCreating model...")
    model = TwoHeadedResNet(dropout_prob=0.1)
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        enable_csv_logging=True,
        experiment_name=f"easy_positions_{'_'.join(args.position_types)}"
    )
    
    # Training
    print(f"\nStarting training on easy positions ({args.position_types})...")
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save analysis results
    analysis_results = {
        'train_analysis': train_analysis,
        'val_analysis': val_analysis,
        'position_types': args.position_types,
        'max_examples': args.max_examples
    }
    
    with open(save_path / "easy_position_analysis.json", 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    # Train
    results = trainer.train(
        num_epochs=args.num_epochs,
        save_dir=str(save_path)
    )
    
    print(f"\nTraining complete! Results saved to {args.save_dir}")
    
    # Final analysis
    print("\n=== Final Model Analysis ===")
    trainer.model.eval()
    
    # Test on a few easy positions
    with torch.no_grad():
        for i, (board, policy, value) in enumerate(val_dataset):
            if i >= 5:  # Test first 5 positions
                break
            
            board = board.unsqueeze(0).to(device)
            policy_pred, value_pred = trainer.model(board)
            value_prob = torch.sigmoid(value_pred).item()
            
            print(f"Position {i+1}: True={value.item():.3f}, Pred={value_prob:.3f}, Error={abs(value_prob - value.item()):.3f}")


if __name__ == "__main__":
    main() 