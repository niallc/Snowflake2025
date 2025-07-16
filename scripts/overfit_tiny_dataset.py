#!/usr/bin/env python3
"""
Overfit the model on a tiny dataset (e.g., one game from a .pkl.gz file) to sanity-check the model and training loop.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from hex_ai.data_pipeline import StreamingProcessedDataset
from hex_ai.models import create_model
from hex_ai.training_utils import get_device
from hex_ai.training import PolicyValueLoss
import torch.optim as optim


def main():
    parser = argparse.ArgumentParser(description="Overfit the model on a tiny dataset (one game) to sanity-check training.")
    parser.add_argument('file_path', type=str, help='Path to .pkl.gz file (should contain one game)')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay (L2 regularization)')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu, cuda, mps)')
    args = parser.parse_args()

    # Load the tiny dataset (all moves from one game)
    dataset = StreamingProcessedDataset([Path(args.file_path)], chunk_size=10000, shuffle_files=False)
    # Load all samples into memory (should be small)
    all_samples = [dataset[i] for i in range(len(dataset))]
    boards = torch.stack([b for b, p, v in all_samples])
    policies = torch.stack([p for b, p, v in all_samples])
    values = torch.stack([v for b, p, v in all_samples])
    print(f"Loaded {len(boards)} samples from {args.file_path}")

    # Make a single batch
    batch = (boards, policies, values)

    # Model setup
    device = args.device or get_device()
    model = create_model()
    model.dropout.p = args.dropout
    model = model.to(device)
    model.train()

    # Loss and optimizer
    criterion = PolicyValueLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        boards, policies, values = [x.to(device) for x in batch]
        optimizer.zero_grad()
        policy_pred, value_pred = model(boards)
        total_loss, loss_dict = criterion(policy_pred, value_pred, policies, values)
        total_loss.backward()
        optimizer.step()
        if epoch % 100 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch {epoch:6d}: total_loss={loss_dict['total_loss']:.6f} policy_loss={loss_dict['policy_loss']:.6f} value_loss={loss_dict['value_loss']:.6f}")
    print("Training complete.")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        boards, policies, values = [x.to(device) for x in batch]
        policy_pred, value_pred = model(boards)
        total_loss, loss_dict = criterion(policy_pred, value_pred, policies, values)
        print("Final evaluation:")
        print(f"total_loss={loss_dict['total_loss']:.6f} policy_loss={loss_dict['policy_loss']:.6f} value_loss={loss_dict['value_loss']:.6f}")
        # Print predicted vs target value for each sample
        print("Sample-by-sample value predictions:")
        for i in range(len(values)):
            print(f"Sample {i}: target={values[i].item():.3f} pred={value_pred[i].item():.3f}")

if __name__ == "__main__":
    main() 