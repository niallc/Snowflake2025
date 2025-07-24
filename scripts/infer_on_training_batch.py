#!/usr/bin/env python3
"""
Script to run inference on a dumped training batch and compare value head predictions to targets.
"""
import argparse
import os
import pickle
import torch
import numpy as np
from hex_ai.inference.model_wrapper import ModelWrapper

def main():
    parser = argparse.ArgumentParser(description="Run inference on a dumped training batch.")
    parser.add_argument('--batch_file', type=str, required=True, help='Path to dumped batch .pkl file')
    parser.add_argument('--checkpoint', type=str, default="checkpoints/hyperparameter_tuning/loss_weight_sweep_exp2_do0_pw0.001_f537d4_20250722_211936/epoch1_mini15.pt", help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu, cuda, mps)')
    parser.add_argument('--model_type', type=str, default="resnet18", help='Model type (default: resnet18)')
    args = parser.parse_args()

    print(f"Loading batch from: {args.batch_file}")
    with open(args.batch_file, 'rb') as f:
        batch = pickle.load(f)

    # Use the CPU tensors for safety
    boards = batch['boards_cpu']
    values = batch['values_cpu']
    policies = batch['policies_cpu'] if 'policies_cpu' in batch else None
    print(f"Batch shape: boards {boards.shape}, values {values.shape}")

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = ModelWrapper(args.checkpoint, device=args.device, model_type=args.model_type)

    # Run inference
    print("Running inference on batch...")
    with torch.no_grad():
        policy_logits, value_logits = model.batch_predict(boards)
        value_probs = torch.sigmoid(value_logits).squeeze(-1).numpy()  # shape (batch_size,)
        targets = values.squeeze(-1).numpy() if values.ndim > 1 else values.numpy()

    # Print results
    print("\nValue head predictions vs targets:")
    for i, (pred, target) in enumerate(zip(value_probs, targets)):
        print(f"Sample {i:3d}: Predicted value={pred:.4f} | Target={target:.4f} | Error={abs(pred-target):.4f}")
    mse = np.mean((value_probs - targets) ** 2)
    print(f"\nBatch MSE: {mse:.6f}")
    print(f"Batch mean abs error: {np.mean(np.abs(value_probs - targets)):.6f}")
    print(f"Batch mean target: {np.mean(targets):.4f}, mean prediction: {np.mean(value_probs):.4f}")

    # Optionally, print policy head info
    if policies is not None:
        print("\nPolicy head output (first sample):")
        print(policy_logits[0])
        print("Target policy (first sample):")
        print(policies[0])

if __name__ == "__main__":
    main() 