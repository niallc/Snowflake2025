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
    boards = batch['boards'] if 'boards' in batch else batch['boards_cpu']
    values = batch['values'] if 'values' in batch else batch['values_cpu']
    policies = batch.get('policies', batch.get('policies_cpu', None))
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
    summary_lines = []
    for i, (pred, target) in enumerate(zip(value_probs, targets)):
        line = f"Sample {i:3d}: Predicted value={pred:.4f} | Target={target:.4f} | Error={abs(pred-target):.4f}"
        print(line)
        summary_lines.append(line)
    mse = np.mean((value_probs - targets) ** 2)
    mae = np.mean(np.abs(value_probs - targets))
    mean_target = np.mean(targets)
    mean_pred = np.mean(value_probs)
    summary_lines.append(f"\nBatch MSE: {mse:.6f}")
    summary_lines.append(f"Batch mean abs error: {mae:.6f}")
    summary_lines.append(f"Batch mean target: {mean_target:.4f}, mean prediction: {mean_pred:.4f}")

    # If detailed training-time predictions are available, compare them
    if 'value_logits' in batch:
        train_value_logits = batch['value_logits'].squeeze(-1).numpy()
        train_value_probs = torch.sigmoid(torch.tensor(train_value_logits)).numpy()
        # Compare inference vs training-time predictions
        summary_lines.append("\nComparison to training-time predictions:")
        for i, (train_pred, infer_pred) in enumerate(zip(train_value_probs, value_probs)):
            line = f"Sample {i:3d}: Training value={train_pred:.4f} | Inference value={infer_pred:.4f} | Diff={abs(train_pred-infer_pred):.4f}"
            print(line)
            summary_lines.append(line)
        diff_mse = np.mean((train_value_probs - value_probs) ** 2)
        diff_mae = np.mean(np.abs(train_value_probs - value_probs))
        summary_lines.append(f"\nInference vs Training value MSE: {diff_mse:.6f}")
        summary_lines.append(f"Inference vs Training value mean abs error: {diff_mae:.6f}")
        summary_lines.append(f"Training mean: {np.mean(train_value_probs):.4f}, Inference mean: {np.mean(value_probs):.4f}")

    # Optionally, print policy head info
    if policies is not None:
        summary_lines.append("\nPolicy head output (first sample):")
        summary_lines.append(str(policy_logits[0]))
        summary_lines.append("Target policy (first sample):")
        summary_lines.append(str(policies[0]))

    # Save summary to file
    import hashlib
    def short_hash(s):
        return hashlib.sha1(s.encode()).hexdigest()[:8]
    batch_base = os.path.basename(args.batch_file).replace('.pkl', '')
    checkpoint_base = os.path.basename(args.checkpoint).replace('.pt', '')
    summary_filename = f"analysis/debugging/value_head_performance/infer_summary_{batch_base}_{checkpoint_base}_{short_hash(args.batch_file + args.checkpoint)}.txt"
    with open(summary_filename, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"\nSummary written to: {summary_filename}")

if __name__ == "__main__":
    main() 