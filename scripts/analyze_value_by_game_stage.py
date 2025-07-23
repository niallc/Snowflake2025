#!/usr/bin/env python3
"""
Analyze value head performance across different game stages.

This script loads a trained model and analyzes its value predictions
across early, mid, and late game positions to understand where the
value head is struggling.
"""

import argparse
import sys
import os
from pathlib import Path
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hex_ai.inference.model_wrapper import ModelWrapper
from hex_ai.data_pipeline import StreamingSequentialShardDataset, discover_processed_files
from hex_ai.training_utils import get_device

# CONSTANTS
MODEL_BASE_DIR = "checkpoints/hyperparameter_tuning"
# Alternative model names:
# loss_weight_sweep_exp0_do0_pw0.2_794e88_20250722_211936
# loss_weight_sweep_exp1_do0_pw0.7_55b280_20250722_211936
# In general find find recent results from July 23rd directories in MODEL_BASE_DIR
MODEL_NAME = "loss_weight_sweep_exp0_do0_pw0.2_794e88_20250722_211936"
MODEL_FILE_BASENAME = "epoch1_mini100.pt"
FULL_MODEL_PATH = f"{MODEL_BASE_DIR}/{MODEL_NAME}/{MODEL_FILE_BASENAME}"
DATA_DIR = "data/processed/shuffled/"
# DATA_FILE = "shuffled_0000.pkl.gz"
SAVE_DIR = f"analysis/value_by_stage/{MODEL_NAME}"
NUM_SAMPLES = 10000
DEVICE = "mps"
ENABLE_AUGMENTATION = True

def categorize_position(board):
    """Categorize a position as early, mid, or late game based on piece count."""
    # Count pieces on the board
    blue_pieces = np.sum(board[0] > 0)  # Blue channel
    red_pieces = np.sum(board[1] > 0)   # Red channel
    total_pieces = blue_pieces + red_pieces
    
    # Categorize based on total pieces
    if total_pieces < 20:
        return 'early'
    elif total_pieces < 80:
        return 'mid'
    else:
        return 'late'


def analyze_value_predictions(model, dataset, num_samples=10000, device='cpu'):
    """Analyze value predictions across different game stages."""
    model.eval()
    
    # Collect predictions by game stage
    stage_predictions = defaultdict(list)
    stage_targets = defaultdict(list)
    stage_boards = defaultdict(list)
    
    print(f"Analyzing {num_samples} positions...")
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            if i % 1000 == 0:
                print(f"  Processed {i}/{num_samples} positions")
            
            board, policy, value = dataset[i]
            board = board.unsqueeze(0).to(device)
            
            # Get model prediction
            policy_pred, value_pred = model.predict(board)
            value_prob = torch.sigmoid(torch.tensor(value_pred)).item()
            
            # Categorize position
            stage = categorize_position(board[0].cpu().numpy())
            
            # Store results
            stage_predictions[stage].append(value_prob)
            stage_targets[stage].append(value.item())
            stage_boards[stage].append(board[0].cpu().numpy())
    
    # Analyze results
    results = {}
    for stage in ['early', 'mid', 'late']:
        if stage_predictions[stage]:
            preds = np.array(stage_predictions[stage])
            targets = np.array(stage_targets[stage])
            
            # Calculate metrics
            mse = np.mean((preds - targets) ** 2)
            mae = np.mean(np.abs(preds - targets))
            
            # Calculate accuracy (predictions within 0.3 of target)
            accuracy = np.mean(np.abs(preds - targets) < 0.3)
            
            # Calculate bias (mean prediction - mean target)
            bias = np.mean(preds) - np.mean(targets)
            
            # Distribution statistics
            pred_mean = np.mean(preds)
            pred_std = np.std(preds)
            target_mean = np.mean(targets)
            target_std = np.std(targets)
            
            results[stage] = {
                'mse': mse,
                'mae': mae,
                'accuracy': accuracy,
                'bias': bias,
                'pred_mean': pred_mean,
                'pred_std': pred_std,
                'target_mean': target_mean,
                'target_std': target_std,
                'sample_count': len(preds),
                'predictions': preds.tolist(),
                'targets': targets.tolist()
            }
    
    return results


def create_visualizations(results, save_dir):
    """Create visualizations of the analysis results."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Performance metrics by stage
    stages = list(results.keys())
    metrics = ['mse', 'mae', 'accuracy']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        values = [results[stage][metric] for stage in stages]
        axes[i].bar(stages, values)
        axes[i].set_title(f'{metric.upper()} by Game Stage')
        axes[i].set_ylabel(metric.upper())
        
        # Add value labels on bars
        for j, v in enumerate(values):
            axes[i].text(j, v + max(values) * 0.01, f'{v:.3f}', 
                        ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path / 'performance_by_stage.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Prediction vs Target distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, stage in enumerate(stages):
        preds = results[stage]['predictions']
        targets = results[stage]['targets']
        
        axes[i].scatter(targets, preds, alpha=0.5, s=1)
        axes[i].plot([0, 1], [0, 1], 'r--', alpha=0.5)  # Perfect prediction line
        axes[i].set_xlabel('True Value')
        axes[i].set_ylabel('Predicted Value')
        axes[i].set_title(f'{stage.capitalize()} Game')
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'prediction_vs_target.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Prediction distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, stage in enumerate(stages):
        preds = results[stage]['predictions']
        targets = results[stage]['targets']
        
        axes[i].hist(preds, bins=50, alpha=0.7, label='Predictions', density=True)
        axes[i].hist(targets, bins=50, alpha=0.7, label='Targets', density=True)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'{stage.capitalize()} Game Distribution')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'prediction_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze value head performance by game stage")
    parser.add_argument('--model_path', type=str, 
                        default=FULL_MODEL_PATH,
                        required=False, help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Data directory')
    parser.add_argument('--max_data_files', type=int, default=1, help='Maximum number of data files to use')
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR, help='Save directory')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to analyze')
    parser.add_argument('--device', type=str, default=DEVICE, help='Device to use')
    parser.add_argument('--enable_augmentation', action='store_true', help='Enable data augmentation')
    
    args = parser.parse_args()
    
    # Setup
    device = get_device() if args.device is None else torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model_wrapper = ModelWrapper(args.model_path, device=device)
    
    # Load data
    print("Loading data...")
    data_files = discover_processed_files(args.data_dir, max_files=args.max_data_files)
    print(f"Found {len(data_files)} data files")
    print(f"Using {args.max_data_files} data files (full data, >97million, examples, is far larger than required)")
    input_data = data_files[:args.max_data_files]

    dataset = StreamingSequentialShardDataset(
        input_data,
        enable_augmentation=args.enable_augmentation,
        max_examples_unaugmented=args.num_samples,
    )
    
    # Analyze
    print("Analyzing value predictions by game stage...")
    results = analyze_value_predictions(
        model_wrapper, 
        dataset, 
        num_samples=args.num_samples,
        device=device
    )
    
    # Print summary
    print("\n=== Analysis Summary ===")
    for stage in ['early', 'mid', 'late']:
        if stage in results:
            r = results[stage]
            print(f"\n{stage.capitalize()} Game ({r['sample_count']} samples):")
            print(f"  MSE: {r['mse']:.4f}")
            print(f"  MAE: {r['mae']:.4f}")
            print(f"  Accuracy: {r['accuracy']:.2%}")
            print(f"  Bias: {r['bias']:.4f}")
            print(f"  Pred Mean: {r['pred_mean']:.4f} ± {r['pred_std']:.4f}")
            print(f"  Target Mean: {r['target_mean']:.4f} ± {r['target_std']:.4f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(results, args.save_dir)
    
    # Save results
    results_file = Path(args.save_dir) / "analysis_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nAnalysis complete! Results saved to {args.save_dir}")
    print(f"  - JSON results: {results_file}")
    print(f"  - Visualizations: {args.save_dir}/")


if __name__ == "__main__":
    main() 