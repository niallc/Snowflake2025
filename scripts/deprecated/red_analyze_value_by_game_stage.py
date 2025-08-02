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
import gzip
import pickle
import csv

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hex_ai.inference.model_wrapper import ModelWrapper
from hex_ai.data_pipeline import StreamingSequentialShardDataset, discover_processed_files
from hex_ai.training_utils import get_device
from hex_ai.data_utils import get_player_to_move_from_board, get_valid_policy_target
from hex_ai.config import PLAYER_CHANNEL, BOARD_SIZE

# CONSTANTS
MODEL_BASE_DIR = "checkpoints/hyperparameter_tuning"
MODEL_NAMES = [
    "loss_weight_sweep_exp2_do0_pw0.001_f537d4_20250722_211936",
    "loss_weight_sweep_exp1_do0_pw0.7_55b280_20250722_211936",
    "loss_weight_sweep_exp0_do0_pw0.2_794e88_20250722_211936",
]
CHECKPOINT_MINI_EPOCHS = [15, 30, 60, 85, 100]
CHECKPOINT_TEMPLATE = "epoch1_mini{}.pt"
DATA_DIR = "data/processed/shuffled/"
# DATA_FILE = "shuffled_0000.pkl.gz"
SAVE_DIR = "analysis/value_by_stage/"
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
            # The below line produced the warning:
            # /Users/niallHome/Documents/programming/Snowflake2025/scripts/analyze_value_by_game_stage.py:84: 
            # UserWarning: To copy construct from a tensor, it is recommended to use 
            # sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), 
            # rather than torch.tensor(sourceTensor).
            # Fix: use detach().clone() if value_pred is a tensor
            if isinstance(value_pred, torch.Tensor):
                value_tensor = value_pred.detach().clone()
            else:
                value_tensor = torch.tensor(value_pred)
            value_prob = torch.sigmoid(value_tensor).item()
            
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
            
            # Strict accuracy (within 0.3 of target)
            strict_accuracy = np.mean(np.abs(preds - targets) < 0.3)
            # Inverse strict accuracy: within 0.3 of the *wrong* label
            inverse_strict_accuracy = np.mean(np.abs(preds - (1 - targets)) < 0.3)
            # Classification accuracy (threshold at 0.5)
            class_preds = (preds >= 0.5).astype(np.float32)
            class_targets = (targets >= 0.5).astype(np.float32)
            class_accuracy = np.mean(class_preds == class_targets)
            
            # Confident call accuracy: among predictions with pred > 0.7 or pred < 0.3, fraction correct
            confident_mask = (preds > 0.7) | (preds < 0.3)
            if np.any(confident_mask):
                confident_correct = np.mean(np.abs(preds[confident_mask] - targets[confident_mask]) < 0.3)
                confident_frac = np.mean(confident_mask)
            else:
                confident_correct = float('nan')
                confident_frac = 0.0
            # Correlation between predictions and targets
            if len(preds) > 1 and np.std(preds) > 0 and np.std(targets) > 0:
                correlation = np.corrcoef(preds, targets)[0, 1]
            else:
                correlation = float('nan')
            
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
                'strict_accuracy': strict_accuracy,
                'inverse_strict_accuracy': inverse_strict_accuracy,
                'classification_accuracy': class_accuracy,
                'confident_call_accuracy': confident_correct,
                'confident_call_fraction': confident_frac,
                'correlation': correlation,
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
    metrics = ['mse', 'mae', 'strict_accuracy', 'classification_accuracy']
    metric_labels = ['MSE', 'MAE', 'Strict Accuracy (0.7)', 'Classification Accuracy']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[stage][metric] for stage in stages]
        axes[i].bar(stages, values)
        axes[i].set_title(f'{label} by Game Stage')
        axes[i].set_ylabel(label)
        # Add value labels on bars
        for j, v in enumerate(values):
            axes[i].text(j, v + max(values) * 0.01, f'{v:.3f}', 
                        ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path / 'performance_by_stage.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Prediction vs Target distributions (scatter)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, stage in enumerate(stages):
        preds = results[stage]['predictions']
        targets = results[stage]['targets']
        # Fix: increase marker size and alpha for visibility
        axes[i].scatter(targets, preds, alpha=0.2, s=10, color='blue', edgecolors='none')
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


def load_examples_from_file(file_path, max_examples=100000):
    """Load all examples from a single .pkl.gz file."""
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    examples = data['examples'] if 'examples' in data else []
    if max_examples is not None:
        examples = examples[:max_examples]
    return examples


def main():
    parser = argparse.ArgumentParser(description="Analyze value head performance by game stage")
    parser.add_argument('--model_path', type=str, 
                        default=None,
                        required=False, help='Path to trained model (if set, only this model is analyzed)')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Data directory')
    parser.add_argument('--max_data_files', type=int, default=1, help='Maximum number of data files to use')
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR, help='Save directory')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to analyze')
    parser.add_argument('--device', type=str, default=DEVICE, help='Device to use')
    parser.add_argument('--enable_augmentation', action='store_true', help='Enable data augmentation')
    
    args = parser.parse_args()
    
    device = get_device() if args.device is None else torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data ONCE for all models
    print("Loading data...")
    data_files = discover_processed_files(args.data_dir)
    print(f"Found {len(data_files)} data files")
    print(f"Using {args.max_data_files} data files (full data, >97million, examples, is far larger than required)")
    input_data = data_files[:args.max_data_files]
    all_examples = []
    for file_path in input_data:
        examples = load_examples_from_file(file_path, max_examples=args.num_samples)
        all_examples.extend(examples)
        if len(all_examples) >= args.num_samples:
            all_examples = all_examples[:args.num_samples]
            break
    print(f"Loaded {len(all_examples)} examples for analysis.")
    class ExampleDataset:
        def __init__(self, examples):
            self.examples = examples
        def __len__(self):
            return len(self.examples)
        def __getitem__(self, idx):
            ex = self.examples[idx]
            # TODO: Replace manual preprocessing with preprocess_example_for_model from hex_ai.data_utils
            board_2ch = ex['board']
            # Add player-to-move channel
            player_to_move = get_player_to_move_from_board(board_2ch)
            player_channel = np.full((1, BOARD_SIZE, BOARD_SIZE), float(player_to_move), dtype=board_2ch.dtype)
            board_3ch = np.concatenate([board_2ch, player_channel], axis=0)
            board = torch.tensor(board_3ch, dtype=torch.float32)
            # Use central utility for policy label
            policy_numpy = get_valid_policy_target(ex['policy'], use_uniform=False)
            # Avoid torch warning: 
            # "To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone()"
            if isinstance(policy_numpy, torch.Tensor):
                policy = policy_numpy.detach().clone()
            else:
                policy = torch.tensor(policy_numpy, dtype=torch.float32)
            value = torch.tensor(ex['value'], dtype=torch.float32)
            return board, policy, value
    
    dataset = ExampleDataset(all_examples)
    
    # Model selection logic
    model_results = []  # List of dicts: {model_name, checkpoint, stage, metrics...}
    if args.model_path:
        # Only analyze the provided model_path
        model_paths = [("custom_model", args.model_path)]
    else:
        # Analyze all checkpoints for all model names
        model_paths = []
        for model_name in MODEL_NAMES:
            for epoch in CHECKPOINT_MINI_EPOCHS:
                checkpoint_file = CHECKPOINT_TEMPLATE.format(epoch)
                full_path = f"{MODEL_BASE_DIR}/{model_name}/{checkpoint_file}"
                print(f"Checking for checkpoint: {full_path}")  # Debug print
                if os.path.exists(full_path):
                    model_paths.append((model_name, full_path))
    print(f"Analyzing {len(model_paths)} model checkpoints...")
    
    # Analyze each model/checkpoint
    for model_name, model_path in model_paths:
        print(f"\n--- Analyzing {model_name} | {os.path.basename(model_path)} ---")
        try:
            model_wrapper = ModelWrapper(model_path, device=device)
            results = analyze_value_predictions(
                model_wrapper, 
                dataset, 
                num_samples=args.num_samples,
                device=device
            )
            # Store results for each stage
            for stage in ['early', 'mid', 'late']:
                if stage in results:
                    r = results[stage]
                    entry = {
                        'model': model_name,
                        'checkpoint': os.path.basename(model_path),
                        'stage': stage,
                        'mse': r['mse'],
                        'mae': r['mae'],
                        'strict_accuracy': r['strict_accuracy'],
                        'classification_accuracy': r['classification_accuracy'],
                        'bias': r['bias'],
                        'pred_mean': r['pred_mean'],
                        'pred_std': r['pred_std'],
                        'target_mean': r['target_mean'],
                        'target_std': r['target_std'],
                        'sample_count': r['sample_count'],
                        'full_summary': r
                    }
                    model_results.append(entry)
        except Exception as e:
            print(f"[ERROR] Failed to analyze {model_name} {model_path}: {e}")
    
    # Print metrics table
    print("\n=== Metrics Table (all models/checkpoints) ===")
    header = [
        "Model", "Checkpoint", "Stage", "MSE", "MAE", "StrictAcc(0.7)", "InvStrictAcc(0.7)",
        "ClassAcc", "ConfCallAcc", "ConfCallFrac", "Correlation", "Bias", "PredMean", "TargetMean", "N"
    ]
    print("\t".join(header))
    for entry in model_results:
        print(f"{entry['model']}\t{entry['checkpoint']}\t{entry['stage']}\t"
              f"{entry['mse']:.4f}\t{entry['mae']:.4f}\t{entry['strict_accuracy']:.2%}\t"
              f"{entry['inverse_strict_accuracy']:.2%}\t{entry['classification_accuracy']:.2%}\t"
              f"{entry['confident_call_accuracy']:.2%}\t{entry['confident_call_fraction']:.2%}\t"
              f"{entry['correlation']:.3f}\t{entry['bias']:.4f}\t"
              f"{entry['pred_mean']:.4f}\t{entry['target_mean']:.4f}\t{entry['sample_count']}")
    
    # Write CSV of full results
    csv_path = os.path.join(args.save_dir, "analysis_results.csv")
    os.makedirs(args.save_dir, exist_ok=True)
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(header)
        for entry in model_results:
            writer.writerow([
                entry['model'],
                entry['checkpoint'],
                entry['stage'],
                f"{entry['mse']:.4f}",
                f"{entry['mae']:.4f}",
                f"{entry['strict_accuracy']:.4f}",
                f"{entry['inverse_strict_accuracy']:.4f}",
                f"{entry['classification_accuracy']:.4f}",
                f"{entry['confident_call_accuracy']:.4f}",
                f"{entry['confident_call_fraction']:.4f}",
                f"{entry['correlation']:.4f}",
                f"{entry['bias']:.4f}",
                f"{entry['pred_mean']:.4f}",
                f"{entry['target_mean']:.4f}",
                entry['sample_count']
            ])
    print(f"\nCSV of results written to: {csv_path}")
    
    # Identify top 2 checkpoints for each metric (across all models/stages)
    metrics_to_check = ['strict_accuracy', 'classification_accuracy', 'mse', 'mae']
    for metric in metrics_to_check:
        print(f"\n=== Top 2 Checkpoints for {metric} ===")
        # For accuracy metrics, higher is better; for mse/mae, lower is better
        reverse = metric in ['strict_accuracy', 'classification_accuracy']
        sorted_entries = sorted(model_results, key=lambda x: x[metric], reverse=reverse)
        for i, entry in enumerate(sorted_entries[:2]):
            print(f"\n[{i+1}] {entry['model']} | {entry['checkpoint']} | {entry['stage']} | {metric}: {entry[metric]:.4f}")
            # Print full summary for this entry
            for k, v in entry['full_summary'].items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                elif isinstance(v, int):
                    print(f"  {k}: {v}")
                else:
                    pass  # Don't print long lists
    
    print("\nAnalysis complete for all models.")


if __name__ == "__main__":
    main() 