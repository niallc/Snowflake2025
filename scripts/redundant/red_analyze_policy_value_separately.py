#!/usr/bin/env python3
"""
Analyze policy and value network performance separately.
- Load the best model from hyperparameter tuning
- Extract separate policy and value outputs
- Analyze their individual performance
- Visualize policy predictions vs targets
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple
import pandas as pd

from hex_ai.models import TwoHeadedResNet
from hex_ai.data_pipeline import StreamingAugmentedProcessedDataset
from torch.utils.data import DataLoader

def load_best_model(results_dir: str = "checkpoints/scaled_tuning") -> Tuple[TwoHeadedResNet, Dict]:
    """Load the best model from hyperparameter tuning results."""
    results_dir = Path(results_dir)
    
    # Find the best experiment
    best_exp = None
    best_loss = float('inf')
    
    for exp_dir in results_dir.iterdir():
        if exp_dir.is_dir() and exp_dir.name != "__pycache__":
            results_file = exp_dir / "experiment_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    exp_data = json.load(f)
                
                if 'best_val_loss' in exp_data:
                    val_loss = exp_data['best_val_loss']
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_exp = exp_dir.name
    
    if best_exp is None:
        raise ValueError("No valid experiments found!")
    
    print(f"Loading best model from experiment: {best_exp}")
    print(f"Best validation loss: {best_loss:.6f}")
    
    # Load model configuration
    config_file = results_dir / best_exp / "config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Create model with same configuration
    model = TwoHeadedResNet(dropout_prob=config['hyperparameters']['dropout_prob'])
    
    # Load best model weights
    model_path = results_dir / best_exp / "best_model.pt"
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config

def analyze_policy_value_performance(model: TwoHeadedResNet, 
                                   data_loader: DataLoader,
                                   device: torch.device) -> Dict:
    """Analyze policy and value network performance separately."""
    model.eval()
    model.to(device)
    
    policy_losses = []
    value_losses = []
    policy_accuracies = []
    value_accuracies = []
    
    policy_predictions = []
    policy_targets = []
    value_predictions = []
    value_targets = []
    
    with torch.no_grad():
        for batch_idx, (board_states, policy_targets_batch, value_targets_batch) in enumerate(data_loader):
            board_states = board_states.to(device)
            policy_targets_batch = policy_targets_batch.to(device)
            value_targets_batch = value_targets_batch.to(device)
            
            # Forward pass
            policy_logits, value_logits = model(board_states)
            
            # Policy analysis
            policy_probs = F.softmax(policy_logits, dim=1)
            policy_loss = F.cross_entropy(policy_logits, policy_targets_batch)
            policy_losses.append(policy_loss.item())
            
            # Policy accuracy (top-1)
            target_indices = torch.argmax(policy_targets_batch, dim=1)
            policy_pred = torch.argmax(policy_logits, dim=1)
            policy_acc = (policy_pred == target_indices).float().mean().item()
            policy_accuracies.append(policy_acc)
            
            # Value analysis
            value_probs = torch.sigmoid(value_logits).squeeze()
            value_loss = F.binary_cross_entropy_with_logits(value_logits.view(-1), value_targets_batch.float().view(-1))
            value_losses.append(value_loss.item())
            
            # Value accuracy (threshold at 0.5)
            value_pred = (value_probs > 0.5).float()
            value_acc = (value_pred == value_targets_batch.float()).float().mean().item()
            value_accuracies.append(value_acc)
            
            # Store predictions and targets for detailed analysis
            policy_predictions.extend(policy_probs.cpu().numpy())
            policy_targets.extend(policy_targets_batch.cpu().numpy())
            value_predictions.extend(value_probs.cpu().numpy())
            value_targets.extend(value_targets_batch.cpu().numpy())
            
            if batch_idx >= 10:  # Limit to first 10 batches for analysis
                break
    
    return {
        'policy_losses': policy_losses,
        'value_losses': value_losses,
        'policy_accuracies': policy_accuracies,
        'value_accuracies': value_accuracies,
        'policy_predictions': np.array(policy_predictions),
        'policy_targets': np.array(policy_targets),
        'value_predictions': np.array(value_predictions),
        'value_targets': np.array(value_targets)
    }

def plot_policy_analysis(results: Dict, save_path: str = None):
    """Plot policy network analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Policy loss over batches
    axes[0, 0].plot(results['policy_losses'])
    axes[0, 0].set_title('Policy Loss Over Batches')
    axes[0, 0].set_xlabel('Batch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Policy accuracy over batches
    axes[0, 1].plot(results['policy_accuracies'])
    axes[0, 1].set_title('Policy Accuracy Over Batches')
    axes[0, 1].set_xlabel('Batch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Policy prediction distribution
    axes[1, 0].hist(results['policy_predictions'].flatten(), bins=50, alpha=0.7)
    axes[1, 0].set_title('Policy Prediction Distribution')
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Frequency')
    
    # Policy target vs prediction correlation
    target_probs = results['policy_targets'].flatten()
    pred_probs = results['policy_predictions'].flatten()
    axes[1, 1].scatter(target_probs, pred_probs, alpha=0.1)
    axes[1, 1].plot([0, 1], [0, 1], 'r--', linewidth=2)
    axes[1, 1].set_title('Policy: Target vs Prediction')
    axes[1, 1].set_xlabel('Target Probability')
    axes[1, 1].set_ylabel('Predicted Probability')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Policy analysis saved to: {save_path}")
    
    plt.show()

def plot_value_analysis(results: Dict, save_path: str = None):
    """Plot value network analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Value loss over batches
    axes[0, 0].plot(results['value_losses'])
    axes[0, 0].set_title('Value Loss Over Batches')
    axes[0, 0].set_xlabel('Batch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Value accuracy over batches
    axes[0, 1].plot(results['value_accuracies'])
    axes[0, 1].set_title('Value Accuracy Over Batches')
    axes[0, 1].set_xlabel('Batch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Value prediction distribution
    axes[1, 0].hist(results['value_predictions'], bins=50, alpha=0.7)
    axes[1, 0].set_title('Value Prediction Distribution')
    axes[1, 0].set_xlabel('Predicted Win Probability')
    axes[1, 0].set_ylabel('Frequency')
    
    # Value target vs prediction correlation
    axes[1, 1].scatter(results['value_targets'], results['value_predictions'], alpha=0.5)
    axes[1, 1].plot([0, 1], [0, 1], 'r--', linewidth=2)
    axes[1, 1].set_title('Value: Target vs Prediction')
    axes[1, 1].set_xlabel('Target Win Probability')
    axes[1, 1].set_ylabel('Predicted Win Probability')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Value analysis saved to: {save_path}")
    
    plt.show()

def print_separate_performance_summary(results: Dict):
    """Print a summary of separate policy and value performance."""
    print("\n" + "="*80)
    print("SEPARATE POLICY AND VALUE NETWORK ANALYSIS")
    print("="*80)
    
    # Policy network summary
    avg_policy_loss = np.mean(results['policy_losses'])
    avg_policy_acc = np.mean(results['policy_accuracies'])
    
    print(f"\n📊 POLICY NETWORK PERFORMANCE:")
    print(f"   Average Loss: {avg_policy_loss:.4f}")
    print(f"   Average Accuracy: {avg_policy_acc:.4f} ({avg_policy_acc*100:.1f}%)")
    print(f"   Loss Range: {min(results['policy_losses']):.4f} - {max(results['policy_losses']):.4f}")
    print(f"   Accuracy Range: {min(results['policy_accuracies']):.4f} - {max(results['policy_accuracies']):.4f}")
    
    # Value network summary
    avg_value_loss = np.mean(results['value_losses'])
    avg_value_acc = np.mean(results['value_accuracies'])
    
    print(f"\n🎯 VALUE NETWORK PERFORMANCE:")
    print(f"   Average Loss: {avg_value_loss:.4f}")
    print(f"   Average Accuracy: {avg_value_acc:.4f} ({avg_value_acc*100:.1f}%)")
    print(f"   Loss Range: {min(results['value_losses']):.4f} - {max(results['value_losses']):.4f}")
    print(f"   Accuracy Range: {min(results['value_accuracies']):.4f} - {max(results['value_accuracies']):.4f}")
    
    # Combined performance
    total_loss = avg_policy_loss + avg_value_loss
    print(f"\n🔗 COMBINED PERFORMANCE:")
    print(f"   Total Average Loss: {total_loss:.4f}")
    print(f"   Policy Loss Weight: {avg_policy_loss/total_loss:.2%}")
    print(f"   Value Loss Weight: {avg_value_loss/total_loss:.2%}")
    
    # Prediction analysis
    print(f"\n📈 PREDICTION ANALYSIS:")
    print(f"   Policy Predictions - Mean: {np.mean(results['policy_predictions']):.4f}, Std: {np.std(results['policy_predictions']):.4f}")
    print(f"   Value Predictions - Mean: {np.mean(results['value_predictions']):.4f}, Std: {np.std(results['value_predictions']):.4f}")
    print(f"   Policy Targets - Mean: {np.mean(results['policy_targets']):.4f}, Std: {np.std(results['policy_targets']):.4f}")
    print(f"   Value Targets - Mean: {np.mean(results['value_targets']):.4f}, Std: {np.std(results['value_targets']):.4f}")

if __name__ == "__main__":
    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load best model
    model, config = load_best_model()
    
    # Create small dataset for analysis
    from hex_ai.data_pipeline import discover_processed_files
    data_files = discover_processed_files("data/processed")
    dataset = StreamingAugmentedProcessedDataset(
        data_files=data_files,
        max_examples=500,  # Small sample for analysis
        shuffle_files=True,
        # NOTE: Augmentation has been disabled for analysis.
        #       Is this what you intend?
        enable_augmentation=False # Disable augmentation for analysis
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False,
        num_workers=0  # Avoid multiprocessing issues
    )
    
    print(f"Analyzing {len(dataset)} examples...")
    
    # Analyze separate performance
    results = analyze_policy_value_performance(model, dataloader, device)
    
    # Print summary
    print_separate_performance_summary(results)
    
    # Create plots
    plot_policy_analysis(results, "policy_analysis.png")
    plot_value_analysis(results, "value_analysis.png")
    
    print(f"\nAnalysis complete! Check the generated plots for detailed visualizations.") 