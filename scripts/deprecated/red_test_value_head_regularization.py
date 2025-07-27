#!/usr/bin/env python3
"""
Test script to demonstrate value head regularization with different learning rates and weight decay.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hex_ai.training import Trainer
from hex_ai.models import TwoHeadedResNet

def test_value_head_regularization():
    """Test the value head regularization functionality."""
    
    print("Testing Value Head Regularization")
    print("=" * 50)
    
    # Create a simple model
    model = TwoHeadedResNet()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dummy data
    batch_size = 4
    dummy_data = torch.randn(batch_size, 3, 13, 13)
    dummy_policies = torch.randn(batch_size, 169)
    dummy_values = torch.randn(batch_size, 1)
    
    dataset = TensorDataset(dummy_data, dummy_policies, dummy_values)
    loader = DataLoader(dataset, batch_size=batch_size)
    
    # Test different configurations
    configs = [
        {"name": "Default", "value_lr_factor": 1.0, "value_wd_factor": 1.0},
        {"name": "Slow Value Learning", "value_lr_factor": 0.1, "value_wd_factor": 1.0},
        {"name": "High Value Regularization", "value_lr_factor": 1.0, "value_wd_factor": 5.0},
        {"name": "Both", "value_lr_factor": 0.1, "value_wd_factor": 5.0},
    ]
    
    base_lr = 0.001
    base_wd = 1e-4
    
    for config in configs:
        print(f"\n{config['name']}:")
        print("-" * 30)
        
        trainer = Trainer(
            model=TwoHeadedResNet(),  # Fresh model for each test
            train_loader=loader,
            learning_rate=base_lr,
            weight_decay=base_wd,
            value_learning_rate_factor=config["value_lr_factor"],
            value_weight_decay_factor=config["value_wd_factor"]
        )
        
        # Show parameter groups
        for i, group in enumerate(trainer.optimizer.param_groups):
            group_name = "Value Head" if i == 1 else "Other Parameters"
            print(f"  {group_name}:")
            print(f"    Learning Rate: {group['lr']:.6f}")
            print(f"    Weight Decay: {group['weight_decay']:.6f}")
            print(f"    Parameters: {len(group['params'])}")
        
        # Test a forward pass
        model.eval()
        with torch.no_grad():
            policy_pred, value_pred = model(dummy_data)
            print(f"  Policy output shape: {policy_pred.shape}")
            print(f"  Value output shape: {value_pred.shape}")

if __name__ == "__main__":
    test_value_head_regularization() 