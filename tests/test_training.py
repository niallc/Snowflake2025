"""
Tests for the training module.

This module tests the training loop, loss computation, checkpointing,
and progress tracking functionality.
"""

import unittest
import torch
import tempfile
import os
from pathlib import Path
import numpy as np

from hex_ai.training import PolicyValueLoss, Trainer, create_trainer, train_model
from hex_ai.models import create_model
from hex_ai.dataset import HexDataset
from hex_ai.config import BOARD_SIZE, POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE


class TestPolicyValueLoss(unittest.TestCase):
    """Test the combined policy-value loss function."""
    
    def setUp(self):
        """Set up test data."""
        self.batch_size = 4
        self.policy_pred = torch.randn(self.batch_size, POLICY_OUTPUT_SIZE)
        self.value_pred = torch.randn(self.batch_size, 1)
        self.policy_target = torch.randn(self.batch_size, POLICY_OUTPUT_SIZE)
        self.value_target = torch.randn(self.batch_size, 1)
        
        # Normalize policy target to probabilities
        self.policy_target = torch.softmax(self.policy_target, dim=1)
    
    def test_loss_computation(self):
        """Test that loss computation works correctly."""
        criterion = PolicyValueLoss()
        
        total_loss, loss_dict = criterion(
            self.policy_pred, self.value_pred,
            self.policy_target, self.value_target
        )
        
        # Check that loss is a scalar tensor
        self.assertEqual(total_loss.dim(), 0)
        self.assertGreater(total_loss.item(), 0)
        
        # Check that loss dictionary contains expected keys
        expected_keys = ['total_loss', 'policy_loss', 'value_loss']
        for key in expected_keys:
            self.assertIn(key, loss_dict)
            self.assertIsInstance(loss_dict[key], float)
    
    def test_loss_weights(self):
        """Test that loss weights work correctly."""
        criterion = PolicyValueLoss(policy_weight=2.0, value_weight=0.5)
        
        total_loss, loss_dict = criterion(
            self.policy_pred, self.value_pred,
            self.policy_target, self.value_target
        )
        
        # Check that weights are applied
        self.assertGreater(loss_dict['policy_loss'], 0)
        self.assertGreater(loss_dict['value_loss'], 0)
    
    def test_loss_gradients(self):
        """Test that loss computation supports gradients."""
        criterion = PolicyValueLoss()
        
        # Enable gradients
        self.policy_pred.requires_grad_(True)
        self.value_pred.requires_grad_(True)
        
        total_loss, _ = criterion(
            self.policy_pred, self.value_pred,
            self.policy_target, self.value_target
        )
        
        # Check that gradients can be computed
        total_loss.backward()
        
        self.assertIsNotNone(self.policy_pred.grad)
        self.assertIsNotNone(self.value_pred.grad)


class TestTrainer(unittest.TestCase):
    """Test the Trainer class."""
    
    def setUp(self):
        """Set up test data."""
        self.model = create_model("resnet18")
        
        # Create dummy data
        self.train_data = [
            "http://www.trmph.com/hex/board#13,a1b2c3",
            "http://www.trmph.com/hex/board#13,a1b2c3d4e5f6g7",
            "http://www.trmph.com/hex/board#13,a1b2c3d4e5f6g7h8i9j10k11l12m13",
        ] * 10  # 30 samples
        
        self.val_data = [
            "http://www.trmph.com/hex/board#13,a1b2c3",
            "http://www.trmph.com/hex/board#13,a1b2c3d4e5f6g7",
        ] * 5  # 10 samples
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = create_trainer(
            self.model, self.train_data, self.val_data,
            batch_size=4, learning_rate=0.001
        )
        
        self.assertIsInstance(trainer, Trainer)
        self.assertEqual(trainer.current_epoch, 0)
        self.assertEqual(trainer.best_val_loss, float('inf'))
        self.assertEqual(len(trainer.training_history), 0)
    
    def test_trainer_without_validation(self):
        """Test trainer without validation data."""
        trainer = create_trainer(
            self.model, self.train_data, None,
            batch_size=4, learning_rate=0.001
        )
        
        self.assertIsNone(trainer.val_loader)
    
    def test_save_load_checkpoint(self):
        """Test checkpoint saving and loading."""
        trainer = create_trainer(
            self.model, self.train_data, self.val_data,
            batch_size=4, learning_rate=0.001
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test_checkpoint.pt"
            
            # Save checkpoint
            train_metrics = {'total_loss': 1.0, 'policy_loss': 0.5, 'value_loss': 0.5}
            val_metrics = {'total_loss': 0.8, 'policy_loss': 0.4, 'value_loss': 0.4}
            
            trainer.save_checkpoint(checkpoint_path, train_metrics, val_metrics)
            
            # Verify file exists
            self.assertTrue(checkpoint_path.exists())
            
            # Load checkpoint
            trainer.load_checkpoint(checkpoint_path)
            
            # Check that state was loaded
            self.assertEqual(trainer.best_val_loss, 0.8)
    
    def test_training_epoch(self):
        """Test training for one epoch."""
        trainer = create_trainer(
            self.model, self.train_data, self.val_data,
            batch_size=4, learning_rate=0.001
        )
        
        # Train for one epoch
        metrics = trainer.train_epoch()
        
        # Check that metrics are returned
        expected_keys = ['total_loss', 'policy_loss', 'value_loss']
        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], float)
            self.assertGreaterEqual(metrics[key], 0)
    
    def test_validation(self):
        """Test validation."""
        trainer = create_trainer(
            self.model, self.train_data, self.val_data,
            batch_size=4, learning_rate=0.001
        )
        
        # Run validation
        metrics = trainer.validate()
        
        # Check that metrics are returned
        expected_keys = ['total_loss', 'policy_loss', 'value_loss']
        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], float)
            self.assertGreaterEqual(metrics[key], 0)
    
    def test_validation_without_val_loader(self):
        """Test validation when no validation loader is provided."""
        trainer = create_trainer(
            self.model, self.train_data, None,
            batch_size=4, learning_rate=0.001
        )
        
        # Run validation
        metrics = trainer.validate()
        
        # Should return empty dict
        self.assertEqual(metrics, {})


class TestTrainingIntegration(unittest.TestCase):
    """Test training integration."""
    
    def test_train_model_function(self):
        """Test the convenience train_model function."""
        model = create_model("resnet18")
        
        train_data = [
            "http://www.trmph.com/hex/board#13,a1b2c3",
            "http://www.trmph.com/hex/board#13,a1b2c3d4e5f6g7",
        ] * 5  # 10 samples
        
        val_data = [
            "http://www.trmph.com/hex/board#13,a1b2c3",
        ] * 3  # 3 samples
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = train_model(
                model=model,
                train_data=train_data,
                val_data=val_data,
                num_epochs=2,  # Short training
                batch_size=2,   # Small batch size
                learning_rate=0.001,
                save_dir=temp_dir
            )
            
            # Check that results are returned
            self.assertIn('history', results)
            self.assertIn('best_val_loss', results)
            
            # Check that history has expected length
            self.assertEqual(len(results['history']), 2)
            
            # Check that checkpoints were saved
            checkpoint_files = list(Path(temp_dir).glob("*.pt"))
            self.assertGreater(len(checkpoint_files), 0)
            
            # Check that training history was saved
            history_file = Path(temp_dir) / "training_history.json"
            self.assertTrue(history_file.exists())


if __name__ == '__main__':
    unittest.main() 