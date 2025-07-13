"""
Integration tests for the complete Hex AI pipeline.

This module tests the entire pipeline from data loading through model training,
saving, and loading to ensure everything works together correctly.
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import tempfile
import os
import numpy as np
from pathlib import Path
import gzip
import pickle
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from resumable_shard_processing import process_all_data_resumable
from hex_ai.models import TwoHeadedResNet, create_model, count_parameters
from hex_ai.dataset import create_sample_data, create_dataloader
from hex_ai.config import BOARD_SIZE, NUM_PLAYERS, POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test data and model."""
        # Create sample data
        self.boards, self.policies, self.values = create_sample_data(batch_size=8)
        
        # Create model
        self.model = create_model("resnet18")
        
        # Create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Create loss functions
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.BCEWithLogitsLoss()
    
    def test_data_dimensions(self):
        """Test that data has correct dimensions."""
        self.assertEqual(self.boards.shape, (8, NUM_PLAYERS, BOARD_SIZE, BOARD_SIZE))
        self.assertEqual(self.policies.shape, (8, POLICY_OUTPUT_SIZE))
        self.assertEqual(self.values.shape, (8, VALUE_OUTPUT_SIZE))
    
    def test_model_forward_pass(self):
        """Test that model can process data and output correct dimensions."""
        # Forward pass
        policy_logits, value_logit = self.model(self.boards)
        
        # Check output dimensions
        self.assertEqual(policy_logits.shape, (8, POLICY_OUTPUT_SIZE))
        self.assertEqual(value_logit.shape, (8, VALUE_OUTPUT_SIZE))
        
        # Check that outputs are reasonable
        self.assertTrue(torch.isfinite(policy_logits).all())
        self.assertTrue(torch.isfinite(value_logit).all())
    
    def test_loss_computation(self):
        """Test that loss functions work with model outputs."""
        # Forward pass
        policy_logits, value_logit = self.model(self.boards)
        
        # Compute losses
        policy_loss = self.policy_loss_fn(policy_logits, self.policies.argmax(dim=1))
        value_loss = self.value_loss_fn(value_logit, self.values)
        
        # Check that losses are finite and positive
        self.assertTrue(torch.isfinite(policy_loss))
        self.assertTrue(torch.isfinite(value_loss))
        self.assertGreater(policy_loss.item(), 0)
        self.assertGreater(value_loss.item(), 0)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        # Forward pass
        policy_logits, value_logit = self.model(self.boards)
        
        # Compute total loss
        policy_loss = self.policy_loss_fn(policy_logits, self.policies.argmax(dim=1))
        value_loss = self.value_loss_fn(value_logit, self.values)
        total_loss = policy_loss + value_loss
        
        # Backward pass
        total_loss.backward()
        
        # Check that gradients exist
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all())
    
    def test_optimizer_step(self):
        """Test that optimizer can update model parameters."""
        # Get initial parameters
        initial_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                initial_params[name] = param.data.clone()
        
        # Forward pass
        policy_logits, value_logit = self.model(self.boards)
        policy_loss = self.policy_loss_fn(policy_logits, self.policies.argmax(dim=1))
        value_loss = self.value_loss_fn(value_logit, self.values)
        total_loss = policy_loss + value_loss
        
        # Backward pass and optimizer step
        total_loss.backward()
        self.optimizer.step()
        
        # Check that parameters changed
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertFalse(torch.allclose(initial_params[name], param.data))
    
    def test_model_save_load(self):
        """Test that model can be saved and loaded."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, temp_path)
            
            # Create new model and optimizer
            new_model = create_model("resnet18")
            new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
            
            # Load saved state
            checkpoint = torch.load(temp_path, weights_only=False)
            new_model.load_state_dict(checkpoint['model_state_dict'])
            new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Test that loaded model produces same output
            # Set both models to eval mode to ensure deterministic behavior
            self.model.eval()
            new_model.eval()
            
            with torch.no_grad():
                original_policy, original_value = self.model(self.boards)
                loaded_policy, loaded_value = new_model(self.boards)
                
                torch.testing.assert_close(original_policy, loaded_policy)
                torch.testing.assert_close(original_value, loaded_value)
                
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_dataloader_integration(self):
        """Test that DataLoader works with the model."""
        # Create temporary directory with dummy data
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy .trmph files with proper format
            for i in range(3):
                dummy_file = os.path.join(temp_dir, f"game_{i}.trmph")
                with open(dummy_file, 'w') as f:
                    f.write(f"http://www.trmph.com/hex/board#13,a{i+1}b{i+2}c{i+3} {i % 2}\n")
            
            # Create DataLoader
            dataloader = create_dataloader(temp_dir, batch_size=2, num_workers=0)
            
            # Test that we can iterate through the dataloader
            batch_count = 0
            for batch_idx, (boards, policies, values) in enumerate(dataloader):
                batch_count += 1
                
                # Check batch dimensions (last batch might be smaller)
                expected_batch_size = min(2, 3 - batch_idx * 2)  # 3 files, batch_size=2
                self.assertEqual(boards.shape[0], expected_batch_size)
                self.assertEqual(boards.shape[1:], (NUM_PLAYERS, BOARD_SIZE, BOARD_SIZE))
                self.assertEqual(policies.shape[1], POLICY_OUTPUT_SIZE)
                self.assertEqual(values.shape[1], VALUE_OUTPUT_SIZE)
                
                # Test that model can process this batch
                policy_logits, value_logit = self.model(boards)
                self.assertEqual(policy_logits.shape, (expected_batch_size, POLICY_OUTPUT_SIZE))
                self.assertEqual(value_logit.shape, (expected_batch_size, VALUE_OUTPUT_SIZE))
                
                if batch_count >= 2:  # Limit to avoid infinite loop
                    break
    
    def test_device_transfer(self):
        """Test that model works on different devices."""
        # Test CPU (should always work)
        model_cpu = self.model.cpu()
        boards_cpu = self.boards.cpu()
        policy_logits, value_logit = model_cpu(boards_cpu)
        self.assertEqual(policy_logits.device, torch.device('cpu'))
        self.assertEqual(value_logit.device, torch.device('cpu'))
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = self.model.cuda()
            boards_cuda = self.boards.cuda()
            policy_logits, value_logit = model_cuda(boards_cuda)
            self.assertEqual(policy_logits.device, torch.device('cuda'))
            self.assertEqual(value_logit.device, torch.device('cuda'))
    
    def test_model_summary(self):
        """Test that model summary functions work."""
        # Test parameter counting
        num_params = count_parameters(self.model)
        self.assertGreater(num_params, 10_000_000)  # ResNet-18 should have >10M params
        self.assertLess(num_params, 15_000_000)     # But less than 15M
    
    def test_training_step(self):
        """Test a complete training step."""
        # Set model to training mode
        self.model.train()
        
        # Forward pass
        policy_logits, value_logit = self.model(self.boards)
        
        # Compute losses
        policy_loss = self.policy_loss_fn(policy_logits, self.policies.argmax(dim=1))
        value_loss = self.value_loss_fn(value_logit, self.values)
        total_loss = policy_loss + value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Check that loss decreased (or at least didn't explode)
        self.assertTrue(torch.isfinite(total_loss))
        self.assertGreater(total_loss.item(), 0)
    
    def test_evaluation_mode(self):
        """Test that model works in evaluation mode."""
        # Set model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            policy_logits, value_logit = self.model(self.boards)
            
            # Check outputs
            self.assertEqual(policy_logits.shape, (8, POLICY_OUTPUT_SIZE))
            self.assertEqual(value_logit.shape, (8, VALUE_OUTPUT_SIZE))
            
            # Check that gradients are not computed
            self.assertFalse(policy_logits.requires_grad)
            self.assertFalse(value_logit.requires_grad)


class TestGlobalSharding(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.source_dir = Path(self.temp_dir.name) / "source"
        self.processed_dir = Path(self.temp_dir.name) / "processed"
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        # Create 3 dummy .trmph files with 7 unique games total
        games = [
            "http://www.trmph.com/hex/board#13,a1b2c3 1\n",
            "http://www.trmph.com/hex/board#13,a1b2c3d4e5f6g7 0\n",
            "http://www.trmph.com/hex/board#13,a1b2c3d4e5f6g7h8i9j10k11l12m13 1\n",
            "http://www.trmph.com/hex/board#13,a1b2c3d4e5f6g7h8 0\n",
            "http://www.trmph.com/hex/board#13,a1b2c3d4e5 1\n",
            "http://www.trmph.com/hex/board#13,a1b2c3d4e5f6 0\n",
            "http://www.trmph.com/hex/board#13,a1b2c3d4 1\n",
        ]
        # Write unique games to each file
        with open(self.source_dir / "file_0.trmph", "w") as f:
            f.write(games[0])
            f.write(games[1])
        with open(self.source_dir / "file_1.trmph", "w") as f:
            f.write(games[2])
            f.write(games[3])
            f.write(games[4])
        with open(self.source_dir / "file_2.trmph", "w") as f:
            f.write(games[5])
            f.write(games[6])

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_global_sharding(self):
        # Use a small shard size to force multiple shards
        shard_files = process_all_data_resumable(
            source_dir=str(self.source_dir),
            processed_dir=str(self.processed_dir),
            games_per_shard=3
        )
        # 7 games, 3 per shard => 3 shards (3, 3, 1)
        self.assertEqual(len(shard_files), 3)
        # Check for unique shard names
        shard_names = [f.name for f in shard_files]
        self.assertEqual(len(shard_names), len(set(shard_names)), "Shard names are not unique!")
        # Check for no clobbering (all files exist)
        for f in shard_files:
            self.assertTrue(f.exists(), f"Shard file {f} does not exist!")
        # Check all games are present, no duplicates or missing
        all_games = set()
        total_games = 0
        for f in shard_files:
            with gzip.open(f, 'rb') as g:
                data = pickle.load(g)
                total_games += data['num_games']
                for board in data['boards']:
                    all_games.add(board.numpy().tobytes())
        self.assertEqual(total_games, 7)
        self.assertEqual(len(all_games), 7, "Duplicate or missing games in shards!")
        # Check shard sizes (except last)
        for f in shard_files[:-1]:
            with gzip.open(f, 'rb') as g:
                data = pickle.load(g)
                self.assertEqual(data['num_games'], 3)
        # Last shard can be smaller
        with gzip.open(shard_files[-1], 'rb') as g:
            data = pickle.load(g)
            self.assertGreaterEqual(data['num_games'], 1)


if __name__ == '__main__':
    unittest.main() 