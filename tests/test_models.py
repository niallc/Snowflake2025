"""
Tests for the Hex AI model architectures.

This module contains unit tests to ensure the ResNet models work correctly
and produce the expected output shapes.
"""

import unittest
import os
import torch
import torch.nn as nn
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

# Environment validation is now handled automatically in hex_ai/__init__.py

from hex_ai.inference.model_wrapper import ModelWrapper
from hex_ai.model_spec import model_spec_from_model
from hex_ai.models import (
    ResNetBlock,
    TwoHeadedBottleneckPool3x3PolicyResNet,
    TwoHeadedBottleneckPoolResNet,
    TwoHeadedResNet,
    count_parameters,
    create_model,
    get_model_summary,
    get_supported_model_types,
)
from hex_ai.config import BOARD_SIZE, NUM_PLAYERS, POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE


class TestResNetBlock(unittest.TestCase):
    """Test cases for the ResNetBlock class."""
    
    def test_resnet_block_creation(self):
        """Test that ResNetBlock can be created."""
        block = ResNetBlock(in_channels=64, out_channels=64, stride=1)
        self.assertIsInstance(block, ResNetBlock)
    
    def test_resnet_block_forward(self):
        """Test that ResNetBlock forward pass works."""
        block = ResNetBlock(in_channels=64, out_channels=64, stride=1)
        x = torch.randn(2, 64, 13, 13)
        output = block(x)
        
        # Check output shape
        self.assertEqual(output.shape, (2, 64, 13, 13))
    
    def test_resnet_block_stride_2(self):
        """Test ResNetBlock with stride=2 (downsampling)."""
        block = ResNetBlock(in_channels=64, out_channels=128, stride=2)
        x = torch.randn(2, 64, 13, 13)
        output = block(x)
        
        # Check output shape (should be downsampled)
        self.assertEqual(output.shape, (2, 128, 7, 7))
    
    def test_resnet_block_shortcut(self):
        """Test that shortcut connection works correctly."""
        block = ResNetBlock(in_channels=64, out_channels=128, stride=2)
        x = torch.randn(2, 64, 13, 13)
        output = block(x)
        
        # Output should be the sum of main path and shortcut
        self.assertEqual(output.shape, (2, 128, 7, 7))


class TestTwoHeadedResNet(unittest.TestCase):
    """Test cases for the TwoHeadedResNet class."""
    
    def test_model_creation(self):
        """Test that TwoHeadedResNet can be created."""
        model = TwoHeadedResNet()
        self.assertIsInstance(model, TwoHeadedResNet)
    
    def test_model_forward(self):
        """Test that model forward pass works."""
        model = TwoHeadedResNet()
        x = torch.randn(4, 3, BOARD_SIZE, BOARD_SIZE)
        move_stage = torch.rand(4)  # Random move stages in [0,1]
        policy_logits, value_logit = model(x, move_stage)
        
        # Check output shapes
        self.assertEqual(policy_logits.shape, (4, POLICY_OUTPUT_SIZE))
        self.assertEqual(value_logit.shape, (4, VALUE_OUTPUT_SIZE))
        
        # Check that value outputs are in [-1, 1] range due to tanh
        self.assertTrue(torch.all(value_logit >= -1) and torch.all(value_logit <= 1))
    
    def test_model_parameters(self):
        """Test that model has reasonable number of parameters."""
        model = TwoHeadedResNet()
        num_params = count_parameters(model)
        
        # KataGo-inspired model should have reasonable number of parameters
        self.assertGreater(num_params, 1_000_000)   # At least 1M
        self.assertLess(num_params, 50_000_000)     # Less than 50M
    
    def test_model_summary(self):
        """Test that model summary works."""
        model = TwoHeadedResNet()
        summary = get_model_summary(model)
        
        # Check that summary contains expected information
        self.assertIn("Total Parameters", summary)
        self.assertIn("TwoHeadedResNet", summary)
        self.assertIn("Policy Head", summary)
        self.assertIn("Value Head", summary)
    
    def test_model_device_transfer(self):
        """Test that model can be moved to different devices."""
        model = TwoHeadedResNet()
        
        # Test CPU
        model_cpu = model.cpu()
        x = torch.randn(2, 3, BOARD_SIZE, BOARD_SIZE)
        move_stage = torch.rand(2)
        policy, value = model_cpu(x, move_stage)
        self.assertEqual(policy.device, torch.device('cpu'))
        self.assertEqual(value.device, torch.device('cpu'))
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            x_cuda = x.cuda()
            move_stage_cuda = move_stage.cuda()
            policy, value = model_cuda(x_cuda, move_stage_cuda)
            self.assertEqual(policy.device, torch.device('cuda'))
            self.assertEqual(value.device, torch.device('cuda'))
    
    def test_model_gradients(self):
        """Test that model can compute gradients."""
        model = TwoHeadedResNet()
        x = torch.randn(2, 3, BOARD_SIZE, BOARD_SIZE, requires_grad=True)
        move_stage = torch.rand(2)
        policy_logits, value_logit = model(x, move_stage)
        
        # Compute loss and backward pass
        loss = policy_logits.sum() + value_logit.sum()
        loss.backward()
        
        # Check that gradients were computed
        self.assertIsNotNone(x.grad)
    
    def test_model_batch_sizes(self):
        """Test that model works with different batch sizes."""
        model = TwoHeadedResNet()
        
        batch_sizes = [1, 4, 8, 16]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3, BOARD_SIZE, BOARD_SIZE)
            move_stage = torch.rand(batch_size)
            policy_logits, value_logit = model(x, move_stage)
            
            self.assertEqual(policy_logits.shape, (batch_size, POLICY_OUTPUT_SIZE))
            self.assertEqual(value_logit.shape, (batch_size, VALUE_OUTPUT_SIZE))
    
    def test_value_head_architecture(self):
        """Test the enhanced value head architecture."""
        # Test the new KataGo-inspired value head (always has bottleneck)
        model = TwoHeadedResNet()
        x = torch.randn(2, 3, BOARD_SIZE, BOARD_SIZE)
        move_stage = torch.rand(2)
        policy, value = model(x, move_stage)
        
        # Check value range is [-1, 1] due to tanh
        self.assertTrue(torch.all(value >= -1) and torch.all(value <= 1))
        
        # Check that the value head has the expected structure
        self.assertTrue(hasattr(model.value_head, 'k_outputs'))
        self.assertEqual(model.value_head.k_outputs, 4)

    def test_value_head_output_layers_keep_explicit_neutral_initialization(self):
        """Value-head output layers should preserve the intended near-neutral init."""
        for model in (TwoHeadedResNet(), create_model("katago_bottleneck_pool")):
            out_k_weight = model.value_head.out_k.weight.detach()
            out_k_bias = model.value_head.out_k.bias.detach()
            comb_weight = model.value_head.comb.weight.detach()

            self.assertTrue(torch.allclose(out_k_weight, torch.zeros_like(out_k_weight)))
            self.assertTrue(torch.allclose(out_k_bias, torch.zeros_like(out_k_bias)))
            expected_comb = torch.full_like(comb_weight, 1.0 / model.value_head.k_outputs)
            self.assertTrue(torch.allclose(comb_weight, expected_comb))


class TestModelFactory(unittest.TestCase):
    """Test cases for the model factory function."""
    
    def test_create_model_katago_inspired(self):
        """Test creating KataGo-inspired model."""
        model = create_model("katago_inspired")
        self.assertIsInstance(model, TwoHeadedResNet)

    def test_create_model_katago_bottleneck_pool_defaults(self):
        """Test creating the pooled bottleneck challenger with family defaults."""
        model = create_model("katago_bottleneck_pool")
        self.assertIsInstance(model, TwoHeadedBottleneckPoolResNet)
        self.assertEqual(model.num_blocks, 13)
        self.assertEqual(model.trunk_channels, 224)
        self.assertEqual(model.global_block_indices, (3, 8))
        self.assertEqual(model.value_head.pool_mode, "mean_max")

    def test_create_model_katago_bottleneck_pool_3x3_policy_defaults(self):
        """Test creating the 3x3-policy pooled bottleneck challenger."""
        model = create_model("katago_bottleneck_pool_3x3_policy")
        self.assertIsInstance(model, TwoHeadedBottleneckPool3x3PolicyResNet)
        self.assertEqual(model.num_blocks, 13)
        self.assertEqual(model.trunk_channels, 224)
        self.assertEqual(model.global_block_indices, (3, 8))
        self.assertEqual(model.value_head.pool_mode, "mean_max")

    def test_supported_model_types_include_3x3_policy_variant(self):
        """Test that the supported-type registry includes the new family."""
        self.assertIn("katago_bottleneck_pool_3x3_policy", get_supported_model_types())

    def test_create_model_board_size_propagates(self):
        """Test that board_size is carried into model construction and output shapes."""
        board_size = 9
        model = create_model("katago_bottleneck_pool", board_size=board_size)
        self.assertEqual(model.board_size, board_size)
        self.assertEqual(model.policy_head.board_size, board_size)

        x = torch.randn(2, 3, board_size, board_size)
        move_stage = torch.rand(2)
        policy_logits, value = model(x, move_stage)

        self.assertEqual(policy_logits.shape, (2, board_size * board_size))
        self.assertEqual(value.shape, (2, VALUE_OUTPUT_SIZE))

    def test_model_rejects_board_size_mismatch(self):
        """Test that model board_size metadata matches runtime board tensors."""
        model = create_model("katago_bottleneck_pool", board_size=9)
        x = torch.randn(1, 3, 13, 13)
        move_stage = torch.rand(1)

        with self.assertRaises(ValueError):
            model(x, move_stage)
    
    def test_create_model_invalid_type(self):
        """Test that invalid model type raises error."""
        with self.assertRaises(ValueError):
            create_model("invalid_model")

    def test_model_summary_describes_3x3_policy_bottleneck_pool_variant(self):
        """Test that summary text keeps pooled-bias trunk details for head-only variants."""
        model = create_model("katago_bottleneck_pool_3x3_policy")
        summary = get_model_summary(model)
        self.assertIn("pooled-bias blocks at 1-indexed positions [4, 9]", summary)
        self.assertIn("3x3 + global-bias + LayerNorm policy head", summary)

    def test_model_wrapper_reloads_checkpoint_for_3x3_policy_variant(self):
        """Test spec-based checkpoint reload for the new bottleneck-pool family."""
        model = create_model("katago_bottleneck_pool_3x3_policy")
        checkpoint = {
            "model_spec": model_spec_from_model(model).to_dict(),
            "model_state_dict": model.state_dict(),
        }

        with TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)

            wrapper = ModelWrapper(str(checkpoint_path), device="cpu")

        self.assertEqual(
            wrapper.model_spec.model_type, "katago_bottleneck_pool_3x3_policy"
        )
        self.assertIsInstance(wrapper.model, TwoHeadedBottleneckPool3x3PolicyResNet)


class TestAppDevModelPaths(unittest.TestCase):
    """Regression tests for dev-server model-path normalization."""

    def test_normalize_requested_model_path_strips_checkpoints_prefix(self):
        from hex_ai.web import app_dev

        relative_path = "checkpoints/foo/bar.pt.gz"
        absolute_path = os.path.abspath(relative_path)

        self.assertEqual(
            app_dev._normalize_requested_model_path(relative_path),
            "foo/bar.pt.gz",
        )
        self.assertEqual(
            app_dev._normalize_requested_model_path(absolute_path),
            "foo/bar.pt.gz",
        )

    def test_register_dynamic_model_normalizes_prefixed_relative_path(self):
        from hex_ai.web import app_dev

        model_id = "test_dynamic_model"
        try:
            app_dev.register_dynamic_model(model_id, "checkpoints/foo/bar.pt.gz")
            self.assertEqual(app_dev.DYNAMIC_MODELS[model_id], "foo/bar.pt.gz")
        finally:
            app_dev.DYNAMIC_MODELS.pop(model_id, None)

if __name__ == '__main__':
    unittest.main() 
