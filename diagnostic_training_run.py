#!/usr/bin/env python3
"""
Diagnostic script for a small training run to verify everything works correctly.
"""

import logging
import os
import time
import psutil
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hex_ai.config import (
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, 
    DEVICE, CHECKPOINT_DIR, BOARD_SIZE, POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE
)
from hex_ai.data_processing import DataProcessor, ProcessedDataset, create_processed_dataloader
from hex_ai.models import TwoHeadedResNet, create_model
from hex_ai.training import Trainer, PolicyValueLoss

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_memory_usage(context: str = ""):
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)
    logger.info(f"[{context}] Memory usage: {mem:.2f} MB")

def check_data_files():
    """Check available data files and their sizes."""
    data_dir = Path("./data/twoNetGames")
    trmph_files = list(data_dir.glob("*.trmph"))
    
    logger.info(f"Found {len(trmph_files)} .trmph files")
    
    # Sort by size and show top 10
    file_sizes = [(f, f.stat().st_size) for f in trmph_files]
    file_sizes.sort(key=lambda x: x[1], reverse=True)
    
    logger.info("Top 10 largest files:")
    for i, (file_path, size) in enumerate(file_sizes[:10]):
        size_mb = size / (1024 * 1024)
        logger.info(f"  {i+1}. {file_path.name}: {size_mb:.2f} MB")
    
    # Find a small file for testing
    small_files = [f for f, size in file_sizes if size < 100 * 1024]  # < 100KB
    if small_files:
        test_file = small_files[0]
        logger.info(f"Selected small test file: {test_file.name} ({test_file.stat().st_size / 1024:.1f} KB)")
        return test_file
    else:
        # Use the smallest file
        test_file = file_sizes[-1][0]
        logger.info(f"Selected smallest file: {test_file.name} ({test_file.stat().st_size / 1024:.1f} KB)")
        return test_file

def test_dataset_loading(test_file: Path):
    """Test dataset loading and sample data."""
    logger.info(f"Testing dataset loading with {test_file}")
    
    try:
        # Process the test file first
        processor = DataProcessor()
        shard_files = processor.process_file(test_file, games_per_shard=100, compress=True)
        
        if not shard_files:
            logger.error("No shard files created")
            return False
        
        logger.info(f"Created {len(shard_files)} shard files")
        
        # Load processed dataset
        dataset = ProcessedDataset(shard_files)
        logger.info(f"Dataset loaded successfully: {len(dataset)} training examples")
        
        # Test a few samples
        for i in range(min(3, len(dataset))):
            try:
                board, policy, value = dataset[i]
                logger.info(f"Sample {i}: board shape {board.shape}, policy shape {policy.shape}, value {value.item():.3f}")
                
                # Check data ranges
                logger.info(f"  Board range: [{board.min():.3f}, {board.max():.3f}]")
                logger.info(f"  Policy range: [{policy.min():.3f}, {policy.max():.3f}], sum: {policy.sum():.3f}")
                logger.info(f"  Value: {value.item():.3f}")
                
            except Exception as e:
                logger.error(f"Error loading sample {i}: {e}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return False

def test_model_forward_pass():
    """Test model forward pass with sample data."""
    logger.info("Testing model forward pass...")
    
    # Create model
    model = create_model("resnet18")
    model = model.to(DEVICE)
    model.eval()
    
    # Create sample data
    batch_size = 4
    boards = torch.randn(batch_size, 2, BOARD_SIZE, BOARD_SIZE).to(DEVICE)
    
    # Test forward pass
    with torch.no_grad():
        policy_logits, value_logit = model(boards)
        
        logger.info(f"Model output shapes:")
        logger.info(f"  Policy logits: {policy_logits.shape}")
        logger.info(f"  Value logit: {value_logit.shape}")
        
        # Check output ranges
        logger.info(f"  Policy range: [{policy_logits.min():.3f}, {policy_logits.max():.3f}]")
        logger.info(f"  Value range: [{value_logit.min():.3f}, {value_logit.max():.3f}]")
        
        # Test softmax on policy
        policy_probs = torch.softmax(policy_logits, dim=1)
        logger.info(f"  Policy probs sum: {policy_probs.sum(dim=1)}")
        
    return True

def test_loss_function():
    """Test the loss function with sample data."""
    logger.info("Testing loss function...")
    
    # Create sample data
    batch_size = 4
    boards = torch.randn(batch_size, 2, BOARD_SIZE, BOARD_SIZE)
    policies = torch.randn(batch_size, POLICY_OUTPUT_SIZE)
    policies = torch.softmax(policies, dim=1)  # Convert to probabilities
    values = torch.rand(batch_size, VALUE_OUTPUT_SIZE)  # [0, 1] range
    
    # Create model and get outputs
    model = create_model("resnet18")
    model.eval()
    
    with torch.no_grad():
        policy_logits, value_logit = model(boards)
        
        # Test loss function
        loss_fn = PolicyValueLoss()
        total_loss, loss_dict = loss_fn(policy_logits, value_logit, policies, values)
        
        logger.info(f"Loss values:")
        logger.info(f"  Total loss: {total_loss.item():.6f}")
        logger.info(f"  Policy loss: {loss_dict['policy_loss']:.6f}")
        logger.info(f"  Value loss: {loss_dict['value_loss']:.6f}")
        
        # Test gradient computation
        model.train()
        policy_logits, value_logit = model(boards)
        total_loss, loss_dict = loss_fn(policy_logits, value_logit, policies, values)
        
        total_loss.backward()
        
        # Check gradients
        grad_norm = 0
        param_count = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
                param_count += 1
        
        grad_norm = grad_norm ** 0.5
        logger.info(f"  Gradient norm: {grad_norm:.6f}")
        logger.info(f"  Parameters with gradients: {param_count}")
        
    return True

def test_small_training_run(test_file: Path):
    """Run a very small training run to test the full pipeline."""
    logger.info("Starting small training run...")
    
    # Process the test file first
    processor = DataProcessor()
    shard_files = processor.process_file(test_file, games_per_shard=50, compress=True)
    
    if not shard_files:
        logger.error("No shard files created")
        return False
    
    logger.info(f"Created {len(shard_files)} shard files")
    
    # Create data loaders with small batch size
    train_loader = create_processed_dataloader(
        shard_files, 
        batch_size=4,  # Small batch size
        shuffle=True,
        num_workers=0
    )
    
    val_loader = create_processed_dataloader(
        shard_files,  # Use same files for validation
        batch_size=4,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    model = create_model("resnet18")
    model = model.to(DEVICE)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=LEARNING_RATE,
        device=DEVICE
    )
    
    # Run a few epochs
    logger.info("Training for 2 epochs...")
    start_time = time.time()
    
    try:
        results = trainer.train(
            num_epochs=2,
            save_dir="./test_checkpoints",
            max_checkpoints=2,
            compress_checkpoints=True
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.1f} seconds")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.6f}")
        
        # Print training metrics
        for epoch, metrics in enumerate(results['train_metrics']):
            logger.info(f"Epoch {epoch+1}:")
            logger.info(f"  Train loss: {metrics['loss']:.6f}")
            logger.info(f"  Train policy loss: {metrics['policy_loss']:.6f}")
            logger.info(f"  Train value loss: {metrics['value_loss']:.6f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

def main():
    """Run all diagnostics."""
    logger.info("=" * 60)
    logger.info("HEX AI TRAINING DIAGNOSTICS")
    logger.info("=" * 60)
    
    log_memory_usage("Start")
    
    # Check data files
    logger.info("\n1. Checking data files...")
    test_file = check_data_files()
    
    # Test dataset loading
    logger.info("\n2. Testing dataset loading...")
    if not test_dataset_loading(test_file):
        logger.error("Dataset loading failed!")
        return
    
    # Test model forward pass
    logger.info("\n3. Testing model forward pass...")
    if not test_model_forward_pass():
        logger.error("Model forward pass failed!")
        return
    
    # Test loss function
    logger.info("\n4. Testing loss function...")
    if not test_loss_function():
        logger.error("Loss function test failed!")
        return
    
    # Test small training run
    logger.info("\n5. Testing small training run...")
    if not test_small_training_run(test_file):
        logger.error("Small training run failed!")
        return
    
    log_memory_usage("End")
    
    logger.info("\n" + "=" * 60)
    logger.info("ALL DIAGNOSTICS PASSED!")
    logger.info("Ready for full training run.")
    logger.info("=" * 60)

if __name__ == "__main__":
    main() 