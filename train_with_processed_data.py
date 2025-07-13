#!/usr/bin/env python3
"""
Training script using processed Hex game data from .pkl.gz shards.
"""

import logging
import os
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader

from hex_ai.config import (
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, 
    DEVICE, CHECKPOINT_DIR, BOARD_SIZE
)
from hex_ai.models import create_model
from hex_ai.training import Trainer
from hex_ai.data_processing import create_processed_dataloader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main training function using processed Hex game data."""
    
    # Data paths
    processed_dir = Path("./data/processed")
    
    logger.info(f"Looking for processed data in: {processed_dir}")
    
    # Find all processed shard files
    shard_files = list(processed_dir.glob("*.pkl.gz"))
    if not shard_files:
        raise FileNotFoundError(f"No processed shard files found in {processed_dir}")
    
    logger.info(f"Found {len(shard_files)} processed shard files")
    
    # Split into train/validation (80/20 split)
    random.shuffle(shard_files)
    split_idx = int(0.8 * len(shard_files))
    train_files = shard_files[:split_idx]
    val_files = shard_files[split_idx:]
    
    logger.info(f"Training shards: {len(train_files)}")
    logger.info(f"Validation shards: {len(val_files)}")
    
    # Create data loaders
    logger.info("Creating training data loader...")
    train_loader = create_processed_dataloader(
        train_files,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0  # Start with 0 for debugging
    )
    
    logger.info("Creating validation data loader...")
    val_loader = create_processed_dataloader(
        val_files,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Log dataset sizes
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    
    # Test a batch to verify data format
    logger.info("Testing data format...")
    for batch_idx, (boards, policies, values) in enumerate(train_loader):
        logger.info(f"Sample batch {batch_idx}:")
        logger.info(f"  Boards shape: {boards.shape}")
        logger.info(f"  Policies shape: {policies.shape}")
        logger.info(f"  Values shape: {values.shape}")
        logger.info(f"  Board range: [{boards.min():.3f}, {boards.max():.3f}]")
        logger.info(f"  Policy sum: {policies.sum(dim=1)}")
        logger.info(f"  Value range: [{values.min():.3f}, {values.max():.3f}]")
        break
    
    # Create model
    logger.info("Creating model...")
    model = create_model("resnet18")
    model = model.to(DEVICE)
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=LEARNING_RATE,
        device=DEVICE
    )
    
    # Start training
    logger.info("Starting training with processed data...")
    trainer.train(
        num_epochs=NUM_EPOCHS, 
        save_dir=str(CHECKPOINT_DIR),
        max_checkpoints=5,  # Keep only 5 most recent checkpoints
        compress_checkpoints=True  # Use compression to reduce size
    )
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {trainer.best_val_loss}")

if __name__ == "__main__":
    main() 