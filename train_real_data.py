#!/usr/bin/env python3
"""
Training script using real Hex game data from .trmph files.
"""

import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hex_ai.config import (
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, 
    DEVICE, CHECKPOINT_DIR, BOARD_SIZE
)
from hex_ai.dataset import HexDataset
from hex_ai.models import TwoHeadedResNet
from hex_ai.training import Trainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main training function using real Hex game data."""
    
    # Data paths
    data_dir = Path("./data/twoNetGames")
    train_file = data_dir / "twoNetGames_13x13_mk1_test.trmph"  # Start with test file
    val_file = data_dir / "twoNetGames_13x13_mk1_test.trmph"    # Use same file for now
    
    logger.info(f"Using training data from: {train_file}")
    logger.info(f"Using validation data from: {val_file}")
    
    # Check if files exist
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Validation file not found: {val_file}")
    
    # Create datasets
    logger.info("Creating training dataset...")
    train_dataset = HexDataset(train_file, board_size=BOARD_SIZE)
    
    logger.info("Creating validation dataset...")
    val_dataset = HexDataset(val_file, board_size=BOARD_SIZE)
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0  # Start with 0 for debugging
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    logger.info("Creating model...")
    model = TwoHeadedResNet(resnet_depth=18)
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
    logger.info("Starting training with real data...")
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