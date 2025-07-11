#!/usr/bin/env python3
"""
Example training script for Hex AI model.

This script demonstrates how to train a model using the training module.
"""

import logging
from hex_ai.models import create_model
from hex_ai.training import train_model
from hex_ai.data_utils import convert_to_matrix_format
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_data(num_samples: int = 100) -> list:
    """Create dummy training data for testing."""
    dummy_data = []
    
    # Create some simple trmph strings
    base_patterns = [
        "http://www.trmph.com/hex/board#13,a1b2c3",
        "http://www.trmph.com/hex/board#13,a1b2c3d4e5f6g7",
        "http://www.trmph.com/hex/board#13,a1b2c3d4e5f6g7h8i9j10k11l12m13",
        "http://www.trmph.com/hex/board#11,a8h1b8h2c8h3d8h4e8h5f8h6g8h7a7a6b6b5c5c4d4d3e3e2f2f1g1",  # Blue winner
        "http://www.trmph.com/hex/board#11,a8h1b8h2c8h3d8h4e8h5f8h6g8h7b1a1a3a2c1b2c2b3a4c3b4d3e3d4c5e4f4e5d6f5g5f6g6f7e7g7",  # Red winner
    ]
    
    for i in range(num_samples):
        # Randomly select and modify a base pattern
        pattern = np.random.choice(base_patterns)
        # Add some random moves to make it unique
        if "a1b2c3" in pattern:
            extra_moves = "d4e5f6" if i % 2 == 0 else "g7h8i9"
            pattern = pattern.replace("a1b2c3", f"a1b2c3{extra_moves}")
        
        dummy_data.append(pattern)
    
    return dummy_data

def main():
    """Main training function."""
    logger.info("Creating dummy training data...")
    train_data = create_dummy_data(50)
    val_data = create_dummy_data(10)
    
    logger.info(f"Created {len(train_data)} training samples and {len(val_data)} validation samples")
    
    # Create model
    logger.info("Creating model...")
    model = create_model("resnet18")
    
    # Train model
    logger.info("Starting training...")
    results = train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        num_epochs=5,  # Short training for testing
        batch_size=8,   # Small batch size for testing
        learning_rate=0.001,
        save_dir="test_checkpoints"
    )
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
    
    # Print some training history
    for epoch_data in results['history'][-3:]:  # Last 3 epochs
        epoch = epoch_data['epoch']
        train_loss = epoch_data['train']['total_loss']
        val_loss = epoch_data['val']['total_loss'] if epoch_data['val'] else None
        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f if val_loss else 'N/A'}")

if __name__ == "__main__":
    main() 