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
    """Create dummy training data for testing, ensuring no duplicate moves."""
    dummy_data = []
    
    # All possible moves for 13x13
    letters = 'abcdefghijklm'
    numbers = [str(i+1) for i in range(13)]
    all_moves_13 = [l+n for l in letters for n in numbers]
    # All possible moves for 7x7
    letters7 = 'abcdefg'
    numbers7 = [str(i+1) for i in range(7)]
    all_moves_7 = [l+n for l in letters7 for n in numbers7]
    
    base_patterns = [
        ("http://www.trmph.com/hex/board#13,", ["a1", "b2", "c3"]),
        ("http://www.trmph.com/hex/board#13,", ["a1", "b2", "c3", "d4", "e5", "f6", "g7"]),
        ("http://www.trmph.com/hex/board#13,", ["a1", "b2", "c3", "d4", "e5", "f6", "g7", "h8", "i9", "j10", "k11", "l12", "m13"]),
        ("http://www.trmph.com/hex/board#11,", ["a8", "h1", "b8", "h2", "c8", "h3", "d8", "h4", "e8", "h5", "f8", "h6", "g8", "h7", "a7", "a6", "b6", "b5", "c5", "c4", "d4", "d3", "e3", "e2", "f2", "f1", "g1"]),
        ("http://www.trmph.com/hex/board#11,", ["a8", "h1", "b8", "h2", "c8", "h3", "d8", "h4", "e8", "h5", "f8", "h6", "g8", "h7", "b1", "a1", "a3", "a2", "c1", "b2", "c2", "b3", "a4", "c3", "b4", "d3", "e3", "d4", "c5", "e4", "f4", "e5", "d6", "f5", "g5", "f6", "g6", "f7", "e7", "g7"]),
    ]
    
    for i in range(num_samples):
        base_url, base_moves = base_patterns[np.random.randint(len(base_patterns))]
        moves = list(base_moves)
        if base_url.endswith('#13,'):
            available_moves = set(all_moves_13) - set(moves)
        else:
            available_moves = set(all_moves_7) - set(moves)
        # Add up to 3 extra unique moves
        extra_moves = np.random.choice(list(available_moves), size=min(3, len(available_moves)), replace=False)
        moves.extend(extra_moves)
        trmph = base_url + ''.join(moves)
        dummy_data.append(trmph)
    
    return dummy_data

def log_epoch_summary(epoch, train_loss, val_loss):
    """Utility to log epoch summary with proper formatting."""
    if val_loss is not None:
        val_str = f"{val_loss:.4f}"
    else:
        val_str = "N/A"
    logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_str}")

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
        log_epoch_summary(epoch, train_loss, val_loss)

if __name__ == "__main__":
    main() 