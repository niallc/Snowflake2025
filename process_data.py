#!/usr/bin/env python3
"""
Script to process raw .trmph files into efficient training formats.

This script:
1. Converts .trmph files to processed tensors
2. Shards data into manageable chunks
3. Compresses data for efficient storage
4. Creates training-ready datasets
"""

import logging
from pathlib import Path

from hex_ai.data_processing import DataProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Process raw data into efficient training format."""
    
    # Initialize processor
    processor = DataProcessor(processed_dir="data/processed")
    
    # Process the test file first to see how it works
    test_file = Path("data/twoNetGames/twoNetGames_13x13_mk1_test.trmph")
    
    logger.info("Processing test file...")
    shard_files = processor.process_file(
        test_file, 
        games_per_shard=50,  # Small shards for testing
        compress=True
    )
    
    logger.info(f"Created {len(shard_files)} shard files:")
    for shard_file in shard_files:
        size_mb = shard_file.stat().st_size / (1024 * 1024)
        logger.info(f"  {shard_file.name}: {size_mb:.2f} MB")
    
    # Now process a larger file
    large_file = Path("data/twoNetGames/twoNetGames_13x13_mk1_d2b6_1759.trmph")
    
    if large_file.exists():
        logger.info("Processing larger file...")
        large_shards = processor.process_file(
            large_file,
            games_per_shard=500,  # Larger shards for bigger files
            compress=True
        )
        
        logger.info(f"Created {len(large_shards)} shard files from large file:")
        for shard_file in large_shards:
            size_mb = shard_file.stat().st_size / (1024 * 1024)
            logger.info(f"  {shard_file.name}: {size_mb:.2f} MB")
    
    logger.info("Data processing completed!")

if __name__ == "__main__":
    main() 