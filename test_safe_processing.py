#!/usr/bin/env python3
"""
Test the safe processing with a small subset of data.
"""

import logging
from pathlib import Path
from safe_shard_processing import process_all_data_safely

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_safe_processing():
    """Test safe processing with a small subset of files."""
    logger.info("Testing safe processing with small subset...")
    
    # Create a test directory with just a few small files
    test_source_dir = "data/test_twoNetGames"
    test_processed_dir = "data/test_processed"
    
    # Create test directory
    test_source_path = Path(test_source_dir)
    test_source_path.mkdir(parents=True, exist_ok=True)
    
    # Copy a few small files to test directory
    source_path = Path("data/twoNetGames")
    small_files = []
    
    for file_path in source_path.glob("*.trmph"):
        if file_path.stat().st_size < 100 * 1024:  # Less than 100KB
            small_files.append(file_path)
            if len(small_files) >= 3:  # Just 3 small files
                break
    
    if not small_files:
        logger.error("No small files found for testing")
        return False
    
    # Copy files to test directory
    for file_path in small_files:
        import shutil
        shutil.copy2(file_path, test_source_path / file_path.name)
        logger.info(f"Copied {file_path.name} to test directory")
    
    # Process the test files
    try:
        shard_files = process_all_data_safely(
            source_dir=test_source_dir,
            processed_dir=test_processed_dir,
            games_per_shard=100,  # Smaller shards for testing
        )
        
        logger.info(f"Test processing completed successfully!")
        logger.info(f"Created {len(shard_files)} shard files")
        
        # Clean up test files
        import shutil
        shutil.rmtree(test_source_path, ignore_errors=True)
        shutil.rmtree(test_processed_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        logger.error(f"Test processing failed: {e}")
        return False

if __name__ == "__main__":
    test_safe_processing() 