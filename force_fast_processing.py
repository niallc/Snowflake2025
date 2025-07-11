#!/usr/bin/env python3
"""
Force fast data processing - processes all files regardless of existing progress.
Use this to complete the dataset processing.
"""

import time
from pathlib import Path
from fast_data_processing import process_file_fast
import logging
from typing import List

logger = logging.getLogger(__name__)


def process_all_data_force(source_dir: str = "data/twoNetGames", 
                          processed_dir: str = "data/processed",
                          games_per_shard: int = 1000,
                          compress: bool = True,
                          num_workers: int = None) -> List[Path]:
    """Process all data, forcing processing of all files."""
    
    source_path = Path(source_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    
    if num_workers is None:
        import multiprocessing as mp
        num_workers = min(mp.cpu_count(), 8)
    
    # Get existing progress info
    existing_shards = list(processed_path.glob("*.pkl.gz")) + list(processed_path.glob("*.pkl"))
    logger.info(f"Found {len(existing_shards)} existing shards")
    
    # Find all source files
    trmph_files = list(source_path.glob("*.trmph"))
    trmph_files.sort()  # Ensure consistent ordering
    
    if not trmph_files:
        raise FileNotFoundError(f"No .trmph files found in {source_dir}")
    
    logger.info(f"Found {len(trmph_files)} source files to process")
    logger.info(f"Using {num_workers} workers for parallel processing")
    logger.info("FORCING PROCESSING OF ALL FILES")
    
    # Process all files
    start_time = time.time()
    all_shard_files = []
    
    for i, trmph_file in enumerate(trmph_files, 1):
        logger.info(f"Processing file {i}/{len(trmph_files)}: {trmph_file.name}")
        
        try:
            shard_files = process_file_fast(
                trmph_file, processed_path, games_per_shard, compress, num_workers
            )
            all_shard_files.extend(shard_files)
            
        except Exception as e:
            logger.error(f"Error processing {trmph_file.name}: {e}")
            continue
    
    processing_time = time.time() - start_time
    
    # Final summary
    final_shards = list(processed_path.glob("*.pkl.gz")) + list(processed_path.glob("*.pkl"))
    logger.info(f"\n{'='*60}")
    logger.info("FORCE DATA PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Files processed: {len(trmph_files)}")
    logger.info(f"New shards created: {len(all_shard_files)}")
    logger.info(f"Total shards now: {len(final_shards)}")
    logger.info(f"Processing time: {processing_time:.1f} seconds")
    if processing_time > 0:
        logger.info(f"Games per second: {len(all_shard_files) * games_per_shard / processing_time:.1f}")
    logger.info(f"{'='*60}")
    
    return all_shard_files


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Process all data with force
    process_all_data_force() 