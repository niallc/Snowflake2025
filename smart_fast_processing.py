#!/usr/bin/env python3
"""
Smart fast data processing that can resume from existing progress.
"""

import time
from pathlib import Path
from fast_data_processing import process_file_fast, load_games_from_file_fast
import logging
from typing import List

logger = logging.getLogger(__name__)


def get_processed_files_info(processed_dir: Path) -> dict:
    """Get information about already processed files."""
    shard_files = list(processed_dir.glob("*.pkl.gz")) + list(processed_dir.glob("*.pkl"))
    
    if not shard_files:
        return {"total_shards": 0, "total_games": 0, "processed_files": set()}
    
    # Count total games
    total_games = 0
    for shard_file in shard_files:
        if shard_file.suffix == '.gz':
            import gzip
            import pickle
            with gzip.open(shard_file, 'rb') as f:
                shard_data = pickle.load(f)
        else:
            import pickle
            with open(shard_file, 'rb') as f:
                shard_data = pickle.load(f)
        total_games += shard_data['num_games']
    
    return {
        "total_shards": len(shard_files),
        "total_games": total_games,
        "processed_files": set()  # We'll populate this based on file patterns
    }


def estimate_processed_files(source_dir: Path, processed_info: dict) -> set:
    """Estimate which source files have been processed based on shard count."""
    # This is a rough estimate - assumes files are processed in order
    # and each file produces roughly similar number of shards
    total_shards = processed_info["total_shards"]
    
    if total_shards == 0:
        return set()
    
    # Get all source files
    trmph_files = list(source_dir.glob("*.trmph"))
    trmph_files.sort()  # Ensure consistent ordering
    
    # Estimate how many files were processed
    # This is approximate - we'll refine it by checking actual file contents
    estimated_files_processed = min(len(trmph_files), total_shards // 2)  # Rough estimate
    
    return set(trmph_files[:estimated_files_processed])


def verify_file_processing(source_file: Path, processed_dir: Path) -> bool:
    """Check if a specific source file has been processed by looking for its content patterns."""
    # This is a simplified check - in practice, you might want to hash file contents
    # For now, we'll assume files are processed in order and check file timestamps
    
    # Get the most recent shard file
    shard_files = list(processed_dir.glob("*.pkl.gz")) + list(processed_dir.glob("*.pkl"))
    if not shard_files:
        return False
    
    # Check if the source file is older than the most recent shard
    latest_shard = max(shard_files, key=lambda f: f.stat().st_mtime)
    source_mtime = source_file.stat().st_mtime
    shard_mtime = latest_shard.stat().st_mtime
    
    # If source file is older than the latest shard, it was likely processed
    return source_mtime < shard_mtime


def process_all_data_smart(source_dir: str = "data/twoNetGames", 
                          processed_dir: str = "data/processed",
                          games_per_shard: int = 1000,
                          compress: bool = True,
                          num_workers: int = None) -> List[Path]:
    """Process all data with smart resumption from existing progress."""
    
    source_path = Path(source_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    
    if num_workers is None:
        import multiprocessing as mp
        num_workers = min(mp.cpu_count(), 8)
    
    # Get existing progress info
    processed_info = get_processed_files_info(processed_path)
    logger.info(f"Found {processed_info['total_shards']} existing shards with {processed_info['total_games']} games")
    
    # Find all source files
    trmph_files = list(source_path.glob("*.trmph"))
    trmph_files.sort()  # Ensure consistent ordering
    
    if not trmph_files:
        raise FileNotFoundError(f"No .trmph files found in {source_dir}")
    
    logger.info(f"Found {len(trmph_files)} source files to process")
    logger.info(f"Using {num_workers} workers for parallel processing")
    
    # Determine which files to skip
    files_to_process = []
    files_to_skip = []
    
    for trmph_file in trmph_files:
        if verify_file_processing(trmph_file, processed_path):
            files_to_skip.append(trmph_file)
        else:
            files_to_process.append(trmph_file)
    
    logger.info(f"Files to skip (already processed): {len(files_to_skip)}")
    logger.info(f"Files to process: {len(files_to_process)}")
    
    if not files_to_process:
        logger.info("All files already processed!")
        return []
    
    # Process remaining files
    start_time = time.time()
    all_shard_files = []
    
    for i, trmph_file in enumerate(files_to_process, 1):
        logger.info(f"Processing file {i}/{len(files_to_process)}: {trmph_file.name}")
        
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
    final_info = get_processed_files_info(processed_path)
    logger.info(f"\n{'='*60}")
    logger.info("SMART DATA PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Files processed: {len(files_to_process)}")
    logger.info(f"Files skipped: {len(files_to_skip)}")
    logger.info(f"New shards created: {len(all_shard_files)}")
    logger.info(f"Total shards now: {final_info['total_shards']}")
    logger.info(f"Total games now: {final_info['total_games']}")
    logger.info(f"Processing time: {processing_time:.1f} seconds")
    if processing_time > 0:
        logger.info(f"Games per second: {len(all_shard_files) * games_per_shard / processing_time:.1f}")
    logger.info(f"{'='*60}")
    
    return all_shard_files


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Process all data with smart resumption
    process_all_data_smart() 