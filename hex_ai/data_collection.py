"""
Data collection orchestration for Hex AI.

This module provides utilities for collecting and organizing training data from multiple sources.
It handles the business logic of finding, combining, and cleaning trmph files.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

from .data_utils import find_trmph_files, extract_games_from_file, remove_duplicates
from .data_config import log_processed_files

logger = logging.getLogger(__name__)


def collect_and_organize_data(
    source_dirs: List[Path], 
    output_dir: Path, 
    chunk_size: int = 20000
) -> Dict:
    """
    Collect all training data from multiple sources and organize it.
    
    Args:
        source_dirs: List of source directories to search for .trmph files
        output_dir: Directory to write organized data to
        chunk_size: Number of games per chunk file
        
    Returns:
        Dictionary with collection statistics
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all TRMPH files
    all_files = find_trmph_files(source_dirs)
    if not all_files:
        logger.error("No .trmph files found in any source directory")
        return {"error": "No files found"}
    
    # Extract all games from all files
    all_games = []
    source_stats = {}
    processed_files_by_source = {}
    
    for source_dir, file_path in all_files:
        logger.info(f"Processing {file_path}")
        games = extract_games_from_file(file_path)
        all_games.extend(games)
        
        # Track statistics by source
        source_name = source_dir.name
        if source_name not in source_stats:
            source_stats[source_name] = {"files": 0, "games": 0}
            processed_files_by_source[source_dir] = []
        source_stats[source_name]["files"] += 1
        source_stats[source_name]["games"] += len(games)
        processed_files_by_source[source_dir].append(file_path)
        
        logger.info(f"  Extracted {len(games)} games from {file_path.name}")
    
    logger.info(f"Total games extracted: {len(all_games)}")
    
    # Remove duplicates
    unique_games = remove_duplicates(all_games)
    
    # Split into chunks
    chunks = []
    for i in range(0, len(unique_games), chunk_size):
        chunk = unique_games[i:i + chunk_size]
        chunks.append(chunk)
    
    logger.info(f"Split {len(unique_games)} games into {len(chunks)} chunks")
    
    # Write chunks to files
    for i, chunk in enumerate(chunks):
        chunk_filename = f"collected_chunk_{i:03d}.trmph"
        chunk_path = output_dir / chunk_filename
        with open(chunk_path, 'w') as f:
            for game in chunk:
                f.write(game + '\n')
        logger.info(f"Wrote chunk {i} with {len(chunk)} games to {chunk_path}")
    
    # Write summary
    summary_path = output_dir / "collection_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Training Data Collection Summary\n")
        f.write(f"================================\n")
        f.write(f"Collection date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Total source directories: {len(source_dirs)}\n")
        f.write(f"Total input files: {len(all_files)}\n")
        f.write(f"Total games extracted: {len(all_games)}\n")
        f.write(f"Unique games after deduplication: {len(unique_games)}\n")
        f.write(f"Duplicates removed: {len(all_games) - len(unique_games)}\n")
        f.write(f"Output chunks: {len(chunks)}\n")
        f.write(f"Games per chunk: ~{chunk_size}\n")
        
        f.write(f"\nSource directories:\n")
        for source_dir in source_dirs:
            f.write(f"  {source_dir}\n")
        
        f.write(f"\nStatistics by source:\n")
        for source_name, stats in source_stats.items():
            f.write(f"  {source_name}: {stats['files']} files, {stats['games']} games\n")
        
        f.write(f"\nInput files:\n")
        for source_dir, file_path in all_files:
            f.write(f"  {file_path}\n")
        
        f.write(f"\nOutput files:\n")
        for i in range(len(chunks)):
            f.write(f"  collected_chunk_{i:03d}.trmph\n")
    
    # Log processed files for tracking (but don't move them)
    for source_dir, processed_files in processed_files_by_source.items():
        log_processed_files(source_dir, processed_files)
        logger.info(f"Logged {len(processed_files)} processed files for {source_dir}")
    
    logger.info(f"Collection complete! Summary written to {summary_path}")
    
    return {
        "total_files": len(all_files),
        "total_games": len(all_games),
        "unique_games": len(unique_games),
        "duplicates_removed": len(all_games) - len(unique_games),
        "chunks_created": len(chunks),
        "source_stats": source_stats,
        "processed_files_by_source": processed_files_by_source
    }


def combine_and_clean_files(input_dir: Path, output_dir: Path, chunk_size: int = 20000):
    """Combine all TRMPH files, remove duplicates, and split into chunks."""
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all TRMPH files
    trmph_files = list(input_dir.glob("*.trmph"))
    if not trmph_files:
        logger.error(f"No .trmph files found in {input_dir}")
        # Still create a summary file even if no files were found
        summary_path = output_dir / "processing_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Self-play data preprocessing summary\n")
            f.write(f"=====================================\n")
            f.write(f"Input directory: {input_dir}\n")
            f.write(f"Output directory: {output_dir}\n")
            f.write(f"Total input files: 0\n")
            f.write(f"Total games extracted: 0\n")
            f.write(f"Unique games after deduplication: 0\n")
            f.write(f"Duplicates removed: 0\n")
            f.write(f"Output chunks: 0\n")
            f.write(f"Games per chunk: ~{chunk_size}\n")
            f.write(f"\nNo .trmph files found in input directory.\n")
        logger.info(f"Processing complete! Summary written to {summary_path}")
        return
    
    # Extract all games from all files
    all_games = []
    for file_path in trmph_files:
        logger.info(f"Processing {file_path}")
        games = extract_games_from_file(file_path)
        all_games.extend(games)
        logger.info(f"  Extracted {len(games)} games from {file_path.name}")
    
    logger.info(f"Total games extracted: {len(all_games)}")
    
    # Remove duplicates
    unique_games = remove_duplicates(all_games)
    
    # Split into chunks
    chunks = []
    for i in range(0, len(unique_games), chunk_size):
        chunk = unique_games[i:i + chunk_size]
        chunks.append(chunk)
    
    logger.info(f"Split {len(unique_games)} games into {len(chunks)} chunks of ~{chunk_size} games each")
    
    # Write chunks to files
    for i, chunk in enumerate(chunks):
        chunk_filename = f"cleaned_chunk_{i:03d}.trmph"
        chunk_path = output_dir / chunk_filename
        with open(chunk_path, 'w') as f:
            for game in chunk:
                f.write(game + '\n')
        logger.info(f"Wrote chunk {i} with {len(chunk)} games to {chunk_path}")
    
    # Write summary
    summary_path = output_dir / "processing_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Self-play data preprocessing summary\n")
        f.write(f"=====================================\n")
        f.write(f"Input directory: {input_dir}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Total input files: {len(trmph_files)}\n")
        f.write(f"Total games extracted: {len(all_games)}\n")
        f.write(f"Unique games after deduplication: {len(unique_games)}\n")
        f.write(f"Duplicates removed: {len(all_games) - len(unique_games)}\n")
        f.write(f"Output chunks: {len(chunks)}\n")
        f.write(f"Games per chunk: ~{chunk_size}\n")
        f.write(f"\nInput files:\n")
        for file_path in trmph_files:
            f.write(f"  {file_path.name}\n")
        f.write(f"\nOutput files:\n")
        for i in range(len(chunks)):
            f.write(f"  cleaned_chunk_{i:03d}.trmph\n")
    
    logger.info(f"Processing complete! Summary written to {summary_path}")


def parse_shard_range(range_str: str, data_dir: str = None) -> tuple:
    """
    Parse shard range string and return (start, end) tuple.
    
    Args:
        range_str: Range string in format "start-end" or "all"
        data_dir: Optional data directory for validation (not used in parsing)
        
    Returns:
        Tuple of (start, end) where end=None means use all available shards
        
    Raises:
        ValueError: If range format is invalid
    """
    if range_str.lower() == 'all':
        return (0, None)  # None means use all available shards
    
    if '-' not in range_str:
        raise ValueError(f"Invalid shard range format: {range_str}. Use 'start-end' or 'all'")
    
    try:
        start, end = range_str.split('-', 1)
        start = int(start)
        end = int(end)
        
        if start < 0 or end < 0:
            raise ValueError(f"Shard numbers must be non-negative: {range_str}")
        if start > end:
            raise ValueError(f"Start shard must be <= end shard: {range_str}")
        
        return (start, end)
    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError(f"Invalid shard range format: {range_str}. Use 'start-end' or 'all'")
        raise


def parse_shard_ranges(shard_ranges: list, data_dirs: list, logger=None) -> tuple:
    """
    Parse multiple shard ranges and convert to skip_files and max_files format.
    
    Args:
        shard_ranges: List of shard range strings (e.g., ["251-300", "all"])
        data_dirs: List of data directory paths for validation
        logger: Optional logger for warnings
        
    Returns:
        Tuple of (skip_files_list, max_files_list) - one value per data directory
        
    Raises:
        ValueError: If parsing fails or validation fails
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Set default shard ranges if not provided
    if not shard_ranges:
        shard_ranges = ["all"] * len(data_dirs)
        logger.info(f"Using default shard ranges: {shard_ranges}")
    
    # Validate lengths match
    if len(shard_ranges) != len(data_dirs):
        raise ValueError(f"Number of shard ranges ({len(shard_ranges)}) must match number of data directories ({len(data_dirs)})")
    
    # Parse shard ranges and convert to skip_files and max_files format
    skip_files = []
    max_files = []
    for i, (range_str, data_dir) in enumerate(zip(shard_ranges, data_dirs)):
        try:
            start, end = parse_shard_range(range_str, data_dir)
            
            if end is None:  # 'all' case
                skip_files.append(0)
                max_files.append(None)  # No limit
            else:
                # For range start-end, we skip the first 'start' files and limit to 'end-start+1' files
                skip_files.append(start)
                max_files.append(end - start + 1)  # Number of files in the range
                
                # Check if the data directory has enough shards
                data_path = Path(data_dir)
                if data_path.exists():
                    shard_files = list(data_path.glob("shuffled_*.pkl.gz"))
                    max_shard = len(shard_files) - 1
                    if end > max_shard:
                        logger.warning(f"Data directory {data_dir} only has shards 0-{max_shard}, but range specifies up to {end}")
                else:
                    logger.warning(f"Data directory {data_dir} does not exist")
                    
        except ValueError as e:
            raise ValueError(f"Error parsing shard range for {data_dir}: {e}")
    
    logger.info(f"Parsed shard ranges: {list(zip(data_dirs, shard_ranges, skip_files, max_files))}")
    return skip_files, max_files
