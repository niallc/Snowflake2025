"""
Data collection orchestration for Hex AI.

This module provides utilities for collecting and organizing training data from multiple sources.
It handles the business logic of finding, combining, and cleaning trmph files.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import re

from .data_utils import find_trmph_files, extract_games_from_file, remove_duplicates
from .data_config import log_processed_files

logger = logging.getLogger(__name__)


def parse_tournament_date_from_dirname(dirname: str) -> Optional[datetime]:
    """
    Parse tournament date from directory name.
    
    Supports formats like:
    - deterministic_tournament_20250908_1625
    - tournament_50games_2models_250908_09
    
    Args:
        dirname: Directory name to parse
        
    Returns:
        Parsed datetime or None if parsing fails
    """
    # Pattern for deterministic_tournament_YYYYMMDD_HHMM
    deterministic_pattern = r'deterministic_tournament_(\d{8})_(\d{4})'
    match = re.match(deterministic_pattern, dirname)
    if match:
        date_str, time_str = match.groups()
        try:
            return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M")
        except ValueError:
            pass
    
    # Pattern for tournament_*games_*models_YYMMDD_HH
    tournament_pattern = r'tournament_.*_(\d{6})_(\d{2})'
    match = re.match(tournament_pattern, dirname)
    if match:
        date_str, hour_str = match.groups()
        try:
            # Convert YYMMDD to YYYYMMDD (assuming 20xx)
            full_date_str = f"20{date_str}"
            return datetime.strptime(f"{full_date_str}_{hour_str}00", "%Y%m%d_%H%M")
        except ValueError:
            pass
    
    return None


def find_tournament_directories(
    source_dir: Path,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> List[Path]:
    """
    Find tournament directories within a source directory, optionally filtered by date.
    
    Args:
        source_dir: Directory to search for tournament subdirectories
        start_date: Only include tournaments from this date onwards
        end_date: Only include tournaments up to this date
        
    Returns:
        List of tournament directory paths
    """
    if not source_dir.exists():
        logger.warning(f"Source directory does not exist: {source_dir}")
        return []
    
    tournament_dirs = []
    
    for item in source_dir.iterdir():
        if not item.is_dir():
            continue
            
        # Parse date from directory name
        tournament_date = parse_tournament_date_from_dirname(item.name)
        
        if tournament_date is None:
            # Not a tournament directory, skip
            continue
            
        # Apply date filters
        if start_date and tournament_date < start_date:
            continue
        if end_date and tournament_date > end_date:
            continue
            
        tournament_dirs.append(item)
    
    # Sort by date
    tournament_dirs.sort(key=lambda d: parse_tournament_date_from_dirname(d.name))
    
    logger.info(f"Found {len(tournament_dirs)} tournament directories in {source_dir}")
    if start_date or end_date:
        logger.info(f"Date filter: {start_date} to {end_date}")
    
    return tournament_dirs


def find_trmph_files_with_date_filter(
    source_dirs: List[Path],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    include_tournaments: bool = True,
    include_selfplay: bool = True
) -> List[Tuple[Path, Path]]:
    """
    Find TRMPH files with optional date filtering and source type filtering.
    
    Args:
        source_dirs: List of source directories to search
        start_date: Only include files from this date onwards
        end_date: Only include files up to this date
        include_tournaments: Whether to include tournament data
        include_selfplay: Whether to include self-play data
        
    Returns:
        List of (source_dir, file_path) tuples
    """
    all_files = []
    
    for source_dir in source_dirs:
        if not source_dir.exists():
            logger.warning(f"Source directory does not exist: {source_dir}")
            continue
            
        # Handle tournament directories
        if include_tournaments:
            tournament_dirs = find_tournament_directories(source_dir, start_date, end_date)
            for tournament_dir in tournament_dirs:
                # Find .trmph files within tournament directory
                trmph_files = list(tournament_dir.glob("*.trmph"))
                for trmph_file in trmph_files:
                    all_files.append((source_dir, trmph_file))
                    logger.debug(f"Found tournament file: {trmph_file}")
        
        # Handle direct .trmph files (self-play data)
        if include_selfplay:
            trmph_files = list(source_dir.glob("*.trmph"))
            for trmph_file in trmph_files:
                # Check file modification time if date filtering is requested
                if start_date or end_date:
                    file_mtime = datetime.fromtimestamp(trmph_file.stat().st_mtime)
                    if start_date and file_mtime < start_date:
                        continue
                    if end_date and file_mtime > end_date:
                        continue
                
                all_files.append((source_dir, trmph_file))
                logger.debug(f"Found self-play file: {trmph_file}")
    
    logger.info(f"Found {len(all_files)} TRMPH files total")
    return all_files


def collect_tournament_data_since_date(
    source_dirs: List[Path],
    output_dir: Path,
    since_date: datetime,
    chunk_size: int = 20000
) -> Dict:
    """
    Collect tournament data since a specific date.
    
    Args:
        source_dirs: List of source directories to search
        output_dir: Directory to write organized data to
        since_date: Only collect data from this date onwards
        chunk_size: Number of games per chunk file
        
    Returns:
        Dictionary with collection statistics
    """
    logger.info(f"Collecting tournament data since {since_date}")
    
    # Find tournament directories since the specified date
    all_tournament_dirs = []
    for source_dir in source_dirs:
        tournament_dirs = find_tournament_directories(source_dir, start_date=since_date)
        all_tournament_dirs.extend(tournament_dirs)
    
    if not all_tournament_dirs:
        logger.warning(f"No tournament directories found since {since_date}")
        return {"error": "No tournament directories found"}
    
    # Find all TRMPH files in these tournament directories
    all_files = []
    for tournament_dir in all_tournament_dirs:
        trmph_files = list(tournament_dir.glob("*.trmph"))
        for trmph_file in trmph_files:
            all_files.append((tournament_dir.parent, trmph_file))
    
    if not all_files:
        logger.warning("No .trmph files found in tournament directories")
        return {"error": "No TRMPH files found"}
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
        chunk_filename = f"tournament_chunk_{i:03d}.trmph"
        chunk_path = output_dir / chunk_filename
        with open(chunk_path, 'w') as f:
            for game in chunk:
                f.write(game + '\n')
        logger.info(f"Wrote chunk {i} with {len(chunk)} games to {chunk_path}")
    
    # Write summary
    summary_path = output_dir / "tournament_collection_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Tournament Data Collection Summary\n")
        f.write(f"===================================\n")
        f.write(f"Collection date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Since date: {since_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Total source directories: {len(source_dirs)}\n")
        f.write(f"Total tournament directories: {len(all_tournament_dirs)}\n")
        f.write(f"Total input files: {len(all_files)}\n")
        f.write(f"Total games extracted: {len(all_games)}\n")
        f.write(f"Unique games after deduplication: {len(unique_games)}\n")
        f.write(f"Duplicates removed: {len(all_games) - len(unique_games)}\n")
        f.write(f"Output chunks: {len(chunks)}\n")
        f.write(f"Games per chunk: ~{chunk_size}\n")
        
        f.write(f"\nSource directories:\n")
        for source_dir in source_dirs:
            f.write(f"  {source_dir}\n")
        
        f.write(f"\nTournament directories:\n")
        for tournament_dir in all_tournament_dirs:
            f.write(f"  {tournament_dir}\n")
        
        f.write(f"\nStatistics by source:\n")
        for source_name, stats in source_stats.items():
            f.write(f"  {source_name}: {stats['files']} files, {stats['games']} games\n")
        
        f.write(f"\nInput files:\n")
        for source_dir, file_path in all_files:
            f.write(f"  {file_path}\n")
        
        f.write(f"\nOutput files:\n")
        for i in range(len(chunks)):
            f.write(f"  tournament_chunk_{i:03d}.trmph\n")
    
    # Log processed files for tracking
    for source_dir, processed_files in processed_files_by_source.items():
        log_processed_files(source_dir, processed_files)
        logger.info(f"Logged {len(processed_files)} processed files for {source_dir}")
    
    logger.info(f"Tournament collection complete! Summary written to {summary_path}")
    
    return {
        "total_tournament_dirs": len(all_tournament_dirs),
        "total_files": len(all_files),
        "total_games": len(all_games),
        "unique_games": len(unique_games),
        "duplicates_removed": len(all_games) - len(unique_games),
        "chunks_created": len(chunks),
        "source_stats": source_stats,
        "processed_files_by_source": processed_files_by_source
    }


def collect_and_organize_data(
    source_dirs: List[Path], 
    output_dir: Path, 
    chunk_size: int = 20000,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    include_tournaments: bool = True,
    include_selfplay: bool = True
) -> Dict:
    """
    Collect all training data from multiple sources and organize it.
    
    Args:
        source_dirs: List of source directories to search for .trmph files
        output_dir: Directory to write organized data to
        chunk_size: Number of games per chunk file
        start_date: Only include files from this date onwards
        end_date: Only include files up to this date
        include_tournaments: Whether to include tournament data
        include_selfplay: Whether to include self-play data
        
    Returns:
        Dictionary with collection statistics
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all TRMPH files with optional date filtering
    if start_date or end_date or not include_tournaments or not include_selfplay:
        all_files = find_trmph_files_with_date_filter(
            source_dirs, start_date, end_date, include_tournaments, include_selfplay
        )
    else:
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
