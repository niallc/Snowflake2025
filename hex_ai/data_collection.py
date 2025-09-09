"""
Data collection orchestration for Hex AI.

This module provides utilities for collecting and organizing training data from multiple sources.
It handles the business logic of finding, combining, and cleaning trmph files.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Pattern
from datetime import datetime, timedelta
import re

from .data_utils import find_trmph_files, extract_games_from_file, remove_duplicates
from .data_config import log_processed_files

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION: Directory Naming Patterns and Assumptions
# =============================================================================
# 
# This section contains all the assumptions about directory naming conventions.
# If you change your directory naming patterns, update these configurations.
#
# Current assumptions:
# 1. Tournament directories follow specific naming patterns with embedded dates
# 2. Tournament directories contain .trmph files directly
# 3. Self-play data may be in directories or as direct .trmph files
#
# To update for different naming conventions:
# 1. Modify the regex patterns below
# 2. Update the date parsing logic if needed
# 3. Test with your actual directory structure
# =============================================================================

# Tournament directory naming patterns
# These regex patterns are used to identify and parse tournament directories
TOURNAMENT_PATTERNS = {
    # Pattern: deterministic_tournament_YYYYMMDD_HHMM
    # Example: deterministic_tournament_20250908_1625
    'deterministic': {
        'pattern': re.compile(r'deterministic_tournament_(\d{8})_(\d{4})'),
        'date_format': '%Y%m%d_%H%M',
        'description': 'Deterministic tournament directories with full timestamp'
    },
    
    # Pattern: tournament_*games_*models_YYMMDD_HH
    # Example: tournament_50games_2models_250908_09
    'tournament': {
        'pattern': re.compile(r'tournament_.*_(\d{6})_(\d{2})'),
        'date_format': '%Y%m%d_%H%M',  # Will be converted from YYMMDD to YYYYMMDD
        'description': 'General tournament directories with abbreviated date'
    }
}

# File extensions to look for
TRMPH_EXTENSION = '.trmph'

# Default date assumptions
DEFAULT_CENTURY_PREFIX = '20'  # For YYMMDD -> YYYYMMDD conversion

# Validation settings
MIN_TOURNAMENT_DIRS_FOR_WARNING = 5  # Warn if fewer than this many tournament dirs found
MAX_DATE_LOOKBACK_DAYS = 62  # Refuse to look back more than this many days


def parse_tournament_date_from_dirname(dirname: str) -> Optional[datetime]:
    """
    Parse tournament date from directory name using configured patterns.
    
    This function tries each configured tournament pattern in order and returns
    the first successful match. If no patterns match, returns None.
    
    Args:
        dirname: Directory name to parse
        
    Returns:
        Parsed datetime or None if no patterns match
        
    Raises:
        ValueError: If a pattern matches but date parsing fails (indicates config error)
    """
    for pattern_name, pattern_config in TOURNAMENT_PATTERNS.items():
        pattern = pattern_config['pattern']
        date_format = pattern_config['date_format']
        
        match = pattern.match(dirname)
        if match:
            try:
                if pattern_name == 'tournament':
                    # Handle abbreviated date format (YYMMDD -> YYYYMMDD)
                    date_str, hour_str = match.groups()
                    full_date_str = f"{DEFAULT_CENTURY_PREFIX}{date_str}"
                    date_time_str = f"{full_date_str}_{hour_str}00"
                else:
                    # Handle full date format
                    date_str, time_str = match.groups()
                    date_time_str = f"{date_str}_{time_str}"
                
                return datetime.strptime(date_time_str, date_format)
                
            except ValueError as e:
                # This indicates a configuration error - the pattern matched but date parsing failed
                raise ValueError(
                    f"Configuration error: Pattern '{pattern_name}' matched directory '{dirname}' "
                    f"but date parsing failed. Pattern: {pattern.pattern}, "
                    f"Date format: {date_format}, Error: {e}"
                )
    
    return None


def validate_tournament_patterns() -> List[str]:
    """
    Validate that all configured tournament patterns are syntactically correct.
    
    Returns:
        List of validation errors (empty if all patterns are valid)
    """
    errors = []
    
    for pattern_name, pattern_config in TOURNAMENT_PATTERNS.items():
        try:
            # Test that the pattern compiles (should already be compiled, but double-check)
            pattern = pattern_config['pattern']
            if not isinstance(pattern, re.Pattern):
                errors.append(f"Pattern '{pattern_name}' is not a compiled regex pattern")
                continue
            
            # Test that the date format is valid
            date_format = pattern_config['date_format']
            try:
                # Try parsing a test date to validate the format
                test_date = datetime(2025, 1, 1, 12, 0)
                test_str = test_date.strftime(date_format)
                parsed_date = datetime.strptime(test_str, date_format)
                if parsed_date != test_date:
                    errors.append(f"Pattern '{pattern_name}' date format '{date_format}' is not reversible")
            except ValueError as e:
                errors.append(f"Pattern '{pattern_name}' has invalid date format '{date_format}': {e}")
            
            # Test that the pattern has the expected number of groups
            if pattern_name == 'deterministic':
                if pattern.groups != 2:
                    errors.append(f"Pattern '{pattern_name}' should have 2 groups (date, time), but has {pattern.groups}")
            elif pattern_name == 'tournament':
                if pattern.groups != 2:
                    errors.append(f"Pattern '{pattern_name}' should have 2 groups (date, hour), but has {pattern.groups}")
                    
        except Exception as e:
            errors.append(f"Error validating pattern '{pattern_name}': {e}")
    
    return errors


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
        
    Raises:
        ValueError: If configuration is invalid or date range is unreasonable
        FileNotFoundError: If source directory doesn't exist
    """
    # Validate configuration first
    config_errors = validate_tournament_patterns()
    if config_errors:
        raise ValueError(f"Tournament pattern configuration errors: {'; '.join(config_errors)}")
    
    # Validate date range
    if start_date and end_date and start_date > end_date:
        raise ValueError(f"Start date {start_date} is after end date {end_date}")
    
    if start_date:
        days_back = (datetime.now() - start_date).days
        if days_back > MAX_DATE_LOOKBACK_DAYS:
            raise ValueError(
                f"Start date {start_date} is {days_back} days ago, which exceeds "
                f"maximum lookback of {MAX_DATE_LOOKBACK_DAYS} days. "
                f"Adjust MAX_DATE_LOOKBACK_DAYS in configuration if needed."
            )
    
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    
    if not source_dir.is_dir():
        raise ValueError(f"Source path is not a directory: {source_dir}")
    
    tournament_dirs = []
    parse_errors = []
    
    try:
        for item in source_dir.iterdir():
            if not item.is_dir():
                continue
                
            # Parse date from directory name
            try:
                tournament_date = parse_tournament_date_from_dirname(item.name)
            except ValueError as e:
                # Configuration error - this is serious
                parse_errors.append(f"Directory '{item.name}': {e}")
                continue
            
            if tournament_date is None:
                # Not a tournament directory, skip silently
                continue
                
            # Apply date filters
            if start_date and tournament_date < start_date:
                continue
            if end_date and tournament_date > end_date:
                continue
                
            tournament_dirs.append(item)
        
        # Report configuration errors if any occurred
        if parse_errors:
            error_msg = f"Configuration errors detected while parsing directories:\n" + "\n".join(parse_errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Sort by date
        tournament_dirs.sort(key=lambda d: parse_tournament_date_from_dirname(d.name))
        
        # Warn if very few tournament directories found
        if len(tournament_dirs) < MIN_TOURNAMENT_DIRS_FOR_WARNING:
            logger.warning(
                f"Only found {len(tournament_dirs)} tournament directories in {source_dir}. "
                f"This might indicate:\n"
                f"1. Directory naming patterns have changed (check TOURNAMENT_PATTERNS config)\n"
                f"2. No recent tournament data available\n"
                f"3. Source directory structure is different than expected"
            )
        
        logger.info(f"Found {len(tournament_dirs)} tournament directories in {source_dir}")
        if start_date or end_date:
            logger.info(f"Date filter: {start_date} to {end_date}")
        
        return tournament_dirs
        
    except Exception as e:
        logger.error(f"Error scanning directory {source_dir}: {e}")
        raise


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


def test_tournament_patterns_with_examples() -> Dict[str, List[str]]:
    """
    Test the configured tournament patterns with example directory names.
    
    This function helps validate that the patterns work as expected with
    typical directory names. Add your actual directory names here for testing.
    
    Returns:
        Dictionary mapping pattern names to lists of test results
    """
    # Example directory names for testing
    # Add your actual directory names here to test the patterns
    test_examples = [
        "deterministic_tournament_20250908_1625",
        "deterministic_tournament_20250903_0750", 
        "tournament_50games_2models_250908_09",
        "tournament_400games_3models_250907_01",
        "not_a_tournament_dir",
        "some_other_directory",
    ]
    
    results = {}
    
    for pattern_name, pattern_config in TOURNAMENT_PATTERNS.items():
        pattern = pattern_config['pattern']
        results[pattern_name] = []
        
        for example in test_examples:
            match = pattern.match(example)
            if match:
                try:
                    parsed_date = parse_tournament_date_from_dirname(example)
                    if parsed_date:
                        results[pattern_name].append(f"✓ '{example}' -> {parsed_date}")
                    else:
                        results[pattern_name].append(f"✗ '{example}' -> pattern matched but date parsing failed")
                except Exception as e:
                    results[pattern_name].append(f"✗ '{example}' -> error: {e}")
            else:
                results[pattern_name].append(f"- '{example}' -> no match")
    
    return results


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
