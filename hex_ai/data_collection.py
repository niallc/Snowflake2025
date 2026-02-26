"""
Data collection orchestration for Hex AI.

This module provides utilities for collecting and organizing training data from multiple sources.
It handles the business logic of finding, combining, and cleaning trmph files.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from .data_utils import find_trmph_files, extract_games_from_file, remove_duplicates
from .data_config import log_processed_files
from .move_provenance import (
    MoveProvenanceRecord,
    load_move_provenance_sidecar,
    make_move_provenance_record,
    sidecar_path_for_trmph,
)
from hex_ai.data_pipeline import discover_training_data_files_all, discover_training_data_files_by_shards

logger = logging.getLogger(__name__)

MAX_DATE_LOOKBACK_DAYS = 62  # Fail fast to avoid accidentally collecting months of data.


def _validate_date_range(start_date: Optional[datetime], end_date: Optional[datetime]) -> None:
    if start_date and end_date and start_date > end_date:
        raise ValueError(f"Start date {start_date} is after end date {end_date}")
    
    if start_date:
        days_back = (datetime.now() - start_date).days
        if days_back > MAX_DATE_LOOKBACK_DAYS:
            raise ValueError(
                f"Start date {start_date} is {days_back} days ago, which exceeds "
                f"MAX_DATE_LOOKBACK_DAYS={MAX_DATE_LOOKBACK_DAYS}."
            )


def find_trmph_files_with_date_filter(
    source_dirs: List[Path],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> List[Tuple[Path, Path]]:
    """
    Find `.trmph` files (recursively) and optionally filter them by filesystem modification time.
    
    Args:
        source_dirs: List of source directories to search
        start_date: Only include files with mtime >= start_date
        end_date: Only include files with mtime <= end_date
        
    Returns:
        List of (source_dir, file_path) tuples
    """
    _validate_date_range(start_date, end_date)

    all_files = find_trmph_files(source_dirs)
    if not (start_date or end_date):
        return all_files

    filtered: List[Tuple[Path, Path]] = []
    for source_dir, file_path in all_files:
        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        if start_date and file_mtime < start_date:
            continue
        if end_date and file_mtime > end_date:
            continue
        filtered.append((source_dir, file_path))

    logger.info(f"Filtered {len(all_files)} -> {len(filtered)} .trmph files by mtime")
    return filtered

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
    
    all_files = find_trmph_files_with_date_filter(source_dirs, start_date=since_date)
    if not all_files:
        logger.warning(f"No .trmph files found with mtime >= {since_date}")
        return {"error": "No TRMPH files found"}

    dirs_with_data = sorted({file_path.parent for _, file_path in all_files})
    
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
        f.write(f"Directories with data: {len(dirs_with_data)}\n")
        f.write(f"Total input files: {len(all_files)}\n")
        f.write(f"Total games extracted: {len(all_games)}\n")
        f.write(f"Unique games after deduplication: {len(unique_games)}\n")
        f.write(f"Duplicates removed: {len(all_games) - len(unique_games)}\n")
        f.write(f"Output chunks: {len(chunks)}\n")
        f.write(f"Games per chunk: ~{chunk_size}\n")
        
        f.write(f"\nSource directories:\n")
        for source_dir in source_dirs:
            f.write(f"  {source_dir}\n")
        
        f.write(f"\nDirectories with data:\n")
        for d in dirs_with_data:
            f.write(f"  {d}\n")
        
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
        "total_dirs_with_data": len(dirs_with_data),
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
    end_date: Optional[datetime] = None
) -> Dict:
    """
    Collect all training data from multiple sources and organize it.
    
    Args:
        source_dirs: List of source directories to search for .trmph files
        output_dir: Directory to write organized data to
        chunk_size: Number of games per chunk file
        start_date: Only include files with mtime >= start_date
        end_date: Only include files with mtime <= end_date
        
    Returns:
        Dictionary with collection statistics
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all TRMPH files with optional date filtering
    if start_date or end_date:
        all_files = find_trmph_files_with_date_filter(source_dirs, start_date, end_date)
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


def combine_and_clean_files(
    input_dirs: List[Path] | Path,
    output_dir: Path,
    chunk_size: int = 20000,
    policy_provenance_mode: str = "off",
):
    """
    Combine TRMPH files, remove duplicates, split into chunks, and optionally propagate provenance.

    Args:
        input_dirs: Source directory (or directories) containing .trmph files
        output_dir: Output directory for cleaned chunks
        chunk_size: Number of games per cleaned chunk
        policy_provenance_mode: 'off' or 'require'
    """
    if policy_provenance_mode not in {"off", "require"}:
        raise ValueError(
            f"Invalid policy_provenance_mode {policy_provenance_mode!r} "
            "(expected 'off' or 'require')"
        )
    provenance_required = policy_provenance_mode == "require"

    if isinstance(input_dirs, Path):
        normalized_input_dirs = [input_dirs]
    else:
        normalized_input_dirs = [Path(d) for d in input_dirs]

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all TRMPH files from all input directories
    all_trmph_files: List[Path] = []
    for input_dir in normalized_input_dirs:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory {input_dir} does not exist - this is likely a configuration error")
        trmph_files = sorted(input_dir.glob("*.trmph"))
        all_trmph_files.extend(trmph_files)
        logger.info(f"Found {len(trmph_files)} .trmph files in {input_dir}")
    
    if not all_trmph_files:
        raise RuntimeError(
            f"No .trmph files found in any input directory: {[str(d) for d in normalized_input_dirs]}. "
            "This suggests a configuration error or empty directories."
        )
    
    # Extract all games from all files
    all_games: List[str] = []
    game_to_provenance: Dict[str, MoveProvenanceRecord] = {}
    for file_path in all_trmph_files:
        logger.info(f"Processing {file_path}")
        games = extract_games_from_file(file_path)
        if provenance_required:
            sidecar_path = sidecar_path_for_trmph(file_path)
            provenance_records = load_move_provenance_sidecar(sidecar_path)
            if len(provenance_records) != len(games):
                raise ValueError(
                    f"Provenance record count mismatch for {file_path}: "
                    f"expected {len(games)}, found {len(provenance_records)} in {sidecar_path}"
                )

            for game_index, game_line in enumerate(games):
                record = provenance_records[game_index]
                if record.game_index != game_index:
                    raise ValueError(
                        f"Provenance game_index mismatch for {file_path}: "
                        f"expected {game_index}, found {record.game_index}"
                    )
                existing = game_to_provenance.get(game_line)
                if existing is None:
                    game_to_provenance[game_line] = record
                elif (
                    existing.move_codes != record.move_codes
                    or existing.policy_train_mask != record.policy_train_mask
                ):
                    raise ValueError(
                        "Conflicting provenance for duplicate game line across input files: "
                        f"{game_line[:80]}..."
                    )

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
        if provenance_required:
            sidecar_path = sidecar_path_for_trmph(chunk_path)
            with open(sidecar_path, "w", encoding="utf-8") as f:
                for game_index, game_line in enumerate(chunk):
                    source_record = game_to_provenance.get(game_line)
                    if source_record is None:
                        raise ValueError(
                            f"Missing source provenance while writing cleaned sidecar for {chunk_path}"
                        )
                    record = make_move_provenance_record(
                        game_index=game_index,
                        move_codes=source_record.move_codes,
                    )
                    f.write(record.to_json_line())
                    f.write("\n")
            logger.info(
                f"Wrote chunk provenance sidecar for chunk {i}: {sidecar_path}"
            )
    
    # Write summary
    summary_path = output_dir / "processing_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Self-play data preprocessing summary\n")
        f.write(f"=====================================\n")
        f.write(f"Input directories: {[str(d) for d in normalized_input_dirs]}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Policy provenance mode: {policy_provenance_mode}\n")
        f.write(f"Total input files: {len(all_trmph_files)}\n")
        f.write(f"Total games extracted: {len(all_games)}\n")
        f.write(f"Unique games after deduplication: {len(unique_games)}\n")
        f.write(f"Duplicates removed: {len(all_games) - len(unique_games)}\n")
        f.write(f"Output chunks: {len(chunks)}\n")
        f.write(f"Games per chunk: ~{chunk_size}\n")
        if provenance_required:
            f.write(f"Provenance sidecars written: yes\n")
        f.write(f"\nInput files:\n")
        for file_path in all_trmph_files:
            f.write(f"  {file_path}\n")
        f.write(f"\nOutput files:\n")
        for i in range(len(chunks)):
            f.write(f"  cleaned_chunk_{i:03d}.trmph\n")
            if provenance_required:
                f.write(f"  cleaned_chunk_{i:03d}.provenance.jsonl\n")
    
    logger.info(f"Processing complete! Summary written to {summary_path}")


def parse_shard_range(range_str: str, data_dir: str = None) -> tuple:
    """
    Parse a single shard range segment and return (start, end) tuple.
    
    Args:
        range_str: Single range segment in format "start-end" or "all"
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


def expand_shard_range_spec(range_str: str, data_dir: str = None) -> Optional[List[int]]:
    """
    Expand a shard range spec into explicit shard numbers.

    Supported formats:
    - "all"
    - "start-end"
    - Comma-separated non-contiguous ranges, e.g. "0-206,208-498"

    Args:
        range_str: Shard range specification string
        data_dir: Optional data directory for context in errors

    Returns:
        Sorted list of shard numbers, or None for "all"

    Raises:
        ValueError: If any segment is invalid
    """
    normalized = range_str.strip().lower()
    if normalized == "all":
        return None

    segments = [segment.strip() for segment in range_str.split(",") if segment.strip()]
    if not segments:
        raise ValueError("Shard range spec cannot be empty")

    if any(segment.lower() == "all" for segment in segments):
        if len(segments) > 1:
            raise ValueError(
                f"Invalid shard range format: {range_str}. "
                "Use either 'all' or comma-separated 'start-end' segments, not both."
            )
        return None

    expanded = []
    for segment in segments:
        start, end = parse_shard_range(segment, data_dir)
        if end is None:
            return None
        expanded.extend(range(start, end + 1))

    # Deduplicate and sort so overlapping segments behave predictably.
    return sorted(set(expanded))


def validate_shard_ranges(data_dirs: List[str], 
                         shard_ranges: List[str], 
                         context_name: str = "data",
                         logger=None) -> None:
    """
    Validate shard ranges for data directories, handling "None" ranges by skipping directories.
    
    Args:
        data_dirs: List of data directory paths
        shard_ranges: List of shard range strings (e.g., ["251-300", "0-206,208-498", "all", "None"])
        context_name: Context name for logging (e.g., "training", "validation")
        logger: Optional logger instance
        
    Raises:
        RuntimeError: If validation fails or no data files found
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    
    for i, (data_dir, shard_range) in enumerate(zip(data_dirs, shard_ranges)):
        try:
            # Skip directories with "None" shard range
            if shard_range.lower() == "none":
                logger.info(f"{context_name.title()} directory {i+1}: Skipping {data_dir} (range: {shard_range})")
                continue
            
            # Parse shard range for this directory
            shard_numbers = expand_shard_range_spec(shard_range, data_dir)
            
            # Discover files in this directory using shard-based approach
            process_context = f"{context_name} data validation"
            
            if shard_numbers is None:  # 'all' case - get all files
                data_files = discover_training_data_files_all(data_dir, process_context=process_context)
            else:
                # Use explicit shard-based approach
                data_files = discover_training_data_files_by_shards(data_dir, shard_numbers, process_context=process_context)
            
            if not data_files:
                raise RuntimeError(f"No {context_name} data files found in {data_dir} with range {shard_range}")
            
            logger.info(f"{context_name.title()} directory {i+1}: Found {len(data_files)} shards in {data_dir} (range: {shard_range})")
            
        except Exception as e:
            message = str(e)
            if "Missing expected shards" in message:
                message = (
                    f"{message} "
                    "Tip: if your dataset has intentional gaps, use a comma-separated range "
                    '(for example, "0-206,208-498"), or use "all" to load every available shard.'
                )
            logger.error(f"Failed to validate {context_name} data in {data_dir}: {message}")
            raise RuntimeError(f"Failed to validate {context_name} data in {data_dir}: {message}")
