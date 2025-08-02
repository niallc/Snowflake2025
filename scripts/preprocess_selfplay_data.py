#!/usr/bin/env python3
"""
Preprocess self-play data files.

This script helps clean up self-play data by:
1. Combining multiple .trmph files into a single file
2. Removing duplicate games
3. Stripping header information
4. Splitting into manageable chunks for processing

Usage:
    python scripts/preprocess_selfplay_data.py --input-dir data/sf25/jul29 --output-dir data/cleaned
"""

import argparse
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import List, Set
import hashlib

# Environment validation is now handled automatically in hex_ai/__init__.py


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/preprocess_selfplay.log'),
            logging.StreamHandler()
        ]
    )


def find_trmph_files(input_dir: Path) -> List[Path]:
    """Find all .trmph files in the input directory."""
    trmph_files = list(input_dir.glob("*.trmph"))
    logging.info(f"Found {len(trmph_files)} .trmph files in {input_dir}")
    return sorted(trmph_files)


def extract_games_from_file(file_path: Path) -> List[str]:
    """Extract game lines from a TRMPH file, skipping headers."""
    games = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Game lines start with #13, (board size + format)
            # Header lines start with just # (metadata)
            if line and line.startswith('#13,'):
                # Remove any inline comments (everything after # that's not part of the game)
                if ' # ' in line:
                    line = line.split(' # ')[0].strip()
                games.append(line)
    return games


def remove_duplicates(games: List[str]) -> List[str]:
    """Remove duplicate games while preserving order."""
    seen = set()
    unique_games = []
    
    for game in games:
        if game not in seen:
            seen.add(game)
            unique_games.append(game)
    
    logging.info(f"Removed {len(games) - len(unique_games)} duplicate games")
    return unique_games


def split_games_into_chunks(games: List[str], chunk_size: int = 20000) -> List[List[str]]:
    """Split games into chunks of specified size."""
    chunks = []
    for i in range(0, len(games), chunk_size):
        chunk = games[i:i + chunk_size]
        chunks.append(chunk)
    
    logging.info(f"Split {len(games)} games into {len(chunks)} chunks of ~{chunk_size} games each")
    return chunks


def write_chunk_to_file(chunk: List[str], output_path: Path, chunk_num: int):
    """Write a chunk of games to a file."""
    with open(output_path, 'w') as f:
        for game in chunk:
            f.write(game + '\n')
    
    logging.info(f"Wrote chunk {chunk_num} with {len(chunk)} games to {output_path}")


def combine_and_clean_files(input_dir: Path, output_dir: Path, chunk_size: int = 20000):
    """Combine all TRMPH files, remove duplicates, and split into chunks."""
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all TRMPH files
    trmph_files = find_trmph_files(input_dir)
    if not trmph_files:
        logging.error(f"No .trmph files found in {input_dir}")
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
        logging.info(f"Processing complete! Summary written to {summary_path}")
        return
    
    # Extract all games from all files
    all_games = []
    for file_path in trmph_files:
        logging.info(f"Processing {file_path}")
        games = extract_games_from_file(file_path)
        all_games.extend(games)
        logging.info(f"  Extracted {len(games)} games from {file_path.name}")
    
    logging.info(f"Total games extracted: {len(all_games)}")
    
    # Remove duplicates
    unique_games = remove_duplicates(all_games)
    
    # Split into chunks
    chunks = split_games_into_chunks(unique_games, chunk_size)
    
    # Write chunks to files
    for i, chunk in enumerate(chunks):
        chunk_filename = f"cleaned_chunk_{i:03d}.trmph"
        chunk_path = output_dir / chunk_filename
        write_chunk_to_file(chunk, chunk_path, i)
    
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
    
    logging.info(f"Processing complete! Summary written to {summary_path}")


def main():
    """Main entry point."""
    # Environment validation is now handled automatically in hex_ai/__init__.py
    
    # Setup logging
    setup_logging()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Preprocess self-play data files")
    parser.add_argument("--input-dir", required=True, help="Directory containing .trmph files")
    parser.add_argument("--output-dir", required=True, help="Output directory for cleaned files")
    parser.add_argument("--chunk-size", type=int, default=20000, help="Number of games per chunk (default: 20000)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logging.error(f"Input directory {input_dir} does not exist")
        return 1
    
    logging.info(f"Starting preprocessing of self-play data")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Chunk size: {args.chunk_size}")
    
    try:
        combine_and_clean_files(input_dir, output_dir, args.chunk_size)
        logging.info("Preprocessing completed successfully!")
        return 0
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 