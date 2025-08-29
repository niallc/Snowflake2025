#!/usr/bin/env python3
"""
Script to clean up TRMPH files by removing games with duplicate moves and duplicate games.

This script processes all .trmph files in data directories and removes:
1. Lines that contain duplicate moves anywhere in the game sequence
2. Duplicate game lines (entire lines that are identical)

The script can handle both types of duplicates and can be run in dry-run mode to just report findings.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Set

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hex_ai.utils.format_conversion import trmph_to_moves, strip_trmph_preamble, split_trmph_moves, trmph_move_to_rowcol
from hex_ai.config import BOARD_SIZE

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def has_duplicate_moves(trmph_line: str) -> bool:
    """
    Check if a TRMPH line contains duplicate moves anywhere in the game.
    
    Args:
        trmph_line: A single TRMPH line from a file
        
    Returns:
        True if the line contains duplicate moves, False otherwise
    """
    try:
        # Extract moves from TRMPH line
        if not trmph_line.startswith('http://www.trmph.com/hex/board#13,'):
            return False
            
        # Parse TRMPH line
        parts = trmph_line.split('#13,')
        if len(parts) != 2:
            return False
            
        moves_winner = parts[1]
        
        # Find the winner indicator (b or r) at the end
        if moves_winner.endswith(' b'):
            moves_str = moves_winner[:-2]
        elif moves_winner.endswith(' r'):
            moves_str = moves_winner[:-2]
        else:
            # No winner indicator, use the whole string
            moves_str = moves_winner
            
        # Convert TRMPH moves to row,col coordinates
        moves = trmph_to_moves(f"#13,{moves_str}")
        
        # Check for duplicates
        unique_moves = set(moves)
        return len(unique_moves) != len(moves)
        
    except Exception as e:
        logger.warning(f"Error parsing TRMPH line: {e}")
        return False

def clean_trmph_file(input_path: str, output_path: str = None, dry_run: bool = False, 
                    remove_duplicate_moves: bool = True, remove_duplicate_games: bool = True) -> Tuple[int, int, int, int]:
    """
    Clean a TRMPH file by removing lines with duplicate moves and/or duplicate games.
    
    Args:
        input_path: Path to input TRMPH file
        output_path: Path to output file (if None, overwrites input file)
        dry_run: If True, only report findings without modifying files
        remove_duplicate_moves: Whether to remove lines with duplicate moves
        remove_duplicate_games: Whether to remove duplicate game lines
        
    Returns:
        Tuple of (original_lines, kept_lines, duplicate_moves_removed, duplicate_games_removed)
    """
    if output_path is None and not dry_run:
        output_path = input_path
        
    original_lines = 0
    kept_lines = 0
    duplicate_moves_removed = 0
    duplicate_games_removed = 0
    
    # Read all lines first to count them
    with open(input_path, 'r') as f:
        lines = f.readlines()
        original_lines = len(lines)
    
    # Strip whitespace for processing
    stripped_lines = [line.rstrip('\n\r') for line in lines]
    
    # Track seen games for duplicate detection
    seen_games = set()
    
    # Process lines
    if dry_run:
        # Just count duplicates without writing
        for line_num, line in enumerate(stripped_lines, 1):
            should_keep = True
            
            # Check for duplicate moves
            if remove_duplicate_moves and has_duplicate_moves(line):
                logger.debug(f"Would skip line {line_num} with duplicate moves: {line[:50]}...")
                duplicate_moves_removed += 1
                should_keep = False
            
            # Check for duplicate games (entire line)
            if should_keep and remove_duplicate_games and line in seen_games:
                logger.debug(f"Would skip line {line_num} with duplicate game: {line[:50]}...")
                duplicate_games_removed += 1
                should_keep = False
            
            if should_keep:
                seen_games.add(line)
                kept_lines += 1
    else:
        # Write clean ones to output file
        with open(output_path, 'w') as f:
            for line_num, line in enumerate(stripped_lines, 1):
                should_keep = True
                
                # Check for duplicate moves
                if remove_duplicate_moves and has_duplicate_moves(line):
                    logger.debug(f"Skipping line {line_num} with duplicate moves: {line[:50]}...")
                    duplicate_moves_removed += 1
                    should_keep = False
                
                # Check for duplicate games (entire line)
                if should_keep and remove_duplicate_games and line in seen_games:
                    logger.debug(f"Skipping line {line_num} with duplicate game: {line[:50]}...")
                    duplicate_games_removed += 1
                    should_keep = False
                
                if should_keep:
                    seen_games.add(line)
                    f.write(line + '\n')
                    kept_lines += 1
                
    if dry_run:
        logger.info(f"Checked {input_path}: {original_lines} lines, would remove {duplicate_moves_removed} duplicate moves, {duplicate_games_removed} duplicate games")
    else:
        logger.info(f"Processed {input_path}: {original_lines} -> {kept_lines} lines (removed {duplicate_moves_removed} duplicate moves, {duplicate_games_removed} duplicate games)")
    
    return original_lines, kept_lines, duplicate_moves_removed, duplicate_games_removed

def check_directory_for_duplicates(directory_path: str, dry_run: bool = True, 
                                 remove_duplicate_moves: bool = True, 
                                 remove_duplicate_games: bool = True) -> Tuple[int, int, int, int]:
    """
    Check all TRMPH files in a directory for duplicate moves and duplicate games.
    
    Args:
        directory_path: Path to directory to check
        dry_run: If True, only report findings without modifying files
        remove_duplicate_moves: Whether to remove lines with duplicate moves
        remove_duplicate_games: Whether to remove duplicate game lines
        
    Returns:
        Tuple of (total_files, total_lines, total_duplicate_moves, total_duplicate_games)
    """
    dir_path = Path(directory_path)
    
    if not dir_path.exists():
        logger.warning(f"Directory {directory_path} does not exist")
        return 0, 0, 0, 0
    
    # Find all .trmph files recursively
    trmph_files = list(dir_path.rglob("*.trmph"))
    
    if not trmph_files:
        logger.info(f"No .trmph files found in {directory_path}")
        return 0, 0, 0, 0
    
    logger.info(f"Found {len(trmph_files)} TRMPH files to check in {directory_path}")
    
    total_original = 0
    total_kept = 0
    total_duplicate_moves = 0
    total_duplicate_games = 0
    
    # Process each file
    for file_path in trmph_files:
        try:
            original, kept, duplicate_moves, duplicate_games = clean_trmph_file(
                str(file_path), 
                dry_run=dry_run,
                remove_duplicate_moves=remove_duplicate_moves,
                remove_duplicate_games=remove_duplicate_games
            )
            total_original += original
            total_kept += kept
            total_duplicate_moves += duplicate_moves
            total_duplicate_games += duplicate_games
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    total_removed = total_duplicate_moves + total_duplicate_games
    
    if total_removed > 0:
        logger.info(f"Summary for {directory_path}: {len(trmph_files)} files, {total_original} lines")
        logger.info(f"  - Duplicate moves removed: {total_duplicate_moves}")
        logger.info(f"  - Duplicate games removed: {total_duplicate_games}")
        logger.info(f"  - Total removed: {total_removed} ({total_removed/total_original*100:.2f}%)")
    else:
        logger.info(f"Summary for {directory_path}: {len(trmph_files)} files, {total_original} lines, no duplicates found")
    
    return len(trmph_files), total_original, total_duplicate_moves, total_duplicate_games

def main():
    """Main function to check all TRMPH files in multiple directories."""
    directories_to_check = [
        "data/twoNetGames",
        "data/sf25", 
        "data/tournament_play"
    ]
    
    # Configuration
    dry_run = True  # Set to False to actually clean the files
    remove_duplicate_moves = True
    remove_duplicate_games = True
    
    total_files = 0
    total_lines = 0
    total_duplicate_moves = 0
    total_duplicate_games = 0
    
    for directory in directories_to_check:
        logger.info(f"\n{'='*60}")
        logger.info(f"Checking directory: {directory}")
        logger.info(f"{'='*60}")
        
        files, lines, duplicate_moves, duplicate_games = check_directory_for_duplicates(
            directory, 
            dry_run=dry_run,
            remove_duplicate_moves=remove_duplicate_moves,
            remove_duplicate_games=remove_duplicate_games
        )
        total_files += files
        total_lines += lines
        total_duplicate_moves += duplicate_moves
        total_duplicate_games += duplicate_games
    
    total_removed = total_duplicate_moves + total_duplicate_games
    
    logger.info(f"\n{'='*60}")
    logger.info(f"OVERALL SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total files checked: {total_files}")
    logger.info(f"Total lines: {total_lines}")
    logger.info(f"Duplicate moves found: {total_duplicate_moves}")
    logger.info(f"Duplicate games found: {total_duplicate_games}")
    logger.info(f"Total duplicates found: {total_removed}")
    if total_lines > 0:
        logger.info(f"Overall duplicate rate: {total_removed/total_lines*100:.2f}%")
    
    if dry_run:
        logger.info(f"\nThis was a dry run. Set dry_run=False to actually clean the files.")

if __name__ == "__main__":
    main()
