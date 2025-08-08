#!/usr/bin/env python3
"""
Safe script to fix winner labels in .trmph files.

This script processes all .trmph files in data/sf25/jul29/ and corrects the winner labels
by using the game engine's winner detection function instead of the incorrect 'b' labels.

SAFE VERSION: Writes corrected files to a new directory instead of modifying originals.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hex_ai.inference.game_engine import HexGameState
from hex_ai.utils.format_conversion import parse_trmph_game_record
from hex_ai.config import TRMPH_BLUE_WIN, TRMPH_RED_WIN

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def determine_winner_from_trmph(trmph_string: str) -> Optional[str]:
    """
    Determine the winner of a game from its TRMPH string.
    
    Args:
        trmph_string: The TRMPH game string (e.g., "#13,a4g7e9e8f8f7h7h6j5")
        
    Returns:
        "blue" if blue wins, "red" if red wins, None if game is incomplete
    """
    try:
        # Create game state from TRMPH string
        game_state = HexGameState.from_trmph(trmph_string)
        
        # Check if game is over and get winner
        if game_state.game_over:
            return game_state.winner
        else:
            # Game is incomplete - no winner yet
            return None
            
    except Exception as e:
        logger.error(f"Error processing TRMPH string '{trmph_string}': {e}")
        return None


def convert_winner_to_trmph_format(winner: str) -> str:
    """
    Convert winner string to TRMPH format.
    
    Args:
        winner: "blue" or "red"
        
    Returns:
        TRMPH winner indicator ("b" or "r")
    """
    if winner == "blue":
        return TRMPH_BLUE_WIN
    elif winner == "red":
        return TRMPH_RED_WIN
    else:
        raise ValueError(f"Invalid winner: {winner}")


def process_trmph_file(input_file_path: Path, output_file_path: Path, dry_run: bool = False) -> Tuple[int, int, int]:
    """
    Process a single .trmph file and fix winner labels.
    
    Args:
        input_file_path: Path to the input .trmph file
        output_file_path: Path where the corrected file should be written
        dry_run: If True, don't write changes, just report what would be changed
        
    Returns:
        Tuple of (total_games, fixed_games, error_games)
    """
    logger.info(f"Processing file: {input_file_path}")
    
    total_games = 0
    fixed_games = 0
    error_games = 0
    
    # Read all lines from the input file
    with open(input_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Process each line
    new_lines = []
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        logger.debug(f"Line {line_num}: Processing '{line}'")
        
        # Skip empty lines and comments (but not TRMPH game lines that start with #13,)
        if not line or (line.startswith('#') and not line.startswith('#13,')):
            logger.debug(f"Line {line_num}: Skipping comment/empty line")
            new_lines.append(line)
            continue
        
        # Parse the game record
        try:
            trmph_string, current_winner = parse_trmph_game_record(line)
            total_games += 1
            logger.debug(f"Line {line_num}: Parsed game with current winner '{current_winner}': {trmph_string}")
            
            # Determine the actual winner
            actual_winner = determine_winner_from_trmph(trmph_string)
            logger.debug(f"Line {line_num}: Determined actual winner: {actual_winner}")
            
            if actual_winner is None:
                logger.warning(f"Line {line_num}: Game incomplete, skipping: {trmph_string}")
                new_lines.append(line)  # Keep original line
                error_games += 1
                continue
            
            # Convert to TRMPH format
            correct_winner = convert_winner_to_trmph_format(actual_winner)
            logger.debug(f"Line {line_num}: Converted to TRMPH format: {correct_winner}")
            
            # Check if correction is needed
            if current_winner != correct_winner:
                logger.info(f"Line {line_num}: Fixing winner from '{current_winner}' to '{correct_winner}' for game: {trmph_string}")
                fixed_games += 1
                new_lines.append(f"{trmph_string} {correct_winner}")
            else:
                logger.debug(f"Line {line_num}: No correction needed, winner already correct")
                new_lines.append(line)  # No change needed
                
        except Exception as e:
            logger.error(f"Line {line_num}: Error processing line '{line}': {e}")
            new_lines.append(line)  # Keep original line
            error_games += 1
    
    # Write the corrected file to the output location
    if not dry_run:
        # Ensure output directory exists
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines) + '\n')
        logger.info(f"Updated file: {output_file_path}")
    else:
        logger.info(f"DRY RUN: Would update file: {output_file_path}")
    
    return total_games, fixed_games, error_games


def main():
    """Main function to process all .trmph files in the target directory."""
    # Set up paths
    input_data_dir = Path("data/sf25/jul29")
    output_data_dir = Path("data/sf25/jul29_winner_fixed")
    
    if not input_data_dir.exists():
        logger.error(f"Input data directory does not exist: {input_data_dir}")
        sys.exit(1)
    
    # Find all .trmph files
    trmph_files = list(input_data_dir.glob("*.trmph"))
    
    if not trmph_files:
        logger.error(f"No .trmph files found in {input_data_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(trmph_files)} .trmph files to process")
    logger.info(f"Output directory: {output_data_dir}")
    
    # Check for dry run flag
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        logger.info("DRY RUN MODE: No files will be modified")
    
    # Process each file
    total_files = len(trmph_files)
    total_games_processed = 0
    total_games_fixed = 0
    total_games_error = 0
    
    for i, input_file_path in enumerate(trmph_files, 1):
        logger.info(f"Processing file {i}/{total_files}: {input_file_path.name}")
        
        # Create output file path
        output_file_path = output_data_dir / input_file_path.name
        
        try:
            games, fixed, errors = process_trmph_file(input_file_path, output_file_path, dry_run=dry_run)
            total_games_processed += games
            total_games_fixed += fixed
            total_games_error += errors
            
        except Exception as e:
            logger.error(f"Error processing file {input_file_path}: {e}")
            total_games_error += 1
    
    # Summary
    logger.info("=" * 60)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Files processed: {total_files}")
    logger.info(f"Total games: {total_games_processed}")
    logger.info(f"Games fixed: {total_games_fixed}")
    logger.info(f"Games with errors: {total_games_error}")
    
    if dry_run:
        logger.info("DRY RUN COMPLETED - No files were modified")
    else:
        logger.info(f"All corrected files have been written to: {output_data_dir}")
        logger.info("Original files in data/sf25/jul29/ remain unchanged")


if __name__ == "__main__":
    main() 