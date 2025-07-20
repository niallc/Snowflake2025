#!/usr/bin/env python3
"""
Process all .trmph files into sharded .pkl.gz files with network-ready data.

This script:
1. Finds all .trmph files in the data directory
2. Processes them into training examples using extract_training_examples_from_game
3. Saves results in sharded .pkl.gz files with provenance tracking
4. Handles errors gracefully and provides progress tracking
5. Can be resumed if interrupted
"""

import os
import sys
import glob
import pickle
import gzip
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime

# Add the hex_ai directory to the path
sys.path.append('hex_ai')

from hex_ai.data_utils import load_trmph_file, extract_training_examples_from_game
from hex_ai.utils.format_conversion import parse_trmph_game_record

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trmph_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrmphProcessor:
    def __init__(self, data_dir: str = "data", output_dir: str = "processed_data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'files_processed': 0,
            'files_failed': 0,
            'all_games': 0,           # Total games attempted across all files
            'valid_games': 0,         # Successfully processed games
            'skipped_games': 0,       # Games that couldn't be processed
            'total_examples': 0,      # Total training examples generated
            'start_time': time.time()
        }
        
        # Find all trmph files
        self.trmph_files = list(self.data_dir.rglob("*.trmph"))
        logger.info(f"Found {len(self.trmph_files)} .trmph files to process")
        
    def process_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single .trmph file and return statistics."""
        file_stats = {
            'file_path': str(file_path),
            'all_games': 0,           # Total games attempted (including invalid ones)
            'valid_games': 0,         # Successfully processed games
            'skipped_games': 0,       # Games that couldn't be processed (format errors, etc.)
            'examples_generated': 0,  # Total training examples created
            'file_error': None        # File-level error (if any)
        }
        
        try:
            logger.info(f"Processing {file_path}")
            
            # Load the trmph file
            try:
                games = load_trmph_file(str(file_path))
                logger.info(f"  Loaded {len(games)} games from {file_path}")
            except (FileNotFoundError, ValueError) as e:
                # File-level error - can't process this file at all
                file_stats['file_error'] = str(e)
                logger.error(f"File error processing {file_path}: {e}")
                return file_stats
            
            # Process each game
            all_examples = []
            for i, game_line in enumerate(games):
                file_stats['all_games'] += 1
                
                try:
                    # Parse the game record
                    try:
                        trmph_url, winner = parse_trmph_game_record(game_line)
                    except ValueError as e:
                        logger.warning(f"    Game {i+1} has wrong format: {repr(game_line)}: {e}")
                        file_stats['skipped_games'] += 1
                        continue
                    
                    # Extract training examples
                    try:
                        examples = extract_training_examples_from_game(trmph_url, winner)
                        if examples:
                            all_examples.extend(examples)
                            file_stats['valid_games'] += 1
                            file_stats['examples_generated'] += len(examples)
                        else:
                            logger.warning(f"    Game {i+1} in {file_path.name} produced no examples")
                            file_stats['skipped_games'] += 1
                    except Exception as e:
                        logger.warning(f"    Error extracting examples from game {i+1} in {file_path.name}: {e}")
                        file_stats['skipped_games'] += 1
                        
                except Exception as e:
                    # Catch any other unexpected errors during game processing
                    logger.warning(f"    Unexpected error processing game {i+1} in {file_path.name}: {e}")
                    file_stats['skipped_games'] += 1
            
            # Save processed examples
            if all_examples:
                output_file = self.output_dir / f"{file_path.stem}_processed.pkl.gz"
                try:
                    with gzip.open(output_file, 'wb') as f:
                        pickle.dump({
                            'examples': all_examples,
                            'source_file': str(file_path),
                            'processing_stats': file_stats,
                            'processed_at': datetime.now().isoformat()
                        }, f)
                    logger.info(f"  Saved {len(all_examples)} examples to {output_file}")
                except Exception as e:
                    logger.error(f"    Error saving output file {output_file}: {e}")
                    file_stats['file_error'] = f"Failed to save output: {e}"
            else:
                logger.info(f"  No valid examples generated from {file_path}")
            
            return file_stats
            
        except Exception as e:
            # Catch any other unexpected file-level errors
            file_stats['file_error'] = str(e)
            logger.error(f"Unexpected error processing {file_path}: {e}")
            return file_stats
    
    def process_all_files(self, max_files: Optional[int] = None) -> Dict[str, Any]:
        """Process all trmph files and return overall statistics."""
        files_to_process = self.trmph_files[:max_files] if max_files else self.trmph_files
        
        logger.info(f"Starting processing of {len(files_to_process)} files")
        
        for i, file_path in enumerate(files_to_process):
            logger.info(f"Progress: {i+1}/{len(files_to_process)} ({((i+1)/len(files_to_process)*100):.1f}%)")
            
            file_stats = self.process_single_file(file_path)
            
            # Update overall statistics
            self.stats['files_processed'] += 1
            if file_stats['file_error']:
                self.stats['files_failed'] += 1
            
            self.stats['all_games'] += file_stats['all_games']
            self.stats['valid_games'] += file_stats['valid_games']
            self.stats['skipped_games'] += file_stats['skipped_games']
            self.stats['total_examples'] += file_stats['examples_generated']
            
            # Log progress every 10 files
            if (i + 1) % 10 == 0:
                elapsed = time.time() - self.stats['start_time']
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info(f"  Processed {i+1} files, {self.stats['total_examples']} examples, "
                          f"rate: {rate:.2f} files/sec")
        
        # Final statistics
        elapsed = time.time() - self.stats['start_time']
        self.stats['elapsed_time'] = elapsed
        self.stats['files_per_second'] = len(files_to_process) / elapsed if elapsed > 0 else 0
        
        logger.info("Processing complete!")
        logger.info(f"  Files processed: {self.stats['files_processed']}")
        logger.info(f"  Files failed: {self.stats['files_failed']}")
        logger.info(f"  All games attempted: {self.stats['all_games']}")
        logger.info(f"  Valid games: {self.stats['valid_games']}")
        logger.info(f"  Skipped games: {self.stats['skipped_games']}")
        logger.info(f"  Success rate: {self.stats['valid_games']/self.stats['all_games']*100:.1f}%" if self.stats['all_games'] > 0 else "Success rate: N/A")
        logger.info(f"  Total examples: {self.stats['total_examples']}")
        logger.info(f"  Elapsed time: {elapsed:.1f} seconds")
        logger.info(f"  Rate: {self.stats['files_per_second']:.2f} files/sec")
        
        return self.stats
    
    def create_combined_dataset(self) -> None:
        """Combine all processed files into a single dataset."""
        logger.info("Creating combined dataset...")
        
        all_examples = []
        processed_files = list(self.output_dir.glob("*_processed.pkl.gz"))
        
        for file_path in processed_files:
            try:
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    all_examples.extend(data['examples'])
                    logger.info(f"  Loaded {len(data['examples'])} examples from {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        if all_examples:
            # Create combined dataset
            combined_file = self.output_dir / "combined_dataset.pkl.gz"
            with gzip.open(combined_file, 'wb') as f:
                pickle.dump({
                    'examples': all_examples,
                    'total_examples': len(all_examples),
                    'source_files': len(processed_files),
                    'created_at': datetime.now().isoformat()
                }, f)
            
            logger.info(f"Created combined dataset with {len(all_examples)} examples")
            logger.info(f"Saved to {combined_file}")
        else:
            logger.warning("No examples found to combine")

def main():
    """Main processing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process all .trmph files into training data")
    parser.add_argument("--data-dir", default="data", help="Directory containing .trmph files")
    parser.add_argument("--output-dir", default="processed_data", help="Output directory for processed files")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process (for testing)")
    parser.add_argument("--combine", action="store_true", help="Create combined dataset after processing")
    
    args = parser.parse_args()
    
    # Create processor
    processor = TrmphProcessor(data_dir=args.data_dir, output_dir=args.output_dir)
    
    # Process files
    stats = processor.process_all_files(max_files=args.max_files)
    
    # Create combined dataset if requested
    if args.combine:
        processor.create_combined_dataset()
    
    # Save final statistics
    stats_file = Path(args.output_dir) / "processing_stats.json"
    import json
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Final statistics saved to {stats_file}")

if __name__ == "__main__":
    main() 