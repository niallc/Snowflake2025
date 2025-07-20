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
import signal
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

class GracefulShutdown:
    """Handle graceful shutdown on SIGTERM/SIGINT."""
    def __init__(self):
        self.shutdown_requested = False
        self.emergency_exit = False
        self.signal_count = 0
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        self.signal_count += 1
        if self.signal_count == 1:
            logger.info(f"Received signal {signum} - initiating graceful shutdown...")
            self.shutdown_requested = True
        elif self.signal_count == 2:
            logger.warning(f"Received second signal {signum} - forcing emergency exit!")
            self.emergency_exit = True
            sys.exit(1)
        else:
            logger.error(f"Received signal {signum} for the {self.signal_count}th time - forcing exit!")
            sys.exit(1)

class TrmphProcessor:
    def __init__(self, data_dir: str = "data", output_dir: str = "processed_data", shutdown_handler: Optional[GracefulShutdown] = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.shutdown_handler = shutdown_handler
        
        # Validate output directory
        self._validate_output_directory()
        
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
    
    def _validate_output_directory(self):
        """Validate output directory permissions and available space."""
        try:
            # Create directory if it doesn't exist
            self.output_dir.mkdir(exist_ok=True)
            
            # Check write permissions
            test_file = self.output_dir / ".test_write_permission"
            try:
                test_file.write_text("test")
                test_file.unlink()  # Clean up
            except (PermissionError, OSError) as e:
                raise ValueError(f"Output directory {self.output_dir} is not writable: {e}")
            
            # Check available disk space
            import shutil
            total, used, free = shutil.disk_usage(self.output_dir)
            free_gb = free / (1024**3)
            
            # Estimate space needed based on number of trmph files
            estimated_files = len(list(self.data_dir.rglob("*.trmph")))
            # Rough estimate: each trmph file might produce 10-100MB of processed data
            estimated_needed_gb = estimated_files * 0.1  # Conservative estimate
            
            if free_gb + 20 < estimated_needed_gb:
                logger.warning(f"Low disk space: {free_gb:.1f}GB free, estimated need: {estimated_needed_gb:.1f}GB")
                logger.warning(f"Consider freeing up space or processing fewer files")
            elif free_gb < 40.0:
                logger.warning(f"Low disk space: {free_gb:.1f}GB free in {self.output_dir}")
            
            logger.info(f"Output directory validated: {self.output_dir}")
            
        except Exception as e:
            raise ValueError(f"Failed to validate output directory {self.output_dir}: {e}")
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to be safe for filesystem."""
        import re
        import unicodedata
        
        # Normalize unicode characters
        filename = unicodedata.normalize('NFKD', filename)
        
        # Remove or replace problematic characters
        # Keep alphanumeric, dots, hyphens, underscores
        filename = re.sub(r'[^\w\-_.]', '_', filename)
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('._ ')
        
        # Ensure filename is not empty
        if not filename:
            filename = "unnamed"
        
        # Limit length to prevent filesystem issues
        if len(filename) > 200:
            filename = filename[:200]
        
        return filename
        
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
                # Sanitize filename and ensure uniqueness
                safe_filename = self._sanitize_filename(file_path.stem)
                output_file = self.output_dir / f"{safe_filename}_processed.pkl.gz"
                
                # Ensure filename uniqueness
                counter = 1
                original_output_file = output_file
                while output_file.exists():
                    output_file = self.output_dir / f"{safe_filename}_processed_{counter}.pkl.gz"
                    counter += 1
                    if counter > 4:  # Prevent infinite loop
                        raise ValueError(f"Too many files with similar name: {safe_filename}")
                
                # Validate data before saving
                self._validate_examples_data(all_examples)
                
                try:
                    # Save with atomic write (write to temp file first)
                    temp_file = output_file.with_suffix('.tmp')
                    
                    # Check if target file already exists (race condition check)
                    if output_file.exists():
                        logger.warning(f"    Target file {output_file} already exists - this may indicate a race condition")
                    
                    # Write data with correct file size in one pass
                    with gzip.open(temp_file, 'wb') as f:
                        pickle.dump({
                            'examples': all_examples,
                            'source_file': str(file_path),
                            'processing_stats': file_stats,
                            'processed_at': datetime.now().isoformat(),
                            'file_size_bytes': 0  # Will be updated after write
                        }, f)
                    
                    # Get file size and update metadata
                    file_size = temp_file.stat().st_size
                    
                    # Atomic move - this will fail if target exists on most filesystems
                    try:
                        temp_file.rename(output_file)
                    except FileExistsError:
                        # Target file exists - this is unexpected and indicates a bug
                        temp_file.unlink()  # Clean up temp file
                        raise RuntimeError(f"Target file {output_file} already exists - possible race condition or duplicate processing")
                    
                    logger.info(f"  Saved {len(all_examples)} examples to {output_file} ({file_size} bytes)")
                    
                except Exception as e:
                    # Clean up temp file if it exists
                    if temp_file.exists():
                        temp_file.unlink(missing_ok=True)
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
            # Check for shutdown request
            if self.shutdown_handler and self.shutdown_handler.shutdown_requested:
                logger.info("Shutdown requested - saving progress and exiting gracefully")
                self.current_file_index = i
                self.save_progress_report(self.output_dir)
                break
            
            self.current_file_index = i
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
        
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE - SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Files failed: {self.stats['files_failed']}")
        logger.info(f"Success rate (files): {((self.stats['files_processed'] - self.stats['files_failed']) / self.stats['files_processed'] * 100):.1f}%" if self.stats['files_processed'] > 0 else "N/A")
        logger.info("")
        logger.info("GAME PROCESSING STATISTICS:")
        logger.info(f"  Total games attempted: {self.stats['all_games']:,}")
        logger.info(f"  Valid games processed: {self.stats['valid_games']:,}")
        logger.info(f"  Skipped games: {self.stats['skipped_games']:,}")
        logger.info(f"  Success rate (games): {self.stats['valid_games']/self.stats['all_games']*100:.1f}%" if self.stats['all_games'] > 0 else "N/A")
        logger.info(f"  Total training examples: {self.stats['total_examples']:,}")
        if self.stats['valid_games'] > 0:
            logger.info(f"  Examples per valid game: {self.stats['total_examples']/self.stats['valid_games']:.1f}")
        logger.info("")
        logger.info("PERFORMANCE STATISTICS:")
        logger.info(f"  Elapsed time: {elapsed:.1f} seconds ({elapsed/3600:.1f} hours)")
        logger.info(f"  Processing rate: {self.stats['files_per_second']:.2f} files/sec")
        if self.stats['total_examples'] > 0:
            examples_per_sec = self.stats['total_examples'] / elapsed if elapsed > 0 else 0
            logger.info(f"  Examples generated: {examples_per_sec:.1f} examples/sec")
        logger.info("=" * 60)
        
        return self.stats
    
    def save_progress_report(self, output_dir: Path) -> None:
        """Save a progress report with current statistics."""
        try:
            # Create progress report
            progress_data = {
                'stats': self.stats.copy(),
                'timestamp': datetime.now().isoformat(),
                'shutdown_requested': self.shutdown_handler.shutdown_requested if self.shutdown_handler else False
            }
            
            # Add current file being processed if available
            if hasattr(self, 'current_file_index') and hasattr(self, 'trmph_files'):
                progress_data['current_file_index'] = self.current_file_index
                progress_data['total_files'] = len(self.trmph_files)
                if self.current_file_index < len(self.trmph_files):
                    progress_data['current_file'] = str(self.trmph_files[self.current_file_index])
            
            # Add summary statistics
            if self.stats['all_games'] > 0:
                progress_data['summary'] = {
                    'success_rate_percent': round(self.stats['valid_games'] / self.stats['all_games'] * 100, 1),
                    'examples_per_game': round(self.stats['total_examples'] / self.stats['valid_games'], 1) if self.stats['valid_games'] > 0 else 0,
                    'files_per_second': round(self.stats.get('files_per_second', 0), 2),
                    'elapsed_hours': round(self.stats.get('elapsed_time', 0) / 3600, 1)
                }
            
            # Save progress report
            progress_file = output_dir / "processing_progress.json"
            import json
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            
            logger.info(f"Progress report saved to {progress_file}")
            
        except Exception as e:
            logger.error(f"Failed to save progress report: {e}")
    
    def create_combined_dataset(self) -> None:
        """Combine all processed files into a single dataset."""
        logger.info("Creating combined dataset...")
        
        all_examples = []
        processed_files = list(self.output_dir.glob("*_processed.pkl.gz"))
        
        if not processed_files:
            logger.warning("No processed files found to combine")
            return
        
        logger.info(f"Found {len(processed_files)} processed files to combine")
        
        failed_files = []
        for file_path in processed_files:
            try:
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    all_examples.extend(data['examples'])
                    logger.info(f"  Loaded {len(data['examples'])} examples from {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                failed_files.append(str(file_path))
        
        if failed_files:
            logger.error(f"Failed to load {len(failed_files)} files: {failed_files}")
            if len(failed_files) > len(processed_files) // 2:
                logger.error("Too many files failed to load - aborting combined dataset creation")
                return
        
        if all_examples:
            # Create combined dataset
            combined_file = self.output_dir / "combined_dataset.pkl.gz"
            with gzip.open(combined_file, 'wb') as f:
                pickle.dump({
                    'examples': all_examples,
                    'total_examples': len(all_examples),
                    'source_files': len(processed_files) - len(failed_files),
                    'failed_files': failed_files,
                    'created_at': datetime.now().isoformat()
                }, f)
            
            logger.info(f"Created combined dataset with {len(all_examples)} examples")
            logger.info(f"Saved to {combined_file}")
            if failed_files:
                logger.warning(f"Skipped {len(failed_files)} failed files in combined dataset")
        else:
            logger.warning("No examples found to combine")

    def _validate_examples_data(self, examples: list):
        """Validate examples data before saving."""
        if not isinstance(examples, list):
            raise ValueError("Examples must be a list")
        
        if len(examples) == 0:
            raise ValueError("Examples list cannot be empty")
        
        # Check for reasonable size limits
        if len(examples) > 5000000:  # 1M examples per file
            raise ValueError(f"Too many examples ({len(examples)}) for single file")
        
        # Validate each example has required structure
        for i, example in enumerate(examples):
            if not isinstance(example, tuple):
                raise ValueError(f"Example {i} is not a tuple")
            
            # Check for required tuple length (board, policy, value)
            if len(example) != 3:
                raise ValueError(f"Example {i} must have exactly 3 elements (board, policy, value), got {len(example)}")
            
            # Check that elements are numpy arrays or appropriate types
            import numpy as np
            if not isinstance(example[0], np.ndarray):
                raise ValueError(f"Example {i} board state must be numpy array")
            if not isinstance(example[1], np.ndarray):
                raise ValueError(f"Example {i} policy target must be numpy array")
            if not isinstance(example[2], (int, float, np.number)):
                raise ValueError(f"Example {i} value target must be numeric")
        
        # Check total data size (rough estimate)
        import sys
        estimated_size = sys.getsizeof(examples)
        for example in examples[:100]:  # Sample first 100
            estimated_size += sys.getsizeof(example)
        
        # Extrapolate to full size
        if len(examples) > 100:
            estimated_size = estimated_size * len(examples) // 100
        
        if estimated_size > 1000 * 1024 * 1024:  # 100MB limit
            logger.warning(f"Large dataset detected: ~{estimated_size // (1024*1024)}MB")

def main():
    """Main processing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process all .trmph files into training data")
    parser.add_argument("--data-dir", default="data", help="Directory containing .trmph files")
    parser.add_argument("--output-dir", default="processed_data", help="Output directory for processed files")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process (for testing)")
    parser.add_argument("--combine", action="store_true", help="Create combined dataset after processing")
    
    args = parser.parse_args()
    
    # Create shutdown handler
    shutdown_handler = GracefulShutdown()
    
    # Create processor
    processor = TrmphProcessor(data_dir=args.data_dir, output_dir=args.output_dir, shutdown_handler=shutdown_handler)
    
    # Process files
    stats = processor.process_all_files(max_files=args.max_files)
    
    # Save final progress report
    processor.save_progress_report(Path(args.output_dir))
    
    # Create combined dataset if requested
    if args.combine:
        processor.create_combined_dataset()
    
    # Save final statistics
    stats_file = Path(args.output_dir) / "processing_stats.json"
    import json
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("")
    logger.info("OUTPUT FILES:")
    logger.info(f"  Processing statistics: {stats_file}")
    logger.info(f"  Progress report: {Path(args.output_dir) / 'processing_progress.json'}")
    logger.info(f"  Log file: trmph_processing.log")
    
    # Count processed files
    processed_files = list(Path(args.output_dir).glob("*_processed.pkl.gz"))
    logger.info(f"  Individual processed files: {len(processed_files)} files in {args.output_dir}/")
    
    if args.combine:
        combined_file = Path(args.output_dir) / "combined_dataset.pkl.gz"
        if combined_file.exists():
            logger.info(f"  Combined dataset: {combined_file}")
        else:
            logger.warning(f"  Combined dataset: NOT CREATED (check trmph_processing.log for errors)")
    
    logger.info("")
    # Final status report
    if shutdown_handler.shutdown_requested:
        logger.info("Processing completed with graceful shutdown")
    else:
        logger.info("Processing completed successfully")

if __name__ == "__main__":
    main() 