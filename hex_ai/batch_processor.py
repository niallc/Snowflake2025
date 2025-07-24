"""
Batch processing utilities for trmph files.

This module provides:
- BatchProcessor class for processing multiple trmph files
- Data validation functions
- Combined dataset creation
- Resume functionality for interrupted runs
"""

import time
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

from .data_utils import load_trmph_file
from .utils.format_conversion import parse_trmph_game_record
from .file_utils import (
    GracefulShutdown, atomic_write_pickle_gz, validate_output_directory,
    sanitize_filename, save_progress_report
)

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Process multiple trmph files in batch with progress tracking and error handling."""
    
    VERSION = "1.1"  # Current version of the processor
    
    def __init__(self, data_dir: str = "data", output_dir: str = "processed_data", 
                 shutdown_handler: Optional[GracefulShutdown] = None, run_tag: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.shutdown_handler = shutdown_handler
        self.run_tag = run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Validate output directory
        validate_output_directory(self.output_dir, self.data_dir)
        
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
        
        # State file for resume functionality
        self.state_file = self.output_dir / "processing_state.json"
        self.state = self._load_state()
        
        # Check for existing processing and handle resume
        self._handle_resume()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load existing processing state if available."""
        if not self.state_file.exists():
            return self._create_new_state()
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            # Validate state file
            if not self._validate_state(state):
                logger.warning("Invalid state file found - creating new state")
                return self._create_new_state()
            
            return state
        except Exception as e:
            logger.warning(f"Failed to load state file: {e} - creating new state")
            return self._create_new_state()
    
    def _create_new_state(self) -> Dict[str, Any]:
        """Create a new processing state."""
        return {
            "version": self.VERSION,
            "run_tag": self.run_tag,
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_files": len(self.trmph_files),
            "processed_files": [],
            "failed_files": [],
            "current_file": None,
            "current_file_started": None,
            "stats": self.stats.copy()
        }
    
    def _validate_state(self, state: Dict[str, Any]) -> bool:
        """Validate state file structure and content."""
        required_keys = ["version", "run_tag", "started_at", "total_files", "processed_files", "failed_files"]
        if not all(key in state for key in required_keys):
            return False
        
        # Check if version is compatible (for future version handling)
        if state.get("version") != self.VERSION:
            logger.info(f"State file version {state.get('version')} differs from current version {self.VERSION}")
            # For now, we'll accept different versions, but log it
        
        return True
    
    def _save_state(self):
        """Save current processing state."""
        self.state.update({
            "last_updated": datetime.now().isoformat(),
            "stats": self.stats.copy()
        })
        
        try:
            # Atomic write for state file
            temp_state_file = self.state_file.with_suffix('.tmp')
            with open(temp_state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
            temp_state_file.rename(self.state_file)
        except Exception as e:
            logger.error(f"Failed to save state file: {e}")
    
    def _detect_partial_files(self) -> Set[str]:
        """Detect and clean up partial files (files with .tmp extension)."""
        partial_files = set()
        
        # Look for .tmp files that aren't actively being written
        for tmp_file in self.output_dir.glob("*.tmp"):
            try:
                # Check if file is older than 5 minutes (likely abandoned)
                if time.time() - tmp_file.stat().st_mtime > 300:
                    logger.info(f"Found abandoned partial file: {tmp_file}")
                    tmp_file.unlink()
                    partial_files.add(str(tmp_file))
            except Exception as e:
                logger.warning(f"Error handling partial file {tmp_file}: {e}")
        
        return partial_files
    
    def _handle_resume(self):
        """Handle resume logic when existing processing is detected."""
        if not self.state["processed_files"] and not self.state["failed_files"]:
            # No previous processing
            return
        
        # Detect and clean partial files
        partial_files = self._detect_partial_files()
        
        # Show resume information
        logger.info("=" * 60)
        logger.info("RESUME DETECTED")
        logger.info("=" * 60)
        logger.info(f"Previous run started: {self.state['started_at']}")
        logger.info(f"Run tag: {self.state['run_tag']}")
        logger.info(f"Files completed: {len(self.state['processed_files'])}")
        logger.info(f"Files failed: {len(self.state['failed_files'])}")
        logger.info(f"Total files: {self.state['total_files']}")
        
        # Show more detailed information about existing files
        if self.state["processed_files"]:
            logger.info("")
            logger.info("COMPLETED FILES:")
            for i, file_info in enumerate(self.state["processed_files"][:5]):  # Show first 5
                file_path = Path(file_info["file"])
                output_path = Path(file_info["output"]) if file_info["output"] else "No output"
                logger.info(f"  {i+1}. {file_path.name}")
                logger.info(f"     Output: {output_path.name if output_path != 'No output' else 'No output'}")
                logger.info(f"     Completed: {file_info['completed_at']}")
            if len(self.state["processed_files"]) > 5:
                logger.info(f"  ... and {len(self.state['processed_files']) - 5} more completed files")
        
        if self.state["failed_files"]:
            logger.info("")
            logger.info("FAILED FILES:")
            for i, file_info in enumerate(self.state["failed_files"][:5]):  # Show first 5
                file_path = Path(file_info["file"])
                logger.info(f"  {i+1}. {file_path.name}")
                logger.info(f"     Error: {file_info['error'][:100]}{'...' if len(file_info['error']) > 100 else ''}")
                logger.info(f"     Attempted: {file_info['attempted_at']}")
            if len(self.state["failed_files"]) > 5:
                logger.info(f"  ... and {len(self.state['failed_files']) - 5} more failed files")
        
        logger.info("")
        logger.info(f"Output directory: {self.output_dir.absolute()}")
        
        if partial_files:
            logger.info(f"Cleaned up {len(partial_files)} partial files")
        
        if self.state["current_file"]:
            logger.info(f"Currently processing: {self.state['current_file']}")
        
        # Get user input for resume strategy
        resume_strategy = self._get_resume_strategy()
        
        if resume_strategy == "abandon":
            logger.info("Processing abandoned by user")
            raise KeyboardInterrupt("Processing abandoned by user")
        
        elif resume_strategy == "skip_completed":
            # Skip already completed and failed files
            completed_files = {p["file"] for p in self.state["processed_files"]}
            failed_files = {p["file"] for p in self.state["failed_files"]}
            skip_files = completed_files | failed_files
            
            # Normalize paths for comparison (remove data/ prefix if present)
            normalized_skip_files = set()
            for skip_file in skip_files:
                # Remove data/ prefix if present
                if skip_file.startswith("data/"):
                    normalized_skip_files.add(skip_file[5:])  # Remove "data/"
                else:
                    normalized_skip_files.add(skip_file)
            
            # Debug logging (commented out for production)
            # logger.debug(f"Completed files in state: {completed_files}")
            # logger.debug(f"Failed files in state: {failed_files}")
            # logger.debug(f"Normalized skip files: {normalized_skip_files}")
            # logger.debug(f"Original files: {[f.name for f in self.trmph_files]}")
            
            self.trmph_files = [f for f in self.trmph_files if f.name not in normalized_skip_files]
            logger.info(f"Resuming with {len(self.trmph_files)} remaining files")
            
        elif resume_strategy == "retry_failed":
            # Retry only failed files
            failed_files = {p["file"] for p in self.state["failed_files"]}
            self.trmph_files = [f for f in self.trmph_files if str(f) in failed_files]
            logger.info(f"Retrying {len(self.trmph_files)} failed files")
            
        elif resume_strategy == "fresh_start":
            # Start fresh - clear state and output
            logger.info("Starting fresh - clearing existing output")
            self._clear_existing_output()
            self.state = self._create_new_state()
            self._save_state()  # Save the new state
            
        else:  # skip_current
            # Skip current file and continue from next
            if self.state["current_file"]:
                current_idx = next((i for i, f in enumerate(self.trmph_files) 
                                  if str(f) == self.state["current_file"]), -1)
                if current_idx >= 0:
                    self.trmph_files = self.trmph_files[current_idx + 1:]
                    logger.info(f"Resuming from next file: {len(self.trmph_files)} files remaining")
    
    def _get_resume_strategy(self) -> str:
        """Get user input for resume strategy."""
        print("\nResume options:")
        print("1. Skip completed files and continue")
        print("2. Retry failed files only")
        print("3. Skip current file and continue from next")
        print("4. Start fresh (delete all existing output)")
        print("5. Abandon processing")
        
        while True:
            try:
                choice = input("\nEnter choice (1-5): ").strip()
                if choice == "1":
                    return "skip_completed"
                elif choice == "2":
                    return "retry_failed"
                elif choice == "3":
                    return "skip_current"
                elif choice == "4":
                    return "fresh_start"
                elif choice == "5":
                    return "abandon"
                else:
                    print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
            except KeyboardInterrupt:
                logger.info("Resume cancelled by user")
                raise
    
    def _clear_existing_output(self):
        """Clear existing processed files."""
        for file_path in self.output_dir.glob("*_processed.pkl.gz"):
            try:
                file_path.unlink()
                logger.info(f"Removed existing file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {e}")
        
        # Remove state file
        if self.state_file.exists():
            self.state_file.unlink()
    
    def _mark_file_started(self, file_path: Path):
        """Mark a file as started processing."""
        self.state["current_file"] = str(file_path)
        self.state["current_file_started"] = datetime.now().isoformat()
        self._save_state()
    
    def _mark_file_completed(self, file_path: Path, output_file: Path, file_stats: Dict[str, Any]):
        """Mark a file as completed."""
        self.state["processed_files"].append({
            "file": str(file_path),
            "output": str(output_file),
            "completed_at": datetime.now().isoformat(),
            "stats": file_stats
        })
        self.state["current_file"] = None
        self.state["current_file_started"] = None
        self._save_state()
    
    def _mark_file_failed(self, file_path: Path, error: str):
        """Mark a file as failed."""
        self.state["failed_files"].append({
            "file": str(file_path),
            "error": error,
            "attempted_at": datetime.now().isoformat()
        })
        self.state["current_file"] = None
        self.state["current_file_started"] = None
        self._save_state()
    
    def process_single_file(self, file_path: Path, file_idx: int, position_selector: str = "all") -> Dict[str, Any]:
        """Process a single .trmph file and return statistics."""
        file_stats = {
            'file_path': str(file_path),
            'all_games': 0,           # Total games attempted (including invalid ones)
            'valid_games': 0,         # Successfully processed games
            'skipped_games': 0,       # Games that couldn't be processed (format errors, etc.)
            'examples_generated': 0,  # Total training examples created
            'file_error': None        # File-level error (if any)
        }
        
        # Mark file as started
        self._mark_file_started(file_path)
        
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
                self._mark_file_failed(file_path, str(e))
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
                        # Create game_id with file_idx and line_idx (i+1 for 1-based line numbers)
                        game_id = (file_idx, i+1)
                        from .data_utils import extract_training_examples_with_selector_from_game
                        examples = extract_training_examples_with_selector_from_game(trmph_url, winner, game_id, position_selector=position_selector)
                        if examples:
                            # Use dictionary format directly - no conversion needed
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
                safe_filename = sanitize_filename(file_path.stem)
                output_file = self.output_dir / f"{safe_filename}_processed.pkl.gz"
                
                # Ensure filename uniqueness
                counter = 1
                while output_file.exists():
                    output_file = self.output_dir / f"{safe_filename}_processed_{counter}.pkl.gz"
                    counter += 1
                    if counter > 4:  # Prevent infinite loop
                        raise ValueError(f"Too many files with similar name: {safe_filename}")
                
                # Validate data before saving
                self._validate_examples_data(all_examples)
                
                try:
                    # Save with atomic write
                    data = {
                        'examples': all_examples,
                        'source_file': str(file_path),
                        'processing_stats': file_stats,
                        'processed_at': datetime.now().isoformat(),
                        'file_size_bytes': 0  # Will be updated after write
                    }
                    
                    atomic_write_pickle_gz(data, output_file)
                    
                    # Get file size for logging
                    file_size = output_file.stat().st_size
                    logger.info(f"  Saved {len(all_examples)} examples to {output_file} ({file_size} bytes)")
                    
                    # Mark file as completed
                    self._mark_file_completed(file_path, output_file, file_stats)
                    
                except Exception as e:
                    logger.error(f"    Error saving output file {output_file}: {e}")
                    file_stats['file_error'] = f"Failed to save output: {e}"
                    self._mark_file_failed(file_path, f"Failed to save output: {e}")
            else:
                logger.info(f"  No valid examples generated from {file_path}")
                # Mark as completed even if no examples (file was processed successfully)
                self._mark_file_completed(file_path, Path(""), file_stats)
            
            return file_stats
            
        except Exception as e:
            # Catch any other unexpected file-level errors
            file_stats['file_error'] = str(e)
            logger.error(f"Unexpected error processing {file_path}: {e}")
            self._mark_file_failed(file_path, str(e))
            return file_stats
    
    def process_all_files(self, max_files: Optional[int] = None, position_selector: str = "all") -> Dict[str, Any]:
        """Process all trmph files and return overall statistics."""
        files_to_process = self.trmph_files[:max_files] if max_files else self.trmph_files
        
        logger.info(f"Starting processing of {len(files_to_process)} files")
        
        for i, file_path in enumerate(files_to_process):
            # Check for shutdown request
            if self.shutdown_handler and self.shutdown_handler.shutdown_requested:
                logger.info("Shutdown requested - saving progress and exiting gracefully")
                self.current_file_index = i
                save_progress_report(
                    self.stats, self.output_dir, self.shutdown_handler,
                    i, len(files_to_process), str(file_path)
                )
                break
            
            self.current_file_index = i
            logger.info(f"Progress: {i+1}/{len(files_to_process)} ({((i+1)/len(files_to_process)*100):.1f}%)")
            
            file_stats = self.process_single_file(file_path, file_idx=i, position_selector=position_selector)
            
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
                import gzip
                import pickle
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
            data = {
                'examples': all_examples,
                'total_examples': len(all_examples),
                'source_files': len(processed_files) - len(failed_files),
                'failed_files': failed_files,
                'created_at': datetime.now().isoformat()
            }
            
            atomic_write_pickle_gz(data, combined_file)
            
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
        if len(examples) > 5000000:  # 5M examples per file
            raise ValueError(f"Too many examples ({len(examples)}) for single file")
        
        # Validate each example has required structure
        for i, example in enumerate(examples):
            if not isinstance(example, dict):
                raise ValueError(f"Example {i} is not a dictionary")
            
            # Check for required dictionary keys (board, policy, value)
            if not all(key in example for key in ['board', 'policy', 'value']):
                raise ValueError(f"Example {i} missing required keys: {example.keys()}")
            
            # Check that elements are numpy arrays or appropriate types
            import numpy as np
            if not isinstance(example['board'], np.ndarray):
                raise ValueError(f"Example {i} board state must be numpy array")
            if example['policy'] is not None and not isinstance(example['policy'], np.ndarray):
                raise ValueError(f"Example {i} policy target must be numpy array or None")
            if not isinstance(example['value'], (int, float, np.number)):
                raise ValueError(f"Example {i} value target must be numeric")
        
        # Check total data size (rough estimate)
        import sys
        estimated_size = sys.getsizeof(examples)
        for example in examples[:100]:  # Sample first 100
            estimated_size += sys.getsizeof(example)
        
        # Extrapolate to full size
        if len(examples) > 10000:
            estimated_size = estimated_size * len(examples) // 10
        
        if estimated_size >  1024 * 1024:  # 1GB limit
            logger.warning(f"Large dataset detected: ~{estimated_size // (1024*1024)}MB") 