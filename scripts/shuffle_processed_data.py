#!/usr/bin/env python3
"""
Shuffle processed data to address value head fingerprinting issues.

This script implements a two-phase shuffling process:
1. Distribute games across buckets to break game-level correlations
2. Consolidate and shuffle each bucket to create final shuffled dataset

The process handles memory constraints by processing data in manageable chunks
and ensures games are properly distributed across the final shuffled files.
"""

import sys
import os
import logging
import json
import random
import gzip
import pickle
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

# Add the hex_ai directory to the path
sys.path.append('hex_ai')

from hex_ai.file_utils import GracefulShutdown, atomic_write_pickle_gz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_shuffling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataShuffler:
    """Handles the two-phase data shuffling process."""
    
    def __init__(self, 
                 input_dir: str = "data/processed/data",
                 output_dir: str = "data/processed/shuffled",
                 temp_dir: str = "data/processed/temp_buckets",
                 num_buckets: int = 500,
                 resume_enabled: bool = True,
                 cleanup_temp: bool = True,
                 validation_enabled: bool = True):
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.num_buckets = num_buckets
        self.resume_enabled = resume_enabled
        self.cleanup_temp = cleanup_temp
        self.validation_enabled = validation_enabled
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'total_examples': 0,
            'buckets_completed': 0,
            'start_time': time.time()
        }
        
        # Progress tracking
        self.progress_file = self.output_dir / "shuffling_progress.json"
        self.progress = self._load_progress()
        
        # Shutdown handler
        self.shutdown_handler = GracefulShutdown()
    
    def _load_progress(self) -> Dict[str, Any]:
        """Load existing progress if available."""
        if not self.progress_file.exists() or not self.resume_enabled:
            return self._create_new_progress()
        
        try:
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
            
            logger.info("Resuming from previous run")
            logger.info(f"Files processed: {len(progress.get('processed_files', []))}")
            logger.info(f"Buckets completed: {len(progress.get('completed_buckets', []))}")
            
            return progress
        except Exception as e:
            logger.warning(f"Failed to load progress file: {e}")
            return self._create_new_progress()
    
    def _create_new_progress(self) -> Dict[str, Any]:
        """Create new progress tracking structure."""
        return {
            'started_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'processed_files': [],
            'completed_buckets': [],
            'current_phase': 'distribution',
            'stats': self.stats.copy()
        }
    
    def _save_progress(self):
        """Save current progress."""
        self.progress.update({
            'last_updated': datetime.now().isoformat(),
            'stats': self.stats.copy()
        })
        
        try:
            temp_progress_file = self.progress_file.with_suffix('.tmp')
            with open(temp_progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
            temp_progress_file.rename(self.progress_file)
        except Exception as e:
            logger.error(f"Failed to save progress file: {e}")
    
    def _load_pkl_gz(self, file_path: Path) -> Dict[str, Any]:
        """Load data from a .pkl.gz file."""
        try:
            with gzip.open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise
    
    def _write_bucket_file(self, bucket_idx: int, examples: List[Dict], source_files: List[str]):
        """Write examples to a bucket file."""
        # Use input filename in bucket filename to avoid overwriting
        input_filename = Path(source_files[0]).stem  # Remove .pkl.gz extension
        bucket_file = self.temp_dir / f"{input_filename}_bucket_{bucket_idx:04d}.pkl.gz"
        
        bucket_data = {
            'examples': examples,
            'bucket_id': bucket_idx,
            'source_files': source_files,
            'created_at': datetime.now().isoformat(),
            'num_examples': len(examples)
        }
        
        try:
            atomic_write_pickle_gz(bucket_data, bucket_file)
            logger.debug(f"Written bucket {bucket_idx} from {input_filename} with {len(examples)} examples")
        except Exception as e:
            logger.error(f"Failed to write bucket {bucket_idx} from {input_filename}: {e}")
            raise
    
    def _write_shuffled_file(self, bucket_idx: int, examples: List[Dict], source_files: List[str]):
        """Write shuffled examples to final output file."""
        shuffled_file = self.output_dir / f"shuffled_{bucket_idx:04d}.pkl.gz"
        
        shuffled_data = {
            'examples': examples,
            'shuffling_stats': {
                'num_buckets': self.num_buckets,
                'bucket_id': bucket_idx,
                'total_examples': len(examples),
                'shuffled_at': datetime.now().isoformat(),
                'source_files': source_files
            }
        }
        
        try:
            atomic_write_pickle_gz(shuffled_data, shuffled_file)
            logger.info(f"Written shuffled file {bucket_idx} with {len(examples)} examples")
        except Exception as e:
            logger.error(f"Failed to write shuffled file {bucket_idx}: {e}")
            raise
    
    def _distribute_to_buckets(self, input_files: List[Path]):
        """Phase 1: Distribute games from input files across buckets."""
        logger.info(f"Starting Phase 1: Distribution to {self.num_buckets} buckets")
        
        for file_idx, input_file in enumerate(input_files):
            if self.shutdown_handler.shutdown_requested:
                logger.info("Shutdown requested, stopping distribution")
                break
            
            if str(input_file) in self.progress['processed_files']:
                logger.info(f"Skipping already processed file: {input_file.name}")
                continue
            
            logger.info(f"Processing file {file_idx + 1}/{len(input_files)}: {input_file.name}")
            
            try:
                data = self._load_pkl_gz(input_file)
                examples = data['examples']
                
                # Collect examples by bucket for THIS input file only
                bucket_examples = [[] for _ in range(self.num_buckets)]
                
                # Distribute examples directly across buckets
                # This breaks game-level correlations: with 500 buckets and ≤169 moves per game,
                # each bucket contains at most one position from any given game.
                # During training, the trainer will process ~200k records before seeing another
                # position from the same game, making fingerprinting extremely difficult.
                # See write_ups/data_shuffling_specification.md for detailed explanation.
                for example_idx, example in enumerate(examples):
                    bucket_idx = example_idx % self.num_buckets
                    bucket_examples[bucket_idx].append(example)
                
                # Write bucket files for THIS input file immediately
                for bucket_idx, examples_for_bucket in enumerate(bucket_examples):
                    if examples_for_bucket:  # Only write if we have examples for this bucket
                        self._write_bucket_file(bucket_idx, examples_for_bucket, [str(input_file)])
                
                # Update progress
                self.progress['processed_files'].append(str(input_file))
                self.stats['files_processed'] += 1
                self.stats['total_examples'] += len(examples)
                self._save_progress()
                
            except Exception as e:
                logger.error(f"Error processing {input_file}: {e}")
                raise  # Make file processing errors fatal
        
        logger.info("Phase 1 completed")
    
    def _consolidate_and_shuffle_bucket(self, bucket_idx: int):
        """Phase 2: Consolidate and shuffle a single bucket.
        
        This method:
        1. Loads all bucket files for this bucket index from different input files
        2. Concatenates all examples into a single list
        3. Shuffles the examples to break any remaining correlations
        4. Writes the final shuffled file
        5. Optionally cleans up the temporary bucket files
        """
        logger.info(f"Processing bucket {bucket_idx}")
        
        # Find all bucket files for this bucket index
        bucket_pattern = f"*_bucket_{bucket_idx:04d}.pkl.gz"
        bucket_files = list(self.temp_dir.glob(bucket_pattern))
        
        if not bucket_files:
            logger.warning(f"No files found for bucket {bucket_idx}")
            return
        
        # Load and consolidate examples from all bucket files
        all_examples = []
        source_files = []
        
        for bucket_file in bucket_files:
            try:
                data = self._load_pkl_gz(bucket_file)
                examples = data['examples']
                file_source_files = data.get('source_files', [])
                
                all_examples.extend(examples)
                source_files.extend(file_source_files)
                    
            except Exception as e:
                logger.error(f"Error loading {bucket_file}: {e}")
                raise  # Make file processing errors fatal
        
        if not all_examples:
            logger.warning(f"No examples found for bucket {bucket_idx}")
            return
        
        # Shuffle examples to break any remaining correlations
        # This is the final randomization step that addresses value head fingerprinting
        logger.info(f"Shuffling {len(all_examples)} examples in bucket {bucket_idx}")
        random.shuffle(all_examples)
        
        # Write final shuffled file with complete source tracking
        self._write_shuffled_file(bucket_idx, all_examples, source_files)
        
        # Update progress tracking
        self.progress['completed_buckets'].append(bucket_idx)
        self.stats['buckets_completed'] += 1
        self._save_progress()
        
        # Clean up temporary bucket files if requested
        if self.cleanup_temp:
            for bucket_file in bucket_files:
                try:
                    bucket_file.unlink()
                except Exception as e:
                    logger.error(f"Failed to delete {bucket_file}: {e}")
                    raise  # Make cleanup errors fatal
    
    def _consolidate_and_shuffle_all_buckets(self):
        """Phase 2: Consolidate and shuffle all buckets."""
        logger.info(f"Starting Phase 2: Consolidation and shuffling of {self.num_buckets} buckets")
        
        for bucket_idx in range(self.num_buckets):
            if self.shutdown_handler.shutdown_requested:
                logger.info("Shutdown requested, stopping consolidation")
                break
            
            if bucket_idx in self.progress['completed_buckets']:
                logger.info(f"Skipping already completed bucket: {bucket_idx}")
                continue
            
            self._consolidate_and_shuffle_bucket(bucket_idx)
        
        logger.info("Phase 2 completed")
    
    def _validate_output(self):
        """Validate the shuffled output data."""
        if not self.validation_enabled:
            return
        
        logger.info("Validating shuffled output...")
        
        shuffled_files = list(self.output_dir.glob("shuffled_*.pkl.gz"))
        total_examples = 0
        bucket_sizes = []
        
        for shuffled_file in shuffled_files:
            try:
                data = self._load_pkl_gz(shuffled_file)
                examples = data['examples']
                total_examples += len(examples)
                bucket_sizes.append(len(examples))
                
            except Exception as e:
                logger.error(f"Error validating {shuffled_file}: {e}")
        
        # Report validation results
        logger.info(f"Validation complete:")
        logger.info(f"  Total shuffled files: {len(shuffled_files)}")
        logger.info(f"  Total examples: {total_examples}")
        
        # Check for even distribution across buckets
        if bucket_sizes:
            max_bucket_size = max(bucket_sizes)
            min_bucket_size = min(bucket_sizes)
            avg_bucket_size = sum(bucket_sizes) / len(bucket_sizes)
            
            logger.info(f"  Max bucket size: {max_bucket_size}")
            logger.info(f"  Min bucket size: {min_bucket_size}")
            logger.info(f"  Avg bucket size: {avg_bucket_size:.2f}")
            
            # Check if distribution is reasonably even (within 50% of average)
            if max_bucket_size > avg_bucket_size * 1.5:
                logger.warning("Uneven bucket distribution detected - some buckets may be significantly larger than others")
    
    def shuffle_data(self):
        """Main method to run the complete shuffling process."""
        logger.info("Starting data shuffling process")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Number of buckets: {self.num_buckets}")
        
        # Find input files
        input_files = list(self.input_dir.glob("*.pkl.gz"))
        if not input_files:
            logger.error(f"No .pkl.gz files found in {self.input_dir}")
            return
        
        logger.info(f"Found {len(input_files)} input files")
        
        try:
            # Phase 1: Distribution
            if self.progress['current_phase'] == 'distribution':
                self._distribute_to_buckets(input_files)
                self.progress['current_phase'] = 'consolidation'
                self._save_progress()
            
            # Phase 2: Consolidation and shuffling
            if self.progress['current_phase'] == 'consolidation':
                self._consolidate_and_shuffle_all_buckets()
                self.progress['current_phase'] = 'completed'
                self._save_progress()
            
            # Validation
            if self.progress['current_phase'] == 'completed':
                self._validate_output()
            
            # Final statistics
            elapsed_time = time.time() - self.stats['start_time']
            logger.info("=" * 60)
            logger.info("SHUFFLING COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"Files processed: {self.stats['files_processed']}")
            logger.info(f"Total examples: {self.stats['total_examples']}")
            logger.info(f"Buckets completed: {self.stats['buckets_completed']}")
            logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
            logger.info(f"Output files: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error during shuffling process: {e}")
            raise


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Shuffle processed data to address value head fingerprinting")
    parser.add_argument("--input-dir", default="data/processed/data", 
                       help="Directory containing processed .pkl.gz files")
    parser.add_argument("--output-dir", default="data/processed/shuffled", 
                       help="Output directory for shuffled files")
    parser.add_argument("--temp-dir", default="data/processed/temp_buckets", 
                       help="Temporary directory for bucket files")
    parser.add_argument("--num-buckets", type=int, default=500, 
                       help="Number of buckets for distribution")
    parser.add_argument("--no-resume", action="store_true", 
                       help="Disable resume functionality")
    parser.add_argument("--no-cleanup", action="store_true", 
                       help="Keep temporary bucket files")
    parser.add_argument("--no-validation", action="store_true", 
                       help="Skip output validation")
    
    args = parser.parse_args()
    
    # Create shuffler
    shuffler = DataShuffler(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        temp_dir=args.temp_dir,
        num_buckets=args.num_buckets,
        resume_enabled=not args.no_resume,
        cleanup_temp=not args.no_cleanup,
        validation_enabled=not args.no_validation
    )
    
    # Run shuffling process
    shuffler.shuffle_data()


if __name__ == "__main__":
    main() 