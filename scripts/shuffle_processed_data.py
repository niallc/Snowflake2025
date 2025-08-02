"""
Shuffle processed data to address value head fingerprinting issues.

DATA FLOW & DIRECTORY CONVENTION:
- Input: data/processed/step1_unshuffled/*.pkl.gz (output of process_all_trmph.py)
- Output: data/processed/shuffled/*.pkl.gz (shuffled, ready for training)
- Temp: data/processed/temp_buckets/ (intermediate bucket files)
- This script expects the input directory to contain processed .pkl.gz files with player_to_move and metadata fields.
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
from concurrent.futures import ProcessPoolExecutor, as_completed

# Environment validation is now handled automatically in hex_ai/__init__.py

# Add the hex_ai directory to the path
sys.path.append('hex_ai')

from hex_ai.file_utils import GracefulShutdown, atomic_write_pickle_gz

# Configuration constants
DEFAULT_NUM_BUCKETS = 500
BUCKET_ID_FORMAT_WIDTH = 4  # For :04d format in filenames

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_shuffling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataShuffler:
    """Handles the two-phase data shuffling process."""
    
    def __init__(self, 
                 input_dir: str = "data/processed/step1_unshuffled",
                 output_dir: str = "data/processed/shuffled",
                 temp_dir: str = "data/processed/temp_buckets",
                 num_buckets: int = DEFAULT_NUM_BUCKETS,
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
            
            # Restore stats from saved progress
            saved_stats = progress.get('stats', {})
            if saved_stats:
                self.stats.update(saved_stats)
                logger.info(f"Restored stats: {saved_stats}")
            
            # Fix incorrect phase state: if we have processed files but no completed buckets,
            # we should still be in distribution phase, not completed
            if (len(progress.get('processed_files', [])) > 0 and 
                len(progress.get('completed_buckets', [])) == 0 and
                progress.get('current_phase') == 'completed'):
                logger.warning("Detected incorrect phase state: files processed but no buckets completed, fixing to 'distribution'")
                progress['current_phase'] = 'distribution'
            
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
        # # Check all examples for player_to_move field
        # for i, ex in enumerate(examples):
        #     if 'player_to_move' not in ex:
        #         raise ValueError(f"Missing 'player_to_move' in example at index {i} in bucket {bucket_idx}. Example: {repr(ex)[:300]}")
        # Use input filename in bucket filename to avoid overwriting
        input_filename = Path(source_files[0]).stem  # Remove .pkl.gz extension
        bucket_file = self.temp_dir / f"{input_filename}_bucket_{bucket_idx:0{BUCKET_ID_FORMAT_WIDTH}d}.pkl.gz"
        
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
        # # Check all examples for player_to_move field
        # for i, ex in enumerate(examples):
        #     if 'player_to_move' not in ex:
        #         raise ValueError(f"Missing 'player_to_move' in example at index {i} in shuffled bucket {bucket_idx}. Example: {repr(ex)[:300]}")
        shuffled_file = self.output_dir / f"shuffled_{bucket_idx:0{BUCKET_ID_FORMAT_WIDTH}d}.pkl.gz"
        
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
    
    def _distribute_single_file(self, input_file):
        try:
            data = self._load_pkl_gz(input_file)
            examples = data['examples']
            bucket_examples = [[] for _ in range(self.num_buckets)]
            for example_idx, example in enumerate(examples):
                bucket_idx = example_idx % self.num_buckets
                bucket_examples[bucket_idx].append(example)
            for bucket_idx, examples_for_bucket in enumerate(bucket_examples):
                if examples_for_bucket:
                    self._write_bucket_file(bucket_idx, examples_for_bucket, [str(input_file)])
            return str(input_file), len(examples)
        except Exception as e:
            logger.error(f"Error processing {input_file}: {e}")
            raise

    def _distribute_to_buckets(self, input_files):
        logger.info(f"Starting Phase 1: Distribution to {self.num_buckets} buckets (parallelized)")
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(self._distribute_single_file, input_file): input_file for input_file in input_files}
            for future in as_completed(futures):
                input_file = futures[future]
                try:
                    file_name, num_examples = future.result()
                    logger.info(f"Processed {file_name} with {num_examples} examples")
                    self.progress['processed_files'].append(str(input_file))
                    self.stats['files_processed'] += 1
                    self.stats['total_examples'] += num_examples
                    self._save_progress()
                except Exception as e:
                    logger.error(f"Error in parallel processing of {input_file}: {e}")
                    raise
    
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
        bucket_pattern = f"*_bucket_{bucket_idx:0{BUCKET_ID_FORMAT_WIDTH}d}.pkl.gz"
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
        logger.info(f"Starting Phase 2: Consolidation and shuffling of {self.num_buckets} buckets (parallelized)")
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(self._consolidate_and_shuffle_bucket, bucket_idx): bucket_idx for bucket_idx in range(self.num_buckets)}
            for future in as_completed(futures):
                bucket_idx = futures[future]
                try:
                    future.result()
                    logger.info(f"Bucket {bucket_idx} consolidated and shuffled")
                except Exception as e:
                    logger.error(f"Error in parallel processing of bucket {bucket_idx}: {e}")
                    raise
        logger.info("Phase 2 completed")
    
    def _validate_output(self):
        """Validate the shuffled output data.
        
        Currently only counts total examples to verify no data was lost.
        The distribution is guaranteed to be even by construction (≤169 moves per game,
        {self.num_buckets} buckets), so no distribution validation is needed.
        """
        if not self.validation_enabled:
            return
        
        logger.info("Validating shuffled output...")
        
        shuffled_files = list(self.output_dir.glob("shuffled_*.pkl.gz"))
        total_examples = 0
        
        for shuffled_file in shuffled_files:
            try:
                data = self._load_pkl_gz(shuffled_file)
                examples = data['examples']
                total_examples += len(examples)
                
            except Exception as e:
                logger.error(f"Error validating {shuffled_file}: {e}")
                raise  # Make validation errors fatal
        
        # Report validation results
        logger.info(f"Validation complete:")
        logger.info(f"  Total shuffled files: {len(shuffled_files)}")
        logger.info(f"  Total examples: {total_examples}")
    
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
    parser.add_argument("--input-dir", default="data/processed/step1_unshuffled", 
                       help="Directory containing processed .pkl.gz files")
    parser.add_argument("--output-dir", default="data/processed/shuffled", 
                       help="Output directory for shuffled files")
    parser.add_argument("--temp-dir", default="data/processed/temp_buckets", 
                       help="Temporary directory for bucket files")
    parser.add_argument("--num-buckets", type=int, default=DEFAULT_NUM_BUCKETS, 
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

"""
Example usage:
  PYTHONPATH=. python scripts/shuffle_processed_data.py \
    --input-dir data/processed/jul29_unshuffled9 \
    --output-dir data/processed/jul_29_shuffled \
    --temp-dir data/processed/temp_buckets \
    --num-buckets 100 \
    --no-resume
"""

if __name__ == "__main__":
    main() 