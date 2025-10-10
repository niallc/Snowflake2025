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

from hex_ai.data_pipeline import DataShuffler, DEFAULT_NUM_BUCKETS, BUCKET_ID_FORMAT_WIDTH

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


# DataShuffler class is now imported from hex_ai.data_pipeline


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