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

import sys
import logging
from pathlib import Path

# Add the hex_ai directory to the path
sys.path.append('hex_ai')

from hex_ai.batch_processor import BatchProcessor
from hex_ai.file_utils import GracefulShutdown

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


def main():
    """Main processing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process all .trmph files into training data")
    parser.add_argument("--data-dir", default="data", help="Directory containing .trmph files")
    parser.add_argument("--output-dir", default="processed_data", help="Output directory for processed files")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process (for testing)")
    parser.add_argument("--combine", action="store_true", help="Create combined dataset after processing")
    parser.add_argument("--run-tag", help="Tag for this processing run (default: timestamp)")
    
    args = parser.parse_args()
    
    # Create shutdown handler
    shutdown_handler = GracefulShutdown()
    
    # Create processor
    processor = BatchProcessor(
        data_dir=args.data_dir, 
        output_dir=args.output_dir, 
        shutdown_handler=shutdown_handler,
        run_tag=args.run_tag
    )
    
    # Process files
    stats = processor.process_all_files(max_files=args.max_files)
    
    # Save final progress report
    from hex_ai.file_utils import save_progress_report
    save_progress_report(stats, Path(args.output_dir), shutdown_handler)
    
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