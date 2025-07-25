"""
Process all .trmph files into sharded .pkl.gz files with network-ready data.

DATA FLOW & OUTPUT FORMAT:
- This script finds all .trmph files in the data directory and processes them into training examples.
- Each example includes:
    - board: (2, N, N) numpy array
    - policy: (N*N,) numpy array or None
    - value: float
    - player_to_move: int (0=Blue, 1=Red) [NEW, required for all downstream code]
    - metadata: dict with game_id, position_in_game, winner, etc.
- Output files include a 'source_file' field (the original .trmph file) and are tracked in processing_state.json.
- The game_id in each example's metadata can be mapped back to the original .trmph file using the state file and file lookup utilities in hex_ai/data_utils.py.
- The player_to_move field is critical for correct model training and inference; its absence will cause downstream failures.

"""

import sys
import logging
import os
from pathlib import Path

from hex_ai.system_utils import check_virtual_env
check_virtual_env("hex_ai_env")

# Add the hex_ai directory to the path
sys.path.append('hex_ai')

from hex_ai.batch_processor import BatchProcessor
from hex_ai.file_utils import GracefulShutdown
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    parser.add_argument("--output-dir", default="data/processed/step1_unshuffled", help="Output directory for processed files")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process (for testing)")
    parser.add_argument("--combine", action="store_true", help="Create combined dataset after processing")
    parser.add_argument("--run-tag", help="Tag for this processing run (default: timestamp)")
    parser.add_argument("--position-selector", default="all", choices=["all", "final", "penultimate"], help="Which positions to extract from each game: all, final, or penultimate")
    
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
    
    # Find all .trmph files to process
    trmph_files = processor.trmph_files
    if args.max_files:
        trmph_files = trmph_files[:args.max_files]

    # Parallel processing of files
    def process_file_wrapper(file_idx_file):
        file_idx, file_path = file_idx_file
        # Create a new BatchProcessor for each process to avoid shared state issues
        from hex_ai.batch_processor import BatchProcessor
        proc = BatchProcessor(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            shutdown_handler=None,  # Not needed in worker
            run_tag=args.run_tag
        )
        return proc.process_single_file(file_path, file_idx, position_selector=args.position_selector)

    stats_list = []
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(process_file_wrapper, (i, file_path)): file_path for i, file_path in enumerate(trmph_files)}
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                stats = future.result()
                stats_list.append(stats)
                logger.info(f"Processed {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

    # Combine stats (if needed)
    # Save final progress report
    from hex_ai.file_utils import save_progress_report
    save_progress_report(stats_list, Path(args.output_dir), shutdown_handler)
    
    if args.combine:
        processor.create_combined_dataset()
    
    # Save final statistics
    stats_file = Path(args.output_dir) / "processing_stats.json"
    import json
    with open(stats_file, 'w') as f:
        json.dump(stats_list, f, indent=2)
    
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