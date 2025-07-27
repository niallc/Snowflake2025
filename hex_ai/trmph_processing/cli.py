"""
Command-line interface for TRMPH processing.

This module provides the CLI functionality for processing TRMPH files,
separating the command-line logic from the core processing logic.
"""

import sys
import logging
import argparse
from pathlib import Path

# Import hex_ai modules
from hex_ai.system_utils import check_virtual_env
from hex_ai.batch_processor import BatchProcessor
from hex_ai.file_utils import GracefulShutdown

# Import our processing modules
from .config import ProcessingConfig
from .processor import TRMPHProcessor

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging for the CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/trmph_processing.log'),
            logging.StreamHandler()
        ]
    )


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process all .trmph files into training data")
    parser.add_argument("--data-dir", default="data", help="Directory containing .trmph files")
    parser.add_argument("--output-dir", default="data/processed/step1_unshuffled", help="Output directory for processed files")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process (for testing)")
    parser.add_argument("--combine", action="store_true", help="Create combined dataset after processing")
    parser.add_argument("--run-tag", help="Tag for this processing run (default: timestamp)")
    parser.add_argument("--position-selector", default="all", choices=["all", "final", "penultimate"], help="Which positions to extract from each game: all, final, or penultimate")
    parser.add_argument("--max-workers", type=int, default=6, help="Number of worker processes to use (default: 6)")
    parser.add_argument("--sequential", action="store_true", help="Process files sequentially (for debugging)")
    
    return parser.parse_args()


def create_config_from_args(args):
    """Create ProcessingConfig from parsed arguments."""
    return ProcessingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_files=args.max_files,
        position_selector=args.position_selector,
        run_tag=args.run_tag,
        max_workers=1 if args.sequential else args.max_workers,
        combine_output=args.combine
    )


def process_files(config):
    """Process TRMPH files using the given configuration."""
    # Create processor and process files
    processor = TRMPHProcessor(config)
    results = processor.process_all_files()
    return results


def create_combined_dataset(config):
    """Create combined dataset if requested."""
    logger.info("Creating combined dataset...")
    shutdown_handler = GracefulShutdown()
    
    batch_processor = BatchProcessor(
        data_dir=config.data_dir,
        output_dir=config.output_dir,
        shutdown_handler=shutdown_handler,
        run_tag=config.run_tag
    )
    batch_processor.create_combined_dataset()
    logger.info("Combined dataset created successfully")


def print_output_summary(config, results):
    """Print summary of output files and results."""
    logger.info("")
    logger.info("OUTPUT FILES:")
    logger.info(f"  Processing statistics: {Path(config.output_dir) / 'processing_stats.json'}")
    logger.info(f"  Log file: logs/trmph_processing.log")
    
    # Count processed files
    processed_files = list(Path(config.output_dir).glob("*_processed.pkl.gz"))
    logger.info(f"  Individual processed files: {len(processed_files)} files in {config.output_dir}/")
    
    if config.combine_output:
        combined_file = Path(config.output_dir) / "combined_dataset.pkl.gz"
        if combined_file.exists():
            logger.info(f"  Combined dataset: {combined_file}")
        else:
            logger.warning(f"  Combined dataset: NOT CREATED (check logs/trmph_processing.log for errors)")
    
    logger.info("")


def main():
    """Main CLI entry point."""
    # Check virtual environment
    check_virtual_env("hex_ai_env")
    
    # Setup logging
    setup_logging()
    
    # Parse arguments
    args = parse_arguments()
    
    # Create configuration
    config = create_config_from_args(args)
    logger.info(f"Configuration: {config}")
    
    # Create shutdown handler
    shutdown_handler = GracefulShutdown()
    
    try:
        # Process files
        results = process_files(config)
        
        # Handle combined dataset creation if requested
        if args.combine:
            create_combined_dataset(config)
        
        # Print output summary
        print_output_summary(config, results)
        
        # Final status report
        if shutdown_handler.shutdown_requested:
            logger.info("Processing completed with graceful shutdown")
        else:
            logger.info("Processing completed successfully")
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 