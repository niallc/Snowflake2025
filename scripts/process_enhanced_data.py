#!/usr/bin/env python3
"""
Enhanced data processing script for Hex AI.

This script reprocesses TRMPH files into the new enhanced format with:
- Comprehensive metadata tracking
- Flexible value sampling tiers  
- Memory-efficient stratified processing
- Game correlation breaking
- Repeated moves handling
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hex_ai.enhanced_data_processing import (
    create_stratified_dataset, create_chunked_dataset
)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler('enhanced_processing.log'),
            logging.StreamHandler()
        ]
    )


def find_trmph_files(data_dir: Path) -> List[Path]:
    """Find all TRMPH files in the data directory."""
    trmph_files = list(data_dir.rglob("*.trmph"))
    trmph_files.sort()
    
    if not trmph_files:
        raise ValueError(f"No TRMPH files found in {data_dir}")
    
    return trmph_files


def main():
    parser = argparse.ArgumentParser(description="Enhanced TRMPH data processing")
    
    # Input/output arguments
    parser.add_argument('--data_dir', type=str, default="data",
                       help='Directory containing TRMPH files')
    parser.add_argument('--output_dir', type=str, default="data/enhanced_processed",
                       help='Directory to save processed files')
    
    # Processing strategy
    parser.add_argument('--strategy', choices=['stratified', 'chunked'], 
                       default='stratified',
                       help='Processing strategy to use')
    
    # Stratified processing options
    parser.add_argument('--positions_per_pass', type=int, default=5,
                       help='Number of positions to process per pass (stratified only)')
    parser.add_argument('--max_positions_per_game', type=int, default=200,
                       help='Maximum positions to consider per game')
    
    # Chunked processing options  
    parser.add_argument('--games_per_chunk', type=int, default=10000,
                       help='Number of games per chunk (chunked only)')
    
    # General options
    parser.add_argument('--include_trmph', action='store_true',
                       help='Include full TRMPH strings in metadata')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate paths
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    output_dir = Path(args.output_dir)
    
    # Find TRMPH files
    logger.info(f"Searching for TRMPH files in {data_dir}")
    trmph_files = find_trmph_files(data_dir)
    
    if args.max_files:
        trmph_files = trmph_files[:args.max_files]
        logger.info(f"Limited to {len(trmph_files)} files for testing")
    
    logger.info(f"Found {len(trmph_files)} TRMPH files")
    
    # Display processing configuration
    logger.info("Processing configuration:")
    logger.info(f"  Strategy: {args.strategy}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Include TRMPH strings: {args.include_trmph}")
    
    if args.strategy == 'stratified':
        logger.info(f"  Positions per pass: {args.positions_per_pass}")
        logger.info(f"  Max positions per game: {args.max_positions_per_game}")
    else:
        logger.info(f"  Games per chunk: {args.games_per_chunk}")
    
    # Start processing
    start_time = time.time()
    
    try:
        if args.strategy == 'stratified':
            processed_files, lookup_file = create_stratified_dataset(
                trmph_files=trmph_files,
                output_dir=output_dir,
                positions_per_pass=args.positions_per_pass,
                include_trmph=args.include_trmph,
                max_positions_per_game=args.max_positions_per_game
            )
        else:  # chunked
            processed_files, lookup_file = create_chunked_dataset(
                trmph_files=trmph_files,
                output_dir=output_dir,
                games_per_chunk=args.games_per_chunk,
                include_trmph=args.include_trmph
            )
        
        # Processing complete
        elapsed_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        logger.info(f"Processed files: {len(processed_files)}")
        logger.info(f"Lookup table: {lookup_file}")
        logger.info(f"Output directory: {output_dir}")
        
        # Display file sizes
        total_size = sum(f.stat().st_size for f in processed_files)
        logger.info(f"Total output size: {total_size / (1024**3):.2f} GB")
        
        # Display first few files
        logger.info("Generated files:")
        for i, file_path in enumerate(processed_files[:5]):
            size_mb = file_path.stat().st_size / (1024**2)
            logger.info(f"  {file_path.name} ({size_mb:.1f} MB)")
        
        if len(processed_files) > 5:
            logger.info(f"  ... and {len(processed_files) - 5} more files")
        
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main() 