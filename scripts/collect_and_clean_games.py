#!/usr/bin/env python3
"""
Game collection and cleaning script for Hex AI.

This script handles the first step of the data processing pipeline:
1. Collecting .trmph game files from multiple sources
2. Cleaning and deduplicating games
3. Splitting into manageable chunks

This is the first step before the full processing pipeline that converts games to training positions.

Usage:
    # Collect from multiple sources
    python scripts/collect_and_clean_games.py collect --output-dir data/collected/sf25_20250127
    
    # Process single directory
    python scripts/collect_and_clean_games.py process --input-dir data/sf25/jul29 --output-dir data/cleaned
    
    # Use defaults
    python scripts/collect_and_clean_games.py collect  # Uses default output dir with today's date
    python scripts/collect_and_clean_games.py process --input-dir data/sf25/jul29  # Uses default output dir
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Environment validation is now handled automatically in hex_ai/__init__.py
import hex_ai
from hex_ai.data_collection import (
    collect_and_organize_data, combine_and_clean_files, collect_tournament_data_since_date,
    validate_tournament_patterns, test_tournament_patterns_with_examples
)
from hex_ai.data_config import (
    DEFAULT_SOURCE_DIRS, DEFAULT_CHUNK_SIZE,
    get_collected_dir_name, get_cleaned_dir_name
)


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/process_data.log'),
            logging.StreamHandler()
        ]
    )


def collect_mode(args):
    """Handle data collection from multiple sources."""
    source_dirs = [Path(d) for d in args.source_dirs]
    output_dir = Path(args.output_dir) if args.output_dir else get_collected_dir_name()
    
    # Validate source directories
    for source_dir in source_dirs:
        if not source_dir.exists():
            logging.error(f"Source directory {source_dir} does not exist")
            return 1
    
    logging.info(f"Starting data collection")
    logging.info(f"Source directories: {[str(d) for d in source_dirs]}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Chunk size: {args.chunk_size}")
    
    try:
        stats = collect_and_organize_data(source_dirs, output_dir, args.chunk_size)
        if "error" in stats:
            return 1
        
        logging.info("Data collection completed successfully!")
        logging.info(f"Collected {stats['unique_games']} unique games from {stats['total_files']} files")
        return 0
    except Exception as e:
        logging.error(f"Data collection failed: {e}")
        return 1


def process_mode(args):
    """Handle single directory processing."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else get_cleaned_dir_name()
    
    if not input_dir.exists():
        logging.error(f"Input directory {input_dir} does not exist")
        return 1
    
    logging.info(f"Starting single directory processing")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Chunk size: {args.chunk_size}")
    
    try:
        combine_and_clean_files(input_dir, output_dir, args.chunk_size)
        logging.info("Single directory processing completed successfully!")
        return 0
    except Exception as e:
        logging.error(f"Single directory processing failed: {e}")
        return 1


def test_mode(args):
    """Test tournament pattern configuration and validation."""
    logging.info("Testing tournament pattern configuration...")
    
    # Test pattern validation
    config_errors = validate_tournament_patterns()
    if config_errors:
        logging.error("Configuration validation failed:")
        for error in config_errors:
            logging.error(f"  - {error}")
        return 1
    else:
        logging.info("✓ Configuration validation passed")
    
    # Test patterns with examples
    logging.info("\nTesting patterns with example directory names:")
    test_results = test_tournament_patterns_with_examples()
    
    for pattern_name, results in test_results.items():
        logging.info(f"\nPattern '{pattern_name}':")
        for result in results:
            logging.info(f"  {result}")
    
    # Test with actual source directories if provided
    if args.source_dirs:
        logging.info(f"\nTesting with actual source directories:")
        for source_dir_str in args.source_dirs:
            source_dir = Path(source_dir_str)
            if not source_dir.exists():
                logging.warning(f"  Source directory {source_dir} does not exist")
                continue
            
            try:
                from hex_ai.data_collection import find_tournament_directories
                tournament_dirs = find_tournament_directories(source_dir)
                logging.info(f"  {source_dir}: Found {len(tournament_dirs)} tournament directories")
                
                # Show first few examples
                for i, tournament_dir in enumerate(tournament_dirs[:3]):
                    logging.info(f"    - {tournament_dir.name}")
                if len(tournament_dirs) > 3:
                    logging.info(f"    ... and {len(tournament_dirs) - 3} more")
                    
            except Exception as e:
                logging.error(f"  {source_dir}: Error - {e}")
    
    logging.info("\n✓ Pattern testing completed")
    return 0


def tournament_mode(args):
    """Handle tournament data collection since a specific date."""
    # Validate configuration first
    config_errors = validate_tournament_patterns()
    if config_errors:
        logging.error("Tournament pattern configuration errors:")
        for error in config_errors:
            logging.error(f"  - {error}")
        logging.error("\nRun 'python scripts/collect_and_clean_games.py test' to diagnose configuration issues.")
        return 1
    
    source_dirs = [Path(d) for d in args.source_dirs]
    output_dir = Path(args.output_dir) if args.output_dir else get_collected_dir_name("tournament_data")
    
    # Parse since date
    if args.since_days:
        since_date = datetime.now() - timedelta(days=args.since_days)
    elif args.since_date:
        since_date = datetime.strptime(args.since_date, "%Y-%m-%d")
    else:
        # Default to 7 days ago
        since_date = datetime.now() - timedelta(days=7)
    
    # Validate source directories
    for source_dir in source_dirs:
        if not source_dir.exists():
            logging.error(f"Source directory {source_dir} does not exist")
            return 1
    
    logging.info(f"Starting tournament data collection")
    logging.info(f"Source directories: {[str(d) for d in source_dirs]}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Since date: {since_date}")
    logging.info(f"Chunk size: {args.chunk_size}")
    
    try:
        stats = collect_tournament_data_since_date(source_dirs, output_dir, since_date, args.chunk_size)
        if "error" in stats:
            return 1
        
        logging.info("Tournament data collection completed successfully!")
        logging.info(f"Collected {stats['unique_games']} unique games from {stats['total_files']} files")
        logging.info(f"Found {stats['total_tournament_dirs']} tournament directories")
        return 0
    except Exception as e:
        logging.error(f"Tournament data collection failed: {e}")
        logging.error("\nThis might indicate:")
        logging.error("1. Directory naming patterns have changed (check TOURNAMENT_PATTERNS config)")
        logging.error("2. Source directory structure is different than expected")
        logging.error("3. Configuration errors in date parsing")
        logging.error("\nRun 'python scripts/collect_and_clean_games.py test' to diagnose issues.")
        return 1


def main():
    """Main entry point."""
    # Setup logging
    setup_logging()
    
    # Create main parser
    parser = argparse.ArgumentParser(
        description="Unified data processing for Hex AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test tournament pattern configuration
  python scripts/collect_and_clean_games.py test
  
  # Test with actual source directories
  python scripts/collect_and_clean_games.py test --source-dirs data/tournament_play
  
  # Collect from multiple sources with default output
  python scripts/collect_and_clean_games.py collect
  
  # Collect from multiple sources with custom output
  python scripts/collect_and_clean_games.py collect --output-dir data/collected/my_data
  
  # Collect tournament data since September 3rd
  python scripts/collect_and_clean_games.py tournament --since-date 2025-09-03
  
  # Collect tournament data from last 7 days
  python scripts/collect_and_clean_games.py tournament --since-days 7
  
  # Collect tournament data from specific source directory
  python scripts/collect_and_clean_games.py tournament --source-dirs data/tournament_play --since-date 2025-09-03
  
  # Process single directory with default output
  python scripts/collect_and_clean_games.py process --input-dir data/sf25/jul29
  
  # Process single directory with custom output
  python scripts/collect_and_clean_games.py process --input-dir data/sf25/jul29 --output-dir data/cleaned/my_data
        """
    )
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Processing mode')
    
    # Collect mode parser
    collect_parser = subparsers.add_parser('collect', help='Collect data from multiple sources')
    collect_parser.add_argument(
        "--source-dirs", nargs="+", 
        default=[str(d) for d in DEFAULT_SOURCE_DIRS],
        help=f"Source directories to search for .trmph files (default: {[str(d) for d in DEFAULT_SOURCE_DIRS]})"
    )
    collect_parser.add_argument(
        "--output-dir", 
        help="Output directory for organized data (default: auto-generated with today's date)"
    )
    collect_parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Number of games per chunk (default: {DEFAULT_CHUNK_SIZE})"
    )
    
    # Test mode parser
    test_parser = subparsers.add_parser('test', help='Test tournament pattern configuration and validation')
    test_parser.add_argument(
        "--source-dirs", nargs="+", 
        help="Optional source directories to test with actual data"
    )
    
    # Tournament mode parser
    tournament_parser = subparsers.add_parser('tournament', help='Collect tournament data since a specific date')
    tournament_parser.add_argument(
        "--source-dirs", nargs="+", 
        default=[str(d) for d in DEFAULT_SOURCE_DIRS],
        help=f"Source directories to search for tournament data (default: {[str(d) for d in DEFAULT_SOURCE_DIRS]})"
    )
    tournament_parser.add_argument(
        "--output-dir", 
        help="Output directory for tournament data (default: auto-generated with today's date)"
    )
    tournament_parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Number of games per chunk (default: {DEFAULT_CHUNK_SIZE})"
    )
    tournament_parser.add_argument(
        "--since-date", 
        help="Collect data since this date (format: YYYY-MM-DD)"
    )
    tournament_parser.add_argument(
        "--since-days", type=int,
        help="Collect data since N days ago (alternative to --since-date)"
    )
    
    # Process mode parser
    process_parser = subparsers.add_parser('process', help='Process a single directory')
    process_parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing .trmph files"
    )
    process_parser.add_argument(
        "--output-dir",
        help="Output directory for cleaned files (default: auto-generated with today's date)"
    )
    process_parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Number of games per chunk (default: {DEFAULT_CHUNK_SIZE})"
    )
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return 1
    
    # Route to appropriate mode
    if args.mode == 'collect':
        return collect_mode(args)
    elif args.mode == 'test':
        return test_mode(args)
    elif args.mode == 'tournament':
        return tournament_mode(args)
    elif args.mode == 'process':
        return process_mode(args)
    else:
        logging.error(f"Unknown mode: {args.mode}")
        return 1


if __name__ == "__main__":
    exit(main())
